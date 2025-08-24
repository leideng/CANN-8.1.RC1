# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
basic_lstm_cell
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.constant_util import SIZE_SIXTEEN as C0


# 'pylint: disable=locally-disabled,ungrouped-imports,import-error,too-many-arguments,too-many-locals
def sigmoid(shape, tensor_allgate_ub, tensor_one,
            product_info, symbol, tensor_list, scope_list, operation_list):
    """
    the function of sigmoid
    Parameters
    ----------
    shape : tensor shape
    tensor_allgate_ub : tensor
    symbol : tensor symbol
    Returns
    -------
    """
    dtype_c = tensor_allgate_ub.dtype
    const_num_neg_one = tvm.const(-1, dtype=dtype_c)
    const_num_one = tvm.const(1, dtype=dtype_c)
    tensor_ub_neg_allgate = tvm.compute(shape,
                                        lambda a, b, c, d: tensor_allgate_ub[a, b, c, d] * const_num_neg_one,
                                        name="tensor_gate_neg_" + symbol)
    tensor_list["tensor_gate_neg_" + symbol] = tensor_ub_neg_allgate
    scope_list["tensor_gate_neg_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_gate_neg_" + symbol] = "vector_mul"
    tensor_ub_allgate_exp_fp16 = tensor_ub_neg_allgate
    if product_info["mini"]:
        tensor_ub_allgate_exp_fp16 = tvm.compute(shape,
                                                 lambda *i: shape_util.cast(tensor_ub_neg_allgate(*i), "float16"),
                                                 name="tensor_gate_exp_fp16_" + symbol)
        tensor_list["tensor_gate_exp_fp16_" + symbol] = tensor_ub_allgate_exp_fp16
        scope_list["tensor_gate_exp_fp16_" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_gate_exp_fp16_" + symbol] = "vector_conv"
    tensor_ub_allgate_exp = tvm.compute(shape,
                                        lambda *i: tvm.exp(tensor_ub_allgate_exp_fp16(*i)),
                                        name="tensor_gate_exp_" + symbol)
    tensor_list["tensor_gate_exp_" + symbol] = tensor_ub_allgate_exp
    scope_list["tensor_gate_exp_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_gate_exp_" + symbol] = "vector_exp"

    tensor_ub_allgate_exp_fp32 = tensor_ub_allgate_exp

    if not product_info["hisi_es"] and dtype_c != tensor_ub_allgate_exp.dtype:
        tensor_ub_allgate_exp_fp32 = tvm.compute(shape,
                                                 lambda *i:
                                                 shape_util.cast(tensor_ub_allgate_exp(*i), dtype_c),
                                                 name="tensor_gate_exp_fp32_" + symbol)
        tensor_list["tensor_gate_exp_fp32_" + symbol] = tensor_ub_allgate_exp_fp32
        scope_list["tensor_gate_exp_fp32_" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_gate_exp_fp32_" + symbol] = "vector_conv"
    tensor_ub_allgate_add = tvm.compute(shape,
                                        lambda *i:
                                        tensor_ub_allgate_exp_fp32(*i) + const_num_one,
                                        name="tensor_gate_add_" + symbol)
    tensor_list["tensor_gate_add_" + symbol] = tensor_ub_allgate_add
    scope_list["tensor_gate_add_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_gate_add_" + symbol] = "vector_add"

    if product_info["cloud"]:
        tensor_ub_allgate_sigmoid = tvm.compute(shape,
                                                lambda *i: tensor_one(*i) / tensor_ub_allgate_add(*i),
                                                name="tensor_gate_sigmoid_" + symbol)
        tensor_newton_mul2 = tensor_ub_allgate_sigmoid
        tensor_list["tensor_gate_sigmoid_" + symbol] = tensor_newton_mul2
        scope_list["tensor_gate_sigmoid_" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_gate_sigmoid_" + symbol] = "vector_div"
    else:
        tensor_ub_allgate_sigmoid = tvm.compute(shape,
                                                lambda *i:
                                                const_num_one / tensor_ub_allgate_add(*i),
                                                name="tensor_gate_sigmoid_" + symbol)
        tensor_list["tensor_gate_sigmoid_tmp_" + symbol] = tensor_ub_allgate_sigmoid
        scope_list["tensor_gate_sigmoid_tmp_" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_gate_sigmoid_tmp_" + symbol] = "vector_rec"
        tensor_newton_mul2 = newton_iteration(shape,
                                              tensor_ub_allgate_sigmoid,
                                              tensor_ub_allgate_add,
                                              symbol, tensor_list,
                                              scope_list, operation_list)
        tensor_list["tensor_gate_sigmoid_" + symbol] = tensor_newton_mul2
        scope_list["tensor_gate_sigmoid_" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_gate_sigmoid_" + symbol] = "vector_mul"


# 'pylint: disable=too-many-arguments
def newton_iteration(shape, tensor_x_rec,
                     tensor_x, symbol, tensor_list, scope_list, operation_list):
    """
    the function of newton_iteration
    Parameters
    ----------
    shape : tensor shape
    tensor_x_rec : tensor
    tensor_x : tensor
    symbol : tensor symbol
    Returns
    -------
    """
    dtype_c = tensor_x_rec.dtype
    const_num_neg_two = tvm.const(-2, dtype=dtype_c)
    const_num_neg_one = tvm.const(-1, dtype=dtype_c)

    tensor_newton_mul0 = tvm.compute(shape,
                                     lambda *i: tensor_x_rec(*i) * tensor_x(*i),
                                     name="tensor_newton_mul0_" + symbol)
    tensor_list["tensor_newton_mul0_" + symbol] = tensor_newton_mul0
    scope_list["tensor_newton_mul0_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_newton_mul0_" + symbol] = "vector_mul"
    tensor_newton_add = tvm.compute(shape,
                                    lambda *i: tensor_newton_mul0(*i) + const_num_neg_two,
                                    name="tensor_newton_add_" + symbol)
    tensor_list["tensor_newton_add_" + symbol] = tensor_newton_add
    scope_list["tensor_newton_add_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_newton_add_" + symbol] = "vector_add"
    tensor_newton_mul1 = tvm.compute(shape,
                                     lambda *i: tensor_newton_add(*i) * tensor_x_rec(*i),
                                     name="tensor_newton_mul1_" + symbol)
    tensor_list["tensor_newton_mul1_" + symbol] = tensor_newton_mul1
    scope_list["tensor_newton_mul1_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_newton_mul1_" + symbol] = "vector_mul"
    tensor_newton_mul2 = tvm.compute(shape,
                                     lambda *i: tensor_newton_mul1(*i) * const_num_neg_one,
                                     name="tensor_newton_mul2_" + symbol)
    return tensor_newton_mul2


# 'pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def tanh(shape, tensor, product_info, symbol,
         tensor_list, scope_list, operation_list):
    """
    the function of tanh
    Parameters
    ----------
    shape : tensor shape
    tensor : tensor
    symbol : tensor symbol
    Returns
    -------
    """
    dtype_c = tensor.dtype
    const_num_one = tvm.const(1, dtype=dtype_c)
    const_num_two = tvm.const(-2, dtype=dtype_c)
    const_fp32_min = tvm.const(2 ** (-126), dtype=dtype_c)

    tensor_ub_two_abs = tvm.compute(shape, lambda *i: tvm.abs(tensor(*i)), name="tensor_ub_two_abs_" + symbol)
    tensor_list["tensor_ub_two_abs_" + symbol] = tensor_ub_two_abs
    scope_list["tensor_ub_two_abs_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_ub_two_abs_" + symbol] = "vector_abs"

    tensor_ub_two = tvm.compute(shape, lambda *i: tensor_ub_two_abs(*i) * const_num_two,
                                name="tensor_ub_two_" + symbol)
    tensor_list["tensor_ub_two_" + symbol] = tensor_ub_two
    scope_list["tensor_ub_two_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_ub_two_" + symbol] = "vector_mul"

    tensor_ub_exp_fp16 = tensor_ub_two
    if product_info["mini"] and dtype_c == "float32":
        tensor_ub_exp_fp16 = tvm.compute(shape,
                                         lambda *i:
                                         shape_util.cast(tensor_ub_two(*i), "float16"),
                                         name="tensor_ub_exp_fp16_" + symbol)
        tensor_list["tensor_ub_exp_fp16_" + symbol] = tensor_ub_exp_fp16
        scope_list["tensor_ub_exp_fp16_" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_ub_exp_fp16_" + symbol] = "vector_conv"

    tensor_ub_exp = tvm.compute(shape,
                                lambda *i: tvm.exp(tensor_ub_exp_fp16(*i)),
                                name="tensor_ub_exp_" + symbol)
    tensor_list["tensor_ub_exp_" + symbol] = tensor_ub_exp
    scope_list["tensor_ub_exp_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_ub_exp_" + symbol] = "vector_exp"

    tensor_ub_exp_fp32 = tensor_ub_exp
    if not product_info["hisi_es"] and tensor_ub_exp.dtype != "float32":
        tensor_ub_exp_fp32 = tvm.compute(shape,
                                         lambda *i:
                                         shape_util.cast(tensor_ub_exp(*i), "float32"),
                                         name="tensor_ub_exp_fp32_" + symbol)
        tensor_list["tensor_ub_exp_fp32_" + symbol] = tensor_ub_exp_fp32
        scope_list["tensor_ub_exp_fp32_" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_ub_exp_fp32_" + symbol] = "vector_conv"

    tensor_mul_temp = tvm.compute(
        shape, lambda *i: tensor_ub_exp_fp32(*i) * tensor(*i), name="tensor_mul_temp_" + symbol)
    tensor_list["tensor_mul_temp_" + symbol] = tensor_mul_temp
    scope_list["tensor_mul_temp_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_mul_temp_" + symbol] = "vector_mul"

    tensor_sub_temp = tvm.compute(
        shape, lambda *i: tensor(*i) - tensor_mul_temp(*i), name="tensor_sub_temp_" + symbol)
    tensor_list["tensor_sub_temp_" + symbol] = tensor_sub_temp
    scope_list["tensor_sub_temp_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_sub_temp_" + symbol] = "vector_sub"

    tenosr_add_min_temp = tvm.compute(
        shape, lambda *i: tensor_ub_two_abs(*i) + const_fp32_min, name="tenosr_add_min_temp_" + symbol)
    tensor_list["tenosr_add_min_temp_" + symbol] = tenosr_add_min_temp
    scope_list["tenosr_add_min_temp_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tenosr_add_min_temp_" + symbol] = "vector_add"

    tenosr_add_1_temp = tvm.compute(
        shape, lambda *i: tensor_ub_exp_fp32(*i) + const_num_one, name="tenosr_add_1_temp_" + symbol)
    tensor_list["tenosr_add_1_temp_" + symbol] = tenosr_add_1_temp
    scope_list["tenosr_add_1_temp_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tenosr_add_1_temp_" + symbol] = "vector_add"

    tenosr_down_temp = tvm.compute(
        shape, lambda *i: tenosr_add_1_temp(*i) * tenosr_add_min_temp(*i), name="tenosr_down_temp_" + symbol)
    tensor_list["tenosr_down_temp_" + symbol] = tenosr_down_temp
    scope_list["tenosr_down_temp_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tenosr_down_temp_" + symbol] = "vector_mul"

    if product_info["mini"]:
        tensor_ub_rec = tvm.compute(shape,
                                    lambda *i: const_num_one / tenosr_down_temp(*i),
                                    name="tensor_ub_rec_" + symbol)
        tensor_list["tensor_ub_rec_" + symbol] = tensor_ub_rec
        scope_list["tensor_ub_rec_" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_ub_rec_" + symbol] = "vector_rec"
    else:
        const_num_one = tvm.const(1.0, dtype=tenosr_down_temp.dtype)
        tensor_one = tvm.compute(shape, lambda *i: const_num_one, name='tensor_one' + symbol)
        tensor_list["tensor_one" + symbol] = tensor_one
        scope_list["tensor_one" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_one" + symbol] = "vector_dup"

        tensor_ub_rec = tvm.compute(shape,
                                    lambda *i: tensor_one(*i) / tenosr_down_temp(*i),
                                    name="tensor_ub_rec_" + symbol)
        tensor_list["tensor_ub_rec_" + symbol] = tensor_ub_rec
        scope_list["tensor_ub_rec_" + symbol] = tbe_platform.scope_ubuf
        operation_list["tensor_ub_rec_" + symbol] = "vector_div"

    tensor_newton_mul2 = newton_iteration(shape, tensor_ub_rec, tenosr_down_temp,
                                          symbol, tensor_list,
                                          scope_list, operation_list)
    tensor_list["tensor_ub_tanh_newton_" + symbol] = tensor_newton_mul2
    scope_list["tensor_ub_tanh_newton_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_ub_tanh_newton_" + symbol] = "vector_mul"

    tensor_ub_tanh = tvm.compute(shape,
                                 lambda *i:
                                 tensor_sub_temp(*i) * tensor_newton_mul2(*i),
                                 name="tensor_ub_tanh_" + symbol)
    tensor_list["tensor_ub_tanh_" + symbol] = tensor_ub_tanh
    scope_list["tensor_ub_tanh_" + symbol] = tbe_platform.scope_ubuf
    operation_list["tensor_ub_tanh_" + symbol] = "vector_mul"


# 'pylint: disable=locally-disabled,too-many-boolean-expressions,invalid-name,too-many-arguments,too-many-locals
def basiclstm_cell_check(x, h, c, w, b, ct, ht, it, ft, jt, ot, tanhct):
    """
    the main function of check basic_lstm_cell
    Parameters
    ----------
    x : matmul left x
    h : matmul left h
    c : lstm cell state last time
    w : matmul right w
    b : matmul bias
    ct : lstm cell state this time
    ht : lstm cell output
    it : input gate
    jt : new gate
    ft : forget gate
    ot : output gate
    tanhct : tanh(ot)
    Returns
    -------
    """
    # Check x, h, w, ht dtype
    if x["dtype"] != "float16" or h["dtype"] != "float16" or w["dtype"] != "float16" or ht["dtype"] != "float16":
        error_detail = "x, h, w, ht only support float16"
        error_manager_vector.raise_err_two_input_dtype_invalid("basic_lstm_cell", "x,h", "w,ht", error_detail)
    # Check c, b, ct, it, ft, jt, ot, tanhct dtype
    if c["dtype"] != b["dtype"] or b["dtype"] != ct["dtype"] or \
            ct["dtype"] != it["dtype"] or it["dtype"] != ft["dtype"] or \
            ft["dtype"] != jt["dtype"] or jt["dtype"] != ot["dtype"] or \
            ot["dtype"] != tanhct["dtype"]:
        error_detail = "the dtype of c, b, ct, it, ft, jt, ot and tanhct is not match"
        error_manager_vector.raise_err_two_input_dtype_invalid("basic_lstm_cell", "c, b, ct, it",
                                                               "ft, jt, ot, tanhct", error_detail)
    if c["dtype"] not in ["float16", "float32"]:
        rule_desc = "dtype of c should be one of [float16,float32]"
        error_manager_vector.raise_err_check_params_rules("basic_lstm_cell", rule_desc, "c", c["dtype"])

    if len(x["ori_shape"]) != 2:
        error_detail = "ori_shape'rank of x should be 2"
        error_manager_vector.raise_err_input_shape_invalid("basic_lstm_cell", "x", error_detail)
    if len(h["ori_shape"]) != 2:
        error_detail = "ori_shape'rank of h should be 2"
        error_manager_vector.raise_err_input_shape_invalid("basic_lstm_cell", "h", error_detail)
    if len(c["ori_shape"]) != 2:
        error_detail = "ori_shape'rank of c should be 2"
        error_manager_vector.raise_err_input_shape_invalid("basic_lstm_cell", "c", error_detail)

    if len(w["ori_shape"]) != 2:
        error_detail = "ori_shape'rank of w should be 2"
        error_manager_vector.raise_err_input_shape_invalid("basic_lstm_cell", "w", error_detail)
    if len(b["ori_shape"]) != 1:
        error_detail = "ori_shape'rank of b should be 1"
        error_manager_vector.raise_err_input_shape_invalid("basic_lstm_cell", "b", error_detail)

    batch_dim_x, input_dim_x = x["ori_shape"]
    batch_dim_h, output_dim_h = h["ori_shape"]
    batch_dim_c, output_dim_c = c["ori_shape"]
    w_h_dim, w_w_dim = w["ori_shape"]
    b_dim = b["ori_shape"][0]

    if batch_dim_x != batch_dim_h or batch_dim_x != batch_dim_c:
        error_detail = "the batch_dim is wrong: x {} h {} c {}, please check!" \
            .format(batch_dim_x, batch_dim_h, batch_dim_c)
        error_manager_vector.raise_err_input_shape_invalid("basic_lstm_cell", "batch_dim", error_detail)

    if output_dim_h != output_dim_c:
        error_detail = "output_dim_h {} should be equal to output_dim_c {}, please check!" \
            .format(output_dim_h, output_dim_c)
        error_manager_vector.raise_err_input_shape_invalid("basic_lstm_cell", "output_dim", error_detail)

    if (output_dim_h + input_dim_x) != w_h_dim or output_dim_h != w_w_dim // 4:
        error_detail = "the shape of w is wrong: output_dim_h+input_dim_x " \
                       "{}, output_dim_h {}, w_w_dim {}, w_h_dim {}, please check!" \
            .format(output_dim_h + input_dim_x, output_dim_h, w_w_dim, w_h_dim)
        error_manager_vector.raise_err_input_shape_invalid("basic_lstm_cell", "w", error_detail)
    if output_dim_h != b_dim // 4:
        error_detail = "the shape of b is wrong: output_dim_h {}, b_dim {}, please check!" \
            .format(output_dim_h, b_dim)
        error_manager_vector.raise_err_input_shape_invalid("basic_lstm_cell", "b", error_detail)


# 'pylint: disable=too-many-locals,too-many-statements,unnecessary-lambda,cell-var-from-loop
def get_matmul_tensor(x, h, c, w, b, build_list, tensor_list, scope_list, operation_list, is_hisi_es):
    """
    get_matmul_tensor
    """
    shape_x = x.get("shape")
    shape_h = h.get("shape")
    shape_c = c.get("shape")
    dtype_x = x.get("dtype").lower()
    dtype_c = c.get("dtype").lower()
    dtype_b = b.get("dtype").lower()
    input_dim, batch_dim = shape_x[0:2]
    hidden_dim = shape_h[0]
    output_dim = hidden_dim
    shape_b = b.get("shape")
    shape_b = (shape_b[0] // 16, 16)
    shape_xh = (batch_dim, input_dim + hidden_dim, C0, C0)
    shape_w = w.get("shape")
    shape_w_split = list(shape_w)
    shape_w_split[1] = shape_w_split[1] // 4

    # Inputs in gm
    tensor_x = tvm.placeholder(shape_x, name='tensor_x', dtype=dtype_x)
    tensor_h = tvm.placeholder(shape_h, name='tensor_h', dtype=dtype_x)
    tensor_c = tvm.placeholder(shape_c, name='tensor_c', dtype=dtype_c)
    tensor_w = tvm.placeholder(shape_w, name='tensor_w', dtype=dtype_x)
    tensor_b = tvm.placeholder(shape_b, name='tensor_b', dtype=dtype_c)
    build_list["x"] = tensor_x
    build_list["h"] = tensor_h
    build_list["c"] = tensor_c
    build_list["w"] = tensor_w
    build_list["b"] = tensor_b

    symbol = ["it", "jt", "ft", "ot"]

    def _index_w(str_name, *index):
        if str_name == "it":
            return index[0], index[1], index[2], index[3]
        if str_name == "jt":
            return index[0], index[1] + output_dim, index[2], index[3]
        if str_name == "ft":
            return index[0], index[1] + output_dim * 2, index[2], index[3]
        return index[0], index[1] + output_dim * 3, index[2], index[3]

    def _index_bias(str_name):
        if str_name == "it":
            return 0
        if str_name == "jt":
            return 1
        if str_name == "ft":
            return 2
        return 3

    matmul_type = "float32"
    if is_hisi_es:
        matmul_type = "float16"

    for t in symbol:
        # caoncat x and h into 1 tensor,copy to L1
        tensor_xh_l1_tmp = tvm.compute(shape_xh,
                                       lambda *indice:
                                       tvm.select(indice[1] < input_dim,
                                                  tensor_x[indice[1],
                                                           indice[0],
                                                           indice[2],
                                                           indice[3]],
                                                  tensor_h[indice[1] - input_dim,
                                                           indice[0],
                                                           indice[2],
                                                           indice[3]]),
                                       name="tensor_xh_l1_" + t, tag="concat")
        tensor_list["tensor_xh_l1_" + t] = tensor_xh_l1_tmp
        scope_list["tensor_xh_l1_" + t] = tbe_platform.scope_cbuf
        # optimazition: copy one time
        operation_list["tensor_xh_l1_" + t] = "dma_copy"

        # copy  xh  to L1
        tensor_xh_l0a_tmp = tvm.compute(shape_xh,
                                        lambda *i: tensor_xh_l1_tmp(*i),
                                        name='tensor_xh_l0a_' + t)
        tensor_list["tensor_xh_l0a_" + t] = tensor_xh_l0a_tmp
        scope_list["tensor_xh_l0a_" + t] = tbe_platform.scope_ca
        # optimazition: copy one time
        operation_list["tensor_xh_l0a_" + t] = "dma_copy"
        # copy w to L1 buf
        tensor_w_l1_tmp = tvm.compute(shape_w_split,
                                      lambda *i: tensor_w(*_index_w(t, *i)),
                                      name='tensor_w_l1_' + t)
        tensor_list["tensor_w_l1_" + t] = tensor_w_l1_tmp
        scope_list["tensor_w_l1_" + t] = tbe_platform.scope_cbuf
        operation_list["tensor_w_l1_" + t] = "dma_copy"

        # copy W from L1 to L0 B
        tensor_w_l0b_tmp = tvm.compute(shape_w_split,
                                       lambda *i: tensor_w_l1_tmp(*i),
                                       name='tensor_w_l0b_' + t)
        tensor_list["tensor_w_l0b_" + t] = tensor_w_l0b_tmp
        scope_list["tensor_w_l0b_" + t] = tbe_platform.scope_cb
        operation_list["tensor_w_l0b_" + t] = "dma_copy"

        # copy bias to ubuf ,split the
        tensor_b_ub_tmp = tvm.compute(shape_b,
                                      lambda i0, i1:
                                      tensor_b[_index_bias(t) *
                                               output_dim + i0, i1],
                                      name='tensor_b_ub_' + t)
        tensor_list["tensor_b_ub_" + t] = tensor_b_ub_tmp
        scope_list["tensor_b_ub_" + t] = tbe_platform.scope_ubuf
        operation_list["tensor_b_ub_" + t] = "dma_copy"

        #
        tensor_b_ub_true_tmp = tensor_b_ub_tmp
        if not is_hisi_es and dtype_b == "float16":
            tensor_b_ub_true_tmp = tvm.compute(shape_b,
                                               lambda *i:
                                               shape_util.cast(tensor_b_ub_tmp(*i), "float32"),
                                               name="tensor_b_ub_true_" + t)
            tensor_list["tensor_b_ub_true_" + t] = tensor_b_ub_true_tmp
            scope_list["tensor_b_ub_true_" + t] = tbe_platform.scope_ubuf
            operation_list["tensor_b_ub_true_" + t] = "vector_conv"

        # broadcast bias from [ouput_dim//16,16] to [output_dim//16,N//16,16,16]
        tensor_b_loc_tmp = tvm.compute(shape_h,
                                       lambda i0, i1, i2, i3:
                                       tensor_b_ub_true_tmp[i0, i3],
                                       name='tensor_b_loc_' + t)
        tensor_list["tensor_b_loc_" + t] = tensor_b_loc_tmp
        scope_list["tensor_b_loc_" + t] = tbe_platform.scope_cc
        operation_list["tensor_b_loc_" + t] = "dma_copy"
        # DO MATMUL
        reduce_kb = tvm.reduce_axis((0, input_dim + output_dim), name='reduce_kb_' + t)
        reduce_kp = tvm.reduce_axis((0, C0), name='reduce_kp_' + t)
        tensor_matmul_l0c_tmp = tvm.compute(
            shape_h, lambda nb, mb, mp, np: tvm.sum(
                (tensor_xh_l0a_tmp[mb, reduce_kb, mp, reduce_kp] *
                 tensor_w_l0b_tmp[reduce_kb, nb, np, reduce_kp]).astype(matmul_type), axis=[reduce_kb, reduce_kp]),
            name='tensor_matmul_l0c_' + t, attrs={'input_order': 'positive'})
        tensor_list["tensor_matmul_l0c_" + t] = tensor_matmul_l0c_tmp
        scope_list["tensor_matmul_l0c_" + t] = tbe_platform.scope_cc
        # Matmul accumulation it + b_it
        tensor_matmul_result_l0c_tmp = tvm.compute(shape_h,
                                                   lambda *i: tensor_b_loc_tmp(*i) + tensor_matmul_l0c_tmp(*i),
                                                   name="tensor_matmul_result_l0c_" + t)
        tensor_list["tensor_matmul_result_l0c_" + t] = tensor_matmul_result_l0c_tmp
        scope_list["tensor_matmul_result_l0c_" + t] = tbe_platform.scope_cc
        operation_list["tensor_matmul_result_l0c_" + t] = "phony_insn"

        # copy matmul result from l0c to ub
        gate_ub_tmp = tvm.compute(shape_h,
                                  lambda *i: tensor_list["tensor_matmul_result_l0c_" + t](*i),
                                  name=t + "_ub")
        tensor_list[t + "_ub"] = gate_ub_tmp
        scope_list[t + "_ub"] = tbe_platform.scope_ubuf
        operation_list[t + "_ub"] = "dma_copy"


# 'pylint: disable=too-many-locals,too-many-statements,unnecessary-lambda,cell-var-from-loop
def get_activate_tensor(forget_bias, build_list, tensor_list, scope_list, operation_list, product_info):
    """
    get activate tensor
    """
    unreused_flag = True
    ft_ub = tensor_list["ft_ub"]
    shape_gate = ft_ub.shape
    const_forget_bias = tvm.const(forget_bias, dtype=ft_ub.dtype)
    ft_ub_temp_fbias = tvm.compute(shape_gate,
                                   lambda *i: ft_ub(*i) + const_forget_bias,
                                   name="ft_ub_temp_fbias")
    tensor_list["ft_ub_temp_fbias"] = ft_ub_temp_fbias
    scope_list["ft_ub_temp_fbias"] = tbe_platform.scope_ubuf
    operation_list["ft_ub_temp_fbias"] = "vector_add"

    const_num_one = tvm.const(1.0, dtype=ft_ub.dtype)
    tensor_one = tvm.compute(shape_gate, lambda *i: const_num_one, name='tensor_one')
    if product_info["cloud"]:
        tensor_list["tensor_one"] = tensor_one
        scope_list["tensor_one"] = tbe_platform.scope_ubuf
        operation_list["tensor_one"] = "vector_dup"

    symbol = ["ot", "it", "jt", "ft"]
    for t in symbol:
        temp_tensor = tensor_list["ft_ub_temp_fbias"] if t == "ft" else tensor_list[t + "_ub"]
        if t == "jt":
            # Do tanh(jt) calculation
            tanh(shape_gate, temp_tensor, product_info, "jt", tensor_list, scope_list, operation_list)
            activate_tmp = tensor_list["tensor_ub_tanh_jt"]
        else:
            # Do sigmoid(It,Ft,ot) calculation
            sigmoid(shape_gate, temp_tensor, tensor_one, product_info, t, tensor_list, scope_list, operation_list)
            activate_tmp = tensor_list["tensor_gate_sigmoid_" + t]

        activate_tmp_x = tvm.compute(shape_gate, lambda *i: activate_tmp(*i), name=t + "_ub2gm")
        tensor_list[t + "_ub2gm"] = activate_tmp_x
        scope_list[t + "_ub2gm"] = tbe_platform.scope_ubuf
        operation_list[t + "_ub2gm"] = "dma_copy"

        src_b_type = build_list["b"].dtype
        if activate_tmp_x.dtype != src_b_type:
            activate_tmp_ub_true = tvm.compute(shape_gate,
                                               lambda *i:
                                               shape_util.cast(activate_tmp_x(*i), src_b_type),
                                               name="tensor_" + t + "_ub_true")
            operation_list["tensor_" + t + "_ub_true"] = "vector_conv"
        else:
            activate_tmp_ub_true = activate_tmp_x
            operation_list["tensor_" + t + "_ub_true"] = "phony_insn"
            unreused_flag = False
        tensor_list["tensor_" + t + "_ub_true"] = activate_tmp_ub_true
        scope_list["tensor_" + t + "_ub_true"] = tbe_platform.scope_ubuf

        # Move it, jt, ft, ot to GM
        gate_t_gm = tvm.compute(shape_gate, lambda *i: activate_tmp_ub_true(*i), name=t)
        tensor_list[t] = gate_t_gm
        operation_list[t] = "dma_copy"
        build_list[t] = gate_t_gm

        # Move it, jt, ft, ot back to ub(Fake)
        gate_t_ub_fake = tvm.compute(shape_gate, lambda *i: gate_t_gm(*i), name=t + '_ub_fake')
        tensor_list[t + "_ub_fake"] = gate_t_ub_fake
        scope_list[t + "_ub_fake"] = tbe_platform.scope_ubuf
        operation_list[t + "_ub_fake"] = "phony_insn"

        # Move it, jt, ft, ot back to ub(Fake)
        gate_data_type = "float32" if not product_info["hisi_es"] else "float16"
        gate_t_ub_fake_true = tvm.compute(shape_gate,
                                          lambda *i:
                                          shape_util.cast(gate_t_ub_fake(*i), gate_data_type),
                                          name=t + "_ub_fake_true")

        tensor_list[t + "_ub_fake_true"] = gate_t_ub_fake_true
        scope_list[t + "_ub_fake_true"] = tbe_platform.scope_ubuf
        operation_list[t + "_ub_fake_true"] = "phony_insn"

    # get tensor c copy it to ub
    tensor_c = build_list["c"]
    src_c_type = tensor_c.dtype
    tensor_c_ub = tvm.compute(shape_gate,
                              lambda *i: tensor_c(*i),
                              name='tensor_c_ub')
    tensor_list["tensor_c_ub"] = tensor_c_ub
    scope_list["tensor_c_ub"] = tbe_platform.scope_ubuf
    operation_list["tensor_c_ub"] = "dma_copy"

    tensor_c_ub_true = tensor_c_ub
    # cast to float32
    if not product_info["hisi_es"] and src_c_type == "float16":
        tensor_c_ub_true = tvm.compute(shape_gate,
                                       lambda *i:
                                       shape_util.cast(tensor_c_ub(*i), "float32"),
                                       name="tensor_c_ub_true")
        tensor_list["tensor_c_ub_true"] = tensor_c_ub_true
        scope_list["tensor_c_ub_true"] = tbe_platform.scope_ubuf
        operation_list["tensor_c_ub_true"] = "vector_conv"
    tensor_cf_ub = tvm.compute(shape_gate,
                               lambda *i: tensor_c_ub_true(*i) * tensor_list["ft_ub_fake_true"](*i),
                               name="tensor_cf_ub")
    tensor_list["tensor_cf_ub"] = tensor_cf_ub
    scope_list["tensor_cf_ub"] = tbe_platform.scope_ubuf
    operation_list["tensor_cf_ub"] = "vector_mul"
    tensor_ij_ub = tvm.compute(shape_gate,
                               lambda *i: tensor_list["jt_ub_fake_true"](*i) * tensor_list["it_ub_fake_true"](*i),
                               name="tensor_ij_ub")
    tensor_list["tensor_ij_ub"] = tensor_ij_ub
    scope_list["tensor_ij_ub"] = tbe_platform.scope_ubuf
    operation_list["tensor_ij_ub"] = "vector_mul"
    tensor_ct_ub = tvm.compute(shape_gate,
                               lambda *i: tensor_cf_ub(*i) + tensor_ij_ub(*i),
                               name="tensor_ct_ub")
    tensor_list["tensor_ct_ub"] = tensor_ct_ub
    scope_list["tensor_ct_ub"] = tbe_platform.scope_ubuf
    operation_list["tensor_ct_ub"] = "vector_add"

    # cast to original data type
    if not product_info["hisi_es"] and tensor_ct_ub.dtype != src_c_type:
        tensor_ct_ub_true = tvm.compute(shape_gate,
                                        lambda *i:
                                        shape_util.cast(tensor_ct_ub(*i), src_c_type),
                                        name="tensor_ct_ub_true")
        operation_list["tensor_ct_ub_true"] = "vector_conv"
    else:
        tensor_ct_ub_true = tensor_ct_ub
        operation_list["tensor_ct_ub_true"] = "phony_insn"
    tensor_list["tensor_ct_ub_true"] = tensor_ct_ub_true
    scope_list["tensor_ct_ub_true"] = tbe_platform.scope_ubuf

    # Move ct to gm
    ct = tvm.compute(shape_gate, lambda *i: tensor_ct_ub_true(*i), name="ct")
    build_list["ct"] = ct
    tensor_list["ct"] = ct
    operation_list["ct"] = "dma_copy"

    # Move ct back(Fake)
    tensor_ct_ub_fake = tvm.compute(shape_gate, lambda *i: ct(*i), name="ct_ub_fake")
    tensor_list["tensor_ct_ub_fake"] = tensor_ct_ub_fake
    scope_list["tensor_ct_ub_fake"] = tbe_platform.scope_ubuf
    operation_list["tensor_ct_ub_fake"] = "phony_insn"

    ct_fake_true_type = "float32" if not product_info["hisi_es"] else "float16"
    tensor_ct_ub_fake_true = tvm.compute(shape_gate,
                                         lambda *i:
                                         shape_util.cast(tensor_ct_ub_fake(*i), ct_fake_true_type),
                                         name="tensor_ct_ub_fake_true")
    tensor_list["tensor_ct_ub_fake_true"] = tensor_ct_ub_fake_true
    scope_list["tensor_ct_ub_fake_true"] = tbe_platform.scope_ubuf
    operation_list["tensor_ct_ub_fake_true"] = "phony_insn"

    # calc tanh(ct)
    tanh(shape_gate, tensor_ct_ub_fake_true, product_info,
         "ct", tensor_list, scope_list, operation_list)

    # move tanh(ct) to gm
    tensor_tanh_ct_ub_true = tensor_list["tensor_ub_tanh_ct"]
    if not product_info["hisi_es"] and tensor_tanh_ct_ub_true.dtype != src_c_type:
        tensor_tanh_ct_ub_true = tvm.compute(shape_gate,
                                             lambda *i:
                                             shape_util.cast(tensor_list["tensor_ub_tanh_ct"](*i), src_c_type),
                                             name="tensor_tanh_ct_ub_true")
        tensor_list["tensor_tanh_ct_ub_true"] = tensor_tanh_ct_ub_true
        scope_list["tensor_tanh_ct_ub_true"] = tbe_platform.scope_ubuf
        operation_list["tensor_tanh_ct_ub_true"] = "vector_conv"

    tanhct = tvm.compute(shape_gate, lambda *i: tensor_tanh_ct_ub_true(*i), name="tanhct")

    build_list["tanhct"] = tanhct
    tensor_list["tanhct"] = tanhct
    operation_list["tanhct"] = "dma_copy"

    # Move tanh(ct) back(Fake)
    tensor_tanhct_ub_fake = tvm.compute(shape_gate, lambda *i: tanhct(*i), name="tensor_tanhct_ub_fake")
    tensor_list["tensor_tanhct_ub_fake"] = tensor_tanhct_ub_fake
    scope_list["tensor_tanhct_ub_fake"] = tbe_platform.scope_ubuf
    operation_list["tensor_tanhct_ub_fake"] = "phony_insn"

    tanhct_fake_true_type = "float32" if not product_info["hisi_es"] else "float16"
    tensor_tanhct_ub_fake_true = tvm.compute(shape_gate,
                                             lambda *i:
                                             shape_util.cast(tensor_tanhct_ub_fake(*i), tanhct_fake_true_type),
                                             name="tensor_tanhct_ub_fake_true")
    tensor_list["tensor_tanhct_ub_fake_true"] = tensor_tanhct_ub_fake_true
    scope_list["tensor_tanhct_ub_fake_true"] = tbe_platform.scope_ubuf
    operation_list["tensor_tanhct_ub_fake_true"] = "phony_insn"

    tensor_ht_ub = tvm.compute(shape_gate,
                               lambda *i:
                               tensor_list["ot_ub_fake_true"](*i) *
                               tensor_tanhct_ub_fake_true(*i),
                               name="tensor_ht_ub")
    tensor_list["tensor_ht_ub"] = tensor_ht_ub
    scope_list["tensor_ht_ub"] = tbe_platform.scope_ubuf
    operation_list["tensor_ht_ub"] = "vector_mul"

    tensor_ht_ub_true = tensor_ht_ub
    if not product_info["hisi_es"] and build_list["h"].dtype == "float16":
        tensor_ht_ub_true = tvm.compute(shape_gate,
                                        lambda *i:
                                        shape_util.cast(tensor_ht_ub(*i), "float16"),
                                        name="tensor_ht_ub_true")
        tensor_list["tensor_ht_ub_true"] = tensor_ht_ub_true
        scope_list["tensor_ht_ub_true"] = tbe_platform.scope_ubuf
        operation_list["tensor_ht_ub_true"] = "vector_conv"
    # Move ht to gm
    ht = tvm.compute(shape_gate, lambda *i: tensor_ht_ub_true(*i), name="ht")
    tensor_list["ht"] = ht
    build_list["ht"] = ht

    return unreused_flag


def basic_lstm_cell_compute(x, h, c, w, b, forget_bias, product_info):
    """
    This is the compute func
    """
    build_list = {}
    tensor_list = {}
    scope_list = {}
    operation_list = {}

    get_matmul_tensor(x, h, c, w, b, build_list, tensor_list, scope_list, operation_list, product_info["hisi_es"])
    unreused_flag = get_activate_tensor(forget_bias, build_list, tensor_list, scope_list, operation_list, product_info)
    return build_list, tensor_list, scope_list, operation_list, unreused_flag


# 'pylint: disable=too-many-locals,too-many-statements
def get_tilling(x, h_t, product_info):
    """
    This is tiling func
    """
    block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    l0_size = tbe_platform.get_soc_spec(tbe_platform.L0B_SIZE)
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // 2
    ub_limit = ub_size // 4

    x_shape = x["shape"]
    ht_shape = h_t["shape"]
    input_dim = x_shape[0]
    n_dim = x_shape[1]
    out_dim = ht_shape[0]

    tilling_info = {}
    # block tilling
    n_core = True
    if n_dim % block_num == 0:
        n_core = False
        block_n_npart = block_num
    else:
        block_n_npart = 1

    if out_dim % block_num == 0 and n_core:
        block_out_npart = block_num
    else:
        block_out_npart = 1

    block_n_factor = n_dim // block_n_npart
    block_out_factor = out_dim // block_out_npart

    dtype_mad_size = 2

    # fix type to fp32
    dtype_size = 4

    def _decrement_out_factor(temp_out_factor, block_out):
        div = (block_out // temp_out_factor) + 1
        while block_out % div != 0 and div < block_out:
            div = div + 1
        res = block_out // div

        return res

    def _get_ub_used_size(n_factor, out_factor):
        res = (1 * n_factor * out_factor * C0 * C0 + n_factor * C0 + 3 * out_factor * C0) * dtype_size
        return res

    while (_get_ub_used_size(block_n_factor, block_out_factor) > ub_limit) and block_n_factor > 1:
        block_n_factor = _decrement_out_factor(block_n_factor, n_dim)

    while (_get_ub_used_size(block_n_factor, block_out_factor) > ub_limit) and block_out_factor > 1:
        block_out_factor = _decrement_out_factor(block_out_factor, out_dim)

    # ub tilling
    if product_info["hisi_es"]:
        l0_size = 32768
        k_factor = 64
    else:
        k_factor = 128

    one_mn_size = k_factor * C0 * C0 * dtype_mad_size
    n_factor = min(int(l0_size / one_mn_size), block_n_factor)
    out_factor = min(int(l0_size / one_mn_size), block_out_factor)

    def gcd(var1, var2):
        var1, var2 = (var1, var2) if var1 >= var2 else (var1, var2)
        while var2:
            var1, var2 = var2, var1 % var2
        return var1

    if input_dim == out_dim and input_dim <= k_factor:
        l1_factor = input_dim
    elif (input_dim > out_dim and input_dim % out_dim == 0) or (input_dim < out_dim and out_dim % input_dim == 0):
        l1_factor = min(input_dim, out_dim)
    else:
        l1_factor = gcd(input_dim, out_dim)
    tilling_info["block_n_factor"] = block_n_factor
    tilling_info["block_out_factor"] = block_out_factor
    tilling_info["k_factor"] = k_factor
    tilling_info["n_factor"] = n_factor
    tilling_info["out_factor"] = out_factor
    tilling_info["l1_factor"] = l1_factor

    return tilling_info


# 'pylint: disable=too-many-locals,too-many-statements,too-many-branches
def basic_lstm_cell_schedule(tensor_list, scope_list,
                             operation_list, build_list,
                             tilling_info, unreused_flag, kernel_name):
    """
    do the schedule for the LSTM compute.
    """
    ht = tensor_list["ht"]
    schedule_list = [ht.op]
    s = tvm.create_schedule(schedule_list)

    for key in tensor_list.keys():
        if key in scope_list.keys():
            s[tensor_list[key]].set_scope(scope_list[key])
        if key in operation_list.keys():
            s[tensor_list[key]].emit_insn(s[tensor_list[key]].op.axis[0], operation_list[key])

    s[tensor_list["tensor_xh_l1_ot"]].reused_by(tensor_list["tensor_xh_l1_it"],
                                                tensor_list["tensor_xh_l1_ft"],
                                                tensor_list["tensor_xh_l1_jt"])
    s[tensor_list["tensor_xh_l0a_ot"]].reused_by(tensor_list["tensor_xh_l0a_it"],
                                                 tensor_list["tensor_xh_l0a_ft"],
                                                 tensor_list["tensor_xh_l0a_jt"])

    # handle matmul info
    mad_pattern = tbe_platform.GEMM_MODE
    # split matmul
    symbol = ["ot", "it", "jt", "ft"]
    l1_factor = tilling_info["l1_factor"]
    for t in symbol:
        s[tensor_list["tensor_b_loc_" + t]].reused_by(
            tensor_list["tensor_matmul_l0c_" + t],
            tensor_list["tensor_matmul_result_l0c_" + t])
        tmp = tensor_list["tensor_matmul_l0c_" + t]

        block_n_o, block_n_i = s[tmp].split(tmp.op.axis[1],
                                            factor=tilling_info["block_n_factor"])
        block_out_o, block_out_i = s[tmp].split(tmp.op.axis[0],
                                                factor=tilling_info["block_out_factor"])
        l1_n_outer, l1_n_inner = s[tmp].split(block_n_i,
                                              factor=tilling_info["n_factor"])  # safe
        l1_out_outer, l1_out_inner = s[tmp].split(block_out_i,
                                                  factor=tilling_info["out_factor"])
        l1_k_outer, l1_k_inner = s[tmp].split(tmp.op.reduce_axis[0],
                                              factor=tilling_info["k_factor"])
        l0_n_outer, l0_n_inner = s[tmp].split(l1_n_inner,
                                              factor=tilling_info["n_factor"])
        l0_out_outer, l0_out_inner = s[tmp].split(l1_out_inner,
                                                  factor=tilling_info["out_factor"])

        l0_k_outer, l0_k_inner = s[tmp].split(l1_k_inner,
                                              factor=tilling_info["k_factor"])
        s[tmp].reorder(block_n_o, block_out_o, l1_n_outer, l1_out_outer,
                       l1_k_outer, l0_n_outer, l0_out_outer, l0_k_outer,
                       l0_n_inner, l0_out_inner, tmp.op.axis[2],
                       tmp.op.axis[3], l0_k_inner, tmp.op.reduce_axis[1])
        s[tensor_list["tensor_xh_l0a_" + t]].compute_at(s[tmp], l0_k_outer)
        s[tensor_list["tensor_w_l0b_" + t]].compute_at(s[tmp], l0_k_outer)
        if l1_factor != 1:
            s[tensor_list["tensor_xh_l1_" + t]].split(s[tensor_list["tensor_xh_l1_" + t]].op.axis[1], factor=l1_factor)
        s[tensor_list["tensor_xh_l1_" + t]].compute_at(s[tmp], l1_k_outer)
        s[tensor_list["tensor_w_l1_" + t]].compute_at(s[tmp], l1_k_outer)
        mad_dict = {"mad_pattern": mad_pattern,
                    "k_outer": [l1_k_outer, l0_k_outer],
                    "init_bias": 1}
        s[tmp].emit_insn(l0_n_inner, 'mad', mad_dict)

    # split ht  origin linmu
    ht_1 = ht.shape[1].value
    axis_1_o, axis_1_i = s[ht].split(ht.op.axis[1], factor=tilling_info["block_n_factor"])

    axis_1_i_0, axis_1_i_i = s[ht].split(axis_1_i, factor=tilling_info["n_factor"])
    axis_0_o, axis_0_i = s[ht].split(ht.op.axis[0],
                                     factor=tilling_info["block_out_factor"])
    axis_0_o_o, axis_0_o_i = s[ht].split(axis_0_o, factor=1)
    axis_0_i_o, axis_0_i_i = s[ht].split(axis_0_i, factor=tilling_info["out_factor"])

    s[ht].reorder(axis_1_o, axis_0_o_o, axis_0_o_i,
                  axis_1_i_0, axis_0_i_o, axis_1_i_i, axis_0_i_i)
    compute_at_axis = axis_0_o_i

    for t in symbol:
        s[tensor_list["tensor_xh_l1_" + t]].double_buffer()
        s[tensor_list["tensor_w_l1_" + t]].double_buffer()
        s[tensor_list["tensor_b_ub_" + t]].double_buffer()

    s[tensor_list["tensor_c_ub"]].double_buffer()

    s[tensor_list["it_ub"]].double_buffer()
    s[tensor_list["ft_ub"]].double_buffer()
    s[tensor_list["ot_ub"]].double_buffer()
    s[tensor_list["jt_ub"]].double_buffer()

    block_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    if (ht_1 // tilling_info["block_n_factor"]) > 1:
        core_outer = s[ht].split(axis_1_o, nparts=block_num)
        s[ht].bind(core_outer[0], tvm.thread_axis("blockIdx.x"))
    else:
        core_outer = s[ht].split(axis_0_o_o, nparts=block_num)
        s[ht].bind(core_outer[0], tvm.thread_axis("blockIdx.x"))

    special_symbol = {"tensor_xh_l0a_it", "tensor_xh_l0a_ft",
                      "tensor_xh_l0a_ot", "tensor_xh_l0a_jt",
                      "tensor_w_l0b_it", "tensor_w_l0b_ft",
                      "tensor_w_l0b_ot", "tensor_w_l0b_jt",
                      "tensor_xh_l1_it", "tensor_xh_l1_ft",
                      "tensor_xh_l1_ot", "tensor_xh_l1_jt",
                      "tensor_w_l1_it", "tensor_w_l1_ft",
                      "tensor_w_l1_ot", "tensor_w_l1_jt", "ht"}

    for key in tensor_list.keys():
        if key not in special_symbol:
            s[tensor_list[key]].compute_at(s[ht], compute_at_axis)

    # Move result back (Fake)
    tensor_list["it_ub_fake_ub"] = s.cache_read(tensor_list["it_ub_fake_true"], tbe_platform.scope_ubuf,
                                                [tensor_list["tensor_ij_ub"]])
    tensor_list["jt_ub_fake_ub"] = s.cache_read(tensor_list["jt_ub_fake_true"], tbe_platform.scope_ubuf,
                                                [tensor_list["tensor_ij_ub"]])
    tensor_list["ft_ub_fake_ub"] = s.cache_read(tensor_list["ft_ub_fake_true"], tbe_platform.scope_ubuf,
                                                [tensor_list["tensor_cf_ub"]])
    tensor_list["ot_ub_fake_ub"] = s.cache_read(tensor_list["ot_ub_fake_true"], tbe_platform.scope_ubuf,
                                                [tensor_list["tensor_ht_ub"]])
    for t in ["ot", "it", "ft", "jt"]:
        s[tensor_list[t + "_ub_fake_ub"]].compute_at(s[ht], compute_at_axis)
        s[tensor_list[t + "_ub2gm"]].reused_by(tensor_list[t + "_ub_fake_ub"], tensor_list[t + "_ub_fake_true"])

        if unreused_flag:
            s[tensor_list[t + "_ub2gm"]].unreused_by(tensor_list["tensor_" + t + "_ub_true"])

        s[tensor_list[t + "_ub_fake_ub"]].reused_by(reuse_data=True)

        s[tensor_list[t + "_ub_fake_ub"]].emit_insn(s[tensor_list[t + "_ub_fake_ub"]].op.axis[0], 'phony_insn')
        s[tensor_list[t + "_ub_fake"]].compute_inline()

    # ct
    tensor_ct_ub_fake_ub = s.cache_read(tensor_list["tensor_ct_ub_fake"], tbe_platform.scope_ubuf,
                                        [tensor_list["tensor_ct_ub_fake_true"]])

    s[tensor_ct_ub_fake_ub].compute_at(s[ht], compute_at_axis)
    s[tensor_list["tensor_ct_ub"]].reused_by(tensor_ct_ub_fake_ub, tensor_list["tensor_ct_ub_fake_true"])
    s[tensor_ct_ub_fake_ub].emit_insn(s[tensor_ct_ub_fake_ub].op.axis[0], 'phony_insn')
    s[tensor_list["tensor_ct_ub_fake"]].compute_inline()

    # tanhct
    tensor_tanhct_ub_fake_ub = s.cache_read(tensor_list["tensor_tanhct_ub_fake"], tbe_platform.scope_ubuf,
                                            [tensor_list["tensor_tanhct_ub_fake_true"]])

    s[tensor_tanhct_ub_fake_ub].compute_at(s[ht], compute_at_axis)
    s[tensor_list["tensor_ub_tanh_ct"]].reused_by(tensor_list["tensor_tanhct_ub_fake_true"])
    s[tensor_tanhct_ub_fake_ub].emit_insn(s[tensor_tanhct_ub_fake_ub].op.axis[0], 'phony_insn')
    s[tensor_list["tensor_tanhct_ub_fake"]].compute_inline()
    # ht
    s[ht].emit_insn(s[ht].op.axis[2], 'dma_copy')

    build_symbol = ["x", "h", "c", "w", "b", "ct", "ht", "it", "jt", "ft", "ot", "tanhct"]
    new_build_list = []
    for t in build_symbol:
        if t in build_list.keys():
            new_build_list += [build_list[t]]
    with tbe_build.build_config():
        tvm.build(s, new_build_list, "cce", name=kernel_name)


# Currently not support fusion with dropout
# 'pylint: disable=locally-disabled,too-many-statements,unused-argument,too-many-arguments,too-many-locals
# 'pylint: disable=unnecessary-lambda,invalid-name,too-many-branches,consider-iterating-dictionary
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def basic_lstm_cell(x, h, c, w, b, mask, ct, ht, it, jt, ft, ot, tanhct,
                    keep_prob=1.0, forget_bias=1.0, state_is_tuple=True,
                    activation="tanh", kernel_name="BasicLSTMCell"):
    """
    the main function of the basic_lstm_cell
    Parameters
    ----------
    x : matmul left x
    h : matmul left h
    c : lstm cell state last time
    w : matmul right w
    b : matmul bias
    ct : lstm cell state this time
    ht : lstm cell output
    it : input gate
    jt : new gate
    ft : forget gate
    ot : output gate
    tanhct : tanh(ot)
    keep_prob : dropout Percentage
    forgat_bias : bias of forget_gate, default: 1.0
    state_is_tuple : x and h is tuple, default: true
    activation : activation function, default: tanh
    kernel_name : kernal_name, default: BasicLSTMCell
    Returns
    -------
    """
    basiclstm_cell_check(x, h, c, w, b, ct, ht, it, ft, jt, ot, tanhct)

    is_hisi_es = False
    is_mini = False
    is_cloud = False
    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if tbe_product == "Ascend910":
        is_cloud = True
    elif tbe_product in ("Ascend310P", "Ascend610"):
        is_cloud = True
    elif tbe_product == "Ascend310":
        is_mini = True
    else:
        is_hisi_es = True

    product_info = {}
    product_info["hisi_es"] = is_hisi_es
    product_info["mini"] = is_mini
    product_info["cloud"] = is_cloud
    build_list, tensor_list, scope_list, operation_list, unreused_flag = basic_lstm_cell_compute(x,
                                                                                                 h, c, w, b,
                                                                                                 forget_bias,
                                                                                                 product_info)

    tilling_info = get_tilling(x, h, product_info)
    basic_lstm_cell_schedule(tensor_list, scope_list, operation_list, build_list,
                             tilling_info, unreused_flag, kernel_name)
