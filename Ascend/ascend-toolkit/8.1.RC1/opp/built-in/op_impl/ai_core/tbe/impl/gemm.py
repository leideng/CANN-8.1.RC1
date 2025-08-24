#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
gemm
"""

from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_common import ceil
from impl.util.util_gemm import get_op_support_info_gemm
from impl.util.util_gemm import get_prebuild_pattern


ALPHA_BETA_SHAPE = [1]
MAX_INT32_LENGTH = 2147483647
L1FUSION_INPUT_CTR = 2

ND_LEN = 2
NZ_LEN = 4


def _check_param(input_x1, input_x2, bias, trans_a, trans_b):
    """
    the params check of gemm
    """
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")
    bias_shape = bias.get("ori_shape")

    if len(shape_a) not in (ND_LEN, NZ_LEN):
        error_detail = "len(shape_a) not in (2, 4), len(shape_a)=%s" % len(shape_a)
        error_manager_vector.raise_err_input_shape_invalid("gemm", "A", error_detail)

    if len(shape_b) not in (ND_LEN, NZ_LEN):
        error_detail = "len(shape_b) not in (2, 4), len(shape_b)=%s" % len(shape_b)
        error_manager_vector.raise_err_input_shape_invalid("gemm", "A", error_detail)

    km_shape, m_shape = shape_a[1], shape_a[0]
    if trans_a:
        km_shape, m_shape = shape_a[0], shape_a[1]
    if input_x1.get("ori_format") == "FRACTAL_NZ":
        km_shape, m_shape = (m_shape, km_shape)

    kn_shape, n_shape = shape_b[0], shape_b[1]
    if trans_b:
        kn_shape, n_shape = shape_b[1], shape_b[0]
    if input_x2.get("ori_format") == "FRACTAL_NZ":
        kn_shape, n_shape = (n_shape, kn_shape)

    block_out = tbe_platform.BLOCK_OUT
    block_in = tbe_platform.BLOCK_IN
    block_reduce = tbe_platform.BLOCK_REDUCE if input_x1.get("dtype") == "float16" else tbe_platform.BLOCK_REDUCE_INT8
    if input_x1.get("ori_format") == "ND":
        km_shape = ceil(km_shape, block_reduce)
        m_shape = ceil(m_shape, block_in)
    if input_x2.get("ori_format") == "ND":
        kn_shape = ceil(kn_shape, block_reduce)
        n_shape = ceil(n_shape, block_out)
    bias_shape = list(bias_shape)
    if bias.get("ori_format") == "ND":
        bias_shape[0] = ((bias_shape[0] + block_in - 1) // block_in)
        bias_shape[1] = ((bias_shape[1] + block_out - 1) // block_out)
    if km_shape != kn_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal("gemm", "km_shape", "kn_shape",
                                                                 km_shape, kn_shape, kn_shape)
    if bias.get("ori_format") == "FRACTAL_NZ":
        bias_shape = [bias_shape[1], bias_shape[0]]
    if list(bias_shape)[:2] != [m_shape, n_shape]:
        error_manager_vector.raise_err_input_value_invalid("gemm", "c shape", str([m_shape, n_shape]), str(bias_shape))


def check_supported(
    input_x1,
    input_x2,
    bias,
    alpha,
    beta,
    output_y=None,
    trans_a=False,
    trans_b=False,
    kernel_name="gemm"):
    """
    the k-dims of input_x1 must be equal with that of input_x2
    the m-dims of input_x1 must be equal with that of bias
    the n-dims of input_x2 must be equal with that of bias
    """
    try:
        _check_param(input_x1, input_x2, bias, trans_a, trans_b)
        return True, ""
    except Exception:
        reason = "the input_shape is not supported, input_x1_shape:%s, input_x2_shape:%s, "\
                 "bias_shape:%s, trans_a:%s, trans_b:%s"\
                 % (input_x1.get("ori_shape"), input_x2.get("ori_shape"), bias.get("ori_shape"), trans_a, trans_b)
        return False, reason


def get_op_specific_info(input_x1,
    input_x2,
    alpha,
    beta,
    bias=None,
    output_y=None,
    trans_a=False,
    trans_b=False,
    kernel_name="gemm",):
    """
    get the gemm prebuild pattern

    """
    return get_prebuild_pattern(input_x1, op_pattern="GEMM")


def op_select_format(
    input_x1,
    input_x2,
    alpha,
    beta,
    bias=None,
    output_y=None,
    trans_a=False,
    trans_b=False,
    kernel_name="gemm",
):
    """
    Select format according to the following rules.
    1.When ori_format of input_x1 and input_x2 is ND, dtype of input_x1 and input_x2 is float16,
    and ori_shape[n_dim] is a multiple of 16, Op select supports the following format combination:
    |        |  input_x1   |  input_x2  |    bias    |   alpha   |   beta   |  output_y  |
    | :----: |   :----:    |   :----:   |   :----:   |   :----:  |  :----:  |   :----:   |
    | Format | FRACTAL_NZ  | FRACTAL_NZ | FRACTAL_NZ |     ND    |    ND    | FRACTAL_NZ |
    | Dtype  |  float16    |  float16   |  float16   |  float16  |  float16 |  float16   |
    |        | FRACTAL_NZ  | FRACTAL_NZ | FRACTAL_NZ |     ND    |    ND    | FRACTAL_NZ |
    |        |  float16    |  float16   |  float32   |  float32  |  float32 |  float32   |
    |        |     ND      |    ND      |    ND      |     ND    |    ND    |    ND      |
    |        |  float16    |  float16   |  float16   |  float16  |  float16 |  float16   |
    |        |    ND       |    ND      |    ND      |     ND    |    ND    |    ND      |
    |        |  float16    |  float16   |  float32   |  float32  |  float32 |  float32   |
    Example:
    ori_inputs:
    - input_x1 = Tensor(shape(16, 32), dtype="float16", format="ND")
    - input_x2 = Tensor(shape(32, 48), dtype="float16", format="ND")
    - bias = Tensor(shape(16, 48), dtype="float16", format="ND")
    - alpha = Tensor(shape(1, ), dtype="float16", format="ND")
    - beta = Tensor(shape(1,), dtype="float16", format="ND")

    Op Select can work with FRACTAL_NZ:
    - input_x1 = Tensor(shape(2, 1, 16, 16),  dtype="float16", format="FRACTAL_NZ")
    - input_x2 = Tensor(shape(3, 2, 16, 16),  dtype="float16", format="FRACTAL_NZ")
    - bias = Tensor(shape(3, 1, 16, 16)),  dtype="float16", format="FRACTAL_NZ")
    - alpha = Tensor(shape(1, ),  dtype="float16", format="ND")
    - beta = Tensor(shape(1,),  dtype="float16", format="ND")

    2.When ori_format of input_x1 and input_x2 is ND, dtype of input_x1 and input_x2 is int8,
    and ori_shape[n_dim] is a multiple of 16, Op select supports the following format combination:
    |        |  input_x1   |  input_x2  |    bias    |   alpha   |   beta   |  output_y  |
    | :----: |   :----:    |   :----:   |   :----:   |   :----:  |  :----:  |   :----:   |
    | Format | FRACTAL_NZ  | FRACTAL_Z  |     ND     |     ND    |    ND    | FRACTAL_NZ |
    | Dtype  |    int8     |   int8     |   int32    |   int32   |   int32  |   int32    |
    |        | FRACTAL_NZ  | FRACTAL_Z  | FRACTAL_NZ |     ND    |    ND    | FRACTAL_NZ |
    |        |    int8     |   int8     |  float32   |  float32  |  float32 |  float32   |
    |        |     ND      |    ND      |    ND      |     ND    |    ND    |    ND      |
    |        |   int8      |   int8     |   int32    |    int32  |  int32   |   int32    |
    |        |    ND       |    ND      |    ND      |     ND    |    ND    |    ND      |
    |        |   int8      |   int8     |  float32   |  float32  |  float32 |  float32   |

    Example:
    ori_inputs:
    - input_x1 = Tensor(shape(16, 32), dtype="int8", format="ND")
    - input_x2 = Tensor(shape(32, 48), dtype="int8", format="ND")
    - bias = Tensor(shape(16, 48), dtype="int32", format="ND")
    - alpha = Tensor(shape(1, ), dtype="int32", format="ND")
    - beta = Tensor(shape(1,), dtype="int32", format="ND")

    Op Select can work  with FRACTAL_NZ and FRACTAL_Z:
    - input_x1 = Tensor(shape(1, 1, 16, 32),  dtype="int8", format="FRACTAL_NZ")
    - input_x2 = Tensor(shape(3, 1, 32, 16),  dtype="int8", format="FRACTAL_Z")
    - bias = Tensor(shape(16, 48)),  dtype="int32", format="ND")
    - alpha = Tensor(shape(1, ),  dtype="int32", format="ND")
    - beta = Tensor(shape(1,),  dtype="int32", format="ND")

    3.When ori_format of input_x1 and input_x2 is ND, dtype of input_x1 and input_x2 is float16,
    and ori_shape[n_dim] is not a multiple of 16, Op select supports the following format combination:
    |        |  input_x1   |  input_x2  |    bias    |   alpha   |   beta   |  output_y  |
    | :----: |   :----:    |   :----:   |   :----:   |   :----:  |  :----:  |   :----:   |
    | Format | FRACTAL_NZ  | FRACTAL_NZ | FRACTAL_NZ |     ND    |    ND    | FRACTAL_NZ |
    | Dtype  |  float16    |  float16   |  float16   |  float16  |  float16 |  float16   |
    |        | FRACTAL_NZ  | FRACTAL_NZ | FRACTAL_NZ |     ND    |    ND    | FRACTAL_NZ |
    |        |  float16    |  float16   |  float32   |  float32  |  float32 |  float32   |
    Example:
    ori_inputs:
    - input_x1 = Tensor(shape(16, 32), dtype="float16", format="ND")
    - input_x2 = Tensor(shape(32, 52), dtype="float16", format="ND")
    - bias = Tensor(shape(16, 52), dtype="float16", format="ND")
    - alpha = Tensor(shape(1, ), dtype="float16", format="ND")
    - beta = Tensor(shape(1,), dtype="float16", format="ND")

    Op Select can work with FRACTAL_NZ:
    - input_x1 = Tensor(shape(2, 1, 16, 16),  dtype="float16", format="FRACTAL_NZ")
    - input_x2 = Tensor(shape(4, 2, 16, 16),  dtype="float16", format="FRACTAL_NZ")
    - bias = Tensor(shape(4, 1, 16, 16)),  dtype="float16", format="FRACTAL_NZ")
    - alpha = Tensor(shape(1, ),  dtype="float16", format="ND")
    - beta = Tensor(shape(1,),  dtype="float16", format="ND")

    4.When ori_format of input_x1 and input_x2 is ND, dtype of input_x1 and input_x2 is int8,
    and ori_shape[n_dim] is a multiple of 16, Op select supports the following format combination:
    |        |  input_x1   |  input_x2  |    bias    |   alpha   |   beta   |  output_y  |
    | :----: |   :----:    |   :----:   |   :----:   |   :----:  |  :----:  |   :----:   |
    | Format | FRACTAL_NZ  | FRACTAL_Z  |     ND     |     ND    |    ND    | FRACTAL_NZ |
    | Dtype  |    int8     |   int8     |   int32    |   int32   |   int32  |   int32    |
    |        | FRACTAL_NZ  | FRACTAL_Z  | FRACTAL_NZ |     ND    |    ND    | FRACTAL_NZ |
    |        |    int8     |   int8     |  float32   |  float32  |  float32 |  float32   |

    for example:
    ori_inputs:
    - input_x1 = Tensor(shape(16, 32), dtype="int8", format="ND")
    - input_x2 = Tensor(shape(32, 52), dtype="int8", format="ND")
    - bias = Tensor(shape(16, 52), dtype="int32", format="ND")
    - alpha = Tensor(shape(1, ), dtype="int32", format="ND")
    - beta = Tensor(shape(1,), dtype="int32", format="ND")

    Op Select can process with FRACTAL_NZ and FRACTAL_Z:
    - input_x1 = Tensor(shape(1, 1, 16, 32),  dtype="int8", format="FRACTAL_NZ")
    - input_x2 = Tensor(shape(4, 1, 32, 16),  dtype="int8", format="FRACTAL_Z")
    - bias = Tensor(shape(16, 51)),  dtype="int32", format="ND")
    - alpha = Tensor(shape(1, ),  dtype="int32", format="ND")
    - beta = Tensor(shape(1,),  dtype="int32", format="ND")
    """

    def _select_format(params):
        input_x1 = params[0]
        input_x2 = params[1]
        shape_b = input_x2.get("ori_shape")
        format_a = input_x1.get("format")
        format_b = input_x2.get("format")
        format_c = bias.get("format")
        need_transdata = False
        if {format_a, format_b, format_c} & {"FRACTAL_NZ", "FRACTAL_Z"}:
            need_transdata = True
        else:
            if trans_b:
                b_n = shape_b[0]
            else:
                b_n = shape_b[1]
            if b_n % tbe_platform.BLOCK_OUT != 0:
                need_transdata = True

        if need_transdata:
            input0 = util_select_op_base.gen_param(
                classify="input0",
                name="a",
                datatype="float16,float16,int8,int8",
                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ",
            )
            input1 = util_select_op_base.gen_param(
                classify="input1",
                name="b",
                datatype="float16,float16,int8,int8",
                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_Z,FRACTAL_Z",
            )
            input2 = util_select_op_base.gen_param(
                classify="input2",
                name="c",
                datatype="float32,float16,int32,float32",
                format="FRACTAL_NZ,FRACTAL_NZ,ND,FRACTAL_NZ",
            )
            output0 = util_select_op_base.gen_param(
                classify="output0",
                name="y",
                datatype="float32,float16,int32,float32",
                format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ",
            )
        else:
            input0 = util_select_op_base.gen_param(
                classify="input0",
                name="a",
                datatype="float16,float16,int8,int8",
                format="ND,ND,ND,ND",
            )
            input1 = util_select_op_base.gen_param(
                classify="input1",
                name="b",
                datatype="float16,float16,int8,int8",
                format="ND,ND,ND,ND",
            )
            input2 = util_select_op_base.gen_param(
                classify="input2",
                name="c",
                datatype="float32,float16,int32,float32",
                format="ND,ND,ND,ND",
            )
            output0 = util_select_op_base.gen_param(
                classify="output0",
                name="y",
                datatype="float32,float16,int32,float32",
                format="ND,ND,ND,ND",
            )
        input3 = util_select_op_base.gen_param(
            classify="input3",
            name="alpha",
            datatype="float32,float16,int32,float32",
            format="ND,ND,ND,ND",
        )
        input4 = util_select_op_base.gen_param(
            classify="input4",
            name="beta",
            datatype="float32,float16,int32,float32",
            format="ND,ND,ND,ND",
        )
        return [input0, input1, input2, input3, input4, output0]

    params = [
        input_x1,
        input_x2,
        alpha,
        beta,
        bias,
        output_y,
        trans_a,
        trans_b,
        kernel_name,
    ]
    param_list = _select_format(params)
    return util_select_op_base.get_dynamic_param_in_json(param_list)


def get_op_support_info(input_x1,
                        input_x2,
                        bias,
                        alpha,
                        beta,
                        output_y=None,
                        trans_a=False,
                        trans_b=False,
                        kernel_name="gemm",):
    """
    get the GEMM split, which only split the m and n, cannot cut k for c

    """
    inputs = [input_x1, input_x2, bias]
    return get_op_support_info_gemm(inputs, output_y, trans_a, trans_b, op_type="gemm")


def _dtype_check(inputs_dtype_list, dst_dtype):
    """
    check the dtype of gemm
    """
    a_dtype, b_dtype, bias_dtype, alpha_dtype, beta_dtype = inputs_dtype_list
    if a_dtype != b_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("gemm", "a", "b", a_dtype, b_dtype)
    if alpha_dtype != beta_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("gemm", "alpha", "beta", alpha_dtype, beta_dtype)
    if alpha_dtype != dst_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("gemm", "alpha", "y", alpha_dtype, dst_dtype)
    if a_dtype == "int8":
        if dst_dtype not in ["int32", "float32"]:
            error_manager_vector.raise_err_dtype_invalid("gemm", "dst_dtype", "int32, float32", dst_dtype)
    elif a_dtype == "float16":
        if dst_dtype not in ["float16", "float32"]:
            error_manager_vector.raise_err_dtype_invalid("gemm", "dst_dtype", "float16, float32", dst_dtype)
    else:
        error_manager_vector.raise_err_dtype_invalid("gemm", "src_dtype", "float16, int8", a_dtype)
    if bias_dtype != dst_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("gemm", "c", "y", bias_dtype, dst_dtype)


def _shape_check(
    shape_a,
    shape_b,
    trans_a,
    trans_b
):
    """
    Check the given input if legal

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
    src_dtype: str
            The data type of input, support "float32", "float16"
    trans_a: bool
            If True, shape_a == transposed before multiplication
    trans_b: bool
            If True, shape_b == transposed before multiplication
    alpha_dtype: str
            The data type of alpha
    beta_dtype: str
            The data type of beta
    dst_dtype: str
            The data type of dst
    bias_dtype: str
            The data type of bias
    Returns None
    """
    if len(shape_a) != ND_LEN and len(shape_a) != NZ_LEN:
        error_detail = "len(shape_a) not in (2, 4), len(shape_a)=%s" % len(shape_a)
        error_manager_vector.raise_err_input_shape_invalid("gemm", "A", error_detail)

    if len(shape_b) != ND_LEN and len(shape_b) != NZ_LEN:
        error_detail = "len(shape_b) not in (2, 4), len(shape_b)=%s" % len(shape_b)
        error_manager_vector.raise_err_input_shape_invalid("gemm", "A", error_detail)

    if len(shape_a) == ND_LEN and len(shape_b) == ND_LEN:
        km_shape = shape_a[0] if trans_a else shape_a[1]
        kn_shape = shape_b[1] if trans_b else shape_b[0]
        if km_shape != kn_shape:
            error_manager_vector.raise_err_inputs_shape_not_equal("gemm", "k_dim of x2", kn_shape, "k_dim of x1",
                                                                  km_shape, km_shape)


def _format_check(
        src_dtype,
        dst_dtype,
        format_a,
        format_b,
        format_bias,
        format_alpha,
        format_beta,
        format_dst
):
    """
    src_dtype: str
        input data type
    dst_type: str
        output data type
    format_a: str
        format of input a
    format_b: str
        format of input b
    format_bias: str
        format of input c
    format_alpha: str
        format of input alpha
    format_beta: str
        format of input beta
    format_dst: str
        format of output
    """
    flow_type = src_dtype + dst_dtype
    format_combine = [
        format_a,
        format_b,
        format_bias,
        format_alpha,
        format_beta,
        format_dst
    ]
    if format_combine == ["ND", "ND", "ND", "ND", "ND", "ND"]:
        return
    support_combine = {
        "float16float32":
            ["FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "ND", "ND", "FRACTAL_NZ"],
        "float16float16":
            ["FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "ND", "ND", "FRACTAL_NZ"],
        "int8int32":
            ["FRACTAL_NZ", "FRACTAL_Z", "ND", "ND", "ND", "FRACTAL_NZ"],
        "int8float32":
            ["FRACTAL_NZ", "FRACTAL_Z", "FRACTAL_NZ", "ND", "ND", "FRACTAL_NZ"],
    }
    if (support_combine.get(flow_type) is not None and support_combine.get(flow_type) != format_combine):
        error_detail = "for src_dtype = %s and dst_type = %s, format need to be %s or %s" % (src_dtype,
                         dst_dtype, "ND, ND, ND, ND, ND, ND", support_combine.get(flow_type))
        error_manager_vector.raise_err_specific_reson("gemm", error_detail)


def _get_input_a(input_a, shape_a, dtype_a, trans_a):
    """
    get input x1 of gemm
    """
    if len(shape_a) == ND_LEN:
        if input_a.get("format") != "ND":
            m1_shape = ceil(shape_a[0], tbe_platform.BLOCK_IN)
            k1_shape = ceil(shape_a[1], tbe_platform.CUBE_MKN[dtype_a]["mac"][1])
            # trans can not be false
            trans_a = not trans_a
            shape_a_temp = [k1_shape, m1_shape, tbe_platform.BLOCK_IN, tbe_platform.CUBE_MKN[dtype_a]["mac"][1]]
            format_a = "FRACTAL_NZ"
        else:
            shape_a_temp = shape_a
            format_a = "ND"
    else:
        trans_a = not trans_a
        shape_a_temp = shape_a
        format_a = "FRACTAL_NZ"
    return shape_a_temp, trans_a, format_a


def _get_input_b(input_b, shape_b, dtype_b, trans_b):
    """
    get input x2 of gemm
    """
    if len(shape_b) == ND_LEN:
        if input_b.get("format") != "ND":
            k1_shape = ceil(shape_b[0], tbe_platform.CUBE_MKN[dtype_b]["mac"][1])
            n1_shape = ceil(shape_b[1], tbe_platform.BLOCK_OUT)
            format_b = input_b.get("format")
            if format_b == "FRACTAL_NZ":
                trans_b = not trans_b
            if trans_b:
                shape_b_temp = [n1_shape, k1_shape, tbe_platform.CUBE_MKN[dtype_b]["mac"][1], tbe_platform.BLOCK_OUT]
            else:
                shape_b_temp = [k1_shape, n1_shape, tbe_platform.BLOCK_OUT, tbe_platform.CUBE_MKN[dtype_b]["mac"][1]]
        else:
            shape_b_temp = shape_b
            format_b = "ND"
    else:
        shape_b_temp = shape_b
        format_b = input_b.get("format")
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b
    return shape_b_temp, trans_b, format_b


def _get_bias(bias):
    """
    align bias shape in NZ format
    """
    shape_bias = list(bias.get("ori_shape"))
    if bias.get("format") == "ND" and bias.get("ori_format") != "ND":
        shape_bias = list(bias.get("shape"))

    shape_bias_temp = shape_bias
    if bias.get("format") != "ND" and len(shape_bias) == ND_LEN:
        shape_bias_temp = (
            ceil(shape_bias[1], tbe_platform.BLOCK_OUT),
            ceil(shape_bias[0], tbe_platform.BLOCK_IN),
            tbe_platform.BLOCK_IN,
            tbe_platform.BLOCK_OUT,
        )
    return shape_bias_temp


def _bias_check(input_x1, input_x2, bias, trans_a, trans_b):
    if input_x1["ori_format"] == "ND" and input_x2["ori_format"] == "ND" and bias["ori_format"] == "ND":
        shape_a = list(input_x1["ori_shape"])
        shape_b = list(input_x2["ori_shape"])
        shape_bias = list(bias["ori_shape"])
        a_m = shape_a[1] if trans_a else shape_a[0]
        b_n = shape_b[0] if trans_b else shape_b[1]
        if shape_bias != [a_m, b_n]:
            error_detail = "c shape not in (a_m, b_n), c shape=%s" % shape_bias
            error_manager_vector.raise_err_input_shape_invalid("gemm", "A", error_detail)

    else:
        shape_a = list(input_x1["shape"])
        shape_b = list(input_x2["shape"])
        shape_bias = list(bias["shape"])
        if len(shape_bias) == ND_LEN:
            shape_bias = [
                ceil(shape_bias[1], tbe_platform.BLOCK_OUT),
                ceil(shape_bias[0], tbe_platform.BLOCK_IN),
            ]
        else:
            shape_bias = shape_bias[:2]
        if input_x2["dtype"] == "int8" and shape_bias != [shape_b[1], shape_a[1]]:
            error_detail = "c shape not in %s, c shape=%s" % (str([shape_a[1], shape_b[1]]), shape_bias)
            error_manager_vector.raise_err_input_shape_invalid("gemm", "c shape", error_detail)
        if input_x2["dtype"] == "float16" and shape_bias != [shape_b[0], shape_a[1]]:
            error_detail = "c shape not in %s, c shape=%s" % (str([shape_a[1], shape_b[0]]), shape_bias)
            error_manager_vector.raise_err_input_shape_invalid("gemm", "c shape", error_detail)
    if len(shape_bias) != ND_LEN and len(shape_bias) != NZ_LEN:
        error_detail = "len(bias_shape) not in (2, 4), len(bias_shape)=%s" % len(shape_bias)
        error_manager_vector.raise_err_input_shape_invalid("gemm", "c", error_detail)


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_BOOL,
    para_check.REQUIRED_ATTR_BOOL,
    para_check.KERNEL_NAME,
)
def gemm(
    input_x1,
    input_x2,
    bias,
    alpha,
    beta,
    output_y=None,
    trans_a=False,
    trans_b=False,
    kernel_name="gemm",
):
    """
    calculating  matrix multiplication with bias, C = alpha*A*B + beta*bias, support input
    data with Nz format.

    Parameters:
    input_x1: dict
            shape and dtype of tensor_a
    input_x2: dict
            shape and dtype of tensor_b
    alpha: shape and dtype of alpha
    beta: shape and dtype of beta
    bias: dict
            Shape of bias, support the input data format with Nz/ND in different scenes
    trans_a:
            whether transpose a
            only support false
    trans_b:
            whether transpose b
            only support false
    Returns
    -------
    None
    """
    # 当ab不都为ND格式时，由外层处理transpose
    if input_x1.get("format") != "ND" or input_x2.get("format") != "ND":
        trans_a = False
        trans_b = False

    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")
    src_dtype = input_x1.get("dtype").lower()
    b_dtype = input_x2.get("dtype").lower()
    dst_dtype = output_y.get("dtype").lower()
    bias_dtype = bias.get("dtype").lower()
    if shape_a is not None and len(shape_a) < ND_LEN:
        shape_a = input_x1.get("shape")

    if shape_b is not None and  len(shape_b) < ND_LEN:
        shape_b = input_x2.get("shape")

    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_a)
    para_check.check_shape_rule(shape_b)

    alpha_dtype = alpha.get("dtype")
    beta_dtype = beta.get("dtype")

    _shape_check(shape_a, shape_b, trans_a, trans_b)
    _dtype_check([src_dtype, b_dtype, bias_dtype, alpha_dtype, beta_dtype], dst_dtype)
    _bias_check(input_x1, input_x2, bias, trans_a, trans_b)
    _format_check(
        src_dtype,
        dst_dtype,
        input_x1.get("format"),
        input_x2.get("format"),
        bias.get("format"),
        alpha.get("format"),
        beta.get("format"),
        output_y.get("format")
    )

    shape_a_temp, trans_a, format_a = _get_input_a(input_x1, shape_a, src_dtype, trans_a)
    shape_b_temp, trans_b, format_b = _get_input_b(input_x2, shape_b, b_dtype, trans_b)
    shape_bias_temp = _get_bias(bias)

    def _gemm_local_compute():
        tensor_a = tvm.placeholder(shape_a_temp, name="tensor_a", dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b_temp, name="tensor_b", dtype=src_dtype)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=alpha_dtype)
        tensor_bias = tvm.placeholder(shape_bias_temp, name="tensor_bias", dtype=dst_dtype)
        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias,
            "kernel_name": kernel_name
        }
        result = tbe.gemm(tensor_a=tensor_a, tensor_b=tensor_b, para_dict=para_dict)

        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)

        tensor_list = [tensor_a, tensor_b, tensor_bias,
                       tensor_alpha, tensor_beta, result]
        config = {
            "print_ir": False,
            "name": kernel_name,
            "tensor_list": tensor_list,
        }
        tbe.build(schedule, config)

    _gemm_local_compute()
