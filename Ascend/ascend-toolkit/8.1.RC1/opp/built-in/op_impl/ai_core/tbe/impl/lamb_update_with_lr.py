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
lamb_update_with_lr
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.common_util import constant
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,invalid-name,unused-variable
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals
def real_div_compute(data_1, data_2, output_z, kernel_name="real_div"):
    """
    calculating data's realdiv, `c = a / b`

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is real_div

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                              param_name_input1="input_greater_realdiv",
                                                              param_name_input2="input_realdiv")
    data_x = tbe.broadcast(data_1, shape_max)
    data_y = tbe.broadcast(data_2, shape_max)
    res = tbe.vdiv(data_x, data_y)

    return res


def select_compute(condition, x1, x2, y, kernel_name="select"):
    """
    compute for select

    Parameters
    ----------
    condition: TVM tensor
        the placeholder of input condition
    x1: TVM tensor
        the placeholder of input x1
    x2: TVM tensor
        the placeholder of input x2
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape = shape_util.shape_to_list(x1.shape)
    con_shape = shape_util.shape_to_list(condition.shape)
    num_dtype = x1.dtype
    bool_dtype = condition.dtype

    if num_dtype in ("int8", "uint8"):
        x1_dtype = "float32"
        ones = tbe.broadcast(tvm.const(1, dtype="float32"),
                             shape, output_dtype="float32")
        x1 = tbe.cast_to(x1, "float32")
        x2 = tbe.cast_to(x2, "float32")
    else:
        x1_dtype = num_dtype
        ones = tbe.broadcast(tvm.const(1, dtype=num_dtype),
                             shape, output_dtype=num_dtype)

    if bool_dtype == "int8":
        if x1_dtype == "int32":
            condition_dtype = tbe.ceil(condition)
        else:
            condition_dtype = tbe.cast_to(condition, x1_dtype)
    else:
        if x1_dtype == "int32":
            condition_dtype = condition
        else:
            condition_dtype = tbe.cast_to(condition, x1_dtype)

    if list(con_shape) != list(shape):
        condition_dtype = tbe.broadcast(condition_dtype, shape)

    condition_opp = tbe.vsub(ones, condition_dtype)

    temp_x = tbe.vmul(x1, condition_dtype)
    temp_y = tbe.vmul(x2, condition_opp)
    res = tbe.vadd(temp_x, temp_y)
    if num_dtype in ("int8", "uint8"):
        res = tbe.cast_to(res, num_dtype)
    return res


def _greater_compare(data, shape, dtype, data_min):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    data : tuple
        two input data
    shape : list or tuple
        shape of input data
    dtype : str
        source data type, support float16,float32,int32,int8,uint8
    data_min : tvm.const
        the minimal data according to dtype

    Returns
    -------
    the compare result
    """
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    if dtype == "int32":
        data_one = \
            tbe.broadcast(tvm.const(1, "float16"), shape, "float16")
    else:
        data_one = tbe.broadcast(tvm.const(1, dtype), shape, dtype)

    res_sub = tbe.vsub(data[1], data[0])
    # to amend sub zero result
    res_sub_zero = tbe.vadd(res_sub, data_min)
    res_min = tbe.vmin(res_sub_zero, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    if dtype == "float32":
        # max num of float32 is 2**126
        # but cce can only support 2**62, so use 62/62/2 to adaptor 126
        max_support_fp32 = tvm.const(2 ** 62, dtype=dtype)
        res_mul1 = tbe.vmuls(res_max, max_support_fp32)
        res_mul2 = tbe.vmuls(res_mul1, max_support_fp32)
        res_mul = tbe.vmuls(res_mul2, tvm.const(2 ** 2, dtype=dtype))
    elif dtype == "float16":
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        max_support_fp16 = tvm.const(2 ** 12, dtype=dtype)
        res_mul1 = tbe.vmuls(res_max, max_support_fp16)
        res_mul = tbe.vmuls(res_mul1, max_support_fp16)
    else:
        res_mul = tbe.cast_to(res_max, "float16")
    res = tbe.vsub(data_one, res_mul)

    return tbe.cast_to(res, "uint8", True)


def greater_compute(x, y, z, kernel_name="greater"):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    x : Tensor
        input data_x
    y : Tensor
        input data_y
    z : dict
        shape and dtype of output data_z
    kernel_name : str
        cce kernel name, default value is "greater"

    Returns
    -------
    the result
    """
    shape_x = shape_util.shape_to_list(x.shape)
    shape_y = shape_util.shape_to_list(y.shape)
    dtype = x.dtype.lower()
    shape_x, shape_y, shape = shape_util.broadcast_shapes(shape_x, shape_y,
                                                          param_name_input1="x",
                                                          param_name_input2="y")

    if dtype in ("int8", "uint8"):
        x = tbe.cast_to(x, "float16")
        y = tbe.cast_to(y, "float16")
        dtype = "float16"

    data_x = tbe.broadcast(x, shape)
    data_y = tbe.broadcast(y, shape)

    if dtype == "float32":
        # minimun num of float32 2**(-126)
        data_min = \
            tbe.broadcast(tvm.const(constant.MIN_FP32, dtype=dtype), shape,
                          dtype)
    elif dtype == "float16":
        # minimun num of float16 2**(-24)
        data_min = \
            tbe.broadcast(tvm.const(constant.MIN_FP16, dtype=dtype), shape,
                          dtype)
    else:
        data_min = tbe.broadcast(tvm.const(1, dtype=dtype), shape,
                                 dtype)

    return _greater_compare((data_x, data_y), shape, dtype, data_min)


def reduce_sum_d_compute(x,
                         y,
                         axis=None,
                         keepdims=None,
                         kernel_name="reduce_sum_d"):
    """reduce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    dtype = x.dtype
    if dtype == "float16" and tbe_platform.api_check_support(
            "tbe.dsl.sum", "float32"):
        x = tbe.cast_to(x, "float32")
    res_sum = tbe.sum(x, axis=axis, keepdims=keepdims)
    res = tbe.cast_to(res_sum, dtype)

    return res


def maximum(input_x, input_y, output_z, kernel_name="maximum"):
    """
    do element-wise maximum operation between two input tensors

    """
    shape1 = shape_util.shape_to_list(input_x.shape)
    shape2 = shape_util.shape_to_list(input_y.shape)
    shape1 = shape_util.scalar2tensor_one(shape1)

    shape2 = shape_util.scalar2tensor_one(shape2)

    shape1, shape2, shape_max = shape_util.broadcast_shapes(shape1, shape2,
                                                            param_name_input1="input_x",
                                                            param_name_input2="input_y")

    data1_tmp1 = tbe.broadcast(input_x, shape_max)
    data2_tmp1 = tbe.broadcast(input_y, shape_max)
    res = tbe.vmax(data1_tmp1, data2_tmp1)
    return res


def minimum(input_x, input_y, output_z, kernel_name="minimum"):
    """
    do element-wise minimum operation between two input tensors
    """
    shape1 = shape_util.shape_to_list(input_x.shape)
    shape2 = shape_util.shape_to_list(input_y.shape)
    shape1 = shape_util.scalar2tensor_one(shape1)
    shape2 = shape_util.scalar2tensor_one(shape2)

    para_check.check_shape(shape1, param_name="input_x")
    para_check.check_shape(shape2, param_name="input_y")

    check_list = ["float16", "float32", "int32"]
    dtype = input_x.dtype
    para_check.check_dtype(dtype, check_list, param_name="input_x")

    shape1, shape2, shape_max = shape_util.broadcast_shapes(shape1, shape2,
                                                            param_name_input1="input_x",
                                                            param_name_input2="input_y")

    data1_tmp1 = tbe.broadcast(input_x, shape_max)
    data2_tmp1 = tbe.broadcast(input_y, shape_max)
    res = tbe.vmin(data1_tmp1, data2_tmp1)
    return res


def sub(input_x, input_y, output_z, kernel_name="sub"):
    """
    do element-wise sub operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32,int32
    input_y : dict
        shape and dtype of input, only support float16, float32,int32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : kernel name, default value is "sub"

    Returns
    -------
    None
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                              param_name_input1="input_x",
                                                              param_name_input2="input_y")
    data1_tmp1 = tbe.broadcast(input_x, shape_max)
    data2_tmp1 = tbe.broadcast(input_y, shape_max)
    res = tbe.vsub(data1_tmp1, data2_tmp1)
    return res


def vmul(input_x, input_y, output_z, kernel_name="vmul"):
    """
    do element-wise sub operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32,int32
    input_y : dict
        shape and dtype of input, only support float16, float32,int32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : kernel name, default value is "sub"

    Returns
    -------
    None
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                              param_name_input1="input_x",
                                                              param_name_input2="input_y")

    data1_tmp1 = tbe.broadcast(input_x, shape_max)
    data2_tmp1 = tbe.broadcast(input_y, shape_max)
    res = tbe.vmul(data1_tmp1, data2_tmp1)
    return res


@register_operator_compute("lamb_update_with_lr", op_mode="static", support_fusion=True)
def lamb_update_with_lr_compute(data_input_greater1,
                                data_input_greater_realdiv,
                                data_input_realdiv, data_input_mul0,
                                data_input_mul1, data_input_sub,
                                data_greater_y, data_select_e,
                                data_minimum_y, y,
                                kernel_name="lamb_update_with_lr"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """

    greater0 = greater_compute(data_input_greater1,
                               data_greater_y, {}, kernel_name)
    greater1 = greater_compute(data_input_greater_realdiv,
                               data_greater_y, {}, kernel_name)
    realdiv0 = real_div_compute(data_input_greater_realdiv,
                                data_input_realdiv, {}, kernel_name)

    select0 = select_compute(greater0, realdiv0,
                             data_select_e, {}, kernel_name)
    select1 = select_compute(greater1, select0,
                             data_select_e, {}, kernel_name)

    minimum0 = minimum(select1, data_minimum_y, {}, kernel_name)

    maximum0 = maximum(minimum0, data_greater_y, {}, kernel_name)

    mul0 = vmul(maximum0, data_input_mul0, {}, kernel_name)
    mul1 = vmul(mul0, data_input_mul1, {}, kernel_name)
    res = sub(data_input_sub, mul1, {}, kernel_name)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def lamb_update_with_lr(input_greater1, input_greater_realdiv, input_realdiv,
                        input_mul0, input_mul1, input_sub,
                        greater_y, select_e, minimum_y, y,
                        kernel_name="lamb_update_with_lr"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    shape_input_greater1 = input_greater1.get("shape")
    shape_input_sub = input_sub.get("shape")
    input_dtype = input_sub.get("dtype").lower()

    para_check.check_shape(shape_input_greater1, param_name="input_greater1")
    para_check.check_shape(shape_input_sub, param_name="input_sub")

    shape_input_greater1, shape_input_sub, shape_max = \
        shape_util.broadcast_shapes(shape_input_greater1, shape_input_sub,
                                    param_name_input1="input_greater1",
                                    param_name_input2="input_sub")

    data_input_greater1 = \
        tvm.placeholder(shape_input_greater1,
                        name="data_input_greater1",
                        dtype=input_dtype)

    data_input_greater_realdiv = \
        tvm.placeholder(shape_input_greater1,
                        name="data_input_greater_realdiv",
                        dtype=input_dtype)
    data_input_realdiv = tvm.placeholder(shape_input_greater1,
                                         name="data_input_realdiv",
                                         dtype=input_dtype)
    data_input_mul0 = tvm.placeholder(shape_input_greater1,
                                      name="data_input_mul0",
                                      dtype=input_dtype)
    data_input_mul1 = tvm.placeholder(shape_input_sub,
                                      name="data_input_mul1",
                                      dtype=input_dtype)
    data_input_sub = tvm.placeholder(shape_input_sub,
                                     name="data_input_sub",
                                     dtype=input_dtype)
    data_greater_y = tvm.placeholder(shape_input_greater1,
                                     name="data_greater_y",
                                     dtype=input_dtype)
    data_select_e = tvm.placeholder(shape_input_greater1,
                                    name="data_select_e",
                                    dtype=input_dtype)
    data_minimum_y = tvm.placeholder(shape_input_greater1,
                                     name="data_minimum_y",
                                     dtype=input_dtype)

    res = lamb_update_with_lr_compute(data_input_greater1,
                                      data_input_greater_realdiv,
                                      data_input_realdiv, data_input_mul0,
                                      data_input_mul1, data_input_sub,
                                      data_greater_y, data_select_e,
                                      data_minimum_y, y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_greater1, data_input_greater_realdiv,
                              data_input_realdiv, data_input_mul0,
                              data_input_mul1, data_input_sub, data_greater_y,
                              data_select_e, data_minimum_y, res]}

    tbe.cce_build_code(sch, config)
