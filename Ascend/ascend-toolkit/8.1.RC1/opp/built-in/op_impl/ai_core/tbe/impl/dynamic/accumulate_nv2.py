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
dynamic accumulate_nv2
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    MIN_TENSOR_NUM = 1
    MAX_TENSOR_NUM = 40


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@register_operator_compute("AccumulateNV2", op_mode="dynamic", support_fusion=True)
def _accumulate_nv2_compute(input_x, output_y, num, kernel_name='accumulate_nv2'):
    """
    Process accumulate_nv2 operator.

    Parameters:
    ----------
    input_x : the list of input tensor.

    y : the dict of output.

    num : the size of input.

    kernel_name : cce kernel name, default value is "accumulate_nv2".

    Returns:
    -------
    result : result of accumulate.
    """

    dtype = input_x[0].dtype
    shape = input_x[0].shape
    len_input = len(input_x)

    # in order to improve the accuracy, convert float16 to float32
    has_covert_float32 = dtype in ("float16",) and tbe_platform.api_check_support("tbe.dsl.vadd", "float32")

    if len_input == 1:
        if dtype in ("int8", "uint8"):
            input_tensor = tbe.cast_to(input_x[0], "float16")
            zero_tensor = tbe.broadcast(tvm.const(0, dtype="float16"), shape)
        elif has_covert_float32:
            input_tensor = tbe.cast_to(input_x[0], "float32")
            zero_tensor = tbe.broadcast(tvm.const(0, dtype="float32"), shape)
        else:
            input_tensor = input_x[0]
            zero_tensor = tbe.broadcast(tvm.const(0, dtype=dtype), shape)

        result = tbe.vadd(input_tensor, zero_tensor)
    else:
        for i, input_tensor in enumerate(input_x):
            if dtype in ("int8", "uint8"):
                input_tensor = tbe.cast_to(input_tensor, "float16")
            elif has_covert_float32:
                input_tensor = tbe.cast_to(input_tensor, "float32")

            if i == 0:
                result = input_tensor
            else:
                result = tbe.vadd(result, input_tensor)
 
    # in order to improve the accuracy, convert float32 back to float16
    if has_covert_float32:
        result = tbe.cast_to(result, "float16")

    if dtype in ("int8", "uint8"):
        result = tbe.cast_to(result, dtype)

    return result


def _get_dtype(input_dict):
    """
    Get shape and data type from dictionary.

    Parameters:
    ----------
    input_dict : input dictionary.

    Returns:
    -------
    shape: the shape of input tensor.

    inp_dtype: the lower data type of input tensor.
    """
    check_list = ('float32', 'float16', 'int8', 'uint8', 'int32')
    dtype = input_dict.get('dtype')
    para_check.check_dtype(dtype, check_list, param_name="input_x")
    inp_dtype = dtype.lower()

    return inp_dtype


def _check_all_dtype_same(input_list):
    """
    Check shape and data type of inputs are all same, and return shape and dtype.

    Parameters:
    ----------
    input_list : the list of input dict.

    Returns:
    -------
    shape: the shape of input tensor.

    dtype: the data type of input tensor.
    """

    if input_list is None or len(input_list) < Constant.MIN_TENSOR_NUM:
        error_detail = 'inputs must be a list of at least one Tensor with the same dtype and shape, \
        MIN_TENSOR_NUM:', Constant.MIN_TENSOR_NUM
        error_manager_vector.raise_err_specific_reson("accumulate_nv2", error_detail)

    # ccec compiler does not support more than 40 parameters, so limit it
    if len(input_list) > Constant.MAX_TENSOR_NUM:
        error_manager_vector.raise_err_input_param_not_in_range("accumulate_nv2", "tensor_num", \
        Constant.MIN_TENSOR_NUM, Constant.MAX_TENSOR_NUM, len(input_list))
    dtype = _get_dtype(input_list[0])

    if not all(dtype == _get_dtype(x) for x in input_list):
        error_detail = 'inputs must be a list of at least one Tensor with the same dtype and shape'
        error_manager_vector.raise_err_specific_reson("accumulate_nv2", error_detail)
    return dtype


# 'pylint: disable=too-many-locals
@register_operator("AccumulateNV2")
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def accumulate_nv2(input_x, output_y, num, kernel_name="accumulate_nv2"):
    """
    Returns the element-wise sum of a list of tensors.

    Parameters:
    ----------
    input_x : the list of input dict. support dtype: float16, float32, int8, uint8, int32.

    output_y : the dict of output.

    num : the size of input.

    kernel_name : cce kernel name, default value is "accumulate_nv2".

    Returns:
    -------
    None
    """

    num = len(input_x) if num is None else num
    if len(input_x) != num:
        error_detail = "The size of input and num must be same."
        error_manager_vector.raise_err_two_input_shape_invalid("accumulate_nv2", len(input_x), num, error_detail)

    dtype = _check_all_dtype_same(input_x)
    ins = classify(input_x, OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for inputs in ins:
        with tbe.compute():
            shape_normlize = shape_util.variable_shape(inputs)
            datas = []
            for (i, _), shape_i in zip(enumerate(inputs), shape_normlize):
                datas.append(tvm.placeholder(shape_i,
                                             name="data_%d" % i,
                                             dtype=dtype))
            # _accumulate_nv2_compute
            res = _accumulate_nv2_compute(datas, output_y, num, kernel_name)
            datas.append(res)
            tensors.append(datas[:])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
