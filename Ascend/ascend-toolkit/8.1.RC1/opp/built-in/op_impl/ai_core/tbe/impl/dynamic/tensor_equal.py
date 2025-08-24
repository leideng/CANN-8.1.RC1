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
tensor_equal
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import is_unknown_rank_input
from impl.util.reduce_pattern_adapter import ReducePattern


class Constant:
    """
    The class for Constant
    """
    # define a scalar, minimun num of float32 2^(-126)
    SCALAR_MIN_FP32 = 2 ** (-126)
    # define a scalar, value is 2^(42)
    SCALAR_MUL_FP32 = 2 ** (42)
    # define a scalar, minimun num of float16 2^(-24)
    SCALAR_MIN_FP16 = 2 ** (-24)
    # define a scalar, value is 2^(12)
    SCALAR_MUL_FP16 = 2 ** (12)


@register_operator_compute("Tensor_Equal", op_mode="dynamic", support_fusion=False, support_bfp16=True)
def tensor_equal_compute_use_sub(input_x, input_y, output_z, axes, kernel_name="tensor_equal"):
    '''
    True if two tensors have the same size, elements and dtype, False otherwise

    Parameters:
    ----------
    input_x : TVM tensor
        input tenser x
    input_y: TVM tensor
        input tensor y
    kernel_name: str
        kernel name, default value is "tensor_equal"
    output_z: bool
        output tensor with True or False

    Returns
    ----------------------------------------
    the compare result

    Pseducode
    --------------
    MIN_FP32 = 2^(-126)
    MIN_FP16 = 2^(-24)
    if shape_x != shape_y or dtype_x != dtype_y:
        return False
    sub = abs(input_x - input_y)
    if FP32:
        res_min = min(2^(-126), sub)
        res = abs(res_min*(2^126)-1)
    elif FP16:
        res_min = min(2^(-24), sub)
        res = abs(res_min*(2^24)-1)
    retirm reduce_min(res)
    '''
    dtype_x = input_x.dtype
    dtype_y = input_y.dtype
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)

    if shape_x != shape_y or dtype_x != dtype_y:
        # general False result and return
        input_x = tbe.cast_to(input_x, 'float16', True)
        scalar_zero = tvm.const(0, dtype = 'float16')
        res = tbe.vmuls(input_x, scalar_zero)
        res = tbe.cast_to(res, "float16", True)
        res = tbe.reduce_min(res, axis = axes, keepdims=False)
        res = tbe.cast_to(res, "int8", True)
        return res

    if dtype_x == 'int32':
        input_x = tbe.cast_to(input_x, 'float32')
        input_y = tbe.cast_to(input_y, 'float32')
        dtype_x = 'float32'
        dtype_y = 'float32'
    elif dtype_x in ("int8", "uint8", "bool"):
        input_x = tbe.cast_to(input_x, 'float16')
        input_y = tbe.cast_to(input_y, 'float16')
        dtype_x = 'float16'
        dtype_y = 'float16'

    reduce_all_support = tbe_platform.api_check_support('tbe.dsl.reduce_all', 'bool')
    if reduce_all_support:
        res = tbe.vcmp(input_x, input_y, 'eq', mode='bool')
        res = tbe.reduce_all(res, axes, False)
        res = tbe.cast_to(res, 'int8', True)
        return res

    if dtype_x == 'float16':
        scalar_min = tvm.const(Constant.SCALAR_MIN_FP16, dtype = 'float16')
        scalar_mul = tvm.const(Constant.SCALAR_MUL_FP16, dtype = 'float16')
        scalar_neg_one = tvm.const(-1, dtype = 'float16')
        res_min = tbe.vsub(input_x, input_y)
        res_min = tbe.vabs(res_min)
        res = tbe.vmins(res_min, scalar_min)
        res = tbe.vmuls(res, scalar_mul)
        res = tbe.vmuls(res, scalar_mul)
        res = tbe.vadds(res, scalar_neg_one)
        res = tbe.vabs(res)
        res = tbe.reduce_min(res, axis=axes, keepdims=False)
        res = tbe.cast_to(res, 'int8', True)
    else:
        scalar_min = tvm.const(Constant.SCALAR_MIN_FP32, dtype = 'float32')
        scalar_mul = tvm.const(Constant.SCALAR_MUL_FP32, dtype = 'float32')
        scalar_neg_one = tvm.const(-1, dtype = 'float16')
        res_min = tbe.vsub(input_x, input_y)
        res_min = tbe.vabs(res_min)
        res = tbe.vmins(res_min, scalar_min)
        res = tbe.vmuls(res, scalar_mul)
        res = tbe.vmuls(res, scalar_mul)
        res = tbe.vmuls(res, scalar_mul)
        res = tbe.vadds(res, scalar_neg_one)
        res = tbe.vabs(res)
        res = tbe.reduce_min(res, axis=axes, keepdims=False)
        res = tbe.cast_to(res, 'int8', True)
    return res


@register_operator("Tensor_Equal")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def tensor_equal(input_x, input_y, output_z, kernel_name="tensor_equal"):
    '''
    True if two tensors have the same size and elements, False otherwise

    :param input_x: dict
                input tenser x
    :param input_y: dict
                input tensor y
    :param kernel_name: str
                  kernel name, default value is "tensor_equal"
    :return: none
    '''
    check_tuple = ("float16", "float32", "int32", "int8", "uint8", "bool", "bfloat16")
    input_dtype_y = input_y.get("dtype").lower()
    input_dtype_x = input_x.get("dtype").lower()
    
    para_check.check_dtype(input_dtype_y, check_tuple)
    para_check.check_dtype(input_dtype_x, check_tuple)
    input_y["rel_pos_to_reduce"] = "before"
    input_x["rel_pos_to_reduce"] = "before"

    if is_unknown_rank_input(input_x):
        input_axis = {"shape": [-1], "value": [], "rel_pos_to_reduce": "axis"}
    else:
        shape_x = shape_util.shape_to_list(input_x.get("shape"))
        shape_len = len(shape_x)
        axis = list(range(shape_len))
        input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    # gen extra_params for reduce pattern
    extra_params = dict()
    # set KEEP_DIMS flag
    extra_params.update(ReducePattern.KEEP_DIMS_FALSE)
    # set all reduce pattern
    extra_params.update(ReducePattern.REDUCE_MODE_REDUCE_ALL)
    ins = classify([input_x, input_y, input_axis], OpPatternMode.REDUCE, extra_params)
    schedules, tensors = [], []
    for(_x, _y, _axis) in ins:
        with tbe.compute():
            shape_x, shape_y, shape_axis = shape_util.variable_shape([_x, _y, _axis], op_mode="reduce")
            data_x = tvm.placeholder(shape_x, name='data_1', dtype=input_dtype_x)
            data_y = tvm.placeholder(shape_y, name='data_2', dtype=input_dtype_y)
            res = tensor_equal_compute_use_sub(data_x, data_y, output_z, _axis.get('value'), kernel_name)
            tensors.append([data_x, data_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name":kernel_name,
              "tensor_list":tensors}
    tbe.build(schedules, config)
