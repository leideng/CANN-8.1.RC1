#!/usr/bin/python
# -*- coding: utf-8 -*-
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
fusion common function for dynamic
"""
from __future__ import absolute_import

from tbe.tvm import Tensor
from tbe.tvm.tir import expr
from tbe import tvm
from tbe.dsl.base import operation
from tbe.dsl.instrinsic.cce_emitinsn_params import cceEmitParamsIns


def extract_dict(input_x):
    """
    :param input_x:
    :return:
    """
    if isinstance(input_x, Tensor):
        return {"shape": input_x.shape,
                "range": [(1, 1)] * len(input_x.shape)
                }
    return input_x


def create_placeholder(input_x, shape_x):
    """
    :param input_x:
    :param shape_x:
    :return:
    """
    if isinstance(input_x, Tensor):
        return input_x
    dtype = input_x.get("dtype").lower()
    return tvm.placeholder(shape_x, dtype=dtype)


def normalize_shape(inputs: list):
    """
    :param inputs:
    :return:
    """
    var_t = tvm.Var
    expr_t = expr.BinaryOpExpr

    def get_var(_i):
        for _input in inputs:
            dim_i = _input["shape"][_i]
            if isinstance(dim_i, (var_t, expr_t)):
                return dim_i
        for _input in inputs:
            dim_i = _input["shape"][_i]
            range_i = _input["range"][_i]
            if dim_i == -1:
                return operation.var_inner("_dim_" + str(_i), range_i)

    shapes, ranges = [], []
    for input_i in inputs:
        shapes.append(input_i["shape"])
        ranges.append(input_i["range"])

    d_shapes = [[] for _ in shapes]
    for i in range(len(shapes[0])):
        _var = get_var(i)
        for d_shape, shape in zip(d_shapes, shapes):
            if isinstance(shape[i], (var_t, expr_t)):
                pass
            if shape[i] == -1:
                d_shape.append(_var)
            else:
                d_shape.append(shape[i])
    return d_shapes


def check_fusion_input(inputs: list):
    """
    :param inputs:
    :return:
    """
    tensor_t = Tensor

    for i, input_i in enumerate(inputs):
        if not isinstance(input_i, (tensor_t, dict)):
            raise RuntimeError("The input must be a tensor or dict!")

    if len(list(filter(lambda x: isinstance(x, tensor_t), inputs))) > 1:
        raise RuntimeError("The input tensor number must be less than 1!")


def reuse_input_as_output(output_buffer, input_buffer, kernel_name):
    """
    Declare an input_tensor as output_tensor,
    designed for adam_apply + assign fusion
    """
    existing_dict = cceEmitParamsIns.get_param("InputReuseDict_" + kernel_name)
    if existing_dict is None:
        existing_dict = {}
    if output_buffer in existing_dict.keys():
        raise RuntimeError("Cannot assign two inputs to one output")
    existing_dict[output_buffer] = input_buffer
    cceEmitParamsIns.del_param("InputReuseDict_" + kernel_name)
    cceEmitParamsIns.insert_param_with_penetration(
        "InputReuseDict_" + kernel_name, existing_dict)
