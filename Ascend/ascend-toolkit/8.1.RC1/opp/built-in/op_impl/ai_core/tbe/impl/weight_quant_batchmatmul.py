#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
weight_quant_batchmatmul
"""
from functools import reduce
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from tbe.dsl.compute.weight_quant_bmm_compute import weight_quant_bmm_compute

ND_LENGTH = 2
NZ_LENGTH = 4


def weight_quant_bmm_fuse_compute(tensor_a, tensor_b, tensor_diag,
    tensor_q_bias, tensor_deq_scale, tensor_bias=None, output_z=None,
    adj_x1=False, adj_x2=False, kernel_name="weight_quant_batchmatmul"):
    para_dict = {
        "tensor_q_bias": tensor_q_bias,
        "deq_scale": tensor_deq_scale,
        "bias": tensor_bias,
        "trans_a": adj_x1,
        "trans_b": adj_x2,
        "format_out": output_z.get("format"),
        "shape_out": output_z.get("shape"),
        "dtype_out": output_z.get("dtype").lower(),
        "kernel_name": kernel_name
    }
    return weight_quant_bmm_compute(tensor_a, tensor_b, tensor_diag, para_dict)


def _create_placeholder(input_dict, name):
    in_shape = input_dict.get("shape")
    in_dtype = input_dict.get("dtype").lower()
    attrs = {
        'format': input_dict.get("format"),
        'ori_format': input_dict.get("ori_format"),
        'ori_shape': input_dict.get("ori_shape"),
        'const_value': input_dict.get("const_value", [])
    }
    if name == "tensor_bias":
        in_shape = [reduce(lambda x, y: x * y, in_shape)]
    in_tensor = tvm.placeholder(in_shape, name=name,
                                dtype=in_dtype, attrs=attrs)
    return in_tensor


def _reset_shape_and_format(input_dict):
    in_shape = list(input_dict.get("shape"))
    batch_shape = 1
    if input_dict.get("format") == "FRACTAL_NZ":
        if len(in_shape) > NZ_LENGTH:
            batch_shape = reduce(lambda x, y: x * y, in_shape[:-4])
        in_shape = [batch_shape, ] + list(in_shape[-4:])
    else:
        input_dict["format"] = "ND"
        if len(in_shape) > ND_LENGTH:
            batch_shape = reduce(lambda x, y: x * y, in_shape[:-2])
        in_shape = [batch_shape, ] + list(in_shape[-2:])
    input_dict["shape"] = in_shape


def _get_batch_shape(input_x, input_y):
    a_batch_shape = list(input_x.get("shape"))[:-2]
    b_batch_shape = list(input_y.get("shape"))[:-2]
    batch_a = [1, 1, 1, 1]
    batch_b = [1, 1, 1, 1]
    for i in range(1, len(a_batch_shape) + 1):
        batch_a[-i] = a_batch_shape[-i]
    for i in range(1, len(b_batch_shape) + 1):
        batch_b[-i] = b_batch_shape[-i]
    return batch_a, batch_b


def weight_quant_batchmatmul(input_x, input_y, diagonal_matrix,
                             q_bias, deq_scale, bias=None, output_z=None,
                             adj_x1=False, adj_x2=False,
                             kernel_name="weight_quant_batchmatmul"):
    ori_batch_x1, ori_batch_x2 = _get_batch_shape(input_x, input_y)
    _reset_shape_and_format(input_x)
    _reset_shape_and_format(input_y)
    _reset_shape_and_format(output_z)
    tensor_a = _create_placeholder(input_x, "tensor_a")
    tensor_b = _create_placeholder(input_y, "tensor_b")
    tensor_a.op.attrs["ori_batch_shape"] = ori_batch_x1
    tensor_b.op.attrs["ori_batch_shape"] = ori_batch_x2
    tensor_diag = _create_placeholder(diagonal_matrix, "tensor_diag")
    tensor_q_bias = _create_placeholder(q_bias, "tensor_q_bias")
    tensor_deq_scale = _create_placeholder(deq_scale, "tensor_deq_scale")
    tensor_list = [tensor_a, tensor_b, tensor_diag, tensor_q_bias, tensor_deq_scale]
    if bias:
        tensor_bias = _create_placeholder(bias, "tensor_bias")
        tensor_list.append(tensor_bias)
    else:
        tensor_bias = None
    out = weight_quant_bmm_fuse_compute(tensor_a, tensor_b, tensor_diag, tensor_q_bias,
                                        tensor_deq_scale, tensor_bias, output_z,
                                        adj_x1, adj_x2, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(out)
    tensor_list.append(out)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "double_buffer_non_reuse": True
    }
    tbe.build(sch, config)
