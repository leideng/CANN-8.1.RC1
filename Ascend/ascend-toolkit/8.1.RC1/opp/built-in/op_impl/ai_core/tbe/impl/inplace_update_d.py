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
inplace_update_d
"""
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable = locally-disabled,invalid-name,too-many-locals
# 'pylint: disable = too-many-arguments,unused-argument,no-member
def check_supported(x, v, y, indices, kernel_name="inplace_update_d"):
    """
    True: inplace_update -> inplace_update_d
    False: remain inplace_update
    """
    shape_x = x.get("ori_shape")
    shape_v = v.get("ori_shape")
    inefficiency_shape_list = [[(30001, 3), (1, 3)], [(34686, 3), (1, 3)]]
    if [shape_x, shape_v] in inefficiency_shape_list:
        reason = "input shape x:{0}, v:{1} is inefficiency, op_type:InplaceUpdate not changed".format(shape_x, shape_v)
        return False, reason

    return True, ""


@register_operator_compute("inplace_update_d", op_mode="static", support_fusion=True)
def inplace_update_d_compute(x, v, y, indices, kernel_name="inplace_update_d"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    v : TVM tensor
        the placeholder of v
    y : dict
        dict of output_y, include keys(shape and dtype)
    indices: list
        index list, has the same length as the length of rank 0 of v
    kernel_name : str
        kernel name, default value is "inplace_update"

    Returns
    -------
    output tensor
    """
    res = tbe.inplace_update(x, indices, v)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def inplace_update_d(x, v, y, indices, kernel_name="inplace_update_d"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input tensor x, only support float16, float32
    v : dict
        shape and dtype of input tensor v,
         should be same shape and type as input
    y : dict
        shape and dtype of output tensor should be same shape and type as input
    indices: list
        index list, has the same length as the length of rank 0 of v
    kernel_name : str
        kernel name, default value is "inplace_update"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    shape_v = v.get("shape")
    dtype_x = x.get("dtype")
    dtype_v = v.get("dtype")

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_v, param_name="v")

    check_tuple = ("float16", "float32", "int32")
    input_data_type_x = dtype_x.lower()
    input_data_type_v = dtype_v.lower()
    para_check.check_dtype(input_data_type_x, check_tuple, param_name="x")
    para_check.check_dtype(input_data_type_v, check_tuple, param_name="v")
    indices = list(indices)

    if len(shape_x) != len(shape_v):
        error_detail = "The number of dimension x must be same as dimension v"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "v", error_detail)

    if shape_v[0] != len(indices):
        error_detail = "The length of rank 0 of tensor v must be the same as length of indices"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "v", "indices", error_detail)

    for i in range(1, len(shape_v)):
        if shape_x[i] != shape_v[i]:
            error_detail = "The length of each rank of tensor x must be the same as " \
                           "length of each rank of tensor v except the first dimension"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "v", error_detail)

    for i, _ in enumerate(indices):
        indices[i] = (indices[i] % shape_x[0] + shape_x[0]) % shape_x[0]

    data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_data_type_x)
    data_v = tvm.placeholder(shape_v, name="data_v", dtype=input_data_type_v)

    res = inplace_update_d_compute(data_x, data_v, y, indices, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_v, res]}

    build(sch, config)
