#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
cosine_similarity
"""

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl import constant_util as constant
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


def cosine_similarity_compute_helper(input_x1, input_x2, dim):
    """
    input_x1 : TVM tensor
        the placeholder of input_x1
    input_x2 : TVM tensor
        the placeholder of input_x2
    dim : int
        should be in [ -len(shape_input_x1, len(shape_x1) )
    """

    temp = tbe.vmul(input_x1, input_x2)
    ret = tbe.sum(temp, dim)
    return ret


@register_operator_compute("cosine_similarity", op_mode="static", support_fusion=True)
def cosine_similarity_compute(input_x1, input_x2, output_y, dim, eps):
    """
    calculate data

    Parameters
    ----------
    input_x1 : TVM tensor
        the placeholder of input_x1
    input_x2 : TVM tensor
        the placeholder of input_x2
    dim : int
        should be in [ -len(shape_input_x1, len(shape_x1) )
    eps : float
        error range
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "cosine_similarity"

    Returns
    -------
    output tensor
    """
    # PyTorch(v1.5.1) implementation. Use x / sort(x * x) instead of x / (sort(x) * sort(x))
    # When the value is large, this will lead to the overflow problem.
    # x / (sort(x) * sort(x) implementation
    # The accuracy is satisfied and there is no overflow problem, but it will lead to low performance.

    w12 = cosine_similarity_compute_helper(input_x1, input_x2, dim)
    w1 = cosine_similarity_compute_helper(input_x1, input_x1, dim)
    w2 = cosine_similarity_compute_helper(input_x2, input_x2, dim)

    max_w12_eps = tbe.vmaxs(tbe.vmul(w1, w2), eps * eps)
    n12 = tbe.vsqrt(max_w12_eps)
    output_y = tbe.vdiv(w12, n12)
    return output_y


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def cosine_similarity(input_x1, input_x2, output_y,
                      dim=1, eps=1e-8, kernel_name="cosine_similarity"):
    """
    calculate cosine_similarity of input_x1 and input_x2

    Parameters
    ----------
    input_x1 : dict
        shape and dtype of input
    input_x2 : dict
        shape and dtype of input,should be same shape and dtype as input_x1
    dim : int
        should be in [ -len(shape_input_x1, len(shape_x1) )
    eps : float
        error range
    output_y : dict
        shape and dtype of output
    kernel_name : str
        kernel name, default value is "cosine_similarity"

    Returns
    -------
    None
    """
    shape_x1 = shape_util.scalar2tensor_one(input_x1.get("shape"))
    shape_x2 = shape_util.scalar2tensor_one(input_x2.get("shape"))
    shape_len_x1 = len(shape_x1)

    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_x1)
    para_check.check_shape_rule(shape_x2)
    para_check.check_shape_size(shape_x1, constant.SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(shape_x2, constant.SHAPE_SIZE_LIMIT)

    check_tuple = "float32"
    input_data_type1 = input_x1.get("dtype").lower()
    para_check.check_dtype_rule(input_data_type1, check_tuple)
    input_data_type2 = input_x2.get("dtype").lower()
    para_check.check_dtype_rule(input_data_type2, check_tuple)

    if shape_x1 != shape_x2:
        raise RuntimeError("Shape of x1 {} is not equal to x2 {}.".format(shape_x1, shape_x2))

    if dim < -shape_len_x1 or dim >= shape_len_x1:
        raise RuntimeError("Out of range, dim should be in[", -shape_len_x1, shape_len_x1 - 1, "].")

    data_x1 = tvm.placeholder(shape_x1, name="data_1", dtype=input_data_type1)
    data_x2 = tvm.placeholder(shape_x2, name="data_2", dtype=input_data_type2)

    res = cosine_similarity_compute(data_x1, data_x2, output_y, dim, eps)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"print_dir": False,
              "name": kernel_name,
              "tensor_list": [data_x1, data_x2, res]}
    build(schedule, config)
