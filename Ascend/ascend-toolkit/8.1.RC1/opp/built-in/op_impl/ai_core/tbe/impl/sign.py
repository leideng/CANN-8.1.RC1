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
sign
"""
import functools

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm

NEG_SCALAR_MIN_FP16 = -(2 ** (-24))
NEG_SCALAR_MIN_FP32 = -(2 ** (-126))
SCALAR_MIN_FP16 = 2 ** (-24)
SCALAR_MIN_FP32 = 2 ** (-126)


# 'pylint: disable=unused-argument
@register_operator_compute("sign", op_mode="static", support_fusion=True)
def sign_compute(input_x, output_y, kernel_name="sign"):
    """compute for sign"""
    dtype = input_x.dtype.lower()
    if dtype == "float32":
        data_min = tvm.const(SCALAR_MIN_FP32, dtype=dtype)
        neg_data_min = tvm.const(NEG_SCALAR_MIN_FP32, dtype=dtype)
    elif dtype == "float16":
        data_min = tvm.const(SCALAR_MIN_FP16, dtype=dtype)
        neg_data_min = tvm.const(NEG_SCALAR_MIN_FP16, dtype=dtype)
    else:
        data_min = tvm.const(1, dtype=dtype)
        neg_data_min = tvm.const(-1, dtype=dtype)

    vmax = tbe.vmaxs(input_x, neg_data_min)
    vmin = tbe.vmins(vmax, data_min)
    if dtype == "float32":
        # max num of float32 is 2**126
        max_support_fp32 = tvm.const(2 ** 62, dtype=dtype)
        res_mul1 = tbe.vmuls(vmin, max_support_fp32)
        res_mul2 = tbe.vmuls(res_mul1, max_support_fp32)
        res = tbe.vmuls(res_mul2, tvm.const(2 ** 2, dtype=dtype))
    elif dtype == "float16":
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        max_support_fp16 = tvm.const(2 ** 12, dtype=dtype)
        res_mul1 = tbe.vmuls(vmin, max_support_fp16)
        res = tbe.vmuls(res_mul1, max_support_fp16)
    else:
        res = vmin
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sign(input_x, output_y, kernel_name="sign"):
    """algrithm: if input_x < 0 output_y=0,input_x > 0 output_y=1
    if input_x=0 output_y=0

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is sign

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    para_check.check_shape(shape, param_name="input_x")

    check_list = ["float16", "float32", "int32"]
    inp_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(inp_dtype, check_list, param_name="input_x")
    shape = shape_util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=inp_dtype)

    res = sign_compute(data, output_y, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}
    tbe.cce_build_code(sch, config)
