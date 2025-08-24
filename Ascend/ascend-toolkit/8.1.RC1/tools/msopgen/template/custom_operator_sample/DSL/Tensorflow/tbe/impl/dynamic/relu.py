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
dynamic relu
"""
from __future__ import absolute_import
from functools import reduce as reduceIns

from ..util.platform_adapter import tbe
from ..util.platform_adapter import tvm
from ..util.platform_adapter import tbe_platform
from ..util.platform_adapter import classify
from ..util.platform_adapter import OpPatternMode
from ..util.platform_adapter import shape_util
from ..util.platform_adapter import para_check
from ..util.platform_adapter import register_operator
from ..util.platform_adapter import register_operator_compute

# const value
CONST_ZERO = 0


# pylint: disable=invalid-name,unused-argument,redefined-argument-from-local
@register_operator_compute("Relu", op_mode="dynamic", support_fusion=True)
def relu_compute(x, y, kernel_name="relu"):
    """
    Algrithm : relu(x) = max(x, 0)

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of relu
    """
    inp_dtype = x.dtype
    shape = x.shape
    compatible_dtype = x.dtype

    if inp_dtype == 'int8' and tbe_platform.api_check_support(
            'tbe.dsl.cast_to', 's82f16'):
        x = tbe.cast_to(x, 'float16')
        compatible_dtype = 'float16'
    if tbe_platform.api_check_support('tbe.dsl.vrelu',
                                      compatible_dtype):
        data_res = tbe.vrelu(x)
    else:
        tensor_zero = tbe.broadcast(
            tvm.const(CONST_ZERO, compatible_dtype), shape)
        data_res = tbe.vmax(x, tensor_zero)

    data_res = tbe.cast_to(data_res, inp_dtype)

    return data_res


@register_operator("Relu")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def relu(x, y, kernel_name="relu"):
    """
    Algrithm: relu(x) = max(x, 0)

    Parameters
    ----------
    Algorithm: relu

    Parameters:

    x: dynamic input, include shape, dtype and range

    y: the dict of output

    kernel_name: kernel name, must be string, default value is "relu".

    Returns
    -------
    None
    """

    # check input tensor data_type
    dtype_x = x.get("dtype").lower()
    check_list = ("float16", "float32", "int8", "int32")
    para_check.check_dtype(dtype_x, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([x])
            input_data = tvm.placeholder(shape_x[0], name="input_data",
                                         dtype=dtype_x)
            res = relu_compute(input_data, y, kernel_name)

            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
