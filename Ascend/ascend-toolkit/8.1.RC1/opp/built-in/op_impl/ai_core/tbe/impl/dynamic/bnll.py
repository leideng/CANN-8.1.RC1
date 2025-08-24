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
dynamic bnll
"""
# 'pylint: disable=E0401
# 'pylint: disable=C0412
# 'pylint: disable=W0613
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


@register_operator_compute("Bnll", op_mode="dynamic", support_fusion=True)
def bnll_compute(input_x, output_y, kernel_name):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "bnll"

    Returns
    -------
    output tensor
    """
    const_zero = 0.0
    const_one = 1.0
    const_negative_one = -1.0

    dtype = input_x.dtype
    if dtype == "float16" and tbe_platform.api_check_support('tbe.dsl.vexp', 'float32'):
        input_x = tbe.cast_to(input_x, "float32")
        d_dtype = "float32"
    else:
        d_dtype = "float16"

    scalar_zero = tvm.const(const_zero, d_dtype)
    negative_data = tbe.vmins(input_x, scalar_zero)
    positive_data = tbe.vmaxs(input_x, scalar_zero)
    data_reverse = tbe.vaxpy(positive_data, negative_data, tvm.const(const_negative_one, d_dtype))

    res_vexp = tbe.vexp(data_reverse)
    res_vadds = tbe.vadds(res_vexp, tvm.const(const_one, d_dtype))
    res_vlog = tbe.vlog(res_vadds)
    res = tbe.vadd(res_vlog, positive_data)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bnll(input_x, output_y, kernel_name="bnll"):
    """
    calculating data
    algrithm: y=x+log(1+exp(-x)) if x>0; y=log(1+exp(x)) otherwise

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "bnll"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_shape(shape, param_name="x")

    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_input_assist,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_input_assist])[0]
            x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_dtype)
            res = bnll_compute(x_input, output_y, kernel_name)

            tensors.append([x_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)

