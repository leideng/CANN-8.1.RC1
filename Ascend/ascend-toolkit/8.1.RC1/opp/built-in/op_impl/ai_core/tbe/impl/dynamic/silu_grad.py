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
dynamic silu_grad
`dx = dy * (sigmoid(x) * (1 + x * (1 - sigmoid(x)))`

let:
`A = dy`
`B = sigmoid(x)`
`dx = A * (B * (1 + x * (1 - B)))`
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=invalid-name,too-many-locals,redefined-argument-from-local
@register_operator_compute("silu_grad", op_mode="dynamic", support_fusion=True)
def silu_grad_compute(dy, x, kernel_name="silu_grad"):
    """
    algorithm : silu grad compute
    let:
    A = dy
    B = sigmoid(x)
    then,
    dx = A * (B * (1 + x * (1 - B)))
    Parameters:
    ----------
    dy : dictionary of gradient
    x : dictionary of silu input
    kernel_name : default value is "silu_grad"
    Returns
    -------
    a tenosr
    """
    dtype = dy.dtype.lower()
    if dtype in ("float16", "bfloat16"):
        x = tbe.cast_to(x, "float32")
        dy = tbe.cast_to(dy, "float32")

    # calculate B
    const_num_neg_one = tvm.const(-1, dtype="float32")
    const_num_one = tvm.const(1, dtype="float32")
    tmp_negative = tbe.vmuls(x, const_num_neg_one)
    tmp_exp = tbe.vexp(tmp_negative)
    tmp_add = tbe.vadds(tmp_exp, const_num_one)
    tensor_one = tbe.broadcast(const_num_one, tmp_add.shape)
    sigmoid_res = tbe.vdiv(tensor_one, tmp_add)

    # calculate res
    sigmoid_sub = tbe.vsub(tensor_one, sigmoid_res)
    sigmoid_mul = tbe.vmul(sigmoid_sub, x)
    sigmoid_add = tbe.vadd(tensor_one, sigmoid_mul)
    sigmoid_mul2 = tbe.vmul(sigmoid_res, sigmoid_add)
    res = tbe.vmul(dy, sigmoid_mul2)

    if dtype == "float16":
        return tbe.cast_to(res, "float16")
    elif dtype == "bfloat16":
        res = tbe.round(res, "bfloat16")
    return res


def check_op_dtype(dtype_input, dtype_x):
    """
    check dtypes
    :param dtype_input: str
    :param dtype_x: str
    :return: none
    """
    if dtype_input != dtype_x:
        error_manager_vector.raise_err_two_input_dtype_invalid('silu_grad', "dy", "x",
                                                               "the dtype of dy, x, must be the same")
    check_list = ["bfloat16", "float16", "float32"]
    para_check.check_dtype(dtype_input, check_list)


@register_operator("SiluGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def silu_grad(dy, x, dx, kernel_name="silu_grad"):
    """
    do silu grad

    dx = dy * (sigmoid(x) * (1 + x * (1- sigmoid(x))))
    Parameters:
    ----------
    dy : dictionary of gradient
    x : dictionary of silu input
    dx: dictionary of output
    kernel_name : default value is "silu_grad"
    Returns
    -------
    None
    """
    dtype_grad = dy.get("dtype").lower()
    dtype_x = x.get("dtype").lower()
    para_check.check_elewise_shape_range(
        [dy, x], support_broadcast=True)

    para_check.check_kernel_name(kernel_name)
    check_op_dtype(dtype_grad, dtype_x)

    ins = classify([dy, x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_dy, _x) in ins:
        with tbe.compute():
            shape_grad, shape_x = shape_util.variable_shape([_dy, _x])
            data_grad = tvm.placeholder(shape_grad, dtype=dtype_grad, name="data_grad")
            data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
            res = silu_grad_compute(data_grad, data_x, kernel_name)
            input_list = [data_grad, data_x, res]
            tensors.append(input_list)

        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
