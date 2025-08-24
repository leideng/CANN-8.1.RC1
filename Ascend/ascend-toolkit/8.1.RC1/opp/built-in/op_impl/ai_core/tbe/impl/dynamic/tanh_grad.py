# Copyright 2021 Huawei Technologies Co., Ltd
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
tanh_grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("TanhGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def tanh_grad_compute(y, dy, z, kernel_name="tanh_grad"):
    """
    do element-wise tanh_grad operation between two input tensors

    Parameters
    ----------
    y: TVM tensor
        the placeholder of y input data
    dy: TVM tensor
        the placeholder of dy input data
    z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh_grad

    Returns
    -------
    res : tvm.tensor
        the result of tanh_grad
    """
    shape_list = shape_util.broadcast_shapes(
        y.shape,
        dy.shape,
        param_name_input1="y",
        param_name_input2="dy"
    )
    last_shape = shape_list[2]
    y = tbe.broadcast(y, last_shape)
    dy = tbe.broadcast(dy, last_shape)

    dtype = y.dtype
    if dtype == "float16":
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")

    data1_square = tbe.vmul(y, y)
    data_mul = tbe.vmuls(data1_square, tvm.const(-1, dtype=dtype))
    anuminate = tbe.vadds(data_mul, tvm.const(1, dtype=dtype))
    res = tbe.vmul(anuminate, dy)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=too-many-locals
@register_operator("TanhGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def tanh_grad(y, dy, z, kernel_name="tanh_grad"):
    """
    do element-wise tanh_grad operation between two input tensors

    Parameters
    ----------
    y : dict
        shape and dtype of y input, only support bfloat16, float16, float32
    dy : dict
        shape and dtype of dy input, only support bfloat16, float16, float32
    z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is tanh_grad

    Returns
    -------
    None
    """
    dtype = y.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype, check_list, param_name="y")
    ins = classify([y, dy], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (classify_y, classify_dy) in ins:
        with tbe.compute():
            shape_y, shape_dy = shape_util.variable_shape([classify_y, classify_dy])
            data_y = tvm.placeholder(shape_y, dtype=dtype, name="data_y")
            data_dy = tvm.placeholder(shape_dy, dtype=dtype, name="data_dy")
            res = tanh_grad_compute(data_y, data_dy, z, kernel_name)
            tensors.append([data_y, data_dy, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "print_ir": False, "tensor_list": tensors}
    tbe.build(schedules, config)
