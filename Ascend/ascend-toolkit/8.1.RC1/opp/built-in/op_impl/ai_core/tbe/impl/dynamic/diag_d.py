# Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
dynamic diag_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable = locally-disabled,invalid-name,unused-argument,no-member
@register_operator_compute("DiagD", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def diag_d_compute(x, assit, y, kernel_name="diag_d"):
    """
    diag_d compute
    calculating diag_d(x,help):
    returns a diagonal tensor with a given x values.
    If the shape of x is [D1,...,Dk],the shape of diagonal tensor is
    [D1,...,Dk,D1,...,Dk]
    For example:
    x :    [1, 2, 3]
    res :  [[1, 0, 0]
            [0, 2, 0]
            [0, 0, 3]]

    Parameters
    ----------
    x: TVM tensor
        the placeholder of x
    assit: TVM tensor
        the placeholder of assit
    y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "diag_d"

    Returns
    -------
    res: TVM tensor
        the result of diag compute
    """
    shape_x = shape_util.shape_to_list(x.shape)
    shape_assit = shape_util.shape_to_list(assit.shape)
    shape_x, shape_assit, shape_max = \
        shape_util.broadcast_shapes(shape_x, shape_assit,
                                    param_name_input1="input_x",
                                    param_name_input2="input_y")
    data_x = tbe.broadcast(x, shape_max)
    data_assit = tbe.broadcast(assit, shape_max)
    res = tbe.vmul(data_x, data_assit)

    return res


# 'pylint: disable =too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                           para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def diag_d(x, assist, y, kernel_name="diag_d"):
    """
    algorithm: diag_d
    calculating diag_d(x,help):
    returns a diagonal tensor with a given x values.
    If the shape of x is [D1,...,Dk],the shape of diagonal tensor is
    [D1,...,Dk,D1,...,Dk]
    For example:
    x :    [1, 2, 3]
    res :  [[1, 0, 0]
            [0, 2, 0]
            [0, 0, 3]]

    Parameters
    ----------
    x: dict
        dict with keys(shape and dtype) of x
    assist: dict
        dict with keys(shape and dtype) of assist
    y: dict
        dict with keys(shape and dtype) of y
    kernel_name: str
        kernel name, default value is "diag"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype = x.get("dtype").lower()
    shape_help = assist.get("shape")
    dtype_help = assist.get("dtype").lower()

    if len(shape_x) > 4:
        error_detail = "length of x'shape should be less than 5 but got: %d" % len(shape_x)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_help, param_name="assist")

    check_list = ("float16", "float32", "int32", "bfloat16")
    para_check.check_dtype(dtype, check_list, param_name="x")
    para_check.check_dtype(dtype_help, check_list, param_name="assist")

    ins = classify([x, assist], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for _x, _assist in ins:
        with tbe.compute():
            shape_x, shape_assist = shape_util.variable_shape([_x, _assist])
            x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype)
            assist_input = tvm.placeholder(shape_assist, name="assist_input", dtype=dtype_help)
            res = diag_d_compute(x_input, assist_input, y, kernel_name=kernel_name)

            tensors.append([x_input, assist_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)

