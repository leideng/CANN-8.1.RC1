# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
complex_abs
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=invalid-name,unused-argument
@register_operator_compute("ComplexAbs", op_mode="dynamic", support_fusion=False, support_bfp16=False)
def complex_abs_compute(x, y, Tout=None, kernel_name="complex_abs"):
    """
    algorithm: complex_abs

    Parameters
    ----------
    x: TVM tensor
        the placeholder of x
    y: dict
        dict info of y
    Tout: str
        dtype of output
    kernel_name: str
        kernel name, default value is "complex_abs"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    res = tbe.vabs(x)
    return res


# 'pylint: disable=redefined-builtin
@register_operator("ComplexAbs")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def complex_abs(x, y, Tout=None, kernel_name="complex_abs"):
    """
    algorithm: complex_abs

    calculating data's complex_abs,y= |x|

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support complex32, complex64
    y: dict
        shape and dtype of output, should be same shape and type as input
    Tout: str
        dtype of output
    kernel_name : str
        cce kernel name, default value is complex_abs

    Returns
    -------
    None
    """
    dtype_input = x.get("dtype").lower()
    check_list = ("complex64", "complex32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x, ) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input", dtype=dtype_input)
            res = complex_abs_compute(data_input, y, Tout, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
