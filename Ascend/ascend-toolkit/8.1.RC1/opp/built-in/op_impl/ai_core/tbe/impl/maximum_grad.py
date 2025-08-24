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
maximum_grad
"""
from impl.util.platform_adapter import para_check
from impl import fused_minimum_or_maximum_grad


def check_supported(grads, x1, x2, y1, y2, grad_x=True, grad_y=True, kernel_name="maximum_grad"):

    return True


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,arguments-out-of-order
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def maximum_grad(input_dz, input_x, input_y, output_dx, output_dy,
                 grad_x=True, grad_y=True, kernel_name="maximum_grad"):
    """
    algorithm:
    calculating maximum_grad of the three input data

    Parameters
    ----------
    input_dz : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    input_y : dict
        shape and dtype of y input, only support float16, float32
    output_dx: dict
        shape and dtype of output, should be same shape and type as input
    output_dy: dict
        shape and dtype of output, should be same shape and type as input
    grad_x: bool
        if grad_x is true,output need return dx
    grad_y: bool
        if grad_y is true,output need return dy
    kernel_name : str
        cce kernel name, default value is maximum_grad

    Returns:
    -------
    none.
    """
    shape_dz = input_dz.get("shape")
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype = input_dz.get("dtype").lower()
    fused_minimum_or_maximum_grad.fused_minimum_or_maximum_grad_cce(shape_dz,
                                                                    shape_x,
                                                                    shape_y,
                                                                    grad_x,
                                                                    grad_y,
                                                                    "GE",
                                                                    dtype,
                                                                    kernel_name)
