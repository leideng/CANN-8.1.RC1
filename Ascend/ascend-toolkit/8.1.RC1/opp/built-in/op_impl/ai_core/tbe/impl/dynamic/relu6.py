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
relu6
`f(x) = min(max(0,x), 6)`
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("Relu6", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def relu6_compute(input_x, output_y, kernel_name="relu6"):
    """
    compute of relu6

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of first input data
    output_y: dict
        shape and dtype of output,should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "relu6"

    Returns
    -------
    compute result of relu6
    """
    tmp_res = tbe.vmaxs(input_x, tvm.const(0, input_x.dtype))
    final_res = tbe.vmins(tmp_res, tvm.const(6, input_x.dtype))

    return final_res


@register_operator("Relu6")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def relu6(input_x, output_y, kernel_name="relu6"):
    """
       f(x)= 6(x >= 6)
       f(x)= 0(x <= 0)
       f(x)= x(0<x<6)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output_y, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "relu6"

    Returns
    ------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    vmaxs_support = tbe_platform.api_check_support("tbe.dsl.vmaxs", "float32")
    if not vmaxs_support:
        para_check.check_dtype(dtype_input, ("int32", "float16"), param_name="input_x")

    check_list = ("int32", "float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input",
                                         dtype=dtype_input)
            res = relu6_compute(data_input, output_y, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
