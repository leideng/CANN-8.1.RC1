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
dynamic mul_no_nan
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
@register_operator_compute("MulNoNan", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def mul_no_nan_compute(input_x1, input_x2, output_y, kernel_name="mul_no_nan"):
    """
    calculating data
    np.where(np.equal(y, 0.), np.zeros((), dtype=dtype), np.multiply(x, y))

    Parameters
    ----------
    input_x1 : TVM tensor
        the placeholder of input_x1
    input_x2 : TVM tensor
        the placeholder of input_x2
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "mul_no_nan"

    Returns
    -------
    output tensor
    """
    src_dtype = input_x1.dtype.lower()

    shape_x1, shape_x2, shape_max = shape_util.broadcast_shapes(input_x1.shape,
                                                                input_x2.shape,
                                                                param_name_input1="shape_x1",
                                                                param_name_input2="shape_x2")
    input_x1 = tbe.broadcast(input_x1, shape_max)
    input_x2 = tbe.broadcast(input_x2, shape_max)

    mul_res = tbe.vmul(input_x1, input_x2)

    zero = tvm.const(0, dtype=src_dtype)
    zeros = tbe.broadcast(zero, shape_max)
    cast_flag = False
    if not tbe_platform.api_check_support("tbe.dsl.vcmpsel", src_dtype):
        if tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32"):
            cast_flag = True
            input_x2 = tbe.cast_to(input_x2, "float32")
            zeros = tbe.cast_to(zeros, "float32")
            mul_res = tbe.cast_to(mul_res, "float32")
        else:
            cast_flag = True
            input_x2 = tbe.cast_to(input_x2, "float16")
            zeros = tbe.cast_to(zeros, "float16")
            mul_res = tbe.cast_to(mul_res, "float16")

    res = tbe.vcmpsel(input_x2, zeros, operation='eq', slhs=zeros, srhs=mul_res)
    if cast_flag:
        res = tbe.cast_to(res, src_dtype)

    return res


@register_operator("MulNoNan")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_OUTPUT,
                            para_check.KERNEL_NAME)
def mul_no_nan(x1, x2, y, kernel_name="mul_no_nan"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype of input1
    x2: dict
        shape and dtype of input2
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "mul_no_nan"

    Returns
    -------
    None
    """
    check_tuple = ("float16", "float32", "int32", "bfloat16")
    inputx1_data_type = x1.get("dtype").lower()
    inputx2_data_type = x2.get("dtype").lower()
    para_check.check_dtype(inputx1_data_type, check_tuple)
    para_check.check_dtype(inputx2_data_type, check_tuple)

    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")

    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x1, _x2) in ins:
        with tbe.compute():
            # shape
            shape_x1, shape_x2 = shape_util.variable_shape([_x1, _x2])
            # mul_compute
            data_x1 = tvm.placeholder(shape_x1, dtype=inputx1_data_type, name="data_x1")
            data_x2 = tvm.placeholder(shape_x2, dtype=inputx2_data_type, name="data_x2")
            res = mul_no_nan_compute(data_x1, data_x2, y, kernel_name)
            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
