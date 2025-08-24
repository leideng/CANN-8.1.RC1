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
dynamic adam_apply_one_assign
"""
import uuid

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util import fusion_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    NUM_TEN = 10
    NUM_ZERO = 0
    DYNAMIC_NUM = -1


def shape_broadcast(data_1, data_2):
    """
    broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = \
            shape_util.broadcast_shapes(data_1.shape, data_2.shape, param_name_input1="data_1",
                                        param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)

    return data_1, data_2


# 'pylint: disable=too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals
# 'pylint: disable=too-many-statements,too-many-branches
@register_operator_compute("AdamApplyOneAssign", op_mode="dynamic", support_fusion=True)
def adam_apply_one_assign_compute(data_input0, data_input1, data_input2, data_input3,
                                  data_input4, data_input_mul, data_input_mul1,
                                  data_input_mul2, data_input_mul3, data_input_add2,
                                  output0, output1, output2,
                                  kernel_name="adam_apply_one_assign"):
    """
    apply one adam calculation function

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of square and mul_1
    data_input1: TVM tensor
         the input tensor of mul_2
    data_input2: TVM tensor
         the input tensor of mul_0
    data_input3: TVM tensor
         the input tensor of sub
    data_input4: TVM tensor
         the input tensor of mul_4
    data_input_mul: TVM tensor
         the input tensor of mul_0
    data_input_mul1: TVM tensor
         the input tensor of mul_1
    data_input_mul2: TVM tensor
         the input tensor of mul_2
    data_input_mul3: TVM tensor
         the input tensor of mul_3
    data_input_add2: TVM tensor
         the input tensor of mul_3
    output0: TVM tensor
         the output tensor of add_1
    output1: TVM tensor
         the output tensor of add_0
    output2: TVM tensor
         the output tensor of sub
    kernel_name : str
        kernel name, default value is "adam_apply_one_assign"

    Returns
    -------
    output tensor
    """

    # square
    square_result = tbe.vmul(data_input0, data_input0)

    # mul_3
    square_result, data_input_mul3 = shape_broadcast(square_result,
                                                     data_input_mul3)
    mul_3_result = tbe.vmul(square_result, data_input_mul3)

    # mul_2
    data_input1, data_input_mul2 = shape_broadcast(data_input1,
                                                   data_input_mul2)
    mul_2_result = tbe.vmul(data_input1, data_input_mul2)

    # add_1
    mul_3_result, mul_2_result = shape_broadcast(mul_3_result, mul_2_result)
    output0 = tbe.vadd(mul_2_result, mul_3_result)

    # sqrt
    sqrt_result = tbe.vsqrt(output0)

    # add_2
    data_input_add2, sqrt_result = shape_broadcast(data_input_add2,
                                                   sqrt_result)
    add_2_result = tbe.vadd(sqrt_result, data_input_add2)

    # mul_0
    data_input2, data_input_mul = shape_broadcast(data_input2, data_input_mul)
    mul_0_result = tbe.vmul(data_input2, data_input_mul)

    # mul_1
    data_input0, data_input_mul1 = shape_broadcast(data_input0,
                                                   data_input_mul1)
    mul_1_result = tbe.vmul(data_input0, data_input_mul1)

    # add
    mul_0_result, mul_1_result = shape_broadcast(mul_0_result, mul_1_result)
    output1 = tbe.vadd(mul_0_result, mul_1_result)

    # truediv
    add_2_result, output1 = shape_broadcast(add_2_result, output1)
    output1 = tbe.vmuls(output1, tvm.const(1, dtype=output1.dtype))
    truediv_result = tbe.vdiv(output1, add_2_result)

    # mul_4
    truediv_result, data_input4 = shape_broadcast(truediv_result, data_input4)
    mul_4_result = tbe.vmul(truediv_result, data_input4)

    # sub
    mul_4_result, data_input3 = shape_broadcast(mul_4_result, data_input3)
    output2 = tbe.vsub(data_input3, mul_4_result)

    res = [output0, output1, output2]

    return res


@register_operator("AdamApplyOneAssign")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def adam_apply_one_assign(input0, input1, input2, input3, input4,
                          mul0_x, mul1_x, mul2_x, mul3_x, add2_y,
                          output0, output1, output2, kernel_name="adam_apply_one_assign"):
    """
    function: For bert fuse

    Parameters
    ----------
    input0: dict
         the dict of input of square and mul_1,
         and dtype supports 'float16', 'float32'
    input1: dict
         the dict of input of mul_2, and dtype supports 'float16', 'float32'
    input2: dict
         the dict of input of mul, and dtype supports 'float16', 'float32'
    input3: dict
         the dict of input of sub, and dtype supports 'float16', 'float32'
    input4: dict
         the dict of input of mul_4, and dtype supports 'float16', 'float32'
    mul0_x: dict
         the dict of input of mul_0, and dtype supports 'float16', 'float32'
    mul1_x: dict
         the dict of input of mul_1, and dtype supports 'float16', 'float32'
    mul2_x: dict
         the dict of input of mul_2, and dtype supports 'float16', 'float32'
    mul3_x: dict
         the dict of input of mul_3, and dtype supports 'float16', 'float32'
    add2_y: dict
         the dict of input of add_2, and dtype supports 'float16', 'float32'
    output0: dict
         the dict of output of add_1, and dtype supports 'float16', 'float32'
    output1: dict
         the dict of output of add_0, and dtype supports 'float16', 'float32'
    output2: dict
         the dict of output of sub, and dtype supports 'float16', 'float32'
    kernel_name: str
        cce kernel name, default value is adam_apply_one_assign

    Returns
    -------
    None
    """
    if kernel_name == "adam_apply_one_assign":
        kernel_name += \
            str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4())
        kernel_name = kernel_name.replace('-', 'Z')

    data_dict = {"data_grad": 0, "data_v": 1, "data_m": 2, "data_var": 3,
                 "data_input4": 4, "data_input_mul": 5, "data_input_mul1": 6,
                 "data_input_mul2": 7, "data_input_mul3": 8, "data_input_add2": 9}
    data_dtype = []
    data_dtype.append(input0.get("dtype").lower())
    data_dtype.append(input1.get("dtype").lower())
    data_dtype.append(input2.get("dtype").lower())
    data_dtype.append(input3.get("dtype").lower())
    data_dtype.append(input4.get("dtype").lower())
    data_dtype.append(mul0_x.get("dtype").lower())
    data_dtype.append(mul1_x.get("dtype").lower())
    data_dtype.append(mul2_x.get("dtype").lower())
    data_dtype.append(mul3_x.get("dtype").lower())
    data_dtype.append(add2_y.get("dtype").lower())

    data_inputs = [None] * Constant.NUM_TEN
    data_names = ["data_grad", "data_v", "data_m", "data_var", "data_input4",
                  "data_input_mul", "data_input_mul1", "data_input_mul2",
                  "data_input_mul3", "data_input_add2"]

    dynamic_inputs = [input0, input1, input2, input3, input4,
                      mul0_x, mul1_x, mul2_x, mul3_x, add2_y]
    ins = classify(dynamic_inputs, OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []

    for _dinputs in ins:
        with tbe.compute():
            shape_dinputs = shape_util.variable_shape(_dinputs)
            idx = Constant.NUM_ZERO
            for shape_dinput in shape_dinputs:
                data_inputs[idx] = tvm.placeholder(shape_dinput,
                                                   name=data_names[idx],
                                                   dtype=data_dtype[idx])
                idx += 1

            res = adam_apply_one_assign_compute(data_inputs[data_dict.get("data_grad")],
                                                data_inputs[data_dict.get("data_v")],
                                                data_inputs[data_dict.get("data_m")],
                                                data_inputs[data_dict.get("data_var")],
                                                data_inputs[data_dict.get("data_input4")],
                                                data_inputs[data_dict.get("data_input_mul")],
                                                data_inputs[data_dict.get("data_input_mul1")],
                                                data_inputs[data_dict.get("data_input_mul2")],
                                                data_inputs[data_dict.get("data_input_mul3")],
                                                data_inputs[data_dict.get("data_input_add2")],
                                                output0, output1, output2, kernel_name)
            # Fusion with assign
            tensors.append(data_inputs + list(res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "dummy_placeholder": True}
    tbe.build(schedules, config)
