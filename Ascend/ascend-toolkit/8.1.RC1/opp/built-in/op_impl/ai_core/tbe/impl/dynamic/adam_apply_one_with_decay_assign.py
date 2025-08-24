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
dynamic adam_apply_one_with_decay_assign
"""
import uuid
from impl.util.platform_adapter import tbe_platform
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
    # shape size limit
    NUM_ELEVEN = 11
    NUM_ZERO = 0
    DYNAMIC_NUM = -1


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,too-many-statements
# 'pylint: disable=locally-disabled,invalid-name,too-many-locals
def square_compute(x, kernel_name="square"):
    """
    calculating data's square,y= x*x

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    kernel_name: str
        cce kernel name, default value is "square"

    Returns
    -------
    res: the result of square
    """
    res = tbe.vmul(x, x)
    return res


def mul_compute(x1, x2, kernel_name="mul"):
    """
   calculating data's element-wise mul, c = a .* b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "mul"

   Returns
   -------
   res: output of the data's element-wise mul
   """
    shape_x1, shape_x2, shape_max = \
        shape_util.broadcast_shapes(x1.shape, x2.shape, param_name_input1="x1", param_name_input2="x2")
    data_x1 = tbe.broadcast(x1, shape_max)
    data_x2 = tbe.broadcast(x2, shape_max)
    res = tbe.vmul(data_x1, data_x2)

    return res


def add_compute(x1, x2, kernel_name="add"):
    """
   calculating data's element-wise add, c = a + b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "add"

   Returns
   -------
   res: output of the data's add
   """
    shape_x1, shape_x2, shape_max = \
        shape_util.broadcast_shapes(x1.shape, x2.shape, param_name_input1="x1", param_name_input2="x2")
    data_x1 = tbe.broadcast(x1, shape_max)
    data_x2 = tbe.broadcast(x2, shape_max)
    res = tbe.vadd(data_x1, data_x2)

    return res


def sqrt_compute(x, kernel_name="sqrt"):
    """
    calculating data sqrt,y= x**0.5, mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    res:  the result of sqrt
    """
    input_dtype = x.dtype
    has_improve_precision = False
    if input_dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vsqrt", "float32"):
        x = tbe.cast_to(x, "float32")
        has_improve_precision = True

    res = tbe.vsqrt(x)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


def true_div_compute(x1, x2, kernel_name="true_div"):
    """
    calculating data's realdiv, y = x1 / x2

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    kernel_name: str
        cce kernel name, default value is "true_div"

    Returns
    -------
    res: output of the data's divide
    """
    shape_x1, shape_x2, shape_max = \
        shape_util.broadcast_shapes(x1.shape, x2.shape, param_name_input1="x1", param_name_input2="x2")
    data_x1 = tbe.broadcast(x1, shape_max)
    data_x2 = tbe.broadcast(x2, shape_max)

    res = tbe.vdiv(data_x1, data_x2)

    return res


def sub_compute(x1, x2, kernel_name="sub"):
    """
   calculating data's sub, c = a - b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "sub"

   Returns
   -------
   res : output of the data's sub
   """
    shape_x1, shape_x2, shape_max = \
        shape_util.broadcast_shapes(x1.shape, x2.shape, param_name_input1="x1", param_name_input2="x2")
    data_x1 = tbe.broadcast(x1, shape_max)
    data_x2 = tbe.broadcast(x2, shape_max)
    res = tbe.vsub(data_x1, data_x2)

    return res


def _check_broadcast_shape(input0, input1, input2, input3, input4,
                           const_mul_x, const_mul1_x, const_mul2_x,
                           const_mul3_x, const_mul4_x, add2_y):
    """
    check broadcast shape

    Parameters
    ----------
    all inputs: dict
        the dict of inputs

    Returns
    -------
    the list of inputs shape after broadcast
    """
    shape0 = input0.get("shape")
    para_check.check_shape(shape0, param_name="input0")

    shape1 = input1.get("shape")
    para_check.check_shape(shape1, param_name="input1")

    shape2 = input2.get("shape")
    para_check.check_shape(shape2, param_name="input2")

    shape3 = input3.get("shape")
    para_check.check_shape(shape3, param_name="input3")

    shape4 = input4.get("shape")
    para_check.check_shape(shape4, param_name="input4")

    shapecm0 = const_mul_x.get("shape")
    para_check.check_shape(shapecm0, param_name="const_mul_x")

    shapecm1 = const_mul1_x.get("shape")
    para_check.check_shape(shapecm1, param_name="const_mul1_x")

    shapecm2 = const_mul2_x.get("shape")
    para_check.check_shape(shapecm2, param_name="const_mul2_x")

    shapecm3 = const_mul3_x.get("shape")
    para_check.check_shape(shapecm3, param_name="const_mul3_x")

    shapecm4 = const_mul4_x.get("shape")
    para_check.check_shape(shapecm4, param_name="const_mul4_x")

    shapey = add2_y.get("shape")
    para_check.check_shape(shapey, param_name="add2_y")

    return [shape0, shape1, shape2, shape3, shape4, shapecm0, shapecm1, shapecm2, shapecm3, shapecm4, shapey]


@register_operator_compute("AdamApplyOneWithDecayAssign", op_mode="dynamic", support_fusion=True)
def adam_apply_one_with_decay_assign_compute(input0, input1, input2, input3, input4,
                                             const_mul_x, const_mul1_x, const_mul2_x,
                                             const_mul3_x, const_mul4_x, add2_y):
    """
    calculating data

    Parameters
    ----------
    input0: TVM tensor
        the placeholder of input0
    input1: TVM tensor
        the placeholder of input1
    input2: TVM tensor
        the placeholder of input2
    input3: TVM tensor
        the placeholder of input3
    input4: TVM tensor
        the placeholder of input4
    const_mul_x: TVM tensor
        the placeholder of const_mul_x
    const_mul1_x: TVM tensor
        the placeholder of const_mul1_x
    const_mul2_x: TVM tensor
        the placeholder of const_mul2_x
    const_mul3_x: TVM tensor
        the placeholder of const_mul3_x
    const_mul4_x: TVM tensor
        the placeholder of const_mul4_x
    add2_y: TVM tensor
        the placeholder of add2_y

    Returns
    -------
    y0: TVM tensor
        the tensor of y0
    y1: TVM tensor
        the tensor of y1
    y2: TVM tensor
        the tensor of y2
    """
    square_0 = square_compute(input0, kernel_name="square")
    mul_3 = mul_compute(square_0, const_mul3_x, kernel_name="mul_3")
    mul_2 = mul_compute(input1, const_mul2_x, kernel_name="mul_2")

    y0 = add_compute(mul_2, mul_3, kernel_name="add_1")

    sqrt_0 = sqrt_compute(y0, kernel_name="sqrt")
    add_2 = add_compute(sqrt_0, add2_y, kernel_name="add_2")
    mul_0 = mul_compute(input2, const_mul_x, kernel_name="mul_0")
    mul_1 = mul_compute(input0, const_mul1_x, kernel_name="mul_1")

    y1 = add_compute(mul_0, mul_1, kernel_name="add_0")

    truediv_0 = true_div_compute(y1, add_2, kernel_name="truediv")
    mul_4 = mul_compute(input3, const_mul4_x, kernel_name="mul_4")
    add_3 = add_compute(truediv_0, mul_4, kernel_name="add_3")
    mul_5 = mul_compute(add_3, input4, kernel_name="mul_5")

    y2 = sub_compute(input3, mul_5, kernel_name="sub")

    return y0, y1, y2


# 'pylint: disable=too-many-branches
@register_operator("AdamApplyOneWithDecayAssign")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def adam_apply_one_with_decay_assign(input0,
                                     input1,
                                     input2,
                                     input3,
                                     input4,
                                     const_mul_x,
                                     const_mul1_x,
                                     const_mul2_x,
                                     const_mul3_x,
                                     const_mul4_x,
                                     add2_y,
                                     output0,
                                     output1,
                                     output2,
                                     kernel_name="adam_apply_one_with_decay_assign"):
    """
    calculating data

    Parameters
    ----------
    input0: dict
        shape and dtype of input0
    input1: dict
        shape and dtype of input1
    input2: dict
        shape and dtype of input2
    input3: dict
        shape and dtype of input3
    input4: dict
        shape and dtype of input4
    const_mul_x: dict
        shape and dtype of const_mul_x
    const_mul1_x: dict
        shape and dtype of const_mul1_x
    const_mul2_x: dict
        shape and dtype of const_mul2_x
    const_mul3_x: dict
        shape and dtype of const_mul3_x
    const_mul4_x: dict
        shape and dtype of const_mul4_x
    add2_y: dict
        shape and dtype of add2_y
    output0: dict
        shape and dtype of output0
    output1: dict
        shape and dtype of output1
    output2: dict
        shape and dtype of output2
    kernel_name: str
        kernel name, default value is "adam_apply_one_with_decay_assign"

    Returns
    -------
    None
    """
    if kernel_name == "adam_apply_one_with_decay_assign":
        kernel_name += \
            str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4())
        kernel_name = kernel_name.replace('-', 'Z')
    data_dict = {"data_grad": 0, "data_v": 1, "data_m": 2, "data_var": 3,
                 "data_input4": 4, "const_input_mul": 5, "const_input_mul1": 6,
                 "const_input_mul2": 7, "const_input_mul3": 8, "const_input_mul4": 9,
                 "data_input_add2": 10}
    data_dtype = []
    data_dtype.append(input0.get("dtype").lower())
    data_dtype.append(input1.get("dtype").lower())
    data_dtype.append(input2.get("dtype").lower())
    data_dtype.append(input3.get("dtype").lower())
    data_dtype.append(input4.get("dtype").lower())
    data_dtype.append(const_mul_x.get("dtype").lower())
    data_dtype.append(const_mul1_x.get("dtype").lower())
    data_dtype.append(const_mul2_x.get("dtype").lower())
    data_dtype.append(const_mul3_x.get("dtype").lower())
    data_dtype.append(const_mul4_x.get("dtype").lower())
    data_dtype.append(add2_y.get("dtype").lower())

    data_inputs = [None] * Constant.NUM_ELEVEN
    data_names = ["data_grad", "data_v", "data_m", "data_var", "data_input4",
                  "const_input_mul", "const_input_mul1", "const_input_mul2",
                  "const_input_mul3", "const_input_mul4", "data_input_add2"]

    shape0, shape1, shape2, shape3, shape4, \
    shapecm0, shapecm1, shapecm2, shapecm3, shapecm4, \
    shapey = _check_broadcast_shape(input0, input1, input2, input3, input4,
                                    const_mul_x, const_mul1_x, const_mul2_x,
                                    const_mul3_x, const_mul4_x, add2_y)
    
    dynamic_inputs = [input0, input1, input2, input3, input4,
                      const_mul_x, const_mul1_x, const_mul2_x,
                      const_mul3_x, const_mul4_x, add2_y]
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

            res = adam_apply_one_with_decay_assign_compute(data_inputs[data_dict.get("data_grad")],
                                                           data_inputs[data_dict.get("data_v")],
                                                           data_inputs[data_dict.get("data_m")],
                                                           data_inputs[data_dict.get("data_var")],
                                                           data_inputs[data_dict.get("data_input4")],
                                                           data_inputs[data_dict.get("const_input_mul")],
                                                           data_inputs[data_dict.get("const_input_mul1")],
                                                           data_inputs[data_dict.get("const_input_mul2")],
                                                           data_inputs[data_dict.get("const_input_mul3")],
                                                           data_inputs[data_dict.get("const_input_mul4")],
                                                           data_inputs[data_dict.get("data_input_add2")])
            # Fusion with assign
            tensors.append(data_inputs + list(res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "dummy_placeholder": True}
    tbe.build(schedules, config)
