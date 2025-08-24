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
adam_apply_one
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import build
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util import util_common
from impl.util import util_select_op_base


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,unused-variable
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
            shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="data_1", param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)

    return data_1, data_2


# 'pylint: disable=redeclared-assigned-name
@register_operator_compute("adam_apply_one", op_mode="static", support_fusion=True)
def adam_apply_one_compute(data_input0, data_input1, data_input2, data_input3,
                           data_input4, data_input_mul, data_input_mul1,
                           data_input_mul2, data_input_mul3, data_input_add2,
                           output0, output1, output2,
                           kernel_name="adam_apply_one"):
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
        kernel name, default value is "adam_apply_one"

    Returns
    -------
    output tensor
    """

    # square
    data_input0, data_input0 = shape_broadcast(data_input0, data_input0)
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
    truediv_result = tbe.vdiv(output1, add_2_result)

    # mul_4
    truediv_result, data_input4 = shape_broadcast(truediv_result, data_input4)
    mul_4_result = tbe.vmul(truediv_result, data_input4)

    # sub
    mul_4_result, data_input3 = shape_broadcast(mul_4_result, data_input3)
    output2 = tbe.vsub(data_input3, mul_4_result)

    res = [output0, output1, output2]

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def adam_apply_one(input0, input1, input2, input3, input4,
                   mul0_x, mul1_x, mul2_x, mul3_x, add2_y,
                   output0, output1, output2, kernel_name="adam_apply_one"):
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
        cce kernel name, default value is adam_apply_one

    Returns
    -------
    None
    """
    shape_input0 = shape_util.scalar2tensor_one(input0.get("shape"))
    shape_input1 = shape_util.scalar2tensor_one(input1.get("shape"))
    shape_input2 = shape_util.scalar2tensor_one(input2.get("shape"))
    shape_input3 = shape_util.scalar2tensor_one(input3.get("shape"))
    shape_input4 = shape_util.scalar2tensor_one(input4.get("shape"))
    shape_mul0_x = shape_util.scalar2tensor_one(mul0_x.get("shape"))
    shape_mul1_x = shape_util.scalar2tensor_one(mul1_x.get("shape"))
    shape_mul2_x = shape_util.scalar2tensor_one(mul2_x.get("shape"))
    shape_mul3_x = shape_util.scalar2tensor_one(mul3_x.get("shape"))
    shape_add2_y = shape_util.scalar2tensor_one(add2_y.get("shape"))

    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()
    dtype_input3 = input3.get("dtype").lower()
    dtype_input4 = input4.get("dtype").lower()
    dtype_mul0_x = mul0_x.get("dtype").lower()
    dtype_mul1_x = mul1_x.get("dtype").lower()
    dtype_mul2_x = mul2_x.get("dtype").lower()
    dtype_mul3_x = mul3_x.get("dtype").lower()
    dtype_add2_y = add2_y.get("dtype").lower()

    shape_input0, shape_mul3_x, shape_max_mul3 = \
        shape_util.broadcast_shapes(shape_input0, shape_mul3_x, param_name_input1="input0", param_name_input2="mul3_x")
    shape_input1, shape_mul2_x, shape_max_mul2 = \
        shape_util.broadcast_shapes(shape_input1, shape_mul2_x, param_name_input1="input1", param_name_input2="mul2_x")
    shape_input1, shape_add2_y, shape_max_add2 = \
        shape_util.broadcast_shapes(shape_input1, shape_add2_y, param_name_input1="input1", param_name_input2="add2_y")
    shape_input1, shape_input4, shape_max_mul4 = \
        shape_util.broadcast_shapes(shape_input1, shape_input4, param_name_input1="input1", param_name_input2="input4")
    shape_input1, shape_input3, shape_max_sub = \
        shape_util.broadcast_shapes(shape_input1, shape_input3, param_name_input1="input1", param_name_input2="input3")
    shape_input2, shape_mul0_x, shape_max_mul0 = \
        shape_util.broadcast_shapes(shape_input2, shape_mul0_x, param_name_input1="input2", param_name_input2="mul0_x")
    shape_input0, shape_mul1_x, shape_max_mul1 = \
        shape_util.broadcast_shapes(shape_input0, shape_mul1_x, param_name_input1="input0", param_name_input2="mul1_x")

    data_input0 = tvm.placeholder(shape_input0,
                                  name="data_input0",
                                  dtype=dtype_input0)
    data_input1 = tvm.placeholder(shape_input1,
                                  name="data_input1",
                                  dtype=dtype_input1)
    data_input2 = tvm.placeholder(shape_input2,
                                  name="data_input2",
                                  dtype=dtype_input2)
    data_input3 = tvm.placeholder(shape_input3,
                                  name="data_input3",
                                  dtype=dtype_input3)
    data_input4 = tvm.placeholder(shape_input4,
                                  name="data_input4",
                                  dtype=dtype_input4)
    data_input_mul = tvm.placeholder(shape_mul0_x,
                                     name="data_input_mul",
                                     dtype=dtype_mul0_x)
    data_input_mul1 = tvm.placeholder(shape_mul1_x,
                                      name="data_input_mul1",
                                      dtype=dtype_mul1_x)
    data_input_mul2 = tvm.placeholder(shape_mul2_x,
                                      name="data_input_mul2",
                                      dtype=dtype_mul2_x)
    data_input_mul3 = tvm.placeholder(shape_mul3_x,
                                      name="data_input_mul3",
                                      dtype=dtype_mul3_x)
    data_input_add2 = tvm.placeholder(shape_add2_y,
                                      name="data_input_add2",
                                      dtype=dtype_add2_y)

    res = adam_apply_one_compute(data_input0, data_input1, data_input2,
                                 data_input3, data_input4, data_input_mul,
                                 data_input_mul1, data_input_mul2,
                                 data_input_mul3, data_input_add2,
                                 output0, output1, output2, kernel_name)

    inputlist = [data_input0, data_input1, data_input2, data_input3,
                 data_input4, data_input_mul, data_input_mul1,
                 data_input_mul2, data_input_mul3, data_input_add2]

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res)}

    build(sch, config)


def op_select_format(input0, input1, input2, input3, input4,
                     mul0_x, mul1_x, mul2_x, mul3_x, add2_y,
                     output0, output1, output2, kernel_name="adam_apply_one"):
    """
    1. When all of the inputs' shapes are the same, the Op AdamApplyOne can support format NDC1HWC0 and FRACTAL_Z_3D.
    2. In other scenes, the Op can support format ND.
    """
    shape_input0 = shape_util.scalar2tensor_one(input0.get("ori_shape"))
    shape_input1 = shape_util.scalar2tensor_one(input1.get("ori_shape"))
    shape_input2 = shape_util.scalar2tensor_one(input2.get("ori_shape"))
    shape_input3 = shape_util.scalar2tensor_one(input3.get("ori_shape"))
    shape_input4 = shape_util.scalar2tensor_one(input4.get("ori_shape"))
    shape_mul0_x = shape_util.scalar2tensor_one(mul0_x.get("ori_shape"))
    shape_mul1_x = shape_util.scalar2tensor_one(mul1_x.get("ori_shape"))
    shape_mul2_x = shape_util.scalar2tensor_one(mul2_x.get("ori_shape"))
    shape_mul3_x = shape_util.scalar2tensor_one(mul3_x.get("ori_shape"))
    shape_add2_y = shape_util.scalar2tensor_one(add2_y.get("ori_shape"))

    input_shape_list = [shape_input0, shape_input1, shape_input2, shape_input3,
                        shape_input4, shape_mul0_x, shape_mul1_x, shape_mul2_x,
                        shape_mul3_x, shape_add2_y]

    input_list = [input0, input1, input2, input3, input4,
                  mul0_x, mul1_x, mul2_x, mul3_x, add2_y]

    dtype_list = ["float16", "float32", "bfloat16"]
    dtype_list_out = ["float16", "float32", "bfloat16"]
    support_format = ["ND"] * len(dtype_list)
    shape_equal_flag = True
    for i in range(1, 10):
        if input_shape_list[0] != input_shape_list[i]:
            shape_equal_flag = False
            break

    if shape_equal_flag and not util_common.is_dynamic_input(input_list):
        support_format = support_format + ["NDC1HWC0"] * len(dtype_list)
        support_format = support_format + ["FRACTAL_Z_3D"] * len(dtype_list)
        dtype_list_out = dtype_list_out + dtype_list + dtype_list

    dtype_str = ','.join(dtype_list_out)
    format_str = ','.join(support_format)

    input0_param = util_select_op_base.gen_param(classify="input0", name="input0",
                                                 datatype=dtype_str, format=format_str)
    input1_param = util_select_op_base.gen_param(classify="input1", name="input1",
                                                 datatype=dtype_str, format=format_str)
    input2_param = util_select_op_base.gen_param(classify="input2", name="input2",
                                                 datatype=dtype_str, format=format_str)
    input3_param = util_select_op_base.gen_param(classify="input3", name="input3",
                                                 datatype=dtype_str, format=format_str)
    input4_param = util_select_op_base.gen_param(classify="input4", name="input4",
                                                 datatype=dtype_str, format=format_str)
    input5_param = util_select_op_base.gen_param(classify="input5", name="mul0_x",
                                                 datatype=dtype_str, format=format_str)
    input6_param = util_select_op_base.gen_param(classify="input6", name="mul1_x",
                                                 datatype=dtype_str, format=format_str)
    input7_param = util_select_op_base.gen_param(classify="input7", name="mul2_x",
                                                 datatype=dtype_str, format=format_str)
    input8_param = util_select_op_base.gen_param(classify="input8", name="mul3_x",
                                                 datatype=dtype_str, format=format_str)
    input9_param = util_select_op_base.gen_param(classify="input9", name="add2_y",
                                                 datatype=dtype_str, format=format_str)
    output0_param = util_select_op_base.gen_param(classify="output0", name="output0",
                                                 datatype=dtype_str, format=format_str)
    output1_param = util_select_op_base.gen_param(classify="output1", name="output1",
                                                 datatype=dtype_str, format=format_str)
    output2_param = util_select_op_base.gen_param(classify="output2", name="output2",
                                                 datatype=dtype_str, format=format_str)

    param_list = [input0_param, input1_param, input2_param, input3_param,
                  input4_param, input5_param, input6_param, input7_param,
                  input8_param, input9_param,
                  output0_param, output1_param, output2_param]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json
