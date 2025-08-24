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
index_fill_d
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


#'pylint: disable=unused-argument,too-many-locals,too-many-statements,too-many-arguments,invalid-name
@register_operator_compute("IndexFillD", op_mode="dynamic", support_fusion=True)
def index_fill_d_compute(x, assist1, assist2):
    """
    Main compute logic of index_fill
    Firstly, construct the tensor input of assist1 and assist2,
    where the shape and X of assist1 are the same, and the value is to remove,
    the corresponding position to be filled is 0, and the other positions are 1;
    the shape and X of assist2 are the same,
    and the value is that the corresponding position to be filled is value,
    and all other positions are 0.
    All of the above operations are done in the adaptation layer

    Multiply x and assist1, the add result and assist2.
    Parameters
    ----------
    x : dict
        shape and dtype of input
    assist1 : dict
        shape and dtype of input,should be same shape and dtype as x
    assist2 : dict
        shape and dtype of input,should be same shape and dtype as x
    y : dict
        shape and dtype of output

    Returns
    -------
    output tensor
    """

    if (x.dtype == "float32" or x.dtype == "float16"):
        output_y = tbe.vcmpsel(assist1, 0.0, 'gt', x, assist2)
    elif (x.dtype == "bfloat16"):
        x = tbe.cast_to(x, "float32")
        assist1 = tbe.cast_to(assist1, "float32")
        assist2 = tbe.cast_to(assist2, "float32")
        output_y = tbe.vcmpsel(assist1, 0.0, 'gt', x, assist2)
        output_y = tbe.round(output_y, "bfloat16")
    else:
        x_is_bool = (x.dtype == "int8" or x.dtype == "bool")
        if x_is_bool:
            x = tbe.cast_to(x, "int32")
            assist1 = tbe.cast_to(assist1, "int32")
            assist2 = tbe.cast_to(assist2, "int32")
        temp = tbe.vmul(x, assist1)
        output_y = tbe.vadd(temp, assist2)
        if x_is_bool:
            output_y = tbe.cast_to(output_y, "int8")
    return output_y


@register_operator("IndexFillD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
#'pylint: disable=unused-argument,too-many-locals,too-many-statements,too-many-arguments,invalid-name
def index_fill_d(x, assist1, assist2, y, dim, kernel_name="index_fill_d"):
    """
    Fills the elements of the self tensor with value val by selecting the indices in the order given in index.(PyTorch)
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 2])
    >>> x.index_fill_(1, index, -1)
    tensor([[-1.,  2., -1.],
        [-1.,  5., -1.],
        [-1.,  8., -1.]])
    We implement it in another way.
    Construct two auxiliary tensors in adaptation layer.
    [                   [                 [
      [1, 2, 3],         [0, 1, 0],        [-1, 0, -1],
      [4, 5, 6],   *     [0, 1, 0],    +   [-1, 0, -1],    =
      [7, 8, 9]          [0, 1, 0]         [-1, 0, -1]
    ]                   ]                 ]
        [[-1.,  2., -1.],
        [-1.,  5., -1.],
        [-1.,  8., -1.]]
    Parameters
    ----------
    x : dict
        shape and dtype of input
    assist1 : dict
        shape and dtype of input,should be same shape and dtype as x
    assist2 : dict
        shape and dtype of input,should be same shape and dtype as x
    y : dict
        shape and dtype of outpututil.produce_shapes
    kernel_name : str
        kernel name, default value is "cosine_simlarity"

    Returns
    -------
    None
    """
    x_shape = x.get("shape")
    assist1_shape = assist1.get("shape")
    assist2_shape = assist2.get("shape")

    dtype_x = x.get("dtype").lower()
    dtype_assist1 = assist1.get("dtype").lower()
    dtype_assist2 = assist2.get("dtype").lower()

    para_check.check_shape(x_shape)
    para_check.check_shape(assist1_shape)
    para_check.check_shape(assist2_shape)
    para_check.check_kernel_name(kernel_name)

    schedules = []
    tensors = []
    ins = classify([x, assist1, assist2], OpPatternMode.ELEWISE)

    for (_x, _assist1, _assist2) in ins:
        with tbe.compute():
            shape_x, shape_assist1, shape_assist2 = shape_util.variable_shape([_x, _assist1, _assist2])

            data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_x)
            data_input_as1 = tvm.placeholder(
                shape_assist1, name="data_input_as1", dtype=dtype_assist1)
            data_input_as2 = tvm.placeholder(
                shape_assist2, name="data_input_as2", dtype=dtype_assist2)

            res = index_fill_d_compute(data_input_x, data_input_as1, data_input_as2)
            tensors.append([data_input_x, data_input_as1, data_input_as2, res])

        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
