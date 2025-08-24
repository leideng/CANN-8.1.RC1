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

from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


@register_operator_compute("index_fill_d", op_mode="static", support_fusion=True)
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
    else:
        temp = tbe.vmul(x, assist1)
        output_y = tbe.vadd(temp, assist2)
    return output_y


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_STR)
# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
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
    `[                   [                 [`
    `  [1, 2, 3],         [0, 1, 0],        [-1, 0, -1],`
    `  [4, 5, 6],   *     [0, 1, 0],    +   [-1, 0, -1],    = `
    `  [7, 8, 9]          [0, 1, 0]         [-1, 0, -1]`
    `]                   ]                 ]`
    `    [[-1.,  2., -1.],`
    `    [-1.,  5., -1.],`
    `    [-1.,  8., -1.]]`

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
    shape_x = x.get("shape")
    shape_assist1 = assist1.get("shape")
    shape_assist2 = assist2.get("shape")

    dtype_x = x.get("dtype").lower()
    dtype_assist1 = assist1.get("dtype").lower()
    dtype_assist2 = assist2.get("dtype").lower()

    para_check.check_shape_rule(shape_x)
    para_check.check_shape(shape_x)
    para_check.check_shape_rule(shape_assist1)
    para_check.check_shape(shape_assist1)
    para_check.check_shape_rule(shape_assist2)
    para_check.check_shape(shape_assist2)
    para_check.check_kernel_name(kernel_name)

    # check dim
    shape_x_length = len(shape_x)
    if dim < -shape_x_length or dim >= shape_x_length:
        raise RuntimeError(
            "Out of range, dim should be in [%d, %d], which is [%d]" % (-shape_x_length, shape_x_length - 1, dim))

    data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_x)
    data_input_as1 = tvm.placeholder(
        shape_assist1, name="data_input_as1", dtype=dtype_assist1)
    data_input_as2 = tvm.placeholder(
        shape_assist2, name="data_input_as2", dtype=dtype_assist2)

    res = index_fill_d_compute(data_input_x, data_input_as1, data_input_as2)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_x,
                              data_input_as1,
                              data_input_as2,
                              res]}

    build(schedule, config)
