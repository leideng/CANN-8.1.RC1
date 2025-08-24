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
fill_d
"""

from functools import reduce as functools_reduce

from tbe import tvm
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import shape_util
from te.utils import para_check


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=consider-using-in,invalid-name,redefined-builtin,raise-missing-from
@tbe_platform.fusion_manager.fusion_manager.register("fill_d")
def _fill_compute(data_value, data_output, data_dims, kernel_name="fill_d"):
    """
    Process fill operator

    Parameters
    ----------
    data_value: the placeholder of data input

    data_output : the dict of output

    data_dims: the shape of input

    kernel_name : cce kernel name

    Returns
    -------
    res : result of fill
    """

    in_dtype = data_value.dtype
    shape = data_dims
    # tbe.broadcast supports float16, float32, int32.
    # so convert int8, uint8 to float16
    if in_dtype == "int8" or in_dtype == "uint8":
        data_value = tbe.cast_to(data_value, "float16")

    if functools_reduce(lambda x, y: x*y, data_dims) == 1:
        if in_dtype == "int32":
            tensor_zero = tbe.broadcast(tvm.const(0, "int32"), shape)
            res = tbe.vadd(data_value, tensor_zero)
        elif in_dtype == "float32":
            tensor_zero = tbe.broadcast(tvm.const(0, "float32"), shape)
            res = tbe.vadd(data_value, tensor_zero)
        else:
            tensor_zero = tbe.broadcast(tvm.const(0, "float16"), shape)
            res = tbe.vadd(data_value, tensor_zero)
    else:
        res = tbe.broadcast(data_value, shape, in_dtype)

    if in_dtype == "int8" or in_dtype == "uint8":
        res = tbe.cast_to(res, in_dtype)

    return res


def _check_shape_compatibility(shape_in, shape_out):
    """
    Check if the shape of input tensor is compatible with output tensor.

    Parameters:
    ----------
    shape_in : shape of input tensor.

    shape_out : shape of output tensor.

    Returns:
    -------
    comp_shape_in : new shape_in compatible with shape_out.
    """

    comp_shape_in, comp_shape_out, shape_max = shape_util.broadcast_shapes(
        shape_in, shape_out, param_name_input1="value", param_name_input2="dims")
    try:
        if comp_shape_out != shape_max:
            raise ValueError('shape_in is not compatible with shape_out.')
    except RuntimeError:
        raise ValueError('shape_in is not compatible with shape_out.')
    return comp_shape_in


@para_check.check_op_params(para_check.REQUIRED_INPUT,
para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
para_check.KERNEL_NAME)
def fill_d(value, y, dims, kernel_name="fill_d"):
    """
    do  fill operation

    Parameters:
    ----------
    value:   the dict of input value, include shape and dtype,
             dtype support int8, uint8, int32, float16, float32

    y:  the dict of output

    dims:  the output shape, type support int32

    kernel_name: cce kernel name, default value is "fill_d"

    Returns
    -------
    None
    """
    # get the shape and dtype
    shape_value = value.get("shape")
    dtype_value = value.get("dtype")

    # check whether the shape is right
    para_check.check_shape(dims, param_name="dims")
    para_check.check_shape(shape_value, param_name="value")

    # check whether dtypes are right
    check_list_value = ("int8", "uint8", "int32", "float16", "float32")
    para_check.check_dtype(dtype_value, check_list_value, param_name="value")

    # get 2 input tensors: data_dims, data_value
    compatible_shape_in = _check_shape_compatibility(shape_value, dims)

    dtype_value = dtype_value.lower()
    data_value = tvm.placeholder(compatible_shape_in,
                                 dtype=dtype_value, name="data_value")
    res = _fill_compute(data_value, y, dims, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_value, res),
              "print_ir": False}
    tbe.cce_build_code(sch, config)
