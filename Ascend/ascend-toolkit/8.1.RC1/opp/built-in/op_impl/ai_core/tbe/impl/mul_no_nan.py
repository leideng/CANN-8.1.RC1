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
mul_no_nan
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# 'pylint: disable=pointless-string-statement,unused-argument,invalid-name,too-many-locals
@register_operator_compute("mul_no_nan", op_mode="static", support_fusion=True)
def mul_no_nan_compute(input_x1, input_x2, output_y, kernel_name="mul_no_nan"):
    """
    calculating data

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
    """
    `np.where(np.equal(y, 0.), np.zeros((), dtype=dtype), np.multiply(x, y))`
    """
    src_dtype = input_x1.dtype.lower()
    shape_x1 = shape_util.shape_to_list(input_x1.shape)
    shape_x2 = shape_util.shape_to_list(input_x2.shape)

    shape_x1, shape_x2, shape_max = shape_util.produce_shapes(shape_x1, shape_x2)
    para_check.check_shape_size(shape_max, para_check.SHAPE_SIZE_LIMIT)
    input_x1 = tbe.broadcast(input_x1, shape_max)
    input_x2 = tbe.broadcast(input_x2, shape_max)

    mul_res = tbe.vmul(input_x1, input_x2)
    zero = tvm.const(0, dtype=src_dtype)
    zeros = tbe.broadcast(zero, shape_max)
    res = tbe.vcmpsel(input_x2,
                              zeros,
                              operation='eq',
                              slhs=zeros,
                              srhs=mul_res)
    return res


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
    output_z = y
    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")

    check_tuple = ("float16", "float32", "int32")
    input_data_type = x1.get("dtype").lower()
    para_check.check_dtype(input_data_type, check_tuple)

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x1, shape_x2, param_name_input1="x1",
                                                              param_name_input2="x2")
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    para_check.check_shape(shape_max)

    reshape_x, reshape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(reshape_x, name="data_1", dtype=input_data_type)
    data_y = tvm.placeholder(reshape_y, name="data_2", dtype=input_data_type)
    res = mul_no_nan_compute(data_x, data_y, output_z, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": (data_x, data_y, res)
    }
    build(schedule, config)
