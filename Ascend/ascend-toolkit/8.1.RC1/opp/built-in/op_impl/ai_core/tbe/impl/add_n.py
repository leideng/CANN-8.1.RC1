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
add_n
"""
import functools

from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.dynamic.add_n import add_n_compute_for_batchmatmul


def check_supported(inputs,
                    output,
                    tensor_num,
                    kernel_name="add_n"):
    """
    algorithm: add_n
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    inputs : list or tuple of dict
        A list of Tensor objects,
        each with same shape, range and dtype of first input,
        only support float16, float32, int32.
    output : dict
        shape, range and dtype of output,
        should be broadcast shape and type as input.
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    None
    """
    return True, ""


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,disable=too-many-locals
@register_operator_compute("add_n", op_mode="static", support_fusion=True)
def add_n_compute_for_fusion(datas, output, tensor_num, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders
        all input data
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """
    return add_n_compute_for_batchmatmul(datas, output, tensor_num, kernel_name)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
def add_n_compute(datas, output, tensor_num, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders
        all input data
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """
    data_type = datas[0].dtype
    has_covert_float32 = (data_type == "float16" and
                          tbe_platform.api_check_support("tbe.dsl.vadd", "float32"))

    first_data = datas[0] if not has_covert_float32 else tbe.cast_to(datas[0], "float32")

    res = first_data
    for i, data_n in enumerate(datas):
        if i == 0:
            continue
        temp_data = data_n if not has_covert_float32 else tbe.cast_to(data_n, "float32")
        res = tbe.vadd(res, temp_data)

    if has_covert_float32:
        res = tbe.cast_to(res, "float16")
    return res


@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def add_n(inputs, output, tensor_num, kernel_name="add_n"):
    """
    algorithm: add_n
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    inputs : list or tuple of dict
        A list of Tensor objects, each with same shape and type.
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    None
    """
    input_num = len(inputs)
    if input_num < 2:
        expected_value = "greater than or equal to 2"
        real_value = "less than 2"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "the length of inputs", expected_value,
                                                           real_value)

    if input_num != tensor_num:
        expected_value = tensor_num
        real_value = input_num
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "the length of inputs", expected_value,
                                                           real_value)

    shape_0 = inputs[0].get("shape")
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, shape_0)

    check_list = ("float16", "float32", "int32")
    data = []
    for i, input_dict in enumerate(inputs):
        shape_input = input_dict.get("shape")
        if list(shape_0) != list(shape_input):
            expected_value = list(shape_input)
            real_value = list(shape_0)
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "shape of input", expected_value,
                                                               real_value)
        para_check.check_shape(shape_input, param_name="inputs")
        dtype_input = input_dict.get("dtype").lower()
        para_check.check_dtype(dtype_input, check_list, param_name="inputs")
        data.append(tvm.placeholder(fuseshape, name="data_%d" % i, dtype=dtype_input))

    res = add_n_compute(data, output, tensor_num, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    data.append(res)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": data}

    build(schedule, config)
