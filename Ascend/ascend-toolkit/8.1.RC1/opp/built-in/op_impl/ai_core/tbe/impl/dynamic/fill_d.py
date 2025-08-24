# Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import gen_range


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=consider-using-in,invalid-name,redefined-builtin,raise-missing-from
@register_operator_compute("FillD", op_mode="dynamic", support_fusion=True)
def fill_d_compute(value, y, dims, kernel_name="fill_d"):
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
    src_dtype = value.dtype.lower()
    if src_dtype == "int8" or src_dtype == "uint8":
        value = tbe.cast_to(value, "float16")
    res = tbe.broadcast(value, dims)
    if src_dtype == "int8" or src_dtype == "uint8":
        res = tbe.cast_to(res, src_dtype)

    return res


@register_operator("FillD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
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
    shape_value = list(value.get("shape"))
    dtype_value = value.get("dtype").lower()
    dims_value = list(dims)

    # check whether the shape is right
    para_check.check_shape(dims, param_name="dims")
    para_check.check_shape(shape_value, param_name="value")

    # check whether dtypes are right
    check_list_value = ("int8", "uint8", "int32", "float16", "bfloat16", "float32")
    para_check.check_dtype(dtype_value, check_list_value, param_name="value")

    bro_shape = [1] * (len(dims_value) - len(shape_value)) + shape_value
    bro_shape_range = gen_range(bro_shape)

    value["shape"] = bro_shape
    value["range"] = bro_shape_range

    extra_params = {"disable_optimization": True}
    ins = classify([value], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            shape = shape_util.variable_shape([_input_x])[0]
            data = tvm.placeholder(shape, name="data", dtype=dtype_value)
            res = fill_d_compute(data, y, dims, kernel_name)
            tensors.append([data, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
