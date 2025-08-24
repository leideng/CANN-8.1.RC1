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
fill
"""
from functools import reduce
from operator import mul
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import gen_range
from impl.util.util_compute import only_static_support


# 'pylint: disable=unused-argument,invalid-name
def check_supported(dims, value, y, kernel_name="fill"):
    """
    verify the types of fill supported by tbe
    """
    return True, ""


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("Fill", op_mode="dynamic", support_fusion=False)
def fill_compute(dims, value, y, kernel_name="fill"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    value : a number of float or int
    dtype : string
        the type of input
    kernel_name : str
        kernel name, default value is "fills"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    dims, shape_value, shape_max = \
    shape_util.broadcast_shapes(dims, value.shape,
                                param_name_input1="input_x",
                                param_name_input2="input_y")
    src_dtype = value.dtype
    if src_dtype == "int8" or src_dtype == "uint8":
        value = tbe.cast_to(value, "float16")
    res = tbe.broadcast(value, shape_max)
    if src_dtype == "int8" or src_dtype == "uint8":
        res = tbe.cast_to(res, src_dtype)

    return res


@register_operator("Fill")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def fill(dims, value, y, kernel_name="fill"):
    """
    do  fill operation

    Parameters:
    ----------
    dims : the dict of input
    value :  the dict of input
    y:  the dict of output
    kernel_name : cce kernel name, default value is "fill"

    Returns
    -------
    None
    """
    # get the shape and dtype
    tmp_dtype = value.get("dtype").lower()
    dtype = tmp_dtype if tmp_dtype != "bool" else "int8"
    dtype_dims = dims.get("dtype").lower()
    if is_unknown_rank_input([dims, value]):
        dims, value = [dims, dims] if is_unknown_rank_input(dims) else [value, value]
    else:
        dims["shape"] = [-1]
        dims['range'] = [[0, None]]

        const_value = dims.get('const_value')
        if const_value:
            const_value = list(const_value)
            shape_shape_adapt = [reduce(mul, const_value)]
            shape_range_adapt = gen_range(const_value)
        else:
            shape_shape_adapt = [-1]
            shape_range_adapt = [[0, None]]

        dims["shape"] = shape_shape_adapt
        dims['range'] = shape_range_adapt

    # check whether dtypes are right
    check_list = ("int8", "int32", "int64", "float16", "bfloat16", "float32")
    para_check.check_dtype(dtype, check_list)

    ins = classify([dims, value], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_dims, _value) in ins:
        with tbe.compute():
            shape_dim, shape = shape_util.variable_shape([_dims, _value])
            x_input = tvm.placeholder(shape, name="x_input", dtype=dtype)
            dim_input = tvm.placeholder(shape_dim, name="dim_input", dtype=dtype_dims)

            res = fill_compute(shape_dim, x_input, y, kernel_name=kernel_name)
            tensors.append([dim_input, x_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
