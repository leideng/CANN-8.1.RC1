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
reduce_min_d
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl import reduce_min_d_tik

# define the type of None
NONETYPE = type(None)


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("reduce_min_d", op_mode="static", support_fusion=True)
def reduce_min_d_compute(input_min, output_min, axis, keep_dims, kernel_name="reduce_min_d"):
    """
    Reduce a tensor on a certain axis based on min

    Parameters:
    ----------
    input_min: TVM tensor
        contains input data
    output_min: dict
        dict of output
    axis: int or None
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range (-rank(input_tensor), rank(input_tensor))
    keep_dims: True or False
        if True, retains reduced dimensions with length 1,
        default value is None
    kernel_name: str
        cce kernel name, default value is "reduce_min_d"

    Returns
    -------
    res: TVM tensor
        the reduced tensor
    """
    shape = shape_util.shape_to_list(input_min.shape)
    shape_len = len(shape)
    if not axis:
        axis = range(shape_len)
    if hasattr(axis, 'index'):
        axis = list(axis)
    input_dtype = input_min.dtype.lower()

    if input_dtype != "float32" and input_dtype != "float16":
        input_min = tbe.cast_to(input_min, "float16")
    res_reduce_min = tbe.reduce_min(input_min, axis=axis, keepdims=keep_dims)
    res = tbe.cast_to(res_reduce_min, input_dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_INT),
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_min_d(input_min, output_min, axis, keep_dims=None, kernel_name="reduce_min_d"):
    """
    Reduce a tensor on a certain axis based on min

    Parameters:
    ----------
    input_min: dict
        dict of input, which contains shape and dtype
    output_min: dict
        dict of output, which contains shape and dtype
    axis: int or None
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range (-rank(input_tensor), rank(input_tensor))
    keep_dims: True or False
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name: str
        cce kernel name, default value is "reduce_min_d"

    Returns
    -------
    None
    """
    shape_input = input_min.get("shape")
    dtype_input = input_min.get("dtype")
    para_check.check_shape(shape_input, param_name="input_min")

    check_list = ("float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(dtype_input.lower(), check_list, param_name="input_min")

    shape_len = len(shape_input)

    if not axis:
        axis = range(shape_len)

    if hasattr(axis, 'index'):
        axis = list(axis)

    axis = shape_util.axis_check(shape_len, axis)

    is_5hdc = para_check.check_and_init_5hdc_reduce_support(input_min, axis)
    if not is_5hdc:
        shape_input, axis = shape_util.shape_refine(list(shape_input), axis)
        shape_input, axis = shape_util.simplify_axis_shape(shape_input, axis)

    data_input = tvm.placeholder(shape_input, name="data_input_" + kernel_name, dtype=dtype_input.lower())
    shape_len = len(shape_input)
    if dtype_input.lower() in ("float32", "int32") and len(axis) == 1 \
            and ((axis[0] == (shape_len - 1)) or (axis[0] == -1)):
        input_min["shape"] = tuple(shape_input)
        reduce_min_d_tik.reduce_min_d_tik(input_min, output_min, -1, kernel_name)
    else:
        res = reduce_min_d_compute(data_input, output_min, axis, keep_dims, kernel_name)
        if is_5hdc:
            res.ori_shape = input_min["ori_shape"]
            res.ori_format = input_min["ori_format"]
        with tvm.target.cce():
            sch = auto_schedule(res)

        config = {"name": kernel_name, "tensor_list": [data_input, res]}
        build(sch, config)
