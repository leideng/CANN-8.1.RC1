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
reduce_prod_d
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute

NoneType = type(None)


def check_supported(x, y, axes, keep_dims=None, kernel_name="reduce_prod_d"):
    """
    check the op support situation.
    Go to AICPU when doing reduction with float32 type on the last axis.
    """
    input_shape = x.get("ori_shape")
    input_dtype = x.get("dtype").lower()
    dim_num = len(input_shape)
    for axis in axes:
        if axis in [-1, dim_num - 1] and input_dtype == "float32" and input_shape[-1] == 1:
            reason = "the axis is not supported, axis:%s" % axis
            return False, reason
    return True, ""


# 'pylint: disable=locally-disabled, unused-argument
@register_operator_compute("reduce_prod_d", op_mode="static", support_fusion=True)
def reduce_prod_d_compute(data_input, output_y, axes, keepdims, kernel_name="reduce_prod_d"):
    """
    Reduce a tensor on a certain axes based on product.

    Parameters:
    ----------
    data_input: dict
        shape and dtype of input
    output_y: dict
        shape and dtype of output
    axes: int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keep_dims: bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name: str
        cce kernel name, default value is reduce_prod_d

    Returns
    -------
    res: TVM tensor
    """
    shape = shape_util.shape_to_list(data_input.shape)
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)

    inp_dtype = data_input.dtype

    if data_input.dtype != "float32" or data_input.dtype != "float16":
        if tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
            data_input = tbe.cast_to(data_input, "float32")
        else:
            data_input = tbe.cast_to(data_input, "float16")
    res = tbe.reduce_prod(data_input, axes, keepdims)
    res = tbe.cast_to(res, inp_dtype, f1628IntegerFlag=True)
    return res


# 'pylint: disable=invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_LIST_INT),
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_prod_d(x, y, axes, keep_dims=None, kernel_name="reduce_prod_d"):
    """
    Reduce a tensor on a certain axes based on product.

    Parameters:
    ----------
    x : dict
        shape and dtype of input
    y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keep_dims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name : str
        cce kernel name, default value is reduce_prod_d
    Returns
    -------
    None
    """
    shape = x.get("shape")
    para_check.check_shape(shape, param_name="x")

    inp_dtype = x.get("dtype").lower()
    check_list = ("float16", "float32", "int8", "uint8")
    para_check.check_dtype(inp_dtype, check_list, param_name="x")

    shape_len = len(shape)

    if not axes:
        axes = range(shape_len)

    if hasattr(axes, 'index'):
        axes = list(axes)

    axes = shape_util.axis_check(shape_len, axes)
    para_check.check_reduce_shape_rule(shape)

    shape, axes = shape_util.shape_refine(list(shape), axes)
    shape, axes = shape_util.simplify_axis_shape(shape, axes)

    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)
    with tvm.target.cce():
        res = reduce_prod_d_compute(data_input, y, axes, keep_dims, kernel_name)
        sch = auto_schedule(res)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": [data_input, res]}
    build(sch, config)
