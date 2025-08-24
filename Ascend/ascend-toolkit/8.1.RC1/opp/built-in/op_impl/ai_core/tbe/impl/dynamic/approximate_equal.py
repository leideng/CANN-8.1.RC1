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
approximate_equal
"""
import operator

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


NUM_ONE = 1.0
NUM_ZERO = 0.0


class ApproximateEqualAttrInfo:
    """
    define attr info
    """
    ATTR_TOLERANCE = OpAttr(0, "tolerance", "Float", 1e-5)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals
@register_operator_compute("ApproximateEqual", op_mode="dynamic", support_fusion=True)
def approximate_equal_compute(input_x, input_y, output_z, tolerance,
                              kernel_name="approximate_equal"):
    """
    algorithm: approximate_equal

    calculating abs(x-y) <= tolerance

    Parameters
    ----------
    input_x : the placeholders of input data
    input_y : the placeholders of input data
    tolerance: default 1e-5
    output_z: shape and dtype of output
    kernel_name: cce kernel name, default value is "approximate_equal"
    Returns
    -------
    the function of _approximate_equal_compute
    """

    input_dtype = input_x.dtype
    if input_dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")

    res_vsub = tbe.vsub(input_x, input_y)
    res_vabs = tbe.vabs(res_vsub)

    res_vabs = tbe.cast_to(res_vabs, input_x.dtype)
    tolerance = get_attr_by_cls(tolerance, ApproximateEqualAttrInfo.ATTR_TOLERANCE, input_x.dtype)
    tol_tensor = tbe.broadcast(tolerance, input_x.shape)

    zero_rb_tensor = tbe.broadcast(tvm.const(NUM_ZERO, "float32"), input_x.shape)
    one_rb_tensor = tbe.broadcast(tvm.const(NUM_ONE, "float32"), input_x.shape)
    res = tbe.vcmpsel(res_vabs, tol_tensor, 'le', one_rb_tensor, zero_rb_tensor)

    res = tbe.cast_to(res, "int8")

    return res


@register_operator("ApproximateEqual")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def approximate_equal(input_x, input_y, output_z, tolerance=1e-5,
                      kernel_name="approximate_equal"):
    """
    abs(x-y) <= tolerance
    Parameters
    ----------
    input_x : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    input_y : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    output_z : dict, include shape and dtype, reserve

    tolerance: default 1e-5

    kernel_name : str
        cce kernel name, default value is "approximate_equal"

    Returns
    ------
    None
    """

    in_x_dtype = input_x.get("dtype").lower()
    in_y_dtype = input_y.get("dtype").lower()

    # check input tensor data_type
    check_list = ("float16", "float32")
    para_check.check_dtype(in_x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(in_y_dtype, check_list, param_name="input_y")

    if not operator.eq(in_x_dtype, in_y_dtype):
        error_detail = "dtype of input_x and input_y should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_x", \
                                                               "input_y", error_detail)

    if tolerance is not None and tolerance < 0:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "tolerance", \
                                                           ">= 0", tolerance)

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x_, y_) in ins:
        with tbe.compute():
            shape_x, shape_y = shape_util.variable_shape([x_, y_])
            input_x = tvm.placeholder(shape_x, dtype=in_x_dtype, name="input_x")
            input_y = tvm.placeholder(shape_y, dtype=in_y_dtype, name="input_y")
            res = approximate_equal_compute(input_x, input_y, output_z, tolerance, kernel_name)
            tensors.append([input_x, input_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
    