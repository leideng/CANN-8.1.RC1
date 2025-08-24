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
bias
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_compute import only_static_support
from impl.util.util_common import infershape_for_broadcast
from impl.bias import op_select_format as static_op_select_format


# 'pylint: disable=too-few-public-methods
# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
# 'pylint: disable=too-many-boolean-expressions,too-many-locals,unused-variable
def op_select_format(x, bias, y, axis=1, num_axes=1, bias_from_blob=True, kernel_name="bias"):
    """
    1. when input x's ori_shape is 4, and bias's shape is not 1. The
       Op Bias can support ND and NC1HWC0.
    > for example:
    > x : Tensor of (shape=(16, 16, 16, 16, 16), "NC1HWC0")
    > bias : Tensor of (shape=(16, 16, 16, 16, 16), "NC1HWC0")

    2. In other scenes, all input(x, bias) only support ND.
    > for example:
    > x : Tensor of (shape=(2), "ND")
    > bias : Tensor of (shape=(2), "ND")
    """
    return static_op_select_format(x, bias, y, axis, num_axes, bias_from_blob, kernel_name)


# 'pylint: disable=too-few-public-methods
# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
# 'pylint: disable=too-many-boolean-expressions,too-many-locals,unused-variable
def _param_bias_check(shape_x, shape_bias):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_bias : list or tuple.
        shape of bias.

    Returns
    -------
    None
    """
    length_x = len(shape_x)
    length_bias = len(shape_bias)

    if length_bias != 1:
        if length_x != length_bias:
            error_detail = "length_x and length_bias must be equal"
            error_manager_vector.raise_err_two_input_shape_invalid("bias", "length_x",
                                                                   "length_bias", error_detail)


def _get_param_bias_shape_and_range(shape_x, shape_bias, range_bias):
    """
    Function to calculate the shape of bias.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_bias : list or tuple.
        shape of bias.

    Returns
    -------
    None
    """
    length_x = len(shape_x)
    length_bias = len(shape_bias)

    if length_bias == 1:
        shape = [1] * length_x
        bias_range = ((1, 1),) * length_x
    else:
        shape = list(shape_bias)
        bias_range = tuple(range_bias)

    return shape, bias_range


# 'pylint: disable=too-many-branches
def _check_shape_axis(shape_x, shape_bias, axis, num_axes, bias_from_blob):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_bias: list or tuple
        bias's data shape
    axis : int
        A int num indicates shape of bias when bias is from bottom.
    num_axes: int
        A int num indicates shape of bias when bias is from blob.
    bias_from_blob:
        A bool value indicates scale is from blob or bottom.

    Returns
    -------
    None
    """
    length_x = len(shape_x)
    length_bias = len(shape_bias)

    if (axis >= length_x) or (axis < (-length_x)):
        error_detail = "axis should be greater than the length of shape_x"
        error_manager_vector.raise_err_two_input_shape_invalid("bias", "axis", "shape_x",
                                                               error_detail)

    if num_axes < -1:
        expected_value = "greater than -1"
        real_value = "less than or equal -1"
        error_manager_vector.raise_err_input_value_invalid("bias", "num_axes",
                                                           expected_value, real_value)

    if axis < 0:
        axis_ = length_x + axis
    else:
        axis_ = axis

    # from blob
    if bias_from_blob:
        if num_axes == -1:
            bias_num = length_x - axis_
            if length_bias != bias_num:
                error_detail = "length_bias and bias_num must be equal"
                error_manager_vector.raise_err_two_input_shape_invalid("bias", "length_bias",
                                                                       "bias_num", error_detail)

        if num_axes == 0:
            if length_bias != 1:
                error_detail = "bias must be a scalar"
                error_manager_vector.raise_err_two_input_shape_invalid("bias", "length_bias",
                                                                       "shape_bias", error_detail)

        if num_axes > 0:
            num_axis = axis_ + num_axes

            if num_axis > length_x:
                error_detail = "bias shape extends x shape when applied"
                error_manager_vector.raise_err_two_input_shape_invalid("bias", "num_axis",
                                                                       "length_x", error_detail)

            if length_bias != num_axes:
                error_detail = "length_bias and num_axes must be equal"
                error_manager_vector.raise_err_two_input_shape_invalid("bias", "length_bias",
                                                                       "num_axes", error_detail)

    # from bottom
    if not bias_from_blob:
        if length_bias != 1:
            bias_num = axis_ + length_bias

            if bias_num > length_x:
                error_detail = "bias shape extends x shape when applied"
                error_manager_vector.raise_err_two_input_shape_invalid("bias", "bias_num",
                                                                       "length_x", error_detail)


def get_shape_and_bias(shape_x, shape_bias, axis_, num_axes, bias_from_blob, range_bias):
    """
    Function to calculate shape of bias.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_bias: list or tuple
        bias's data shape
    axis_ : int
        A int num indicates shape of bias when bias is from bottom.
    num_axes:
        A int num indicates shape of bias when bias is from blob.
    bias_from_blob:
        A bool value indicates bias is from blob or bottom.

    Returns
    -------
    shape: list
        the shape of bias
    """
    length_x = len(shape_x)
    length_bias = len(shape_bias)
    if bias_from_blob:
        if num_axes == -1:
            shape_left = [1] * axis_
            shape = shape_left + list(shape_bias)
            bias_range = ((1, 1),) * axis_ + tuple(range_bias)
        elif num_axes == 0:
            shape = [1] * length_x
            bias_range = ((1, 1),) * length_x
        else:
            left_length = length_x - num_axes - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_bias) + shape_right
            bias_range = ((1, 1),) * axis_ + tuple(range_bias) + ((1, 1),) * left_length
    else:
        if length_bias == 1 and shape_bias[0] == 1:
            shape = [1] * length_x
            bias_range = ((1, 1),) * length_x
        else:
            left_length = length_x - length_bias - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_bias) + shape_right
            bias_range = ((1, 1),) * axis_ + tuple(range_bias) + ((1, 1),) * left_length

    return shape, bias_range


def _check_dtype(dtype_x, dtype_bias):
    """
    Function to check dtype of input data.

    Parameters
    ----------

    dtype_x: str
        dtype of x data
    dtype_bias: str
        dtype of bias data
    Returns
    -------
    None
    """
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS"):
        if dtype_x == "float32" or dtype_bias == "float32":
            error_detail = "float32 is not support in HISI"
            error_manager_vector.raise_err_two_input_dtype_invalid("bias", "x", "bias",
                                                                   error_detail)
        check_tuple = ("float16",)
    else:
        check_tuple = ("float32", "float16", "bfloat16")

    para_check.check_dtype(dtype_x, check_tuple, param_name="x")
    para_check.check_dtype(dtype_bias, check_tuple, param_name="bias")


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
@register_operator_compute("Bias", op_mode="dynamic", support_fusion=only_static_support, support_bfp16=True)
def bias_compute(x, bias, y, axis, num_axes, bias_from_blob, kernel_name="bias"):
    """
    calculating data
    y = x + bias

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    bias : TVM tensor
        the placeholder of bias
    y : dict
        dict of y, include keys(shape and dtype)
    axis : int
        A int num indicates shape of bias when bias is from bottom.
    num_axes: int
        A int num indicates shape of bias when bias is from blob.
    bias_from_blob:
        A bool value indicates bias is from blob or bottom.
    kernel_name : str
        kernel name, default value is "bias"

    Returns
    -------
    output tensor
    """
    dtype_x = x.dtype
    dtype_bias = bias.dtype

    is_cast = False
    if tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        if dtype_x == "float16":
            is_cast = True
            x = tbe.cast_to(x, 'float32')

        if dtype_bias == "float16":
            bias = tbe.cast_to(bias, 'float32')

    _, _, shape_max = shape_util.broadcast_shapes(x.shape,
                                                  bias.shape,
                                                  param_name_input1="x",
                                                  param_name_input2="bias")

    data_x = tbe.broadcast(x, shape_max)
    data_bias = tbe.broadcast(bias, shape_max)

    res = tbe.vadd(data_x, data_bias)

    if is_cast:
        res = tbe.cast_to(res, dtype_x)

    return res


# 'pylint: disable=too-many-locals
@register_operator("Bias")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def bias(x, bias, y, axis=1, num_axes=1, bias_from_blob=True, kernel_name="bias"):
    """
    algorithm: Bias
    y = x + bias

    Parameters
    ----------
    x : dict
        dict of input, A Tensor for input data.
    bias : dict
        dict of bias,
        A Tensor for bias, to shift to the input data.
    y : dict
        dict of output,
        A Tensor for y, should be same shape and type as x.
    axis : int
        A int num indicates shape of bias when bias is from bottom.
    num_axes: int
        A int num indicates shape of bias when bias is from blob.
    bias_from_blob:
        A bool value indicates bias is from blob or bottom.
    kernel_name : str
        kernel name, default value is "bias"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    shape_bias = bias.get("shape")
    range_x = x.get("range")
    range_bias = bias.get("range")
    dtype_x = x.get("dtype").lower()
    dtype_bias = bias.get("dtype").lower()

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_bias, param_name="bias")
    _check_dtype(dtype_x, dtype_bias)

    length_x_ori = len(x.get("ori_shape"))

    if any((axis is None, num_axes is None, bias_from_blob is None)):
        tbe_context.get_context().add_compile_info("is_unknown_rank", True)
    else:
        shape_bias_new = []

        if length_x_ori == 4:
            _param_bias_check(shape_x, shape_bias)
            shape_bias_new, range_bias_new = _get_param_bias_shape_and_range(shape_x, shape_bias, range_bias)
        else:
            _check_shape_axis(shape_x, shape_bias, axis, num_axes, bias_from_blob)

            length_x = len(shape_x)
            if axis < 0:
                axis_ = length_x + axis
            else:
                axis_ = axis

            shape_bias_new, range_bias_new = \
                get_shape_and_bias(shape_x, shape_bias, axis_, num_axes, bias_from_blob, range_bias)

        bias["shape"] = shape_bias_new
        bias["range"] = range_bias_new
        bias["ori_shape"] = shape_bias_new
        tbe_context.get_context().add_compile_info("boardcast_bias_shape", shape_bias_new)

    _ = infershape_for_broadcast(x.get("shape"), bias.get("shape"), "Bias", "x", "bias")
    ins = classify([x, bias], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []

    for (_x, _bias) in ins:
        with tbe.compute():
            x_shape, bias_shape = \
                shape_util.variable_shape([_x, _bias])
            x_input = tvm.placeholder(x_shape, name="x_input", dtype=dtype_x)
            bias_input = tvm.placeholder(bias_shape, name="bias_input", dtype=dtype_bias)

            res = bias_compute(x_input, bias_input, y, axis, num_axes, bias_from_blob, kernel_name)
            tensors.append([x_input, bias_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
