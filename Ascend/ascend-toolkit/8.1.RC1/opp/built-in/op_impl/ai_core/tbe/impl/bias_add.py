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
bias_add
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.dynamic.bias_add import op_select_format as bias_add_op_select_format


# 'pylint: disable=too-many-locals,redefined-builtin,unused-argument
# 'pylint: disable=too-many-statements,too-many-branches,invalid-name
def op_select_format(x, bias, y, data_format="NHWC", kernel_name="bias_add"):
    """
    1.when the length of x's ori_shape is less than or equal
    to 4 and the first element of the shape of bias is a multiple
    of 16. The Op BiasAdd can support NC1HWC0, NCHW and NHWC.

        for example:
        x : Tensor of (shape=(16, 16, 16, 16), "NHWC")
        bias : Tensor of (shape=(16, 16, 16, 16), "NHWC")
        The Op BiasAdd can process with NC1HWC0
        x : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        bias : Tensor of (shape=(2), "ND")

    2.when the length of x's ori_shape is greater then 4 and
    the first element of the shape of bias is a multiple of 16.
    The Op BiasAdd can support NDHWC, NCDHW, NDC1HWC0.

        x : Tensor of (shape=(16, 1, 16, 16, 16), "NDHWC")
        bias : Tensor of (shape=(2), "ND")
    """
    return bias_add_op_select_format(x, bias, y, data_format, kernel_name)


@register_operator_compute("bias_add", op_mode="static", support_fusion=True)
def bias_add_compute(x, bias, y, data_format, kernel_name="bias_add"):
    """
    calculating data's bias add

    Parameters
    ----------
    x : tvm tensor
              x data x
    bias : tvm tensor
              x data y
    y : tvm tensor
              y data
    data_format: A string.
                'N...C' and 'NC...' are supported.
    kernel_name : string
                  cce kernel name, default value is "bias_add"

    Returns
    -------
    res : y of the data's bias add
    """
    shape_x = shape_util.shape_to_list(x.shape)
    bias_broad = tbe.broadcast(bias, shape_x)
    res = tbe.vadd(x, bias_broad)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def bias_add(x, bias, y, data_format="NHWC", kernel_name="bias_add"):
    """
    algorithm: bias_and
    Reduce a tensor on a certain axis based on min

    Parameters
    ----------
    x : dict
              the shape and dtype of the tensor x
    bias : dict
              the shape and dtype of the tensor y
    y :  dict
              the shape and dtype of the tensor z
    data_format: A string.
                'N...C' and 'NC...' are supported.
    kernel_name : string
                  cce kernel name, default value is "bias_add"
    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    shape_bias = bias.get("shape")
    dtype_x = x.get("dtype").lower()
    dtype_bias = bias.get("dtype").lower()
    dtype_y = y.get("dtype").lower()
    data_format = data_format.upper()

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_bias, param_name="bias")

    check_tuple = ("float16", "float32", "int32")
    data_format_tuple = ("NCHW", "NHWC", "NDHWC", "NCDHW")
    para_check.check_dtype(dtype_x, check_tuple, param_name="x")
    para_check.check_dtype(dtype_bias, check_tuple, param_name="bias")
    para_check.check_dtype(dtype_y, check_tuple, param_name="y")

    if dtype_x != dtype_bias:
        error_detail = "dtype of x and bias should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", "bias", error_detail)

    if data_format not in data_format_tuple:
        excepted_format_list = "NCHW, NHWC, NDHWC, NCDHW"
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "data_format", \
                                                            excepted_format_list, data_format)

    if x.get("format") is not None and x.get("format").upper() == "NC1HWC0":
        ori_format_x = x.get("ori_format").upper()
        ori_shape_x = x.get("ori_shape")
        if len(shape_x) != 5:
            error_detail = "shape'rank of x should be 5 when input format is NC1HWC0"
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

        if ori_format_x != data_format:
            error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", data_format, ori_format_x)
        if bias.get("format") is not None and bias.get("format").upper() == "NC1HWC0":
            ori_shape_bias = bias.get("ori_shape")
            if ori_format_x == "NCHW" and ori_shape_x[1] != ori_shape_bias[0]:
                error_detail = "data_format is NCHW, shape_bias must be equal to the second axis of shape_x"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
            elif ori_format_x == "NHWC" and ori_shape_x[-1] != ori_shape_bias[0]:
                error_detail = "data_format is NHWC, shape_bias must be equal to the second axis of shape_x"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
        else:
            if ori_format_x == "NCHW" and ori_shape_x[1] != shape_bias[0]:
                error_detail = "data_format is NCHW, shape_bias must be equal to the second axis of shape_x"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
            elif ori_format_x == "NHWC" and ori_shape_x[-1] != shape_bias[0]:
                error_detail = "data_format is NHWC, shape_bias must be equal to the second axis of shape_x"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
        shape_bias = (1, shape_x[1], 1, 1, shape_x[4])

    elif x.get("format") is not None and x.get("format").upper() == "NDHWC":
        if len(shape_x) != 5:
            error_detail = "shape'rank of x should be 5 when input format is NDHWC"
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

        if shape_x[4] != shape_bias[0]:
            error_detail = "data_format is NDHWC, shape_bias must be equal to the fifth axis of shape_x"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
        shape_bias = (1, 1, 1, 1, shape_x[4])

    elif x.get("format") is not None and x.get("format").upper() == "NCDHW":
        if len(shape_x) != 5:
            error_detail = "shape'rank of x should be 5 when input format is NCDHW"
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
        if shape_x[1] != shape_bias[0]:
            error_detail = "data_format is NDHWC, shape_bias must be equal to the second axis of shape_x"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
        shape_bias = (1, shape_x[1], 1, 1, 1)

    elif x.get("format") is not None and x.get("format").upper() == "NDC1HWC0":
        if len(shape_x) != 6:
            error_detail = "shape'rank of x should be 6 when input format is NDC1HWC0"
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
        ori_shape_x = x.get("ori_shape")
        if x.get("ori_format").upper() == "NDHWC":
            if ori_shape_x[4] != shape_bias[0]:
                error_detail = "data_format is NDHWC, shape_bias must be equal to the fifth axis of shape_x"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
        elif x.get("ori_format").upper() == "NCDHW":
            if ori_shape_x[1] != shape_bias[0]:
                error_detail = "data_format is NDHWC, shape_bias must be equal to the second axis of shape_x"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
        shape_bias = (1, 1, shape_x[2], 1, 1, shape_x[5])

    else:
        shape_bias = shape_util.shape_refine(shape_bias)
        if data_format == "NCHW":
            if len(shape_x) < 2 or len(shape_x) > 4:
                error_detail = "shape'rank of x should be in range 2D to 4D when input format is NCHW"
                error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
            if shape_x[1] != shape_bias[0]:
                error_detail = "data_format is NCHW, shape_bias must be equal to the second axis of shape_x"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
            shape_bias = (1, shape_x[1],)
            for i in range(2, len(shape_x)):
                shape_bias = shape_bias + (1,)
        else:
            if len(shape_x) < 2:
                error_detail = "shape'rank of x should be larger than 2D"
                error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
            if shape_x[-1] != shape_bias[0]:
                error_detail = "data_format is NHWC, shape_bias must be equal to the last axis of shape_x"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "bias", error_detail)
            shape_bias = ()
            for i in range(0, len(shape_x)):
                if i != len(shape_x) - 1:
                    shape_bias = shape_bias + (1,)
                else:
                    shape_bias = shape_bias + (shape_x[-1],)

    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
    bias = tvm.placeholder(shape_bias, name="bias", dtype=dtype_bias)

    res = bias_add_compute(data_x, bias, y, data_format, kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_x, bias, res]}
    build(sch, config)
