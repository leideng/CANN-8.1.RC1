#!/usr/bin/python
# -*- coding: utf-8 -*-
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
batch_norm_grad
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode


# 'pylint: disable=locally-disabled,too-many-locals,too-many-arguments
# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator_compute("batch_norm_grad", op_mode="static", support_fusion=True)
def batch_norm_grad_compute(y_backprop, x, scale, reserve_space_1,
                            reserve_space_2,
                            x_backprop, scale_backprop,
                            offset_backprop, reserve_space_4,
                            reserve_space_5, epsilon, data_format, is_training,
                            kernel_name="batch_norm_grad"):
    """
    Compute for batch_norm_grad
    if is_training is True
        x_backprop:(y-(np.sum(y*(x-reserve_space_1)*np.power((reserve_space_2 +
        epsilon), (-0.5))))*(1/m)*(x-reserve_space_1)*np.power((reserve_space_2
        + epsilon), (-0.5))-(np.sum(y))*(1/m))*(scale*np.power((reserve_space_2
        + epsilon), (-0.5)))
    if is_training is False
        x_backprop:y*scale*np.power((reserve_space_2 + epsilon), (-0.5))
    scale_backprop:
    np.sum(y*(x-reserve_space_1)*np.power((reserve_space_2 + epsilon), (-0.5)))
    offset_backprop: np.sum(y)
    reserve_space_4:[0.]
    reserve_space_5:[0.]

    Parameters
    ----------
    y_backprop: TVM tensor
         the placeholder of y_backprop.
    x: TVM tensor
        the placeholder of x.
    scale: TVM tensor
        the placeholder of scale.
    reserve_space_1: TVM tensor
        the placeholder of reserve_space_1.
    reserve_space_2: TVM tensor
        the placeholder of reserve_space_2.
    x_backprop: dict
        dict of x_backprop, include keys(shape, dtype and format).
    scale_backprop: dict
        dict of scale_backprop, include keys(shape, dtype and format).
    offset_backprop: dict
        dict of offset_backprop, include keys(shape, dtype and format).
    reserve_space_4: dict
        dict of reserve_space_4, include keys(shape, dtype and format).
    reserve_space_5: dict
        dict of reserve_space_5, include keys(shape, dtype and format).
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    data_format: str
        An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
        Either "NHWC" (default) or "NCHW".
    is_training: bool
        An optional `bool`. Defaults to `True`.
        A bool value to indicate the operation is for training (default)
        or inference.
    kernel_name: str
        kernel name, default value is "batch_norm_grad"

    Returns
    -------
    res_list: TVM tensor list
        the result of batch_norm_grad compute
    """

    shape_x = shape_util.shape_to_list(x.shape)
    shape_scale = shape_util.shape_to_list(scale.shape)
    format_data = x_backprop.get("format")
    y_backprop, x_cast = _get_parms_cast(y_backprop, x)

    if format_data == "NHWC":
        axis = [0, 1, 2]
        num = shape_x[0] * shape_x[1] * shape_x[2]
    else:
        axis = [0, 2, 3]
        num = shape_x[0] * shape_x[2] * shape_x[3]
    num_rec = 1.0 / num

    reserve_space_1_broadcast = tbe.broadcast(reserve_space_1, shape_x)

    data_sub = tbe.vsub(x_cast, reserve_space_1_broadcast)
    data_adds = tbe.vadds(reserve_space_2, epsilon)
    data_rsqrt = tbe.vsqrt(data_adds)
    scalar_one = 1.0
    data_cast = tbe.broadcast(tvm.const(scalar_one, "float32"), shape_scale)
    data_rsqrts = tbe.vdiv(data_cast, data_rsqrt)
    data_rsqrts_broadcast = tbe.broadcast(data_rsqrts, shape_x)
    input_xl = tbe.vmul(data_sub, data_rsqrts_broadcast)
    scale_backprop_mul = tbe.vmul(y_backprop, input_xl)
    scale_backprop = tbe.sum(scale_backprop_mul, axis, True)
    offset_backprop = tbe.sum(y_backprop, axis, True)

    coef_mul = tbe.vmul(scale, data_rsqrts)
    coef_mul_broadcast = tbe.broadcast(coef_mul, shape_x)

    # output x_backprop
    if not is_training:
        x_backprop = tbe.vmul(y_backprop, coef_mul_broadcast)
    else:
        y_mean = tbe.vmuls(offset_backprop, num_rec)
        y_mean_broadcast = tbe.broadcast(y_mean, shape_x)
        y_cen = tbe.vsub(y_backprop, y_mean_broadcast)
        coef_vmuls = tbe.vmuls(scale_backprop, num_rec)
        coef_vmuls_broadcast = tbe.broadcast(coef_vmuls, shape_x)
        coef_vmul = tbe.vmul(coef_vmuls_broadcast, input_xl)
        coef_sub = tbe.vsub(y_cen, coef_vmul)
        x_backprop = tbe.vmul(coef_sub, coef_mul_broadcast)

    if x.dtype == "float16":
        x_backprop = tbe.cast_to(x_backprop, "float16")
    scalar_zero = 0.0
    # output_scale
    scale_backprop = tbe.vadds(scale_backprop, tvm.const(scalar_zero, "float32"))
    # output_offset
    offset_backprop = tbe.vadds(offset_backprop, tvm.const(scalar_zero, "float32"))

    if format_data not in ("NC1HWC0", "NDC1HWC0"):
        scale_backprop = tbe.sum(scale_backprop, axis, False)
        offset_backprop = tbe.sum(offset_backprop, axis, False)

    if reserve_space_4 is None and reserve_space_5 is None:
        res_list = [x_backprop, scale_backprop, offset_backprop]
    else:
        reserve_space_4, reserve_space_5 = _get_parms_reserve()
        res_list = [x_backprop, scale_backprop, offset_backprop,
                    reserve_space_4, reserve_space_5]

    return res_list


def _get_parms_reserve():
    """
    get reserve_space_4,reserve_space_5.

    Parameters
    ----------
    None

    Returns:
    -------
    reserve_space_4:TVM tensor
                        [0.]
    reserve_space_5:TVM tensor
                        [0.]
    """
    output_reserve_space_4_new = tbe.broadcast(tvm.const(0, "float32"), (1,), "float32")
    output_reserve_space_4 = tbe.vmuls(output_reserve_space_4_new, 0)
    output_reserve_space_5_new = tbe.broadcast(tvm.const(0, "float32"), (1,), "float32")
    output_reserve_space_5 = tbe.vmuls(output_reserve_space_5_new, 0)
    return output_reserve_space_4, output_reserve_space_5


def _get_parms_cast(y_backprop, x):
    """
    Cast parms to float32.

    Parameters
    ----------
    y_backprop: TVM tensor
        input of y_backprop
    x: TVM tensor
        input of x

    Returns:
    -------
    y_backprop
    x
    """
    dtype_y_backprop = y_backprop.dtype
    dtype_x = x.dtype
    if dtype_y_backprop == "float16":
        y_backprop = tbe.cast_to(y_backprop, "float32")
    if dtype_x == "float16":
        x = tbe.cast_to(x, "float32")
    return y_backprop, x


def _format_check(arg_input, data_format):
    """
    Function to check if the data_format is in line with norms.

    Parameters
    ----------
    input: dict
        dict of input
    data_format: str
        format of input data

    Returns
    -------
    None
    """
    format_data = arg_input.get("format")
    if format_data not in ("NHWC", "NCHW", "NC1HWC0", "NDC1HWC0"):
        error_manager_vector.raise_err_input_format_invalid("batch_norm_grad", "arg_input", \
                                                            ["NHWC", "NCHW", "NC1HWC0", "NDC1HWC0"], format_data)
    if data_format not in ("NHWC", "NCHW", "NCDHW", "NDHWC"):
        error_manager_vector.raise_err_input_format_invalid("batch_norm_grad", "data_format", \
                                                            ["NHWC", "NCHW", "NCDHW", "NDHWC"], data_format)


def _check_shape_len(shape_y_backprop, shape_x, shape_scale,
                     shape_reserve_space_1, shape_reserve_space_2,
                     data_format):
    """
     check shape must be 1D or 4D or 5D .

     Parameters
     ----------
     shape_y_backprop: list or tuple
         shape of y_backprop
     shape_x: list or tuple
         shape of x
     shape_scale: list or tuple
         shape of scale
     shape_reserve_space_1: list or tuple
         shape of reserve_space_1
     shape_reserve_space_2: list or tuple
         shape of reserve_space_2

     Returns
     -------
     None
     """
    if data_format in ("NHWC", "NCHW"):
        if len(shape_x) != 4:
            error_detail = "This operator can only support 4D, " \
                           "but x's shape length is not 4!"
            error_manager_vector.raise_err_input_shape_invalid("batch_norm_grad", \
                                                               "x", error_detail)
        if len(shape_scale) != 1 or len(shape_reserve_space_1) != 1 or len(
                shape_reserve_space_2) != 1:
            error_detail = "input feature map must be 1D format in kernel!" \
                           "but scale,reserve_space_1,reserve_space_2 is %d,%d,%d"\
                           % (len(shape_scale), len(shape_reserve_space_1), len(shape_reserve_space_2))
            error_manager_vector.raise_err_input_shape_invalid("batch_norm_grad", \
                                                               "scale,reserve_space_1,reserve_space_2", error_detail)
    else:
        if len(shape_y_backprop) not in (5, 6) or len(shape_scale) not in (5, 6):
            error_detail = "This operator can only support 5D or 6D, " \
                           "but (y_backprop,x,scale)'s shape length is %d,%d,%d" \
                           % (len(shape_y_backprop), len(shape_x), len(shape_scale))
            error_manager_vector.raise_err_input_shape_invalid("batch_norm_grad", \
                                                               "y_backprop,x,scale", error_detail)
        if len(shape_reserve_space_1) not in (5, 6) or len(shape_reserve_space_2) not in (5, 6):
            error_detail = "the shape length of reserve_space_1 and reserve_space_2 is not 5 or 6!"
            error_manager_vector.raise_err_two_input_shape_invalid("batch_norm_grad", "reserve_space_1", \
                                                               "reserve_space_2", error_detail)


def _check_shape(shape_y_backprop, shape_x, shape_scale, shape_reserve_space_1,
                 shape_reserve_space_2, data_format):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_y_backprop: list or tuple
        shape of y_backprop
    shape_x: list or tuple
        shape of x
    shape_scale: list or tuple
        shape of scale
    shape_reserve_space_1: list or tuple
        shape of reserve_space_1
    shape_reserve_space_2: list or tuple
        shape of reserve_space_2
    Returns
    -------
    None
    """
    para_check.check_shape(shape_y_backprop, param_name="y_backprop")
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_scale, param_name="scale")
    para_check.check_shape(shape_reserve_space_1, param_name="reserve_space_1")
    para_check.check_shape(shape_reserve_space_2, param_name="reserve_space_2")
    if data_format in ("NHWC", "NCHW"):
        shape_c0 = shape_y_backprop[-1] if data_format == "NHWC" else shape_y_backprop[1]
        if shape_scale[-1] != shape_c0:
            error_detail = "shape_scale's last dim must be same as shape_y_backprop's!"
            error_manager_vector.raise_err_two_input_shape_invalid("batch_norm_grad", "scale", \
                                                                   "y_backprop", error_detail)
    else:
        shape_n = shape_scale[0]
        if data_format == "NDC1HWC0":
            shape_h = shape_scale[3]
            shape_w = shape_scale[4]
            dim_c1 = shape_y_backprop[2]
            shape_c1 = shape_scale[2]
        else:
            shape_h = shape_scale[2]
            shape_w = shape_scale[3]
            dim_c1 = shape_y_backprop[1]
            shape_c1 = shape_scale[1]
        if shape_n != 1 or shape_h != 1 or shape_w != 1:
            error_detail = "Dimensions except Dimension C must be one for shape_scale"
            error_manager_vector.raise_err_input_shape_invalid("batch_norm_grad", \
                                                               "scale", error_detail)
        if shape_c1 != dim_c1:
            error_detail = "Dimension C must be equal"
            error_manager_vector.raise_err_input_shape_invalid("batch_norm_grad", \
                                                               "scale", error_detail)


def _change_shape(shape_scale, shape_reserve_space_1, shape_reserve_space_2,
                  data_format):
    """
     change shape

     Parameters
     ----------
     shap_scale: list
         shape of scale
     shape_reserve_space_1: list
         shape of reserve_space_1
     shape_reserve_space_2: list
         shape of reserve_space_2
     data_format: str
         format of input data

     Returns
     -------
     params_shape: dict
         {"shape_scale_change": shape_scale_change,
         "shape_reserve_space_1_change": shape_reserve_space_1_change,
         "shape_reserve_space_2_change": shape_reserve_space_2_change}
     """
    if data_format == "NHWC":
        shape_scale_change = [1, 1, 1] + list(shape_scale)
        shape_reserve_space_1_change = [1, 1, 1] + list(shape_reserve_space_1)
        shape_reserve_space_2_change = [1, 1, 1] + list(shape_reserve_space_2)
    elif data_format == "NCHW":
        shape_scale_change = [1] + list(shape_scale) + [1, 1]
        shape_reserve_space_1_change = [1] + list(shape_reserve_space_1) + [
            1, 1]
        shape_reserve_space_2_change = [1] + list(shape_reserve_space_2) + [
            1, 1]
    elif data_format == "NDC1HWC0":
        shape_scale_change = [shape_scale[0] * shape_scale[1], shape_scale[2], shape_scale[3],
                              shape_scale[4], shape_scale[5]]
        shape_reserve_space_1_change = [shape_reserve_space_1[0] * shape_reserve_space_1[1], shape_reserve_space_1[2],
                                        shape_reserve_space_1[3], shape_reserve_space_1[4], shape_reserve_space_1[5]]
        shape_reserve_space_2_change = [shape_reserve_space_2[0] * shape_reserve_space_2[1], shape_reserve_space_2[2],
                                        shape_reserve_space_2[3], shape_reserve_space_2[4], shape_reserve_space_2[5]]
    else:
        shape_scale_change = shape_scale
        shape_reserve_space_1_change = shape_reserve_space_1
        shape_reserve_space_2_change = shape_reserve_space_2
    params_shape = {"shape_scale_change": shape_scale_change,
                    "shape_reserve_space_1_change":
                        shape_reserve_space_1_change,
                    "shape_reserve_space_2_change":
                        shape_reserve_space_2_change}
    return params_shape


# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@register_operator("BatchNormGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def batch_norm_grad(y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3,
                    x_backprop, scale_backprop,
                    offset_backprop, reserve_space_4, reserve_space_5,
                    epsilon=0.0001, data_format="NHWC", is_training=True,
                    kernel_name="batch_norm_grad"):
    """
    algorithm: batch_norm_grad
    Batch normalization grad.

    Parameters
    ----------
    y_backprop: dict
        dict of y_backprop.
        source data type, support "float16", "float32".
    x: dict
        dict of x.
        source data type, support "float16", "float32".
    scale: dict
        dict of scale.
        source data type, support "float32".
    reserve_space_1: dict
        dict of reserve_space_1.
        source data type, support "float32".
        When is_training is True, a Tensor for the computed batch
        mean to be reused in gradient computation. When is_training is
        False, a Tensor for the population mean to be reused in both
        1st and 2nd order gradient computation.
    reserve_space_2: dict
        dict of reserve_space_2.
        source data type, support "float32".
        When is_training is True, a Tensor for the computed batch
        variance (inverted variance in the cuDNN case) to be reused in
        gradient computation. When is_training is False, a Tensor
        for the population variance to be reused in both 1st and 2nd
        order gradient computation.
    reserve_space_3: dict
        dict of reserve_space_3, this optional input is used to keep
        the parameters in this function same with the op prototype. 
    x_backprop: dict
        dict of output. Has the same type as `y_backprop`.
    scale_backprop: dict
        dict of scale_backprop. Has the same type as `reserve_space_1`.
    offset_backprop: dict
        dict of offset_backprop. Has the same type as `reserve_space_1`.
    reserve_space_4: dict
        dict of reserve_space_4.
    reserve_space_5: dict
        dict of reserve_space_5.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    data_format: str
        An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
        Either "NHWC" (default) or "NCHW".
    is_training: bool
        An optional `bool`. Defaults to `True`.
        A bool value to indicate the operation is for training (default)
        or inference.
    kernel_name: str
        kernel name, default value is "batch_norm_grad"

    Returns
    -------
    None
    """
    shape_y_backprop = y_backprop.get("shape")
    if len(shape_y_backprop) == 2:
        shape_y_backprop = list(shape_y_backprop) + [1, 1]
    shape_x = x.get("shape")
    if len(shape_x) == 2:
        shape_x = list(shape_x) + [1, 1]
    shape_scale = scale.get("shape")
    shape_reserve_space_1 = reserve_space_1.get("shape")
    shape_reserve_space_2 = reserve_space_2.get("shape")

    y_backprop_dtype = y_backprop.get("dtype").lower()
    x_dtype = x.get("dtype").lower()
    scale_dtype = scale.get("dtype").lower()
    reserve_space_1_dtype = reserve_space_1.get("dtype").lower()
    reserve_space_2_dtype = reserve_space_2.get("dtype").lower()

    para_check.check_dtype(y_backprop_dtype, ("float32", "float16"), param_name="y_backprop")
    para_check.check_dtype(x_dtype, ("float32", "float16"), param_name="x")
    para_check.check_dtype(scale_dtype, ("float32",), param_name="scale")
    para_check.check_dtype(reserve_space_1_dtype, ("float32",), param_name="reserve_space_1")
    para_check.check_dtype(reserve_space_2_dtype, ("float32",), param_name="reserve_space_2")
    shape_util.compare_tensor_dict_key(y_backprop, x, "dtype")

    _format_check(x, data_format)
    format_data = x.get("format")

    _check_shape_len(shape_y_backprop, shape_x, shape_scale,
                     shape_reserve_space_1, shape_reserve_space_2,
                     format_data)
    _check_shape(shape_y_backprop, shape_x, shape_scale, shape_reserve_space_1,
                 shape_reserve_space_2, format_data)
    shape_util.compare_tensor_dict_key(y_backprop, x, "shape")
    shape_util.compare_tensor_dict_key(scale, reserve_space_1, "shape")
    shape_util.compare_tensor_dict_key(scale, reserve_space_2, "shape")

    shape_list = _change_shape(shape_scale, shape_reserve_space_1,
                               shape_reserve_space_2, format_data)
    if format_data == "NDC1HWC0":
        shape_y_backprop = [shape_y_backprop[0] * shape_y_backprop[1], shape_y_backprop[2],
                            shape_y_backprop[3], shape_y_backprop[4], shape_y_backprop[5]]
        shape_x = [shape_x[0] * shape_x[1], shape_x[2], shape_x[3], shape_x[4], shape_x[5]]

    y_backprop = tvm.placeholder(shape_y_backprop, name="y_backprop",
                                 dtype=y_backprop_dtype)
    x = tvm.placeholder(shape_x, name="x", dtype=x_dtype)
    scale = tvm.placeholder(shape_list.get("shape_scale_change"),
                            name="scale", dtype=scale_dtype)
    reserve_space_1 = tvm.placeholder(
        shape_list.get("shape_reserve_space_1_change"), name="reserve_space_1",
        dtype=reserve_space_1_dtype)
    reserve_space_2 = tvm.placeholder(
        shape_list.get("shape_reserve_space_2_change"), name="reserve_space_2",
        dtype=reserve_space_2_dtype)

    res_list = batch_norm_grad_compute(y_backprop, x, scale,
                                       reserve_space_1, reserve_space_2,
                                       x_backprop,
                                       scale_backprop, offset_backprop,
                                       reserve_space_4, reserve_space_5,
                                       epsilon, data_format, is_training,
                                       kernel_name=kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res_list)

    tensor_list = [y_backprop, x, scale, reserve_space_1,
                   reserve_space_2] + list(res_list)
    config = {"name": kernel_name,
              "tensor_list": tensor_list}

    tbe.cce_build_code(sch, config)
