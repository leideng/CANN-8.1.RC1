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
bn_training_update_v2
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from tbe import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.dynamic.bn_training_update_v2 import op_select_format as bn_op_select_format


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-arguments
def op_select_format(x, sum, square_sum, scale, offset,
                     y, batch_mean, batch_variance, epsilon,
                     kernel_name="bn_training_update_v2"):
    """
    1. when input(x)'s ori_shape is [1, ? ,1, ?] and the format is NCHW
    the Op BNTrainingUpdateV2 can support NCHW.
    > for example:
    > x : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > sum : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > square_sum : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > scale : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > offset : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > the Op BNTrainingUpdateV2 can process with NC1HWC0:
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > sum : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > square_sum : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > scale : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > offset : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    """
    return bn_op_select_format(x, sum, square_sum, scale, offset,
                               y, batch_mean, batch_variance, epsilon,
                               kernel_name)


def _check_format(data_format, origin_foramt):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_foramt: str
        origin format of data

    Returns
    -------
    None
    """
    if data_format.upper() not in ("NC1HWC0", "NCHW", "NDC1HWC0"):
        error_reson = "The data format only supports NC1HWC0 and NCHW and NDC1HWC0."
        error_manager_vector.raise_err_specific_reson("bn_training_update_v2", error_reson)
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            error_reson = "The origin format only supports NCHW when format is NCHW"
            error_manager_vector.raise_err_specific_reson("bn_training_update_v2", error_reson)


# 'pylint: disable=locally-disabled,too-many-arguments
def _check_shape(shape_x, shape_sum, shape_square_sum,
                 shape_scale, shape_offset, data_format):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_sum: list or tuple
        sum's data shape
    shape_square_sum: list or tuple
        square_sum's data shape
    shape_scale: list or tuple
        scale's data shape
    shape_offset: list or tuple
        offset's data shape

    Returns
    -------
    None
    """
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_sum, param_name="sum")
    para_check.check_shape(shape_square_sum, param_name="square_sum")
    para_check.check_shape(shape_scale, param_name="scale")
    para_check.check_shape(shape_offset, param_name="offset")

    if len(shape_x) not in (5, 6) or len(shape_sum) not in (5, 6) \
            or len(shape_square_sum) not in (5, 6) or len(shape_scale) not in (5, 6) \
            or len(shape_offset) not in (5, 6):
        error_reson = "The data format is 5HD or 6HD, but some input's shape length is not 5 or 6"
        error_manager_vector.raise_err_specific_reson("bn_training_update_v2", error_reson)
    dim_c1 = 0
    dim_c0 = 0
    c1 = 0
    c0 = 0
    if data_format == "NC1HWC0":
        dim_c1 = shape_x[1]
        dim_c0 = shape_x[4]
        c1 = 1
        c0 = 4
    else:
        dim_c1 = shape_x[2]
        dim_c0 = shape_x[5]
        c1 = 2
        c0 = 5

    if shape_sum[c1] != dim_c1 or shape_sum[c0] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update_v2", "Dimension C of x and sum must be equal")
    if shape_square_sum[c1] != dim_c1 or shape_square_sum[c0] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update_v2",
                                                      "Dimension C of x and square_sum must be equal")
    if shape_scale[c1] != dim_c1 or shape_scale[c0] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update_v2",
                                                      "Dimension C of x and scale must be equal")
    if shape_offset[c1] != dim_c1 or shape_offset[c0] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update_v2",
                                                      "Dimension C of x and offset must be equal")


# 'pylint: disable=locally-disabled,too-many-arguments
def _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
    dtype_sum: str
        sum's data type
    dtype_square_sum: str
        square_sum's data type
    dtype_scale: str
        scale's data type
    dtype_offset: str
        offset's data type

    Returns
    -------
    None
    """
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")
    para_check.check_dtype(dtype_sum.lower(), ("float32",), param_name="sum")
    para_check.check_dtype(dtype_square_sum.lower(), ("float32",), param_name="square_sum")
    para_check.check_dtype(dtype_scale.lower(), ("float32",), param_name="scale")
    para_check.check_dtype(dtype_offset.lower(), ("float32",), param_name="offset")


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin
@tbe_platform.fusion_manager.fusion_manager.register("bn_training_update_v2")
def bn_training_update_v2_compute(x, sum, square_sum, scale, offset,
                                  y, batch_mean, batch_variance, epsilon,
                                  kernel_name="bn_training_update_v2"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    factor: float
        A ratio to caculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v2 compute
    """
    shape_x = shape_util.shape_to_list(x.shape)

    data_format = y.get("format")
    origin_format = y.get("ori_format")
    axis = list(range(len(shape_x)))

    # compute process is same when input's format is NDC1HWC0 or NC1HWC0, using N * D = N
    if data_format in ["NC1HWC0", "NDC1HWC0"]:
        axis = [0, 2, 3]
    if data_format == "NCHW":
        if origin_format == "NCHW":
            axis.pop(1)

    num = 1
    for cnt in axis:
        num *= shape_x[cnt]
    num_rec = 1.0/num

    # compute the saved mean of x
    save_mean_reduce = tbe.vmuls(sum, num_rec)

    # compute the saved variance of x
    variance_div = tbe.vmuls(square_sum, num_rec)
    variance_square = tbe.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = tbe.vsub(variance_div, variance_square)

    # compute the oefficient of y
    multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)
    multiplier = tbe.broadcast(multiplier_div, shape_x)

    addend_mul = tbe.vmul(multiplier_div, save_mean_reduce)
    addend_sub = tbe.vsub(offset, addend_mul)
    addend = tbe.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    is_cast = False
    if x.dtype == "float16":
        is_cast = True
        x = tbe.cast_to(x, "float32")

    res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)
    if is_cast:
        res_y = tbe.cast_to(res_y, "float16")

    res = [res_y, save_mean_reduce, save_variance_reduce]

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_training_update_v2(x, sum, square_sum, scale, offset,
                          y, batch_mean, batch_variance, epsilon,
                          kernel_name="bn_training_update_v2"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A 5HD Tensor for sum.
        The output of batch_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A 5HD Tensor for square_sum.
        The output of batch_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A 5HD Tensor for mean.
    offset: dict
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    factor: float
        A retio to caculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")

    dtype_x = x.get("dtype")
    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")

    data_format = x.get("format")
    origin_format = x.get("ori_format")

    _check_format(data_format, origin_format)

    if data_format in ("NC1HWC0", "NDC1HWC0"):
        _check_shape(shape_x, shape_sum, shape_square_sum,
                     shape_scale, shape_offset, data_format)
    else:
        shape_list = [1, 1, 1, 1]
        shape_list[1] = shape_x[1]
        shape_sum = shape_list
        shape_square_sum = shape_list

    _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset)
    if data_format == "NDC1HWC0":
        shape_x = [shape_x[0] * shape_x[1], shape_x[2], shape_x[3], shape_x[4], shape_x[5]]
        shape_sum = [shape_sum[0] * shape_sum[1], shape_sum[2], shape_sum[3], shape_sum[4], shape_sum[5]]
        shape_square_sum = [shape_square_sum[0] * shape_square_sum[1], shape_square_sum[2],
                            shape_square_sum[3], shape_square_sum[4], shape_square_sum[5]]

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input",
                                dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum,
                                       name="square_sum_input",
                                       dtype=dtype_square_sum.lower())
    scale_input = tvm.placeholder(shape_sum, name="scale_input",
                                  dtype=dtype_scale.lower())
    offset_input = tvm.placeholder(shape_sum, name="offset_input",
                                   dtype=dtype_offset.lower())

    res = bn_training_update_v2_compute(x_input, sum_input, square_sum_input,
                                        scale_input, offset_input, y,
                                        batch_mean, batch_variance,
                                        epsilon, kernel_name=kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    tensor_list = [x_input, sum_input, square_sum_input,
                   scale_input, offset_input, ] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)
