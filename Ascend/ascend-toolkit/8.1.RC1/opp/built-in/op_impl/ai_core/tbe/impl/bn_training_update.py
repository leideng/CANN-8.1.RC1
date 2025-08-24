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
bn_training_update
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
import impl.dynamic as dimpl
from tbe import tvm
from te.utils import shape_util
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from tbe.common.buildcfg import get_current_build_config
from impl.util.platform_adapter import tbe_context
from impl.util import util_select_op_base
from impl.dynamic.bn_training_update import get_op_support_info as bn_get_op_support_info


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-arguments,too-many-locals,redefined-builtin
def check_supported(x,
                    sum,
                    square_sum,
                    scale,
                    offset,
                    mean,
                    variance,
                    y,
                    mean_out,
                    variance_out,
                    batch_mean,
                    batch_variance,
                    factor,
                    epsilon,
                    before_split_ori_shape=None,
                    before_split_ori_format=None,
                    kernel_name="bn_training_update"):
    """
    check supported
    """
    return True, ""


def get_op_support_info(x,
                        sum,
                        square_sum,
                        scale,
                        offset,
                        mean,
                        variance,
                        y,
                        mean_out,
                        variance_out,
                        batch_mean,
                        batch_variance,
                        factor,
                        epsilon,
                        before_split_ori_shape=None,
                        before_split_ori_format=None,
                        kernel_name="bn_training_update"):
    """
    get_op_support_info
    """
    return bn_get_op_support_info(x,
                                  sum,
                                  square_sum,
                                  scale,
                                  offset,
                                  mean,
                                  variance,
                                  y,
                                  mean_out,
                                  variance_out,
                                  batch_mean,
                                  batch_variance,
                                  factor,
                                  epsilon,
                                  before_split_ori_shape,
                                  before_split_ori_format,
                                  kernel_name)
    

# 'pylint: disable=locally-disabled,too-many-arguments,redefined-builtin
# 'pylint: disable=locally-disabled,invalid-name,too-many-locals,unused-argument
def op_select_format(x,
                     sum,
                     square_sum,
                     scale,
                     offset,
                     mean,
                     variance,
                     y,
                     mean_out,
                     variance_out,
                     batch_mean,
                     batch_variance,
                     factor,
                     epsilon,
                     before_split_ori_shape=None,
                     before_split_ori_format=None,
                     kernel_name="bn_training_update"):
    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype="float16,float,float16,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input1 = util_select_op_base.gen_param(classify="input1",
                                           name="sum",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input2 = util_select_op_base.gen_param(classify="input2",
                                           name="square_sum",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input3 = util_select_op_base.gen_param(classify="input3",
                                           name="scale",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input4 = util_select_op_base.gen_param(classify="input4",
                                           name="offset",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input5 = util_select_op_base.gen_param(classify="input5",
                                           name="mean",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input6 = util_select_op_base.gen_param(classify="input6",
                                           name="variance",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype="float16,float,float16,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output1 = util_select_op_base.gen_param(classify="output1",
                                            name="mean",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output2 = util_select_op_base.gen_param(classify="output2",
                                            name="variance",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output3 = util_select_op_base.gen_param(classify="output3",
                                            name="batch_mean",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output4 = util_select_op_base.gen_param(classify="output4",
                                            name="batch_variance",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    param_list = [input0, input1, input2, input3, input4, input5,
                  input6, output0, output1, output2, output3, output4]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=locally-disabled,too-many-arguments,redefined-builtin
# 'pylint: disable=locally-disabled,invalid-name,too-many-locals,unused-argument
def _check_shape(shape_x, shape_sum, shape_square_sum, shape_scale, shape_offset, shape_mean, shape_variance, format):
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
    shape_mean: list or tuple
        mean's data shape
    shape_variance: list or tuple
        variance's data shape

    Returns
    -------
    None
    """
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_sum, param_name="sum")
    para_check.check_shape(shape_square_sum, param_name="square_sum")
    para_check.check_shape(shape_scale, param_name="scale")
    para_check.check_shape(shape_offset, param_name="offset")
    para_check.check_shape(shape_mean, param_name="mean")
    para_check.check_shape(shape_variance, param_name="variance")

    if len(shape_x) not in (5, 6) or len(shape_sum) not in (5, 6) or len(shape_square_sum) not in (5, 6) or \
            len(shape_scale) not in (5, 6):
        error_reson = "This operator can only support 5D or 6D, but some input's shape length is not 5 or 6"
        error_manager_vector.raise_err_specific_reson("bn_training_update", error_reson)
    if len(shape_offset) not in (5, 6) or len(shape_mean) not in (5, 6) or len(shape_variance) not in (5, 6):
        error_reson = "This operator can only support 5D or 6, but some input's shape length is not 5 or 6"
        error_manager_vector.raise_err_specific_reson("bn_training_update", error_reson)
    dim_c1 = 0
    dim_c0 = 0
    i = 0
    j = 0
    if format == "NC1HWC0":
        dim_c1 = shape_x[1]
        dim_c0 = shape_x[4]
        i = 1
        j = 4
    else:
        dim_c1 = shape_x[2]
        dim_c0 = shape_x[5]
        i = 2
        j = 5
    if shape_sum[i] != dim_c1 or shape_sum[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", "Dimension C of x and sum must be equal")
    if shape_square_sum[i] != dim_c1 or shape_square_sum[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update",
                                                      "Dimension C of x and square_sum must be equal")
    if shape_scale[i] != dim_c1 or shape_scale[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", "Dimension C of x and scale must be equal")
    if shape_offset[i] != dim_c1 or shape_offset[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", "Dimension C of x and offset must be equal")
    if shape_mean[i] != dim_c1 or shape_mean[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", "Dimension C of x and mean must be equal")
    if shape_variance[i] != dim_c1 or shape_variance[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", "Dimension C of x and mean must be equal")


def _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale, dtype_offset, dtype_mean, dtype_variance):
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
    dtype_mean: str
        mean's data type
    dtype_variance: str
        variance's data type

    Returns
    -------
    None
    """
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")
    para_check.check_dtype(dtype_sum.lower(), ("float32",), param_name="sum")
    para_check.check_dtype(dtype_square_sum.lower(), ("float32",), param_name="square_sum")
    para_check.check_dtype(dtype_scale.lower(), ("float32",), param_name="scale")
    para_check.check_dtype(dtype_offset.lower(), ("float32",), param_name="offset")
    para_check.check_dtype(dtype_mean.lower(), ("float32",), param_name="mean")
    para_check.check_dtype(dtype_variance.lower(), ("float32",), param_name="variance")


@tbe_platform.fusion_manager.fusion_manager.register("bn_training_update")
def bn_training_update_compute(x,
                               sum,
                               square_sum,
                               scale,
                               offset,
                               mean,
                               variance,
                               y,
                               mean_out,
                               variance_out,
                               batch_mean,
                               batch_variance,
                               factor,
                               epsilon,
                               before_split_ori_shape=None,
                               before_split_ori_format=None,
                               kernel_name="bn_training_update"):
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
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    mean_out: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
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
    before_split_ori_shape: list_list_int
        ori_shape for input list, only valid in ffts.
    before_split_ori_format: list_int
        ori_format for input list, only valid in ffts.
    kernel_name: str
        kernel name, default value is "bn_training_update"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update compute
    """
    shape_x = shape_util.shape_to_list(x.shape)
    num = shape_x[0] * shape_x[2] * shape_x[3]
    num_rec = 1.0 / num

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

    if num == 1:
        batch_var_scaler = 0.0
    else:
        batch_var_scaler = float(num) / (num - 1)
    batch_variance = tbe.vmuls(save_variance_reduce, batch_var_scaler)

    factor_reverse = 1.0 - factor
    mean_mul = tbe.vmuls(save_mean_reduce, factor)
    mean_mul_rev = tbe.vmuls(mean, factor_reverse)
    mean = tbe.vadd(mean_mul, mean_mul_rev)

    var_mul = tbe.vmuls(batch_variance, factor)
    var_mul_rev = tbe.vmuls(variance, factor_reverse)
    variance = tbe.vadd(var_mul, var_mul_rev)

    res = [res_y, mean, variance, save_mean_reduce, save_variance_reduce]

    return res


def bn_training_update_prebuild(x,
                                sum,
                                square_sum,
                                scale,
                                offset,
                                mean,
                                variance,
                                y,
                                mean_out,
                                variance_out,
                                batch_mean,
                                batch_variance,
                                factor,
                                epsilon,
                                before_split_ori_shape=None,
                                before_split_ori_format=None,
                                kernel_name="bn_training_update"):
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
    mean: dict
        dict of mean, A 5HD Tensor for mean.
    variance: dict
        dict of variance, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    mean_out: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
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
    before_split_ori_shape: list_list_int
        ori_shape for input list, only valid in ffts.
    before_split_ori_format: list_int
        ori_format for input list, only valid in ffts.
    kernel_name: str
        kernel name, default value is "bn_training_update"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")

    dtype_x = x.get("dtype")
    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")
    format = x.get("format")

    _check_shape(shape_x, shape_sum, shape_square_sum, shape_scale, shape_offset, shape_mean, shape_variance, format)
    _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale, dtype_offset, dtype_mean, dtype_variance)
    if format == "NDC1HWC0":
        shape_x = [shape_x[0] * shape_x[1], shape_x[2], shape_x[3], shape_x[4], shape_x[5]]
        shape_sum = [shape_sum[0] * shape_sum[1], shape_sum[2], shape_sum[3], shape_sum[4], shape_sum[5]]
        shape_square_sum = [
            shape_square_sum[0] * shape_square_sum[1], shape_square_sum[2], shape_square_sum[3], shape_square_sum[4],
            shape_square_sum[5]
        ]
        shape_scale = [shape_scale[0] * shape_scale[1], shape_scale[2], shape_scale[3], shape_scale[4], shape_scale[5]]
        shape_offset = [
            shape_offset[0] * shape_offset[1], shape_offset[2], shape_offset[3], shape_offset[4], shape_offset[5]
        ]
        shape_mean = [shape_mean[0] * shape_mean[1], shape_mean[2], shape_mean[3], shape_mean[4], shape_mean[5]]
        shape_variance = [
            shape_variance[0] * shape_variance[1], shape_variance[2], shape_variance[3], shape_variance[4],
            shape_variance[5]
        ]
    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input", dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum, name="square_sum_input", dtype=dtype_square_sum.lower())
    scale_input = tvm.placeholder(shape_scale, name="scale_input", dtype=dtype_scale.lower())
    offset_input = tvm.placeholder(shape_offset, name="offset_input", dtype=dtype_offset.lower())
    mean_input = tvm.placeholder(shape_mean, name="mean_input", dtype=dtype_mean.lower())
    variance_input = tvm.placeholder(shape_variance, name="variance_input", dtype=dtype_variance.lower())

    res = bn_training_update_compute(x_input,
                                     sum_input,
                                     square_sum_input,
                                     scale_input,
                                     offset_input,
                                     mean_input,
                                     variance_input,
                                     y,
                                     mean_out,
                                     variance_out,
                                     batch_mean,
                                     batch_variance,
                                     factor,
                                     epsilon,
                                     before_split_ori_shape,
                                     before_split_ori_format,
                                     kernel_name=kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    tensor_list = [x_input, sum_input, square_sum_input, scale_input, offset_input, mean_input, variance_input
                   ] + list(res)

    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.OPTION_ATTR_LIST_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def bn_training_update(x,
                       sum,
                       square_sum,
                       scale,
                       offset,
                       mean,
                       variance,
                       y,
                       mean_out,
                       variance_out,
                       batch_mean,
                       batch_variance,
                       factor,
                       epsilon,
                       before_split_ori_shape=None,
                       before_split_ori_format=None,
                       kernel_name="bn_training_update"):
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
    mean: dict
        dict of mean, A 5HD Tensor for mean.
    variance: dict
        dict of variance, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    mean_out: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
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
    before_split_ori_shape: list_list_int
        ori_shape for input list, only valid in ffts.
    before_split_ori_format: list_int
        ori_format for input list, only valid in ffts.
    kernel_name: str
        kernel name, default value is "bn_training_update"

    Returns
    -------
    None
    """
    # dynamic and static code. for now, only static
    if get_current_build_config("enable_op_prebuild") or True:
        bn_training_update_prebuild(x,
                                    sum,
                                    square_sum,
                                    scale,
                                    offset,
                                    mean,
                                    variance,
                                    y,
                                    mean_out,
                                    variance_out,
                                    batch_mean,
                                    batch_variance,
                                    factor,
                                    epsilon,
                                    before_split_ori_shape,
                                    before_split_ori_format,
                                    kernel_name)
    else:
        with tbe_context.op_context.OpContext("static"):
            dimpl.bn_training_update(x,
                                     sum,
                                     square_sum,
                                     scale,
                                     offset,
                                     mean,
                                     variance,
                                     y,
                                     mean_out,
                                     variance_out,
                                     batch_mean,
                                     batch_variance,
                                     factor,
                                     epsilon,
                                     before_split_ori_shape,
                                     before_split_ori_format,
                                     kernel_name)
