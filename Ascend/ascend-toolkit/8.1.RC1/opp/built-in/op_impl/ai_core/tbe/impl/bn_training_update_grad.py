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
bn_training_update_grad
"""
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_select_op_base
from impl.dynamic.bn_training_update_grad import get_op_support_info as bn_get_op_support_info


# 'pylint: disable = unused-argument
# 'pylint: disable=invalid-name,too-many-arguments,consider-using-in
def check_supported(grads,
                    x,
                    batch_mean,
                    batch_variance,
                    diff_scale,
                    diff_offset,
                    epsilon,
                    kernel_name="bn_training_update_grad"):
    """
    check supported
    """
    return True, ""


def get_op_support_info(grads,
                        x,
                        batch_mean,
                        batch_variance,
                        diff_scale,
                        diff_offset,
                        epsilon,
                        kernel_name="bn_training_update_grad"):
    """
    get_op_support_info
    """
    return bn_get_op_support_info(grads,
                                  x,
                                  batch_mean,
                                  batch_variance,
                                  diff_scale,
                                  diff_offset,
                                  epsilon,
                                  kernel_name)


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals
def op_select_format(grads,
                     x,
                     batch_mean,
                     batch_variance,
                     diff_scale,
                     diff_offset,
                     epsilon,
                     kernel_name="bn_training_update_grad"):
    """
    1. when input(grads)'s ori_shape is [1, ? ,1, ?] and the format is NCHW
    the Op BNTrainingUpdateGrad can support NCHW.
    > for example:
    > grads : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > x : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > batch_mean : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > batch_variance : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > the Op BNTrainingUpdateGrad can process with NC1HWC0:
    > grads : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > batch_mean : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > batch_variance : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    """
    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="grads",
                                           datatype="float16,float,float16,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input1 = util_select_op_base.gen_param(classify="input1",
                                           name="x",
                                           datatype="float16,float,float16,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input2 = util_select_op_base.gen_param(classify="input2",
                                           name="batch_mean",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input3 = util_select_op_base.gen_param(classify="input3",
                                           name="batch_variance",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="diff_scale",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output1 = util_select_op_base.gen_param(classify="output1",
                                            name="diff_offset",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")

    param_list = [input0, input1, input2, input3, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _check_format_nd(data_format, origin_foramt):
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
    if data_format.upper() not in ("NC1HWC0", "NDC1HWC0", "NCHW"):
        error_reson = "The data format only supports NC1HWC0 and NCHW and NDC1HWC0."
        error_manager_vector.raise_err_specific_reson("bn_training_update_grad", error_reson)
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            error_reson = "The origin format only supports NCHW when format is NCHW"
            error_manager_vector.raise_err_specific_reson("bn_training_update_grad", error_reson)


@register_operator_compute("bn_training_update_grad", op_mode="static", support_fusion=True)
def bn_training_update_grad_compute(grads, x, batch_mean, batch_variance,
                                    diff_scale, diff_offset, epsilon,
                                    kernel_name="bn_training_update_grad"):
    """
    Compute for bn_training_update_grad_compute
    x_norm:(x-input_reserve_space_1)*
            np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_scale:np.sum(y*(x-input_reserve_space_1)*
                         np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_offset: np.sum(y)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads. Must be one of the following
        type: `float16`, `float32`.
    x: TVM tensor 5D
        the placeholder of x. Must be one of the following
        type: `float16`, `float32`.
    batch_mean: TVM tensor 5D
        the placeholder of batch_mean. Must be one of the following
        type: `float32`.
    batch_variance: TVM tensor 5D
        the placeholder of batch_variance. Must be one of the following
        type: `float32`.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for output diff_scale.
    diff_offset: dict
        dict of diff_offset, A 5D Tensor for output diff_offset.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_training_update_grad"

    Returns
    -------
    res_list: list
       [diff_scale, diff_offset].
    """
    shape_x = shape_util.shape_to_list(x.shape)
    axis = [0, 2, 3]

    if grads.dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        grads = tbe.cast_to(grads, "float32")
    if x.dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        x = tbe.cast_to(x, "float32")
    batch_mean_inverse = tbe.vmuls(batch_mean, tvm.const(-1, dtype=batch_mean.dtype))
    input_mean = tbe.broadcast(batch_mean_inverse, shape_x)
    x_sub = tbe.vadd(x, input_mean)

    data_adds = tbe.vadds(batch_variance, epsilon)
    data_rsqrt = tbe.vsqrt(data_adds)
    shape_var = shape_util.shape_to_list(batch_variance.shape)
    scalar_one = 1
    data_cast = tbe.broadcast(tvm.const(scalar_one, "float32"), shape_var)
    data_rsqrts = tbe.vdiv(data_cast, data_rsqrt)
    rsqrts_broadcast = tbe.broadcast(data_rsqrts, shape_x)
    x_norm = tbe.vmul(x_sub, rsqrts_broadcast)

    scale_mul = tbe.vmul(grads, x_norm)

    diff_scale, diff_offset = tbe.tuple_sum([scale_mul, grads], axis, True)
    res_list = [diff_scale, diff_offset]
    return res_list


def _check_shape(shape_grads, shape_x, shape_batch_mean, shape_batch_variance, data_format):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_grads: list or tuple
        input grads's data shape
    shape_x: list or tuple
        input x's data shape
    shape_batch_mean: list or tuple
        input batch_mean's data shape
    shape_batch_variance: list or tuple
        input batch_variance's data shape
    Returns
    -------
    None
    """
    para_check.check_shape(shape_grads, param_name="grads")
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_batch_mean, param_name="batch_mean")
    para_check.check_shape(shape_batch_variance, param_name="batch_variance")
    dim_c1 = 0
    dim_c0 = 0
    n = 0
    h = 0
    w = 0
    c1 = 0
    c0 = 0
    if data_format == "NC1HWC0":
        dim_c1 = shape_grads[1]
        dim_c0 = shape_grads[4]
        n = shape_batch_mean[0]
        h = shape_batch_mean[2]
        w = shape_batch_mean[3]
        c1 = shape_batch_mean[1]
        c0 = shape_batch_mean[4]
    else:
        dim_c1 = shape_grads[2]
        dim_c0 = shape_grads[5]
        n = shape_batch_mean[0] * shape_batch_mean[1]
        h = shape_batch_mean[3]
        w = shape_batch_mean[4]
        c1 = shape_batch_mean[2]
        c0 = shape_batch_mean[5]

    if len(shape_grads) not in (5, 6) or len(shape_x) not in (5, 6):
        error_manager_vector.raise_err_specific_reson("bn_training_update_grad", "This operator can only"
                                                                                 "support 5D and 6D")
    if dim_c0 != 16:
        error_manager_vector.raise_err_specific_reson("bn_training_update_grad", "shape_grads last dim must be 16")
    if len(shape_batch_mean) not in (5, 6) or len(shape_batch_variance) not in (5, 6):
        error_manager_vector.raise_err_specific_reson("bn_training_update_grad",
                                                      "This operator can only support 5D and 6D")
    if n != 1 or h != 1 or w != 1:
        error_manager_vector.raise_err_specific_reson("bn_training_update_grad",
                                                      "Dimensions except Dimension C must be one for shape_batch_mean")
    if c1 != dim_c1 or c0 != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update_grad", "Dimension C must be equal")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_training_update_grad(grads, x, batch_mean, batch_variance,
                            diff_scale, diff_offset, epsilon=0.0001,
                            kernel_name="bn_training_update_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_training_update_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
    x: dict
        dict of x, A 5D Tensor for input x.
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for output diff_scale.
    diff_offset: dict
        dict of diff_offset, A 5D Tensor for output diff_offset.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_training_update_grad"

    Returns
    -------
    None
    """

    shape_grads = grads.get("shape")
    shape_x = x.get("shape")
    shape_batch_mean = batch_mean.get("shape")
    shape_batch_variance = batch_variance.get("shape")

    dtype_grads = grads.get("dtype")
    dtype_x = x.get("dtype")
    dtype_batch_mean = batch_mean.get("dtype")
    dtype_batch_variance = batch_variance.get("dtype")

    input_grads_dtype = dtype_grads.lower()
    input_x_dtype = dtype_x.lower()
    batch_mean_dtype = dtype_batch_mean.lower()
    batch_variance_dtype = dtype_batch_variance.lower()

    para_check.check_dtype(input_grads_dtype, ("float32", "float16"), param_name="grads")
    para_check.check_dtype(input_x_dtype, ("float32", "float16"), param_name="x")
    para_check.check_dtype(batch_mean_dtype, ("float32",), param_name="batch_mean")
    para_check.check_dtype(batch_variance_dtype, ("float32",), param_name="batch_variance")
    shape_util.compare_tensor_dict_key(grads, x, "dtype")

    data_format = grads.get("format")
    ori_format = grads.get("ori_format")
    _check_format_nd(data_format, ori_format)

    if data_format in ("NC1HWC0", "NDC1HWC0"):
        _check_shape(shape_grads, shape_x,
                     shape_batch_mean, shape_batch_variance, data_format)
    else:
        shape_list = [1, 1, 1, 1]
        shape_list[1] = shape_x[1]
        shape_batch_mean = shape_list
        shape_batch_variance = shape_list

    shape_util.compare_tensor_dict_key(grads, x, "shape")
    shape_util.compare_tensor_dict_key(batch_mean, batch_variance, "shape")
    if data_format == "NDC1HWC0":
        shape_grads = [shape_grads[0] * shape_grads[1], shape_grads[2], shape_grads[3], shape_grads[4], shape_grads[5]]
        shape_batch_mean = [shape_batch_mean[0] * shape_batch_mean[1], shape_batch_mean[2],
                            shape_batch_mean[3], shape_batch_mean[4], shape_batch_mean[5]]
    grads_input = tvm.placeholder(shape_grads, name="grads_input",
                                  dtype=input_grads_dtype)
    x_input = tvm.placeholder(shape_grads, name="x_input", dtype=input_x_dtype)
    batch_mean_input = tvm.placeholder(shape_batch_mean,
                                       name="batch_mean_input",
                                       dtype=batch_mean_dtype)
    batch_variance_input = tvm.placeholder(shape_batch_mean,
                                           name="batch_variance_input",
                                           dtype=batch_variance_dtype)

    res_list = bn_training_update_grad_compute(grads_input, x_input,
                                               batch_mean_input,
                                               batch_variance_input, diff_scale,
                                               diff_offset,
                                               epsilon, kernel_name=kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res_list)
    tensor_list = [grads_input, x_input, batch_mean_input,
                   batch_variance_input] + list(res_list)
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    build(sch, config)
