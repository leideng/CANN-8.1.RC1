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
in_training_reduce_grad
"""

import operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    # minimum positive number greater than 0
    EPSLON = 1e-6


# 'pylint: disable=too-many-locals,too-many-arguments,unused-argument,invalid-name,too-many-function-args
def _check_params(params):
    """
    check parameters
    """
    para_check.check_dtype(params.get("dtype_dy"), ("float16", "float32"), param_name="dy")
    para_check.check_dtype(params.get("dtype_variance"), ("float32",), param_name="variance")

    _check_shape(params)


def _check_shape(params):
    """
    check shape
    """
    if operator.ne(tuple(params.get("shape_dy")), tuple(params.get("shape_x"))):
        error_detail = "shape of dy and x should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("instance_norm_grad", "dy", "x", error_detail)

    if operator.ne(tuple(params.get("shape_var")), tuple(params.get("shape_mean"))):
        error_detail = "shape of variance and mean should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("instance_norm_grad", "variance", "mean", error_detail)

    if operator.ne(tuple(params.get("shape_res_gamma")), tuple(params.get("shape_res_beta"))):
        error_detail = "shape of res_gamma and res_beta should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("instance_norm_grad", "res_gamma", "res_beta",
                                                               error_detail)

    shape_x = params.get("shape_x")
    shape_mean = params.get("shape_mean")
    shape_res_gamma = params.get("shape_res_gamma")
    shape_gamma = params.get("shape_gamma")

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_mean, param_name="mean")
    para_check.check_shape(shape_res_gamma, param_name="res_gamma")
    para_check.check_shape(shape_gamma, param_name="gamma")


def _get_data_gm(params):
    """
    get placeholders
    """
    data_dy = tvm.placeholder(params.get("shape_dy"), name="data_dy", dtype=params.get("dtype_dy"))
    data_x = tvm.placeholder(params.get("shape_x"), name="data_x", dtype=params.get("dtype_dy"))
    data_variance = tvm.placeholder(params.get("shape_var"), name="data_variance", dtype=params.get("dtype_variance"))
    data_mean = tvm.placeholder(params.get("shape_mean"), name="data_mean", dtype=params.get("dtype_variance"))
    data_res_gamma = tvm.placeholder(params.get("shape_res_gamma"),
                                     name="data_res_gamma",
                                     dtype=params.get("dtype_variance"))
    data_res_beta = tvm.placeholder(params.get("shape_res_beta"),
                                    name="data_res_beta",
                                    dtype=params.get("dtype_variance"))
    data_gamma = tvm.placeholder(params.get("shape_gamma"), name="data_gamma", dtype=params.get("dtype_variance"))

    data_gm = (data_dy, data_x, data_variance, data_mean, data_res_gamma, data_res_beta, data_gamma)

    return data_gm


def in_training_reduce_grad_compute(dy,
                                    x,
                                    variance,
                                    mean,
                                    res_gamma,
                                    res_beta,
                                    gamma,
                                    pd_x,
                                    format_dy,
                                    kernel_name="in_training_reduce_grad"):
    """
    DSL description of the layernorm_grad operator's mathematical

    Parameters
    ----------
    dy: TVM tensor
        the placeholder of input dy
    x: TVM tensor
        the placeholder of input x
    variance: TVM tensor
        the placeholder of input variance
    mean: TVM tensor
        the placeholder of input mean
    res_gamma: TVM tensor
        the placeholder of input res_gamma
    res_beta: TVM tensor
        the placeholder of input res_beta
    gamma: TVM tensor
        the placeholder of input gamma
    pd_x: dict
        shape and dtype of output pd_x
    kernel_name: str
        cce kernel name, default value is "in_training_reduce_grad"

    Returns
    -------
    res_list: list
        [res]
    """
    shape_dy = shape_util.shape_to_list(dy.shape)
    shape_var = shape_util.shape_to_list(variance.shape)
    dtye_dy = dy.dtype.lower()

    reduce_axis = []
    if format_dy == "NDC1HWC0":  # only support NDC1HWC0 and NC1HWC0
        reduce_axis = [1, 3]
    else:
        reduce_axis = [2, 3]

    num = 1.0
    for i in reduce_axis:
        num *= shape_dy[i]
    num_rec = 1.0 / num

    if dtye_dy == "float16":
        dy = tbe.cast_to(dy, "float32")
        x = tbe.cast_to(x, "float32")

    data_sqrt = tbe.vsqrt(tbe.vadds(variance, Constant.EPSLON))
    scale_inv = tbe.vmuls(res_gamma, num_rec)
    scale_inv_reverse = tbe.vmuls(res_gamma, (-1.0) * num_rec)
    offset_inv_reverse = tbe.vmuls(res_beta, (-1.0) * num_rec)

    multiplier = tbe.vdiv(scale_inv_reverse, data_sqrt)
    addend_div = tbe.vdiv(mean, data_sqrt)
    addend_mul = tbe.vmul(addend_div, scale_inv)
    addend = tbe.vadd(addend_mul, offset_inv_reverse)

    multiplier_broadcast = tbe.broadcast(multiplier, shape_dy)
    addend_broadcast = tbe.broadcast(addend, shape_dy)

    coef_mul = tbe.vmul(multiplier_broadcast, x)
    coef_add = tbe.vadd(dy, coef_mul)
    coef = tbe.vadd(coef_add, addend_broadcast)

    gamma_broadcast = tbe.broadcast(gamma, shape_var)
    mul_scale = tbe.vdiv(gamma_broadcast, data_sqrt)
    mul_scale_broadcast = tbe.broadcast(mul_scale, shape_dy)

    res = tbe.vmul(coef, mul_scale_broadcast)

    if dtye_dy == "float16":
        res = tbe.cast_to(res, "float16")
    return [res]


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def in_training_reduce_grad(dy,
                            x,
                            variance,
                            mean,
                            res_gamma,
                            res_beta,
                            gamma,
                            pd_x,
                            kernel_name="in_training_reduce_grad"):
    """
    in_training_reduce_grad operator interface implementation

    Parameters
    ----------
    dy: dict
        shape and dtype of input dy, only support float16, float32
    x: dict
        shape and dtype of input x, only support float16, float32
    variance: dict
        shape and dtype of input variance, only support float32
    mean: dict
        shape and dtype of input mean, only support float32
    res_gamma: dict
        shape and dtype of input res_gamma, only support float32
    res_beta: dict
        shape and dtype of input res_beta, only support float32
    gamma: dict
        shape and dtype of input gamma, only support float32
    pd_x: dict
        shape and dtype of output pd_x, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "in_training_reduce_grad"

    Returns
    -------
    None
    """
    shape_dy = dy.get("shape")
    shape_x = x.get("shape")
    shape_variance = variance.get("shape")
    shape_mean = mean.get("shape")
    shape_res_gamma = res_gamma.get("shape")
    shape_res_beta = res_beta.get("shape")
    shape_gamma = gamma.get("shape")
    dtype_dy = dy.get("dtype").lower()
    dtype_variance = variance.get("dtype").lower()
    format_dy = dy.get("format")

    _check_params({
        "shape_dy": shape_dy,
        "shape_x": shape_x,
        "shape_var": shape_variance,
        "shape_mean": shape_mean,
        "shape_res_gamma": shape_res_gamma,
        "shape_res_beta": shape_res_beta,
        "shape_gamma": shape_gamma,
        "dtype_dy": dtype_dy,
        "dtype_variance": dtype_variance,
    })

    if format_dy in ("NDC1HWC0",):
        shape_dy = [shape_dy[0], shape_dy[1], shape_dy[2], shape_dy[3] * shape_dy[4], shape_dy[5]]
        shape_x = [shape_x[0], shape_x[1], shape_x[2], shape_x[3] * shape_x[4], shape_x[5]]
        shape_variance = [
            shape_variance[0], shape_variance[1], shape_variance[2], shape_variance[3] * shape_variance[4],
            shape_variance[5]
        ]
        shape_mean = [shape_mean[0], shape_mean[1], shape_mean[2], shape_mean[3] * shape_mean[4], shape_mean[5]]
        shape_res_gamma = [
            shape_res_gamma[0], shape_res_gamma[1], shape_res_gamma[2], shape_res_gamma[3] * shape_res_gamma[4],
            shape_res_gamma[5]
        ]
        shape_res_beta = [
            shape_res_beta[0], shape_res_beta[1], shape_res_beta[2], shape_res_beta[3] * shape_res_beta[4],
            shape_res_beta[5]
        ]
        shape_gamma = [shape_gamma[0], shape_gamma[1], shape_gamma[2], shape_gamma[3] * shape_gamma[4], shape_gamma[5]]

    data_gm = _get_data_gm({
        "shape_dy": shape_dy,
        "shape_x": shape_x,
        "shape_var": shape_variance,
        "shape_mean": shape_mean,
        "shape_res_gamma": shape_res_gamma,
        "shape_res_beta": shape_res_beta,
        "shape_gamma": shape_gamma,
        "dtype_dy": dtype_dy,
        "dtype_variance": dtype_variance,
    })

    res = in_training_reduce_grad_compute(data_gm[0], data_gm[1], data_gm[2], data_gm[3], data_gm[4], data_gm[5],
                                          data_gm[6], pd_x, format_dy, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": list(data_gm) + list(res)}

    tbe.build(sch, config)
