#!/usr/bin/python
# -*- coding: utf-8 -*-
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
instance_norm
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
def instance_norm_compute(x,
                          gamma,
                          beta,
                          y,
                          mean,
                          variance,
                          data_format="NDHWC",
                          epsilon=1e-6,
                          kernel_name="instance_norm"):
    """
    DSL description of the instancenorm operator's mathematical calculation process

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input x
    gamma: TVM tensor
        the placeholder of input gamma
    beta: TVM tensor
        the placeholder of input beta
    y: dict
        shape and dtype of output y
    mean: dict
        shape and dtype of output mean
    variance: dict
        shape and dtype of output variance
    data_format: str
        A `string` from: `"NDHWC", "NCDHW", "NHWC", "NCHW"`
    epsilon: float
        minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "instance_norm"

    Returns
    -------
    res_tuple: tuple
        (result, result_mean, result_variance)
    """
    shape_x = shape_util.shape_to_list(x.shape)
    dtype_x = x.dtype.lower()
    if dtype_x == "float16":
        x = tbe.cast_to(x, "float32")
        gamma = tbe.cast_to(gamma, "float32")
        beta = tbe.cast_to(beta, "float32")

    axis = []
    if data_format in ("NDHWC",):
        axis = [1, 2, 3]
    elif data_format in ("NCDHW",):
        axis = [2, 3, 4]
    elif data_format in ("NHWC",):
        axis = [1, 2]
    elif data_format in ("NCHW",):
        axis = [2, 3]
    elif data_format in ("ND",):
        axis = list(range(2, len(shape_x)))

    reduce_elts = 1.0
    for i in axis:
        reduce_elts *= shape_x[i]
    mean_cof = reduce_elts**(-1)

    # DSL description of the mean calculation process
    mean_muls = tbe.vmuls(x, mean_cof)
    result_mean = tbe.reduce_sum(mean_muls, axis=axis, keepdims=True)
    mean_broadcast = tbe.broadcast(result_mean, shape_x)
    # DSL description of the variance calculation process
    variance_sub = tbe.vsub(x, mean_broadcast)
    variance_mul = tbe.vmul(variance_sub, variance_sub)
    variance_muls = tbe.vmuls(variance_mul, mean_cof)
    result_variance = tbe.reduce_sum(variance_muls, axis=axis, keepdims=True)
    variance_broadcast = tbe.broadcast(result_variance, shape_x)
    # DSL description of the result calculation process
    epsilon = tvm.const(epsilon, dtype="float32")
    normalize_add = tbe.vadds(variance_broadcast, epsilon)
    normalize_sqrt = tbe.vsqrt(normalize_add, 0)
    tesor_one = tbe.broadcast(tvm.const(1, "float32"), shape_x)
    normalize_rsqrt = tbe.vdiv(tesor_one, normalize_sqrt)
    normalize_sub = tbe.vsub(x, mean_broadcast)
    normalize_mul = tbe.vmul(normalize_sub, normalize_rsqrt)
    gamma_broadcast = tbe.broadcast(gamma, shape_x)
    beta_broadcast = tbe.broadcast(beta, shape_x)
    scale_mul = tbe.vmul(gamma_broadcast, normalize_mul)
    result = tbe.vadd(scale_mul, beta_broadcast)

    if dtype_x == "float16":
        result = tbe.cast_to(result, "float16")
        result_mean = tbe.cast_to(result_mean, "float16")
        result_variance = tbe.cast_to(result_variance, "float16")

    return result, result_mean, result_variance


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def instance_norm(x, gamma, beta, y, mean, variance, data_format="NDHWC", epsilon=1e-6, kernel_name="instance_norm"):
    """
    instancenorm operator interface implementation
    calculating: x, gamma, beta
        mean  = np.mean(x, axis, keepdims=True)
        variance = np.mean(np.power((x - mean),2), axis, keepdims=True)
        result = gamma*((x - mean) / np.sqrt(variance + epsilon)) + beta

    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16, float32
    gamma: dict
        shape and dtype of input gamma, only support float16, float32
    beta: dict
        shape and dtype of input beta, only support float16, float32
    y: dict
        shape and dtype of output y, only support float16, float32
    mean: dict
        shape and dtype of output mean, only support float16, float32
    variance: dict
        shape and dtype of output variance, only support float16, float32
    data_format: str
        A `string` from: `"NDHWC", "NCDHW", "NHWC", "NCHW"`
    epsilon: float
        minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "instance_norm"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    shape_gamma = gamma.get("shape")
    shape_beta = beta.get("shape")
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_gamma, param_name="gamma")
    para_check.check_shape(shape_beta, param_name="beta")
    if len(shape_x) < 2:
        expected_value = "must be greater or equal to 2"
        real_value = str(len(shape_x))
        error_manager_vector.raise_err_input_value_invalid("instacenorm", "input x", expected_value, real_value)

    dtype_x = x.get("dtype").lower()
    dtype_gamma = gamma.get("dtype").lower()
    dtype_beta = beta.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_x, check_list, param_name="x")
    para_check.check_dtype(dtype_gamma, check_list, param_name="gamma")
    para_check.check_dtype(dtype_beta, check_list, param_name="beta")

    format_x = x.get("format")
    if format_x in ("NDHWC",) and len(shape_gamma) == 1 and len(shape_beta) == 1:
        shape_gamma = [1, 1, 1, 1, shape_gamma[0]]
        shape_beta = [1, 1, 1, 1, shape_beta[0]]
    elif format_x in ("NCDHW",) and len(shape_gamma) == 1 and len(shape_beta) == 1:
        shape_gamma = [1, shape_gamma[0], 1, 1, 1]
        shape_beta = [1, shape_beta[0], 1, 1, 1]
    elif format_x in ("NHWC",) and len(shape_gamma) == 1 and len(shape_beta) == 1:
        shape_gamma = [1, 1, 1, shape_gamma[0]]
        shape_beta = [1, 1, 1, shape_beta[0]]
    elif format_x in ("NCHW",) and len(shape_gamma) == 1 and len(shape_beta) == 1:
        shape_gamma = [1, shape_gamma[0], 1, 1]
        shape_beta = [1, shape_beta[0], 1, 1]
    elif format_x in ("ND",) and len(shape_gamma) == 1 and len(shape_beta) == 1:
        shape_gamma = [1, shape_gamma[0]] + [1] * (len(shape_x) - 2)
        shape_beta = [1, shape_beta[0]] + [1] * (len(shape_x) - 2)

    data_x = tvm.placeholder(shape_x, name="x", dtype=dtype_x)
    data_gamma = tvm.placeholder(shape_gamma, name="gamma", dtype=dtype_x)
    data_beta = tvm.placeholder(shape_beta, name="beta", dtype=dtype_x)
    result, result_mean, result_variance = instance_norm_compute(data_x, data_gamma, data_beta, y, mean, variance,
                                                                 format_x, epsilon, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule([result, result_mean, result_variance])

    config = {"name": kernel_name, "tensor_list": [data_x, data_gamma, data_beta, result, result_mean, result_variance]}

    tbe.build(sch, config)
