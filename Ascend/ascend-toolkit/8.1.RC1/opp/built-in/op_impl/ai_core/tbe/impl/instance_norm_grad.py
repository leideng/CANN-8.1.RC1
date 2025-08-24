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
instance_norm_grad
"""

import operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-many-locals,too-many-arguments,unused-argument,invalid-name
def _check_params(params):
    """
    check parameters including shape_dy, shape_x, shape_var, shape_mean, shape_gamma, dtype
    """
    check_list = ("float16", "float32")
    para_check.check_dtype(params.get("dtype"), check_list, param_name="x")

    _check_shape(params)


def _check_shape(params):
    """
    check parameters including shape_dy, shape_x, shape_var, shape_mean and shape_gamma
    """
    if operator.ne(tuple(params.get("shape_dy")), tuple(params.get("shape_x"))):
        error_detail = "shape of dy and x should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("instance_norm_grad", "dy", "x", error_detail)

    if operator.ne(tuple(params.get("shape_var")), tuple(params.get("shape_mean"))):
        error_detail = "shape of variance and mean should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("instance_norm_grad", "variance", "mean", error_detail)

    shape_x = params.get("shape_x")
    shape_mean = params.get("shape_mean")
    shape_gamma = params.get("shape_gamma")

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_mean, param_name="mean")
    para_check.check_shape(shape_gamma, param_name="gamma")


def _get_data_gm(shapes, dtype):
    """
    get placeholders of data_dy, data_x, data_variance, data_mean and data_gamma
    """
    data_dy = tvm.placeholder(shapes.get("shape_dy"), name="data_dy", dtype=dtype)
    data_x = tvm.placeholder(shapes.get("shape_x"), name="data_x", dtype=dtype)
    data_variance = tvm.placeholder(shapes.get("shape_var"), name="data_variance", dtype=dtype)
    data_mean = tvm.placeholder(shapes.get("shape_mean"), name="data_mean", dtype=dtype)
    data_gamma = tvm.placeholder(shapes.get("shape_gamma"), name="data_gamma", dtype=dtype)

    data_gm = (data_dy, data_x, data_variance, data_mean, data_gamma)

    return data_gm


def _get_params(shape_x, format_x):
    """
    compute parameters including param_axis, reduce_axis and mean_num
    """
    param_axis = []
    reduce_axis = []

    if format_x == "NDHWC":
        param_axis = [0, 1, 2, 3]
        reduce_axis = [1, 2, 3]

    mean_num = 1.0
    for i in reduce_axis:
        mean_num *= shape_x[i]

    params = {"param_axis": param_axis, "reduce_axis": reduce_axis, "mean_num": mean_num}

    return params


def _get_pd_xl(data, shape_x):
    """
    compute pd_xl according to data_dy, data_gamma and shape_x
    """
    data_gamma_broadcast = tbe.broadcast(data.get("data_gamma"), shape_x)
    pd_xl = tbe.vmul(data_gamma_broadcast, data.get("data_dy"))

    return pd_xl


def _get_pd_var_front(data):
    """
    compute front part of pd_var according to data_variance
    """
    shape_var = shape_util.shape_to_list(data.get("data_variance").shape)
    default_instance_norm_grad_epsilon = 1e-6
    var_elta = tbe.vadds(data.get("data_variance"), tvm.const(default_instance_norm_grad_epsilon, dtype="float32"))
    var_elta_sqrt = tbe.vsqrt(var_elta)
    tesor_one = tbe.broadcast(tvm.const(1, "float32"), shape_var)
    var_elta_rsqrt = tbe.vdiv(tesor_one, var_elta_sqrt)

    pdvar1_mul = tbe.vmul(var_elta_rsqrt, var_elta_rsqrt)
    pd_var_1 = tbe.vmul(pdvar1_mul, var_elta_rsqrt)

    return pd_var_1, var_elta_rsqrt


def _get_pd_var(data, params, shape_x, pd_xl):
    """
    compute pd_var according to data_x, data_mean, reduce_axis and pd_xl
    """
    pd_var_1, var_elta_rsqrt = _get_pd_var_front(data)

    data_mean_broadcast = tbe.broadcast(data.get("data_mean"), shape_x)
    sub_x_mean = tbe.vsub(data.get("data_x"), data_mean_broadcast)

    pdvar_mul1 = tbe.vmul(pd_xl, sub_x_mean)
    pdvar_sum = tbe.reduce_sum(pdvar_mul1, params.get("reduce_axis"), keepdims=True)
    pdvar_mul3 = tbe.vmul(pdvar_sum, pd_var_1)
    pd_var = tbe.vmuls(pdvar_mul3, tvm.const(-0.5, dtype="float32"))

    return pd_var, var_elta_rsqrt, sub_x_mean


def _get_pd_mean(params, pd_xl, pd_var, var_elta_rsqrt, sub_x_mean):
    """
    compute pd_mean according to reduce_axis, pd_xl, pd_var, var_elta_rsqrt and sub_x_mean
    """
    pdmean1_sum = tbe.reduce_sum(pd_xl, params.get("reduce_axis"), keepdims=True)
    pdmean1_mul = tbe.vmul(pdmean1_sum, var_elta_rsqrt)
    pd_mean = tbe.vmuls(pdmean1_mul, tvm.const(-1.0, dtype="float32"))

    return pd_mean


def _get_pd_x_front(data, params, shape_x):
    """
    compute front part of pd_x according to data, params and shape_x
    """
    pd_xl = _get_pd_xl(data, shape_x)

    pd_var, var_elta_rsqrt, sub_x_mean = _get_pd_var(data, params, shape_x, pd_xl)

    pd_mean = _get_pd_mean(params, pd_xl, pd_var, var_elta_rsqrt, sub_x_mean)

    var_elta_rsqrt = tbe.broadcast(var_elta_rsqrt, shape_x)
    pd_x_1 = tbe.vmul(var_elta_rsqrt, pd_xl)
    pdx2_broadcast = tbe.broadcast(pd_var, shape_x)
    pdx2_mul = tbe.vmul(pdx2_broadcast, sub_x_mean)
    pd_x_2 = tbe.vmuls(pdx2_mul, tvm.const((2 * (params.get("mean_num")**(-1))), dtype="float32"))
    pd_x_3 = tbe.vmuls(pd_mean, tvm.const((params.get("mean_num")**(-1)), dtype="float32"))

    return [pd_x_1, pd_x_2, pd_x_3, var_elta_rsqrt, sub_x_mean]


def _get_pd_x(data, params, shape_x, dtype):
    """
    compute pd_x according to data, params and shape_x
    """
    pd_x_1, pd_x_2, pd_x_3, var_elta_rsqrt, sub_x_mean = _get_pd_x_front(data, params, shape_x)

    pdx_broadcast = tbe.broadcast(pd_x_3, shape_x)
    pdx_add = tbe.vadd(pd_x_1, pd_x_2)
    pd_x = tbe.vadd(pdx_add, pdx_broadcast)
    if dtype == "float16":
        pd_x = tbe.cast_to(pd_x, dtype)

    return pd_x, var_elta_rsqrt, sub_x_mean


def _get_pd_gamma(data, params, var_elta_rsqrt, sub_x_mean, dtype):
    """
    compute pd_gamma according to data, params, var_elta_rsqrt and sub_x_mean
    """
    xl_mul = tbe.vmul(var_elta_rsqrt, sub_x_mean)
    pdga_mul = tbe.vmul(data.get("data_dy"), xl_mul)

    pd_gamma = tbe.reduce_sum(pdga_mul, params.get("param_axis"), keepdims=True)

    if dtype == "float16":
        pd_gamma = tbe.cast_to(pd_gamma, dtype)

    return pd_gamma


def _get_pd_beta(data, params, dtype):
    """
    compute pd_beta according to data and params
    """
    pd_beta = tbe.reduce_sum(data.get("data_dy"), params.get("param_axis"), keepdims=True)

    if dtype == "float16":
        pd_beta = tbe.cast_to(pd_beta, dtype)

    return pd_beta


def _get_res(data, params, shape_x, dtype):
    """
    compute pd_x, pd_gamma, pd_beta according to data, params and shape_x
    """
    pd_x, var_elta_rsqrt, sub_x_mean = _get_pd_x(data, params, shape_x, dtype)

    pd_gamma = _get_pd_gamma(data, params, var_elta_rsqrt, sub_x_mean, dtype)

    pd_beta = _get_pd_beta(data, params, dtype)

    return pd_x, pd_gamma, pd_beta


def _get_pds(data_dy, data_x, data_variance, data_mean, data_gamma, format_x):
    """
    get params and data, compute pd_x, pd_gamma, pd_beta.
    """
    shape_x = shape_util.shape_to_list(data_x.shape)
    dtype_x = data_x.dtype.lower()

    params = _get_params(shape_x, format_x)

    if dtype_x == "float16":
        data_dy = tbe.cast_to(data_dy, "float32")
        data_x = tbe.cast_to(data_x, "float32")
        data_variance = tbe.cast_to(data_variance, "float32")
        data_mean = tbe.cast_to(data_mean, "float32")
        data_gamma = tbe.cast_to(data_gamma, "float32")

    data = {
        "data_dy": data_dy,
        "data_x": data_x,
        "data_variance": data_variance,
        "data_mean": data_mean,
        "data_gamma": data_gamma
    }

    pd_x, pd_gamma, pd_beta = _get_res(data, params, shape_x, dtype_x)

    return pd_x, pd_gamma, pd_beta


def instance_norm_grad_compute(dy,
                               x,
                               variance,
                               mean,
                               gamma,
                               pd_x,
                               pd_gamma,
                               pd_beta,
                               format_x,
                               kernel_name="instance_norm_grad"):
    """
    DSL description of the layernorm_grad operator's mathematical

    Parameters
    ----------
    dy: dict
        shape and dtype of input dy, only support float16, float32
    x: dict
        shape and dtype of input x, only support float16, float32
    variance: dict
        shape and dtype of input variance, only support float16, float32
    mean: dict
        shape and dtype of input mean, only support float16, float32
    gamma: dict
        shape and dtype of input gamma, only support float16, float32
    pd_x: dict
        shape and dtype of output pd_x, only support float16, float32
    pd_gamma: dict
        shape and dtype of output pd_gamma, only support float16, float32
    pd_beta: dict
        shape and dtype of output pd_beta, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "instance_norm_grad"

    Returns
    -------
    res_tuple: tuple
        (pd_x, pd_gamma, pd_beta)
    """
    pd_x, pd_gamma, pd_beta = _get_pds(dy, x, variance, mean, gamma, format_x)
    res_list = [pd_x, pd_gamma, pd_beta]

    return res_list


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def instance_norm_grad(dy, x, variance, mean, gamma, pd_x, pd_gamma, pd_beta, kernel_name="instance_norm_grad"):
    """
    instancenormgrad operator interface implementation
    calculating: dy, x, variance, mean, gamma
        pd_xl    = dy*gamma
        pd_var   = np.sum(((-0.5)*pd_xl*(x - mean)
                   *np.power((variance + epsilon), (-1.5))),
                   reduce_axis, keepdims=True)
        pd_mean  = np.sum(((-1.0)*pd_xl
                   *np.power((variance + epsilon), (-0.5))),
                   reduce_axis, keepdims=True)
        pd_x     = pd_xl*np.power((variance + epsilon), (-0.5))
                   + pd_var*(2.0/m)*(x - mean) + pd_mean*(1.0/m)
        pd_gamma = np.sum((dy*(x - mean)
                   *np.power((variance + epsilon), (-0.5))), parms_axis,
                   keepdims=True)
        pd_beta  = np.sum(dy, parms_axis, keepdims=True)

    Parameters
    ----------
    dy: dict
        shape and dtype of input dy, only support float16, float32
    x: dict
        shape and dtype of input x, only support float16, float32
    variance: dict
        shape and dtype of input variance, only support float16, float32
    mean: dict
        shape and dtype of input mean, only support float16, float32
    gamma: dict
        shape and dtype of input gamma, only support float16, float32
    pd_x: dict
        shape and dtype of output pd_x, only support float16, float32
    pd_gamma: dict
        shape and dtype of output pd_gamma, only support float16, float32
    pd_beta: dict
        shape and dtype of output pd_beta, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "instance_norm_grad"

    Returns
    -------
    None
    """
    shape_dy = dy.get("shape")
    shape_x = x.get("shape")
    shape_variance = variance.get("shape")
    shape_mean = mean.get("shape")
    shape_gamma = gamma.get("shape")
    dtype_x = x.get("dtype").lower()

    _check_params({
        "shape_dy": shape_dy,
        "shape_x": shape_x,
        "shape_var": shape_variance,
        "shape_mean": shape_mean,
        "shape_gamma": shape_gamma,
        "dtype": dtype_x,
        "kernel_name": kernel_name
    })

    format_x = x.get("format")
    if format_x in ("NDHWC",) and len(shape_gamma) == 1:
        shape_gamma = (1, 1, 1, 1, shape_gamma[0])

    data_gm = _get_data_gm(
        {
            "shape_dy": shape_dy,
            "shape_x": shape_x,
            "shape_var": shape_variance,
            "shape_mean": shape_mean,
            "shape_gamma": shape_gamma
        }, dtype_x)

    res = instance_norm_grad_compute(data_gm[0], data_gm[1], data_gm[2], data_gm[3], data_gm[4], pd_x, pd_gamma,
                                     pd_beta, format_x, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": list(data_gm) + list(res)}

    tbe.build(sch, config)
