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
layer_norm_grad
"""
import operator

from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.constant_util import EPSLON
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class LayerNormGradAxis:
    """
    LayerNormGradAxis param
    """
    # reduce axis
    PARAMS_AXIS = []


# 'pylint: disable=too-many-locals,too-many-arguments
# 'pylint: disable=locally-disabled,unused-argument
def _check_params(params_map):
    """
    check parameters including shape_dy, shape_x, shape_var,
    shape_mean, shape_gamma, dtype and kernel_name

    Parameters
    ----------
    params_map: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
        "shape_mean": shape_mean, "shape_gamma": shape_gamma,
        "dtype": dtype, "kernel_name": kernel_name}

    Returns
    -------
    None
    """

    check_list = ("float16", "float32")
    para_check.check_dtype(params_map.get("dtype"), check_list, param_name="input_dy")

    _check_shape(params_map)


def _check_shape(params_map):
    """
    check parameters including shape_dy, shape_x, shape_var,
    shape_mean and shape_gamma

    Parameters
    ----------
    params_map: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
         "shape_mean": shape_mean, "shape_gamma": shape_gamma,
         "dtype": dtype, "kernel_name": kernel_name}

    Returns
    -------
    None
    """
    if operator.ne(tuple(params_map.get("shape_dy")),
                   tuple(params_map.get("shape_x"))):
        error_detail = "shape of input_dy and input_x should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_grad", "input_dy",
                                                               "input_x", error_detail)

    if operator.ne(tuple(params_map.get("shape_var")),
                   tuple(params_map.get("shape_mean"))):
        error_detail = "shape of input_variance and input_mean should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_grad", "input_variance",
                                                               "input_mean", error_detail)

    shape_x = params_map.get("shape_x")
    shape_mean = params_map.get("shape_mean")
    shape_gamma = params_map.get("shape_gamma")

    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_mean, param_name="input_mean")
    para_check.check_shape(shape_gamma, param_name="input_gamma")

    _check_shape_mean(shape_x, shape_mean)
    _check_shape_gamma(shape_x, shape_gamma)


def _check_shape_mean(shape_x, shape_mean):
    """
    check if parameter shape_mean meets the requirements of function

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_mean: list or tuple
        shape of mean

    Returns
    -------
    None
    """
    if len(shape_x) != len(shape_mean):
        error_detail = "length of shape_x and shape_mean should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_grad", "input_x",
                                                               "input_mean", error_detail)

    if shape_mean[-1] != 1:
        error_detail = "value of shape_mean's last dim must be 1"
        error_manager_vector.raise_err_input_shape_invalid("layer_norm_grad", "input_mean", error_detail)

    flag = -1
    for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
        if xtem != mean:
            flag = i
            break

    if flag != -1:
        for i, mean in enumerate(shape_mean):
            if i < flag:
                continue
            if mean != 1:
                error_detail = "value of shape_mean must be 1"
                error_manager_vector.raise_err_input_shape_invalid("layer_norm_grad", "input_mean", error_detail)


def _check_shape_gamma(shape_x, shape_gamma):
    """
    check if parameter shape_gamma meets the requirements of function

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    None
    """
    if len(shape_gamma) > len(shape_x):
        error_detail = "length of shape_gamma can not be longer than shape_x"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_grad", "input_gamma",
                                                               "input_x", error_detail)

    for xtem, gamma in zip(reversed(shape_x), reversed(shape_gamma)):
        if xtem != gamma:
            error_detail = "value of shape_gamma is wrong"
            error_manager_vector.raise_err_input_shape_invalid("layer_norm_grad", "input_gamma", error_detail)


def _update_gamma_shape(shape_x, shape_gamma):
    """
    update shape_gamma for subsequent calculation

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    shape_gamma_new: tuple
        new shape_gamma after update
    params_axis: tuple
        the list of axis for gamma reduce_sum
    """
    params_axis_tmp = []
    if len(shape_x) != len(shape_gamma):
        sub = len(shape_x) - len(shape_gamma)
        shape_gamma = list(shape_gamma)
        for i in range(sub):
            shape_gamma.insert(0, 1)
            params_axis_tmp.append(i)

    shape_gamma_new = tuple(shape_gamma)
    params_axis = tuple(params_axis_tmp)

    return shape_gamma_new, params_axis


def _get_data_gm(shapes, dtype):
    """
    get placeholders of data_dy, data_x, data_variance, data_mean and data_gamma

    Parameters
    ----------
    shapes: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
         "shape_mean": shape_mean, "shape_gamma": shape_gamma}
    dtype: str
        the data type

    Returns
    -------
    data_gm: tuple
        (data_dy, data_x, data_variance, data_mean, data_gamma)
    """
    data_dy = tvm.placeholder(shapes.get("shape_dy"),
                              name="data_dy", dtype=dtype)
    data_x = tvm.placeholder(shapes.get("shape_x"),
                             name="data_x", dtype=dtype)
    data_variance = tvm.placeholder(shapes.get("shape_var"),
                                    name="data_variance", dtype=dtype)
    data_mean = tvm.placeholder(shapes.get("shape_mean"),
                                name="data_mean", dtype=dtype)
    data_gamma = tvm.placeholder(shapes.get("shape_gamma"),
                                 name="data_gamma", dtype=dtype)

    data_gm = [data_dy, data_x, data_variance, data_mean, data_gamma]

    return data_gm


def _get_params(shape_x, shape_mean, shape_gamma):
    """
    compute parameters including param_axis, reduce_axis and mean_num

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_mean: list or tuple
        shape of mean
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    """
    param_axis = LayerNormGradAxis.PARAMS_AXIS

    reduce_axis_tmp = []
    flag = -1
    for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
        if xtem != mean:
            flag = i
            break
    if flag != -1:
        for i in range(flag, len(shape_x)):
            reduce_axis_tmp.append(i)
    else:
        reduce_axis_tmp.append(len(shape_x) - 1)
    reduce_axis = tuple(reduce_axis_tmp)

    mean_num = 1.0
    for i in reduce_axis:
        mean_num *= shape_x[i]

    params = {"param_axis": param_axis, "reduce_axis": reduce_axis,
              "mean_num": mean_num}

    return params


def _get_pd_xl(data, shape_x):
    """
    compute pd_xl according to data_dy, data_gamma and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    shape_x: list or tuple
        shape of x

    Returns
    -------
    pd_xl: tvm.tensor
        data_dy*data_gamma
    """
    data_gamma_cast = tbe.broadcast(data.get("data_gamma"), shape_x)
    pd_xl = tbe.vmul(data_gamma_cast, data.get("data_dy"))

    return pd_xl


def _get_pd_var_front(data, cast_dtype):
    """
    compute front part of pd_var according to data_variance

    Parameters
    ----------
    data: dict
        placeholders after cast

    Returns
    -------
    pd_var_1: tvm.tensor
        np.power((data_variance + EPSLON), (-1.5))
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    """
    var_elta = tbe.vadds(data.get("data_variance"),
                         tvm.const(EPSLON, dtype=cast_dtype))
    var_elta_log = tbe.vlog(var_elta)
    var_elta_mul = tbe.vmuls(var_elta_log,
                             tvm.const(-0.5, dtype=cast_dtype))
    var_elta_2 = tbe.vexp(var_elta_mul)
    pdvar1_mul = tbe.vmul(var_elta_2, var_elta_2)
    pd_var_1 = tbe.vmul(pdvar1_mul, var_elta_2)

    return pd_var_1, var_elta_2


def _get_pd_var(data, params, shape_x, pd_xl, cast_dtype):
    """
    compute pd_var according to data_x, data_mean, reduce_axis and pd_xl

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    pd_xl: tvm.tensor
        data_dy*data_gamma

    Returns
    -------
    pd_var: tvm.tensor
        np.sum(((-0.5)*pd_xl*(data_x - data_mean)
        *np.power((data_variance + EPSLON), (-1.5))), reduce_axis,
        keepdims=True)
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    pd_var_1, var_elta_2 = _get_pd_var_front(data, cast_dtype)

    data_mean_cast = tbe.broadcast(data.get("data_mean"), shape_x)
    sub_x_mean = tbe.vsub(data.get("data_x"), data_mean_cast)

    pdvar_mul1 = tbe.vmul(pd_xl, sub_x_mean)
    pdvar_sum = tbe.sum(pdvar_mul1, params.get("reduce_axis"),
                        keepdims=True)
    pdvar_mul3 = tbe.vmul(pdvar_sum, pd_var_1)
    pd_var = tbe.vmuls(pdvar_mul3, tvm.const(-0.5, dtype=cast_dtype))

    return pd_var, var_elta_2, sub_x_mean


def _get_pd_mean(params, pd_xl, pd_var, var_elta_2, sub_x_mean, cast_dtype):
    """
    compute pd_mean according to reduce_axis, pd_xl, pd_var,
    var_elta_2 and sub_x_mean

    Parameters
    ----------
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    pd_xl: tvm.tensor
        data_dy*data_gamma
    pd_var: tvm.tensor
        np.sum(((-0.5)*pd_xl*(data_x - data_mean)
        *np.power((data_variance + EPSLON), (-1.5))),
        reduce_axis, keepdims=True)
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean

    Returns
    -------
    pd_mean: tvm.tensor
        np.sum(((-1.0)*pd_xl
        *np.power((data_variance + EPSLON), (-0.5))), reduce_axis,
        keepdims=True)
        + pd_var*(1.0/m)*np.sum(((-2.0)*(data_x - data_mean)),
        reduce_axis, keepdims=True)
    """
    pdmean1_sum = tbe.sum(pd_xl, params.get("reduce_axis"),
                          keepdims=True)
    pdmean1_mul = tbe.vmul(pdmean1_sum, var_elta_2)
    pd_mean_1 = tbe.vmuls(pdmean1_mul,
                          tvm.const(-1.0, dtype=cast_dtype))

    pdmean2_mul1 = tbe.vmuls(sub_x_mean,
                             tvm.const(-2.0, dtype=cast_dtype))
    pdmean2_sum = tbe.sum(pdmean2_mul1, params.get("reduce_axis"),
                          keepdims=True)
    pdmean2_mul3 = tbe.vmuls(pdmean2_sum,
                             tvm.const((params.get("mean_num")**(-1)),
                                       dtype=cast_dtype))
    pd_mean_2 = tbe.vmul(pd_var, pdmean2_mul3)

    pd_mean = tbe.vadd(pd_mean_2, pd_mean_1)

    return pd_mean


def _get_pd_x_front(data, params, shape_x, cast_dtype):
    """
    compute front part of pd_x according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x

    Returns
    -------
    pd_x_1: tvm.tensor
        pd_xl*np.power((data_variance + EPSLON), (-0.5))
    pd_x_2: tvm.tensor
        pd_var*(2.0/m)*(data_x - data_mean)
    pd_x_3: tvm.tensor
        pd_mean*(1.0/m)
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    pd_xl = _get_pd_xl(data, shape_x)

    pd_var, var_elta_2, sub_x_mean = _get_pd_var(data, params, shape_x, pd_xl,
                                                 cast_dtype)

    pd_mean = _get_pd_mean(params, pd_xl, pd_var, var_elta_2,
                           sub_x_mean, cast_dtype)

    var_elta_2_cast = tbe.broadcast(var_elta_2, shape_x)
    pd_x_1 = tbe.vmul(var_elta_2_cast, pd_xl)
    pdx2_broad = tbe.broadcast(pd_var, shape_x)
    pdx2_mul = tbe.vmul(pdx2_broad, sub_x_mean)
    pd_x_2 = tbe.vmuls(pdx2_mul,
                       tvm.const((2*(params.get("mean_num")**(-1))),
                                 dtype=cast_dtype))
    pd_x_3 = tbe.vmuls(pd_mean,
                       tvm.const((params.get("mean_num")**(-1)),
                                 dtype=cast_dtype))

    return [pd_x_1, pd_x_2, pd_x_3, var_elta_2_cast, sub_x_mean]


def _get_pd_x(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    pd_x_1, pd_x_2, pd_x_3, var_elta_2_cast, sub_x_mean = \
        _get_pd_x_front(data, params, shape_x, cast_dtype)

    pdx_broad = tbe.broadcast(pd_x_3, shape_x)
    pdx_add = tbe.vadd(pd_x_1, pd_x_2)
    pd_x_ub = tbe.vadd(pdx_add, pdx_broad)
    if dtype == "float16" and cast_dtype == "float32":
        pd_x = tbe.cast_to(pd_x_ub, dtype)
    else:
        pd_x = pd_x_ub

    return pd_x, var_elta_2_cast, sub_x_mean


def _get_pd_gamma(data, params, var_elta_2_cast, sub_x_mean, dtype,
                  cast_dtype):
    """
    compute pd_gamma according to data, params, var_elta_2_cast and sub_x_mean

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    dtype: str
        the data type

    Returns
    -------
    pd_gamma: tvm.tensor
        partial derivation of gamma
    """
    xl_mul = tbe.vmul(var_elta_2_cast, sub_x_mean)
    pdga_mul = tbe.vmul(data.get("data_dy"), xl_mul)

    if params.get("param_axis"):
        pd_gamma_ub = tbe.sum(pdga_mul, params.get("param_axis"),
                              keepdims=True)
        if dtype == "float16" and cast_dtype == "float32":
            pd_gamma = tbe.cast_to(pd_gamma_ub, dtype)
        else:
            return pd_gamma_ub
    else:
        if dtype == "float16" and cast_dtype == "float32":
            pd_gamma = tbe.cast_to(pdga_mul, dtype)
        else:
            return pdga_mul

    return pd_gamma


def _get_pd_beta(data, params, dtype, cast_dtype):
    """
    compute pd_beta according to data and params

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    dtype: str
        the data type

    Returns
    -------
    pd_beta: tvm.tensor
        partial derivation of beta
    """
    if params.get("param_axis"):
        pd_beta_ub = tbe.sum(data.get("data_dy"),
                             params.get("param_axis"), keepdims=True)
        if dtype == "float16" and cast_dtype == "float32":
            pd_beta = tbe.cast_to(pd_beta_ub, dtype)
        else:
            return pd_beta_ub
    elif (not params.get("param_axis")) and dtype == "float16":
        pd_beta = tbe.cast_to(data.get("data_dy"), dtype)
    else:
        pd_beta = tbe.vadds(data.get("data_dy"),
                            tvm.const(0, dtype="float32"))

    return pd_beta


def _get_res(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x, pd_gamma, pd_beta according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    pd_gamma: tvm.tensor
        partial derivation of gamma
    pd_beta: tvm.tensor
        partial derivation of beta
    """
    pd_x, var_elta_2_cast, sub_x_mean = _get_pd_x(data, params, shape_x,
                                                  dtype, cast_dtype)

    pd_gamma = _get_pd_gamma(data, params, var_elta_2_cast, sub_x_mean,
                             dtype, cast_dtype)

    pd_beta = _get_pd_beta(data, params, dtype, cast_dtype)

    return pd_x, pd_gamma, pd_beta


def _get_pds(data_dy, data_x, data_variance, data_mean,
             data_gamma, shape_gamma_ori):
    """
    get params and data, compute pd_x, pd_gamma, pd_beta.

    Parameters
    ----------
    data_dy: TVM tensor
        the placeholder of dy input data
    data_x: TVM tensor
        the placeholder of x input data
    data_variance: TVM tensor
        the placeholder of variance input data
    data_mean: TVM tensor
        the placeholder of mean input data
    data_gamma: TVM tensor
        the placeholder of gamma input data
    shape_gamma_ori: list or tuple
        original shape of gamma

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    pd_gamma: tvm.tensor
        partial derivation of gamma
    pd_beta: tvm.tensor
        partial derivation of beta
    """
    dtype = data_dy.dtype.lower()
    shape_x = shape_util.shape_to_list(data_x.shape)
    shape_mean = shape_util.shape_to_list(data_mean.shape)

    params = _get_params(shape_x, shape_mean, shape_gamma_ori)

    has_improve_precision = False
    cast_dtype = dtype
    if dtype == "float16" and tbe_platform.api_check_support(
            "tbe.dsl.vexp", "float32"):
        has_improve_precision = True
        cast_dtype = "float32"

    if has_improve_precision:
        data_dy = tbe.cast_to(data_dy, "float32")
        data_x = tbe.cast_to(data_x, "float32")
        data_variance = tbe.cast_to(data_variance, "float32")
        data_mean = tbe.cast_to(data_mean, "float32")
        data_gamma = tbe.cast_to(data_gamma, "float32")

    data = {"data_dy": data_dy, "data_x": data_x,
            "data_variance": data_variance,
            "data_mean": data_mean, "data_gamma": data_gamma}

    pd_x, pd_gamma, pd_beta = _get_res(data, params, shape_x,
                                       dtype, cast_dtype)

    return pd_x, pd_gamma, pd_beta


@register_operator_compute("layer_norm_grad", op_mode="static", support_fusion=True)
def layer_norm_grad_compute(input_dy, input_x, input_variance, input_mean,
                            input_gamma, output_pd_x, output_pd_gamma,
                            output_pd_beta, kernel_name="layer_norm_grad"):
    """
    DSL description of the layernorm_grad operator's mathematical
    calculation process

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_pd_x: dict
        shape and dtype of output_pd_x, only support float16, float32
    output_pd_gamma: dict
        shape and dtype of output_pd_gamma, only support float16, float32
    output_pd_beta: dict
        shape and dtype of output_pd_beta, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_grad"

    Returns
    -------
    res_tuple: tuple
        (pd_x, pd_gamma, pd_beta)
    """
    pd_x, pd_gamma, pd_beta = _get_pds(input_dy, input_x, input_variance,
                                       input_mean, input_gamma,
                                       input_gamma.shape)
    res_list = [pd_x, pd_gamma, pd_beta]

    return res_list


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def layer_norm_grad(input_dy, input_x, input_variance, input_mean,
                    input_gamma, output_pd_x, output_pd_gamma,
                    output_pd_beta, kernel_name="layer_norm_grad"):
    """
    algorithm: layernorm_grad
    calculating: gradient of layernorm
                 compute partial derivation of x, gamma and beta
        pd_xl    = data_dy*data_gamma
        pd_var   = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
                   *np.power((data_variance + EPSLON), (-1.5))),
                   reduce_axis, keepdims=True)
        pd_mean  = np.sum(((-1.0)*pd_xl
                   *np.power((data_variance + EPSLON), (-0.5))),
                   reduce_axis, keepdims=True)
                   + pd_var*(1.0/m)
                   *np.sum(((-2.0)*(data_x - data_mean)), reduce_axis,
                   keepdims=True)
        pd_x     = pd_xl*np.power((data_variance + EPSLON), (-0.5))
                   + pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
        pd_gamma = np.sum((data_dy*(data_x - data_mean)
                   *np.power((data_variance + EPSLON), (-0.5))), param_axis,
                   keepdims=True)
        pd_beta  = np.sum(data_dy, param_axis, keepdims=True)

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_y: dict
        shape and dtype of output, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_grad"

    Returns
    -------
    None
    """
    dtype = input_dy.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    shape_variance = input_variance.get("shape")
    shape_mean = input_mean.get("shape")
    shape_gamma = input_gamma.get("shape")

    _check_params({"shape_dy": shape_dy, "shape_x": shape_x,
                   "shape_var": shape_variance,
                   "shape_mean": shape_mean, "shape_gamma": shape_gamma,
                   "dtype": dtype, "kernel_name": kernel_name})

    shape_gamma, LayerNormGradAxis.PARAMS_AXIS = _update_gamma_shape(shape_x, shape_gamma)

    data_gm = _get_data_gm({"shape_dy": shape_dy, "shape_x": shape_x,
                            "shape_var": shape_variance,
                            "shape_mean": shape_mean,
                            "shape_gamma": shape_gamma}, dtype)

    res_list = layer_norm_grad_compute(data_gm[0], data_gm[1], data_gm[2],
                                       data_gm[3], data_gm[4], output_pd_x,
                                       output_pd_gamma, output_pd_beta)

    with tvm.target.cce():
        sch = auto_schedule(res_list)

    tensor_list = list(data_gm) + list(res_list)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    build(sch, config)
