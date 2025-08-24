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
layer_norm_beta_gamma_backprop
"""
# 'pylint: disable=too-many-lines
import tbe.common.register as tbe_register
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check


# 'pylint: disable=unused-argument,too-many-arguments
@tbe_register.register_param_generalization("LayerNormBetaGammaBackprop")
def layer_norm_beta_gamma_backprop_generalization(input_dy, input_x, input_variance, input_mean, output_pd_gamma,
                                                  output_pd_beta, shape_gamma, impl_mode,
                                                  generalize_config=None):
    """
    Parameters
    ----------
    input_dy: TVM tensor
        the placeholder of dy input data
    input_x: TVM tensor
        the placeholder of x input data
    input_variance: TVM tensor
        the placeholder of variance input data
    input_mean: TVM tensor
        the placeholder of mean input data
    output_pd_gamma: dict
        shape and dtype of output, only support float16, float32
    output_pd_beta: dict
        shape and dtype of output, only support float16, float32
    shape_gamma: list or tuple
        original shape of gamma
    generalize_config: None
    """
    # for now only support dy and x is (-1,-1,N), variance and mean is (-1,-1,1), shape_gamma is (N,)
    result = []
    x_shape = input_x["shape"]
    last_dim = x_shape[-1]
    input_dy["shape"] = [-1, -1, last_dim]
    input_x["shape"] = [-1, -1, last_dim]
    input_variance["shape"] = [-1, -1, 1]
    input_mean["shape"] = [-1, -1, 1]
    result.append([input_dy, input_x, input_variance, input_mean, output_pd_gamma, output_pd_beta, shape_gamma])
    return result


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
    shape = shapes.get("shape_dy")
    dim_0 = tbe.var("dim_0")
    dim_1 = tbe.var("dim_1")
    shape_x = (dim_0, dim_1, shape[2])
    shape_mean = (dim_0, dim_1, 1)

    data_dy = tvm.placeholder(shape_x,
                              name="data_dy_layernormgrad_beta_gamma", dtype=dtype)
    data_x = tvm.placeholder(shape_x,
                             name="data_x", dtype=dtype)
    data_variance = tvm.placeholder(shape_mean,
                                    name="data_variance", dtype=dtype)
    data_mean = tvm.placeholder(shape_mean,
                                name="data_mean", dtype=dtype)

    data_gm = (data_dy, data_x, data_variance, data_mean)

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
    param_axis = _update_gamma_shape(shape_x, shape_gamma)[1]

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

    params = \
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
         "mean_num": mean_num}

    return params


def _get_pd_var_front(data, dtype):
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
    # Minimum positive number greater than 0
    epslon = 1e-12

    var_elta = tbe.vadds(data.get("data_variance"),
                         tvm.const(epslon, dtype=dtype))
    var_elta_log = tbe.vlog(var_elta)
    var_elta_mul = tbe.vmuls(var_elta_log,
                             tvm.const(-0.5, dtype=dtype))
    var_elta_2 = tbe.vexp(var_elta_mul)

    return var_elta_2


def _get_pd_var(data, shape_x, dtype):
    """
    compute pd_var according to data_x, data_mean, reduce_axis and pd_xl

    Parameters
    ----------
    data: dict
        placeholders after cast
    shape_x: list or tuple
        shape of x

    Returns
    -------
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    var_elta_2 = _get_pd_var_front(data, dtype)
    data_mean_cast = tbe.broadcast(data.get("data_mean"), shape_x)
    sub_x_mean = tbe.vsub(data.get("data_x"), data_mean_cast)

    return var_elta_2, sub_x_mean


def _get_pd_mean(params, pd_xl, pd_var, var_elta_2, sub_x_mean):
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
        *np.power((data_variance + EPSLON), (-0.5))),
        reduce_axis, keepdims=True)
        + pd_var*(1.0/m)*np.sum(((-2.0)*(data_x - data_mean)),
        reduce_axis, keepdims=True)
    """
    pdmean1_sum = tbe.reduce_sum(pd_xl, params.get("reduce_axis"),
                                 keepdims=True)
    pdmean1_mul = tbe.vmul(pdmean1_sum, var_elta_2)
    pd_mean_1 = tbe.vmuls(pdmean1_mul,
                          tvm.const(-1.0, dtype="float32"))

    pdmean2_mul1 = tbe.vmuls(sub_x_mean,
                             tvm.const(-2.0, dtype="float32"))
    pdmean2_sum = tbe.reduce_sum(pdmean2_mul1, params.get("reduce_axis"),
                                 keepdims=True)
    pdmean2_mul3 = tbe.vmuls(pdmean2_sum,
                             tvm.const((params.get("mean_num")**(-1)),
                                       dtype="float32"))
    pd_mean_2 = tbe.vmul(pd_var, pdmean2_mul3)

    pd_mean = tbe.vadd(pd_mean_2, pd_mean_1)

    return pd_mean


def _get_pd_x_front(data, shape_x, dtype):
    """
    compute front part of pd_x according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    shape_x: list or tuple
        shape of x

    Returns
    -------
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    var_elta_2, sub_x_mean = _get_pd_var(data, shape_x, dtype)

    var_elta_2_cast = tbe.broadcast(var_elta_2, shape_x)

    return var_elta_2_cast, sub_x_mean


def _get_pd_x(data, shape_x, dtype):
    """
    compute pd_x according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    shape_x: list or tuple
        shape of x

    Returns
    -------
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    var_elta_2_cast, sub_x_mean = _get_pd_x_front(data, shape_x, dtype)

    return var_elta_2_cast, sub_x_mean


# 'pylint: disable=unused-argument
def _get_pd_gamma(data, params, var_elta_2_cast, sub_x_mean, dtype):
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
        pd_gamma = tbe.reduce_sum(pdga_mul, params.get("param_axis"),
                                  keepdims=True)

    return pd_gamma


def _get_pd_beta(data, params, dtype):
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
        pd_beta = tbe.reduce_sum(data.get("data_dy"), params.get("param_axis"),
                                 keepdims=True)
    else:
        pd_beta = tbe.vadds(data.get("data_dy"),
                            tvm.const(0, dtype=dtype))

    return pd_beta


def _get_res(data, params, shape_x, dtype):
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
    pd_gamma: tvm.tensor
        partial derivation of gamma
    pd_beta: tvm.tensor
        partial derivation of beta
    """
    var_elta_2_cast, sub_x_mean = _get_pd_x(data, shape_x, dtype)

    xl_mul = tbe.vmul(var_elta_2_cast, sub_x_mean)
    pdga_mul = tbe.vmul(data.get("data_dy"), xl_mul)

    data_dy = data.get("data_dy")
    if params.get("param_axis"):
        data_dy_2 = data.get("data_dy_2")
        if data_dy_2 is None:
            data_dy_2 = data_dy
        pd_gamma = tbe.reduce_sum(pdga_mul, params.get("param_axis"), keepdims=True)
        pd_beta = tbe.reduce_sum(data_dy_2, params.get("param_axis"), keepdims=True)
    else:
        pd_beta = tbe.vadds(data_dy,
                            tvm.const(0, dtype=dtype))
        pd_gamma = pdga_mul

    return pd_gamma, pd_beta


def _get_pds(data_dy, data_x, data_variance, data_mean, shape_gamma_ori):
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
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
        dtype = "float32"
    data_dy_2 = None
    if has_improve_precision:
        data_tmp = data_dy
        data_dy = tbe.cast_to(data_dy, "float32")
        data_dy_2 = tbe.cast_to(data_tmp, "float32")
        data_x = tbe.cast_to(data_x, "float32")
        data_variance = tbe.cast_to(data_variance, "float32")
        data_mean = tbe.cast_to(data_mean, "float32")

    data = {"data_dy": data_dy, "data_x": data_x, "data_dy_2": data_dy_2,
            "data_variance": data_variance,
            "data_mean": data_mean}

    pd_gamma, pd_beta = _get_res(data, params, shape_x, dtype)

    if dtype == "float16" and not has_improve_precision:
        pd_gamma = tbe.cast_to(pd_gamma, "float32")
        pd_beta = tbe.cast_to(pd_beta, "float32")

    return pd_gamma, pd_beta


# 'pylint: disable=unused-argument,too-many-arguments
def layer_norm_beta_gamma_backprop_compute(input_dy, input_x, input_variance,
                                           input_mean, output_pd_gamma,
                                           output_pd_beta, shape_gamma,
                                           kernel_name="layer_norm_beta"
                                                       "_gamma_backprop"):
    """
    DSL description of the layernorm_grad operator's
    mathematical calculation process

    Parameters
    ----------
    input_dy: TVM tensor
        the placeholder of dy input data
    input_x: TVM tensor
        the placeholder of x input data
    input_variance: TVM tensor
        the placeholder of variance input data
    input_mean: TVM tensor
        the placeholder of mean input data
    data_gamma: TVM tensor
        the placeholder of gamma input data
    shape_gamma: list or tuple
        original shape of gamma

    Returns
    -------
    res_tuple: tuple
        (pd_gamma, pd_beta)
    """
    pd_gamma, pd_beta = _get_pds(input_dy, input_x,
                                 input_variance, input_mean, shape_gamma)
    res_list = [pd_gamma, pd_beta]

    return res_list


# 'pylint: disable=too-many-arguments,too-many-locals
@register_operator("LayerNormBetaGammaBackprop", pattern="Layer_norm_beta_gamma_backprop")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def layer_norm_beta_gamma_backprop(input_dy, input_x, input_variance,
                                   input_mean, output_pd_gamma,
                                   output_pd_beta, shape_gamma,
                                   kernel_name="layer_norm_beta_"
                                               "gamma_backprop"):
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
                     *np.sum(((-2.0)*(data_x - data_mean)),
                     reduce_axis, keepdims=True)
          pd_gamma = np.sum((data_dy*(data_x - data_mean)
                     *np.power((data_variance + EPSLON), (-0.5))),
                     param_axis, keepdims=True)
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

    format_dy = input_dy.get("format")

    if format_dy.upper() != "FRACTAL_NZ":
        shape_gamma_ori = shape_gamma

        data_gm = _get_data_gm({"shape_dy": shape_dy, "shape_x": shape_x,
                                "shape_var": shape_variance,
                                "shape_mean": shape_mean}, dtype)

        with tbe.compute():
            res_list = layer_norm_beta_gamma_backprop_compute(data_gm[0],
                                                              data_gm[1],
                                                              data_gm[2],
                                                              data_gm[3],
                                                              output_pd_gamma,
                                                              output_pd_beta,
                                                              shape_gamma_ori)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res_list)

        tensor_list = list(data_gm) + list(res_list)

        config = {"print_ir": False,
                  "name": kernel_name,
                  "tensor_list": tensor_list}

        tbe.build(sch, config)
