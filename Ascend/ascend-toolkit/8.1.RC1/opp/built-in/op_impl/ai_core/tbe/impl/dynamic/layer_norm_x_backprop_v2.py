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
layer_norm_x_backprop_v2
"""
# 'pylint: disable=too-many-lines
from tbe.dsl.base import operation
from impl.layer_norm_x_backprop_v2 import op_select_format as static_op_select_format
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.norm_pattern_adapter import NormPattern
from impl.util.util_common import is_unknown_rank_input


def op_select_format(input_dy,
                     input_x,
                     input_variance,
                     input_mean,
                     input_gamma,
                     output_pd_x,
                     res_for_gamma,
                     kernel_name="layer_norm_x_backprop_v2"):
    """
    function of selecting dynamic format

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32, bfloat16
    input_x: dict
        shape and dtype of input x, only support float16, float32, bfloat16
    input_variance: dict
        shape and dtype of input variance, only support float16, float32, bfloat16
    input_mean: dict
        shape and dtype of input mean, only support float16, float32, bfloat16
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32, bfloat16
    output_pd_x: dict
        shape and dtype of output, only support float16, float32, bfloat16
    res_for_gamma: dict
        shape and dtype of output for gamma, only support float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v2"

    Returns
    -------
    None
    """
    return static_op_select_format(input_dy, input_x, input_variance, input_mean, input_gamma,
                                   output_pd_x, res_for_gamma, kernel_name)


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
    data_dy = tvm.placeholder(shapes.get("shape_dy"), name="data_dy", dtype=dtype)
    data_x = tvm.placeholder(shapes.get("shape_x"), name="data_x", dtype=dtype)
    data_variance = tvm.placeholder(shapes.get("shape_var"), name="data_variance", dtype=dtype)
    data_mean = tvm.placeholder(shapes.get("shape_mean"), name="data_mean", dtype=dtype)
    data_gamma = tvm.placeholder(shapes.get("shape_gamma"), name="data_gamma", dtype=dtype)

    data_gm = (data_dy, data_x, data_variance, data_mean, data_gamma)

    return data_gm


def _get_params_after_classify(shape_x, reduce_axis, params):
    """
    compute parameters including param_axis, reduce_axis and mean_num

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    reduce_axis: list or tuple
        reduce axis of inputs
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
         "mean_num": mean_num}

    Returns
    -------
    None
    """
    reduce_elts = 1
    for idx in reduce_axis:
        reduce_elts *= shape_x[idx]
    if isinstance(reduce_elts, int):
        mean_cofs = reduce_elts ** (-1)
        mean_cof = tvm.const(mean_cofs, dtype="float32")
        mean_cof2 = tvm.const(2*mean_cofs, dtype="float32")
    else:
        mean_cof = tbe.var("mean_cof", dtype="float32")
        mean_cof2 = tbe.var("mean_cof_double", dtype="float32")
        operation.add_compile_info("reduce_mean_cof", True)
    
    params["reduce_axis"] = reduce_axis
    params.update({"mean_num": [mean_cof, mean_cof2]})


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
    reduce_axis_tmp = []
    flag = -1
    for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
        if xtem != 1 and mean == 1:
            reduce_axis_tmp.append(i)
    if not reduce_axis_tmp:
        reduce_axis_tmp = [-1]
    reduce_axis = tuple(reduce_axis_tmp)
    all_axis = set(range(len(shape_x)))
    param_axis = tuple(all_axis - set(reduce_axis))

    params = {"reduce_axis": reduce_axis,
              "param_axis": param_axis}

    return params


def _broadcast_interval_dimension(tensor, shape):
    if shape_util.shape_to_list(tensor.shape)[0] == 1 and shape_util.shape_to_list(tensor.shape)[-1] == 1:
        tmp_shape = [1] + shape[1:]
        tmp_tensor = tbe.broadcast(tensor, tmp_shape)
        tensor_target = tbe.broadcast(tmp_tensor, shape)
        return tensor_target
    tensor_target = tbe.broadcast(tensor, shape)
    return tensor_target


def _get_pd_x(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis, "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type
    cast_dtype: str
        if api_check_support float32, then equal to float32 else float16

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    res_for_gamma: tvm.tensor
        `(data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))`
    """
    data_gamma_cast = tbe.broadcast(data.get("data_gamma"), shape_x)
    pd_xl = tbe.vmul(data_gamma_cast, data.get("data_dy"))

    var_elta = tbe.vadds(data.get("data_variance"), tvm.const(EPSLON, dtype=cast_dtype))
    var_elta_log = tbe.vlog(var_elta)
    var_elta_mul = tbe.vmuls(var_elta_log, tvm.const(-0.5, dtype=cast_dtype))
    var_elta_2 = tbe.vexp(var_elta_mul)
    pdvar1_mul = tbe.vmul(var_elta_2, var_elta_2)
    pd_var_1 = tbe.vmul(pdvar1_mul, var_elta_2)

    data_mean_cast = _broadcast_interval_dimension(data.get("data_mean"), shape_x)
    sub_x_mean = tbe.vsub(data.get("data_x"), data_mean_cast)

    pdvar_mul1 = tbe.vmul(pd_xl, sub_x_mean)
    pdvar_sum = tbe.reduce_sum(pdvar_mul1, params.get("reduce_axis"), keepdims=True)
    pdvar_mul3 = tbe.vmul(pdvar_sum, pd_var_1)
    pd_var = tbe.vmuls(pdvar_mul3, tvm.const(-0.5, dtype=cast_dtype))

    var_elta_2_cast = _broadcast_interval_dimension(var_elta_2, shape_x)
    res_for_gamma = tbe.vmul(sub_x_mean, var_elta_2_cast)

    pdmean1_sum = tbe.reduce_sum(pd_xl, params.get("reduce_axis"), keepdims=True)
    pdmean1_mul = tbe.vmul(pdmean1_sum, var_elta_2)
    pd_mean = tbe.vmuls(pdmean1_mul, tvm.const(-1.0, dtype=cast_dtype))

    var_elta_2_cast = _broadcast_interval_dimension(var_elta_2, shape_x)
    pd_x_1 = tbe.vmul(var_elta_2_cast, pd_xl)
    pdx2_broad = _broadcast_interval_dimension(pd_var, shape_x)
    pdx2_mul = tbe.vmul(pdx2_broad, sub_x_mean)
    pd_x_2 = tbe.vmuls(pdx2_mul, params.get("mean_num")[1])
    pd_x_3 = tbe.vmuls(pd_mean, params.get("mean_num")[0])


    pdx_broad = _broadcast_interval_dimension(pd_x_3, shape_x)
    pdx_add = tbe.vadd(pd_x_1, pd_x_2)
    pd_x_ub = tbe.vadd(pdx_add, pdx_broad)

    if dtype == "float16" and cast_dtype == "float32":
        pd_x = tbe.cast_to(pd_x_ub, dtype)
    elif dtype == "bfloat16" and cast_dtype == "float32":
        pd_x = tbe.round(pd_x_ub, dtype)
    else:
        return pd_x_ub, res_for_gamma

    return pd_x, res_for_gamma


def _get_res(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x, pd_gamma, pd_beta according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis, "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type
    cast_dtype: str
        if api_check_support float32, then equal to float32 else float16

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    res_for_gamma: tvm.tensor
        `(data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))`
    """
    pd_x, res_for_gamma = _get_pd_x(data, params, shape_x, dtype, cast_dtype)

    return pd_x, res_for_gamma


# 'pylint: disable=too-many-arguments
def _get_pds(data_dy, data_x, data_variance, data_mean,
             data_gamma, shape_gamma_ori, params):
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
    res_for_gamma: tvm.tensor
        `(data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))`
    """
    dtype = data_dy.dtype.lower()
    shape_x = shape_util.shape_to_list(data_x.shape)

    has_improve_precision = False
    cast_dtype = dtype
    if dtype == "bfloat16" or (dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32")):
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

    pd_x, res_for_gamma = _get_res(data, params, shape_x, dtype, cast_dtype)

    return pd_x, res_for_gamma


def _update_shape_nz(shape_x, shape_var, shape_gamma):
    """
    function of updating Nz shape

    """
    # ND shape of x >= two dim
    # Nz shape of x >= four dim
    len_x = len(shape_x)
    nz_begin = len_x - 4
    shape_x_nz = []
    for i in range(0, nz_begin):
        shape_x_nz.append(shape_x[i])
    shape_x_nz.append(shape_x[nz_begin])
    shape_x_nz.append(shape_x[nz_begin + 1])
    shape_x_nz.append(shape_x[nz_begin + 2])
    shape_x_nz.append(shape_x[nz_begin + 2])

    # ND shape of var >= two dim
    shape_var_nz = []
    len_var = len(shape_var)
    var_nz_begin = len_var - 2
    for i in range(0, var_nz_begin):
        shape_var_nz.append(shape_var[i])
    shape_var_nz.append(1)
    shape_var_nz.append(shape_x[nz_begin + 1])
    shape_var_nz.append(shape_x[nz_begin + 2])
    shape_var_nz.append(1)

    # ND shape of gamma is one dim
    shape_gamma_nz = []
    for i in range(0, nz_begin):
        shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin])
    shape_gamma_nz.append(1)
    shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin + 2])

    reduce_nz_axis = []
    for i, (xtem, var) in enumerate(zip(shape_x_nz, shape_var_nz)):
        if xtem != 1 and var == 1:
            reduce_nz_axis.append(i)

    all_axis = set(range(len(shape_x_nz)))
    param_axis = tuple(all_axis - set(reduce_nz_axis))

    param_nz = {
        "shape_x_nz": shape_x_nz,
        "shape_var_nz": shape_var_nz,
        "shape_gamma_nz": shape_gamma_nz,
        "reduce_axis": reduce_nz_axis,
        "param_axis": param_axis
    }

    return param_nz


# 'pylint: disable=too-many-arguments
def layer_norm_x_backprop_v2_compute(input_dy, input_x,
                                     input_variance, input_mean,
                                     input_gamma, output_pd_x, output_res_gamma,
                                     params, kernel_name="layer_norm_x_backprop_v2"):
    """
    DSL description of the layernorm_grad operator's mathematical calculation process

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support bfloat16, float16, float32
    input_x: dict
        shape and dtype of input x, only support bfloat16, float16, float32
    input_variance: dict
        shape and dtype of input variance, only support bfloat16, float16, float32
    input_mean: dict
        shape and dtype of input mean, only support bfloat16, float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support bfloat16, float16, float32
    output_pd_x: dict
        shape and dtype of output, only support bfloat16, float16, float32
    output_res_gamma: dict
        shape and dtype of output for gamma, only support float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v2"

    Returns
    -------
    res_tuple: tuple
        (pd_x, res_for_gamma)
    """
    pd_x, res_for_gamma = _get_pds(input_dy, input_x, input_variance, input_mean, input_gamma,
                                   input_gamma.shape, params)
    res_list = [pd_x, res_for_gamma]

    return res_list


@register_operator("LayerNormXBackpropV2", "Norm")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def layer_norm_x_backprop_v2(input_dy, input_x, input_variance, input_mean,
                             input_gamma, output_pd_x, output_res_gamma,
                             kernel_name="layer_norm_x_backprop_v2"):
    """
    algorithm: layernorm_x_backprop_v2
    calculating: gradient of layernorm
                 compute partial derivation of x, gamma and beta
        `pd_xl    = data_dy * data_gamma`
        `pd_var   = np.sum(((-0.5) * pd_xl * (data_x - data_mean) * np.power((data_variance + EPSLON), `
                    `(-1.5))), reduce_axis, keepdims=True)`
        `pd_mean  = np.sum(((-1.0) * pd_xl * np.power((data_variance + EPSLON), (-0.5))),`
                   `reduce_axis, keepdims=True)`
                   `+ pd_var * (1.0 / m) * np.sum(((-2.0) * (data_x - data_mean)),`
                   `reduce_axis, keepdims=True)`
        `pd_x     = pd_xl * np.power((data_variance + EPSLON), (-0.5))`
                   `+ pd_var * (2.0 / m) * (data_x - data_mean) + pd_mean * (1.0 / m)`
        `res_for_gamma = (data_x - data_mean) * np.power((data_variance + EPSLON), (-0.5))`

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support bfloat16, float16, float32
    input_x: dict
        shape and dtype of input x, only support bfloat16, float16, float32
    input_variance: dict
        shape and dtype of input variance, only support bfloat16, float16, float32
    input_mean: dict
        shape and dtype of input mean, only support bfloat16, float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support bfloat16, float16, float32
    output_y: dict
        shape and dtype of output, only support bfloat16, float16, float32
    output_res_gamma: dict
        shape and dtype of output for gamma, only support float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v2"

    Returns
    -------
    None
    """
    dtype = input_dy.get("dtype").lower()
    variance_dtype = input_variance.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    shape_variance = input_variance.get("shape")
    shape_gamma = input_gamma.get("shape")
    format_dy = input_dy.get("format")
    extra_params = {"input_shape_type": [0, 0, 1, 1, 1],
                    "same_input_shape_group": [[0, 1], [2, 3]]}
    global EPSLON
    EPSLON = 1e-5 if dtype in ("float16", "bfloat16") and dtype == variance_dtype else 1e-12

    if is_unknown_rank_input((input_dy, input_x, input_variance,
                              input_mean, input_gamma)):
        reduce_axis = NormPattern.REDUCE_UNKNOWN_MODE
        broadcast_axis = NormPattern.BROADCAST_UNKNOWN_MODE
        extra_params.update(NormPattern.REDUCE_AFTER_TYPE)
        extra_params.update({"compile_broadcast_axes": {2: reduce_axis, 3: reduce_axis,
                                                        4: broadcast_axis}})
        extra_params.update({"broadcast_axes_type": {2: "same_reduce", 3: "same_reduce",
                                                     4: "opposite_reduce"}})
        operation.add_compile_info("unknown_mode", True)

        ins = classify([input_dy, input_x, input_variance, input_mean, input_gamma,
                        reduce_axis], OpPatternMode.NORM, extra_params)
    else:
        if format_dy.upper() == "FRACTAL_NZ":
            params = _update_shape_nz(shape_x, shape_variance, shape_gamma)
            input_dy["shape"] = params.get("shape_x_nz")
            input_x["shape"] = params.get("shape_x_nz")
            input_variance["shape"] = params.get("shape_var_nz")
            input_mean["shape"] = params.get("shape_var_nz")
            input_gamma["shape"] = params.get("shape_gamma_nz")

            for input_tensor in (input_dy, input_x, input_variance, input_mean, input_gamma):
                nz_range = [(1, None)] * len(params.get("shape_x_nz"))
                input_tensor["range"] = nz_range
        else:
            params = _get_params(shape_x, shape_variance, shape_gamma)

        extra_params.update({"compile_broadcast_axes": {2: params.get("reduce_axis"),
                                                        3: params.get("reduce_axis"),
                                                        4: params.get("param_axis")}})

        ins = classify([input_dy, input_x, input_variance, input_mean, input_gamma,
                        params.get("reduce_axis")], OpPatternMode.NORM, extra_params)
    
    schedules = []
    tensors = []
    for (ins_dy, ins_x, ins_variance, ins_mean, ins_gamma, ins_reduce_axis) in ins:
        with tbe.compute():
            shape_dy, shape_x, shape_variance, shape_mean, shape_gamma = \
                shape_util.variable_shape([ins_dy, ins_x, ins_variance, ins_mean, ins_gamma],
                                          op_mode="norm")

            data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype)
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype)
            data_variance = tvm.placeholder(shape_variance, name="data_variance", dtype=variance_dtype)
            data_mean = tvm.placeholder(shape_mean, name="data_mean", dtype=variance_dtype)
            data_gamma = tvm.placeholder(shape_gamma, name="data_gamma", dtype=variance_dtype)

            if is_unknown_rank_input((input_dy, input_x, input_variance,
                                      input_mean, input_gamma)):
                mean_cof = tbe.var("mean_cof", dtype="float32")
                mean_cof2 = tbe.var("mean_cof_double", dtype="float32")
                operation.add_compile_info("reduce_mean_cof", True)
                params = {"reduce_axis": ins_reduce_axis,
                          "mean_num": [mean_cof, mean_cof2]}
            else:
                _get_params_after_classify(shape_x, ins_reduce_axis, params)
            
            res_list = layer_norm_x_backprop_v2_compute(data_dy, data_x, data_variance,
                                                        data_mean, data_gamma, output_pd_x,
                                                        output_res_gamma, params)
            tensor_list = [data_dy, data_x, data_variance, data_mean, data_gamma] + list(res_list)
            tensors.append(tensor_list)
        
        with tvm.target.cce():
            sch = tbe.auto_schedule(res_list)
        schedules.append(sch)
    
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    
    tbe.build(schedules, config)
