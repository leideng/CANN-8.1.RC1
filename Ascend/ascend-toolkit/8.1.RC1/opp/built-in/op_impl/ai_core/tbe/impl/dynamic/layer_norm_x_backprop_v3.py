# Copyright 2023 Huawei Technologies Co., Ltd
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
layer_norm_x_backprop_v3
"""
from tbe.dsl.base import operation
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util import util_select_op_base
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.norm_pattern_adapter import NormPattern
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_dynamic_input


# 'pylint: disable=too-many-lines
# 'pylint: disable = unused-argument,too-many-arguments,too-many-locals,global-variable-undefined
# 'pylint: disable=huawei-too-many-arguments
def get_op_support_info(input_dy,
                        input_x,
                        input_rstd,
                        input_mean,
                        input_gamma,
                        output_pd_x,
                        res_for_gamma,
                        kernel_name="layer_norm_x_backprop_v3"):
    """
    get_op_support_info
    """
    shape_x = input_x.get("shape")
    shape_mean = input_mean.get("shape")
    shape_gamma = input_gamma.get("shape")
    format_dy = input_dy.get("format").upper()
    if format_dy in ("ND", "NCHW", "NHWC", "NC1HWC0"):
        if len(shape_x) == len(shape_gamma):
            axis_split_matrix = []
            flag = -1
            for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
                if xtem != mean:
                    flag = i
                    break
            if flag == -1:
                for i in range(len(shape_x) - 1):
                    split_0 = [
                        SplitInput([0, [i], [-1], [-1]], [1, [i], [-1], [-1]], [2, [i], [-1], [-1]],
                                   [3, [i], [-1], [-1]], [4, [i], [-1], [-1]]),
                        SplitOutput([0, [i]])
                    ]
                    axis_split_matrix.append(split_0)
            else:
                for i in range(flag):
                    split_0 = [
                        SplitInput([0, [i], [-1], [-1]], [1, [i], [-1], [-1]], [2, [i], [-1], [-1]],
                                   [3, [i], [-1], [-1]], [4, [i], [-1], [-1]]),
                        SplitOutput([0, [i]], [1, [i]])
                    ]
                    axis_split_matrix.append(split_0)
        else:
            axis_split_matrix = None

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable = unused-argument
# 'pylint: disable=huawei-too-many-arguments
def op_select_format(input_dy,
                     input_x,
                     input_rstd,
                     input_mean,
                     input_gamma,
                     output_pd_x,
                     res_for_gamma,
                     kernel_name="layer_norm_x_backprop_v3"):
    """
    function of selecting dynamic format

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32, bfloat16
    input_x: dict
        shape and dtype of input x, only support float16, float32, bfloat16
    input_rstd: dict
        shape and dtype of input rstd, only support float16, float32, bfloat16
    input_mean: dict
        shape and dtype of input mean, only support float16, float32, bfloat16
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32, bfloat16
    output_pd_x: dict
        shape and dtype of output, only support float16, float32, bfloat16
    res_for_gamma: dict
        shape and dtype of output for gamma, only support float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v3"

    Returns
    -------
    None
    """
    shape_dy = input_dy.get("ori_shape")
    shape_gamma = input_gamma.get("ori_shape")
    shape_dy = shape_util.scalar2tensor_one(shape_dy)
    shape_gamma = shape_util.scalar2tensor_one(shape_gamma)
    c_0 = 16
    if _check_dynamic_format(shape_dy, shape_gamma, c_0):
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="dy",
                                               datatype="float16,float16,float16,"
                                                        "float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,"
                                                        "float16,float16,float16,"
                                                        "bfloat16,bfloat16,bfloat16,"
                                                        "float16,float16,float16,"
                                                        "bfloat16,bfloat16,bfloat16",
                                               format="NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16,float16,float16,"
                                                        "float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,"
                                                        "float16,float16,float16,"
                                                        "bfloat16,bfloat16,bfloat16,"
                                                        "float16,float16,float16,"
                                                        "bfloat16,bfloat16,bfloat16",
                                               format="NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="rstd",
                                               datatype="float16,float16,float16,"
                                                        "float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,"
                                                        "float,float,float,"
                                                        "float,float,float,"
                                                        "float,float,float,"
                                                        "float,float,float",
                                               format="NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="mean",
                                               datatype="float16,float16,float16,"
                                                        "float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,"
                                                        "float,float,float,"
                                                        "float,float,float,"
                                                        "float,float,float,"
                                                        "float,float,float",
                                               format="NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND")
        input4 = util_select_op_base.gen_param(classify="input4",
                                               name="gamma",
                                               datatype="float16,float16,float16,"
                                                        "float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,"
                                                        "float,float,float,"
                                                        "float,float,float,"
                                                        "float16,float16,float16,"
                                                        "bfloat16,bfloat16,bfloat16",
                                               format="NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="pd_x",
                                                datatype="float16,float16,float16,"
                                                         "float,float,float,"
                                                         "bfloat16,bfloat16,bfloat16,"
                                                         "float16,float16,float16,"
                                                         "bfloat16,bfloat16,bfloat16,"
                                                         "float16,float16,float16,"
                                                         "bfloat16,bfloat16,bfloat16",
                                                format="NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="res_for_gamma",
                                                datatype="float,float,float,"
                                                         "float,float,float,"
                                                         "float,float,float,"
                                                         "float,float,float,"
                                                         "float,float,float,"
                                                         "float,float,float,"
                                                         "float,float,float",
                                                format="NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND,"
                                                       "NCHW,NHWC,ND")
    else:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="dy",
                                               datatype="float16,float,float16,float16,"
                                                        "float,float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,bfloat16",
                                               format="FRACTAL_NZ,NCHW,NHWC,ND,"
                                                      "FRACTAL_NZ,NCHW,NHWC,ND,"
                                                      "FRACTAL_NZ,NCHW,NHWC,ND")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16,float,float16,float16,"
                                                        "float,float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,bfloat16",
                                               format="FRACTAL_NZ,NCHW,NHWC,ND,"
                                                      "FRACTAL_NZ,NCHW,NHWC,ND,"
                                                      "FRACTAL_NZ,NCHW,NHWC,ND")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="rstd",
                                               datatype="float16,float,float16,float16,"
                                                        "float,float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,bfloat16",
                                               format="ND,NCHW,NHWC,ND,"
                                                      "ND,NCHW,NHWC,ND,"
                                                      "ND,NCHW,NHWC,ND")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="mean",
                                               datatype="float16,float,float16,float16,"
                                                        "float,float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,bfloat16",
                                               format="ND,NCHW,NHWC,ND,"
                                                      "ND,NCHW,NHWC,ND,"
                                                      "ND,NCHW,NHWC,ND")
        input4 = util_select_op_base.gen_param(classify="input4",
                                               name="gamma",
                                               datatype="float16,float,float16,float16,"
                                                        "float,float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16,bfloat16",
                                               format="ND,NCHW,NHWC,ND,"
                                                      "ND,NCHW,NHWC,ND,"
                                                      "ND,NCHW,NHWC,ND")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="pd_x",
                                                datatype="float16,float,float16,float16,"
                                                         "float,float,float,float,"
                                                         "bfloat16,bfloat16,bfloat16,bfloat16",
                                                format="FRACTAL_NZ,NCHW,NHWC,ND,"
                                                       "FRACTAL_NZ,NCHW,NHWC,ND,"
                                                       "FRACTAL_NZ,NCHW,NHWC,ND")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="res_for_gamma",
                                                datatype="float,float,float,float,"
                                                         "float,float,float,float,"
                                                         "float,float,float,float",
                                                format="FRACTAL_NZ,NCHW,NHWC,ND,"
                                                       "FRACTAL_NZ,NCHW,NHWC,ND,"
                                                       "FRACTAL_NZ,NCHW,NHWC,ND")

    param_list = [input0, input1, input2, input3, input4, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_dynamic_format(shape_dy, shape_gamma, c_0):
    """
    check dynamic format branch

    """
    if len(shape_dy) < 2 or len(shape_gamma) != 1:
        return True

    # Format of NZ is not supported when the size of reduce axis is larger than 4096
    reduce_axis_threshold = 4096
    if shape_dy[-1] >= reduce_axis_threshold:
        return True

    if shape_dy[-1] % c_0 != 0 or shape_dy[-2] % c_0 != 0 \
            or shape_gamma[-1] % c_0 != 0:
        return True
    return False


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
    get placeholders of data_dy, data_x, data_rstd, data_mean and data_gamma

    Parameters
    ----------
    shapes: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_rstd": shape_rstd,
         "shape_mean": shape_mean, "shape_gamma": shape_gamma}
    dtype: str
        the data type

    Returns
    -------
    data_gm: tuple
        (data_dy, data_x, data_rstd, data_mean, data_gamma)
    """
    data_dy = tvm.placeholder(shapes.get("shape_dy"), name="data_dy", dtype=dtype)
    data_x = tvm.placeholder(shapes.get("shape_x"), name="data_x", dtype=dtype)
    data_rstd = tvm.placeholder(shapes.get("shape_rstd"), name="data_rstd", dtype=dtype)
    data_mean = tvm.placeholder(shapes.get("shape_mean"), name="data_mean", dtype=dtype)
    data_gamma = tvm.placeholder(shapes.get("shape_gamma"), name="data_gamma", dtype=dtype)

    data_gm = (data_dy, data_x, data_rstd, data_mean, data_gamma)

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
        `(data_x - data_mean)*np.power((data_rstd + EPSLON), (-0.5))`
    """
    data_gamma_cast = tbe.broadcast(data.get("data_gamma"), shape_x)
    pd_xl = tbe.vmul(data_gamma_cast, data.get("data_dy"))

    var_elta_2 = data.get("data_rstd")
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
        `(data_x - data_mean)*np.power((data_rstd + EPSLON), (-0.5))`
    """
    pd_x, res_for_gamma = _get_pd_x(data, params, shape_x, dtype, cast_dtype)

    return pd_x, res_for_gamma


# 'pylint: disable=too-many-arguments
# 'pylint: disable=huawei-too-many-arguments
def _get_pds(data_dy, data_x, data_rstd, data_mean,
             data_gamma, shape_gamma_ori, params):
    """
    get params and data, compute pd_x, pd_gamma, pd_beta.

    Parameters
    ----------
    data_dy: TVM tensor
        the placeholder of dy input data
    data_x: TVM tensor
        the placeholder of x input data
    data_rstd: TVM tensor
        the placeholder of rstd input data
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
        `(data_x - data_mean)*np.power((data_rstd + EPSLON), (-0.5))`
    """
    dtype = data_dy.dtype.lower()
    shape_x = shape_util.shape_to_list(data_x.shape)

    has_improve_precision = False
    cast_dtype = dtype
    if dtype == "bfloat16" or (dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32")):
        has_improve_precision = True
        cast_dtype = "float32"

    if has_improve_precision:
        data_dy = tbe.cast_to(data_dy, "float32")
        data_x = tbe.cast_to(data_x, "float32")
        data_rstd = tbe.cast_to(data_rstd, "float32")
        data_mean = tbe.cast_to(data_mean, "float32")
        data_gamma = tbe.cast_to(data_gamma, "float32")

    data = {"data_dy": data_dy, "data_x": data_x,
            "data_rstd": data_rstd,
            "data_mean": data_mean, "data_gamma": data_gamma}

    pd_x, res_for_gamma = _get_res(data, params, shape_x, dtype, cast_dtype)

    return pd_x, res_for_gamma


def _update_shape_nz(shape_x, shape_rstd, shape_gamma):
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
    shape_rstd_nz = []
    len_var = len(shape_rstd)
    var_nz_begin = len_var - 2
    for i in range(0, var_nz_begin):
        shape_rstd_nz.append(shape_rstd[i])
    shape_rstd_nz.append(1)
    shape_rstd_nz.append(shape_x[nz_begin + 1])
    shape_rstd_nz.append(shape_x[nz_begin + 2])
    shape_rstd_nz.append(1)

    # ND shape of gamma is one dim
    shape_gamma_nz = []
    for i in range(0, nz_begin):
        shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin])
    shape_gamma_nz.append(1)
    shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin + 2])

    reduce_nz_axis = []
    for i, (xtem, var) in enumerate(zip(shape_x_nz, shape_rstd_nz)):
        if xtem != 1 and var == 1:
            reduce_nz_axis.append(i)

    all_axis = set(range(len(shape_x_nz)))
    param_axis = tuple(all_axis - set(reduce_nz_axis))

    param_nz = {
        "shape_x_nz": shape_x_nz,
        "shape_rstd_nz": shape_rstd_nz,
        "shape_gamma_nz": shape_gamma_nz,
        "reduce_axis": reduce_nz_axis,
        "param_axis": param_axis
    }

    return param_nz


# 'pylint: disable=too-many-arguments
# 'pylint: disable=huawei-too-many-arguments
def layer_norm_x_backprop_v3_compute(input_dy, input_x,
                                     input_rstd, input_mean,
                                     input_gamma, output_pd_x, output_res_gamma,
                                     params, kernel_name="layer_norm_x_backprop_v3"):
    """
    DSL description of the layernorm_grad operator's mathematical calculation process

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32, bfloat16
    input_x: dict
        shape and dtype of input x, only support float16, float32, bfloat16
    input_rstd: dict
        shape and dtype of input rstd, only support float16, float32, bfloat16
    input_mean: dict
        shape and dtype of input mean, only support float16, float32, bfloat16
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32, bfloat16
    output_pd_x: dict
        shape and dtype of output, only support float16, float32, bfloat16
    output_res_gamma: dict
        shape and dtype of output for gamma, only support float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v3"

    Returns
    -------
    res_tuple: tuple
        (pd_x, res_for_gamma)
    """
    pd_x, res_for_gamma = _get_pds(input_dy, input_x, input_rstd, input_mean, input_gamma,
                                   input_gamma.shape, params)
    res_list = [pd_x, res_for_gamma]

    return res_list


def shape_check(dy, x, rstd, mean, gamma, pd_x, res_gamma):
    format_dy = dy.get("format")
    if format_dy.upper() == "FRACTAL_NZ":
        return True
    if is_unknown_rank_input((dy, x, rstd, mean, gamma, pd_x, res_gamma)):
        return True
    if is_dynamic_input((dy, x, rstd, mean, gamma, pd_x, res_gamma)):
        return True
    shape_dy = dy.get("shape")
    shape_x = x.get("shape")
    shape_rstd = rstd.get("shape")
    shape_mean = mean.get("shape")
    shape_gamma = gamma.get("shape")
    shape_pd_x = pd_x.get("shape")
    shape_res_gamma = res_gamma.get("shape")
    
    if not (shape_dy == shape_x == shape_pd_x == shape_res_gamma):
        raise RuntimeError("shape_dy, shape_x, shape_pd_x, shape_res_gamma must be the same")
    if shape_rstd != shape_mean:
        raise RuntimeError("shape_rstd, shape_mean must be the same")
    # get reduce axis
    shape_gamma = gamma.get("shape")
    start_idx = len(shape_dy) - len(shape_gamma)
    reduce_axis = list(range(start_idx, len(shape_dy)))
    # check rstd/mean shape is [A1,A2,...,An,1,...,1] pattern
    if len(shape_rstd) != len(shape_dy):
        raise RuntimeError("len(shape_rstd) != len(shape_dy)")
    for i, v in enumerate(shape_rstd):
        if (i not in reduce_axis) and (v != shape_dy[i]):
            raise RuntimeError(f"shape_rstd[{i}] != shape_dy[{i}]")
        if (i in reduce_axis) and (v != 1):
            raise RuntimeError(f"shape_rstd[{i}] != 1")
        if (i in reduce_axis) and (shape_gamma[i - start_idx] != shape_dy[i]):
            raise RuntimeError(f"shape_gamma[{i - start_idx}] != shape_dy[{i}]")
    return True


# 'pylint: disable=huawei-too-many-arguments
@register_operator("LayerNormXBackpropV3", "Norm")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def layer_norm_x_backprop_v3(input_dy, input_x, input_rstd, input_mean,
                             input_gamma, output_pd_x, output_res_gamma,
                             kernel_name="layer_norm_x_backprop_v3"):
    """
    algorithm: layernorm_x_backprop_v3
    calculating: gradient of layernorm
                 compute partial derivation of x, gamma and beta
        `pd_xl    = data_dy * data_gamma`
        `pd_var   = np.sum(((-0.5) * pd_xl * (data_x - data_mean) * np.power(data_rstd, `
                    `3)), reduce_axis, keepdims=True)`
        `pd_mean  = np.sum(((-1.0) * pd_xl * data_rstd),`
                   `reduce_axis, keepdims=True)`
                   `+ pd_var * (1.0 / m) * np.sum(((-2.0) * (data_x - data_mean)),`
                   `reduce_axis, keepdims=True)`
        `pd_x     = pd_xl * data_rstd`
                   `+ pd_var * (2.0 / m) * (data_x - data_mean) + pd_mean * (1.0 / m)`
        `res_for_gamma = (data_x - data_mean) * data_rstd`

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32, bfloat16
    input_x: dict
        shape and dtype of input x, only support float16, float32, bfloat16
    input_rstd: dict
        shape and dtype of input rstd, only support float16, float32, bfloat16
    input_mean: dict
        shape and dtype of input mean, only support float16, float32, bfloat16
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32, bfloat16
    output_y: dict
        shape and dtype of output, only support float16, float32, bfloat16
    output_res_gamma: dict
        shape and dtype of output for gamma, only support float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v3"

    Returns
    -------
    None
    """
    check_ret = shape_check(input_dy, input_x, input_rstd, input_mean, input_gamma,
                            output_pd_x, output_res_gamma)
    if not check_ret:
        raise RuntimeError("layer_norm_x_backprop_v3 shape check failed!")
    dtype = input_dy.get("dtype").lower()
    rstd_dtype = input_rstd.get("dtype").lower()
    gamma_dtype = input_gamma.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    shape_rstd = input_rstd.get("shape")
    shape_gamma = input_gamma.get("shape")
    format_dy = input_dy.get("format")
    extra_params = {"input_shape_type": [0, 0, 1, 1, 1],
                    "same_input_shape_group": [[0, 1], [2, 3]]}

    if is_unknown_rank_input((input_dy, input_x, input_rstd, input_mean, input_gamma)):
        reduce_axis = NormPattern.REDUCE_UNKNOWN_MODE
        broadcast_axis = NormPattern.BROADCAST_UNKNOWN_MODE
        extra_params.update(NormPattern.REDUCE_AFTER_TYPE)
        extra_params.update({"compile_broadcast_axes": {2: reduce_axis, 3: reduce_axis,
                                                        4: broadcast_axis}})
        extra_params.update({"broadcast_axes_type": {2: "same_reduce", 3: "same_reduce",
                                                     4: "opposite_reduce"}})
        operation.add_compile_info("unknown_mode", True)

        ins = classify([input_dy, input_x, input_rstd, input_mean, input_gamma,
                        reduce_axis], OpPatternMode.NORM, extra_params)
    else:
        if format_dy.upper() == "FRACTAL_NZ":
            params = _update_shape_nz(shape_x, shape_rstd, shape_gamma)
            input_dy["shape"] = params.get("shape_x_nz")
            input_x["shape"] = params.get("shape_x_nz")
            input_rstd["shape"] = params.get("shape_rstd_nz")
            input_mean["shape"] = params.get("shape_rstd_nz")
            input_gamma["shape"] = params.get("shape_gamma_nz")

            for input_tensor in (input_dy, input_x, input_rstd, input_mean, input_gamma):
                nz_range = [(1, None)] * len(params.get("shape_x_nz"))
                input_tensor["range"] = nz_range
        else:
            params = _get_params(shape_x, shape_rstd, shape_gamma)

        extra_params.update({"compile_broadcast_axes": {2: params.get("reduce_axis"),
                                                        3: params.get("reduce_axis"),
                                                        4: params.get("param_axis")}})

        ins = classify([input_dy, input_x, input_rstd, input_mean, input_gamma,
                        params.get("reduce_axis")], OpPatternMode.NORM, extra_params)

    schedules = []
    tensors = []
    for (ins_dy, ins_x, ins_rstd, ins_mean, ins_gamma, ins_reduce_axis) in ins:
        with tbe.compute():
            shape_dy, shape_x, shape_rstd, shape_mean, shape_gamma = \
                shape_util.variable_shape([ins_dy, ins_x, ins_rstd, ins_mean, ins_gamma],
                                          op_mode="norm")

            data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype)
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype)
            data_rstd = tvm.placeholder(shape_rstd, name="data_rstd", dtype=rstd_dtype)
            data_mean = tvm.placeholder(shape_mean, name="data_mean", dtype=rstd_dtype)
            data_gamma = tvm.placeholder(shape_gamma, name="data_gamma", dtype=gamma_dtype)

            if is_unknown_rank_input((input_dy, input_x, input_rstd,
                                      input_mean, input_gamma)):
                mean_cof = tbe.var("mean_cof", dtype="float32")
                mean_cof2 = tbe.var("mean_cof_double", dtype="float32")
                operation.add_compile_info("reduce_mean_cof", True)
                params = {"reduce_axis": ins_reduce_axis,
                          "mean_num": [mean_cof, mean_cof2]}
            else:
                _get_params_after_classify(shape_x, ins_reduce_axis, params)

            res_list = layer_norm_x_backprop_v3_compute(data_dy, data_x, data_rstd,
                                                        data_mean, data_gamma, output_pd_x,
                                                        output_res_gamma, params)
            tensor_list = [data_dy, data_x, data_rstd, data_mean, data_gamma] + list(res_list)
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res_list)
        schedules.append(sch)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
