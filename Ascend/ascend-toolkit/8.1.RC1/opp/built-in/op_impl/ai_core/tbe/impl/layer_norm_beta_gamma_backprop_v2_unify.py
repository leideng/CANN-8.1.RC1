#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
layer_norm_beta_gamma_backprop_v2_unify
"""
import tbe.common.register as tbe_register
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tuple_sum
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context


# 'pylint: disable = too-few-public-methods
class Constant:
    """
    common constants
    """
    DIM_TWO = 2
    DIM_FOUR = 4


def _update_gamma_shape(shape_gamma):
    """
    update shape_gamma for subsequent calculation
    """
    params_axis_tmp = []
    for i, dim in enumerate(shape_gamma):
        if dim == 1:
            params_axis_tmp.append(i)
    params_axis = tuple(params_axis_tmp)
    return params_axis


def _update_gamma_shape_by_axis(param_axis, processed_param_axis, shape_gamma):
    """
    update shape_gamma for tuple_reduce pattern's axis combination
    """
    if len(param_axis) == len(processed_param_axis):
        return shape_gamma
    skip = 0
    idx = 0
    proc_idx = 0
    shape_gamma_new = []
    for val in shape_gamma:
        if val == 1:
            if proc_idx >= len(processed_param_axis):
                continue
            elif proc_idx < len(processed_param_axis) and \
                 processed_param_axis[proc_idx] + skip == param_axis[idx]:
                idx += 1
                proc_idx += 1
                shape_gamma_new.append(val)
            else:
                idx += 1
                skip += 1
        else:
            shape_gamma_new.append(val)
    return shape_gamma_new


def layer_norm_beta_gamma_backprop_v2_compute(data_dy, res_for_gamma, output_pd_gamma,
                                              output_pd_beta, shape_gamma,
                                              kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
    DSL description of the layernorm_grad operator's
    mathematical calculation process

    Parameters
    ----------
    input_dy: TVM tensor
        the placeholder of dy input data
    res_for_gamma: TVM tensor
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
    dtype = data_dy.dtype.lower()
    shape_x = shape_util.shape_to_list(res_for_gamma.shape)
    param_axis = _update_gamma_shape(shape_gamma)

    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
        dtype = "float32"

    if has_improve_precision:
        data_dy = tbe.cast_to(data_dy, "float32")
        res_for_gamma = tbe.cast_to(res_for_gamma, "float32")

    data_x = tbe.vmul(res_for_gamma, data_dy)
    pd_gamma, pd_beta = tuple_sum([data_x, data_dy], param_axis, keepdims=True)

    if dtype == "float16" and not has_improve_precision:
        pd_gamma = tbe.cast_to(pd_gamma, "float32")
        pd_beta = tbe.cast_to(pd_beta, "float32")

    res_list = [pd_gamma, pd_beta]

    return res_list


def _update_shape_nz(shape_x):
    """
    function of updating Nz shape

    """
    # ND shape of x >= two dim
    # Nz shape of x >= four dim
    len_x = len(shape_x)
    nz_begin = len_x - Constant.DIM_FOUR
    shape_x_nz = []
    for i in range(0, nz_begin):
        shape_x_nz.append(shape_x[i])
    shape_x_nz.append(shape_x[nz_begin])
    shape_x_nz.append(shape_x[nz_begin + 1])
    shape_x_nz.append(shape_x[nz_begin + Constant.DIM_TWO])
    shape_x_nz.append(shape_x[nz_begin + Constant.DIM_TWO])

    # ND shape of gamma is one dim
    shape_gamma_nz = []
    for i in range(0, nz_begin):
        shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin])
    shape_gamma_nz.append(1)
    shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin + Constant.DIM_TWO])

    param_nz_axis = []
    for i, (xtem, gamma) in enumerate(zip(shape_x_nz, shape_gamma_nz)):
        if xtem != gamma or (xtem == 1 and gamma == 1):
            param_nz_axis.append(i)

    param_nz = {"shape_x_nz": shape_x_nz, "shape_gamma_nz": shape_gamma_nz, "param_nz_axis": param_nz_axis}

    return param_nz


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def layer_norm_beta_gamma_backprop_v2_unify(input_dy, res_for_gamma, output_pd_gamma,
                                      output_pd_beta, shape_gamma,
                                      kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
    algorithm: layernorm_grad
    calculating: gradient of layernorm
                 compute partial derivation of x, gamma and beta
    pd_gamma = np.sum(data_dy*res_for_gamma, param_axis, keepdims=True)
    pd_beta  = np.sum(data_dy, param_axis, keepdims=True)

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    res_for_gamma: dict
        shape and dtype of input res_for_gamma, only support float16, float32
    output_pd_gamma: dict
        shape and dtype of output, only support float16, float32
    output_pd_beta: dict
        shape and dtype of output, only support float16, float32
    shape_gamma: list
        shape of gamma
    kernel_name: str
        cce kernel name, default value is "layer_norm_beta_gamma_backprop_v2"

    Returns
    -------
    None
    """
    dtype = input_dy.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    shape_x = res_for_gamma.get("shape")
    dtype_x = res_for_gamma.get("dtype").lower()
    format_dy = input_dy.get("format").upper()

    params = _update_shape_nz(shape_x)
    param_axis = params.get("param_nz_axis")
    input_dy["shape"] = params.get("shape_x_nz")
    res_for_gamma["shape"] = params.get("shape_x_nz")
    shape_gamma = params.get("shape_gamma_nz")
    for input_tensor in (input_dy, res_for_gamma):
        nz_range = [(1, None)] * len(params.get("shape_x_nz"))
        input_tensor["range"] = nz_range

    ins = classify([input_dy, res_for_gamma, param_axis], OpPatternMode.TUPLE_REDUCE)
    schedules = []
    tensors = []
    for (ins_dy, ins_x, ins_param_axis) in ins:
        with tbe.compute():
            shape_gamma = _update_gamma_shape_by_axis(param_axis, ins_param_axis, shape_gamma)
            shape_dy, shape_x = shape_util.variable_shape([ins_dy, ins_x], op_mode="tuple_reduce")
            data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype)
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
            res_list = layer_norm_beta_gamma_backprop_v2_compute(data_dy, data_x, output_pd_gamma,
                                                                 output_pd_beta, shape_gamma)
            tensor_list = [data_dy, data_x] + list(res_list)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res_list)
        schedules.append(sch)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
