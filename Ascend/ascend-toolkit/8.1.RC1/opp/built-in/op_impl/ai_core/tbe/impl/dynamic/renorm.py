#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
renorm
"""

import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.util_attr_common import RenormAttrInfo
from impl.util.util_attr_common import get_attr_by_cls


@register_operator_compute("renorm", support_bfp16=True)
def renorm_compute(input_x, output_y, p, axis, maxnorm,
                   kernel_name="renorm"):
    """
    calculating logits

    if maxnorm>norm(x, p, axis, keepdim=true):
        ratio = maxnorm/norm(x, p, axis, keepdim=true)
    else:
        ratio = 1

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    p: float
        specify L_p norm
    axis: list
        the unprocessed dim
    maxnorm: float
        threshold for comparison
    kernel_name : str
        kernel name, default value is "renorm"

    Returns
    ratio
    -------

    """
    need_cast = False
    dtype = input_x.dtype
    if dtype == "float16":
        need_cast = True
        input_x = tbe.cast_to(input_x, "float32")

    const_maxnorm = get_attr_by_cls(maxnorm, RenormAttrInfo.ATTR_MAXNORM, "float32")
    ext = tvm.const(1e-7, "float32")

    if isinstance(p, float):
        if math.isclose(p, 2.0):
            x_square = tbe.vmul(input_x, input_x)
            x_square_sum = tbe.reduce_sum(x_square, axis, keepdims=True)
            x_l2norm_sqrt = tbe.vsqrt(x_square_sum)
            x_l2norm = tbe.vmins(x_l2norm_sqrt, const_maxnorm)
            ratio = tbe.vdiv(x_l2norm, tbe.vmaxs(x_l2norm_sqrt, ext))
        elif math.isclose(p, 1.0):
            x_sum = tbe.reduce_sum(tbe.vabs(input_x), axis, keepdims=True)
            x_l1norm = tbe.vmins(x_sum, const_maxnorm)
            ratio = tbe.vdiv(x_l1norm, tbe.vmaxs(x_sum, ext))
        else:
            if math.isclose(p, 0.0):
                zero_scalar = tvm.const(0, 'float32')
                one_scalar = tvm.const(1, 'float32')
                tmp_tensor = tbe.vcmpsel(input_x, one_scalar, 'ne', one_scalar, one_scalar)
                x_tmp = tbe.cast_to(tbe.reduce_sum(tmp_tensor, axis, keepdims=True), 'float32')
            else:
                p_log = tbe.vlog(tbe.vabs(input_x))
                p_mul = tbe.vmuls(p_log, p)
                x_sum = tbe.vexp(p_mul)
                x_psum = tbe.reduce_sum(x_sum, axis, keepdims=True)
                p_log_v = tbe.vlog(x_psum)
                p_mul_v = tbe.vmuls(p_log_v, 1 / p)
                x_tmp = tbe.vexp(p_mul_v)
            x_lpnorm_p = tbe.vmins(x_tmp, const_maxnorm)
            x_tmp_ext = tbe.vmaxs(x_tmp, ext)
            ratio = tbe.vdiv(x_lpnorm_p, x_tmp_ext)
    if isinstance(p, dict):
        cof = tbe.var("cof", dtype="float32")
        p_log = tbe.vlog(tbe.vabs(input_x))
        p_mul = tbe.vmuls(p_log, cof)
        x_sum = tbe.vexp(p_mul)
        x_psum = tbe.reduce_sum(x_sum, axis, keepdims=True)
        p_log_v = tbe.vlog(x_psum)
        p_mul_v = tbe.vmuls(p_log_v, 1 / cof)
        x_tmp = tbe.vexp(p_mul_v)
        x_lpnorm_p = tbe.vmins(x_tmp, const_maxnorm)
        x_tmp_ext = tbe.vmaxs(x_tmp, ext)
        ratio = tbe.vdiv(x_lpnorm_p, x_tmp_ext)
    if need_cast:
        ratio = tbe.cast_to(ratio, "float16")
    return ratio


# 'pylint: disable=not-use-list-comprehension,too-many-arguments,too-many-locals
@register_operator("Renorm")
# @para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_DICT_FLOAT,
#                             para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def renorm(input_x, output_y, p, dim, maxnorm,
           kernel_name="renorm", impl_mode="high_performance"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    p: float
        specify L_p norm
    dim: int
        the processed dim
    maxnorm: float
        threshold for comparison
    kernel_name : str
        kernel name, default value is "renorm"

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    # dtype of input must be float16,float32,bfloat16
    check_tuple = ("float16", "float32", "bfloat16")
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype(input_dtype, check_tuple)

    input_x["rel_pos_to_reduce"] = "before"
    if maxnorm is None:
        input_axis = {"shape": [-1], "rel_pos_to_reduce": "axis"}
    else:
        shape_x = input_x["shape"]
        dims = len(shape_x)
        nedims = dims * -1
        axis = [i for i in range(dims)]
        if dim is not None:
            if dim < nedims or dim > (dims - 1):
                raise RuntimeError("Only support {} <= dim <= {} while dim is {}".format(nedims, dims-1, dim))
            if dim < 0:
                dim = dim + dims
            axis.pop(dim)
        input_axis = {"shape":[len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    schedules = []
    tensors = []
    ins = classify([input_x, input_axis], OpPatternMode.REDUCE, {"keepdims": True})

    for (_x, _axis) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_x, _axis], op_mode="reduce")[0]
            input_data = tvm.placeholder(shape_var_new, name="input_data", dtype=input_dtype)
            res = renorm_compute(input_data, output_y, p, _axis["value"], maxnorm,
                                 kernel_name)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
