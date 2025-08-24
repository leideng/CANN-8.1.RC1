#!/usr/bin/python
# -*- coding: utf-8 -*-
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
gn_training_update
"""
from __future__ import division

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-arguments,too-many-locals
def op_select_format(x, sum, square_sum, scale, offset, mean, variance,
                     y, batch_mean, batch_variance,
                     epsilon=0.0001, num_groups=2,
                     kernel_name="gn_training_update"):
    """
    select format dynamically
    """
    input0 = gen_param(classify="input0", name="x",
                       datatype="float16,float,float16,float",
                       format="NCHW,NCHW,NHWC,NHWC")
    input1 = gen_param(classify="input1", name="sum",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input2 = gen_param(classify="input2", name="square_sum",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input3 = gen_param(classify="input3", name="scale",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input4 = gen_param(classify="input4", name="offset",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input5 = gen_param(classify="input5", name="mean",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input6 = gen_param(classify="input6", name="variance",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")

    output0 = gen_param(classify="output0", name="y",
                        datatype="float16,float,float16,float",
                        format="NCHW,NCHW,NHWC,NHWC")
    output1 = gen_param(classify="output1", name="batch_mean",
                        datatype="float,float,float,float",
                        format="ND,ND,ND,ND")
    output2 = gen_param(classify="output2", name="batch_variance",
                        datatype="float,float,float,float",
                        format="ND,ND,ND,ND")
    param_list = [input0, input1, input2, input3, input4,
                  input5, input6,
                  output0, output1, output2]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_rule(data, rule_desc, param_name=para_check.PARAM_NAME):
    """
    The special check rule for tensor
    """
    if data is None or rule_desc is None:
        return
    error_manager_vector.raise_err_check_params_rules("gn_training_update", rule_desc, param_name, data)


def check_input_shape(shape, data_format="NCHW", num_groups=2):
    """
    check_input_shape
    """
    para_check.check_shape(shape, min_rank=4, max_rank=4, param_name="x")
    c_index = data_format.index("C")
    if shape[c_index] % num_groups != 0:
        check_rule("{} and {}".format(shape[c_index], num_groups),
                   "num_groups must divide C channel",
                   "channel and num_groups")


def check_couple_shape(shape_a, shape_b, ori_shape, data_format,
                       num_groups, first_index=False):
    """
    check_couple_shape
    """
    if first_index:
        first_value = ori_shape[0]
    else:
        first_value = 1
    para_check.check_shape(shape_a, min_rank=5, max_rank=5, param_name="x")
    para_check.check_shape(shape_b, min_rank=5, max_rank=5, param_name="x")
    if data_format == "NCHW":
        aim_shape = (first_value, num_groups, 1, 1, 1)
    else:
        aim_shape = (first_value, 1, 1, num_groups, 1)
    if tuple(shape_a) != aim_shape:
        check_rule("{} and {}".format(shape_a, aim_shape),
                   "shape_a must match with aim_shape",
                   "shape_a and aim_shape")
    if tuple(shape_b) != aim_shape:
        check_rule("{} and {}".format(shape_a, aim_shape),
                   "shape_b must match with aim_shape",
                   "shape_b and aim_shape")


@register_operator_compute("gn_training_update", op_mode="static", support_fusion=True)
def gn_training_update_compute(x,
                               scale, offset, mean, variance,
                               sum, square_sum,
                               data_format,
                               y, batch_mean, batch_variance,
                               epsilon, num_groups,
                               kernel_name="gn_training_update"):
    """
    algorithm: group_norm
    group normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    data_format: str
        data format
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    batch_variance: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "in_training_update_v2"

    Returns
    -------
    res: TVM tensor list
        the result of in_training_update_v2 compute
    """
    dtype = x.dtype
    shape = shape_util.shape_to_list(x.shape)
    if dtype == "float16":
        x = tbe.cast_to(x, "float32")
    if data_format == "NCHW":
        num = shape[2] * shape[3] * shape[4]
    else:
        num = shape[1] * shape[2] * shape[4]

    num_rec = 1.0 / num
    # compute the saved mean of x
    compute_mean = tbe.vmuls(sum, num_rec)
    mean_boardcast = tbe.broadcast(compute_mean, shape)

    # compute the saved variance of x
    variance_div = tbe.vmuls(square_sum, num_rec)
    variance_square = tbe.vmul(compute_mean, compute_mean)
    compute_var = tbe.vsub(variance_div, variance_square)

    x_mean = tbe.vsub(x, mean_boardcast)
    multiplier_add = tbe.vadds(compute_var, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    sqrt_boardcast = tbe.broadcast(multiplier_sqrt, shape)
    mean_wo_scale = tbe.vdiv(x_mean, sqrt_boardcast)
    result = mean_wo_scale
    if scale is not None and offset is not None:
        scale = tbe.broadcast(scale, shape)
        offset = tbe.broadcast(offset, shape)
        scale_scale = tbe.vmul(result, scale)
        result = tbe.vadd(scale_scale, offset)

    if dtype == "float16":
        result = tbe.cast_to(result, "float16")

    res = [result, compute_mean, compute_var]
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def gn_training_update(x, sum, square_sum,
                       scale, offset, mean, variance,
                       y, batch_mean, batch_variance,
                       epsilon=0.0001, num_groups=2,
                       kernel_name="gn_training_update"):
    """
    algorithm: group_norm
    group normalization.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    sum: dict
        dict of sum, A Tensor for sum.
        The output of instance_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A Tensor for square_sum.
        The output of instance_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A Tensor for scale.
    offset: dict
        dict of offset, A Tensor for offset.
    mean: dict
        dict of mean, A Tensor for mean.
    variance: dict
        dict of variance, A Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    batch_variance: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    epsilon: float
        A small float number added to the variance of x.
    num_groups: int
        group num
    kernel_name: str
        kernel name, default value is "gn_training_update"

    Returns
    -------
    None
    """
    data_format = x.get("format")
    para_check.check_format(data_format, ("NCHW", "NHWC"), param_name="x")

    # Process x, sum, square_sum
    shape_origin = x.get("shape")
    dtype_x = x.get("dtype")
    check_input_shape(shape_origin, data_format, num_groups)

    if data_format == "NCHW":
        shape_x = [shape_origin[0], num_groups,
                   shape_origin[1] // num_groups, shape_origin[2],
                   shape_origin[3]]

    # Reshape NHWC -> NHW[GD]
    elif data_format == "NHWC":
        shape_x = [shape_origin[0], shape_origin[1], shape_origin[2],
                   num_groups, shape_origin[3] // num_groups]

    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    check_couple_shape(shape_sum, shape_square_sum, shape_origin,
                       data_format, num_groups, True)

    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    para_check.check_dtype(dtype_sum.lower(), ("float32",), param_name="sum")
    para_check.check_dtype(dtype_square_sum.lower(), ("float32",), param_name="square_sum")

    x_input = tvm.placeholder(shape_x, name="x_input",
                              dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input",
                                dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum,
                                       name="square_sum_input",
                                       dtype=dtype_square_sum.lower())
    scale_input, offset_input, mean_input, var_input = None, None, None, None

    # Process scale and offset

    affine = False
    if scale is not None and offset is not None:
        affine = True
        shape_scale = scale.get("shape")
        dtype_scale = scale.get("dtype")
        shape_offset = offset.get("shape")
        dtype_offset = offset.get("dtype")

        check_couple_shape(shape_scale, shape_offset, shape_origin,
                           data_format,
                           num_groups)

        para_check.check_dtype(dtype_scale.lower(), ("float32",), param_name="scale")
        para_check.check_dtype(dtype_offset.lower(), ("float32",), param_name="offset")

        scale_input = tvm.placeholder(shape_scale, name="scale_input",
                                      dtype=dtype_scale.lower())
        offset_input = tvm.placeholder(shape_offset, name="offset_input",
                                       dtype=dtype_offset.lower())

    res = gn_training_update_compute(x_input, scale_input, offset_input,
                                     mean_input, var_input,
                                     sum_input, square_sum_input,
                                     data_format,
                                     y, batch_mean, batch_variance,
                                     epsilon, num_groups,
                                     kernel_name=kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    if affine:
        tensor_list = [x_input, sum_input, square_sum_input,
                       scale_input, offset_input] + list(res)
    else:
        tensor_list = [x_input, sum_input, square_sum_input] \
                      + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    build(sch, config)
