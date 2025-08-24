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
ascend_requant_s16
"""
from functools import reduce as function_reduce
from collections import namedtuple
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl import ascend_quant_util as util


# 'pylint: disable=invalid-name,unused-argument,unnecessary-lambda,too-many-arguments,too-many-locals
@register_operator_compute("AscendRequantS16", op_mode="dynamic", support_fusion=True)
def ascend_requant_s16_compute(x0, req_scale, x1, y0, y1, dual_output, relu_flag, kernel_name='ascend_requant_s16'):
    """
    int16 -> int8

    Parameters:
    ----------
    x0: the placeholder of input
    req_scale: the placeholder of req_scale
    x1: the placeholder of x1
    y0: the dict of output.
    y1: the dict of output1.
    dual_output: dual output flag, default value is False
    relu_flag: the relu mode, default value is False
    kernel_name: cce kernel name, default value is "ascend_requant_s16"

    Returns:
    -------
    res : the result of ascend_requant_s16 which is list
    """
    x_shape = x0.shape
    x_shape_list = shape_util.shape_to_list(x_shape)
    align_shape = x_shape_list.copy()

    ori_shape_req = req_scale.op.attrs["ori_shape"]
    ori_shape_req_list = shape_util.shape_to_list(ori_shape_req)
    req_dim = function_reduce(lambda x, y: x * y, ori_shape_req_list[:])
    tensor_flag = False
    if req_dim > 1:
        tensor_flag = True

    c1_index = 1
    if util.is_nz_format(x0):
        c1_index = len(x_shape) - 4

    align_shape[c1_index] = (align_shape[c1_index] + 1) // 2 * 2
    compute_input = namedtuple('NormalCompute', "req_scale x_shape align_shape c1_index tensor_flag relu_flag")
    res_s16, res_ub = _s16_to_s8_normal_compute(
        x0, x1, compute_input(req_scale, x_shape, align_shape, c1_index, tensor_flag, relu_flag)
        )

    res = _format_transfer(align_shape, res_ub, c1_index)
    if util.is_nz_format(x0):
        res = tvm.compute(align_shape, lambda *i: res[i], name="res", tag="requant_s16_NZ")

    if dual_output:
        return [res, res_s16]

    return [res]


def _s16_to_s8_normal_compute(x, x1, compute_input):
    """
    generate s16_to_s8 compute
    """
    req_scale, x_shape, align_shape, c1_index, tensor_flag, relu_flag = compute_input
    if x1 is not None:
        if relu_flag:
            res_s16 = tvm.compute(x_shape, lambda *indices: tvm.relu(x(*indices) + x1(*indices)),
                                  name="res_s16", tag="requant_s16_vaddrelu")
        else:
            res_s16 = tvm.compute(x_shape, lambda *indices: x(*indices) + x1(*indices),
                                  name="res_s16", tag="requant_s16_vadd")
    else:
        if relu_flag:
            res_s16 = tvm.compute(x_shape, lambda *indices: tvm.relu(x(*indices)), name="res_s16",
                                  tag="requant_s16_relu")
        else:
            res_s16 = tvm.compute(x_shape, lambda *indices: x(*indices), name="res_s16", tag="requant_s16")
    x_shape_list = shape_util.shape_to_list(x_shape)
    deq_cast_input = namedtuple('CastCompute', "res_s16 req_scale align_shape c1_index tensor_flag x_shape_list")
    if tensor_flag:
        res_ub = tvm.compute(align_shape,
                             _deq_cast_compute(
                                 deq_cast_input(res_s16, req_scale, align_shape, c1_index, tensor_flag, x_shape_list)
                                 ),
                             name="s16_to_s8", tag="requant_s16_vector")
    else:
        res_ub = tvm.compute(align_shape,
                             _deq_cast_compute(
                                 deq_cast_input(res_s16, req_scale, align_shape, c1_index, tensor_flag, x_shape_list)
                                 ),
                             name="s16_to_s8", tag="requant_s16_scale")
    return res_s16, res_ub


def _deq_cast_compute(deq_cast_input):
    """
    generate lambda func
    """
    res_s16, req_scale, align_shape, c1_index, tensor_flag, x_shape_list = deq_cast_input
    n_dim = len(align_shape)
    c0_index = n_dim - 1

    def lambda_func(*indice):
        new_indice = [0] * 5
        if tensor_flag:
            new_indice[4] = indice[c0_index]
            new_indice[1] = indice[c1_index]

        return tvm.select(indice[c1_index] < x_shape_list[c1_index],
                          tvm.conv_vdeq(res_s16(*indice), req_scale(*new_indice)).astype("int8"),
                          tvm.const(0, dtype="int8"))

    return lambda_func


def _format_compute(tensor, trans_shape, c1_index):
    """
    generate lambda func
    """
    n_dim = len(trans_shape)
    c0_index = n_dim - 1

    def lambda_func(*indice):
        new_indice = [0] * n_dim
        for i in range(n_dim):
            if i == c0_index:
                new_indice[i] = (indice[c1_index] * 32 + indice[c0_index]) % 16
            elif i == c1_index:
                new_indice[i] = (indice[c1_index] * 32 + indice[c0_index]) // 16
            else:
                new_indice[i] = indice[i]
        return tensor(*new_indice)

    return lambda_func


def _format_transfer(shape, x, c1_index):
    """
    C0 from 16 to 32 for NC1HWC0
    """
    trans_shape = shape[:]
    trans_shape[c1_index] = trans_shape[c1_index] // 2
    trans_shape[-1] = trans_shape[-1] * 2
    res = tvm.compute(trans_shape,
                      _format_compute(x, trans_shape, c1_index),
                      name="data_transfer",
                      tag="requant_s16_data_transfer")
    return res


def _check_params(x0, req_scale):
    """
    check the parameters including dtype, kernel_name, attr
    """
    format_x = x0.get("format")
    dtype_x = x0.get("dtype")

    format_req = req_scale.get("format")
    dtype_req = req_scale.get("dtype")

    check_list = [("int16",), ("uint64",)]
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    para_check.check_dtype(dtype_x, check_list[0], param_name="x")
    para_check.check_dtype(dtype_req, check_list[1], param_name="req_scale")
    para_check.check_format(format_x, format_list, param_name="x")

    para_check.check_format(format_req, ("NC1HWC0",), param_name="req_scale")


@register_operator("AscendRequantS16", pattern="requant_s16")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ascend_requant_s16(x0, req_scale, x1, y0, y1, dual_output=False, relu_flag=False, kernel_name="ascend_requant_s16"):
    """
    int16 -> int8

    Parameters:
    ----------
    x0: the placeholder of input
    req_scale: the placeholder of req_scale
    x1: the placeholder of x1
    y0: the dict of output.
    y1: the dict of output1.
    dual_output: dual output flag, default value is False
    relu_flag: the relu mode, default value is False
    kernel_name: cce kernel name, default value is "ascend_requant_s16"

    Returns:
    -------
    None
    """

    _check_params(x0, req_scale)

    schedules, tensors = [], []
    with tbe.compute():
        shape_x = x0.get("shape")
        format_x = x0.get("format")
        if format_x == "NC1HWC0":
            x0_n = operation.var("x0_n")
            x0_c1 = operation.var("x0_c1")
            x0_hw = operation.var("x0_hw")

            shape_x = []
            shape_x.append(x0_n)
            shape_x.append(x0_c1)
            shape_x.append(x0_hw)
            shape_x.append(16)

        x0_dtype = x0.get("dtype")
        input_x0 = tvm.placeholder(shape_x, x0_dtype, "x0")

        shape_req = req_scale.get("shape")
        dtype_req = req_scale.get("dtype")
        ori_shape_deq = req_scale.get("ori_shape")
        attr = {"ori_shape": ori_shape_deq}
        input_req = tvm.placeholder(shape_req, name="req_scale", dtype=dtype_req, attrs=attr)

        input_x1 = None
        if x1:
            input_x1 = tvm.placeholder(shape_x, "int16", "x1")

        res = ascend_requant_s16_compute(input_x0, input_req, input_x1, y0, y1, dual_output, relu_flag, kernel_name)

        tensors.append([input_x0, input_req, input_x1, res])

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
