#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
add
"""
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils.error_manager import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util import util_common
from impl.dynamic.add import static_reshape
from impl.dynamic.add import calc_input_tensor
from impl.dynamic.add import add_compute_for_batchmatmul
from impl.dynamic.add import op_select_format as add_op_select_format


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
def op_select_format(input_x, input_y, output_z, kernel_name="add"):
    """
    select format dynamically \n
    op_select_format support desc:

    1.when input x's ori_shape is 4, and bias's shape is not 1. \n
    The Op Bias can support
    ND + ND = ND,
    NC1HWC0 + NC1HWC0 = NC1HWC0.

        for example:
        inputs:
            x        ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"
            bias     ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"
        outputs:
            y        ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"

    2.In other scenes, all input(x, bias) only support ND.

        for example:
        inputs:
            x        ori shape = [2] ori_format = "ND"
            bias     ori shape = [2] ori_format = "ND"
        outputs:
            y        ori shape = [2] ori_format = "ND"

    """
    return add_op_select_format(input_x, input_y, output_z, kernel_name)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("add", op_mode="static", support_fusion=True)
def add_compute(input_x, input_y, output_z, is_scene_1d=False, broadcast_flag=True, kernel_name="add"):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    is_scene_1d: bool
        is scene 1d
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : output of the data's add
    """
    x_dtype = input_x.dtype.lower()
    y_dtype = input_y.dtype.lower()
    is_mix_dtype = x_dtype != y_dtype

    if x_dtype in ("uint8", "int8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    if is_mix_dtype:
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")

    if is_scene_1d:
        shape_x = shape_util.shape_to_list(input_x.shape)
        shape_y = shape_util.shape_to_list(input_y.shape)
        if shape_x != shape_y:
            if broadcast_flag:
                # if shape not equal, then apply broadcast.
                shape_x, shape_y, shape_max = para_check.produce_shapes(shape_x, shape_y)
                input_x = tbe.broadcast(input_x, shape_max)
                input_y = tbe.broadcast(input_y, shape_max)
            else:
                input_y = tbe.broadcast(input_y, shape_x)
    else:
        batchmatmul_flag, input_x, input_y = calc_input_tensor(input_x, input_y)
        if batchmatmul_flag:
            return add_compute_for_batchmatmul(input_x, input_y)

    res = tbe.vadd(input_x, input_y)

    if x_dtype in ("uint8", "int8"):
        res = util_common.uint8_int8_overflow_proc(res, x_dtype)

    output_dtype = output_z.get("dtype")
    if res.dtype != output_dtype:
        res = tbe.cast_to(res, output_dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def add(input_x, input_y, output_z, kernel_name="add"):
    """
    algorithm: add
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x : dict
        shape and dtype of first input, only support float16, float32, int32, int8, uint8
    input_y : dict
        shape and dtype of second input, only support float16, float32, int32, int8, uint8
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is add

    Returns
    -------
    None
    """
    # check dtype
    check_tuple = ("float16", "float32", "int32", "int8", "uint8")
    dtype_x = input_x.get("dtype").lower()
    dtype_y = input_y.get("dtype").lower()
    dtype_out = output_z.get("dtype").lower()

    mix_dtype_list = (("float16", "float32", "float32"), ("float32", "float16", "float32"))
    is_valid_mix_dtpye = (dtype_x, dtype_y, dtype_out) in mix_dtype_list

    if not is_valid_mix_dtpye and dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'x', 'y', dtype_x, dtype_y)
    para_check.check_dtype(dtype_x, check_tuple, param_name="input_x")

    vadd_support = tbe_platform.api_check_support("te.lang.cce.vadd", "float32")
    if not vadd_support:
        new_check_list = list(check_tuple)
        new_check_list.remove("float32")
        para_check.check_dtype(dtype_x, new_check_list, param_name="input_x")
        para_check.check_dtype(dtype_y, new_check_list, param_name="input_y")

    broadcast_flag, is_scene_1d = None, None
    shape_x, shape_y, broadcast_flag, is_scene_1d = static_reshape(input_x, input_y)

    data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_1")
    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data_2")
    res = add_compute(data_x, data_y, output_z, is_scene_1d, broadcast_flag, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": (data_x, data_y, res)}
    build(schedule, config)
