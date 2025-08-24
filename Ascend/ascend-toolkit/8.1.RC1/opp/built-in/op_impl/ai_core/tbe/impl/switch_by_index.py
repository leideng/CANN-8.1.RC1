#!/usr/bin/python
# -*- coding: utf-8 -*-
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
switch_by_index
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform


class Constant:
    """
    This class for Constant.
    """
    NUM_ONE = 1
    NUM_TWO = 2


def check_params(x1, x2):
    """
    Check whether the input info are valid.
    """
    x1_shape = x1.get("shape")
    x2_shape = x2.get("shape")
    x1_dtype = x1.get("dtype")
    x2_dtype = x2.get("dtype")

    if (len(x1_shape) != 1 or x1_shape[0] != 1):
        raise RuntimeError("Input0 shape is valid, and the dim and dim size should be 1.")
    if (len(x2_shape) != 1):
        raise RuntimeError("Input1 shape is valid, and the dim and dim size should be 1.")
    if (x1_dtype != 'int32'):
        raise RuntimeError(f"Input dtype {x1_dtype} is valid, only int32 supported.")
    if (x2_dtype != 'uint64'):
        raise RuntimeError(f"Input dtype {x2_dtype} is valid, only uint64 supported.")


# 'pylint:disable=invalid-name,too-many-locals,unused-argument,unused-variable
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.KERNEL_NAME)
def switch_by_index(x1, x2, kernel_name="switch_by_index"):
    """
    the main function of switch_by_index

    Parameters
    ----------
    input_x: dict, info of the index, datatype supports int32

    input_y: dict, shape and datatype, datatype supports uint64

    kernel_name: str
        cce kernel name, default value is "switch_by_index"

    Returns
    -------
    tik_instance: tik_instance
    """
    check_params(x1, x2)

    input_shape_x1 = x1.get("shape")
    input_shape_x2 = x2.get("shape")

    tik_instance = tik.Tik(block_size=tbe_platform.get_block_size())
    input_index = tik_instance.Tensor("int32", input_shape_x1, name="input_index_gm",
                                     scope=tik.scope_gm)
    input_index_ub = tik_instance.Tensor("int32", (Constant.NUM_ONE,), name="input_index_ub",
                                        scope=tik.scope_ubuf)
    tik_instance.data_move(input_index_ub, input_index, 0, 1, 1, 0, 0)

    index = tik_instance.Scalar(dtype="int32", name="index", init_value=input_index_ub[0])
    input_data = tik_instance.Tensor("uint64", input_shape_x2, name="input_data_gm",
                                     scope=tik.scope_gm)
    input_data_ub = tik_instance.Tensor("uint64", (Constant.NUM_TWO,), name="input_data_ub",
                                        scope=tik.scope_ubuf)
    tik_instance.data_move(input_data_ub.reinterpret_cast_to("uint32"),
                           input_data[index].reinterpret_cast_to("uint32"), 0, 1, 1, 0, 0)
    value = tik_instance.Scalar(dtype="uint64", name="value", init_value = input_data_ub[0])
    tik_instance.set_cond_task_id(value)
    tik_instance.BuildCCE(kernel_name, inputs=[input_index, input_data], outputs=[])

    return tik_instance