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
grid_unnormal
"""

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import is_cast_support
from impl.util.platform_adapter import inernal_cast


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("grid_unnormal", op_mode="static", support_fusion=True)
def grid_unnormal_compute(grid, assist, diff, position, align_corners=False, kernel_name="grid_unnormal"):
    """
    algorithm: unnormal grid data
    Parameters
    ----------
    grid : TVM tensor
        the placeholder of grid
    assist : TVM tensor
        the placeholder of assist
    diff: dict
        shape and dtype of output, only support float16, float32
    position: dict
        shape and dtype of output, only support int32
    align_corners : bool.
        An optional bool. If "true", the centers of the corner pixels of
        the input and output tensors are aligned. Defaults to "false" .
    kernel_name : str
        cce kernel name, default value is grid_unnormal

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    grid_tmp1 = tbe.vadds(grid, 1)
    grid_tmp2 = tbe.vmuls(grid_tmp1, 0.5)

    if align_corners:
        input_size = tbe.vadds(assist, -1)
        pos_base = tbe.vmul(grid_tmp2, input_size)
    else:
        tmp1 = tbe.vmul(grid_tmp2, assist)
        pos_base = tbe.vadds(tmp1, -0.5)

    res_pos = tbe.floor(pos_base)
    res_diff = tbe.vsub(pos_base, res_pos)
    return [res_diff, res_pos]


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def grid_unnormal(grid, assist, diff, position, align_corners=False, kernel_name="grid_unnormal"):
    """
    algorithm: unnormal grid data
    Parameters
    ----------
    grid : dict
        shape and dtype of first input, only support float16, float32
    assist : dict
        shape and dtype of second input, only support float16, float32
    diff: dict
        shape and dtype of output, only support float16, float32
    position: dict
        shape and dtype of output, only support int32
    align_corners : bool.
        An optional bool. If "true", the centers of the corner pixels of
        the input and output tensors are aligned. Defaults to "false" .
    kernel_name : str
        cce kernel name, default value is grid_unnormal

    Returns
    -------
    None
    """
    data_grid = tvm.placeholder(grid.get("shape"), dtype=grid.get("dtype"), name="data_grid")
    data_assist = tvm.placeholder(assist.get("shape"), dtype=assist.get("dtype"), name="data_assist")

    res_list = grid_unnormal_compute(data_grid, data_assist, diff, position, align_corners, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = auto_schedule(res_list)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_grid, data_assist, res_list[0], res_list[1]]}
    build(schedule, config)
