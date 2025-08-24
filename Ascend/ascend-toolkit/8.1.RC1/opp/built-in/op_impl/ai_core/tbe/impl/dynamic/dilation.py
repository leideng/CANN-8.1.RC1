#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
dilation
"""
import collections

from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util import util_select_op_base


# 2 means L1 enable
L1FUSION_INPUT_CTR = 2

InputParams = collections.namedtuple('InputParams', ['shape_x', 'dilations', 'pads', 'padding_value'])


def get_op_support_info(x: dict, y: dict, dilations: list, pads: list = None,
                        padding_value: float = 0.0, kernel_name: str = "dilation") -> str:
    """
    the split formation of dilation
    :param x: dict, input
    :param y: dict, output
    :param dilations: list or tuple, dilate
    :param pads: list or tuple or None, pads of input after dilate
    :param padding_value: float, the pad value
    :param kernel_name
    """
    n_axis = 0
    head_overlap_n = -1
    tail_overlap_n = head_overlap_n
    axis_split_matrix = [
        # cut N
        [util_select_op_base.SplitInput([0, [n_axis], [head_overlap_n], [tail_overlap_n]]),
         util_select_op_base.SplitOutput([0, [n_axis]])]
    ]
    axis_reduce_list = None
    l1_mini_size = 0
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, l1_mini_size)

    return op_cal_info_in_json


def _param_check(x, dilations):
    shape_x = x.get("shape")
    if len(shape_x) != len(dilations):
        args_dict = {
            "errCode": "E60002",
            "attr_name": "dim",
            "param1_name": "x",
            "param2_name": "dilations",
            "param1_value": "{}".format(len(shape_x)),
            "param2_value": "{}".format(len(dilations))
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    check_list = ("float16", "float32")
    para_check.check_dtype(x.get("dtype"), check_list, param_name="x")


def _infer_dynamic_shape():
    fmap_batch = operation.var("fmap_batch")
    fmap_c1 = operation.var("fmap_c1")
    fmap_h = operation.var("fmap_h")
    fmap_w = operation.var("fmap_w")
    fmap_c0 = operation.var("fmap_c0")
    dilations_batch = operation.var("dilations_batch")
    dilations_c1 = operation.var("dilations_c1")
    dilations_h = operation.var("dilations_h")
    dilations_w = operation.var("dilations_w")
    dilations_c0 = operation.var("dilations_c0")

    shape_x = (fmap_batch, fmap_c1, fmap_h, fmap_w, fmap_c0)
    dilations = (dilations_batch, dilations_c1, dilations_h, dilations_w, dilations_c0)
    pads = (operation.var("padu"), operation.var("padd"), operation.var("padl"), operation.var("padr"))
    padding_value = operation.var("padding_value")
    input_params = InputParams(shape_x, dilations, pads, padding_value)
    return input_params


def _define_tiling_var():
    operation.var("core_use")
    operation.var("w_split_npart")
    operation.var("h_split_npart")
    operation.var("ub_input_bound")
    operation.var("ub_output_bound")


@tbe_register.register_operator('Dilation')
def dilation(x, y, dilations, pads=None, padding_value=0.0, kernel_name="dilation"):
    """
    dilation, expand dims of x according to parameter dilations, for example:
    when x shape is [1, 1, 2, 2, 16], dilations = [1, 1, 2, 2, 1], output is [1, 1, 3, 3, 16]
    :param x: dict, input
    :param y: dict, output
    :param dilations: list or tuple, only supported h and w dilation now
    :param pads: list or tuple or None
    :param padding_value: float
    :param kernel_name
    """
    _param_check(x, dilations)
    dtype_x = x.get("dtype")
    shape_x, dilations, pads, padding_value = _infer_dynamic_shape()
    _define_tiling_var()
    shape_x_fuse = [shape_x[0] * shape_x[1], *shape_x[2:]]
    dilations_fuse = [dilations[0] * dilations[1], *dilations[2:]]
    tensor_x = tvm.placeholder(shape_x_fuse, dtype=dtype_x, name="x")

    with tbe.compute():
        dilated_x = tbe.dilation(tensor_x, dilations_fuse, pads, padding_value)
    with tvm.target.cce():
        s = tbe.auto_schedule(dilated_x)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": (tensor_x, dilated_x)
    }
    tbe.build(s, config)
