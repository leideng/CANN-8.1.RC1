#!/usr/bin/python
# -*- coding: utf-8 -*-
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
dynamic_gruv2_grad
"""
from tbe.common.register import register_param_generalization
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-many-arguments,unused-argument,unnecessary-pass
@register_param_generalization("DynamicGRUV2Grad")
def dynamic_gruv2_grad_generalization(x, weight_input, weight_hidden, y, init_h, h, dy, dh, update, reset, new,
                                      hidden_new, seq_length, mask, dw_input, dw_hidden, db_input, db_hidden, dx,
                                      dh_prev, direction="UNIDIRECTIONAL", cell_depth=0, keep_prob=-1.0,
                                      cell_clip=-1.0, num_proj=0, time_major=True, gate_order="zrh",
                                      reset_after=True, generalize_config=None):
    """
    only T, b support -1
    input:
    x (T, b, input_size)
    weight_input (input_size, hidden_size * 3)
    weight_hidden (hidden_size, hidden_size * 3)
    y (T, b, hidden_size)
    init_h (b, hidden_size)
    h (T, b, hidden_size)
    dy (T, b, hidden_size)
    dh (b, hidden_size)
    update (T, b, hidden_size)
    reset (T, b, hidden_size)
    new (T, b, hidden_size)
    hidden_new (T, b, hidden_size)

    seq_length (hidden_size) optional input
    mask (hidden_size) optional input

    output:
    dw_input (input_size, hidden_size * 3)
    dw_hidden (hidden_size, hidden_size * 3)
    db_input (hidden_size * 3)
    db_hidden (hidden_size * 3)
    dx (T, b, input_size)
    dh_prev (b, hidden_size)
    """

    param_input_size = x["ori_shape"][2]
    param_hidden_size = y["ori_shape"][2]

    range_x = [(1, -1), (1, -1), (param_input_size, param_input_size)]
    shape_x = (-1, -1, param_input_size)

    range_weight_input = [(weight_input["ori_shape"][0], weight_input["ori_shape"][0]),
                          (weight_input["ori_shape"][1], weight_input["ori_shape"][1])]
    shape_weight_input = (weight_input["ori_shape"][0], weight_input["ori_shape"][1])

    range_weight_hidden = [(weight_hidden["ori_shape"][0], weight_hidden["ori_shape"][0]),
                          (weight_hidden["ori_shape"][1], weight_hidden["ori_shape"][1])]
    shape_weight_hidden = (weight_hidden["ori_shape"][0], weight_hidden["ori_shape"][1])

    range_inputs = [(1, -1), (1, -1), (param_hidden_size, param_hidden_size)]
    shape_inputs = (-1, -1, param_hidden_size)

    x["ori_shape"], x["ori_range"] = shape_x, range_x
    weight_input["ori_shape"], weight_input["ori_range"] = shape_weight_input, range_weight_input
    weight_hidden["ori_shape"], weight_hidden["ori_range"] = shape_weight_hidden, range_weight_hidden
    y["ori_shape"], y["ori_range"] = shape_inputs, range_inputs
    h["ori_shape"], h["ori_range"] = shape_inputs, range_inputs
    dy["ori_shape"], dy["ori_range"] = shape_inputs, range_inputs
    update["ori_shape"], update["ori_range"] = shape_inputs, range_inputs
    reset["ori_shape"], reset["ori_range"] = shape_inputs, range_inputs
    new["ori_shape"], new["ori_range"] = shape_inputs, range_inputs
    hidden_new["ori_shape"], hidden_new["ori_range"] = shape_inputs, range_inputs

    if len(init_h["ori_shape"]) == 2:
        range_init_h = [(1, -1), (param_hidden_size, param_hidden_size)]
        shape_init_h = (-1, param_hidden_size)
    else:
        range_init_h = [(1, 1), (1, -1), (param_hidden_size, param_hidden_size)]
        shape_init_h = (1, -1, param_hidden_size)

    if len(dh["ori_shape"]) == 2:
        range_dh = [(1, -1), (param_hidden_size, param_hidden_size)]
        shape_dh = (-1, param_hidden_size)
    else:
        range_dh = [(1, 1), (1, -1), (param_hidden_size, param_hidden_size)]
        shape_dh = (1, -1, param_hidden_size)

    init_h["ori_shape"], init_h["ori_range"] = shape_init_h, range_init_h
    dh["ori_shape"], dh["ori_range"] = shape_dh, range_dh

    if seq_length is not None and len(seq_length["ori_shape"]) != 0:
        range_seq_length = [(param_hidden_size, param_hidden_size)]
        shape_seq_length = (param_hidden_size,)
        seq_length["ori_shape"], seq_length["ori_range"] = shape_seq_length, range_seq_length

    if mask is not None and len(mask["ori_shape"]) != 0:
        range_mask = [(param_hidden_size, param_hidden_size)]
        shape_mask = (param_hidden_size,)
        mask["ori_shape"], mask["ori_range"] = shape_mask, range_mask

    # outputs
    range_db_input = [(weight_input["ori_shape"][0], weight_input["ori_shape"][0])]
    shape_db_input = (weight_input["ori_shape"][0],)

    range_db_hidden = [(weight_hidden["ori_shape"][0], weight_hidden["ori_shape"][0])]
    shape_db_hidden = (weight_hidden["ori_shape"][0],)

    range_dh_prev = [(1, -1), (param_hidden_size, param_hidden_size)]
    shape_dh_prev = (-1, param_hidden_size)

    dw_input["ori_shape"], dw_input["ori_range"] = shape_weight_input, range_weight_input
    dw_hidden["ori_shape"], dw_hidden["ori_range"] = shape_weight_hidden, range_weight_hidden
    db_input["ori_shape"], db_input["ori_range"] = shape_db_input, range_db_input
    db_hidden["ori_shape"], db_hidden["ori_range"] = shape_db_hidden, range_db_hidden
    dx["ori_shape"], dx["ori_range"] = shape_x, range_x
    dh_prev["ori_shape"], dh_prev["ori_range"] = shape_dh_prev, range_dh_prev

    result = []
    result.append([x, weight_input, weight_hidden, y, init_h, h, dy, dh, update, reset, new, hidden_new,
                   seq_length, mask, dw_input, dw_hidden, db_input, db_hidden, dx, dh_prev,
                   {"direction": direction}, {"cell_depth": cell_depth}, {"keep_prob": keep_prob}, 
                   {"cell_clip": cell_clip}, {"num_proj": num_proj}, {"time_major": time_major},
                   {"gate_order": gate_order}, {"reset_after": reset_after}])
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def dynamic_gruv2_grad(x, weight_input, weight_hidden, y, init_h, h, dy, dh, update, reset, new, hidden_new,
                       seq_length, mask, dw_input, dw_hidden, db_input, db_hidden, dx, dh_prev,
                       direction="UNIDIRECTIONAL", cell_depth=0, keep_prob=-1.0, cell_clip=-1.0, num_proj=0,
                       time_major=True, gate_order="zrh", reset_after=True, kernel_name="dynamic_gruv2_grad"):
    """
    dynamic_gruv2_grad
    """
    pass
