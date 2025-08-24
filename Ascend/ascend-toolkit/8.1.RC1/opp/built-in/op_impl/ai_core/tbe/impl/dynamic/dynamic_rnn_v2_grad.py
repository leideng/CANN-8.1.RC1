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
dynamic_rnn_grad
"""
import tbe.common.register as tbe_register
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-many-arguments,unused-argument,unnecessary-pass
@tbe_register.register_param_generalization("DynamicRNNV2Grad")
def dynamic_rnn_v2_grad_generalization(x,
                                       w_x,
                                       w_h,
                                       y,
                                       init_h,
                                       init_c,
                                       h,
                                       c,
                                       dy,
                                       dh,
                                       dc,
                                       i,
                                       j,
                                       f,
                                       o,
                                       tanhct,
                                       seq_length,
                                       wci,
                                       wcf,
                                       wco,
                                       mask,
                                       dw_x,
                                       dw_h,
                                       db,
                                       dx,
                                       dh_prev,
                                       dc_prev,
                                       dwci,
                                       dwcf,
                                       dwco,
                                       cell_type="LSTM",
                                       direction="UNIDIRECTIONAL",
                                       cell_depth=1,
                                       use_peephole=False,
                                       keep_prob=1.0,
                                       cell_clip=-1.0,
                                       num_proj=0,
                                       time_major=True,
                                       activation='tanh',
                                       recurrent_activation='sigmoid',
                                       gate_order="ijfo",
                                       stateful=False,
                                       merge_mode='concat',
                                       kernel_name="dynamic_rnn_v2_grad",
                                       generalize_config=None):
    """
    only T, b support -1
    """
    # x shape is [batch, input_size]
    param_input_size = x["ori_shape"][1]
    # w_x shape is [input_size, 4 * hidden_size]
    param_hidden_size = w_x["ori_shape"][1] / 4

    range_x = [(1, -1), (param_input_size, param_input_size)]
    shape_x = (-1, param_input_size)

    range_w_x = [(w_x["ori_shape"][0], w_x["ori_shape"][0]),
                 (w_x["ori_shape"][1], w_x["ori_shape"][1])]
    shape_w_x = (w_x["ori_shape"][0], w_x["ori_shape"][1])

    range_w_h = [(w_h["ori_shape"][0], w_h["ori_shape"][0]),
                 (w_h["ori_shape"][1], w_h["ori_shape"][1])]
    shape_w_h = (w_h["ori_shape"][0], w_h["ori_shape"][1])

    range_b = [(w_x["ori_shape"][1], w_x["ori_shape"][1])]  # 4*hidden_size
    shape_b = (w_x["ori_shape"][1],)

    range_cell = [(1, -1), (param_hidden_size, param_hidden_size)]
    shape_cell = (-1, param_hidden_size)

    range_y_3d = [(1, 1), (1, -1), (param_hidden_size, param_hidden_size)]
    shape_y_3d = (1, -1, param_hidden_size)

    range_y = [ (1, -1), (param_hidden_size, param_hidden_size)]
    shape_y = (-1, param_hidden_size)

    if len(y["ori_shape"]) == 3:
        y["ori_shape"], y["ori_range"] = shape_y_3d, range_y_3d
    else:
        y["ori_shape"], y["ori_range"] = shape_y, range_y

    if len(i["ori_shape"]) == 3:
        i["ori_shape"], i["ori_range"] = shape_y_3d, range_y_3d
        j["ori_shape"], j["ori_range"] = shape_y_3d, range_y_3d
        f["ori_shape"], f["ori_range"] = shape_y_3d, range_y_3d
        o["ori_shape"], o["ori_range"] = shape_y_3d, range_y_3d
        tanhct["ori_shape"], tanhct["ori_range"] = shape_y_3d, range_y_3d
    else:
        i["ori_shape"], i["ori_range"] = shape_y, range_y
        j["ori_shape"], j["ori_range"] = shape_y, range_y
        f["ori_shape"], f["ori_range"] = shape_y, range_y
        o["ori_shape"], o["ori_range"] = shape_y, range_y
        tanhct["ori_shape"], tanhct["ori_range"] = shape_y, range_y

    x["ori_shape"], x["ori_range"] = shape_x, range_x
    w_x["ori_shape"], w_x["ori_range"] = shape_w_x, range_w_x
    w_h["ori_shape"], w_h["ori_range"] = shape_w_h, range_w_h
    init_h["ori_shape"], init_h["ori_range"] = shape_cell, range_cell
    init_c["ori_shape"], init_c["ori_range"] = shape_cell, range_cell
    h["ori_shape"], h["ori_range"] = shape_y, range_y
    c["ori_shape"], c["ori_range"] = shape_y, range_y
    if len(dy["ori_shape"]) == 3:
        dy["ori_shape"], dy["ori_range"] = shape_y_3d, range_y_3d
    else:
        dy["ori_shape"], dy["ori_range"] = shape_y, range_y
    dh["ori_shape"], dh["ori_range"] = shape_cell, range_cell
    dc["ori_shape"], dc["ori_range"] = shape_cell, range_cell

    dw_x["ori_shape"], dw_x["ori_range"] = shape_w_x, range_w_x
    dw_h["ori_shape"], dw_h["ori_range"] = shape_w_h, range_w_h
    db["ori_shape"], db["ori_range"] = shape_b, range_b
    dx["ori_shape"], dx["ori_range"] = shape_x, range_x
    dh_prev["ori_shape"], dh_prev["ori_range"] = shape_cell, range_cell
    dc_prev["ori_shape"], dc_prev["ori_range"] = shape_cell, range_cell

    result = []
    result.append([x, w_x, w_h, y, init_h, init_c, h, c, dy, dh, dc, i, j, f, o, tanhct, seq_length,
                   wci, wcf, wco, mask, dw_x, dw_h, db, dx, dh_prev, dc_prev, dwci, dwcf, dwco,
                   {"cell_type": cell_type}, {"direction": direction}, {"cell_depth": cell_depth},
                   {"use_peephole": use_peephole}, {"keep_prob": keep_prob}, {"cell_clip": cell_clip},
                   {"num_proj": num_proj}, {"time_major": time_major}, {"activation": activation}, 
                   {"recurrent_activation": recurrent_activation}, {"gate_order": gate_order}, {"stateful": stateful},
                   {"merge_mode": merge_mode}])
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def dynamic_rnn_v2_grad(x,
                        w_x,
                        w_h,
                        y,
                        init_h,
                        init_c,
                        h,
                        c,
                        dy,
                        dh,
                        dc,
                        i,
                        j,
                        f,
                        o,
                        tanhct,
                        seq_length,
                        wci,
                        wcf,
                        wco,
                        mask,
                        dw_x,
                        dw_h,
                        db,
                        dx,
                        dh_prev,
                        dc_prev,
                        dwci,
                        dwcf,
                        dwco,
                        cell_type="LSTM",
                        direction="UNIDIRECTIONAL",
                        cell_depth=1,
                        use_peephole=False,
                        keep_prob=1.0,
                        cell_clip=-1.0,
                        num_proj=0,
                        time_major=True,
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        gate_order="ijfo",
                        stateful=False,
                        merge_mode='concat',
                        kernel_name="dynamic_rnn_v2_grad"):
    """
    dynamic_rnn_v2_grad
    """
    pass
