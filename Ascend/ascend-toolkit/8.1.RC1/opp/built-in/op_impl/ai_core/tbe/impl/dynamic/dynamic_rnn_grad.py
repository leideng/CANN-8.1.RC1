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
dynamic_rnn_grad
"""
import tbe.common.register as tbe_register
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-many-arguments,unused-argument,unnecessary-pass
@tbe_register.register_param_generalization("DynamicRNNGrad")
def dynamic_rnn_grad_generalization(x,
                                    w,
                                    b,
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
                                    mask,
                                    wci,
                                    wcf,
                                    wco,
                                    dw,
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
                                    keep_prob=-1.0,
                                    cell_clip=-1.0,
                                    num_proj=0,
                                    time_major=True,
                                    forget_bias=0.0,
                                    gate_order= "ijfo",
                                    kernel_name="dynamic_rnn_grad",
                                    generalize_config=None):
    """
    only T, b support -1
    """

    param_input_size = x["ori_shape"][2]
    param_hidden_size = w["ori_shape"][1] / 4

    range_x = [(1, -1), (1, -1), (param_input_size, param_input_size)]
    shape_x = (-1, -1, param_input_size)

    range_w = [(w["ori_shape"][0], w["ori_shape"][0]),
               (w["ori_shape"][1], w["ori_shape"][1])]
    shape_w = (w["ori_shape"][0], w["ori_shape"][1])

    range_b = [(b["ori_shape"][0], b["ori_shape"][0])]
    shape_b = (b["ori_shape"][0],)

    if len(init_h["ori_shape"]) == 2:
        range_init = [(1, -1), (param_hidden_size, param_hidden_size)]
        shape_init = (-1, param_hidden_size)
    else:
        range_init = [(1, 1), (1, -1), (param_hidden_size, param_hidden_size)]
        shape_init = (1, -1, param_hidden_size)

    if len(dh["ori_shape"]) == 2:
        range_dh = [(1, -1), (param_hidden_size, param_hidden_size)]
        shape_dh = (-1, param_hidden_size)
    else:
        range_dh = [(1, 1), (1, -1), (param_hidden_size, param_hidden_size)]
        shape_dh = (1, -1, param_hidden_size)

    range_output = [(1, -1), (1, -1), (param_hidden_size, param_hidden_size)]
    shape_output = (-1, -1, param_hidden_size)

    range_prev = [(1, -1), (param_hidden_size, param_hidden_size)]
    shape_prev = (-1, param_hidden_size)

    x["ori_shape"], x["ori_range"] = shape_x, range_x
    w["ori_shape"], w["ori_range"] = shape_w, range_w
    b["ori_shape"], b["ori_range"] = shape_b, range_b

    y["ori_shape"], y["ori_range"] = shape_output, range_output
    init_h["ori_shape"], init_h["ori_range"] = shape_init, range_init
    init_c["ori_shape"], init_c["ori_range"] = shape_init, range_init
    h["ori_shape"], h["ori_range"] = shape_output, range_output
    c["ori_shape"], c["ori_range"] = shape_output, range_output
    dy["ori_shape"], dy["ori_range"] = shape_output, range_output
    dh["ori_shape"], dh["ori_range"] = shape_dh, range_dh
    dc["ori_shape"], dc["ori_range"] = shape_dh, range_dh
    i["ori_shape"], i["ori_range"] = shape_output, range_output
    j["ori_shape"], j["ori_range"] = shape_output, range_output
    f["ori_shape"], f["ori_range"] = shape_output, range_output
    o["ori_shape"], o["ori_range"] = shape_output, range_output
    tanhct["ori_shape"], tanhct["ori_range"] = shape_output, range_output

    if seq_length is not None and len(seq_length["ori_shape"]) != 0:
        range_seq_length = [(1, -1), (1, -1), (param_hidden_size, param_hidden_size)]
        shape_seq_length = (-1, -1, param_hidden_size)
        seq_length["ori_shape"], seq_length["ori_range"] = shape_seq_length, range_seq_length

    dw["ori_shape"], dw["ori_range"] = shape_w, range_w
    db["ori_shape"], db["ori_range"] = shape_b, range_b
    dx["ori_shape"], dx["ori_range"] = shape_x, range_x
    dh_prev["ori_shape"], dh_prev["ori_range"] = shape_prev, range_prev
    dc_prev["ori_shape"], dc_prev["ori_range"] = shape_prev, range_prev

    result = []
    result.append([x, w, b, y, init_h, init_c, h, c, dy, dh, dc, i, j, f, o, tanhct, seq_length, mask,
                   wci, wcf, wco, dw, db, dx, dh_prev, dc_prev, dwci, dwcf, dwco,
                   {"cell_type": cell_type}, {"direction": direction}, {"cell_depth": cell_depth},
                   {"use_peephole": use_peephole}, {"keep_prob": keep_prob}, {"cell_clip": cell_clip},
                   {"num_proj": num_proj}, {"time_major": time_major}, {"forget_bias": forget_bias}, 
                   {"gate_order": gate_order}])
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def dynamic_rnn_grad(x,
                     w,
                     b,
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
                     mask,
                     wci,
                     wcf,
                     wco,
                     dw,
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
                     keep_prob=-1.0,
                     cell_clip=-1.0,
                     num_proj=0,
                     time_major=True,
                     forget_bias=0.0,
                     gate_order= "ijfo",
                     kernel_name="dynamic_rnn_grad"):
    """
    dynamic_rnn_grad
    """
    pass
