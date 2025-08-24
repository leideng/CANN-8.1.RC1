#!/usr/bin/env python
# coding: utf-8
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
pooling
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from tbe import tvm
from tbe.common.utils import para_check
from tbe.common.utils.errormgr import error_manager_cube as err_man
from impl.dynamic.conv2d import cube_forward_op
from tbe.common.utils import log
from tbe.common.register import register_param_generalization
from tbe.common.utils.op_util.op_util_conv2d import is_support_fixpipe
from tbe.common.register import register_operator

OP_NAME = "Pooling"
SHAPE_LEN = 5
ORI_SHAPE_LEN = 4
DYNAMIC_VALUE = -1
MODE_MAX = 0
MODE_AVG = 1
MODE_SUM = 2
NCHW_IDX = {"N" : 0 , "C" : 1, "H" : 2, "W" : 3}
C0_SIZE = 16
NONETYPE = type(None)
# The tiny groupFractalZ shape that pooling(mode=sum) needs. The shape is (K, N, CUBE_N, CUBE_K)
FILTER_GFZ_TINY_SHAPE_AND_RANGE = {
    "float16": ((1, 1, 16, 16), ((1, 1), (1, 1), (16, 16), (16, 16))),
    "float32": ((2, 1, 16, 8), ((2, 2), (1, 1), (16, 16), (8, 8))),
    "int8": ((1, 2, 16, 32), ((1, 1), (2, 2), (16, 16), (32, 32)))
}


@register_param_generalization("Pooling")
def pooling_generalization(x,
            matrix,
            bias,
            y,
            window=(1, 1),
            stride=(1, 1),
            offset_x=0,
            mode=0,
            pad=(0, 0, 0, 0),
            global_pooling=False,
            ceil_mode=0,
            dilation=(1, 1, 1, 1),
            kernel_name="pooling_cce",
            impl_mode="high_performance",
            generalize_config=None):
    log.debug("Entering pooling generalization, mode: %s", generalize_config)

    if not generalize_config:
        generalize_config = {"mode": "all_shape"}

    generalize_mode = generalize_config.get("mode")
    if generalize_mode != "all_shape":
        err_man.raise_err_specific_user(OP_NAME,
            f"dynamic pooling only supports all_shape generalize_mode, but received {generalize_mode}")

    if mode != MODE_SUM:
        err_man.raise_err_specific_user(OP_NAME,
            f"dynamic pooling only supports SUM mode (mode = {MODE_SUM}), but received {mode}")

    for tensor in [x, y]:
        if len(tensor.get("ori_shape")) != ORI_SHAPE_LEN or len(tensor.get("shape")) != SHAPE_LEN:
            err_man.raise_err_specific_user(OP_NAME, "check input shape size fail")
        tensor["shape"] = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
        tensor["ori_shape"] = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
        tensor["range"] = ((1, None), (1, None), (1, None), (1, None), (1, None))
        tensor["ori_range"] = ((1, None), (1, None), (1, None), (1, None))

    if len(matrix.get("ori_shape")) != ORI_SHAPE_LEN or len(matrix.get("shape")) != ORI_SHAPE_LEN:
        err_man.raise_err_specific_user(OP_NAME, "check matrix shape size fail")
    matrix_dtype = matrix.get("dtype")
    if matrix_dtype not in FILTER_GFZ_TINY_SHAPE_AND_RANGE.keys():
        err_man.raise_err_specific_user(OP_NAME, "The data type {} of 2nd input is not supported.".format(matrix_dtype))
    log.debug("Matrix data type is {}".format(matrix_dtype))
    matrix["shape"] = FILTER_GFZ_TINY_SHAPE_AND_RANGE.get(matrix_dtype)[0]  # 0 is shape tuple
    matrix["ori_shape"] = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
    matrix["range"] = FILTER_GFZ_TINY_SHAPE_AND_RANGE.get(matrix_dtype)[1]  # 1 is shape range tuple
    matrix["ori_range"] = ((1, None), (1, None), (1, None), (1, None))

    bias = None
    window = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
    stride = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
    offset_x = None
    pad = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
    global_pooling = None
    ceil_mode = None
    dilation = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
    generalized_result = [[x, matrix, bias, y, window, stride, offset_x,
                          mode, pad, global_pooling, ceil_mode, dilation,
                          kernel_name, impl_mode]]
    log.debug("Pooling generalization result: %s", generalized_result)
    return generalized_result


# 'pylint: disable = too-many-branches,too-many-statements,variable_type_changed
@register_operator("Pooling")
@para_check.check_input_type(dict, dict, (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list),
                             (int, NONETYPE), int, (tuple, list),
                             (bool, NONETYPE), (int, NONETYPE),
                             (tuple, list), str, str)
def pooling(x,
            matrix,
            bias,
            y,
            window=(1, 1),
            stride=(1, 1),
            offset_x=0,
            mode=0,
            pad=(0, 0, 0, 0),
            global_pooling=False,
            ceil_mode=0,
            dilation=(1, 1, 1, 1),
            kernel_name="pooling_cce",
            impl_mode="high_performance"):
    """
    Dynamic Pooling currently only supports sum mode.
    Pooling sum reuse conv2d implementation with group = cin

    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16,
        shape is 5HD, ori_shape is 4 dims, format is NCHW/NHWC
    matrix: dict, shape and dtype of right matrix, only support float16, shape is 4 dims, format is NCHW
        shape is 4 dims, ori_shape is 4 dims, format is FRACTAL_Z
    bias: dict, shape and dtype of bias, only support float16, shape is 4 dims, format is NCHW, only use bias in conv2d
    y : output dict, shape and dtype of output_data, only support float16
        shape is 5HD, ori_shape is 4 dims, format is NCHW/NHWC
    window : list or tuple, the window of pooling, only support pooling in H or W
        window must have 4 dim in dynamic pooling
    stride : list or tuple, the stride of pooling window, only support pooling in H or W
    offset_x : avg quantize parmas
    mode : int, the mode of pooling, support 0:max pooling, 1:avg pooling, 2:sum pooling.
           Only mode 2 is supported in dynamic pooling
    pad : list or tuple, the pad of pooling, only support pooling in H or W
    global_pooling : global pooling params.
    ceil_mode : caffe round_mode params, 0:CEIL, 1:FLOOR, default value is DOMI_POOLING_CEIL
    dilation : reserved.
    kernel_name : cce kernel name, default value is "pooling_cce"
    impl_mode : high_precision or high_performance for inference, default value is "high_performance".
    Returns
    -------
    None
    """
    log.debug("Entering dynamic Pooling op")
    if mode != MODE_SUM:
        err_man.raise_err_specific_user(OP_NAME,
            f"dynamic pooling only supports SUM mode (mode = {MODE_SUM}), but recieved {mode}")

    if len(stride) != ORI_SHAPE_LEN:
        err_man.raise_err_specific_user(OP_NAME,
            f"Input stride should be 4 dimentional list, but recieved {stride}")
    if len(window) != ORI_SHAPE_LEN:
        err_man.raise_err_specific_user(OP_NAME,
            f"Input window should be 4 dimentional list, but recieved {window}")

    # start to construct input for cube_forward op
    # data format does not matter in dynamic mode as the shape be generalized anyway
    x_ori_shape = x.get("ori_shape")
    if x_ori_shape is None:
        raise RuntimeError("Invalid, x_ori_shape is None.")
    c_idx = NCHW_IDX.get("C", 1)
    groups = x_ori_shape[c_idx]

    op_option_dict = {"tiny_weight_fractal_flag" : True,
                      "ksize" : window}
    cube_input = {"op_type" : "Pooling",
                  "inputs" : x,
                  "weights" : matrix,
                  "bias" : bias,
                  "offset_w" : 0,
                  "outputs" : y,
                  "strides" : stride,
                  "pads" : pad,
                  "dilations" : dilation,
                  "groups" : groups,
                  "data_format" : "NCHW",
                  "offset_x" : offset_x,
                  "kernel_name" : kernel_name,
                  "op_option_dict" : op_option_dict
    }
    cube_forward_op(**cube_input)
