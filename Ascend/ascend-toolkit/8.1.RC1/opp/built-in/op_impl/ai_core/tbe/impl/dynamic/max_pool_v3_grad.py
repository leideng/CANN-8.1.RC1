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
max_pool_v3_grad
"""

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util import util_common
import tbe
from tbe.common.utils import shape_util
from tbe import tvm
import te.platform as tbe_platform
from tbe.common.platform import ASCEND_910B
from tbe.common.platform import ASCEND_910_93
from tbe.common.platform import SHORT_SOC_VERSION
from tbe.dsl.compute.pooling.max_pool_grad import max_pool_grad as max_pool_grad_compute


# 'pylint: disable=dangerous-default-value
# 'pylint: disable=too-few-public-methods,too-many-statements,too-many-branches,no-self-use,huawei-too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-many-lines
# 'pylint: disable=too-many-lines,too-many-locals,too-many-statements,unused-variable,too-many-arguments
def op_select_format(orig_x,
                     orig_y,
                     grads,
                     y,
                     ksize,
                     strides,
                     padding="CALCULATED",
                     pads=[0, 0, 0, 0],
                     data_format="NCHW",
                     global_pooling=False,
                     ceil_mode=False,
                     kernel_name="max_pool_v3_grad"):
    input_dtype = orig_x.get("dtype").lower()
    soc_ver = tbe_platform.get_soc_spec(SHORT_SOC_VERSION)
    x1_shape = list(orig_x.get("ori_shape"))
    x2_shape = list(orig_y.get("ori_shape"))
    grad_shape = list(grads.get("ori_shape"))

    if soc_ver in (ASCEND_910B, ASCEND_910_93):
        x1_format,  x2_format,  grad_format,  y_format = \
            ["NC1HWC0", "NC1HWC0"], ["NC1HWC0", "NC1HWC0"], ["NC1HWC0", "NC1HWC0"], ["NC1HWC0", "NC1HWC0"]
        x1_dy_format, x2_dy_format, grad_dy_format, y_dy_format = \
            ["NC1HWC0", "NC1HWC0"], ["NC1HWC0", "NC1HWC0"], ["NC1HWC0", "NC1HWC0"], ["NC1HWC0", "NC1HWC0"]
        x1_dtype, x2_dtype, grad_dtype, y_dtype = \
            ["float", "float16"], ["float", "float16"], ["float", "float16"], ["float", "float16"]
    else:
        x1_format,  x2_format,  grad_format,  y_format = \
            ["NC1HWC0", ], ["NC1HWC0", ], ["NC1HWC0", ], ["NC1HWC0", ]
        x1_dy_format, x2_dy_format, grad_dy_format, y_dy_format = \
            ["NC1HWC0", ], ["NC1HWC0", ], ["NC1HWC0", ], ["NC1HWC0", ]
        x1_dtype, x2_dtype, grad_dtype, y_dtype = \
            ["float16", ], ["float16", ], ["float16", ], ["float16", ]

    input0 = gen_param(classify="input0", name="orig_input",
                       datatype=",".join(x1_dtype),
                       format=",".join(x1_format),
                       unknownshape_format=",".join(x1_dy_format))
    input1 = gen_param(classify="input1", name="orig_output",
                       datatype=",".join(x2_dtype),
                       format=",".join(x2_format),
                       unknownshape_format=",".join(x2_dy_format))
    input2 = gen_param(classify="input2", name="grad",
                       datatype=",".join(grad_dtype),
                       format=",".join(grad_format),
                       unknownshape_format=",".join(grad_dy_format))
    output0 = gen_param(classify="output0", name="out_grad",
                        datatype=",".join(y_dtype),
                        format=",".join(y_format),
                        unknownshape_format=",".join(y_dy_format))

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=dangerous-default-value
# 'pylint: disable=too-few-public-methods,too-many-statements,too-many-branches,no-self-use,huawei-too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-many-lines
# 'pylint: disable=too-many-lines,too-many-locals,too-many-statements,unused-variable,too-many-arguments
def check_supported(orig_x,
                     orig_y,
                     grads,
                     y,
                     ksize,
                     strides,
                     padding="CALCULATED",
                     pads=[0, 0, 0, 0],
                     data_format="NCHW",
                     global_pooling=False,
                     ceil_mode=False,
                     kernel_name="max_pool_v3_grad"):
    """
    check whether ai_core is supported
    """
    if util_common.is_unknown([orig_x, orig_y]):
        return True, ""

    input_dtype = orig_x.get("dtype").lower()
    if input_dtype in ("float32",):
        return True, ""

    return False, ""


# 'pylint: disable=dangerous-default-value,too-many-locals,
# 'pylint: disable=invalid-name,unused-argument,huawei-too-many-arguments
@register_operator("MaxPoolV3Grad", pattern="PoolGrad")
def max_pool_v3_grad(orig_x,
                     orig_y,
                     grads,
                     y,
                     ksize,
                     strides,
                     padding="CALCULATED",
                     pads=[0, 0, 0, 0],
                     data_format="NCHW",
                     global_pooling=False,
                     ceil_mode=False,
                     kernel_name="max_pool_v3_grad"):
    """
    main function of max_pool_v3_grad

    Parameters
    ----------
    ori_input: dict
        shape and data type of ori_input
    ori_output: dict
        shape and data type of ori_output
    grad: dict
        shape and data type of grad
    y: dict
        shape and data type of y
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: str
        value from `SAME`, `VALID`, 'CALCULATED'
    pads: list or tuple
        padding size of height and width while padding is CALCULATED
    data_format: str
        value from `NCHW`, `NHWC`
    global_pooling: bool
        if true, ksize and padding will be invalid
    ceil_mode: bool
        whether use ceil fuction to calculate the output of height and width
    kernel_name: str

    Returns
    -------
    return the tik api function
    """
    """
    main function of max_pool3d_grad

    Parameters
    ----------
    orig_x: dict
        shape and data type of max_pool3d's forward_input
    orig_y: dict
        shape and data type of max_pool3d's forward_output
    grads: dict
        shape and data type of grads
    y: dict
        shape and data type of y
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    pads: list or tuple
        the fill value of input
    data_format: str
        value from `NDHWC`, `NCDHW`
    kernel_name: str

    Returns
    -------
    return the tik api function
    """
    x_dtype = orig_x.get("dtype")
    input_format, ori_format = orig_x["format"], orig_x["ori_format"]
    window_axes, normalized_ksize, normalized_strides = [], [], []
    for i, (i_axis, i_ksize, i_stride) in enumerate(zip(ori_format, ksize, strides)):
        if i_axis in ('D', 'H', 'W'):
            window_axes.append(i)
            if orig_x["shape"] == (-2, ):
                i_ksize, i_stride = -1, -1
            normalized_ksize.append(i_ksize)
            normalized_strides.append(i_stride)

    if input_format == "NC1HWC0":
        window_axes = [2, 3]
    elif input_format == "NDC1HWC0":
        window_axes = [1, 3, 4]

    padding_dims = [[pads[i], pads[i+1]] for i in range(0, len(pads) - 1, 2)]

    ins = tbe.dsl.classify([orig_x, orig_y, grads, window_axes], "PoolGrad")
    schedules, tensors = [], []
    for (x_c, y_c, dy_c, window_axes) in ins:
        with tbe.dsl.compute():
            x_v, y_v, dy_v, ksize_v, strides_v, pads_v = shape_util.variable_shape([x_c, y_c, dy_c,
                    window_axes, normalized_ksize, normalized_strides, []], op_mode="PoolGrad")

            ph_x = tvm.placeholder(x_v, dtype=x_dtype, name="x")
            ph_y = tvm.placeholder(y_v, dtype=x_dtype, name="y")
            ph_dy = tvm.placeholder(dy_v, dtype=x_dtype, name="dy")

            dx = max_pool_grad_compute(ph_x, ph_y, ph_dy,
                                       window_axes,
                                       ksize_v,
                                       strides_v,
                                       padding_mode=padding,
                                       padding_dimensions=padding_dims,
                                       ceil_mode=ceil_mode)

            tensors.append((ph_x, ph_y, ph_dy, dx))

        with tvm.target.cce():
            schedule = tbe.dsl.auto_schedule(dx)
            schedules.append(schedule)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)
