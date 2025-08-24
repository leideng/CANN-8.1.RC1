# Copyright 2019 Huawei Technologies Co., Ltd
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
clip_by_value
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("clip_by_value", op_mode="static", support_fusion=True)
def clip_by_value_compute(input_t, clip_value_min, clip_value_max, output_t,
                          kernel_name="clip_by_value"):
    """
    algorithm: clip_by_value
    data = max  if data > max
    data = min  if data < min

    Parameters
    ----------
    input_t: TVM tensor
        the placeholders of input data
    clip_value_min: TVM tensor
        the placeholders of clip_value_min
    clip_value_max: TVM tensor
        the placeholders of data_max
    output_t: dict
        shape and dtype of output
    kernel_name: str
        kernel name, default value is "clip_by_value"

    Returns
    -------
    res: TVM tensor
        result of compute
    """
    input_shape = shape_util.shape_to_list(input_t.shape)
    shape_min_org = shape_util.shape_to_list(clip_value_min.shape)
    shape_max_org = shape_util.shape_to_list(clip_value_max.shape)
    _, _, shape_broadcast = \
        shape_util.broadcast_shapes(input_shape, shape_max_org, param_name_input1="input_t",
                                    param_name_input2="clip_value_max")

    if list(shape_max_org) != list(shape_broadcast):
        clip_value_max = tbe.broadcast(clip_value_max, shape_broadcast)

    if list(input_shape) != list(shape_broadcast):
        input_t = tbe.broadcast(input_t, shape_broadcast)

    res_min = tbe.vmin(input_t, clip_value_max)
    _, _, shape_broadcast2 = \
        shape_util.broadcast_shapes(shape_min_org, shape_broadcast, param_name_input1="clip_value_min",
                                    param_name_input2="input_t_broadcast")

    if list(shape_min_org) != list(shape_broadcast2):
        clip_value_min = tbe.broadcast(clip_value_min, shape_broadcast2)

    if list(shape_broadcast) != list(shape_broadcast2):
        res_min = tbe.broadcast(res_min, shape_broadcast2)

    res = tbe.vmax(res_min, clip_value_min)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def clip_by_value(input_t, clip_value_min, clip_value_max,
                  output_t, kernel_name="clip_by_value"):
    """
    algorithm: clip_by_value
    Clips tensor values to a specified min and max.
    Given a tensor t, this operation returns a tensor of
    the same type and shape as t
    with its values clipped to clip_value_min and clip_value_max.
    Any values less than clip_value_min are set to clip_value_min.
    Any values greater than clip_value_max are set to clip_value_max.

    Parameters
    ----------
    input_t: dict with keys(shape and dtype)
           input tensor
    clip_value_min: dict with keys(shape and dtype) or scaler
           The minimum value to clip by.
    clip_value_max: dict with keys(shape and dtype) or scaler
           The minimum value to clip by.
    output_t: dict
           info of output tensor with the same shape as input.
    kernel_name: str
           kernel name, default value is "clip_by_value"

    Returns
    -------
    None
    """
    shape_x = input_t.get("shape")
    dtype = input_t.get("dtype")
    shape_min = clip_value_min.get("shape")
    shape_max = clip_value_max.get("shape")
    if set(shape_min) == {1} and set(shape_max) == {1}:
        shape_x = shape_util.shape_refine(shape_x)
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, ("float16", "float32", "int32"), param_name="input_t")
    shape_x, shape_max, shape_broadcast = \
        shape_util.broadcast_shapes(shape_x, shape_max, param_name_input1="input_t",
                                    param_name_input2="clip_value_max")
    shape_min, shape_broadcast, _ = \
        shape_util.broadcast_shapes(shape_min, shape_broadcast, param_name_input1="clip_value_min",
                                    param_name_input2="input_t_broadcast")

    para_check.check_shape(shape_x, param_name="input_t")
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_dtype)

    data_value = {}
    para_check.check_shape(shape_min, param_name="clip_value_min")
    data_value["min"] = tvm.placeholder(shape_min, name="data_min", dtype=input_dtype)
    para_check.check_shape(shape_max, param_name="clip_value_max")
    data_value["max"] = tvm.placeholder(shape_max, name="data_max", dtype=input_dtype)

    res = clip_by_value_compute(data_x, data_value["min"], data_value["max"], output_t, kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_value["min"], data_value["max"], res]}
    build(sch, config)
