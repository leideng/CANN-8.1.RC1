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
dynamic clip_by_value_v2
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("ClipByValueV2", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def clip_by_value_v2_compute(input_t,
                             clip_value_min,
                             clip_value_max,
                             output_t,
                             kernel_name="clip_by_value_v2"):
    """
    algorithm: clip_by_value_v2
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
        kernel name, default value is "clip_by_value_v2"

    Returns
    -------
    res: TVM tensor
        result of compute
    """
    _, _, shape_broadcast = \
        shape_util.broadcast_shapes(input_t.shape, clip_value_max.shape, param_name_input1="input_t",
                                    param_name_input2="clip_value_max")

    _, _, shape_broadcast2 = \
        shape_util.broadcast_shapes(clip_value_min.shape,
                                    shape_broadcast,
                                    param_name_input1="clip_value_min",
                                    param_name_input2="input_t_broadcast")

    input_t = tbe.broadcast(input_t, shape_broadcast2)
    clip_value_max = tbe.broadcast(clip_value_max, shape_broadcast2)
    clip_value_min = tbe.broadcast(clip_value_min, shape_broadcast2)

    res_max = tbe.vmax(input_t, clip_value_min)
    res = tbe.vmin(res_max, clip_value_max)
    return res


@register_operator("ClipByValueV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def clip_by_value_v2(input_t,
                     clip_value_min,
                     clip_value_max,
                     output_t,
                     kernel_name="clip_by_value_v2"):
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
           kernel name, default value is "clip_by_value_v2"

    Returns
    -------
    None
    """
    input_dtype = input_t.get("dtype").lower()
    para_check.check_dtype(input_dtype, ("bfloat16", "float16", "float32", "int32", "int64"), param_name="input_t")

    schedules, tensors = [], []
    ins = classify([input_t, clip_value_min, clip_value_max], OpPatternMode.ELEWISE_WITH_BROADCAST)
    for (_input_t, _min, _max) in ins:
        with tbe.compute():
            shape_t, shape_min, shape_max = shape_util.variable_shape([_input_t, _min, _max])
            data_t = tvm.placeholder(shape_t, dtype=input_dtype, name="data_x")
            data_min = tvm.placeholder(shape_min, dtype=input_dtype, name="data_mask")
            data_max = tvm.placeholder(shape_max, dtype=input_dtype, name="data_value")
            res = clip_by_value_v2_compute(data_t, data_min, data_max, output_t, kernel_name)
            tensors.append([data_t, data_min, data_max, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              }
    tbe.build(schedules, config)
