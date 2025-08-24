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
dynamic clip_by_norm_no_div_sum
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
@register_operator_compute("ClipByNormNoDivSum", op_mode="dynamic", support_fusion=True)
def clip_by_norm_no_div_sum_compute(data_input_x,
                                    data_greater_zeros,
                                    data_select_ones,
                                    data_maximum_ones,
                                    y,
                                    kernel_name="clip_by_norm_no_div_sum"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """
    dtype = data_input_x.dtype
    if dtype == "float32" and not tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        data_input_x = tbe.cast_to(data_input_x, "float16")
        data_greater_zeros = tbe.cast_to(data_greater_zeros, "float16")
        data_select_ones = tbe.cast_to(data_select_ones, "float16")
        data_maximum_ones = tbe.cast_to(data_maximum_ones, "float16")
    shape_input_x = data_input_x.shape
    shape_greater_zeros = data_greater_zeros.shape
    shape_select_ones = data_select_ones.shape

    before_vcmpsel_broadcast = shape_util.unify_broadcast_shapes(
        [shape_input_x, shape_greater_zeros, shape_select_ones]
    )

    broad_input_x = tbe.broadcast(data_input_x, before_vcmpsel_broadcast[-1])
    broad_greater_zeros = tbe.broadcast(data_greater_zeros, before_vcmpsel_broadcast[-1])
    broad_select_ones = tbe.broadcast(data_select_ones, before_vcmpsel_broadcast[-1])

    greater_select_res = tbe.vcmpsel(broad_input_x, broad_greater_zeros, "gt", broad_input_x, broad_select_ones)

    sqrt_res = tbe.vsqrt(greater_select_res)

    less_select_res = tbe.vcmpsel(broad_input_x, broad_greater_zeros, "le", broad_input_x, sqrt_res)

    shape_maximum_ones = data_maximum_ones.shape
    after_vcmpsel_broadcast = shape_util.unify_broadcast_shapes([before_vcmpsel_broadcast[-1], shape_maximum_ones])
    broadcast_maximum_ones = tbe.broadcast(data_maximum_ones, after_vcmpsel_broadcast[-1])
    less_select_res = tbe.broadcast(less_select_res, after_vcmpsel_broadcast[-1])

    res = tbe.vmax(less_select_res, broadcast_maximum_ones)
    if dtype == "float32" and not tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        res = tbe.cast_to(res, "float32")
    return res


@register_operator("ClipByNormNoDivSum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def clip_by_norm_no_div_sum(x, greater_zeros, select_ones, maximum_ones, y,
                            kernel_name="clip_by_norm_no_div_sum"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    schedules, tensors = [], []
    ins = classify([x, greater_zeros, select_ones, maximum_ones], OpPatternMode.ELEWISE_WITH_BROADCAST)

    for (_x, _greater, _zeros, _maximum) in ins:
        with tbe.compute():
            shape_x, shape_greater, shape_zeros, shape_maximum = shape_util.variable_shape([_x,
                                                                                            _greater,
                                                                                            _zeros,
                                                                                            _maximum])

            data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
            data_greater = tvm.placeholder(shape_greater, dtype=dtype_x, name="data_greater")
            data_zeros = tvm.placeholder(shape_zeros, dtype=dtype_x, name="data_zeros")
            data_maximum = tvm.placeholder(shape_maximum, dtype=dtype_x, name="data_maximum")
            res = clip_by_norm_no_div_sum_compute(data_x, data_greater, data_zeros, data_maximum, y,
                                                  kernel_name)
            tensors.append([data_x, data_greater, data_zeros, data_maximum, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              }
    tbe.build(schedules, config)
