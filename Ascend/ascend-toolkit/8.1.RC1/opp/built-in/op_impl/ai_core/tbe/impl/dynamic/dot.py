# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
dynamic dot
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import OpImplMode
from impl.util.util_soc_common import after_v200
from impl.util.util_common import check_op_impl_mode
from impl.util import util_common


def need_cast(impl_mode, dtype, vdiv_support):
    return (
        ((not after_v200()) or (dtype != "int32")) and
        ((dtype == "bfloat16") or
         (dtype in ("uint8", "int8") and vdiv_support) or
         (impl_mode == "high_precision" and dtype != "float32"))
    )


# 'pylint: disable=invalid-name,unused-argument
@register_operator_compute("Dot", op_mode="dynamic", support_fusion=True)
def dot_compute(input_x1, input_x2, axis, kernel_name="dot", impl_mode="high_precision"):
    """
    :param input_x1: one tesnor must be 1d   
    :param input_x2: another tensor must be 1d
    :param axis: which axis to reduce
    :param kernel_name: dot
    :param impl_mode: assign high_performance or high_precision
    :return: the dot result
    """
    x_dtype = input_x1.dtype.lower()
    vdiv_support = tbe_platform.api_check_support("tbe.dsl.vmod", "float32")

    if after_v200() and x_dtype in ("int8", "uint8"):
        input_x1 = tbe.cast_to(input_x1, 'int32')
        input_x2 = tbe.cast_to(input_x2, 'int32')
        mul_tmp = tbe.vmul(input_x1, input_x2)
        red_res = tbe.reduce_sum(mul_tmp, axis)
        red_res = util_common.reduce_int_cast_to_b8(red_res, x_dtype)
    elif need_cast(impl_mode, x_dtype, vdiv_support):
        input_x1 = tbe.cast_to(input_x1, 'float32')
        input_x2 = tbe.cast_to(input_x2, 'float32')
        mul_tmp = tbe.vmul(input_x1, input_x2)
        red_res = tbe.reduce_sum(mul_tmp, axis)
        if (x_dtype in ("uint8", "int8") and vdiv_support):
            red_res = util_common.uint8_int8_overflow_proc(red_res, x_dtype)
        elif x_dtype == "bfloat16":
            red_res = tbe.round(red_res, x_dtype)
        else:
            red_res = tbe.cast_to(red_res, x_dtype)
    else:
        mul_tmp = tbe.vmul(input_x1, input_x2)
        red_res = tbe.reduce_sum(mul_tmp, axis)
        
    return red_res


# 'pylint: disable=redefined-builtin
@register_operator("Dot")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, 
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def dot(input_x1, input_x2, output_z, kernel_name="dot", impl_mode="high_precision"):
    """
    :param input_x1: one tesnor must be 1d
    :param input_x2: another tensor must be 1d
    :param output_z: must be 1d
    :param kernel_name: dot
    :param impl_mode: assign high_performance or high_precision
    :return: the dot result
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)
    
    shape_x1 = input_x1.get("shape")
    shape_x2 = input_x2.get("shape")
    dtype_set = ("bfloat16", "float16", "float32", "int8", "int32", "uint8")
    dtype_x1 = input_x1.get("dtype").lower()
    dtype_x2 = input_x2.get("dtype").lower()
    para_check.check_dtype(dtype_x1, dtype_set)
    para_check.check_dtype(dtype_x2, dtype_set)
 
    input_x1["rel_pos_to_reduce"] = "before"
    input_x2["rel_pos_to_reduce"] = "before"
    
    axes = list(range(len(shape_x1)))
    input_axis = {"shape": [len(axes), ], "value": axes, "rel_pos_to_reduce": "axis"}
    ins = classify([input_x1, input_x2, input_axis], OpPatternMode.REDUCE,
                   {"keepdims": False})
                   
    schedules, tensors = [], []
    for (_input_x1, _input_x2, _axes) in ins:
        with tbe.compute():
            shape_x1, shape_x2 = shape_util.variable_shape([_input_x1, _input_x2, _axes], op_mode="reduce")[:2]
            data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=dtype_x1)
            data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=dtype_x2)
            res = dot_compute(data_x1, data_x2, _axes.get("value"))
            tensors.append([data_x1, data_x2, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)