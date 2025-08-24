#!/usr/bin/env python
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
ascend_anti_quant_v2
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput

EXCEPT_PATTERN = "scalar_broadcast"
DTYPE_2_STR_ANTI_MAP = {
    1: "float16",
    27: "bfloat16",
}


def support_ub_fusion():
    """
    check ub fusion support
        fommat is ND can not support ub fusion
    """
    inputs = tbe_context.op_context.get_context().get_op_info()[0].inputs
    x_shape = inputs[0].get("shape")
    if x_shape[-1] == -1 or x_shape[-1] % 2 != 0:
        return False

    return True


# 'pylint: disable=too-many-arguments,invalid-name,unused-argument,unnecessary-lambda,too-many-locals
@register_operator_compute("AscendAntiQuantV2", op_mode="dynamic", support_fusion=support_ub_fusion)
def ascend_anti_quant_v2_compute(x, scale, offset, y, dst_type=1, sqrt_mode=False,
                                 kernel_name="ascend_anti_quant_v2"):
    """
    int4 -> float16/bfloat16

    Parameters:
    ----------
    x : the dict of input

    scale : the dict of scale

    offset : the dict of offset

    y : the dict of output

    dst_type : the output data type

    sqrt_mode : the sqrt mode when true the result to do sqrt

    kernel_name : cce kernel name, default value is "ascend_anti_quant_v2"

    Returns:
    -------
    None
    """
    y_dtype = DTYPE_2_STR_ANTI_MAP.get(dst_type)
    scale_cast_f32 = tbe.cast_to(scale, "float32")
    scale_broadcast = tbe.broadcast(scale_cast_f32, x.shape)

    cast_f16_ub = tbe.cast_to(x, "float16")
    cast_f32_ub = tbe.cast_to(cast_f16_ub, "float32")

    if offset is not None:
        offset_cast_f32 = tbe.cast_to(offset, "float32")
        offset_broadcast = tbe.broadcast(offset_cast_f32, x.shape)
        offset_ub = tbe.vadd(cast_f32_ub, offset_broadcast)

        if sqrt_mode:
            scale_sqrt_ub = tbe.vmul(offset_ub, scale_broadcast)
            res = tbe.vmul(scale_sqrt_ub, scale_broadcast)
        else:
            res = tbe.vmul(offset_ub, scale_broadcast)
    else:
        if sqrt_mode:
            scale_sqrt_ub = tbe.vmul(cast_f32_ub, scale_broadcast)
            res = tbe.vmul(scale_sqrt_ub, scale_broadcast)
        else:
            res = tbe.vmul(cast_f32_ub, scale_broadcast)
    
    if y_dtype == "bfloat16":
        res = tbe.round(res, y_dtype)
    else:
        res = tbe.cast_to(res, "float16")
    return res


def _check_params(x, scale, offset, y, dst_type):
    """
    check the parameters including shape, dtype, attr
    """
    x_dtype = x.get("dtype").lower()
    x_check_list = ["int4", "int8"]
    para_check.check_dtype(x_dtype, x_check_list, param_name="x_type")
    y_dtype = y.get("dtype").lower()
    y_check_list = ["float16", "bfloat16"]
    para_check.check_dtype(y_dtype, y_check_list, param_name="y_type")
    if dst_type is not None:
        y_dst_dtype = DTYPE_2_STR_ANTI_MAP.get(dst_type)
        para_check.check_dtype(y_dst_dtype, y_check_list, param_name="dst_type")
    if offset is not None:
        if not util_common.is_unknown([x, scale, offset]):
            shape_x = x.get("shape")
            shape_scale = scale.get("shape")
            shape_offset = offset.get("shape")
            if len(shape_scale) != 1:
                error_detail = "the length of scale shape should be 1."
                error_manager_vector.raise_err_input_shape_invalid("AscendAntiQuantV2", "scale", error_detail)
            if len(shape_offset) != 1:
                error_detail = "the length of offset shape should be 1."
                error_manager_vector.raise_err_input_shape_invalid("AscendAntiQuantV2", "offset", error_detail)

            if shape_scale[0] != 1 and shape_scale[0] != shape_x[-1]:
                error_detail = "the shape of scale should be [1] or equal to shape_x[-1]."
                error_manager_vector.raise_err_input_shape_invalid("AscendAntiQuantV2", "scale", error_detail)
            if shape_offset[0] != 1 and shape_offset[0] != shape_x[-1]:
                error_detail = "the shape of offset should be [1] or equal to shape_x[-1]."
                error_manager_vector.raise_err_input_shape_invalid("AscendAntiQuantV2", "offset", error_detail)
    else:
        if not util_common.is_unknown([x, scale]):
            shape_x = x.get("shape")
            shape_scale = scale.get("shape")

            if len(shape_scale) != 1:
                error_detail = "the length of scale shape should be 1."
                error_manager_vector.raise_err_input_shape_invalid("AscendAntiQuantV2", "scale", error_detail)

            if shape_scale[0] != 1 and shape_scale[0] != shape_x[-1]:
                error_detail = "the shape of scale should be [1] or equal to shape_x[-1]."
                error_manager_vector.raise_err_input_shape_invalid("AscendAntiQuantV2", "scale", error_detail)


def get_op_support_info(x, scale, offset, y, dst_type=1, sqrt_mode=False,
                        kernel_name="ascend_anti_quant_v2"):
    """
    get split info
    """
    dim_x = len(x.get("shape"))
    format_x = x.get("format")
   
    if format_x in ("ND"):
        axis_split_matrix = []
        for i in range(0, dim_x - 1):
            input_list = []
            for j in range(0, 3):
                input_0 = [j, [i], [-1], [-1]]
                input_list.append(input_0)
            split_0 = [SplitInput(*input_list), SplitOutput([0, [i]])]
            axis_split_matrix.append(split_0)
    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


@register_operator("AscendAntiQuantV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def ascend_anti_quant_v2(x, scale, offset, y, dst_type=1, sqrt_mode=False, kernel_name="ascend_anti_quant_v2"):
    """
    int4 -> float16/bfloat16

    Parameters:
    ----------
    x : the dict of input

    scale : the dict of scale

    offset : the dict of offset

    y : the dict of output

    dst_type : the output data type

    sqrt_mode : the sqrt mode when true the result to do sqrt

    kernel_name : cce kernel name, default value is "ascend_anti_quant_v2"

    Returns:
    -------
    None
    """
    _check_params(x, scale, offset, y, dst_type)
    input_dtype = x.get("dtype").lower()
    scale_dtype = scale.get("dtype").lower()

    if dst_type is None:
        dst_type = list(DTYPE_2_STR_ANTI_MAP.keys())[list(DTYPE_2_STR_ANTI_MAP.values()).index(y.get("dtype").lower())]

    schedules, tensors = [], []
    if offset is not None:
        offset_dtype = offset.get("dtype").lower()
        ins = classify([x, scale, offset], OpPatternMode.ELEWISE_WITH_BROADCAST)
        for (_x, _scale, _offset) in ins:
            with tbe.compute():
                x_shape, scale_shape, offset_shape = shape_util.variable_shape([_x, _scale, _offset])
                input_x = tvm.placeholder(x_shape, name="input_x", dtype=input_dtype)
                input_scale = tvm.placeholder(scale_shape, name="input_scale", dtype=scale_dtype)
                input_offset = tvm.placeholder(offset_shape, name="input_offset", dtype=offset_dtype)

                res = ascend_anti_quant_v2_compute(input_x, input_scale, input_offset, y, dst_type, sqrt_mode,
                                                   kernel_name)
                tensors.append([input_x, input_scale, input_offset, res])

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    else:
        ins = classify([x, scale], OpPatternMode.ELEWISE_WITH_BROADCAST)
        for (_x, _scale) in ins:
            if _x.get("pattern") == EXCEPT_PATTERN:
                continue
            with tbe.compute():
                x_shape, scale_shape = shape_util.variable_shape([_x, _scale])
                input_x = tvm.placeholder(x_shape, name="input_x", dtype=input_dtype)
                input_scale = tvm.placeholder(scale_shape, name="input_scale", dtype=scale_dtype)

                res = ascend_anti_quant_v2_compute(input_x, input_scale, None, y, dst_type, sqrt_mode, kernel_name)
                tensors.append([input_x, input_scale, res])

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
