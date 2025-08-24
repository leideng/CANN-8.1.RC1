#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
global_lppool
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.dynamic.power import power_compute
from impl.dynamic.reduce_sum_d import reduce_sum_d_compute
from tbe.common.utils.errormgr import raise_err_input_dtype_not_supported


# 'pylint: disable=too-many-locals,invalid-name
@register_operator_compute("GlobalLpPool", op_mode="dynamic", support_fusion=True)
def global_lppool_compute(x, y, p, axis):
    """
    :param x: input_data ,tvm.tensor
    :param y: output_data
    :param p: power
    :param axis: reduce axis
    :return: result tvm.tensor
    """
    input_dtype = x.dtype
    if input_dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp", "float32") and \
            tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
        x = tbe.cast_to(x, "float32")

    p_x = power_compute(x, y, power=p)
    sum_x = reduce_sum_d_compute(p_x, y, axis, keepdims=True)
    p_y = power_compute(sum_x, y, 1 / p)

    if p_y.dtype != input_dtype:
        p_y = tbe.cast_to(p_y, input_dtype)
    return p_y


# 'pylint: disable=invalid-name
@register_operator("GlobalLpPool")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def global_lppool(input_x, output_y, p=2.0, kernel_name="global_lppool"):
    """
    :param input_x: dict, include shape and dtype, dtype must be one of [float16, float32]
    :param output_y: dict, include shape and dtype, dtype must be one of [float16, float32]
    :param p: power, float, default to 2.0
    :param kernel_name: str, default to global_lppool
    :return:
    """
    shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    para_check.check_shape(shape, param_name="x")
    type_tuple = ("float16", "float32")
    para_check.check_dtype(input_dtype, type_tuple, param_name="x")

    if len(shape) != 4 and len(shape) != 5:
        raise RuntimeError("global_lppool only support NCHW and NCD0D1D2")

    axis = [2, 3]
    if len(shape) == 5:
        axis = [2, 3, 4]

    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    input_x["rel_pos_to_reduce"] = "before"
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    schedules = []
    tensors = []
    ins = classify([input_x, input_axis], OpPatternMode.REDUCE, {"keepdims": True})

    for (_x, axis) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_x, axis], op_mode="reduce")[0]
            data_input = tvm.placeholder(shape_var_new, name="input_data", dtype=input_dtype)

            if cur_cce_product in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"):
                if input_dtype == "float32":
                    raise_err_input_dtype_not_supported("global_lppool", "input_x", "float16", str(input_dtype))

                res = global_lppool_compute(data_input, output_y, p, axis.get("value"))
            else:
                res = global_lppool_compute(data_input, output_y, p, axis.get("value"))
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
