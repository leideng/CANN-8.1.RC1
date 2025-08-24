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
is_close
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_attr_common import IsCloseAttrInfo
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_soc_common import after_v200


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("IsClose", op_mode="dynamic", support_fusion=True)
def is_close_compute(input_x1, input_x2, output_y, rtol=1e-05, atol=1e-08, equal_nan=False, kernel_name="is_close"):
    """
    calculating a new tensor with bool elements representing if each element of input_x1 is "close" to the corresponding
    element of input_x2.
    Closeness is defined as:∣input_x1−input_x2∣≤atol+rtol×∣input_x2∣

    Parameters
    ----------
    input_x1: TVM tensor
        the placeholder of first input data
    input_x2: TVM tensor
        the placeholder of second input data
    output_y: dict
        shape and dtype of output, should be broadcast shape and bool type
    rtol: float
        absolute tolerance, default value 1e-08
    atol: float
        relative tolerance, default value is 1e-05
    equal_nan: bool
        if True, then two NaN s will be considered equal, default value is False
    kernel_name: str
        cce kernel name, default value is is_close

    Returns
    -------
    res : output of the data's isclose
    """
    shape_x1, shape_x2, shape_broad = shape_util.broadcast_shapes(input_x1.shape, input_x2.shape)
    input_x1 = tbe.broadcast(input_x1, shape_broad)
    input_x2 = tbe.broadcast(input_x2, shape_broad)

    input_dtype = input_x1.dtype
    if not tbe_platform.api_check_support("tbe.dsl.vabs", input_dtype):
        input_x1 = tbe.cast_to(input_x1, "float16")
        input_x2 = tbe.cast_to(input_x2, "float16")

    actual_error = tbe.vabs(tbe.vsub(input_x1, input_x2))
    rtol_sca = get_attr_by_cls(rtol, IsCloseAttrInfo.ATTR_RTOL, "float16")
    temp = tbe.vabs(tbe.vmuls(input_x2, rtol_sca))
    atol_sca = get_attr_by_cls(atol, IsCloseAttrInfo.ATTR_ATOL, "float16")
    allowed_error = tbe.vadds(temp, atol_sca)

    if not tbe_platform.api_check_support("tbe.dsl.vcmp", input_dtype):
        actual_error = tbe.cast_to(actual_error, "float16")
        allowed_error = tbe.cast_to(allowed_error, "float16")
    if after_v200() and (input_dtype in ["float16", "float32", "bfloat16"]):
        lval = tbe.vcmp(input_x1, input_x2, operation='eq')
        lval = tbe.cast_to(lval, "float16")
        if equal_nan:
            input_x1_nan = tbe.vcmp(input_x1, input_x1, operation="eq")
            input_x2_nan = tbe.vcmp(input_x2, input_x2, operation="eq")
            input_x1_nan_res = tbe.cast_to(input_x1_nan, "float16")
            input_x2_nan_res = tbe.cast_to(input_x2_nan, "float16")
            rval_nan_res = tbe.vadd(input_x1_nan_res, input_x2_nan_res)
            rval_nan = tbe.vcmp(rval_nan_res, 0, operation="eq")
            lval = tbe.vadd(lval, tbe.cast_to(rval_nan, "float16"))
        is_less_loss = tbe.vcmp(actual_error, allowed_error, operation="le")
        
        positive_infs = tbe.broadcast(tvm.const(float("inf"), input_dtype), shape_broad)  # isfinite(actual_error)
        negative_infs = tbe.broadcast(tvm.const(float("-inf"), input_dtype), shape_broad)
        neg_res_temp = tbe.vcmp(actual_error, negative_infs, "eq", "bit")
        actual_error = tbe.vsel(neg_res_temp, positive_infs, actual_error)
        isfinite_res = tbe.vcmp(actual_error, positive_infs, "ne")  
        
        isfinite_res = tbe.cast_to(isfinite_res, "int32")  # isfinite(actual_error)  and  actual_error<=allowed_error
        is_less_loss = tbe.cast_to(is_less_loss, "int32")
        data_and = tbe.vand(isfinite_res, is_less_loss)
        data_and = tbe.cast_to(data_and, "float16")
        const_num_one = tbe.broadcast(tvm.const(1, dtype="float16"), shape_broad)
        const_num_zero = tbe.broadcast(tvm.const(0, dtype="float16"), shape_broad)
        return tbe.cast_to(tbe.vcmpsel(tbe.vadd(lval, data_and),
                                       const_num_zero, 'eq', const_num_zero, const_num_one), "int8")

    return tbe.vcmp(actual_error, allowed_error, operation='le')


# 'pylint: disable=redefined-builtin
@register_operator("IsClose")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def is_close(x1, x2, output_y, rtol=1e-05, atol=1e-08, equal_nan=False, kernel_name="is_close"):
    """
    algorithm: is_close
    calculating a new tensor with bool elements representing if each element of x1 is "close" to the corresponding
    element of x2.
    Closeness is defined as:∣x1−x2∣≤atol+rtol×∣x2∣

    Parameters
    ----------
    x1 : dict
        shape and dtype of first input, only support float16, float32, int32
    x2 : dict
        shape and dtype of second input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be broadcast shape and bool type
    rtol: float
        absolute tolerance, default value 1e-08
    atol: float
        relative tolerance, default value is 1e-05
    equal_nan: bool
        if True, then two NaN s will be considered equal, default value is False
    kernel_name : str
        cce kernel name, default value is is_close

    Returns
    -------
    None
    """
    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape(shape_x1)
    para_check.check_shape(shape_x2)

    input_data_type = x1.get("dtype").lower()
    check_tuple = ("float16", "float32", "int32", "bfloat16")
    para_check.check_dtype_rule(input_data_type, check_tuple)

    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x1, _x2) in ins:
        with tbe.compute():
            shape_x1, shape_x2 = shape_util.variable_shape([_x1, _x2])
            shape_x1, shape_x2, shape_broad = shape_util.broadcast_shapes(shape_x1,
                                                                        shape_x2,
                                                                        param_name_input1="shape_x1",
                                                                        param_name_input2="shape_x2")
            if shape_x1[-1] == 1 and shape_x2[-1] == 1 and shape_broad[-1] == 1:
                shape_x1 = shape_x1 if len(shape_x1) == 1 else shape_x1[:-1]
                shape_x2 = shape_x2 if len(shape_x2) == 1 else shape_x2[:-1]

            dt_x1 = tvm.placeholder(shape_x1, name="data_1", dtype=input_data_type)
            dt_x2 = tvm.placeholder(shape_x2, name="data_2", dtype=input_data_type)
            if input_data_type == "float16" or input_data_type == "bfloat16" or input_data_type == "int32":
                dt_x1_trans = tbe.cast_to(dt_x1, "float32")
                dt_x2_trans = tbe.cast_to(dt_x2, "float32")
                res = is_close_compute(dt_x1_trans, dt_x2_trans, output_y, rtol, atol, equal_nan, kernel_name)
            else:
                res = is_close_compute(dt_x1, dt_x2, output_y, rtol, atol, equal_nan, kernel_name)

        tensors.append([dt_x1, dt_x2, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors, "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
