#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
hardtanh_grad
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import HardtanhGradAttrInfo
from impl.util.util_soc_common import after_v200

# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=pointless-string-statement,no-else-return,unused-argument,invalid-name
@register_operator_compute("HardtanhGrad", op_mode="dynamic", support_fusion=True)
def hardtanh_grad_compute(input_result, input_grad, output_y, min_val=-1.0, max_val=1.0, kernel_name="hardtanh_grad"):
    """
    calculating data

    Parameters
    ----------
    input_result : TVM tensor
        the placeholder of input_x
    input_grad : TVM tensor
        the placeholder of input_y
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    min_val : float
        default to -1.0
    max_val : float
        default to 1.0
    kernel_name : str
        kernel name, default value is "hardtanh_grad"

    Returns
    -------
    output tensor
    """

    """
    Please refer to the TE DSL Manual, And code here with TE DSL.
    """
    in_data_type = input_result.dtype.lower()
    f_min = get_attr_by_cls(min_val, HardtanhGradAttrInfo.ATTR_MIN_VAL, "float32")
    f_max = get_attr_by_cls(max_val, HardtanhGradAttrInfo.ATTR_MAX_VAL, "float32")
    max_tensor = tbe.broadcast(f_max, input_result.shape, output_dtype="float32")
    min_tensor = tbe.broadcast(f_min, input_result.shape, output_dtype="float32")
    res_float32 = None
    if in_data_type in ["float16", "bfloat16"]:
        input_grad = tbe.cast_to(input_grad, "float32")
        input_result = tbe.cast_to(input_result, "float32")
    
    compare_if_nan = 1.0
    compare_if_nan_tensor = tbe.broadcast(compare_if_nan, input_result.shape, output_dtype="float32")

    lval = tbe.vcmp(compare_if_nan_tensor, input_result, operation='le')
    rval = tbe.vcmp(compare_if_nan_tensor, input_result, operation='gt')
    lval = tbe.cast_to(lval, "float32")
    rval = tbe.cast_to(rval, "float32")
    
    # socs those support inf
    if after_v200():
        # 910B: To generate a mask tensor that distinguishes whether the elements of input tensor are nan or not
        not_nan_mask = tbe.cast_to(tbe.vadd(lval, rval), "bool")

        # control value in maximum & minimum
        less_than_max_mask = tbe.vcmp(input_result, max_tensor, operation='lt')
        less_than_max_mask = tbe.cast_to(less_than_max_mask, "float32")
        greater_than_min_mask = tbe.vcmp(input_result, min_tensor, operation='gt')
        greater_than_min_mask = tbe.cast_to(greater_than_min_mask, "float32")
        
        # To generate a mask tensor that distinguishes whether the elements of input tensor are in maximum & minimum
        in_range_mask = tbe.vmul(greater_than_min_mask, less_than_max_mask)
        
        # select the result according to the mask tensor
        grad_product = tbe.vmul(in_range_mask, input_grad)
        res_float32 = tbe.vsel(not_nan_mask, grad_product, input_grad)

     # socs those don't support inf
    else:
        inf_fp32_val = 1e-10
        tmp_min = tbe.vmins(input_result, f_max)

        # control value in maximum & minimum
        tmp_max = tbe.vmaxs(tmp_min, f_min)
        if max_val != 0:
            sub_max = tbe.vsub(tmp_max, max_tensor)
        else:
            sub_max = tmp_max
        if min_val != 0:
            sub_min = tbe.vsub(tmp_max, min_tensor)
        else:
            sub_min = tmp_max
        mul_max_min = tbe.vmul(sub_max, sub_min)

        add_inf = tbe.vadds(mul_max_min, tvm.const(inf_fp32_val, dtype="float32"))
        div_res = tbe.vdiv(mul_max_min, add_inf)
        
        res_float32 = tbe.vmul(div_res, input_grad)

    if in_data_type == "float16":
        return tbe.cast_to(res_float32, in_data_type)
    elif in_data_type == "bfloat16":
        return tbe.round(res_float32, in_data_type)
    else:
        return res_float32


@register_operator("HardtanhGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def hardtanh_grad(result, grad, y, min_val, max_val, kernel_name="hardtanh_grad"):
    """
    calculating data

    Parameters
    ----------
    result : dict
        shape and dtype of input
    grad : dict
        shape and dtype of input, should be same shape and type as result
    y : dict
        shape and dtype of input, should be same shape and type as input
    min_val:
        minimum value of the linear region range.
    max_val:
        maximum value of the linear region range.
    kernel_name : str
        kernel name, default value is "hardtanh_grad"

    Returns
    -------
    None
    """
    result_dtype = (result.get("dtype")).lower()
    grad_dtype = (grad.get("dtype")).lower()

    """
    operator check
    """
    para_check.check_kernel_name(kernel_name)

    check_tuple = ("bfloat16", "float16", "float32")
    para_check.check_dtype(grad_dtype, check_tuple)
    para_check.check_dtype(result_dtype, check_tuple)

    if grad_dtype != result_dtype:
        raise RuntimeError("grad datatype %s and result datatype %s should be equal!" % (grad_dtype, result_dtype))

    """
    operator compute, invoke hardtanh_grad_compute
    """
    ins = classify([result, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_result, _grad) in ins:
        with tbe.compute():
            shape_result, shape_grad = shape_util.variable_shape([_result, _grad])
            data_result = tvm.placeholder(shape_result, name="data_result", dtype=result_dtype)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=grad_dtype)
            res = hardtanh_grad_compute(data_result, data_grad, y, min_val, max_val, kernel_name)
            tensors.append([data_result, data_grad, res])

        """
        auto schedule
        """
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    """
    operator build
    """
    config = {"name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)