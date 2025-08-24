"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

threshold
"""

# 'pylint: disable=invalid-name,unused-argument
import functools

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.util_attr_common import ThresholdAttrInfo
from impl.util.util_attr_common import get_attr_by_cls


# 'pylint: disable=redefined-outer-name
@register_operator_compute("threshold", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def threshold_compute(input_x, output_y, threshold, kernel_name="threshold", impl_mode=None):
    """
    compare data with threshold,x > threshold ? 1; 0

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    output_y: dict
        shape and dtype of output, should be broadcast shape and type as input
    threshold: float
        parameter of the operator, value of threshold
    kernel_name: str
        cce kernel name, default value is threshold
    impl_mode: str
        high_precision for inference, default value is "None"
        
    Returns
    -------
    res: if data is bigger than threshold return 1,else return 0
    """
    
    # define scalar
    SCALAR_MAX_DIGIT = 10000000.0
    
    # switch dtype of input_x
    input_dtype = input_x.dtype
    if input_dtype == "float16":
        if impl_mode == OpImplMode.HIGH_PRECISION:
            input_x = tbe.cast_to(input_x, "float32")
        SCALAR_MAX_DIGIT = 10000.0

    #`sub_threshold = x - threshold`
    x_dtype = input_x.dtype
    threshold_scalar = get_attr_by_cls(threshold, ThresholdAttrInfo.ATTR_THRESHOLD, input_dtype)
    sub_threshold = tbe.vadds(input_x, -threshold_scalar)
    
    # `one_data = x > threshold ? min(x-1, 1) : 0`
    zero_data = tbe.vmaxs(sub_threshold, 0.0)
    one_data = tbe.vmins(zero_data, 1.0)

    # zoom in x in (0,1) to 1
    zoom_data_tmp = tbe.vmuls(one_data, SCALAR_MAX_DIGIT)
    res_tmp = tbe.vmins(zoom_data_tmp, 1.0)
    zoom_data = tbe.vmuls(res_tmp, SCALAR_MAX_DIGIT)
    res = tbe.vmins(zoom_data, 1.0)
    
    # switch dtype back 
    if x_dtype != input_dtype:
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def threshold(input_x, output_y, threshold=0.0, kernel_name="threshold", impl_mode=None):
    """
    algorithm: threshold
    compare data with threshold: x > threshold ? 1; 0

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be broadcast shape and type as input
    threshold: float
        parameter of the operator
    kernel_name : str
        kernel name, default value is "threshold"
    impl_mode: str
        high_precision for inference, default value is "None"
        
    Returns
    -------
    None
    """

    # get and check the shape and dtype
    shape = input_x.get("shape")
    input_data_type = input_x.get("dtype").lower()
    para_check.check_shape(shape, param_name="input_x")
    para_check.check_dtype(input_data_type, ["float16", "float32", "bfloat16"], param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input",
                                         dtype=input_data_type)
            res = threshold_compute(data_input, output_y, threshold, kernel_name, impl_mode)
            
            tensors.append([data_input, res])
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False,
              "name": kernel_name,
              "need_build": False,
              "tensor_list": tensors}
    tbe.build(schedules, config)
