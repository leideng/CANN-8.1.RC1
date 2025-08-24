# /usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

GetFloatStatus
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check


# 'pylint: disable=invalid-name,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def tensor_redirect(x, output_x, kernel_name="tensor_redirect"):
    """
    the main function of TensorRedirect

    Parameters
    ----------
    x: dict,shape and datatype
    output_x: dict,shape and datatype
    kernel_name: cce kernel name, default value is "tensor_redirect"

    Returns
    -------
    tik_instance: tik_instance
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    input_dtype = dtype.lower()
    check_list = ["float16", "float32", "int8", "int32", "uint8",
                  "int16", "uint16", "uint32", "int64", "uint64"]
    para_check.check_dtype(input_dtype, check_list)

    tik_instance = tik.Tik()

    input_addr = tik_instance.Tensor(input_dtype, shape, name="input_addr",
                                     scope=tbe_platform.scope_gm)
    output_data = tik_instance.Tensor(input_dtype, [64,],
                                      name="output_data",
                                      scope=tbe_platform.scope_gm)

    data_ub = tik_instance.Tensor(input_dtype, [64,], name="data_ub",
                                  scope=tbe_platform.scope_ubuf)
    tik_instance.data_move(data_ub, input_addr, 0, 1, 1, 0, 0)
    tik_instance.data_move(output_data, data_ub, 0, 1, 1, 0, 0)
    tik_instance.BuildCCE(kernel_name, inputs=[input_addr],
                          outputs=[output_data])
    return tik_instance
