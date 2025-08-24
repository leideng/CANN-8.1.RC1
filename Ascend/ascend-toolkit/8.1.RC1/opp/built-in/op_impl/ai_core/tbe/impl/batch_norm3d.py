#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

batch_norm_3d
"""
from impl.util.platform_adapter import para_check
from impl.batch_norm import batch_norm


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin, too-many-locals, too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def batch_norm3d(x, scale, offset, mean, variance, y, batch_mean,
                 batch_variance, reserve_space_1, reserve_space_2,
                 epsilon=0.0001, data_format="NCDHW",
                 is_training=True, kernel_name="batch_norm3d"):
    """
    algorithm: fused_batch_norm_3d
    Batch normalization.
    Note that the size of 5D Tensors are defined by "NDC1HWC0".
    The input tensor's dimension C should be equal.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    scale: dict
        dict of scale,
        A Tensor for scaling factor, to scale the normalized x.
    offset: dict
        dict of offset, A Tensor for offset, to shift to the normalized x.
    mean: dict
        dict of mean, A Tensor for population mean.
        Used for inference only, must be empty for training.
    variance: dict
        dict of variance, A Tensor for population variance.
        Used for inference only, must be empty for training.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`. Has the same type as `mean`.
    batch_variance: dict
        dict of batch_var, A `Tensor`. Has the same type as `variance`.
    reserve_space_1: dict
        dict of reserve_space_1, A `Tensor`.
    reserve_space_2: dict
        dict of reserve_space_2, A `Tensor`.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.001`.
    data_format: str
        The data format for x and y. Support "NC1HWC0" only.
    is_training: bool
        A bool value indicates the operation for train (default) or inference.
    kernel_name: str
        kernel name, default value is "batch_norm"

    Returns
    -------
    None
    """

    batch_norm(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1,
               reserve_space_2, None, epsilon, data_format, is_training, kernel_name)
