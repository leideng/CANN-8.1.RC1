#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
batch_norm_grad
"""
from impl.util.platform_adapter import para_check
from impl.batch_norm_grad import batch_norm_grad


# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def batch_norm3d_grad(y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3,
                      x_backprop, scale_backprop,
                      offset_backprop, reserve_space_4, reserve_space_5,
                      epsilon=0.0001, data_format="NCDHW", is_training=True,
                      kernel_name="batch_norm_grad"):
    """
    algorithm: batch_norm3d_grad
    Batch normalization grad.

    Parameters
    ----------
    y_backprop: dict
        dict of y_backprop.
        source data type, support "float16", "float32".
    x: dict
        dict of x.
        source data type, support "float16", "float32".
    scale: dict
        dict of scale.
        source data type, support "float32".
    reserve_space_1: dict
        dict of reserve_space_1.
        source data type, support "float32".
        When is_training is True, a Tensor for the computed batch
        mean to be reused in gradient computation. When is_training is
        False, a Tensor for the population mean to be reused in both
        1st and 2nd order gradient computation.
    reserve_space_2: dict
        dict of reserve_space_2.
        source data type, support "float32".
        When is_training is True, a Tensor for the computed batch
        variance (inverted variance in the cuDNN case) to be reused in
        gradient computation. When is_training is False, a Tensor
        for the population variance to be reused in both 1st and 2nd
        order gradient computation.
    reserve_space_3: dict
        dict of reserve_space_3, this optional input is used to keep
        the parameters in this function same with the op prototype. 
    x_backprop: dict
        dict of output. Has the same type as `y_backprop`.
    scale_backprop: dict
        dict of scale_backprop. Has the same type as `reserve_space_1`.
    offset_backprop: dict
        dict of offset_backprop. Has the same type as `reserve_space_1`.
    reserve_space_4: dict
        dict of reserve_space_4.
    reserve_space_5: dict
        dict of reserve_space_5.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    data_format: str
        An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
        Either "NHWC" (default) or "NCHW".
    is_training: bool
        An optional `bool`. Defaults to `True`.
        A bool value to indicate the operation is for training (default)
        or inference.
    kernel_name: str
        kernel name, default value is "batch_norm_grad"

    Returns
    -------
    None
    """
    batch_norm_grad(y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3, x_backprop,
                    scale_backprop, offset_backprop, reserve_space_4, reserve_space_5,
                    epsilon, data_format, is_training, kernel_name=kernel_name)
