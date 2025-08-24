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
bn_3d_training_reduce
"""

import te.lang.cce as tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.bn_training_reduce import bn_training_reduce


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin, too-many-locals
def _reduce_compute_5hd(x):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data

    Returns
    -------
    res: TVM tensor list
        the result of bn_3d_training_reduce compute
    """
    square_x = tbe.vmul(x, x)

    axis = [0, 2, 3]
    sum_x, square_sum_x = tbe.tuple_sum([x, square_x], axis, True)

    res = [sum_x, square_sum_x]

    return res


@register_operator_compute("bn3d_training_reduce", op_mode="static", support_fusion=True)
def bn3d_training_reduce_compute(x, sum, square_sum,
                                 kernel_name="bn3d_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    res: TVM tensor list
        the result of bn_3d_training_reduce compute
    """
    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")

    res = _reduce_compute_5hd(x)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bn3d_training_reduce(x, sum, square_sum,
                         kernel_name="bn3d_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    None
    """
    bn_training_reduce(x, sum, square_sum, kernel_name=kernel_name)
