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
dynamic bn3d_training_reduce
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.dynamic.bn_training_reduce import bn_training_reduce
from impl.dynamic.bn_training_reduce import bn_training_reduce_compute


# 'pylint: disable=redefined-builtin
@register_operator_compute("BN3DTrainingReduce", op_mode="dynamic", support_fusion=True)
def bn3d_training_reduce_compute(x, sum, square_sum, kernel_name="bn3d_training_reduce"):
    """
    algorithm: part of batch_norm
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
    return bn_training_reduce_compute(x, sum, square_sum, kernel_name)


@register_operator("BN3DTrainingReduce")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def bn3d_training_reduce(x, sum, square_sum, kernel_name="bn3d_training_reduce"):
    """
    algorithm: part of batch_norm
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
