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
dynamic bn_3d_training_update_grad
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.dynamic.bn_training_update_grad import bn_training_update_grad
from impl.dynamic.bn_training_update_grad import bn_training_update_grad_compute


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
# 'pylint: disable=invalid-name,redefined-builtin,too-many-statements
@register_operator_compute("BN3DTrainingUpdateGrad", op_mode="dynamic", support_fusion=True)
def bn3d_training_update_grad_compute(grads,
                                      x,
                                      batch_mean,
                                      batch_variance,
                                      diff_scale,
                                      diff_offset,
                                      epsilon,
                                      kernel_name="bn3d_training_update_grad"):
    """
    Compute for bn_training_update_grad_compute
    x_norm:(x-input_reserve_space_1)*
            np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_scale:np.sum(y*(x-input_reserve_space_1)*
                         np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_offset: np.sum(y)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads. Must be one of the following
        type: `float16`, `float32`.
    x: TVM tensor 5D
        the placeholder of x. Must be one of the following
        type: `float16`, `float32`.
    batch_mean: TVM tensor 5D
        the placeholder of batch_mean. Must be one of the following
        type: `float32`.
    batch_variance: TVM tensor 5D
        the placeholder of batch_variance. Must be one of the following
        type: `float32`.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for output diff_scale.
    diff_offset: dict
        dict of diff_offset, A 5D Tensor for output diff_offset.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_3d_training_update_grad"

    Returns
    -------
    res_list: list
       [diff_scale, diff_offset].
    """
    return bn_training_update_grad_compute(grads,
                                           x,
                                           batch_mean,
                                           batch_variance,
                                           diff_scale,
                                           diff_offset,
                                           epsilon,
                                           kernel_name=kernel_name)


# 'pylint: disable=too-many-arguments
@register_operator("BN3DTrainingUpdateGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn3d_training_update_grad(grads,
                              x,
                              batch_mean,
                              batch_variance,
                              diff_scale,
                              diff_offset,
                              epsilon=0.0001,
                              kernel_name="bn3d_training_update_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_training_update_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
    x: dict
        dict of x, A 5D Tensor for input x.
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for output diff_scale.
    diff_offset: dict
        dict of diff_offset, A 5D Tensor for output diff_offset.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_3d_training_update_grad"

    Returns
    -------
    None
    """

    bn_training_update_grad(grads,
                            x,
                            batch_mean,
                            batch_variance,
                            diff_scale,
                            diff_offset,
                            epsilon,
                            kernel_name=kernel_name)
