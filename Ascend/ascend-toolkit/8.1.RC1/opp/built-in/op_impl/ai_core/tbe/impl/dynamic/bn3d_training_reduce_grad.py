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
dynamic bn_3d_training_reduce_grad
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_compute import only_static_support
from impl.dynamic.bn_training_reduce_grad import bn_training_reduce_grad
from impl.dynamic.bn_training_reduce_grad import bn_training_reduce_grad_compute


# 'pylint: disable=unused-argument,too-many-locals
# 'pylint: disable=invalid-name,redefined-builtin,too-many-statements,too-many-arguments
@register_operator_compute("BN3DTrainingReduceGrad", op_mode="dynamic", support_fusion=only_static_support)
def bn3d_training_reduce_grad_compute(grads,
                                      x,
                                      diff_scale,
                                      diff_offset,
                                      scale,
                                      batch_mean,
                                      batch_variance,
                                      y,
                                      epsilon,
                                      kernel_name="bn3d_training_reduce_grad"):
    """
    Compute for batch_norm_grad
    y:(grads*scale*np.power((batch_variance + epsilon), (-0.5)))+
      np.sum(grads*scale*(-0.5)*x_norm*np.power((batch_variance+epsilon),(-1))))
      *(2/m)+np.sum(grads*scale*(-1)*
      np.power((batch_variance+epsilon),(-0.5)))*(1/m)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads.
        Must be one of the following type: `float16`, `float32`, `bfloat16`.
    x: TVM tensor 5D
        the placeholder of x.
        Must be one of the following type: `float32`, 'float16`, `bfloat16`.
    diff_scale: TVM tensor 5D
        the placeholder of diff_scale.
        Must be one of the following type: `float32`.
    diff_offset: TVM tensor 5D
         the placeholder of diff_offset.
         Must be one of the following types: `float32`.
    scale: TVM tensor 5D
        the placeholder of scale.
        Must be one of the following types: `float32`.
    batch_mean: dict 5D
        the placeholder of batch_mean.
        Must be one of the following types: `float32`.
    batch_variance: dict 5D
        the placeholder of batch_variance.
        Must be one of the following types: `float32`.
    y: dict
        dict of y, include keys(shape and dtype).
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_3d_training_reduce_grad"

    Returns
    -------
    res: TVM tensor
    """
    return bn_training_reduce_grad_compute(grads,
                                           x,
                                           diff_scale,
                                           diff_offset,
                                           scale,
                                           batch_mean,
                                           batch_variance,
                                           y,
                                           epsilon, None, None,
                                           kernel_name=kernel_name)


# 'pylint: disable=too-many-arguments
@register_operator("BN3DTrainingReduceGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def bn3d_training_reduce_grad(grads,
                              x,
                              diff_scale,
                              diff_offset,
                              scale,
                              batch_mean,
                              batch_variance,
                              y,
                              epsilon=0.0001,
                              kernel_name="bn3d_training_reduce_grad"):
    """
    algorithm: batch_norm_grad
    bn_training_reduce_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
        source data type, support "float32", "float16", "bfloat16".
    x: dict
        dict of s, A 5D Tensor for input x.
        source data type, support "float32", "float16", "bfloat16".
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for input diff_scale.
        The output of bn_training_update_grad.
        source data type, support "float32".
    diff_offset: dict
        dict of diff_offset, A 5HD Tensor for input diff_offset.
        The output of bn_training_update_grad.
        source data type, support "float32".
    scale: dict
        dict of scale, A 5HD Tensor for input scale.
        source data type, support "float32".
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
        source data type, support "float32".
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
        source data type, support "float32".
    y: dict
        dict of output, A `Tensor`. Has the same type as `grads`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_3d_training_reduce_grad"

    Returns
    -------
    None
    """
    bn_training_reduce_grad(grads,
                            x,
                            diff_scale,
                            diff_offset,
                            scale,
                            batch_mean,
                            batch_variance,
                            y,
                            epsilon, None, None,
                            kernel_name=kernel_name)
