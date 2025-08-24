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
bn_3d_training_reduce_grad
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpImplMode
from impl.bn_training_reduce_grad import bn_training_reduce_grad


# 'pylint: disable=unused-argument,too-many-locals
# 'pylint: disable=invalid-name,redefined-builtin,too-many-statements,too-many-arguments
@register_operator_compute("bn3d_training_reduce_grad", op_mode="static", support_fusion=True)
def bn3d_training_reduce_grad_compute(grads, x, diff_scale, diff_offset, scale,
                                      batch_mean, batch_variance, y, epsilon,
                                      kernel_name="bn3d_training_reduce_grad"):
    """
    Compute for batch_norm_train_reduce_grad
    y:(grads*scale*np.power((batch_variance + epsilon), (-0.5)))+
      np.sum(grads*scale*(-0.5)*x_norm*np.power((batch_variance+epsilon),(-1))))
      *(2/m)+np.sum(grads*scale*(-1)*
      np.power((batch_variance+epsilon),(-0.5)))*(1/m)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads.
        Must be one of the following type: `float16`, `float32`.
    x: TVM tensor 5D
        the placeholder of x.
        Must be one of the following type: `float32`, 'float16.
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
    shape_grads = shape_util.shape_to_list(grads.shape)
    num = shape_grads[0] * shape_grads[2] * shape_grads[3]
    num_rec = 1.0 / num
    is_cast = False
    if grads.dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        is_cast = True
        grads = tbe.cast_to(grads, "float32")

    if x.dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        x = tbe.cast_to(x, "float32")

    data_sqrt = tbe.vsqrt(tbe.vadds(batch_variance, epsilon), impl_mode=OpImplMode.HIGH_PERFORMANCE)
    scale_inv = tbe.vmuls(diff_scale, num_rec)
    scale_inv_reverse = tbe.vmuls(diff_scale, (-1.0)*num_rec)
    offset_inv_reverse = tbe.vmuls(diff_offset, (-1.0)*num_rec)

    multiplier = tbe.vdiv(scale_inv_reverse, data_sqrt)
    addend_div = tbe.vdiv(batch_mean, data_sqrt)
    addend_mul = tbe.vmul(addend_div, scale_inv)
    addend = tbe.vadd(addend_mul, offset_inv_reverse)

    multiplier_broadcast = tbe.broadcast(multiplier, shape_grads)
    addend_broadcast = tbe.broadcast(addend, shape_grads)

    coef_mul = tbe.vmul(multiplier_broadcast, x)
    coef_add = tbe.vadd(grads, coef_mul)
    coef = tbe.vadd(coef_add, addend_broadcast)

    mul_scale = tbe.vdiv(scale, data_sqrt)
    mul_scale_broadcast = tbe.broadcast(mul_scale, shape_grads)

    res = tbe.vmul(coef, mul_scale_broadcast)

    if is_cast:
        res = tbe.cast_to(res, "float16")
    return res


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn3d_training_reduce_grad(grads, x, diff_scale, diff_offset, scale,
                              batch_mean, batch_variance, y, epsilon=0.0001,
                              kernel_name="bn3d_training_reduce_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_training_reduce_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
        source data type, support "float32", "float16".
    x: dict
        dict of s, A 5D Tensor for input x.
        source data type, support "float32", "float16".
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
    bn_training_reduce_grad(grads, x, diff_scale, diff_offset, scale, batch_mean,
                            batch_variance, y, epsilon, None, None, kernel_name=kernel_name)
