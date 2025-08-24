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
bn_infer_grad
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


class BNInferGradAttrInfo:
    """
    define BNInferGrad attr info
    """
    ATTR_EPSILON = OpAttr(0, "epsilon", "Float", 0.0000001)


def support_ub_fusion():
    """
    check ub fusion support
    """
    inputs = tbe_context.op_context.get_context().get_op_info()[0].inputs
    storage_format = inputs[0].get("format").upper()
    if storage_format == "NCHW":
        return False

    return True


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
# 'pylint: disable=locally-disabled,too-many-arguments
@register_operator_compute("BNInferGrad", op_mode="dynamic", support_fusion=support_ub_fusion, support_bfp16=True)
def bn_infer_grad_compute(grads, scale, batch_variance, x_backprop, epsilon, kernel_name="bn_infer_grad"):
    """
    Compute for bn_infer_grad_compute
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
    scale: TVM tensor 5D
        the placeholder of x. Must be one of the following
        type: `float32`.
    batch_variance: TVM tensor 5D
        the placeholder of batch_variance. Must be one of the following
        type: `float32`.
    x_backprop: dict
        dict of x_norm, A 5D Tensor for output x_norm.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_infer_grad"

    Returns
    -------
    res: x_backprop
    """
    is_cast = False
    if grads.dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        is_cast = True
        grads = tbe.cast_to(grads, "float32")

    data_adds = tbe.vadds(batch_variance, epsilon)
    data_rsqrt = tbe.vsqrt(data_adds)
    data_cast = tbe.broadcast(tvm.const(1, "float32"), batch_variance.shape)
    data_rsqrts = tbe.vdiv(data_cast, data_rsqrt)

    scale_mul = tbe.vmul(scale, data_rsqrts)
    scale_mul_broadcast = tbe.broadcast(scale_mul, grads.shape)
    res = tbe.vmul(scale_mul_broadcast, grads)
    if is_cast:
        res = tbe.cast_to(res, "float16")

    return res


def _update_shape(grads, scale, batch_variance):
    """
    update shape when format is NCHW
    """
    storage_format = grads.get("format").upper()
    if storage_format != "NCHW":
        # only NCHW will update dims
        return grads, scale, batch_variance

    # fill shape c to 1c11 base x
    storage_shape = list(grads.get("shape"))
    storage_range = list(grads.get("range", list()))
    fill_shape = storage_shape.copy()
    if not util_common.is_unknown_rank_input([grads]) and len(storage_shape) > 1:
        fill_shape = [1] * len(storage_shape)
        fill_range = [[1, 1]] * len(storage_shape)
        fill_shape[1] = storage_shape[1]
        fill_range[1] = [fill_shape[1], fill_shape[1]] if fill_shape[1] > 0 else storage_range[1]

        scale["shape"] = fill_shape
        batch_variance["shape"] = fill_shape
        scale["range"] = fill_range
        batch_variance["range"] = fill_range

    return grads, scale, batch_variance


@register_operator("BNInferGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_infer_grad(grads, scale, batch_variance, x_backprop, epsilon=0.0001, kernel_name="bn_infer_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_infer_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
    scale: dict
        dict of scale, A 5D Tensor for input scale.
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
        have the same shape with scale
    x_backprop: dict
        dict of x_backprop, A 5D Tensor for output x_backprop.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_infer_grad"

    Returns
    -------
    None
    """

    shape_grads = grads.get("shape")
    shape_scale = scale.get("shape")
    shape_batch_variance = batch_variance.get("shape")

    input_grads_dtype = grads.get("dtype").lower()
    input_scale_dtype = scale.get("dtype").lower()
    batch_variance_dtype = batch_variance.get("dtype").lower()

    para_check.check_dtype(input_grads_dtype, ("float32", "float16", "bfloat16"), param_name="grads")
    para_check.check_dtype(input_scale_dtype, ("float32",), param_name="scale")
    para_check.check_dtype(batch_variance_dtype, ("float32",), param_name="batch_variance")

    grads, scale, batch_variance = _update_shape(grads, scale, batch_variance)

    ins = classify([grads, scale, batch_variance], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_grads, _scale, _batch_variance) in ins:
        with tbe.compute():
            shape_grads, shape_scale, shape_batch_variance = shape_util.variable_shape(
                [_grads, _scale, _batch_variance])
            grads_input = tvm.placeholder(shape_grads, name="grads_input", dtype=input_grads_dtype)
            scale_input = tvm.placeholder(shape_scale, name="x_input", dtype=input_scale_dtype)
            batch_variance_input = tvm.placeholder(shape_batch_variance,
                                                   name="batch_variance_input",
                                                   dtype=batch_variance_dtype)
            epsilon_input = get_attr_by_cls(epsilon, BNInferGradAttrInfo.ATTR_EPSILON, batch_variance_dtype)

            res = bn_infer_grad_compute(grads_input,
                                        scale_input,
                                        batch_variance_input,
                                        x_backprop,
                                        epsilon_input,
                                        kernel_name=kernel_name)
            tensors.append([grads_input, scale_input, batch_variance_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
