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
bn_infer
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


class BNInferAttrInfo:
    """
    define BNInfer attr info
    """
    ATTR_EPSILON = OpAttr(0, "epsilon", "Float", 0.0000001)


def _check_dtype(dtype_x, dtype_scale, dtype_offset, dtype_mean, dtype_variance):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
    dtype_scale: str
        scale's data type
    dtype_offset: str
        offset's data type
    dtype_mean: str
        mean's data type
    dtype_variance: str
        variance's data type

    Returns
    -------
    None
    """
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32", "bfloat16"), param_name="x")
    para_check.check_dtype(dtype_scale.lower(), ("float32",), param_name="scale")
    para_check.check_dtype(dtype_offset.lower(), ("float32",), param_name="offset")
    para_check.check_dtype(dtype_mean.lower(), ("float32",), param_name="mean")
    para_check.check_dtype(dtype_variance.lower(), ("float32",), param_name="variance")


def support_ub_fusion():
    """
    check ub fusion support
    """
    inputs = tbe_context.op_context.get_context().get_op_info()[0].inputs
    storage_format = inputs[0].get("format").upper()
    if storage_format == "NCHW" or storage_format == "NCDHW":
        return False

    return True


# 'pylint: disable=locally-disabled,invalid-name,too-many-arguments
# 'pylint: disable=locally-disabled,too-many-locals,unused-argument
@register_operator_compute("BNInfer", op_mode="dynamic", support_fusion=support_ub_fusion, support_bfp16=True)
def bn_infer_compute(x, scale, offset, mean, variance, y, epsilon, kernel_name="bn_inf"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization for inference

    Parameters
    ----------
    x: TVM tensor
        contains x data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    res: TVM tensor
        the result of bn_training_update_v2 compute for inference
    """
    _, _, _, _, _, shape_max = shape_util.unify_broadcast_shapes(
        [x.shape, scale.shape, offset.shape, mean.shape, variance.shape])

    multiplier_add = tbe.vadds(variance, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)

    addend_mul = tbe.vmul(multiplier_div, mean)
    addend_sub = tbe.vsub(offset, addend_mul)

    # compute the batch normalization of x
    is_cast = False
    if x.dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        is_cast = True
        x = tbe.cast_to(x, "float32")
        multiplier_div = tbe.cast_to(multiplier_div, "float32")
        addend_sub = tbe.cast_to(addend_sub, "float32")

    multiplier_div = tbe.broadcast(multiplier_div, shape_max)
    addend_sub = tbe.broadcast(addend_sub, shape_max)
    x = tbe.broadcast(x, shape_max)
    res = tbe.vadd(tbe.vmul(multiplier_div, x), addend_sub)
    if is_cast:
        res = tbe.cast_to(res, "float16")

    return res


# pylint: disable=too-many-return-values
def _update_shape(x, scale, offset, mean, variance):
    """
    update shape when format is NCHW
    """
    storage_format = x.get("format").upper()
    if storage_format != "NCHW" and storage_format != "NCDHW":
        # only NCHW will update dims
        return x, scale, offset, mean, variance

    # fill shape c to 1c11 base x
    storage_shape = list(x.get("shape"))
    storage_range = list(x.get("range", list()))
    fill_shape = storage_shape.copy()
    fill_range = list()

    def check_c_dim(in_name, in_shape):
        storage_ori_format = x.get("ori_format").upper()
        if storage_ori_format != "NCHW" and storage_ori_format != "NCDHW":
            return
        x_ori_shape = x.get("ori_shape")
        if len(in_shape) == 1 and len(x_ori_shape) > 1 and in_shape[0] != x_ori_shape[1]:
            error_detail = f"{in_name} c dim_value: {in_shape[0]} should equal input_x c dim_value: {x_ori_shape[1]}"
            error_manager_vector.raise_err_input_shape_invalid("BNInfer", in_name, error_detail)

    if not util_common.is_unknown_rank_input([x]) and len(storage_shape) > 1:
        scale_ori_shape = scale.get("ori_shape")
        offset_ori_shape = offset.get("ori_shape")
        mean_ori_shape = mean.get("ori_shape")
        variance_ori_shape = variance.get("ori_shape")
        check_c_dim("scale", scale_ori_shape)
        check_c_dim("offset", offset_ori_shape)
        check_c_dim("mean", mean_ori_shape)
        check_c_dim("variance", variance_ori_shape)
        fill_shape = [1] * len(storage_shape)
        fill_range = [[1, 1]] * len(storage_shape)
        fill_shape[1] = storage_shape[1]
        fill_range[1] = [fill_shape[1], fill_shape[1]] if fill_shape[1] > 0 else storage_range[1]

        scale["shape"] = fill_shape
        offset["shape"] = fill_shape
        mean["shape"] = fill_shape
        variance["shape"] = fill_shape

        scale["range"] = fill_range
        offset["range"] = fill_range
        mean["range"] = fill_range
        variance["range"] = fill_range

    return x, scale, offset, mean, variance


@register_operator("BNInfer")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_infer(x, scale, offset, mean, variance, y, epsilon, kernel_name="bn_infer"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization for inference

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    scale: dict
        dict of scale, A 5HD Tensor for scale.
        can be broadcast to x.shape
    offset: dict
        dict of offset, A 5HD Tensor for offset.
        can be broadcast to x.shape
    mean: dict
        dict of mean, A `Tensor`.
        dict of scale, A 5HD Tensor for mean.
        can be broadcast to x.shape
    variance: dict
        dict of batch_variance, A `Tensor`.
        dict of offset, A 5HD Tensor for variance.
        can be broadcast to x.shape
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_infer"

    Returns
    -------
    None
    """

    dtype_x = x.get("dtype").lower()
    dtype_scale = scale.get("dtype").lower()
    dtype_offset = offset.get("dtype").lower()
    dtype_mean = mean.get("dtype").lower()
    dtype_variance = variance.get("dtype").lower()

    data_format = x.get("format").upper()
    data_format_list = ("NC1HWC0", "NCHW", "NHWC", "NCDHW")
    if data_format not in data_format_list:
        error_manager_vector.raise_err_check_params_rules("BNInfer", "Format only support 5HD,NCHW,NHWC,NCDHW", "x",
                                                          data_format)

    x, scale, offset, mean, variance = _update_shape(x, scale, offset, mean, variance)

    _check_dtype(dtype_x, dtype_scale, dtype_offset, dtype_mean, dtype_variance)

    extra_params = {"same_input_shape_group": [[0], [1, 2, 3, 4]]}
    ins = classify([x, scale, offset, mean, variance], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []
    for (_x, _scale, _offset, _mean, _variance) in ins:
        with tbe.compute():
            shape_list = shape_util.variable_shape([_x, _scale, _offset, _mean, _variance])
            x_input = tvm.placeholder(shape_list[0], name="x_input", dtype=dtype_x.lower())
            scale_input = tvm.placeholder(shape_list[1], name="scale_input", dtype=dtype_scale.lower())
            offset_input = tvm.placeholder(shape_list[2], name="offset_input", dtype=dtype_offset.lower())
            mean_input = tvm.placeholder(shape_list[3], name="mean_input", dtype=dtype_mean.lower())
            variance_input = tvm.placeholder(shape_list[4], name="variance_input", dtype=dtype_variance.lower())
            epsilon_input = get_attr_by_cls(epsilon, BNInferAttrInfo.ATTR_EPSILON, dtype_variance)
            res = bn_infer_compute(x_input,
                                   scale_input,
                                   offset_input,
                                   mean_input,
                                   variance_input,
                                   y,
                                   epsilon_input,
                                   kernel_name=kernel_name)
            tensors.append([x_input, scale_input, offset_input, mean_input, variance_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
