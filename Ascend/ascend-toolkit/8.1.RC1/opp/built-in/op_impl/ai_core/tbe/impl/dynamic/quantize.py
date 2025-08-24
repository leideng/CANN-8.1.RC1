# Copyright 2021 Huawei Technologies Co., Ltd
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
dynamic sigmoid
"""
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.util_soc_common import after_v200


# 'pylint: disable=unused-argument,huawei-too-many-arguments,too-many-locals,invalid-name
@register_operator_compute("Quantize", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def quantize_compute(x, scales, zero_points, y, dtype, axis=1, kernel_name="quantize_per_channel"):
    dtype_input = x.dtype
    dtype_scales = scales.dtype
    if dtype_input != "float32":
        x = tbe.cast_to(x, "float32")
    if dtype_scales != "float32":
        scales = tbe.cast_to(scales, "float32")

    if zero_points is not None:
        zero_points = tbe.cast_to(zero_points, "float32")
        zero_points = tbe.broadcast(zero_points, x.shape)        
        scales = tbe.broadcast(scales, x.shape)
        x = tbe.vdiv(x, scales)
        x = tbe.vadd(x, zero_points)
    else:
        x_shape, y_shape, z_shape = shape_util.broadcast_shapes(x.shape,
                                                                scales.shape,
                                                                param_name_input1="x",
                                                                param_name_input2="scales")
        x = tbe.broadcast(x, z_shape)
        scales = tbe.broadcast(scales, z_shape)
        x = tbe.vdiv(x, scales)

    if (dtype == "torch.quint8") and tbe_platform.api_check_support("tbe.dsl.clip", "float32"):
        x = tbe.clip(x, 255, 0)
    x = tbe.round(x)
    if dtype == "torch.qint8":
        res = tbe.cast_to(x, "int8")
    elif dtype == "torch.quint8":
        res = tbe.cast_to(x, "uint8")
    else:
        res = tbe.cast_to(x, "int32")
    return res


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def quantize(x, scales, zero_points, y, dtype, axis=1, kernel_name="quantize_per_channel"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input_x
    scales : dict
        shape and dtype of input_scales
    zero_points : dict
        shape and dtype of input_zero_points
    y : dict
        shape and dtype of output_y, should be same shape as input, dtype is same as the quantified type
    axis: int
        the processed dim
    dtype:
        quantified type
    kernel_name : str
        kernel name, default value is "quantize"

    Returns
    -------
    None
    """

    dtype_x = x.get("dtype")
    dtype_scale = scales.get("dtype")
    check_x_tuple = ("float16", "float32", "bfloat16")
    check_scale_tuple = ("float32", "float16", "bfloat16")
    check_zp_tuple = ("int8", "uint8", "int32", "bfloat16")
    check_dtype = ("torch.qint8", "torch.quint8", "torch.qint32")
    if dtype_x not in check_x_tuple:
        raise RuntimeError("X only support %s while dtype is %s" % (",".join(check_x_tuple), dtype_x))
    if dtype_scale not in check_scale_tuple:
        raise RuntimeError("Scales only support %s while dtype is %s" % (",".join(check_scale_tuple), dtype_scale))
    if dtype not in check_dtype:
        raise RuntimeError("Dtype only support %s while dtype is %s" % (",".join(check_dtype), dtype))

    para_check.check_kernel_name(kernel_name)

    schedules, tensors = [], []
    if zero_points is not None:
        dtype_zp = zero_points.get("dtype")
        if dtype_zp not in check_zp_tuple:
            raise RuntimeError("Zero_points only support %s while dtype is %s" % (",".join(check_zp_tuple), dtype_zp))
        ins = classify([x, scales, zero_points], OpPatternMode.ELEWISE_WITH_BROADCAST)
        for (_x, s, z) in ins:
            with tbe.compute():
                shape_x, shape_scales, shape_zero_points = shape_util.variable_shape([_x, s, z])
                data_x = tvm.placeholder(shape_x, name="x", dtype=dtype_x)
                data_scales = tvm.placeholder(shape_scales, name="scales", dtype=dtype_scale)
                data_zero_points = tvm.placeholder(shape_zero_points, name="zero_points", dtype=dtype_zp)
                res = quantize_compute(data_x, data_scales, data_zero_points, y, dtype, axis, kernel_name)
                tensors.append([data_x, data_scales, data_zero_points, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    else:
        ins = classify([x, scales], OpPatternMode.ELEWISE_WITH_BROADCAST)
        for (_x, s) in ins:
            with tbe.compute():
                shape_x, shape_scales = shape_util.variable_shape([_x, s])
                data_x = tvm.placeholder(shape_x, name="x", dtype=dtype_x)
                data_scales = tvm.placeholder(shape_scales, name="scales", dtype=dtype_scale)
                res = quantize_compute(data_x, data_scales, None, y, dtype, axis, kernel_name)
                tensors.append([data_x, data_scales, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors
    }
    tbe.build(schedules, config)
