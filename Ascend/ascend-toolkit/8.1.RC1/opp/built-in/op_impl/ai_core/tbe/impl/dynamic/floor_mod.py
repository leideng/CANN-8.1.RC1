# Copyright 2020 Huawei Technologies Co., Ltd
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
dynamic floor_mod
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_soc_common import after_v200
from impl.util.util_soc_common import is_support_inf_nan


class Constant(object):
    """
    Define constant in this class
    """
    FP32_MAX_VALID = 2 ** 24


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=invalid-name,redefined-builtin,too-many-locals,unused-argument,no-else-raise,unnecessary-lambda
def check_supported(x1, x2, y, kernel_name="floor_mod", impl_mode="high_performance"):
    """

    Parameters
    ----------
    x1: TVM tensor
        input tensor has shape, dtype and range attributes
    x2: TVM tensor
        input tensor has shape, dtype and range attributes
    y: dict
        dict with keys(shape, dtype and range) of output
    kernel_name : str
        cce kernel name, default value is "floor_mod"
    impl_mode : assign high_performance or high_precision

    Returns
    -------
    True or False
    """

    dtype = x1.get("dtype").lower()
    context = tbe_context.op_context.get_context()
    impl_mode = context.get_addition("op_impl_mode_dict").get("FloorMod")

    reason = "When the dtype is float32 and impl_mode is high_precision, " \
             "calculation will be implemented by aicpu."
    if dtype == "float32" and impl_mode == "high_precision":
        return False, reason
    else:
        return True, ""


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=too-many-locals,redefined-argument-from-local,too-many-statements
@register_operator_compute("FloorMod", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def floor_mod_compute(x1, x2, y, kernel_name="floor_mod", impl_mode="high_performance"):
    """
    Compute remainder of division
    res = x1 - floor(input_data_x / input_data_y) * input_data_y

    Parameters
    ----------
    x1: TVM tensor
        input tensor has shape, dtype and range attributes
    x2: TVM tensor
        input tensor has shape, dtype and range attributes
    y: dict
        dict with keys(shape, dtype and range) of output
    kernel_name : str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """

    dtype = x1.dtype
    _, _, shape = shape_util.broadcast_shapes(x1.shape, x2.shape,
                                              param_name_input1="x1",
                                              param_name_input2="x2")

    if dtype == "int64":
        x1 = tbe.broadcast(x1, shape)
        x2 = tbe.broadcast(x2, shape)
        res = tbe.vmod(x1, x2)
        return res
    
    # calculate result, using float32 for better precision
    def _mod(x1, x2):
        has_improve_precision = False
        input_x_fp32 = x1
        input_y_fp32 = x2
        if tbe_platform.api_check_support("te.lang.cce.vdiv",
                                          "float32"):
            input_x_fp32 = tbe.cast_to(x1, "float32")
            input_y_fp32 = tbe.cast_to(x2, "float32")
            has_improve_precision = True

        input_x_fp32 = tbe.broadcast(input_x_fp32, shape)
        input_y_fp32 = tbe.broadcast(input_y_fp32, shape)

        res_quot = tbe.vdiv(input_x_fp32, input_y_fp32)

        if tbe_platform.api_check_support("te.lang.cce.floor", "f322f32") and \
            dtype != "int32":
            res_quot = tbe.floor(res_quot, "float32")
        elif tbe_platform.api_check_support("te.lang.cce.floor",
                                          res_quot.dtype):
            res_quot = tbe.floor(res_quot)
        else:
            res_quot = tbe.cast_to(res_quot, "float16")
            res_quot = tbe.floor(res_quot)

        if dtype != "int32":
            if has_improve_precision:
                result = tbe.cast_to(res_quot, "float32")
            else:
                result = tbe.cast_to(res_quot, "float16")
            result = tbe.vmul(result, input_y_fp32)
            res_rem = tbe.vsub(input_x_fp32, result)
            if has_improve_precision:
                res_rem = tbe.cast_to(res_rem, dtype)
            res_quot = tbe.cast_to(res_quot, dtype)
        else:
            x2_broad = tbe.broadcast(x2, shape)
            x1_broad = tbe.broadcast(x1, shape)
            result = tbe.vmul(res_quot, x2_broad)
            res_rem = tbe.vsub(x1_broad, result)
            
        return res_quot, res_rem

    if impl_mode == "high_performance" or (dtype in ["float16", "float32"]):
        _, res = _mod(x1, x2)
        if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend910B", "Ascend910_93") and \
           dtype != "int32":
            tensor_one = tbe.broadcast(tvm.const(1.0, dtype), shape)
            tensor_zero = tbe.broadcast(tvm.const(0.0, dtype), shape)
            tensor_inf = tbe.broadcast(tvm.const(float("inf"), dtype), shape)
            const_nan = tvm.const(float("nan"), dtype)
            x1 = tbe.broadcast(x1, shape)
            x2 = tbe.broadcast(x2, shape)

            abs_x2 = tbe.vabs(x2)
            x2_mask = tbe.vcmp(abs_x2, tensor_inf, "eq", "bit")
            res = tbe.vsel(x2_mask, x1, res)

            abs_x1 = tbe.vabs(x1)
            x1_mask = tbe.vcmp(abs_x1, tensor_inf, "eq", "bit")
            res = tbe.vsel(x1_mask, const_nan, res)

            res_add_x2 = tbe.vadd(res, x2)

            res_not_zero_mask = tbe.vcmp(res, tensor_zero, "ne", "bit")
            res_not_zero = tbe.vsel(res_not_zero_mask, tensor_one, tensor_zero)
            x2_signbit = tbe.vsignbit(x2)
            res_signbit = tbe.vsignbit(res)
            sign_diff = tbe.vsub(x2_signbit, res_signbit)
            res_not_zero_sign_diff_x2 = tbe.vmul(res_not_zero, sign_diff)
            res_mask = tbe.vcmp(res_not_zero_sign_diff_x2, tensor_zero, "ne", "bit")
            res = tbe.vsel(res_mask, res_add_x2, res)
    else:
        # x1 can not be converted to a fp32 number absolute equality when its dtype is int32 and value is bigeer
        # than 2^24 sometimes, so we use 2^24 as an intermediate constant to get the exact result.
        fp32_max_valid_tensor = tbe.broadcast(Constant.FP32_MAX_VALID, shape)
        quot_x_tmp, res_x_tmp = _mod(x1, fp32_max_valid_tensor)
        _, res_tmp_y = _mod(fp32_max_valid_tensor, x2)
        res = tbe.vmul(quot_x_tmp, res_tmp_y)
        res = tbe.vadd(res, res_x_tmp)
        _, res = _mod(res, x2)

    return res


@register_operator("FloorMod")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def floor_mod(x1, x2, y, kernel_name="floor_mod", impl_mode="high_performance"):
    """
    calculate the remainder of division, support fp16,fp32,int32
    res = x1 -floor(input_data_x / input_data_y)* input_data_y

    Parameters
    ----------
    x1: dict
        dict{"shape":tuple or list,"dtype":str, "range": tuple or list}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32,bf16
    x2: dict
        dict{"shape":tuple or list,"dtype":str, "range": tuple or list}
        shape of data
        the data type, src_dtype equals  of dst_dtype, support fp16,fp32,int32,bf16
    y: dict, reserved field
        dict with keys(shape, dtype and range) of output
    kernel_name: str
        cce kernel name, default value is "floor_mod"
    impl_mode: str
        impl_mode, default value is "high_performance"

    Returns
    ------
    None
    """

    # check input tensor data_type
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    check_list = ["float16", "float32", "int32", "bfloat16"]
    if tbe_platform.api_check_support("tbe.dsl.vmod", "int64"):
        check_list.append("int64")

    para_check.check_dtype(dtype_x, check_list, param_name="x1")
    para_check.check_dtype(dtype_y, check_list, param_name="x2")
    if dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal("floor_mod", 'x1', 'x2', str(dtype_x), str(dtype_y))

    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe.compute():
            shape_x, shape_y = shape_util.variable_shape([x1, x2])
            input_data_x = tvm.placeholder(shape_x, name="input_data_x",
                                           dtype=dtype_x)
            input_data_y = tvm.placeholder(shape_y, name="input_data_y",
                                           dtype=dtype_y)
            res = floor_mod_compute(input_data_x, input_data_y, y, kernel_name, impl_mode)

            tensors.append([input_data_x, input_data_y, res])
        with tvm.target.cce():
            auto_sch = tbe.auto_schedule(res)
        schedules.append(auto_sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
