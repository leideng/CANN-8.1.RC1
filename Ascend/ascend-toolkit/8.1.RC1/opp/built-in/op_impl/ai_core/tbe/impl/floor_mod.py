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
floor_mod
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context


class Constant:
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
        input tensor has shape and dtype attributes
    x2: TVM tensor
        input tensor has shape and dtype attributes
    y: dict
        dict with keys(shape, dtype) of output
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


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# 'pylint: disable=unused-variable
@register_operator_compute("floor_mod", op_mode="static", support_fusion=True)
def floor_mod_compute(x1, x2, y, kernel_name="floor_mod", impl_mode="high_performance"):
    """
    Compute remainder of division
    res= x1 - floor(input_data_x / input_data_y) * input_data_y

    Parameters
    ----------
    x1: TVM tensor
        input tensor has shape and dtype attributes
    x2: TVM tensor
        input tensor has shape and dtype attributes
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """
    # calculate result, using float32 for better precision
    dtype = x1.dtype
    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    shape_x, shape_y, shape = shape_util.broadcast_shapes(shape_x, shape_y,
                                                         param_name_input1="x1",
                                                         param_name_input2="x2")

    def _mod(x1, x2):
        has_improve_precision = False
        input_x_fp32 = x1
        input_y_fp32 = x2
        if tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
            input_x_fp32 = tbe.cast_to(x1, "float32")
            input_y_fp32 = tbe.cast_to(x2, "float32")
            has_improve_precision = True

        input_x_fp32 = tbe.broadcast(input_x_fp32, shape)
        input_y_fp32 = tbe.broadcast(input_y_fp32, shape)

        res_quot = tbe.vdiv(input_x_fp32, input_y_fp32)

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
        else:
            x2_broad = tbe.broadcast(x2, shape)
            x1_broad = tbe.broadcast(x1, shape)
            result = tbe.vmul(res_quot, x2_broad)
            res_rem = tbe.vsub(x1_broad, result)

        return res_quot, res_rem

    if impl_mode == "high_performance":
        _, res = _mod(x1, x2)
    else:
        # x1 can not be converted to a fp32 number absolute equality when its dtype is int32 and value is bigeer
        # than 2^24 sometimes, so we use 2^24 as an intermediate constant to get the exact result.
        fp32_max_valid_tensor = tbe.broadcast(Constant.FP32_MAX_VALID, shape)
        quot_x_tmp, res_x_tmp = _mod(x1, fp32_max_valid_tensor)
        quot_tmp_y, res_tmp_y = _mod(fp32_max_valid_tensor, x2)
        res = tbe.vmul(quot_x_tmp, res_tmp_y)
        res = tbe.vadd(res, res_x_tmp)
        _, res = _mod(res, x2)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def floor_mod(x1, x2, y, kernel_name="floor_mod", impl_mode="high_performance"):
    """
    calculate the remainder of division, support fp16,fp32,int32
    res = x1 -floor(input_data_x / input_data_y)* input_data_y

    Parameters
    ----------
    x1: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32
    x2: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32
    y: dict, reserved field
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "floor_mod"
    impl_mode: str
        impl_mode, default value is "high_performance"

    Returns
    ------
    None
    """
    # get dtype and shape attributes
    dtype_x = x1.get("dtype").lower()
    shape_x = x1.get("shape")
    dtype_y = x2.get("dtype").lower()
    shape_y = x2.get("shape")

    # check_kernel_name & shape
    para_check.check_shape(shape_x, param_name="x1")
    para_check.check_shape(shape_y, param_name="x2")

    # check input tensor data_type
    check_list = ("float16", "float32", "int32")
    para_check.check_dtype(dtype_x, check_list, param_name="x1")
    para_check.check_dtype(dtype_y, check_list, param_name="x2")

    if dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x1", "x2", dtype_x, dtype_y)

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                              param_name_input1="x1",
                                                              param_name_input2="x2")
    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)

    input_data_x = tvm.placeholder(shape_x, name="input_data_x", dtype=dtype_x)
    input_data_y = tvm.placeholder(shape_y, name="input_data_y", dtype=dtype_y)
    res = floor_mod_compute(input_data_x, input_data_y, y, kernel_name, impl_mode)
    with tvm.target.cce():
        auto_sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, input_data_y, res]}
    build(auto_sch, config)
