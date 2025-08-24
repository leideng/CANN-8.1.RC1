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
dynamic equal
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import check_support_block_size_16
from impl.common_util import get_dtype
from impl.util_soc_common import is_v200


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-locals,redefined-argument-from-local
@register_operator_compute("Equal", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def equal_compute(input_x, input_y, output_z, kernel_name="equal"):
    """
    compute for equal

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x, has shape, dtype and range attributes
    input_y: TVM tensor
        the placeholder of input_y, has shape, dtype and range attributes
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "equal"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    # `define a scalar, value = 2**(-126), minimun num of float32 2**(-126)`
    scalar_min_fp32 = 2**(-126)
    # `define a scalar, value = 2**(50)`
    scalar_mul_fp32 = 2**50
    # `define a scalar, value = 2**(26)`
    scalar_mul2_fp32 = 2**26
    # `define a scalar, value = 2**(-24), minimun num of float16 2**(-24)`
    scalar_min_fp16 = 2**(-24)
    # `define a scalar, value = 2**(12)`
    scalar_mul_fp16 = 2**12
    # `define a scalar, value = 1`
    scalar_one_value = 1

    dtype_x = input_x.dtype
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_broad = shape_util.broadcast_shapes(shape_x,
                                                                shape_y,
                                                                param_name_input1="input_x",
                                                                param_name_input2="input_y")

    if check_support_block_size_16():
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16") 
        return equal_compute_with_cmp(input_x, input_y, shape_broad)

    type_range = ("int64", "uint64", "float32", "float16") if is_v200() else ("int64", "uint64")
    if dtype_x in type_range:
        res = equal_compute_with_cmp(input_x, input_y, shape_broad)
        return res

    if dtype_x == "int32" and tbe_platform.api_check_support("tbe.dsl.vcmp", dtype_x):
        return equal_compute_with_cmp(input_x, input_y, shape_broad)

    if dtype_x == "float32":
        scalar_min = tvm.const(scalar_min_fp32, dtype="float32")
        scalar_mul = tvm.const(scalar_mul_fp32, dtype="float32")
        scalar_mul1 = tvm.const(scalar_mul2_fp32, dtype="float32")
        scalar_one = tvm.const(-1 * scalar_one_value, dtype="float32")
    else:
        scalar_min = tvm.const(scalar_min_fp16, dtype="float16")
        scalar_mul = tvm.const(scalar_mul_fp16, dtype="float16")
        scalar_one = tvm.const(-1 * scalar_one_value, dtype="float16")

    if dtype_x in ("int8", "uint8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    x_brod = tbe.broadcast(input_x, shape_broad)
    y_brod = tbe.broadcast(input_y, shape_broad)

    res_vsub = tbe.vsub(x_brod, y_brod)
    if tbe_platform.api_check_support("te.lang.cce.vabs", res_vsub.dtype):
        res_vabs = tbe.vabs(res_vsub)
    else:
        res_vsub = tbe.cast_to(res_vsub, "float32")
        res_vabs = tbe.vabs(res_vsub)
    res_min = tbe.vmins(res_vabs, scalar_min)
    res_vmul = tbe.vmuls(res_min, scalar_mul)
    res_vmul1 = tbe.vmuls(res_vmul, scalar_mul)

    if dtype_x == "float32":
        res_vmul2 = tbe.vmuls(res_vmul1, scalar_mul1)
        res_vsub1 = tbe.vadds(res_vmul2, scalar_one)
        res_vabs1 = tbe.vabs(res_vsub1)
    else:
        res_vsub1 = tbe.vadds(res_vmul1, scalar_one)
        res_vabs1 = tbe.vabs(res_vsub1)

    res = tbe.cast_to(res_vabs1, "int8", True)
    return res


def equal_compute_with_cmp(input_x, input_y, shape_broad):
    """
    b64 compute for equal: input data type is int64 or uint64
    """
    input_x = tbe.broadcast(input_x, shape_broad)
    input_y = tbe.broadcast(input_y, shape_broad)
    res = tbe.vcmp(input_x, input_y, "eq", mode="bool")
    return res


@register_operator("Equal")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def equal(input_x, input_y, output_z, kernel_name="equal"):
    """
    Returns the truth value of (x = y) element-wise

    Parameters
    ----------
    input_x: dict
        dict{"shape":tuple or list,"dtype":str, range: tuple or list}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32,uint8,int8,bfloat16
    input_y: dict
        dict{"shape":tuple or list,"dtype":str, range: tuple or list}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32,uint8,int8,bfloat16
    output_z: dict, reserved field
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "equal"

    Returns
    -------
    None
    """

    # check input tensor data_type
    x_dtype = get_dtype(input_x)
    if x_dtype == "uint32":
        x_dtype = "int32"
        input_x["dtype"] = "int32"
    y_dtype = get_dtype(input_y)
    if y_dtype == "uint32":
        y_dtype = "int32"
        input_y["dtype"] = "int32"
    check_list = ["bfloat16", "float16", "float32", "int64", "uint64", "int32", "uint8", "int8"]
    if check_support_block_size_16():
        check_list.append("int16")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("equal", 'x_dtype', 'y_dtype', str(x_dtype), str(y_dtype))

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_x, input_y) in ins:
        with tbe.compute():
            x_shape, y_shape = shape_util.variable_shape([input_x, input_y])
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, y_dtype, "tensor_y")
            res = equal_compute(tensor_x, tensor_y, output_z, kernel_name)
            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors, "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
