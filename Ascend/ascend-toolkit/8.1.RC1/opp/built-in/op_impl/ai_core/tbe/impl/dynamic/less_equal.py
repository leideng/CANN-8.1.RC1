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
less_equal
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util_soc_common import is_v200


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for Constant
    """
    SCALAR_MIN_FP32 = 2**(-126)
    SCALAR_MUL_FP32 = 2**50
    SCALAR_MUL2_FP32 = 2**26
    SCALAR_MIN_FP16 = 2**(-24)
    SCALAR_MUL_FP16 = 2**12


def less_equal_compute_with_cmp(input_x, input_y, shape_broadcast):
    """
    b64 compute for less_equal: input data type is int64 or uint64
    """
    input_x = tbe.broadcast(input_x, shape_broadcast)
    input_y = tbe.broadcast(input_y, shape_broadcast)

    res = tbe.vcmp(input_x, input_y, "le", "bool")

    return res


def less_equal_compute_b32(input_x, input_y, shape_broadcast):
    """
    b32 compute for greater_equal: input data type is int32
    """
    input_x = tbe.broadcast(input_x, shape_broadcast)
    input_y = tbe.broadcast(input_y, shape_broadcast)
    res_min = tbe.vmin(input_x, input_y)
    res = tbe.vcmp(input_x, res_min, "eq", "bool")

    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
# 'pylint: disable=redefined-argument-from-local
@register_operator_compute("LessEqual", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def less_equal_compute(input_x, input_y, output_z, kernel_name="less_equal"):
    """
    compute for less_equal

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "less_equal"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_x = input_x.dtype
    shape_x, shape_y, shape_broadcast = shape_util.broadcast_shapes(input_x.shape,
                                                                    input_y.shape,
                                                                    param_name_input1="input_x",
                                                                    param_name_input2="input_y")

    type_range = ("int64", "uint64", "float32", "float16") if is_v200() else ("int64", "uint64")
    if dtype_x in type_range:
        return less_equal_compute_with_cmp(input_x, input_y, shape_broadcast)
    elif dtype_x == "int32" and tbe_platform.api_check_support("tbe.dsl.vcmp", "int32"):
        return less_equal_compute_b32(input_x, input_y, shape_broadcast)
    elif dtype_x == "float32":
        scalar_min = tvm.const(Constant.SCALAR_MIN_FP32, dtype="float32")
        scalar_mul = tvm.const(Constant.SCALAR_MUL_FP32, dtype="float32")
        scalar_mul1 = tvm.const(Constant.SCALAR_MUL2_FP32, dtype="float32")
        scalar_neg_one = tvm.const(-1, dtype="float32")
    else:
        scalar_min = tvm.const(Constant.SCALAR_MIN_FP16, dtype="float16")
        scalar_mul = tvm.const(Constant.SCALAR_MUL_FP16, dtype="float16")
        scalar_neg_one = tvm.const(-1, dtype="float16")

    if dtype_x in ("int8", "uint8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    input_x = tbe.broadcast(input_x, shape_broadcast)
    input_y = tbe.broadcast(input_y, shape_broadcast)

    res_max = tbe.vmax(input_x, input_y)
    res_vsub = tbe.vsub(input_y, res_max)
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
        res_vsub1 = tbe.vadds(res_vmul2, scalar_neg_one)
        res_vabs1 = tbe.vabs(res_vsub1)
    else:
        res_vsub1 = tbe.vadds(res_vmul1, scalar_neg_one)
        res_vabs1 = tbe.vabs(res_vsub1)

    res = tbe.cast_to(res_vabs1, "int8", True)

    return res


@register_operator("LessEqual")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def less_equal(input_x, input_y, output_z, kernel_name="less_equal"):
    """
    Returns the truth value of (x <= y) element-wise

    Parameters
    ----------
    input_x: dict
        dict{"shape":tuple or list, "dtype":str, range: tuple or list},
        shape, range, and dtype of first input,
        support float16,float32,int32,int8,uint8,int64,uint64
    input_y: dict
        dict{"shape":tuple or list, "dtype":str, range: tuple or list},
        shape, range, and dtype of first input,
        support float16,float32,int32,int8,uint8,int64,uint64
    output_z: dict
        dict of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is "less_equal"

    Returns
    -------
    None
    """
    # check input tensor data_type
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32", "int32", "uint8", "int8", "int64", "uint64")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal('less_equal', 'input_x', 'input_y', str(x_dtype),
                                                              str(y_dtype))

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_x, input_y) in ins:
        with tbe.compute():
            # shape
            x_shape, y_shape = shape_util.variable_shape([input_x, input_y])

            # less_equal compute
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, y_dtype, "tensor_y")
            res = less_equal_compute(tensor_x, tensor_y, output_z, kernel_name)

            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
