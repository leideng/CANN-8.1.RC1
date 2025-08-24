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
ones_like
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
@register_operator_compute("OnesLike", op_mode="dynamic", support_fusion=False)
def ones_like_compute(input_x, output_y, kernel_name="ones_like"):
    """
    Given a tensor, this operation returns a tensor of the same
    type and shape as `tensor` with all elements set to 1.

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: TVM tensor
        the placeholder of output data
    kernel_name : str
        cce kernel name, default value is "ones_like"

    Returns
    -------
    res: TVM tensor
        the result of ones_like_compute
    """
    src_dtype = input_x.dtype.lower()
    if src_dtype == "bfloat16":
        one = tvm.const(1, dtype="float16")
        one_src = tbe.broadcast(one, input_x.shape)
        one_src = tbe.cast_to(one_src, "float32")
        one_src = tbe.round(one_src, "bfloat16")
        return one_src

    src_dtype = "int8" if src_dtype == "bool" else src_dtype
    dst_type = src_dtype
    src_type_list = ("int8", "uint8")
    dst_type_list = ("int8", "uint8")
    if src_dtype in src_type_list:
        src_dtype = "float16"
    one = tvm.const(1, dtype=src_dtype)
    one_src = tbe.broadcast(one, input_x.shape)
    if src_dtype in dst_type_list:
        one_src = tbe.cast_to(one_src, dst_type, f1628IntegerFlag=True)
    else:
        one_src = tbe.cast_to(one_src, dst_type)

    return one_src


@register_operator("OnesLike")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def ones_like(x, y, kernel_name="ones_like"):
    """
    output a tensor of all one, shape and dtype is same of input

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32,
        int32,int8,uint8,bfloat16
    y: dict
        shape and dtype of output data
    kernel_name: str
        cce kernel name, default value is "ones_like"

    Returns
    ------
    None
    """
    dtype_input = x.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "int8", "uint8", "bfloat16", "bool")
    src_dtype = dtype_input
    para_check.check_dtype(src_dtype, check_list, param_name="x")
    schedules, tensors = [], []
    ins = classify([x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input", dtype=src_dtype)
            res = ones_like_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            auto_sch = tbe.auto_schedule(res)
        schedules.append(auto_sch)
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
