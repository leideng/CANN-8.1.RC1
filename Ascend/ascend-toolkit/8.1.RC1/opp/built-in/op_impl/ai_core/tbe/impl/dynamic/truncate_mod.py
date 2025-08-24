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
truncate_mod
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_soc_common import is_v200


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("TruncateMod", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def truncate_mod_compute(input_x, input_y, output_z,
                         kernel_name="truncate_mod"):
    """
    truncate_mod compute
    calculating data's truncatemod, res = x - truncate(x/y)*y

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "truncate_mod"

    Returns
    -------
    res: TVM tensor
        the result of truncate_mod(input_x,input_y)
    """
    shape_list = shape_util.broadcast_shapes(input_x.shape, input_y.shape,
                                             param_name_input1="input_x",
                                             param_name_input2="input_y")
    dtype = input_x.dtype
    tran_x = tbe.cast_to(input_x, "float32")
    tran_y = tbe.cast_to(input_y, "float32")
    data_x_broad = tbe.broadcast(tran_x, shape_list[2])
    data_y_broad = tbe.broadcast(tran_y, shape_list[2])

    vdiv_data = tbe.vdiv(data_x_broad, data_y_broad)
    if tbe_platform.api_check_support("tbe.dsl.trunc", "f322f32") and is_v200():
        cast_truncate_data = tbe.trunc(vdiv_data, "float32")
    else:
        truncate_data = tbe.cast_to(vdiv_data, "int32")
        cast_truncate_data = tbe.cast_to(truncate_data, "float32")
    mul_data = tbe.vmul(cast_truncate_data, data_y_broad)
    sub_data = tbe.vsub(data_x_broad, mul_data)
    res = tbe.cast_to(sub_data, dtype)

    return res


@register_operator("TruncateMod")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def truncate_mod(input_x, input_y, output_z, kernel_name="truncate_mod"):
    """
    algorithm: truncatemod
    calculating data's truncate, res = x - truncate(x/y)*y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "truncatemod"

    Returns
    -------
    None
    """
    dtype_x = input_x.get("dtype").lower()
    dtype_y = input_y.get("dtype").lower()

    check_list = ("bfloat16", "float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    para_check.check_dtype(dtype_y, check_list, param_name="input_y")

    if dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal("TruncateMod", "input_x", "input_y",
                                                              str(dtype_x), str(dtype_y))

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_x, _input_y) in ins:
        with tbe.compute():
            x_shape, y_shape = \
                shape_util.variable_shape([_input_x, _input_y])
            data1 = tvm.placeholder(x_shape, dtype=dtype_x, name="data1")
            data2 = tvm.placeholder(y_shape, dtype=dtype_y, name="data2")
            res = truncate_mod_compute(data1, data2, output_z, kernel_name)
            tensors.append([data1, data2, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensors
    }

    tbe.build(schedules, config)
