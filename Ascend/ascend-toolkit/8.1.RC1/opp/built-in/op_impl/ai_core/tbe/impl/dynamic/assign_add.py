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
dynamic assign_add
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("AssignAdd", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def assign_add_compute(ref, value, output, kernel_name="assign_add"):
    """
    implementation of operator assign_add

    Parameters
    ----------
    ref: placeholder
        contains ref data
    value: placeholder
        contains value data
    output: placeholder
        save the result data.
    kernel_name: str
        kernel name, default value is "assign_add"
    """
    ref_dtype = ref.dtype.lower()
    value_dtype = value.dtype.lower()
    is_mix_dtype = ref_dtype != value_dtype

    if ref_dtype in ("int8", "uint8"):
        ref = tbe.cast_to(ref, "float16")
        value = tbe.cast_to(value, "float16")

    if is_mix_dtype:
        value = tbe.cast_to(value, "float32")

    res = tbe.vadd(ref, value)

    if ref_dtype in ("int8", "uint8"):
        res = tbe.cast_to(res, ref_dtype)

    return res


@register_operator("AssignAdd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def assign_add(ref, value, output, kernel_name="assign_add"):
    """
    algorithm: assign_add
    update ref by adding value to it
    calculating data's add, a = a + b

    Parameters
    ----------
    ref: dict
        dict of input_ref, include shape and dtype,
    value: dict
        dict of input_value, include shape and dtype,
        Must have the same shape and dtype as input_ref
    output: dict
        dict of output
    kernel_name : str
        cce kernel name, default value is assign_add

    Returns
    -------
    None
    """
    ref_dtype = ref.get("dtype").lower()
    value_dtype = value.get("dtype").lower()
    output_dtype = output.get("dtype").lower()
    mix_dtype_list = (("float32", "float16", "float32"), ("float32", "bfloat16", "float32"))
    is_valid_mix_dtpye = (ref_dtype, value_dtype, output_dtype) in mix_dtype_list
    if not is_valid_mix_dtpye and ref_dtype != value_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("assign_add", "ref", "value",
                                                              str(ref_dtype), str(value_dtype))

    ins = classify([ref, value], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_ref, _value) in ins:
        with tbe.compute():
            shape_ref, shape_value = shape_util.variable_shape([_ref, _value])

            data_ref = tvm.placeholder(shape_ref, name="data_ref", dtype=ref_dtype)
            data_value = tvm.placeholder(shape_value, name="data_value", dtype=value_dtype)

            res = assign_add_compute(data_ref, data_value, output, kernel_name)

            tensor_list = [data_ref, data_value, res]
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
