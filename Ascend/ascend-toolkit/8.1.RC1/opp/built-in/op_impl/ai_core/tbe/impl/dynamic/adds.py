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
dynamic adds
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


class AddsAttrInfo:
    """
    define attr info
    """
    ATTR_VALUE = OpAttr(0, "value", "Float")


# 'pylint: disable=locally-disabled,invalid-name,unused-argument,too-many-locals
@register_operator_compute("Adds", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def adds_compute(x, y, value, kernel_name="adds"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    y : dict
        dict of output
    value : a number of float or int
    kernel_name : str
        kernel name, default value is "adds"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    dtype = x.dtype
    if dtype == "int16":
        x = tbe.cast_to(x, "int32")
    value = get_attr_by_cls(value, AddsAttrInfo.ATTR_VALUE, x.dtype)
    res = tbe.vadds(x, value)
    if dtype == "int16":
        res = tbe.cast_to(res, dtype)
    return res


@register_operator("Adds")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def adds(x, y, value, kernel_name="adds"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    value: a number of float
    kernel_name : str
        kernel name, default value is "adds"

    Returns
    --------
    None
    """
    dtype = x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16", "int32", "int64")
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend910B", "Ascend910_93"):
        check_list += ("int16",)
    para_check.check_dtype(dtype, check_list, param_name="x")
    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            shape = shape_util.variable_shape([_x])[0]
            data_input = tvm.placeholder(shape, name="data_input", dtype=dtype)
            res = adds_compute(data_input, y, value)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
    