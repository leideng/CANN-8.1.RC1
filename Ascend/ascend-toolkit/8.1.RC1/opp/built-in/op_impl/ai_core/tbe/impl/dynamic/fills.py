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
fills
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


class FillsAttrInfo:
    """
    define Fills attr info
    """
    ATTR_VALUE = OpAttr(0, "value", "Float")


# 'pylint: disable=invalid-name,unused-argument
@register_operator_compute("Fills", op_mode="dynamic", support_fusion=True)
def fills_compute(x, value, dtype, kernel_name="fills"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    value : a number of float or int
    dtype : string
        the type of input
    kernel_name : str
        kernel name, default value is "fills"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    value_var = get_attr_by_cls(value, FillsAttrInfo.ATTR_VALUE, dtype)
    res = tbe.broadcast(value_var, x.shape)
    return res


@register_operator("Fills")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def fills(x, y, value, kernel_name="fills"):
    """
    do  fills operation

    Parameters:
    ----------
    x : the dict of output
    y :  the dict of output
    value:  scalar  value,
    kernel_name : cce kernel name, default value is "fills"

    Returns
    -------
    None
    """
    # get the dtype
    dtype = x.get("dtype").lower()

    # check whether dtypes are right
    check_list = ("uint8", "int8", "int32", "int64", "float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x])[0]
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype)

            res = fills_compute(data_x, value, dtype)
            tensors.append([data_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
