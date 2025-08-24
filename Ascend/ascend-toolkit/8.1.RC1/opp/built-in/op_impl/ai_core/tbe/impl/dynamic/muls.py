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
dynamic muls
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_compute import check_batchmatmul_fuse


# 'pylint: disable=too-few-public-methods
class MulsAttrInfo:
    """
    define attr info
    """
    ATTR_VALUE = OpAttr(0, "value", "Float")


def muls_compute_for_batchmatmul(lhs_tensor, const_tensor):
    """
    calculating data's muls, c = a * b
    Parameters
    ----------
    lhs_tensor: TVM tensor
        the placeholder of first input data
    const_value: Const tensor
    Returns
    -------
    res : output of the lhs_tensor + const_tensor
    """
    if "para_name" in lhs_tensor.op.attrs:
        para_name = lhs_tensor.op.attrs["para_name"]
        para_name += "_muls"
    else:
        para_name = "muls"
    batch_shape = shape_util.shape_to_list(lhs_tensor.op.attrs["batch_shape"])
    res = tbe.vmuls(lhs_tensor, const_tensor)
    res.op.attrs["batch_shape"] = batch_shape
    res.op.attrs["para_name"] = para_name

    return res


# 'pylint: disable=too-many-locals,unused-argument
@register_operator_compute("Muls", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def muls_compute(input_x, output_y, value, kernel_name="muls"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    scalar : a number of float or int
    kernel_name : str
        kernel name, default value is "muls"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    input_dtype = input_x.dtype
    # get tvm.var attr whether attr is None, else get tvm.const
    if input_dtype == "complex32":
        value = get_attr_by_cls(value, MulsAttrInfo.ATTR_VALUE, "float16")
    elif input_dtype == "complex64":
        value = get_attr_by_cls(value, MulsAttrInfo.ATTR_VALUE, "float32")
    else:
        value = get_attr_by_cls(value, MulsAttrInfo.ATTR_VALUE, input_dtype)
    batch_matmul_fuse_flag = check_batchmatmul_fuse(input_x)
    if batch_matmul_fuse_flag:
        return muls_compute_for_batchmatmul(input_x, value)
    res = tbe.vmuls(input_x, value)
    return res


@register_operator("Muls")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def muls(input_x, output_y, value, kernel_name="muls"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of output, should be same shape and type as x
    value : float
        scale
    kernel_name : str
        kernel name, default value is "muls"

    Returns
    -------
    None
    """
    x_dtype = input_x.get("dtype")
    input_dtype = x_dtype.lower()

    check_list = ["bfloat16", "float16", "float32", "int32", "int16", "int64", "complex32", "complex64"]
    para_check.check_dtype(x_dtype, check_list)
    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input", dtype=input_dtype)
            res = muls_compute(data_input, output_y, value)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
