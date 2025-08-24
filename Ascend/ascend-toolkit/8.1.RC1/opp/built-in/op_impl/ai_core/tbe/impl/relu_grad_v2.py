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
relu_grad_v2
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import tbe_platform
from impl.dynamic.relu_grad_v2 import get_op_support_info as relu_get_op_support_info


# 'pylint: disable=locally-disabled,too-many-argument,unused-argument,invalid-name
def get_op_support_info(gradients, mask, backprops, kernel_name="relu_grad_v2"):
    """
    get_op_support_info
    """
    return relu_get_op_support_info(gradients, mask, backprops, kernel_name="relu_grad_v2")


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("relu_grad_v2", op_mode="static", support_fusion=True)
def relu_grad_v2_compute(gradients, mask, backprops, kernel_name="relu_grad_v2", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).

    Parameters
    ----------
    gradients: TVM tensor
        input tensor of grad
    mask: TVM tensor
        input tensor of relu output
    backprops: dict
        output dict of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad_v2"

    Returns
    -------
    res: TVM tensor
        the result of relu_grad_compute
    """
    dtype = gradients.dtype
    trans_type = dtype

    # enable_fp32_high_percision: condition 1 and condition 2
    # static op don't support high_precision
    # condition 1: not milan soc, because milan soc support high_percision
    # condition 2. impl_mode equal OpImplMode.HIGH_PRECISION
    enable_fp32_high_percision = not tbe_platform.api_check_support("tik.vcopy") and \
                                impl_mode == OpImplMode.HIGH_PRECISION

    if dtype in ("float32",) and enable_fp32_high_percision:
        ones_tensor = tbe.broadcast(tvm.const(1, "float16"), gradients.shape)
        zeros_tensor = tbe.broadcast(tvm.const(0, "float16"), gradients.shape)
        gradients_mask = tbe.vsel(mask, ones_tensor, zeros_tensor)
        gradients_mask = tbe.cast_to(gradients_mask, "float32")
        result = tbe.vmul(gradients_mask, gradients)
        return result

    # need cast int8 or uint8 to float16
    if dtype in ("int8", "uint8"):
        gradients = tbe.cast_to(gradients, "float16")
        trans_type = "float16"

    result = tbe.vsel(mask, gradients, tvm.const(0, trans_type))

    # cast int8 or uint8 back
    if dtype in ("int8", "uint8"):
        result = tbe.cast_to(result, dtype, f1628IntegerFlag=True)

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def relu_grad_v2(gradients, mask, backprops, kernel_name="relu_grad_v2", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).
    support dtype:float16,float32,int32,int8,uint8

    Parameters
    ----------
    gradients: dict
        dict of grad
    mask: dict
        dict of relu output mask
    backprops: dict
        output of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad_v2"

    Returns
    -------
    None
    """
    shape_input_gradients = gradients.get("shape")
    shape_input_features = mask.get("shape")

    para_check.check_shape(shape_input_gradients, param_name="gradients")
    para_check.check_shape(shape_input_features, param_name="mask")

    dtype_input_gradients = gradients.get("dtype").lower()
    dtype_input_features = mask.get("dtype").lower()

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(dtype_input_gradients, check_list, param_name="gradients")
    para_check.check_dtype(dtype_input_features, ("uint8"), param_name="mask")

    shape_in = list(shape_input_gradients)

    # make sure the last dim of input feature is 8's multipules
    if shape_in[-1] % 8 != 0:
        shape_in[-1] = (shape_in[-1] + 7) // 8 * 8

    data_input_gradients = tvm.placeholder(tuple(shape_in), name="data_input_gradients", dtype=dtype_input_gradients)
    data_input_features = tvm.placeholder(shape_input_features, name="data_input_features", dtype=dtype_input_features)

    res = relu_grad_v2_compute(data_input_gradients, data_input_features, backprops, kernel_name, impl_mode)
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input_gradients, data_input_features, res]}
    build(sch, config)
