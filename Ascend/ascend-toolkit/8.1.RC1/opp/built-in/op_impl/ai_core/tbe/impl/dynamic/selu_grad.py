# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
selu_grad dynamic   
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# define product of scale
SCALE = 1.0507009873554804934193349852946
# define product of scale and alpha
SCALE_ALPHA_PRODUCT = 1.7580993408473768599402175208123


@register_operator_compute("SeluGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def selu_grad_compute(input_gradients, input_outputs, y, kernel_name="selu_grad"):
    """
    Computes SeluGrad backprops: `gradients * (outputs + scale * alpha)` 
    if outputs < 0, `scale * gradients`
    SCALE = 1.0507009873554804934193349852946
    SCALE_ALPHA_PRODUCT = 1.7580993408473768599402175208123

    Parameters
    ----------
    input_gradients: TVM tensor
        input tensor has shape and dtype attributes
    input_outputs: TVM tensor
        input tensor has shape and dtype attributes
    y: TVM tensor
        output tensor has shape and dtype attributes
    kernel_name : str
        cce kernel name, default value is "selu_grad"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """
    input_data = input_gradients
    dtype = input_data.dtype

    if dtype in ("int8", "uint8"):
        input_gradients = tbe.cast_to(input_gradients, "float16")
        input_outputs = tbe.cast_to(input_outputs, "float16")
        type_tmp = "float16"
    elif dtype in ("int32",):
        input_gradients = tbe.cast_to(input_gradients, "float32")
        input_outputs = tbe.cast_to(input_outputs, "float32")
        type_tmp = "float32"    
    else:
        type_tmp = dtype

    tensor_zero = tbe.broadcast(tvm.const(0, dtype=type_tmp), input_data.shape)
    adds_out_sa = tbe.vadds(input_outputs, tvm.const(SCALE_ALPHA_PRODUCT, dtype=type_tmp))
    mul_gra_outsa = tbe.vmul(input_gradients, adds_out_sa)
    muls_s_gra = tbe.vmuls(input_gradients, tvm.const(SCALE, dtype=type_tmp))
    greater_res = tbe.vcmpsel(input_outputs, tensor_zero, "lt", tensor_zero, muls_s_gra)
    smaller_res = tbe.vcmpsel(input_outputs, tensor_zero, "lt", mul_gra_outsa, tensor_zero)
    res = tbe.vadd(greater_res, smaller_res)

    # cast int8 or uint8 back
    if dtype in ("int8", "uint8", "int32"):
        res = tbe.cast_to(res, dtype, f1628IntegerFlag=True)

    return res


@register_operator("SeluGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def selu_grad(input_gradients, input_outputs, y, kernel_name="selu_grad"):
    """
    Computes SeluGrad backprops: `gradients * (outputs + scale * alpha)` 
    if outputs < 0, `scale * gradients`
    support dtype:float16,float32,int32,int8,uint8,bfloat16

    Parameters
    ----------
    input_gradients: dict
        gradients backpropagated to the Selu op.
    input_outputs: dict
        outputs of the Selu op.
    y: dict
        gradients to backpropagate to the Selu inputs.
    kernel_name: str
        cce kernel name, default value is "selu_grad"

    Returns
    ------
    None
    """
    gradients_dtype = input_gradients.get("dtype").lower()
    outputs_dtype = input_outputs.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(gradients_dtype, check_list, param_name="input_gradients")
    para_check.check_dtype(outputs_dtype, check_list, param_name="input_outputs")
    if gradients_dtype != outputs_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "input_gradients", "input_outputs",
                                                              gradients_dtype, outputs_dtype)
    ins = classify([input_gradients, input_outputs], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_gradients, _outputs) in ins:
        with tbe.compute():
            gradients_shape, outputs_shape = shape_util.variable_shape([_gradients, _outputs])
            data_gradients = tvm.placeholder(gradients_shape, name="data_gradients", 
                                             dtype=gradients_dtype)
            data_outputs = tvm.placeholder(outputs_shape, name="data_outputs", dtype=outputs_dtype)
            res = selu_grad_compute(data_gradients, data_outputs, y, kernel_name)
            tensors.append([data_gradients, data_outputs, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
