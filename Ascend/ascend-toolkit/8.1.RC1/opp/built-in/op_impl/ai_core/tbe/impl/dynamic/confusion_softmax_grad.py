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
confusion_softmax_grad
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from tbe.common.utils import shape_util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


def op_select_format(grad, x, y, kernel_name="confusion_softmax_grad"):
    """
    select format depend on the shape.
    """
    is_dynamic_shape = util_common.is_dynamic_input([grad, x])
    op_dtype_dyn = "bfloat16,bfloat16,bfloat16,float16,float16,float16,float,float,float,bfloat16,float16,float"
    op_dtype_stc = "bfloat16,bfloat16,bfloat16,float16,float16,float16,float,float,float,\
        bfloat16,bfloat16,float16,float16,float,float"
    op_format_dyn = "FRACTAL_NZ,NC1HWC0,ND,FRACTAL_NZ,NC1HWC0,ND,FRACTAL_NZ,NC1HWC0,ND,NDC1HWC0,NDC1HWC0,NDC1HWC0"
    op_format_stc = "FRACTAL_NZ,NC1HWC0,ND,FRACTAL_NZ,NC1HWC0,ND,FRACTAL_NZ,NC1HWC0,ND,\
        FRACTAL_Z_3D,NDC1HWC0,FRACTAL_Z_3D,NDC1HWC0,FRACTAL_Z_3D,NDC1HWC0"
    op_dtype = op_dtype_dyn if is_dynamic_shape else op_dtype_stc
    op_format = op_format_dyn if is_dynamic_shape else op_format_stc

    input0 = gen_param(classify="input0", name="grad", datatype=op_dtype, format=op_format,
                       unknownshape_format=op_format)
    input1 = gen_param(classify="input1", name="x", datatype=op_dtype, format=op_format,
                       unknownshape_format=op_format)
    output0 = gen_param(classify="output0", name="y", datatype=op_dtype, format=op_format,
                        unknownshape_format=op_format)
    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json
    

# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = shape_util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = tbe.broadcast(tensor, temp_shape)
    tensor = tbe.broadcast(tensor, shape)
    return tensor


@register_operator_compute("ConfusionSoftmaxGrad", op_mode="dynamic", support_fusion=False, support_bfp16=True)
def confusion_softmax_grad_compute(grad, x, reduce_axis, y,
                                   kernel_name="confusion_softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    y = grad - sum(grad * x)

    Parameters
    ----------
    grad: TVM tensor
        the placeholder of first input data
    x: TVM tensor
        the placeholder of second input data
    y: dict
        the dict of output data
    kernel_name: str
        cce kernel name, default value is "confusion_softmax_grad"

    Returns
    -------
    res: TVM tensor
        the result of confusion_softmax_grad_compute
    """
    dtype = grad.dtype
    shape_input = shape_util.shape_to_list(x.shape)

    data_vmul = tbe.vmul(grad, x)
    if dtype == "float16":
        data_vmul = tbe.cast_to(data_vmul, "float32")

    data_sum = tbe.reduce_sum(data_vmul, axis=reduce_axis, keepdims=True)

    if dtype == "float16":
        data_sum = tbe.cast_to(data_sum, "float16")

    data_sum_tmp = _broadcast_nz(data_sum, shape_input)

    res = tbe.vsub(grad, data_sum_tmp)

    return res


@register_operator("ConfusionSoftmaxGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def confusion_softmax_grad(grad, x, y, kernel_name="confusion_softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    y = grad - sum(grad * x)

    Parameters
    ----------
    grad: dict
        shape and dtype of first input, only support bfloat16, float16, float32
    x: dict
        shape and dtype of second input, only support bfloat16, float16, float32
    y: dict
        shape and dtype of output data, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "confusion_softmax_grad"

    Returns
    -------
    None
    """
    grad = util_common.update_shape_base_other_format_dynamic(grad)
    x = util_common.update_shape_base_other_format_dynamic(x)
    shape_grad = grad.get("shape")
    shape_x = x.get("shape")
    dtype_grad = grad.get("dtype")
    ori_shape_grad = grad.get("ori_shape")
    format_grad = grad.get("format")
    ori_format_grad = grad.get("ori_format")

    shape_util.compare_tensor_dict_key(grad, x, "dtype")
    para_check.check_shape(shape_grad, param_name="grad")
    para_check.check_shape(shape_x, param_name="x")

    check_list = ("bfloat16", "float16", "float32")
    input_dtype = dtype_grad.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="grad")

    extra_params = {"input_shape_type": [0, 0]}
    reduce_axis = [len(ori_shape_grad) - 1, ]
    reduce_axis = util_common.update_axis_for_other_format(ori_shape_grad, reduce_axis[0],
                                                            format_grad, ori_format_grad, True)
    if not isinstance(reduce_axis, list):
        reduce_axis = [reduce_axis]
    ins = classify([grad, x, reduce_axis], OpPatternMode.NORM, extra_params)

    tensors, schedules = [], []
    for (x1, x2, reduce_axis) in ins:
        with tbe.compute():
            shape_grad, shape_x = shape_util.variable_shape([x1, x2], op_mode="norm")
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=input_dtype)
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_dtype)
            res = confusion_softmax_grad_compute(data_grad, data_x, reduce_axis, y,
                                                 kernel_name=kernel_name)
            tensors.append([data_grad, data_x] + [res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
