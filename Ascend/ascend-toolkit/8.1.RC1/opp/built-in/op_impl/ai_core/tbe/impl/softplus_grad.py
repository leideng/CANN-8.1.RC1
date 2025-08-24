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
softplus_grad
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from tbe import tvm
from te.utils.error_manager import error_manager_vector


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
# 'pylint: disable=unused-variable
@tbe_platform.fusion_manager.fusion_manager.register("softplus_grad")
def softplus_grad_compute(input_gradients, input_features, output_backprops,
                          kernel_name="softplus_grad"):
    """
    Computes softplus gradients for a softplus operation.
    The gradients: "dy * exp(x) / (1 + exp(x))".

    Parameters
    ----------
    input_gradients: TVM tensor
        The backpropagated gradients to the corresponding softplus operation.
    input_features: TVM tensor
        The input_features passed as input to the corresponding softplus operation.
        source data type support "float16", "float32", "int32", "int8", "uint8".
    output_backprops: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softplus_grad".

    Returns
    -------
    res: TVM tensor
        output tensor. has the same type as "input_gradients".
    """
    shape_dy = shape_util.shape_to_list(input_gradients.shape)
    shape_x = shape_util.shape_to_list(input_features.shape)
    dtype = input_gradients.dtype

    if list(shape_dy) != list(shape_x):
        shape_dy, shape_x, shape_max = \
            shape_util.broadcast_shapes(shape_dy, shape_x,
                                        param_name_input1="input_gradients",
                                        param_name_input2="input_features")
        input_gradients = tbe.broadcast(input_gradients, shape_max, dtype)
        input_features = tbe.broadcast(input_features, shape_max, dtype)
    else:
        shape_max = shape_dy

    if dtype != "float32":
        input_gradients = tbe.cast_to(input_gradients, "float32")
        input_features = tbe.cast_to(input_features, "float32")

    data_exp_tmp = tbe.vexp(input_features)
    data_add_tmp = tbe.vadds(data_exp_tmp, 1)
    data_div_tmp = tbe.vdiv(data_exp_tmp, data_add_tmp)
    res_tmp = tbe.vmul(input_gradients, data_div_tmp)

    if dtype == "float16":
        res = tbe.cast_to(res_tmp, "float16")
    elif dtype in ("int32", "int8", "uint8"):
        data_zero = tbe.broadcast(tvm.const(0, "float16"), shape_max, "float16")
        res_min = tbe.vmin(res_tmp, data_zero)
        res_max = tbe.vmax(res_tmp, data_zero)
        res_max_int = tbe.floor(res_max)
        res_min_int = tbe.ceil(res_min)
        res = tbe.vadd(res_max_int, res_min_int)
    else:
        res = res_tmp

    if dtype == "int8":
        res = tbe.cast_to(res, "int8")
    elif dtype == "uint8":
        res = tbe.cast_to(res, "uint8")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def softplus_grad(input_gradients, input_features, output_backprops,
                  kernel_name="softplus_grad"):
    """
    Computes softplus gradients for a softplus operation.
    The gradients: "dy * exp(x) / (1 + exp(x))".

    Parameters
    ----------
    input_gradients: dict
        The backpropagated gradients to the corresponding softplus operation.
    input_features: dict
        The input_features passed as input to the corresponding softplus operation.
        source data type support "float16", "float32", "int32", "int8", "uint8".
    output_backprops: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softplus_grad".

    Returns
    -------
    None
    """
    shape_dy = input_gradients.get("shape")
    dtype_dy = input_gradients.get("dtype")
    shape_x = input_features.get("shape")
    dtype_x = input_features.get("dtype")

    if dtype_dy.lower() != dtype_x.lower():
        error_detail = "Dtype of tensor input_gradients and input_features must be same!"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_gradients", \
                                                               "input_features", error_detail)
    dtype = dtype_dy

    para_check.check_shape(shape_dy, param_name="input_gradients")
    para_check.check_shape(shape_x, param_name="input_features")

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_gradients")
    shape_dy, shape_x, shape_max = \
        shape_util.broadcast_shapes(shape_dy, shape_x,
                                    param_name_input1="input_gradients",
                                    param_name_input2="input_features")
    reshape_dy, reshape_x = shape_util.refine_shapes_for_broadcast(shape_dy, shape_x)

    data_dy = tvm.placeholder(reshape_dy, name="data_dy", dtype=input_dtype)
    data_x = tvm.placeholder(reshape_x, name="data_x", dtype=input_dtype)

    res = softplus_grad_compute(data_dy, data_x, output_backprops,
                                kernel_name=kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_dy, data_x, res]}
    tbe.cce_build_code(sch, config)
