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
dynamic in_training_update_grad
"""
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tuple_sum
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    # minimum positive number greater than 0
    EPSLON = 1e-6


def op_select_format(dy, x, variance, mean, diff_scale, diff_offset, epsilon, kernel_name="in_training_update_grad"):
    """
    1. when input(dy)'s ori_shape is [1, ? ,1, ?] and the format is NCHW
    the Op INTrainingUpdateGrad can support NCHW.
    > for example:
    > dy : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > x : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > variance : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > mean : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > the Op INTrainingUpdateGrad can process with NC1HWC0:
    > dy : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > variance : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > mean : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    """
    format_x = x.get("ori_format").upper()
    origin_shape = x.get("ori_shape")

    if format_x == "NCHW" and len(origin_shape) == 4 and origin_shape[0] == 1 and origin_shape[2] == 1:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="dy",
                                               datatype="float16,float,float16,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16,float,float16,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="variance",
                                               datatype="float,float,float,float",
                                               format="NCHW, NCHW,NC1HWC0,NC1HWC0")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="mean",
                                               datatype="float,float,float,float",
                                               format="NCHW, NCHW,NC1HWC0,NC1HWC0")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="diff_scale",
                                                datatype="float,float,float,float",
                                                format="NCHW, NCHW,NC1HWC0,NC1HWC0")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="diff_offset",
                                                datatype="float,float,float,float",
                                                format="NCHW, NCHW,NC1HWC0,NC1HWC0")
    else:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="dy",
                                               datatype="float16,float,float16,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16,float,float16,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="variance",
                                               datatype="float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="mean",
                                               datatype="float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="diff_scale",
                                                datatype="float,float,float,float",
                                                format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="diff_offset",
                                                datatype="float,float,float,float",
                                                format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")

    param_list = [input0, input1, input2, input3, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=too-many-statements,too-many-arguments,too-many-locals,invalid-name,unused-argument
@register_operator_compute("INTrainingUpdateGrad", op_mode="dynamic", support_fusion=True)
def in_training_update_grad_compute(dy,
                                    x,
                                    variance,
                                    mean,
                                    res_gamma,
                                    res_beta,
                                    kernel_name="in_training_update_grad",
                                    reduce_axis=None):
    """
    DSL description of the layernorm_grad operator's mathematical

    Parameters
    ----------
    dy: TVM tensor
        the placeholder of input dy
    x: TVM tensor
        the placeholder of input x
    variance: TVM tensor
        the placeholder of input variance
    mean: TVM tensor
        the placeholder of input mean
    res_gamma: dict
        shape and dtype of output res_gamma
    res_beta: dict
        shape and dtype of output res_beta
    kernel_name: str
        cce kernel name, default value is "in_training_update_grad"
    reduce_axis: list
        reduce axis of input shape

    Returns
    -------
    res_list: list
        [res_gamma, res_beta]
    """
    data_format = res_gamma.get("format").upper()
    if not reduce_axis and data_format in ("NC1HWC0", "NCHW"):
        axis = [2, 3]
    elif not reduce_axis and data_format in ("NDC1HWC0",):
        axis = [1, 3, 4]
    else:
        axis = reduce_axis

    if dy.dtype == "float16":
        dy = tbe.cast_to(dy, "float32")

    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")

    shape_dy = shape_util.shape_to_list(dy.shape)
    shape_var = shape_util.shape_to_list(variance.shape)

    mean_inverse = tbe.vmuls(mean, tvm.const(-1, dtype="float32"))
    mean_inverse_broadcast = tbe.broadcast(mean_inverse, shape_dy)
    x_sub = tbe.vadd(x, mean_inverse_broadcast)

    data_adds = tbe.vadds(variance, Constant.EPSLON)
    data_rsqrt = tbe.vsqrt(data_adds)
    data_one = tbe.broadcast(tvm.const(1, "float32"), shape_var)

    data_rsqrts = tbe.vdiv(data_one, data_rsqrt)
    rsqrts_broadcast = tbe.broadcast(data_rsqrts, shape_dy)
    x_norm = tbe.vmul(x_sub, rsqrts_broadcast)

    scale_mul = tbe.vmul(dy, x_norm)

    res_gamma, res_beta = tuple_sum([scale_mul, dy], reduce_axis, True)

    return [res_gamma, res_beta]


# 'pylint: disable=too-many-statements,too-many-arguments,too-many-locals,invalid-name,unused-argument
@register_operator("INTrainingUpdateGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def in_training_update_grad(dy, x, variance, mean, res_gamma, res_beta, kernel_name="in_training_update_grad"):
    """
    in_training_update_grad operator interface implementation

    Parameters
    ----------
    dy: dict
        shape and dtype of input dy, only support float16, float32
    x: dict
        shape and dtype of input x, only support float16, float32
    variance: dict
        shape and dtype of input variance, only support float32
    mean: dict
        shape and dtype of input mean, only support float32
    res_gamma: dict
        shape and dtype of output res_gamma, only support float32
    res_beta: dict
        shape and dtype of output res_beta, only support float32
    kernel_name: str
        cce kernel name, default value is "in_training_update_grad"

    Returns
    -------
    None
    """
    input_list = [dy, x, variance, mean]
    dtype_dy = dy.get("dtype").lower()
    dtype_x = x.get("dtype").lower()
    dtype_variance = variance.get("dtype").lower()
    dtype_mean = mean.get("dtype").lower()

    para_check.check_dtype(dtype_dy, ("float32", "float16"), param_name="dy")
    para_check.check_dtype(dtype_x, ("float32", "float16"), param_name="x")
    para_check.check_dtype(dtype_variance, ("float32",), param_name="variance")
    para_check.check_dtype(dtype_mean, ("float32",), param_name="mean")

    data_format = dy.get("format")
    if data_format in ("NC1HWC0", "NCHW"):
        list_axis = [2, 3]
    else:
        list_axis = [1, 3, 4]

    if util_common.is_unknown_rank_input(input_list):
        if data_format == "NC1HWC0":
            unknown_rank_shape = [-1, -1, -1, -1, -1]
            unknown_rank_range = [(1, None), (1, None), (1, None), (1, None), (1, None)]
        else:
            unknown_rank_shape = [-1, -1, -1, -1, -1, -1]
            unknown_rank_range = [(1, None), (1, None), (1, None), (1, None), (1, None), (1, None)]
        for input_dict in input_list:
            input_dict["shape"] = unknown_rank_shape
            input_dict["range"] = unknown_rank_range

    extra_params = {"compile_broadcast_axis": {2: list_axis, 3: list_axis}}
    ins = classify([dy, x, variance, mean, list_axis], OpPatternMode.TUPLE_REDUCE, extra_params=extra_params)
    schedules, tensors = [], []
    for (_dy, _x, _variance, _mean, _reduce_axis) in ins:
        with tbe.compute():
            shape_dy, shape_x, shape_mean, shape_variance = shape_util.variable_shape(
                [_dy, _x, _variance, _mean], op_mode=OpPatternMode.TUPLE_REDUCE)
            input_dy = tvm.placeholder(shape_dy, name="input_dy", dtype=dtype_dy)
            input_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
            input_mean = tvm.placeholder(shape_mean, name="input_mean", dtype=dtype_variance)
            input_variance = tvm.placeholder(shape_variance, name="input_variance", dtype=dtype_mean)
            res = in_training_update_grad_compute(input_dy, input_x, input_mean, input_variance, res_gamma, res_beta,
                                                  kernel_name, _reduce_axis)

            tensor_list = [input_dy, input_x, input_mean, input_variance] + list(res)
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
