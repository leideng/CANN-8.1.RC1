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
dynamic in_training_update_v2
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_compute import only_static_support
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_soc_common import after_v200


class AttrInfo:
    """
    define attr info
    """
    ATTR_MOMENTUM = OpAttr(0, "momentum", "Float", 0.1)
    ATTR_EPSILON = OpAttr(1, "epsilon", "Float", 0.00001)


# 'pylint: disable=redefined-builtin
def op_select_format(x,
                     sum,
                     square_sum,
                     gamma,
                     beta,
                     mean,
                     variance,
                     y,
                     batch_mean,
                     batch_variance,
                     momentum,
                     epsilon,
                     kernel_name="in_training_update_v2"):
    """
    select format dynamically
    """
    input_format = "NC1HWC0, NC1HWC0"
    ori_format = x.get("ori_format")
    if ori_format in ("NDHWC", "NCDHW"):
        input_format = "NDC1HWC0, NDC1HWC0"

    input0 = gen_param(classify="input0", name="x", datatype="float16,float", format=input_format)
    input1 = gen_param(classify="input1", name="sum", datatype="float,float", format=input_format)
    input2 = gen_param(classify="input2", name="square_sum", datatype="float,float", format=input_format)
    input3 = gen_param(classify="input3", name="gamma", datatype="float,float", format=input_format)
    input4 = gen_param(classify="input4", name="beta", datatype="float,float", format=input_format)
    input5 = gen_param(classify="input5", name="mean", datatype="float,float", format=input_format)
    input6 = gen_param(classify="input6", name="variance", datatype="float,float", format=input_format)
    output0 = gen_param(classify="output0", name="y", datatype="float16,float", format=input_format)
    output1 = gen_param(classify="output1", name="batch_mean", datatype="float,float", format=input_format)
    output2 = gen_param(classify="output2", name="batch_variance", datatype="float,float", format=input_format)

    param_list = [input0, input1, input2, input3, input4, input5, input6, output0, output1, output2]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_gamma, dtype_beta):
    """check input dtype"""
    para_check.check_dtype(dtype_x, ("float16", "float32"))
    para_check.check_dtype(dtype_sum, ("float32",))
    para_check.check_dtype(dtype_square_sum, ("float32",))
    para_check.check_dtype(dtype_gamma, ("float32",))
    para_check.check_dtype(dtype_beta, ("float32",))


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
@register_operator_compute("INTrainingUpdate", op_mode="dynamic", support_fusion=only_static_support)
def in_training_update_v2_compute(x,
                                  sum,
                                  square_sum,
                                  gamma,
                                  beta,
                                  mean,
                                  variance,
                                  y,
                                  mean_out,
                                  variance_out,
                                  momentum,
                                  epsilon,
                                  kernel_name="in_training_update_v2"):
    """
    DSL description of the instancenorm operator's mathematical calculation process

    x: dict
        the placeholder of input x
    sum: dict
        the placeholder of input sum
    square_sum: dict
        the placeholder of input square_sum
    gamma: dict
        the placeholder of input gamma
    beta: dict
        the placeholder of input beta
    mean: dict
        the placeholder of input mean
    variance: dict
        the placeholder of input variance
    y: dict
        shape and dtype of output y
    batch_mean: dict
        shape and dtype of output batch_mean
    batch_variance: dict
        shape and dtype of output batch_variance
    momentum: float
        A ratio to calculate the update mean or variance
    epsilon: float
        A small float number added to the variance of x
    kernel_name: str
        cce kernel name, default value is "in_training_update_v2"

    Returns
    -------
    res_list: list
        [result, result_mean, result_variance]
    """
    dtype_x = x.dtype.lower()

    if not util_common.is_unknown([y]):
        data_format = y.get("format").upper()
        shape_y = y.get("shape")
        if data_format in ("NC1HWC0",):
            reduce_dims = [shape_y[2], shape_y[3]]
        elif data_format in ("NDC1HWC0",):
            reduce_dims = [shape_y[1], shape_y[3], shape_y[4]]
        else:
            reduce_dims = None

        num = 1
        if reduce_dims:
            for dim in reduce_dims:
                num *= dim

        num_bw = 1.0 / num
        num_rec = tvm.const(num_bw, dtype="float32")

        if num == 1:
            batch_var_scalar = 0.0
        else:
            batch_var_scalar = float(num) / (num - 1)
    else:
        num_rec = tbe.var("num_rec", dtype="float32")
        batch_var_scalar = tbe.var("batch_var_scaler", dtype="float32")

    if dtype_x == "float16":
        x = tbe.cast_to(x, "float32")

    momentum = get_attr_by_cls(momentum, AttrInfo.ATTR_MOMENTUM, "float32")
    epsilon = get_attr_by_cls(epsilon, AttrInfo.ATTR_EPSILON, "float32")

    # compute the saved mean of x
    save_mean_reduce = tbe.vmuls(sum, num_rec)

    # compute the saved variance of x
    variance_div = tbe.vmuls(square_sum, num_rec)
    variance_square = tbe.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = tbe.vsub(variance_div, variance_square)
    if after_v200():
        save_variance_reduce = tbe.vmaxs(save_variance_reduce, tvm.const(0.0, save_variance_reduce.dtype))

    # compute the coefficient of y
    if gamma is not None and beta is not None:
        multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
        multiplier_sqrt = tbe.vsqrt(multiplier_add)
        gamma = tbe.broadcast(gamma, sum.shape)
        multiplier_div = tbe.vdiv(gamma, multiplier_sqrt)
        multiplier = tbe.broadcast(multiplier_div, x.shape)

        addend_mul = tbe.vmul(multiplier_div, save_mean_reduce)
        beta = tbe.broadcast(beta, sum.shape)
        addend_sub = tbe.vsub(beta, addend_mul)
        addend = tbe.broadcast(addend_sub, x.shape)

        x_mul = tbe.vmul(multiplier, x)
        res_y = tbe.vadd(x_mul, addend)
    else:
        mean_broadcast = tbe.broadcast(save_mean_reduce, x.shape)
        x_mean = tbe.vsub(x, mean_broadcast)
        multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
        multiplier_sqrt = tbe.vsqrt(multiplier_add)
        sqrt_broadcast = tbe.broadcast(multiplier_sqrt, x.shape)
        res_y = tbe.vdiv(x_mean, sqrt_broadcast)

    if dtype_x == "float16":
        res_y = tbe.cast_to(res_y, dtype_x)

    result_mean = save_mean_reduce
    result_variance = tbe.vmuls(save_variance_reduce, batch_var_scalar)

    # if input mean and var, use input values and momentum to update
    if mean is not None and variance is not None:
        factor_reverse = 1.0 - momentum
        mean_mul = tbe.vmuls(save_mean_reduce, momentum)
        mean_mul_rev = tbe.vmuls(mean, factor_reverse)
        result_mean = tbe.vadd(mean_mul, mean_mul_rev)

        var_mul = tbe.vmuls(result_variance, momentum)
        var_mul_rev = tbe.vmuls(variance, factor_reverse)
        result_variance = tbe.vadd(var_mul, var_mul_rev)

    return [res_y, result_mean, result_variance]


@register_operator("INTrainingUpdateV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def in_training_update_v2(x,
                          sum,
                          square_sum,
                          gamma,
                          beta,
                          mean,
                          variance,
                          y,
                          batch_mean,
                          batch_variance,
                          momentum=0.1,
                          epsilon=0.00001,
                          kernel_name="in_training_update_v2"):
    """
    instancenorm operator interface implementation

    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16, float32
    sum: dict
        shape and dtype of input sum, only support float32
    square_sum: dict
        shape and dtype of input square_sum, only support float32
    gamma: dict
        shape and dtype of input gamma, only support float32
    beta: dict
        shape and dtype of input beta, only support float32
    mean: dict
        shape and dtype of input mean, only support float32
    variance: dict
        shape and dtype of input variance, only support float32
    y: dict
        shape and dtype of output y, only support float16, float32
    batch_mean: dict
        shape and dtype of output batch_mean, only support float32
    batch_variance: dict
        shape and dtype of output batch_variance, only support float32
    momentum: float
        A ratio to calculate the update mean or variance
    epsilon: float
        A small float number added to the variance of x
    kernel_name: str
        cce kernel name, default value is "in_training_update_v2"

    Returns
    -------
    None
    """
    input_list = [x, sum, square_sum, gamma, beta]
    dtype_x = x.get("dtype").lower()
    dtype_sum = sum.get("dtype").lower()
    dtype_sqrsum = square_sum.get("dtype").lower()
    dtype_gamma = gamma.get("dtype").lower()
    dtype_beta = beta.get("dtype").lower()
    data_format = x.get("format")
    _check_dtype(dtype_x, dtype_sum, dtype_sqrsum, dtype_gamma, dtype_beta)

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
        if mean and variance:
            mean["shape"] = unknown_rank_shape
            mean["range"] = unknown_rank_range
            variance["shape"] = unknown_rank_shape
            variance["range"] = unknown_rank_range

    schedules = []
    tensors = []

    if not mean and not variance:
        ins_list = [x, sum, square_sum, gamma, beta]
        ins = classify(ins_list, OpPatternMode.ELEWISE_WITH_BROADCAST)
        for (ins_x, ins_sum, ins_square_sum, ins_gamma, ins_beta) in ins:
            with tbe.compute():
                shape_x, shape_sum, shape_sqrsum, shape_gamma, shape_beta = shape_util.variable_shape(
                    [ins_x, ins_sum, ins_square_sum, ins_gamma, ins_beta])

                in_x = tvm.placeholder(shape_x, name="x", dtype=dtype_x)
                in_sum = tvm.placeholder(shape_sum, name="sum", dtype=dtype_sum)
                in_sqrsum = tvm.placeholder(shape_sqrsum, name="sqrsum", dtype=dtype_sum)
                in_gamma = tvm.placeholder(shape_gamma, name="gamma", dtype=dtype_sum)
                in_beta = tvm.placeholder(shape_beta, name="beta", dtype=dtype_sum)
                res = in_training_update_v2_compute(in_x,
                                                    in_sum,
                                                    in_sqrsum,
                                                    in_gamma,
                                                    in_beta,
                                                    None,
                                                    None,
                                                    y,
                                                    batch_mean,
                                                    batch_variance,
                                                    momentum,
                                                    epsilon,
                                                    kernel_name=kernel_name)
                tensors.append([in_x, in_sum, in_sqrsum, in_gamma, in_beta] + res)

                with tvm.target.cce():
                    sch = tbe.auto_schedule(res)
                schedules.append(sch)
    else:
        ins_list = [x, sum, square_sum, gamma, beta, mean, variance]
        ins = classify(ins_list, OpPatternMode.ELEWISE_WITH_BROADCAST)
        for (ins_x, ins_sum, ins_square_sum, ins_gamma, ins_beta, ins_mean, ins_variance) in ins:
            with tbe.compute():
                shape_x, shape_sum, shape_sqrsum, shape_gamma, shape_beta, shape_mean, shape_variance = \
                    shape_util.variable_shape([ins_x, ins_sum, ins_square_sum, ins_gamma, ins_beta, ins_mean,
                                               ins_variance])

                in_x = tvm.placeholder(shape_x, name="x", dtype=dtype_x)
                in_sum = tvm.placeholder(shape_sum, name="sum", dtype=dtype_sum)
                in_sqrsum = tvm.placeholder(shape_sqrsum, name="sqrsum", dtype=dtype_sum)
                in_gamma = tvm.placeholder(shape_gamma, name="gamma", dtype=dtype_sum)
                in_beta = tvm.placeholder(shape_beta, name="beta", dtype=dtype_sum)
                in_mean = tvm.placeholder(shape_mean, name="mean", dtype=dtype_sum)
                in_variance = tvm.placeholder(shape_variance, name="variance", dtype=dtype_sum)
                res = in_training_update_v2_compute(in_x,
                                                    in_sum,
                                                    in_sqrsum,
                                                    in_gamma,
                                                    in_beta,
                                                    in_mean,
                                                    in_variance,
                                                    y,
                                                    batch_mean,
                                                    batch_variance,
                                                    momentum,
                                                    epsilon,
                                                    kernel_name=kernel_name)
                tensors.append([in_x, in_sum, in_sqrsum, in_gamma, in_beta, in_mean, in_variance] + res)

                with tvm.target.cce():
                    sch = tbe.auto_schedule(res)
                schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
