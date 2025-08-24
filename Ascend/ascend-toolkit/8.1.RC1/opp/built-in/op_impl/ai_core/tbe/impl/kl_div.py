# Copyright 2020 Huawei Technologies Co., Ltd
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
kl_div
"""
import functools

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
import te.lang.cce as tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_select_op_base


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
def op_select_format(input_x, input_target, output_y, reduction, kernel_name="kl_div"):
    """
    select format dynamically 
    op_select_format support desc:

    1.When reduction is "none". 

    The output format is the same as the input.

        for example:
        inputs:
            input_x         shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
            input_target      shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
        outputs:
            output_y         shape = [16, 16, 16, 16, 16] format = "NC1HWC0"

    2.In other scenes, all output_y only support ND.

        for example:
        inputs:
            input_x         shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
            input_target      shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
        outputs:
            output_y         shape = [1,] format = "ND"

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x, dtype only support fp16 and fp32.
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    output_y : dict
        shape and dtype of output.Dtype must be same as input_x
    reduction: str
        reduction="batchmean" or "sum" or "none".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
        "none": no reduction will be applied
    kernel_name : str
        cce kernel name, default value is "kl_div"

    Returns
    ------
    None
    """
    input_format = "ND,NC1HWC0,FRACTAL_Z,HWCN,FRACTAL_NZ,C1HWNCoC0,ND,NC1HWC0,FRACTAL_Z,HWCN,FRACTAL_NZ,C1HWNCoC0"
    if reduction == "none":
        output_format = input_format
    else:
        output_format = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"

    dtype = "float,float,float,float,float,float,float16,float16,float16,float16,float16,float16"
    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                           datatype=dtype,
                                           format=input_format,
                                           unknownshape_format=input_format)
    input1 = util_select_op_base.gen_param(classify="input1", name="target",
                                           datatype=dtype,
                                           format=input_format,
                                           unknownshape_format=input_format)
    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                            datatype=dtype,
                                            format=output_format,
                                            unknownshape_format=output_format)
    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
@register_operator_compute("kl_div", op_mode="static", support_fusion=True)
def kl_div_compute(input_x, input_target, output_y, reduction, batch_size, kernel_name="kl_div"):
    """
    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input_target : TVM tensor
        the placeholder of input_target
    output_y : dict
        shape and dtype of output, should be same shape and type as input_x
    reduction: str
        Specifies the reduction to apply to the output:
        reduction="batchmean" or "sum" or "none".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
        "none": no reduction will be applied
    batch_size: int
        Equal to the first dimension value of the input shape.
    kernel_name : str
        cce kernel name, default value is "kl_div"

    Returns
    ------
    compute result of kl_div
    """
    input_dtype = input_x.dtype
    log_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vlog", "float32")
    if log_support_fp32 and input_dtype == "float32":
        # for float32, take the maximum value between input_target and 1.18e-38, so that
        # the argument of natural logarithm will be a positive number.
        log_target_max = tbe.vmaxs(input_target, tvm.const(1.18e-38, dtype=input_dtype))
        log_target = tbe.vlog(log_target_max)
    else:
        log_target_max = tbe.vmaxs(input_target, tvm.const(1.18e-7, dtype=input_dtype))
        log_target = tbe.vlog(log_target_max)

    tmp_result = tbe.vsub(log_target, input_x)
    output_pos = tbe.vmul(input_target, tmp_result)

    target_gt_zero = tbe.vmaxs(input_target, 0)

    if input_dtype == "float16":
        # algorithm : `Y = X*1024/(X*1024+ESP_MIN)`
        # for float16, add a small number which value is 1.18e-7, so that the
        # divisor is not equal to 0, and for accuracy, multiply by a number
        # which value is 1024.
        mul_big = tbe.vmuls(target_gt_zero, 1024)
        add_espmin = tbe.vadds(mul_big, 1.18e-7)
        y_espmin = tbe.vdiv(mul_big, add_espmin)
    if input_dtype == "float32":
        # algorithm : `Y = X/(X+ESP_MIN)`
        # for float32, add a small number which value is 1.18e-38, so that
        # the divisor is not equal to 0.
        add_espmin = tbe.vadds(target_gt_zero, 1.18e-38)
        y_espmin = tbe.vdiv(target_gt_zero, add_espmin)

    output_res = tbe.vmul(y_espmin, output_pos)

    if reduction == "batchmean":
        output_res = tbe.vmuls(output_res, 1.0 / batch_size)
        final_res = tbe.sum(output_res, axis=0)
    elif reduction == "sum":
        final_res = tbe.sum(output_res, axis=0)
    elif reduction == "none":
        final_res = output_res
    else:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, 'reduction',
                                                           ("batchmean", "sum", "none"), reduction)

    return final_res


def _check_parameter(input_x, input_target):
    """
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    Returns
    ------
    None
    """
    shape_x = input_x.get("shape")
    shape_target = input_target.get("shape")
    para_check.check_shape(shape_x, param_name="input_x")
    if list(shape_x) != list(shape_target):
        error_manager_vector.raise_err_inputs_shape_not_equal('kl_div', 'input_x', 'input_target', shape_x,
                                                              shape_target, shape_x)

    # check input tensor data_type
    dtype_x = input_x.get("dtype").lower()
    dtype_target = input_target.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    if dtype_x != dtype_target:
        error_manager_vector.raise_err_inputs_dtype_not_equal('kl_div', 'input_x', 'input_target', dtype_x,
                                                              dtype_target)

    if dtype_x == "float32" and not tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        error_manager_vector.raise_err_input_dtype_not_supported('kl_div', 'input_x', ('float16', ), dtype_x)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def kl_div(input_x, input_target, output_y, reduction, kernel_name="kl_div"):
    """
    Calcuate Kullback-Leibler divergence.

    output_pos = input_target * (log(input_target) - input_x)
    output = where(input_target > 0, output_pos, zeros)
    reduced = reduce_sum_all(output)
    if reduction = "batchmean":
        final_res = reduce / input.dim[0]
    elif reduction = "sum":
        final_res = reduced
    else:
        final_res = output
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x, dtype only support fp16 and fp32.
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    output_y : dict
        shape and dtype of output.Dtype must be same as input_x
    reduction: str
        reduction="batchmean" or "sum" or "none".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
        "none": no reduction will be applied
    kernel_name : str
        cce kernel name, default value is "kl_div"

    Returns
    ------
    None
    """
    # check input parameter
    _check_parameter(input_x, input_target)

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    batch_size = shape_x[0]
    shape_one_dim = [functools.reduce(lambda x, y: x * y, shape_x[:])]
    data_x = tvm.placeholder(shape_one_dim, name="data_x", dtype=dtype_x)
    data_target = tvm.placeholder(shape_one_dim, name="data_target", dtype=dtype_x)

    final_res = kl_div_compute(data_x, data_target, output_y, reduction, batch_size, kernel_name=kernel_name)
    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(final_res)

    config = {"name": kernel_name, "tensor_list": (data_x, data_target, final_res)}

    tbe.cce_build_code(auto_sch, config)
