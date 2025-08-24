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
Dynamic Activation Universal Linear Quant Clamp Min Gradient
"""
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.reduce_pattern_adapter import ReducePattern


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    SHAPE_SIZE_LIMIT = 2 ** 31


def check_input_shape(input_x, input_y, input_z, kernel_name):
    """
    check inputs shape and range
    """
    input_x_shape = input_x.get('shape')
    input_x_range = input_x.get('range')
    input_x_size = 1
    for index, input_x_shape_value in enumerate(input_x_shape):
        if input_x_shape_value == -1:
            if input_x_range[index][1] is None:
                input_x_size *= Constant.SHAPE_SIZE_LIMIT
            else:
                input_x_size *= input_x_range[index][1]
        else:
            input_x_size *= input_x_shape_value
    if input_x_size > Constant.SHAPE_SIZE_LIMIT:
        error_detail = "The shape size of y_grad must be smaller than {}!".format(Constant.SHAPE_SIZE_LIMIT)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, 'y_grad', error_detail)

    input_y_shape = input_y.get('shape')
    input_z_shape = input_z.get('shape')

    if input_x_shape != input_y_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'y_grad', 'clamp_min_mask', input_x_shape, input_y_shape, input_x_shape)

    if input_x_shape != input_z_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'y_grad', 'x_clamped_loss', input_x_shape, input_z_shape, input_x_shape)


# 'pylint: disable=unused-argument
@register_operator_compute('ActULQClampMinGrad', op_mode='dynamic', support_fusion=False)
def act_ulq_clamp_min_grad_compute(
    y_grad, clamp_min_mask, x_clamped_loss, axis, kernel_name='act_ulq_clamp_min_grad'):
    """
    Function: Calculate the gradient of minimum clamped value.

    Parameters:
    ----------
    y_grad: the placeholder of gradient

    clamp_min_mask : the placeholder of clamp_min_mask

    x_clamped_loss : the placeholder of x_clamped_loss

    output: the dict of output

    kernel_name: cce kernel name, default value is "act_ulq_clamp_min_grad"

    Returns : A Tensor with float32 and (1,).
    -------
    """
    shape = y_grad.shape
    dtype = y_grad.dtype
    min_dtype = clamp_min_mask.dtype
    if min_dtype == "int8":
        clamp_min_mask = tbe.cast_to(clamp_min_mask, dtype)

    signal = tbe.broadcast(tvm.const(1, dtype), shape)
    signal = tbe.vsub(clamp_min_mask, signal)
    signal = tbe.vabs(clamp_min_mask)

    x_min_grad = tbe.vadd(x_clamped_loss, signal)

    clamp_min_grad = tbe.vmul(y_grad, x_min_grad)
    clamp_min_grad = tbe.reduce_sum(clamp_min_grad, axis)

    return clamp_min_grad


# 'pylint: disable=too-many-locals,too-many-branches,too-many-statements
@register_operator('ActULQClampMinGrad')
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
    para_check.KERNEL_NAME)
def act_ulq_clamp_min_grad(input_x, input_y, input_z, output, kernel_name='act_ulq_clamp_min_grad'):
    """
    ----------
    Parameters:
    ----------
    input_x : the placeholder of y_grad

    input_y : the placeholder of clamp_min_mask

    input_z : the placeholder of x_clamped_loss

    output : the dict of clamp_min_grad

    Returns : None
    ----------
    """
    if not is_unknown_rank_input([input_x, input_y, input_z]):
        check_input_shape(input_x, input_y, input_z, kernel_name)

    check_list = ['float16', 'float32']
    input_x_type = input_x.get('dtype').lower()
    input_y_type = input_y.get('dtype').lower()
    input_z_type = input_z.get('dtype').lower()

    para_check.check_dtype_rule(input_x_type, check_list, 'y_grad')
    if input_y_type == "bool" or input_y_type == "int8":
        input_y_type = "int8"
    elif input_x_type != input_y_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, 'y_grad', 'clamp_min_mask', input_x_type, input_y_type)

    if input_x_type != input_z_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, 'y_grad', 'x_clamped_loss', input_x_type, input_z_type)

    input_x["rel_pos_to_reduce"] = "before"
    input_y["rel_pos_to_reduce"] = "before"
    input_z["rel_pos_to_reduce"] = "before"

    # gen reduce axis input dict
    input_axis = {"shape": [-1], "value": [], "rel_pos_to_reduce": "axis"}
    # gen extra_params for reduce pattern
    extra_params = dict()
    # set KEEP_DIMS flag
    extra_params.update(ReducePattern.KEEP_DIMS_FALSE)
    # set all reduce pattern
    extra_params.update(ReducePattern.REDUCE_MODE_REDUCE_ALL)

    ins = classify([input_x, input_y, input_z, input_axis], OpPatternMode.REDUCE, extra_params)
    schedules, tensors = [], []
    for (y_grad, clamp_min_mask, x_clamped_loss, axis) in ins:
        with tbe.compute():
            shape_y_grad, shape_clamp_min_mask, shape_x_clamped_loss = shape_util.variable_shape(
                [y_grad, clamp_min_mask, x_clamped_loss, axis], op_mode='reduce')[0: 3]

            data_y_grad = tvm.placeholder(shape_y_grad, input_x_type, 'data_y_grad')
            data_clamp_min_mask = tvm.placeholder(shape_clamp_min_mask, input_y_type, 'data_clamp_min_mask')
            data_x_clamped_loss = tvm.placeholder(shape_x_clamped_loss, input_x_type, 'data_x_clamped_loss')

            res = act_ulq_clamp_min_grad_compute(
                data_y_grad, data_clamp_min_mask, data_x_clamped_loss, axis.get('value'), kernel_name)
            tensors.append([data_y_grad, data_clamp_min_mask, data_x_clamped_loss, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        'name': kernel_name,
        'print_ir': False,
        'tensor_list': tensors,
        'bool_storage_as_1bit': False}

    tbe.build(schedules, config)
