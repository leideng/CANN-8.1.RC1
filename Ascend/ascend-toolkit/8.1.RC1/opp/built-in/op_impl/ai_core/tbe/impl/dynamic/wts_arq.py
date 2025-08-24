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
dynamic wts_arq
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


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    SHAPE_SIZE_LIMIT = 2 ** 31
    EPS = 0.001


def check_input_shape(w, w_min, w_max, kernel_name):
    """
    check inputs shape and range
    """
    w_shape = w.get('shape')
    w_min_shape = w_min.get('shape')
    w_max_shape = w_max.get('shape')
    w_range = w.get('range')
    w_min_range = w_min.get('range')
    w_max_range = w_max.get('range')

    if w_min_shape != w_max_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'w_min', 'w_max', w_min_shape, w_max_shape, w_min_shape)

    if len(w_shape) != len(w_min_shape):
        error_detail = "The shape size of w_min&w_max must be same as w!"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, 'w_min', 'w_max', error_detail)

    for i, j in zip(w_shape, w_min_shape):
        if j not in (i, 1):
            error_detail = "The shape value of w_min&w_max must be same as w or equal to 1!"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, 'w_min', 'w_max', error_detail)

    w_size = 1
    for i, w_shape_i in enumerate(w_shape):
        if w_shape_i == -1:
            if w_range[i][1] is None:
                w_size *= Constant.SHAPE_SIZE_LIMIT
            else:
                w_size *= w_range[i][1]
            if w_min_shape[i] == -1:
                if w_range[i] != w_min_range[i]:
                    error_manager_vector.raise_err_inputs_shape_not_equal(
                        kernel_name, 'w', 'w_min', w_range, w_min_range, w_range)
                if w_range[i] != w_max_range[i]:
                    error_manager_vector.raise_err_inputs_shape_not_equal(
                        kernel_name, 'w', 'w_max', w_range, w_max_range, w_range)
        else:
            w_size *= w_shape_i

    if w_size > Constant.SHAPE_SIZE_LIMIT:
        error_detail = "The shape size of w must be smaller than {}!".format(Constant.SHAPE_SIZE_LIMIT)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, 'w', error_detail)


# 'pylint: disable=invalid-name, too-many-locals
@register_operator_compute('WtsARQ', op_mode='dynamic', support_fusion=False)
def wts_arq_compute(w, w_min, w_max, num_bits, offset_flag):
    """
    wts_arq compute
    do fake quantize for weights

    Parameters
    ----------
    w: TVM tensor
        the placeholder of w
    w_min: TVM tensor
        the placeholder of w_min
    w_max: TVM tensor
        the placeholder of w_max
    num_bits: int
        the bits num used for quantize
    offset_flag: bool
        whether use offset for quantize

    Returns
    -------
    y: TVM tensor
        the fake quantized weights
    """
    _, _, shape_broadcast = shape_util.broadcast_shapes(w.shape, w_min.shape)
    _, _, shape_broadcast = shape_util.broadcast_shapes(w.shape, w_max.shape)
    w = tbe.broadcast(w, shape_broadcast)
    w_min = tbe.broadcast(w_min, shape_broadcast)
    w_max = tbe.broadcast(w_max, shape_broadcast)

    const_0 = tvm.const(0.0, w.dtype)
    w_min = tbe.vmins(w_min, const_0)
    w_max = tbe.vmaxs(w_max, const_0)

    # const defination
    const_eps = tvm.const(Constant.EPS, 'float16')
    const_1 = tvm.const(1.0, 'float16')

    if offset_flag:
        # multiply w_max/w_min with const_step_reciprocal firstly,
        # incase (w_max - w_min) overflow float16
        const_step_reciprocal = tvm.const(1.0 / (2 ** num_bits - 1), w.dtype)
        scale_upper_bound = tbe.vmuls(w_max, const_step_reciprocal)
        scale_low_bound = tbe.vmuls(w_min, const_step_reciprocal)
        scale = tbe.vsub(scale_upper_bound, scale_low_bound)

        scale = tbe.cast_to(scale, 'float16')
        scale = tbe.vcmpsel(scale, const_eps, operation='lt', slhs=const_1, srhs=scale)
        scale = tbe.cast_to(scale, w.dtype)

        offset = tbe.round(tbe.vdiv(w_min, scale))
        offset = tbe.vmuls(offset, tvm.const(-1, w.dtype))
        const_bias = tbe.broadcast(tvm.const(2 ** (num_bits - 1), 'int32'), shape_broadcast)
        offset = tbe.vsub(offset, const_bias)
    else:
        const_step_low = tvm.const(2 ** (num_bits - 1), w.dtype)
        const_step_high = tvm.const(2 ** (num_bits - 1) - 1, w.dtype)

        step_low = tbe.broadcast(const_step_low, shape_broadcast, w.dtype)
        step_upper = tbe.broadcast(const_step_high, shape_broadcast, w.dtype)

        scale_1 = tbe.vdiv(tbe.vabs(w_min), step_low)
        scale_2 = tbe.vdiv(w_max, step_upper)
        scale = tbe.vmax(scale_1, scale_2)

        scale = tbe.cast_to(scale, 'float16')
        scale = tbe.vcmpsel(scale, const_eps, operation='lt', slhs=const_1, srhs=scale)
        scale = tbe.cast_to(scale, w.dtype)

    y = tbe.vdiv(w, scale)
    y = tbe.round(y)
    if offset_flag:
        y = tbe.vadd(y, offset)
    const_int8_low = tvm.const(-1 * 2 ** (num_bits - 1), y.dtype)
    const_int8_high = tvm.const(2 ** (num_bits - 1) - 1, y.dtype)
    y = tbe.vmaxs(y, const_int8_low)
    y = tbe.vmins(y, const_int8_high)
    if offset_flag:
        y = tbe.vsub(y, offset)
    y = tbe.cast_to(y, 'float16')
    y = tbe.cast_to(y, w.dtype)
    y = tbe.vmul(y, scale)

    return y


# 'pylint: disable=invalid-name, too-many-locals, too-many-arguments, too-many-branches, too-many-statements
@register_operator('WtsARQ')
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
    para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def wts_arq(w, w_min, w_max, y, num_bits=8, offset_flag=False, kernel_name='wts_arq'):
    """
    algorithm: weights adaptive range quantization
    get scale and offset, do fake quantize for weights

    Parameters
    ----------
    w: dict
        dict with keys(shape and dtype) of w
    w_min: dict
        dict with keys(shape and dtype) of w_min
    w_max: dict
        dict with keys(shape and dtype) of w_max
    y: dict
        dict with keys(shape and dtype) of y
    num_bits: int
        the bits num used for quantize
    offset_flag: bool
        whether use offset for quantize
    kernel_name : str
        cce kernel name, default value is "wts_arq"

    Returns
    -------
    None
    """
    if num_bits != 8:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'num_bits', 8, 8, num_bits)

    if not is_unknown_rank_input([w, w_min, w_max]):
        check_input_shape(w, w_min, w_max, kernel_name)

    check_list = ['float16', 'float32']
    w_type = w.get('dtype').lower()
    w_min_type = w_min.get('dtype').lower()
    w_max_type = w_max.get('dtype').lower()
    para_check.check_dtype_rule(w_type, check_list, 'w')

    if w_type != w_min_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'w', 'w_min', w_type, w_min_type)

    if w_type != w_max_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'w', 'w_max', w_type, w_max_type)

    ins = classify([w, w_min, w_max], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_w, _w_min, _w_max) in ins:
        with tbe.compute():
            shape_w, shape_w_min, shape_w_max = shape_util.variable_shape([_w, _w_min, _w_max])

            data_w = tvm.placeholder(shape_w, w_type, 'data_w')
            data_w_min = tvm.placeholder(shape_w_min, w_type, 'data_w_min')
            data_w_max = tvm.placeholder(shape_w_max, w_type, 'data_w_max')

            res = wts_arq_compute(data_w, data_w_min, data_w_max, num_bits, offset_flag)

            tensors.append([data_w, data_w_min, data_w_max, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        'name': kernel_name,
        'print_ir': False,
        'tensor_list': tensors,
        'bool_storage_as_1bit': True}

    tbe.build(schedules, config)
