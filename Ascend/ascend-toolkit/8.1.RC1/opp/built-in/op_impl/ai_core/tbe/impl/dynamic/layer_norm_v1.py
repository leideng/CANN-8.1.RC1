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
layer_norm
"""
from copy import deepcopy
from tbe import tvm
from tbe.dsl.base import operation
from tbe.common.utils.errormgr import error_manager_vector
from tbe.common.platform import get_soc_spec
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from .layer_norm_tik import layer_normalize
from .layer_norm_tik import if_tik_support
from .layer_norm_tik import _check_input_mode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    LAST_DIM_RANGE_SET = ((1, 1), (2, 64), (65, 2000), (2001, None))


def to_frac_z_axis(ori_shape, ori_axis):
    """
    judge the format is fractal NZ

    Parameters
    ----------
    ori_shape: list or tuple
        original shape of input
    ori_axis: list or tuple
        original axis of original shape to operate

    Returns
    -------
    output: list
        axis of the fractal Nz shape
    """

    frac_z_axis = list(ori_axis)
    shape_len = len(ori_shape)
    axis_count = len(frac_z_axis)
    axis_negative_1 = shape_len - 1
    axis_negative_2 = shape_len - 2
    for i in range(axis_count):
        axis_index = (frac_z_axis[i] + shape_len) % shape_len
        if axis_index == axis_negative_1:
            if frac_z_axis[i] > shape_len - 2:
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 2)
        elif axis_index == axis_negative_2:
            frac_z_axis[i] = axis_index + 1
            frac_z_axis.append(axis_index + 2)
        else:
            frac_z_axis[i] = axis_index
    return frac_z_axis


def _broadcast_nz(tensor, shape):
    """
    broadcast_nz
    """
    broadcast_axes = []
    src_shape = shape_util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and broadcast_axes[1] - broadcast_axes[0] != 1 and broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = tbe.broadcast(tensor, temp_shape)
    tensor = tbe.broadcast(tensor, shape)
    return tensor


def layer_norm_compute_nz(input_x, input_gamma, input_beta,
                          output_y, output_mean, output_variance,
                          ori_reduce_axis, reduce_axis, begin_params_axis,
                          epsilon, kernel_name="layer_norm",
                          impl_mode="high_performance"):
    """
    DSL description of the layernorm operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    output_data: dict
        shape and dtype of output
    ori_reduce_axis: list
      the reduce  axis of ori_shape
    reduce_axis: list
      the reduce  axis of  shape
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "cce_layernorm"

    Returns
    -------
    res_tuple: tuple
        (mean, variance, result)
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    input_x1 = input_x
    cast_dtype = dtype
    cast_dtype_precision = dtype
    is_cast = False
    is_support_vexp = tbe_platform.api_check_support("te.lang.cce.vexp", "float32")
    tbe_context.get_context().add_compile_info("is_support_vexp", is_support_vexp)
    if dtype == "float16" and ((is_support_vexp and impl_mode == "high_performance")
                               or impl_mode == "high_precision"):
        cast_dtype = "float32"
        cast_dtype_precision = "float32"
        input_x = tbe.cast_to(input_x, "float32")
        input_x1 = tbe.cast_to(input_x1, "float32")
        input_gamma = tbe.cast_to(input_gamma, "float32")
        input_beta = tbe.cast_to(input_beta, "float32")
        is_cast = True

    # Calculate the scaling ratio of the average
    reduce_elts = 1.0
    for i in reduce_axis:
        reduce_elts *= shape_x[i]
    if isinstance(reduce_elts, float):
        mean_cofs = reduce_elts ** (-1)
        mean_cof = tvm.const(mean_cofs, dtype=cast_dtype)
    else:
        mean_cof = tbe.var("mean_cof", dtype=cast_dtype)
        operation.add_compile_info("reduce_mean_cof_dtype", cast_dtype)

    # DSL description of the mean calculation process
    mean_muls = tbe.vmuls(input_x, mean_cof)
    mean = tbe.reduce_sum(mean_muls, axis=reduce_axis, keepdims=True)

    if is_cast:
        mean_16 = tbe.cast_to(mean, "float16")
        mean = tbe.cast_to(mean_16, "float32")
    # DSL description of the variance calculation process
    mean_variance_broadcast = _broadcast_nz(mean, shape_x)
    variance_sub = tbe.vsub(input_x1, mean_variance_broadcast)
    variance_mul = tbe.vmul(variance_sub, variance_sub)
    variance_muls = tbe.vmuls(variance_mul, mean_cof)
    variance = tbe.reduce_sum(variance_muls, axis=reduce_axis, keepdims=True)
    if is_cast:
        variance_16 = tbe.cast_to(variance, "float16")
        variance = tbe.cast_to(variance_16, "float32")
    normalize_sub = variance_sub

    # DSL description of the normalize calculation process
    if impl_mode == "high_performance" and is_support_vexp:
        epsilon = tvm.const(epsilon, dtype=cast_dtype)
        variance_normalize_broadcast = _broadcast_nz(variance, shape_x)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_log = tbe.vlog(normalize_add)
        normalize_log_mul = \
            tbe.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
        normalize_exp = tbe.vexp(normalize_log_mul)
        normalize_mul = tbe.vmul(normalize_sub, normalize_exp)
    else:
        tesor_one = tbe.broadcast(tvm.const
                                  (1, cast_dtype_precision),
                                  shape_x)
        variance_normalize_broadcast = _broadcast_nz(variance, shape_x)
        epsilon = tvm.const(epsilon, dtype=cast_dtype_precision)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_sqrt = tbe.vsqrt(normalize_add)
        normalize_rsqrt = tbe.vdiv(tesor_one, normalize_sqrt)
        normalize_mul = tbe.vmul(normalize_sub, normalize_rsqrt)

    # DSL description of the scale and translate calculation process
    if begin_params_axis == 0:
        scale_mul = tbe.vmul(input_gamma, normalize_mul)
        res = tbe.vadd(scale_mul, input_beta)
    else:
        gamma_broadcast = _broadcast_nz(input_gamma, shape_x)
        beta_broadcast = _broadcast_nz(input_beta, shape_x)
        scale_mul = tbe.vmul(gamma_broadcast, normalize_mul)
        res = tbe.vadd(scale_mul, beta_broadcast)

    if is_cast:
        res = tbe.cast_to(res, "float16")
        return mean_16, variance_16, res

    return mean, variance, res


def layer_norm_compute(input_x,
                       input_gamma,
                       input_beta,
                       output_y,
                       output_mean,
                       output_variance,
                       reduce_axis,
                       begin_params_axis,
                       epsilon,
                       kernel_name="layer_norm",
                       impl_mode="high_performance"):
    """
    DSL description of the layernorm operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    output_data: dict
        shape and dtype of output
    reduce_axis: list
      the reduce axis
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "cce_layernorm"

    Returns
    -------
    res_tuple: tuple
        (mean, variance, result)
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    input_x1 = input_x
    cast_dtype = dtype
    cast_dtype_precision = dtype
    is_cast = False
    is_support_vexp = tbe_platform.api_check_support("te.lang.cce.vexp", "float32")
    tbe_context.get_context().add_compile_info("is_support_vexp", is_support_vexp)
    if dtype == "float16" and ((is_support_vexp and impl_mode == "high_performance")
                               or impl_mode == "high_precision"):
        cast_dtype = "float32"
        cast_dtype_precision = "float32"
        input_x = tbe.cast_to(input_x, "float32")
        input_x1 = tbe.cast_to(input_x1, "float32")
        input_gamma = tbe.cast_to(input_gamma, "float32")
        input_beta = tbe.cast_to(input_beta, "float32")
        is_cast = True

    reduce_elts = 1.0
    for i in reduce_axis:
        reduce_elts *= shape_x[i]
    if isinstance(reduce_elts, float):
        mean_cofs = reduce_elts ** (-1)
        mean_cof = tvm.const(mean_cofs, dtype=cast_dtype)
    else:
        mean_cof = tbe.var("mean_cof", dtype=cast_dtype)
        operation.add_compile_info("reduce_mean_cof_dtype", cast_dtype)

    # DSL description of the mean calculation process
    mean_muls = tbe.vmuls(input_x, mean_cof)
    mean = tbe.reduce_sum(mean_muls, axis=reduce_axis, keepdims=True)
    # workspace special case
    if is_cast:
        mean_16 = tbe.cast_to(mean, "float16")
        mean = tbe.cast_to(mean_16, "float32")

    # DSL description of the variance calculation process
    mean_variance_broadcast = tbe.broadcast(mean, shape_x)
    variance_sub = tbe.vsub(input_x1, mean_variance_broadcast)
    variance_mul = tbe.vmul(variance_sub, variance_sub)
    variance_muls = tbe.vmuls(variance_mul, mean_cof)
    variance = tbe.reduce_sum(variance_muls, axis=reduce_axis, keepdims=True)
    if is_cast:
        variance_16 = tbe.cast_to(variance, "float16")
        variance = tbe.cast_to(variance_16, "float32")
    normalize_sub = variance_sub

    # DSL description of the normalize calculation process
    if impl_mode == "high_performance" and is_support_vexp:
        epsilon = tvm.const(epsilon, dtype=cast_dtype)
        variance_normalize_broadcast = tbe.broadcast(variance, shape_x)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_log = tbe.vlog(normalize_add)
        normalize_log_mul = tbe.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
        normalize_exp = tbe.vexp(normalize_log_mul)
        normalize_mul = tbe.vmul(normalize_sub, normalize_exp)
    else:
        tesor_one = tbe.broadcast(tvm.const(1, cast_dtype_precision), shape_x)
        variance_normalize_broadcast = tbe.broadcast(variance, shape_x)
        epsilon = tvm.const(epsilon, dtype=cast_dtype_precision)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_sqrt = tbe.vsqrt(normalize_add)
        normalize_rsqrt = tbe.vdiv(tesor_one, normalize_sqrt)
        normalize_mul = tbe.vmul(normalize_sub, normalize_rsqrt)

    # DSL description of the scale and translate calculation process
    if begin_params_axis == 0:
        scale_mul = tbe.vmul(input_gamma, normalize_mul)
        res = tbe.vadd(scale_mul, input_beta)
    else:
        gamma_broadcast = tbe.broadcast(input_gamma, shape_x)
        beta_broadcast = tbe.broadcast(input_beta, shape_x)
        scale_mul = tbe.vmul(gamma_broadcast, normalize_mul)
        res = tbe.vadd(scale_mul, beta_broadcast)

    if is_cast:
        res = tbe.cast_to(res, "float16")
        return mean_16, variance_16, res

    return mean, variance, res


def layer_norm_v1(input_x,
                  input_gamma,
                  input_beta,
                  output_y,
                  output_mean,
                  output_variance,
                  begin_norm_axis,
                  begin_params_axis,
                  epsilon=1e-12,
                  kernel_name="layer_norm",
                  impl_mode="high_performance"):
    """
    layernorm operator interface implementation
    calculating: x, gamma, beta
        mean  = np.mean(x, reduce_axis, keepdims=True)
        variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True)
        result = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

    Parameters
    ----------
    input_x : dict
        shape and dtype of input x, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    input_beta: dict
        shape and dtype of input beta, only support float16, float32
    output_y: dict
        shape and dtype of output, only support float16, float32
    begin_norm_axis: int
      The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "layernorm"

    Returns
    -------
    None
    """
    shape_x = list(input_x.get("shape"))
    range_x = list(input_x.get("range"))
    ori_shape_x = list(input_x.get("ori_shape"))
    input_format = input_x.get("format").upper()
    input_gamma_format = input_gamma.get("format").upper()
    input_beta_format = input_beta.get("format").upper()

    check_list = ("float16", "float32")
    dtype = input_x.get("dtype").lower()
    dtype_gamma = input_gamma.get("dtype").lower()
    dtype_beta = input_gamma.get("dtype").lower()
    para_check.check_dtype(dtype, check_list, param_name="input_x")
    para_check.check_dtype(dtype_gamma, check_list, param_name="input_gamma")
    para_check.check_dtype(dtype_beta, check_list, param_name="input_gamma")

    shape_gamma = list(input_gamma.get("shape"))
    shape_beta = list(input_beta.get("shape"))

    ub_size = get_soc_spec("UB_SIZE")
    core_num = get_soc_spec("CORE_NUM")
    tik_mode = _check_input_mode(input_x)
    # 8kb for scalar in ub
    ub_max_byte = ub_size - 8192

    tik_support = if_tik_support(input_x, input_gamma, input_beta,
                                 output_y, output_mean, output_variance,
                                 begin_norm_axis, begin_params_axis, epsilon)

    atomic_clean_diff_shape = False
    add_compile_info_dict = {
        "input_format": input_format,
        "core_num": core_num,
        "begin_norm_axis": begin_norm_axis,
        "begin_params_axis": begin_params_axis,
        "is_tik_support": tik_support,
        "tik_mode": tik_mode,
        "ub_max_byte": ub_max_byte,
        "atomic_clean_diff_shape": atomic_clean_diff_shape
    }
    _add_compile_info_ops(add_compile_info_dict)
    if tik_support:
        tik_instance = layer_normalize(input_x,
                                       input_gamma,
                                       input_beta,
                                       output_y,
                                       output_mean,
                                       output_variance,
                                       begin_norm_axis,
                                       begin_params_axis,
                                       epsilon,
                                       kernel_name,
                                       atomic_clean_diff_shape)
    else:
        if input_format == "FRACTAL_NZ":
            begin_norm_axis = shape_util.axis_check(len(ori_shape_x), begin_norm_axis)
            begin_params_axis = shape_util.axis_check(len(ori_shape_x), begin_params_axis)

            if input_gamma_format == "FRACTAL_NZ" or input_beta_format == "FRACTAL_NZ":
                error_detail = "gamma and beta not support Nz in bert"
                error_manager_vector.raise_err_two_input_format_invalid(kernel_name, "input_gamma",
                                                                        "input_beta", error_detail)
            if shape_gamma != shape_beta:
                error_detail = "gamma and beta's shape must be same."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_gamma",
                                                                       "input_beta", error_detail)
            if ori_shape_x[begin_params_axis:] != shape_gamma:
                error_detail = "x or gamma or begin_params_axis is wrong."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x",
                                                                       "input_gamma", error_detail)
            if len(shape_gamma) > 1:
                error_detail = "shape of gamma or beta only support 1D in bert"
                error_manager_vector.raise_err_input_shape_invalid(kernel_name, "input_gamma", error_detail)

            if begin_params_axis != 0:
                for i in range(begin_params_axis):
                    shape_gamma.insert(i, 1)
            shape_gamma[-2] = shape_x[-4]
            shape_gamma[-1] = 1
            shape_gamma.append(1)
            shape_gamma.append(shape_x[-1])
            shape_beta = shape_gamma
            index_list = tuple(range(len(ori_shape_x)))
            ori_reduce_axis = index_list[begin_norm_axis:]
            reduce_axis = to_frac_z_axis(ori_shape_x, ori_reduce_axis)
            broadcast_axis = index_list[:begin_params_axis]
        else:
            begin_norm_axis = shape_util.axis_check(len(shape_x), begin_norm_axis)
            begin_params_axis = shape_util.axis_check(len(shape_x), begin_params_axis)

            if shape_gamma != shape_beta:
                error_detail = "gamma and beta's shape must be same."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_gamma", "input_beta",
                                                                       error_detail)
            no_need_fix_gamma = False
            no_need_fix_beta = False
            if shape_x[begin_params_axis:] != shape_gamma:
                if len(shape_x) == len(shape_gamma):
                    no_need_fix_gamma = True
                    no_need_fix_beta = True
                else:
                    error_detail = "x or gamma or begin_params_axis is wrong."
                    error_manager_vector.raise_err_two_input_shape_invalid(
                        kernel_name, "x", "input_gamma", error_detail)
            # make shape_x,shape_gamma,shape_beta dim same
            if begin_params_axis != 0 and not no_need_fix_gamma:
                for i in range(begin_params_axis):
                    shape_gamma.insert(i, 1)
            if begin_params_axis != 0 and not no_need_fix_beta:
                for i in range(begin_params_axis):
                    shape_beta.insert(i, 1)
            index_list = tuple(range(len(shape_x)))
            reduce_axis = index_list[begin_norm_axis:]
            broadcast_axis = index_list[:begin_params_axis]

        input_gamma["shape"] = tuple(shape_gamma)
        input_beta["shape"] = tuple(shape_beta)

        ins = _classify(input_x, input_gamma, input_beta, reduce_axis, broadcast_axis, input_format)
        schedules, tensors = [], []
        var_list = []
        for i in range(len(shape_x) - 1):
            dim_axis = operation.var_inner("_dim_" + str(i), range_x[i])
            var_list.append(dim_axis)
        var_list.append(operation.var_inner("_dim_" + str(len(shape_x) - 1)))

        for (dy_shape_x, dy_shape_gamma, dy_shape_beta, dy_reduce_axis) in ins[:1]:
            x_last_dim_range = dy_shape_x["range"][-1]
            if x_last_dim_range[0] == x_last_dim_range[-1]:
                x_last_dim_range_set = [x_last_dim_range]
            else:
                x_last_dim_range_set = Constant.LAST_DIM_RANGE_SET
            for _, rn in enumerate(x_last_dim_range_set):
                with tbe.compute():
                    x_var, gamma_var, beta_var, _ = _reduce_variable_shape(
                        [dy_shape_x, dy_shape_gamma, dy_shape_beta, dy_reduce_axis], var_list, rn, input_format)
                    data_x = tvm.placeholder(x_var, name="x", dtype=dtype)
                    data_gamma = tvm.placeholder(gamma_var, name="gamma", dtype=dtype)
                    data_beta = tvm.placeholder(beta_var, name="beta", dtype=dtype)

                    if input_format == "FRACTAL_NZ":
                        mean, variance, res = layer_norm_compute_nz(data_x, data_gamma, data_beta,
                                                                    output_y, output_mean, output_variance,
                                                                    ori_reduce_axis, dy_reduce_axis.get("value"),
                                                                    begin_params_axis, epsilon,
                                                                    kernel_name, impl_mode)
                    else:
                        mean, variance, res = layer_norm_compute(data_x, data_gamma, data_beta,
                                                                 output_y, output_mean, output_variance,
                                                                 dy_reduce_axis.get("value"),
                                                                 begin_params_axis, epsilon,
                                                                 kernel_name, impl_mode)
                    tensors.append([data_x, data_gamma, data_beta, res, mean, variance])
                with tvm.target.cce():
                    sch = tbe.auto_schedule([res, mean, variance])

                schedules.append(sch)

        config = {
            "print_ir": False,
            "name": kernel_name,
            "tensor_list": tensors
        }

        tbe.build(schedules, config)


def _add_compile_info_ops(add_compile_info_dict):
    for k, v in add_compile_info_dict.items():
        tbe_context.get_context().add_compile_info(k, v)


def _classify(input_x, input_gamma, input_beta, reduce_axis, broadcast_axis, input_format):
    """
     Parameters
    ----------
    input_x : dict
        shape and dtype of input x, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    input_beta: dict
        shape and dtype of input beta, only support float16, float32
    reduce_axis: index_list[begin_norm_axis:]
    broadcast_axis: index_list[:begin_params_axis]
    input_format: format of input_x
    """
    input_x, input_gamma, input_beta = generate_reduce_input((input_x, input_gamma, input_beta), input_format)
    x_range = input_x.get("range")
    gamma_shape = input_gamma.get("shape")
    dynamic_index = []
    for i, val in enumerate(gamma_shape):
        if val == -1:
            dynamic_index.append(i)
    gamma_range = input_gamma.get("range")
    beta_shape = input_beta.get("shape")
    beta_range = input_beta.get("range")
    input_x_list = _generate_all_ins(input_x)
    outs = []
    for ix in input_x_list:
        new_gamma_shape = deepcopy(gamma_shape)
        new_beta_shape = deepcopy(beta_shape)
        for i in dynamic_index:
            new_gamma_shape[i] = ix[i]
            new_beta_shape[i] = ix[i]
        if -1 not in ix and -2 not in ix:
            mode = para_check.CONST
        else:
            mode = para_check.ORIGINAL

        ixd = {'shape': ix, 'range': x_range, 'mode': mode, 'rel_pos_to_reduce': 'before'}
        igd = {'shape': new_gamma_shape, 'range': gamma_range, 'mode': mode, 'rel_pos_to_reduce': 'before'}
        ibd = {'shape': new_beta_shape, 'range': beta_range, 'mode': mode, 'rel_pos_to_reduce': 'before'}
        outs.append([ixd, igd, ibd])

    for ins in outs:
        ins.append({'shape': reduce_axis, 'value': reduce_axis, 'rel_pos_to_reduce': 'axis'})
    return outs


def _fuse_shape_operation(x_dict, res_fuse_axis):
    """
    fuse_shape_operation
    """
    x_shape = x_dict.get("shape")
    x_range = x_dict.get("range")
    for fx in res_fuse_axis:
        reduce_num = 1
        res = [x_shape[d] for d in fx]
        reduce_num = 1 if res == [1] * len(fx) else -1

        x_shape = x_shape[:fx[0]] + [reduce_num] + x_shape[fx[-1] + 1:]
        x_range = x_range[:fx[0] + 1] + x_range[fx[-1] + 1:]
    x_dict["shape"] = x_shape
    x_dict["range"] = x_range
    return x_dict


def _process_all_unknown_shape(shape_list, range_list):
    """
    process input include shape -2
    """
    all_unknown_shape_len = 8
    for single_shape in shape_list:
        if tuple(single_shape) != (-2,):
            all_unknown_shape_len = len(single_shape)
            break

    for idx, single_shape in enumerate(shape_list):
        if tuple(single_shape) == (-2,):
            shape_list[idx] = [-1] * all_unknown_shape_len
            range_list[idx] = [(0, None)] * all_unknown_shape_len
    return shape_list, range_list


def generate_reduce_input(inputs_before_reduce, input_format):
    """
    generate_reduce_input
    """
    shape_local = [list(x["shape"]) for x in inputs_before_reduce]
    range_local = [
        list(x.get("range")) if list(x.get("range")) else [(1, None)] * len(shape_local[0])
        for x in inputs_before_reduce
    ]

    shape_list, range_list = _process_all_unknown_shape(shape_local, range_local)
    max_len = max((len(x_shape) for x_shape in shape_list))
    new_shape_local = [[1] * (max_len - len(x_shape)) + x_shape if len(x_shape) != max_len else x_shape
                       for x_shape in shape_list]
    if input_format == "FRACTAL_NZ":
        new_range_local = [[(1, None)] * max_len if len(x_range) != max_len else x_range
                           for x_range in range_list]
    else:
        new_range_local = [[(1, None)] * (max_len - len(x_range)) + x_range if len(x_range) != max_len else x_range
                           for x_range in range_list]

    for index, _ in enumerate(new_shape_local):
        for idx, _ in enumerate(new_shape_local[index]):
            if new_range_local[index][idx][0] == new_range_local[index][idx][1]:
                new_shape_local[index][idx] = new_range_local[index][idx][0]
    for xid, xval in enumerate(new_shape_local[0]):
        if xval != -1:
            new_range_local[0][xid] = (xval, xval)
    for i, x in enumerate(inputs_before_reduce):
        x["shape"] = new_shape_local[i]
        x["range"] = new_range_local[i]
    return inputs_before_reduce


def _generate_all_ins(inputx):
    """
    generate_all_ins
    """
    x_shape = inputx["shape"]
    x_range = inputx["range"]
    x_len = len(x_shape)
    outs = []

    def _generate_all_combination(itern, out_list):
        res = []
        for dims in itern:
            if not out_list:
                res.append([dims])
            else:
                sub_out_list = deepcopy(out_list)
                for out in sub_out_list:
                    out.append(dims)
                res += sub_out_list
        return res

    for i in range(x_len):
        dim = x_shape[i]
        dim_range = x_range[i]
        if dim == -1 and list(dim_range)[0] == 1:
            itern = (-1, 1)
        else:
            itern = (dim,)
        outs = _generate_all_combination(itern, outs)
    return outs


def _reduce_variable_shape(inputs, var_list, dim_ln_range, x_dtype=None):
    """
    variable shape for reduce ops
    """
    inputs_before_reduce, inputs_after_reduce, input_axis = [], [], []
    for single_input in inputs:
        input_type = single_input.get("rel_pos_to_reduce")
        if input_type == "axis":
            input_axis.append(single_input)
        elif input_type == "after":
            inputs_after_reduce.append(single_input)
        else:
            inputs_before_reduce.append(single_input)

    if not inputs:
        return []
    mode = inputs_before_reduce[0].get("mode")
    if mode is None:
        mode = para_check.ORIGINAL
    operation.get_context().add("mode", mode)
    current_compute = operation.get_context().get_current_compute()
    if current_compute:
        current_compute.add("mode", mode)

        ori_axis = input_axis[0].get("ori_axis")
        if ori_axis is not None:
            current_compute.add("ori_axis", ori_axis)
        axis_dtype = input_axis[0].get("axis_dtype")
        if axis_dtype is not None:
            current_compute.add("axis_dtype", axis_dtype)

    shape_local = [x["shape"] for x in inputs_before_reduce]
    current_compute.add("dim_ln_range", dim_ln_range)
    current_compute.add("input_format", x_dtype)

    shape_before_reduce = []

    for i, _ in enumerate(shape_local):
        single_shape_before_reduce = []
        for index in range(len(shape_local[i])):
            if shape_local[i][index] == -1:
                _var = var_list[index]
                single_shape_before_reduce.append(_var)
            else:
                single_shape_before_reduce.append(shape_local[i][index])
        shape_before_reduce.append(single_shape_before_reduce)

    shape_out = []
    for i, single_input in enumerate(inputs):
        input_type = single_input.get("rel_pos_to_reduce")
        if input_type == "before":
            shape_out.append(shape_before_reduce[i][:])
        else:
            shape_out.append(input_axis[0].get("shape")[:])
    return shape_out
