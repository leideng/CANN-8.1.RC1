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
# =============================================================================
"""
ascend_quant
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe_platform
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util import util_common
from impl import ascend_quant_util as quant_util


def is_enable_loop_partition():
    """
    reg base 平台需要打开此开关，以保证性能收益
    """
    soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if soc_version in ["Ascend610Lite", "BS9SX2A", "MC61AM21A"]:
        return True

    return False


def check_supported(x, y, scale, offset, sqrt_mode=False, round_mode="Round", dst_type=2,
                         kernel_name="ascend_quant"):
    """
    check whether dynamic is supported
    """
    if util_common.is_unknown([x, y]):
        return True, ""
    input_format = x.get("format")
    if input_format in ("ND", "NCHW"):
        return True, ""
    return False, ""


# 'pylint: disable=too-many-arguments,invalid-name,unused-argument,unnecessary-lambda,too-many-locals
@register_operator_compute("AscendQuant", op_mode="dynamic", support_fusion=True)
def ascend_quant_compute(x, y, scale, offset, sqrt_mode=False, round_mode="Round", dst_type=2,
                         kernel_name="ascend_quant"):
    """
    float16/float32 -> int8

    Parameters:
    ----------
    x : the tensor of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    dst_type : the output data type

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    y_format = y.get("format")
    dtype = x.dtype
    if y_format in ("NC1HWC0", "FRACTAL_NZ"):
        if quant_util.is_support_a100(is_quant=True) and x.op.name == "res_conv2d":
            scale_exp = tvm.const(scale, "float32")
            offset_exp = tvm.const(offset, "float32")

            ni, ci1, hiwi, ci0 = (i.value for i in x.shape)
            dst_type = 2
            align_num = 2 if dst_type == 2 else 4
            ci1_align = (ci1 + align_num - 1) // align_num

            input_shape = ni, ci1, hiwi, ci0
            output_shape = ni, ci1_align, hiwi, ci0 * align_num

            reform_x = tvm.compute(output_shape,
                                _reform_compute_generate(x, input_shape, output_shape, None, False),
                                name='reform_input')
            res = tvm.compute(
                output_shape,
                lambda *indice:
                tvm.quant_cast(reform_x(*indice), scale_exp, offset_exp, \
                               quant_util.Constant.DTYPE_2_STR_MAP.get(dst_type)),
                name='res_quant')
            return res

        
        in_shape = shape_util.shape_to_list(x.shape)
        nz_format_flag = quant_util.is_nz_format(x, True)

        c1_dim = in_shape[1]
        c1_index = 1
        if nz_format_flag:
            c1_index = len(in_shape) - 4
            c1_dim = in_shape[c1_index]

        y_dtype = quant_util.Constant.DTYPE_2_STR_MAP.get(dst_type)
        c1_transform = quant_util.Constant.C1_TRANS_MAP.get(y_dtype)

        read_shape, out_shape = _get_shape_info(in_shape, nz_format_flag, c1_transform)

        input_ub = _input_compute_generate(x, read_shape, c1_index, c1_transform)

        if dtype == "float32":
            cast_f16_ub = tvm.compute(read_shape, lambda *indice: shape_util.cast(input_ub(*indice), "float16"),
                                    name="cast_f16_ub", tag="cast_f16_ub")
            cast_i8_ub = _compute_scale(cast_f16_ub, in_shape, out_shape, (scale, offset, sqrt_mode, y_dtype),
                                        nz_format_flag)
        else:
            cast_i8_ub = _compute_scale(input_ub, in_shape, out_shape, (scale, offset, sqrt_mode, y_dtype),
                                        nz_format_flag)

        res = tvm.compute(out_shape, lambda *indice: cast_i8_ub(*indice), name="res", tag="quant",
                        attrs={"scale": scale,
                                "sqrt_mode": sqrt_mode,
                                "offset": offset,
                                "round_mode": round_mode,
                                "c1_dim": c1_dim,
                                "c1_transform": c1_transform})
    else:
        tbe_context.get_context().add_compile_info("dsl_compile", True)
        cast_f16_ub = tbe.cast_to(x, "float16")
        if sqrt_mode:
            scale_sqrt_ub = tbe.vmuls(cast_f16_ub, scale)
            scale_ub = tbe.vmuls(scale_sqrt_ub, scale)
        else:
            scale_ub = tbe.vmuls(cast_f16_ub, scale)
        offset_ub = tbe.vadds(scale_ub, offset)
        round_ub = _round_tensor(offset_ub, round_mode)
        y_dtype = quant_util.Constant.DTYPE_2_STR_MAP.get(dst_type)
        res = tbe.cast_to(round_ub, "int8")
    return res


def _round_tensor(offset_ub, round_mode):
    if round_mode == "Round":
        return tbe.round(offset_ub)
    elif round_mode == "Floor":
        return tbe.floor(offset_ub)
    elif round_mode == "Ceil":
        return tbe.ceil(offset_ub)
    else:
        return tbe.trunc(offset_ub)


def _get_shape_info(in_shape, nz_format_flag, c1_transform):
    """
    the compute of scale

    Parameters
    ----------
    in_shape: the shape of input tensor
    nz_format_flag: the format of output tensor

    Returns
    -------
    read_shape, out_shape
    """
    c0_index = len(in_shape) - 1
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4
    out_shape = in_shape[:]
    read_shape = in_shape[:]
    read_shape[c1_index] = (read_shape[c1_index] + c1_transform - 1) // c1_transform * c1_transform
    for dim, _ in enumerate(in_shape):
        if dim == c0_index:
            out_shape[dim] = in_shape[dim] * c1_transform
        if dim == c1_index:
            out_shape[dim] = (in_shape[dim] + c1_transform - 1) // c1_transform
    return read_shape, out_shape


def _check_params(x, round_mode, dst_type, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr
    """
    shape = x.get("shape")
    x_format = x.get("format")
    ori_shape = x.get("ori_shape")
    ori_format = x.get("ori_format")
    pos_c = ori_format.find('C')
    if pos_c < 0 or pos_c >= len(ori_shape):
        pos_c = len(ori_shape) - 1
    is_inner_format = False
    if x_format in ("NC1HWC0", "FRACTAL_NZ", "NDC1HWC0"):
        is_inner_format = True

    if x_format == "NDC1HWC0":
        para_check.check_shape(shape, min_rank=6, max_rank=6, param_name="x")
    if x_format == "NC1HWC0":
        para_check.check_shape(shape, min_rank=5, max_rank=5, param_name="x")
    if x_format == "FRACTAL_NZ":
        para_check.check_shape(shape, min_rank=4, param_name="x")

    round_mode_list = ["Round", "Ceil", "Floor", "Trunc"]
    if round_mode not in round_mode_list:
        rule = "round_mode only support [Round, Ceil, Floor, Trunc]"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "round_mode", round_mode)

    y_dtype = quant_util.Constant.DTYPE_2_STR_MAP.get(dst_type)
    y_check_list = ["int8"]
    para_check.check_dtype(y_dtype, y_check_list, param_name="dst_type")

    if ori_shape[pos_c] == -1 and is_inner_format:
        expected_value = "greater than -1"
        real_value = "less than or equal -1"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "c dimension",
                                                           expected_value, real_value)


def _input_compute_generate(x, read_shape, c1_index, c1_transform):
    """
    generate lambda func
    """
    dtype = x.dtype
    in_shape = shape_util.shape_to_list(x.shape)
    c1_dim = in_shape[c1_index]
    c1_is_var = bool(isinstance(c1_dim, tvm.Var))
    if not c1_is_var and c1_dim % c1_transform == 0:
        res = tvm.compute(in_shape, lambda *i: x(*i),
                          name="input_ub", tag="input_ub", attrs={"c_out": c1_dim, "c1_transform": c1_transform})
    else:
        zero = tvm.const(0, dtype=dtype)
        res = tvm.compute(read_shape,
                          lambda *indice: tvm.select(indice[c1_index] <= in_shape[c1_index] - 1, x(*indice), zero),
                          name='input_ub', tag="input_ub", attrs={"c_out": c1_dim, "c1_transform": c1_transform})
    return res


def _reform_by_vadds(input_tensor, input_shape, output_shape, offset_val, nz_format_flag):
    """
    5 dim input tensor C0 change

    Parameters
    ----------
    input_tensor: input tensor
    input_shape: the shape of input tensor
    output_shape: the shape of output tensor
    offset_val: the val of offset
    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    vadds_vector = tvm.compute(output_shape,
                               _reform_compute_generate(input_tensor, input_shape, output_shape,
                                                        (True, offset_val, -1), nz_format_flag),
                               name="reform_by_vadds", tag="reform_by_vadds")

    return vadds_vector


def _reform_compute_generate(tensor, in_shape, out_shape, val_info, nz_format_flag):
    """
    generate lambda func

    Parameters
    ----------
    tensor: input tensor
    in_shape: the shape of input tensor
    out_shape: the shape of output tensor
    val_info: the val info of offset,scale
    nz_format_flag: the format of input tensor

    Returns
    -------
    res lambda_func
    """
    in_shape = list(in_shape)
    out_shape = list(out_shape)
    n_dim = len(in_shape)

    c0_index = n_dim - 1
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4

    def lambda_func(*indice):
        """
        c1,c0 reform compute
        """
        new_indice = [0] * n_dim
        for i in range(n_dim):
            if i == c0_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] + indice[c0_index]) % in_shape[c0_index]
            elif i == c1_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] + indice[c0_index]) // in_shape[c0_index]
            else:
                new_indice[i] = indice[i]

        if val_info[0]:
            return tensor(*new_indice) + val_info[1]

        return tensor(*new_indice) * val_info[2]

    return lambda_func


def _reform_by_vmuls(input_tensor, input_shape, output_shape, scale_val, nz_format_flag):
    """
    5 dim input tensor C0 change

    Parameters
    ----------
    input_tensor: input tensor
    input_shape: the shape of input tensor
    output_shape: the shape of output tensor
    scale_val: the val of scale
    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    vmuls_vector = tvm.compute(output_shape,
                               _reform_compute_generate(input_tensor, input_shape, output_shape,
                                                        (False, -1, scale_val), nz_format_flag),
                               name="reform_by_vmuls", tag="reform_by_vmuls")

    return vmuls_vector


def _compute_offset(in_tensor, in_shape, out_shape, attr_list, nz_format_flag):
    """
    the compute of scale

    Parameters
    ----------
    in_tensor: input tensor
    in_shape: the shape of input tensor
    out_shape: the shape of output tensor
    attr_list: the attr list
    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    offset = attr_list[0]
    reform_flag = attr_list[1]
    scale = attr_list[2]
    y_dtype = attr_list[3]
    if offset != 0 or scale == 1:
        offset_value = tvm.const(offset, "float16")
        if reform_flag:
            offset_ub = _reform_by_vadds(in_tensor, in_shape, out_shape, offset_value, nz_format_flag)
        else:
            offset_ub = tvm.compute(out_shape, lambda *indice: in_tensor(*indice) + offset_value,
                                    name="offset_ub", tag="offset_ub")
        if y_dtype == "int8":
            res = tvm.compute(out_shape,
                              lambda *indice: shape_util.cast(offset_ub(*indice), "int8"), name="cast_i8_ub",
                              tag="cast_i8_ub")
        else:
            res = tvm.compute(out_shape,
                              lambda *indice: shape_util.cast(offset_ub(*indice), "int4"), name="cast_i4_ub",
                              tag="cast_i4_ub")

    else:
        if y_dtype == "int8":
            res = tvm.compute(out_shape,
                              lambda *indice: shape_util.cast(in_tensor(*indice), "int8"), name="cast_i8_ub",
                              tag="cast_i8_ub")
        else:
            res = tvm.compute(out_shape,
                              lambda *indice: shape_util.cast(in_tensor(*indice), "int4"), name="cast_i4_ub",
                              tag="cast_i4_ub")
    return res


def _compute_scale(in_tensor, in_shape, out_shape, attr_list, nz_format_flag):
    """
    the compute of scale

    Parameters
    ----------
    in_tensor: input tensor
    in_shape: the shape of input tensor
    out_shape: the shape of output tensor
    attr_list: the attr list
    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    scale = attr_list[0]
    offset = attr_list[1]
    sqrt_mode = attr_list[2]
    y_dtype = attr_list[3]
    if scale != 1:
        scale_value = tvm.const(scale, "float16")
        scale_ub = _reform_by_vmuls(in_tensor, in_shape, out_shape, scale_value, nz_format_flag)
        if sqrt_mode:
            scale_sqrt_ub = tvm.compute(out_shape, lambda *indice: scale_ub(*indice) * scale_value,
                                        name="scale_sqrt_ub", tag="scale_sqrt_ub")
            res = _compute_offset(scale_sqrt_ub, in_shape, out_shape, (offset, False, scale, y_dtype), nz_format_flag)
        else:
            res = _compute_offset(scale_ub, in_shape, out_shape, (offset, False, scale, y_dtype), nz_format_flag)
    else:
        res = _compute_offset(in_tensor, in_shape, out_shape, (offset, True, scale, y_dtype), nz_format_flag)
    return res


def get_op_support_info(x, y, scale, offset, sqrt_mode=False, round_mode="Round", dst_type=2,
                        kernel_name="ascend_quant"):
    """
    get split info
    """
    return quant_util.get_quant_support_info(x, l1_fusion_enable=1)


@register_operator("AscendQuant")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def ascend_quant(x, y, scale, offset, sqrt_mode=False, round_mode="Round", dst_type=2, kernel_name="ascend_quant"):
    """
    float16/float32 -> int8

    Parameters:
    ----------
    x : the dict of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    dst_type : the output data type

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    _check_params(x, round_mode, dst_type, kernel_name)
    input_dtype = x.get("dtype").lower()
    x_format = x.get("format")
    if x_format in ("NC1HWC0", "FRACTAL_NZ"):
        ins = classify([x], OpPatternMode.ASCEND_QUANT)
    else:
        ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            if x_format in ("NC1HWC0", "FRACTAL_NZ"):
                input_shape = shape_util.variable_shape([_x], OpPatternMode.ASCEND_QUANT)
            else:
                input_shape = shape_util.variable_shape([_x], OpPatternMode.ELEWISE)
            input_x = tvm.placeholder(input_shape[0], name="input_x", dtype=input_dtype)
            res = ascend_quant_compute(input_x, y, scale, offset, sqrt_mode, round_mode, dst_type, kernel_name)
            tensors.append([input_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    if x_format in ("NC1HWC0", "FRACTAL_NZ") and is_enable_loop_partition():
        config = {"print_ir": False,
                  "name": kernel_name,
                  "build_args": {"enable_loop_partition": True},
                  "tensor_list": tensors}
    else:
        config = {"print_ir": False,
                  "name": kernel_name,
                  "tensor_list": tensors}
    tbe.build(schedules, config)
