# Copyright 2022 Huawei Technologies Co., Ltd
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
layer_norm_beta_gamma_backprop_v2
"""
from functools import reduce
from impl.util import util_common
from impl.util import util_gemm
from impl.util import util_select_op_base
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tuple_sum
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable = too-few-public-methods
class Constant:
    """
    common constants
    """
    DIM_FOUR = 4


def cal_prod_list(list_cal):
    return reduce(lambda x, y: x * y, list_cal)


def cal_fused_dim_num(list_cal):
    return -1 if -1 in list_cal else cal_prod_list(list_cal)


# 'pylint: disable=locally-disabled,unused-argument,too-many-arguments
# 'pylint: disable=too-many-locals,too-many-lines,too-many-function-args
def _check_dynamic_format(shape_dy, shape_gamma, c_0):
    """
    check dynamic format branch

    """
    if -2 in shape_dy or shape_gamma is None:
        return False
    if len(shape_dy) < 2 or len(shape_gamma) != 1:
        return True
    if shape_dy[-1] % c_0 != 0 or shape_dy[-2] % c_0 != 0 \
            or shape_gamma[-1] % c_0 != 0:
        return True
    return False


def _is_special_cases(input_shape):
    white_list_shape = [[64, 64, 1024], [96, 64, 1024]]
    shape_t = list(input_shape)
    if shape_t in white_list_shape:
        return True

    return False


def op_select_format(input_dy, input_x,
                     output_pd_gamma, output_pd_beta, shape_gamma,
                     kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
    function of selecting dynamic format

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support bfloat16, float16, float32
    input_x: dict
        shape and dtype of input x, only support float32
    output_pd_gamma: dict
        shape and dtype of output, only support float32
    output_pd_beta: dict
        shape and dtype of output, only support float32
    shape_gamma: list
        shape of gamma
    kernel_name: str
        cce kernel name, default value is "layer_norm_grad"

    Returns
    -------
    None
    """
    shape_dy = input_dy.get("ori_shape")
    shape_dy = shape_util.scalar2tensor_one(shape_dy)
    c_0 = 16
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")

    if _check_dynamic_format(shape_dy, shape_gamma, c_0) or _is_special_cases(shape_dy):
        input0 = util_select_op_base.gen_param(classify="input0", name="dy",
                                               datatype="float16,float16,float16,float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16",
                                               format="NCHW,NHWC,ND,NCHW,NHWC,ND,NCHW,NHWC,ND")
        input1 = util_select_op_base.gen_param(classify="input1", name="res_for_gamma",
                                               datatype="float,float,float,float,float,float,float,float,float",
                                               format="NCHW,NHWC,ND,NCHW,NHWC,ND,NCHW,NHWC,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="pd_gamma",
                                                datatype="float,float,float,float,float,float,float,float,float",
                                                format="NCHW,NHWC,ND,NCHW,NHWC,ND,NCHW,NHWC,ND")
        output1 = util_select_op_base.gen_param(classify="output1", name="pd_beta",
                                                datatype="float,float,float,float,float,float,float,float,float",
                                                format="NCHW,NHWC,ND,NCHW,NHWC,ND,NCHW,NHWC,ND")
    else:
        input0 = util_select_op_base.gen_param(classify="input0", name="dy",
                                               datatype="float16,float,bfloat16,float16,float16,"
                                                        "float16,float,float,float,"
                                                        "bfloat16,bfloat16,bfloat16",
                                               format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,"
                                                      "ND,NCHW,NHWC,ND,NCHW,NHWC,ND")
        input1 = util_select_op_base.gen_param(classify="input1", name="res_for_gamma",
                                               datatype="float,float,float,float,"
                                                        "float,float,float,float,"
                                                        "float,float,float,float",
                                               format="FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,"
                                                      "ND,NCHW,NHWC,ND,NCHW,NHWC,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="pd_gamma",
                                                datatype="float,float,float,float,"
                                                         "float,float,float,float,"
                                                         "float,float,float,float",
                                                format="ND,ND,ND,NCHW,NHWC,ND,NCHW,"
                                                       "NHWC,ND,NCHW,NHWC,ND")
        output1 = util_select_op_base.gen_param(classify="output1", name="pd_beta",
                                                datatype="float,float,float,float,float,"
                                                         "float,float,float,float,float,"
                                                         "float,float",
                                                format="ND,ND,ND,NCHW,NHWC,ND,NCHW,"
                                                       "NHWC,ND,NCHW,NHWC,ND")

    param_list = [input0, input1, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@tbe_register.register_param_generalization("LayerNormBetaGammaBackpropV2")
def layer_norm_beta_gamma_backprop_v2_generalization(input_dy, res_for_gamma, output_pd_gamma,
                                                     output_pd_beta, shape_gamma,
                                                     kernel_name="layer_norm_beta_gamma_backprop_v2",
                                                     generalize_config=None):
    format_dy = input_dy.get("format").upper()
    shape_dy = input_dy.get("shape")
    len_shape_dy = len(shape_dy)
    len_shape_gamma = len(shape_gamma)
    binary_shape_gamma = shape_gamma
    result = []
    if generalize_config.get("mode") == "keep_rank":
        shape_in = [-1] * len_shape_dy
        range_in = [[1, -1]] * len_shape_dy
        shape_out = [-1] * len_shape_gamma
        range_out = [[1, -1]] * len_shape_gamma

        dic_list = [dic
                    for dic in [input_dy, res_for_gamma, output_pd_gamma, output_pd_beta]
                    for i in range(4)]
        key_list = ["shape", "ori_shape", "range", "ori_range"] * 4
        value_list = [val
                      for val in [shape_in, range_in]
                      for i in range(2)] * 2 + \
                     [val
                      for val in [shape_out, range_out]
                      for i in range(2)] * 2
        for dic, key, value in zip(dic_list, key_list, value_list):
            dic[key] = value
    elif generalize_config.get("mode") == "all_shape":
        if format_dy == "FRACTAL_NZ":
            binary_shape_gamma = [-2] if len_shape_dy == 4 else [-2, -2]
            output_pd_gamma["format"] = "ND"
            output_pd_beta["format"] = "ND"
        else:
            binary_shape_gamma = [-2, -2] if len_shape_gamma == len_shape_dy else [-2]
            for ins in [input_dy, res_for_gamma, output_pd_gamma, output_pd_beta]:
                ins["format"] = "ND"
        for ins in [input_dy, res_for_gamma, output_pd_gamma, output_pd_beta]:
            ins["shape"] = util_gemm.DYNAMIC_UNKNOWN_RANK

    result.append([input_dy, res_for_gamma, output_pd_gamma, output_pd_beta, binary_shape_gamma])
    return result


def layer_norm_beta_gamma_backprop_v2_compute(input_dy, res_for_gamma, output_pd_gamma,
                                              output_pd_beta, shape_gamma,
                                              kernel_name="layer_norm_beta_gamma_backprop_v2",
                                              reduce_axis=None):
    """
    DSL description of the layernorm_grad operator's
    mathematical calculation process

    Parameters
    ----------
    input_dy: TVM tensor
        the placeholder of dy input data
    res_for_gamma: TVM tensor
        the placeholder of x input data
    output_pd_gamma: dict
        shape and dtype of output, only support float32
    output_pd_beta: dict
        shape and dtype of output, only support float32
    shape_gamma: list
        shape of gamma
    kernel_name: str
        cce kernel name, default value is "layer_norm_beta_gamma_backprop_v2"
    reduce_axis: list or tuple
        reduce axis of input shape

    Returns
    -------
    res_tuple: tuple
        (pd_gamma, pd_beta)
    """
    dtype = input_dy.dtype.lower()

    has_improve_precision = False
    if dtype in ("float16", "bfloat16"):
        has_improve_precision = True
        dtype = "float32"

    if has_improve_precision:
        input_dy = tbe.cast_to(input_dy, "float32")

    data_x = tbe.vmul(res_for_gamma, input_dy)
    if reduce_axis:
        pd_gamma, pd_beta = tuple_sum([data_x, input_dy], reduce_axis, keepdims=True)
    else:
        pd_beta = tbe.vadds(input_dy, tvm.const(0, dtype=dtype))
        pd_gamma = data_x

    res_list = [pd_gamma, pd_beta]

    return res_list


def _get_orig_reduce_axis(shape_dy, shape_gamma, format_dy):
    len_shape_dy = len(shape_dy)
    if format_dy == "FRACTAL_NZ":
        param_axis = []
        for i in range(len_shape_dy):
            if i not in (len_shape_dy - 1, len_shape_dy - Constant.DIM_FOUR):
                param_axis.append(i)
    else:
        param_axis = list(range(len_shape_dy - len(shape_gamma)))
    return param_axis


def _get_refine_shape_and_axis(input_dy, res_for_gamma, shape_gamma):
    shape_dy = input_dy.get("shape")
    format_dy = input_dy.get("format").upper()
    if tbe_context.get_context().get_op_mode() == "dynamic":
        if is_unknown_rank_input([input_dy, res_for_gamma]):
            if shape_gamma not in [[-2], [-2, -2]]:
                error_manager_vector.raise_err_specific_reson("LayerNormBetaGammaBackpropV2",
                    "This op does not support unknown rank scenarios other than binary matching.")
            if format_dy == "FRACTAL_NZ":
                refine_shape = [-1, -1, 16, 16] if shape_gamma == [-2] else [-1, -1, -1, 16]
                refine_reduce_axis = [1, 2] if shape_gamma == [-2] else [0, 2]
            else:
                refine_shape = [-1, -1] if shape_gamma == [-2] else [1, -1]
                refine_reduce_axis = [0] if shape_gamma == [-2] else []
        else:  # common dynamic
            if format_dy == "FRACTAL_NZ":
                if len(shape_dy) == 4:
                    refine_shape = shape_dy
                    refine_reduce_axis = [1, 2]
                else:
                    refine_shape = [-1, -1, -1, -1]
                    refine_shape[0] = cal_fused_dim_num(shape_dy[:-4])
                    refine_shape[1] = shape_dy[-4]
                    refine_shape[2] = cal_fused_dim_num(shape_dy[-3:-1])
                    refine_shape[3] = shape_dy[-1]
                    refine_reduce_axis = [0, 2]
            else:
                second_dim_num = cal_fused_dim_num(shape_gamma)
                if len(shape_gamma) != len(shape_dy):
                    refine_shape = [-1, second_dim_num]
                    refine_reduce_axis = [0]
                else:  # elewise
                    refine_shape = [1, second_dim_num]
                    refine_reduce_axis = []
    else:
        refine_shape = shape_dy
        refine_reduce_axis = _get_orig_reduce_axis(shape_dy, shape_gamma, format_dy)

    return refine_shape, refine_reduce_axis


@register_operator("LayerNormBetaGammaBackpropV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.KERNEL_NAME)
def layer_norm_beta_gamma_backprop_v2(input_dy, res_for_gamma, output_pd_gamma,
                                      output_pd_beta, shape_gamma,
                                      kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
    algorithm: layernorm_grad
    calculating: gradient of layernorm
                 compute partial derivation of x, gamma and beta
    pd_gamma = np.sum(input_dy*res_for_gamma, param_axis, keepdims=True)
    pd_beta  = np.sum(input_dy, param_axis, keepdims=True)

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support bfloat16, float16, float32
    res_for_gamma: dict
        shape and dtype of input res_for_gamma, only support float32
    output_pd_gamma: dict
        shape and dtype of output, only support float32
    output_pd_beta: dict
        shape and dtype of output, only support float32
    shape_gamma: list
        shape of gamma
    kernel_name: str
        cce kernel name, default value is "layer_norm_beta_gamma_backprop_v2"

    Returns
    -------
    None
    """
    dtype = input_dy.get("dtype").lower()
    dtype_x = res_for_gamma.get("dtype").lower()

    refine_shape, refine_reduce_axis = _get_refine_shape_and_axis(input_dy, res_for_gamma, shape_gamma)
    input_dy["shape"] = refine_shape
    res_for_gamma["shape"] = refine_shape
    input_dy["range"] = util_common.gen_range(refine_shape)
    res_for_gamma["range"] = util_common.gen_range(refine_shape)

    schedules = []
    tensors = []
    if refine_reduce_axis:
        tbe_context.get_context().add_compile_info("pattern_mode", "TupleReduce")
        ins = classify([input_dy, res_for_gamma, refine_reduce_axis], OpPatternMode.TUPLE_REDUCE)
        for (ins_dy, ins_x, ins_param_axis) in ins:
            with tbe.compute():
                shape_dy, shape_x = shape_util.variable_shape([ins_dy, ins_x], op_mode="tuple_reduce")
                data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype)
                data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
                res_list = layer_norm_beta_gamma_backprop_v2_compute(data_dy, data_x, output_pd_gamma, output_pd_beta,
                                                                     shape_gamma, kernel_name, ins_param_axis)
                tensor_list = [data_dy, data_x] + list(res_list)
                tensors.append(tensor_list)
            with tvm.target.cce():
                sch = tbe.auto_schedule(res_list)
            schedules.append(sch)
    else:
        tbe_context.get_context().add_compile_info("pattern_mode", "ElemWise")
        ins = classify([input_dy, res_for_gamma], OpPatternMode.ELEWISE)
        for (ins_dy, ins_x) in ins:
            with tbe.compute():
                shape_dy, shape_x = shape_util.variable_shape([ins_dy, ins_x])[0:2]
                data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype)
                data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
                res_list = layer_norm_beta_gamma_backprop_v2_compute(data_dy, data_x, output_pd_gamma, output_pd_beta,
                                                                     shape_gamma, kernel_name)
                tensor_list = [data_dy, data_x] + list(res_list)
                tensors.append(tensor_list)
            with tvm.target.cce():
                sch = tbe.auto_schedule(res_list)
            schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
