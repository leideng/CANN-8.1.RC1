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
dynamic bn_training_update
"""
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_compute import only_static_support
from impl.util.util_common import is_unknown
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_soc_common import after_v200
from tbe.common.register import set_fusion_buildcfg
from tbe.dsl.base import operation


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-arguments,too-many-locals,redefined-builtin
def op_select_format(x,
                     sum,
                     square_sum,
                     scale,
                     offset,
                     mean,
                     variance,
                     y,
                     mean_out,
                     variance_out,
                     batch_mean,
                     batch_variance,
                     factor,
                     epsilon,
                     before_split_ori_shape=None,
                     before_split_ori_format=None,
                     kernel_name="bn_training_update"):
    """
    select format dynamically \n
    op_select_format support desc:
        support 5HD
        support NCHW when use dynamic template
    """
    x_support_dtype = ["float16", "float", "bfloat16"]

    scale_support_dtype = ["float"] * len(x_support_dtype)

    support_format = ["NC1HWC0", "NDC1HWC0", "NCDHW"]
    is_dy_template_support = check_supported(x, sum, square_sum, scale, offset, mean, variance, y, mean_out,
                                             variance_out, batch_mean, batch_variance, factor, epsilon, kernel_name)

    if is_dy_template_support[0]:
        support_format.append("NCHW")
        support_format.append("NHWC")

    # unfold dtype and format
    x_dtype_list = list()
    scale_dtype_list = list()
    format_list = list()
    for _format in support_format:
        x_dtype_list += x_support_dtype
        scale_dtype_list += scale_support_dtype
        format_list += [_format] * len(x_support_dtype)

    x_dtype_str = ",".join(x_dtype_list)
    scale_dtype_str = ",".join(scale_dtype_list)
    format_list_str = ",".join(format_list)

    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype=x_dtype_str,
                                           format=format_list_str,
                                           unknownshape_format=format_list_str)
    input1 = util_select_op_base.gen_param(classify="input1",
                                           name="sum",
                                           datatype=scale_dtype_str,
                                           format=format_list_str,
                                           unknownshape_format=format_list_str)
    input2 = util_select_op_base.gen_param(classify="input2",
                                           name="square_sum",
                                           datatype=scale_dtype_str,
                                           format=format_list_str,
                                           unknownshape_format=format_list_str)
    input3 = util_select_op_base.gen_param(classify="input3",
                                           name="scale",
                                           datatype=scale_dtype_str,
                                           format=format_list_str,
                                           unknownshape_format=format_list_str)
    input4 = util_select_op_base.gen_param(classify="input4",
                                           name="offset",
                                           datatype=scale_dtype_str,
                                           format=format_list_str,
                                           unknownshape_format=format_list_str)
    input5 = util_select_op_base.gen_param(classify="input5",
                                           name="mean",
                                           datatype=scale_dtype_str,
                                           format=format_list_str,
                                           unknownshape_format=format_list_str)
    input6 = util_select_op_base.gen_param(classify="input6",
                                           name="variance",
                                           datatype=scale_dtype_str,
                                           format=format_list_str,
                                           unknownshape_format=format_list_str)
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype=x_dtype_str,
                                            format=format_list_str,
                                            unknownshape_format=format_list_str)
    output1 = util_select_op_base.gen_param(classify="output1",
                                            name="mean",
                                            datatype=scale_dtype_str,
                                            format=format_list_str,
                                            unknownshape_format=format_list_str)
    output2 = util_select_op_base.gen_param(classify="output2",
                                            name="variance",
                                            datatype=scale_dtype_str,
                                            format=format_list_str,
                                            unknownshape_format=format_list_str)
    output3 = util_select_op_base.gen_param(classify="output3",
                                            name="batch_mean",
                                            datatype=scale_dtype_str,
                                            format=format_list_str,
                                            unknownshape_format=format_list_str)
    output4 = util_select_op_base.gen_param(classify="output4",
                                            name="batch_variance",
                                            datatype=scale_dtype_str,
                                            format=format_list_str,
                                            unknownshape_format=format_list_str)
    param_list = [input0, input1, input2, input3, input4, input5, input6, output0, output1, output2, output3, output4]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals,invalid-name,unused-argument,redefined-builtin
def check_special_soc():
    if tbe_platform.api_check_support("tik.vcopy"):
        return True

    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    if core_num in (30,):
        return True

    return False


def check_supported(x,
                    sum,
                    square_sum,
                    scale,
                    offset,
                    mean,
                    variance,
                    y,
                    mean_out,
                    variance_out,
                    batch_mean,
                    batch_variance,
                    factor,
                    epsilon,
                    before_split_ori_shape=None,
                    before_split_ori_format=None,
                    kernel_name="bn_training_update"):
    """
    check supported
    """
    if util_common.is_unknown(
        [x, sum, square_sum, scale, offset, mean, variance, y, mean_out, variance_out, batch_mean, batch_variance]):
        return True, ""

    # static shape
    if check_special_soc():
        return True, ""

    return False, ""


def get_op_support_info(x,
                        sum,
                        square_sum,
                        scale,
                        offset,
                        mean,
                        variance,
                        y,
                        mean_out,
                        variance_out,
                        batch_mean,
                        batch_variance,
                        factor,
                        epsilon,
                        before_split_ori_shape=None,
                        before_split_ori_format=None,
                        kernel_name="bn_training_update"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    if format_x == "NC1HWC0":
        axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])]]

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def simplify_shape(ins, data_format):
    if data_format in ("NHWC",):
        dim_c = ins.get("shape")[3]
        ins["shape"] = [1, 1, 1, dim_c]
        c_range = (1, None) if dim_c == -1 else (dim_c, dim_c)
        ins["range"] = [(1, 1), (1, 1), (1, 1), c_range]
    elif data_format in ("NCHW",):
        dim_c = ins.get("shape")[1]
        ins["shape"] = [1, dim_c, 1, 1]
        c_range = (1, None) if dim_c == -1 else (dim_c, dim_c)
        ins["range"] = [(1, 1), c_range, (1, 1), (1, 1)]
    elif data_format in ("NCDHW",):
        dim_c = ins.get("shape")[1]
        ins["shape"] = [1, dim_c, 1, 1, 1]
        c_range = (1, None) if dim_c == -1 else (dim_c, dim_c)
        ins["range"] = [(1, 1), c_range, (1, 1), (1, 1), (1, 1)]
    elif data_format in ("NC1HWC0",):
        dim_c1 = ins.get("shape")[1]
        ins["shape"] = [1, dim_c1, 1, 1, 16]
        c1_range = (1, None) if dim_c1 == -1 else (dim_c1, dim_c1)
        c0_range = (16, 16)
        ins["range"] = [(1, 1), c1_range, (1, 1), (1, 1), c0_range]
    else:
        dim_c1 = ins.get("shape")[2]
        ins["shape"] = [1, 1, dim_c1, 1, 1, 16]
        c1_range = (1, None) if dim_c1 == -1 else (dim_c1, dim_c1)
        c0_range = (16, 16)
        ins["range"] = [(1, 1), (1, 1), c1_range, (1, 1), (1, 1), c0_range]


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals
def _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale, dtype_offset, dtype_mean, dtype_variance):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
    dtype_sum: str
        sum's data type
    dtype_square_sum: str
        square_sum's data type
    dtype_scale: str
        scale's data type
    dtype_offset: str
        offset's data type
    dtype_mean: str
        mean's data type
    dtype_variance: str
        variance's data type

    Returns
    -------
    None
    """
    para_check.check_dtype(dtype_x, ("float16", "float32", "bfloat16"), param_name="x")
    para_check.check_dtype(dtype_sum, ("float32",), param_name="sum")
    para_check.check_dtype(dtype_square_sum, ("float32",), param_name="square_sum")
    para_check.check_dtype(dtype_scale, ("float32",), param_name="scale")
    para_check.check_dtype(dtype_offset, ("float32",), param_name="offset")
    para_check.check_dtype(dtype_mean, ("float32",), param_name="mean")
    para_check.check_dtype(dtype_variance, ("float32",), param_name="variance")


def support_ub_fusion():
    """
    check ub fusion support
        fommat is nchw can not support ub fusion
    """
    inputs = tbe_context.op_context.get_context().get_op_info()[0].inputs
    storage_format = inputs[0].get("format").upper()
    if storage_format == "NCHW":
        return False

    return only_static_support()


def is_ffts_for_bn(before_split_ori_shape, before_split_ori_format):
    """
    is_ffts_for_bn
        charge whether is ffts case
        when both before_split_ori_shape and before_split_ori_format is valid, return True
    """
    is_ori_shape_valid = before_split_ori_shape is not None and len(before_split_ori_shape) > 0
    is_ori_format_valid = before_split_ori_format is not None and len(before_split_ori_format) > 0
    if is_ori_shape_valid and is_ori_format_valid:
        return len(before_split_ori_shape) == len(before_split_ori_format)

    return False


def trans_format_for_bn(bn_format):
    """
    trans_format_for_bn
        trans the string format to int format base inc/external/graph/types.h
    """
    foramt_int_to_string_dict = {0: "NCHW", 1: "NHWC", 30: "NCDHW"}

    return foramt_int_to_string_dict.get(bn_format, "ND")


def trans_shape_for_bn(bn_shape, bn_format):
    """
    trans_shape_for_bn
        update the scale shape with x shape and x format
        bn_format must be ("NCHW", "NHWC")
    """
    if bn_format not in ("NCHW", "NHWC"):
        return bn_shape
    if len(bn_shape) == 1:
        return [1, bn_shape[0], 1, 1] if bn_format == "NCHW" else [1, 1, 1, bn_shape[0]]
    if len(bn_shape) == 2:
        return [bn_shape[0], bn_shape[1], 1, 1] if bn_format == "NCHW" else [bn_shape[0], 1, 1, bn_shape[1]]

    return bn_shape


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals,invalid-name,unused-argument
@register_operator_compute("BNTrainingUpdate", op_mode="dynamic", support_fusion=support_ub_fusion, support_bfp16=True)
def bn_training_update_compute(x,
                               sum,
                               square_sum,
                               scale,
                               offset,
                               mean,
                               variance,
                               y,
                               mean_out,
                               variance_out,
                               batch_mean,
                               batch_variance,
                               factor,
                               epsilon,
                               before_split_ori_shape,
                               before_split_ori_format,
                               kernel_name="bn_training_update"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    y: dict
        dict of output, A 'Tensor'. Has the same type as 'x'.
    mean_out: dict
        dict of mean, A 'Tensor'. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A 'Tensor'.
        The update variance of save variance and running variance.
    batch_mean: dict
        dict of batch_mean, A 'Tensor'.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A 'Tensor'.
        Has the same type as 'batch_mean'.
    factor: float
        A ratio to caculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    before_split_ori_shape: list_list_int
        ori_shape for input list, only valid in ffts.
    before_split_ori_format: list_int
        ori_format for input list, only valid in ffts.
    kernel_name: str
        kernel name, default value is "bn_training_update"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_compute
    """
    # set fusion build config to avoid the problem: fusion pass set dummy_placeholder default = False
    # when the input is unused, the cce will miss the input gm addr and trigger 0x800000
    build_cfg = {'dummy_placeholder': True}
    set_fusion_buildcfg("BNTrainingUpdate", build_cfg)

    if isinstance(factor, float):
        factor_reverse = 1.0 - factor
    else:
        factor_reverse = tbe.var("factor_reverse", dtype="float32")
        tbe_context.get_context().add_compile_info("has_factor_reverse", True)

    mode = operation.get_context().get_current_compute().get("_mode")
    if mode == "const":
        operation.get_context().get_current_compute().add("is_inplace_compute", True)

    factor = get_attr_by_cls(factor, OpAttr(0, "factor", "Float", 0.2), "float32")
    epsilon = get_attr_by_cls(epsilon, OpAttr(1, "epsilon", "Float", 0.0000001), "float32")

    if not util_common.is_unknown([y]):
        if is_ffts_for_bn(before_split_ori_shape, before_split_ori_format):
            # enter ffts cut fix
            data_format = trans_format_for_bn(before_split_ori_format[0])
            shape_y = trans_shape_for_bn(before_split_ori_shape[0], data_format)
        else:
            # static op
            data_format = y.get("format").upper()
            shape_y = y.get("shape")
        if data_format in ("NHWC",):
            reduce_dims = [shape_y[0], shape_y[1], shape_y[2]]
        elif data_format in ("NC1HWC0", "NCHW"):
            reduce_dims = [shape_y[0], shape_y[2], shape_y[3]]
        elif data_format in ("NDC1HWC0",):
            reduce_dims = [shape_y[0], shape_y[1], shape_y[3], shape_y[4]]
        elif data_format in ("NCDHW",):
            reduce_dims = [shape_y[0], shape_y[2], shape_y[3], shape_y[4]]
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
        batch_var_scalar = tbe.var("batch_var_scalar", dtype="float32")

    # compute the saved mean of x
    save_mean_reduce = tbe.vmuls(sum, num_rec)

    # compute the saved variance of x
    variance_div = tbe.vmuls(square_sum, num_rec)
    variance_square = tbe.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = tbe.vsub(variance_div, variance_square)

    # compute the oefficient of y
    if after_v200():
        save_variance_reduce = tbe.vmaxs(save_variance_reduce, tvm.const(0.0, save_variance_reduce.dtype))
        multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
    else:
        multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)
    multiplier_div.op.attrs["tensor_volume"] = 4096
    multiplier = tbe.broadcast(multiplier_div, x.shape)

    addend_mul = tbe.vmul(multiplier_div, save_mean_reduce)
    addend_sub = tbe.vsub(offset, addend_mul)
    addend_sub.op.attrs["tensor_volume"] = 4096
    addend = tbe.broadcast(addend_sub, x.shape)

    # compute the batch normalization of x
    x_dtype = x.dtype
    if x_dtype in ("float16",):
        x = tbe.cast_to(x, "float32")

    res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)

    if x_dtype in ("float16",):
        res_y = tbe.cast_to(res_y, x_dtype)

    batch_variance = tbe.vmuls(save_variance_reduce, batch_var_scalar)

    mean_mul = tbe.vmuls(save_mean_reduce, factor)
    mean_mul_rev = tbe.vmuls(mean, factor_reverse)
    res_mean = tbe.vadd(mean_mul, mean_mul_rev)
    res_mean.op.attrs["inplace_input"] = mean
    res_mean.op.attrs["tensor_volume"] = 4096

    var_mul = tbe.vmuls(batch_variance, factor)
    var_mul_rev = tbe.vmuls(variance, factor_reverse)
    res_variance = tbe.vadd(var_mul, var_mul_rev)
    res_variance.op.attrs["inplace_input"] = mean
    res_variance.op.attrs["tensor_volume"] = 4096

    res = [res_y, res_mean, res_variance, save_mean_reduce, save_variance_reduce]

    return res


def get_tail_clean_info(input_list):
    """
    get tail clean info for bn
    the clean info is a dict like {"tail_clean_info": (clean_offset, clean_count, fill_value)}
        clean_offset: the start addr for clean
        clean_count: the clean count, the dtype is the same as placeholder
        fill_value: fill value, usually the value is 0.0

    Parameters
    ----------
    input_list: list of dict
        list of dict, A 5HD Tensor for input data.

    Returns
    -------
    dict: {"tail_clean_info": (clean_offset, clean_count, fill_value)}
    """
    tail_clean_dict = dict()
    # set default info: (clean_offset, clean_count, clean_value)
    tail_clean_dict["tail_clean_info"] = (0, 0, 0)
    is_dynamic = util_common.is_unknown(input_list)
    if not input_list or not is_dynamic:
        return tail_clean_dict

    data_format = input_list[0].get("format")
    if data_format != "NC1HWC0":
        return tail_clean_dict

    # will set tbe.var to tail_clean_dict
    tbe_context.get_context().add_compile_info("is_need_tail_clean", True)
    clean_offset = tbe.var("clean_offset", dtype="int32")
    clean_count = tbe.var("clean_count", [0, 16], dtype="int32")
    tail_clean_dict["tail_clean_info"] = (clean_offset, clean_count, 0)

    return tail_clean_dict


# 'pylint: disable=too-many-statements,too-many-arguments,too-many-locals,invalid-name,unused-argument
@register_operator("BNTrainingUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_LIST_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def bn_training_update(x,
                       sum,
                       square_sum,
                       scale,
                       offset,
                       mean,
                       variance,
                       y,
                       mean_out,
                       variance_out,
                       batch_mean,
                       batch_variance,
                       factor,
                       epsilon,
                       before_split_ori_shape=None,
                       before_split_ori_format=None,
                       kernel_name="bn_training_update"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A 5HD Tensor for sum.
        The output of batch_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A 5HD Tensor for square_sum.
        The output of batch_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A 5HD Tensor for scale.
    offset: dict
        dict of offset, A 5HD Tensor for offset.
    mean: dict
        dict of mean, A 5HD Tensor for mean.
    variance: dict
        dict of variance, A 5HD Tensor for variance.
    y: dict
        dict of output, A 'Tensor'. Has the same type as 'x'.
    mean_out: dict
        dict of mean, A 'Tensor'. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A 'Tensor'. The update variance of save variance and running variance.
    batch_mean: dict
        dict of batch_mean, A 'Tensor'.
        One of the results that is called save mean.
    batch_variance: dict
        dict of batch_variance, A 'Tensor'.
        Has the same type as 'batch_mean'.
    factor: float
        A ratio to calculate the update mean and variance.
    epsilon: float
        A small float number added to the variance of x.
    before_split_ori_shape: list_list_int
        ori_shape for input list, only valid in ffts, default is None.
    before_split_ori_format: list_int
        ori_format for input list, only valid in ffts, default is None.
    kernel_name: str
        kernel name, default value is "bn_training_update".

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    dtype_sum = sum.get("dtype").lower()
    dtype_square_sum = square_sum.get("dtype").lower()
    dtype_scale = scale.get("dtype").lower()
    dtype_offset = offset.get("dtype").lower()
    dtype_mean = mean.get("dtype").lower()
    dtype_variance = variance.get("dtype").lower()
    _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale, dtype_offset, dtype_mean, dtype_variance)

    data_format = x.get("format")

    # handle dynamic dims
    if is_unknown_rank_input((x, sum, square_sum, scale, offset, mean, variance)) or factor is None or epsilon is None:
        if data_format == "NCHW":
            x["shape"] = [-1, -1, -1, -1]
            x["range"] = [(1, None), (1, None), (1, None), (1, None)]
            dynamic_shape = [1, -1, 1, 1]
            dynamic_range = [(1, 1), (1, None), (1, 1), (1, 1)]
        elif data_format == "NHWC":
            x["shape"] = [-1, -1, -1, -1]
            x["range"] = [(1, None), (1, None), (1, None), (1, None)]
            dynamic_shape = [1, 1, 1, -1]
            dynamic_range = [(1, 1), (1, 1), (1, 1), (1, None)]
        elif data_format == "NCDHW":
            x["shape"] = [-1, -1, -1, -1, -1]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None)]
            dynamic_shape = [1, -1, 1, 1, 1]
            dynamic_range = [(1, 1), (1, None), (1, 1), (1, 1), (1, 1)]
        elif data_format == "NC1HWC0":
            x["shape"] = [-1, -1, -1, -1, 16]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (16, 16)]
            dynamic_shape = [1, -1, 1, 1, 16]
            dynamic_range = [(1, 1), (1, None), (1, 1), (1, 1), (16, 16)]
        else:
            x["shape"] = [-1, -1, -1, -1, -1, 16]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None), (16, 16)]
            dynamic_shape = [1, 1, -1, 1, 1, 16]
            dynamic_range = [(1, 1), (1, 1), (1, None), (1, 1), (1, 1), (16, 16)]
        for input_dict in (sum, square_sum, scale, offset, mean, variance):
            input_dict["shape"] = dynamic_shape
            input_dict["range"] = dynamic_range

    for _ins in (sum, square_sum, scale, offset, mean, variance, mean_out, variance_out, batch_mean, batch_variance):
        if len(_ins.get("shape")) == 1:
            c_dim = _ins.get("shape")[0]
            if data_format in ("NCHW",):
                _ins["shape"] = [1, c_dim, 1, 1]
            elif data_format in ("NHWC",):
                _ins["shape"] = [1, 1, 1, c_dim]
            elif data_format in ("NCDHW",):
                _ins["shape"] = [1, c_dim, 1, 1, 1]

    # support fuzzy compile
    if util_common.is_unknown([x, sum, square_sum, scale, offset, mean, variance]):
        for _ins in (sum, square_sum, scale, offset, mean, variance):
            simplify_shape(_ins, data_format)

    ins_list = [x, sum, square_sum, scale, offset, mean, variance]
    tail_clean_info = get_tail_clean_info(ins_list)
    ins = classify(ins_list, OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedule_list, tensor_list = [], []

    for (ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset, ins_mean, ins_variance) in ins:
        with tbe.compute():
            _shape_x, _shape_sum, _shape_square_sum, _shape_scale, _shape_offset, _shape_mean, _shape_variance = \
                shape_util.variable_shape([ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset, ins_mean,
                                           ins_variance])

            x_input = tvm.placeholder(_shape_x, name="x_input", dtype=dtype_x)
            sum_input = tvm.placeholder(_shape_sum, name="sum_input", dtype=dtype_sum, attrs=tail_clean_info)
            square_sum_input = tvm.placeholder(_shape_square_sum,
                                               name="square_sum_input",
                                               dtype=dtype_square_sum,
                                               attrs=tail_clean_info)
            scale_input = tvm.placeholder(_shape_scale, name="scale_input", dtype=dtype_scale, attrs=tail_clean_info)
            offset_input = tvm.placeholder(_shape_offset,
                                           name="offset_input",
                                           dtype=dtype_offset,
                                           attrs=tail_clean_info)
            mean_input = tvm.placeholder(_shape_mean, name="mean_input", dtype=dtype_mean, attrs=tail_clean_info)
            variance_input = tvm.placeholder(_shape_variance,
                                             name="variance_input",
                                             dtype=dtype_variance,
                                             attrs=tail_clean_info)

            res = bn_training_update_compute(x_input,
                                             sum_input,
                                             square_sum_input,
                                             scale_input,
                                             offset_input,
                                             mean_input,
                                             variance_input,
                                             y,
                                             mean_out,
                                             variance_out,
                                             batch_mean,
                                             batch_variance,
                                             factor,
                                             epsilon,
                                             before_split_ori_shape,
                                             before_split_ori_format,
                                             kernel_name=kernel_name)
            tensor_list.append(
                [x_input, sum_input, square_sum_input, scale_input, offset_input, mean_input, variance_input] +
                list(res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedule_list.append(sch)

    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.build(schedule_list, config)
