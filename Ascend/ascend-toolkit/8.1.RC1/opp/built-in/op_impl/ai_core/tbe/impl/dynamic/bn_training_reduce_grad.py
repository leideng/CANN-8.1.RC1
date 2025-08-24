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
dynamic bn_training_reduce_grad
"""
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.util_compute import only_static_support
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.dynamic.bn_training_update import check_special_soc
from impl.dynamic.bn_training_update import get_tail_clean_info
from impl.dynamic.bn_training_update import is_ffts_for_bn
from impl.dynamic.bn_training_update import trans_format_for_bn
from impl.dynamic.bn_training_update import trans_shape_for_bn


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals,too-many-statements
def op_select_format(grads,
                     x,
                     diff_scale,
                     diff_offset,
                     scale,
                     batch_mean,
                     batch_variance,
                     y,
                     epsilon,
                     before_split_ori_shape=None,
                     before_split_ori_format=None,
                     kernel_name="bn_training_reduce_grad"):
    """
    1. when input(grads)'s ori_shape is [1, ? ,1, ?] and the format is NCHW
    the Op BNTrainingReduceGrad can support NCHW.
    > for example:
    > grads : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > x : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > diff_scale : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > diff_offset : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > scale : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > batch_mean : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > batch_variance : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > the Op BNTrainingReduce can process with NC1HWC0:
    > grads : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > diff_scale : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > diff_offset : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > scale : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > batch_mean : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > batch_variance : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    """
    format_grads = grads.get("ori_format").upper()
    origin_shape = grads.get("ori_shape")
    is_unknow = util_common.is_unknown([grads, x])
    if format_grads == "NCHW" and len(origin_shape) == 4 and origin_shape[0] == 1 and origin_shape[2] == 1:
        format_unknow = "NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NCHW,NHWC,NC1HWC0"
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="grads",
                                               datatype="float16,float,float16,float,float16,float,\
                                                         bfloat16,bfloat16,bfloat16",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format=format_unknow)
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16,float,float16,float,float16,float,\
                                                         bfloat16,bfloat16,bfloat16",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format=format_unknow)
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="diff_scale",
                                               datatype="float,float,float,float,float,float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format=format_unknow)
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="diff_offset",
                                               datatype="float,float,float,float,float,float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format=format_unknow)
        input4 = util_select_op_base.gen_param(classify="input4",
                                               name="scale",
                                               datatype="float,float,float,float,float,float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format=format_unknow)
        input5 = util_select_op_base.gen_param(classify="input5",
                                               name="batch_mean",
                                               datatype="float,float,float,float,float,float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format=format_unknow)
        input6 = util_select_op_base.gen_param(classify="input6",
                                               name="batch_variance",
                                               datatype="float,float,float,float,float,float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format=format_unknow)
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype="float16,float,float16,float,float16,float,\
                                                         bfloat16,bfloat16,bfloat16",
                                                format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                                unknownshape_format=format_unknow)
    else:
        format_unknow = "NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NHWC,NHWC,NCHW,NCHW,NC1HWC0,NDC1HWC0,NHWC,NCHW,\
            NCDHW,NCDHW,NCDHW"
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="grads",
                                               datatype="float16,float,float16,float,float16,float,float16,float,\
                                                         bfloat16,bfloat16,bfloat16,bfloat16,float16,float,bfloat16",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NCDHW,NCDHW,NCDHW",
                                               unknownshape_format=format_unknow)
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16,float,float16,float,float16,float,float16,float,\
                                                         bfloat16,bfloat16,bfloat16,bfloat16,float16,float,bfloat16",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NCDHW,NCDHW,NCDHW",
                                               unknownshape_format=format_unknow)
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="diff_scale",
                                               datatype="float,float,float,float,float,float,float,float,\
                                                         float,float,float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NCDHW,NCDHW,NCDHW",
                                               unknownshape_format=format_unknow)
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="diff_offset",
                                               datatype="float,float,float,float,float,float,float,float,\
                                                         float,float,float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NCDHW,NCDHW,NCDHW",
                                               unknownshape_format=format_unknow)
        input4 = util_select_op_base.gen_param(classify="input4",
                                               name="scale",
                                               datatype="float,float,float,float,float,float,float,float,\
                                                         float,float,float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NCDHW,NCDHW,NCDHW",
                                               unknownshape_format=format_unknow)
        input5 = util_select_op_base.gen_param(classify="input5",
                                               name="batch_mean",
                                               datatype="float,float,float,float,float,float,float,float,\
                                                         float,float,float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NCDHW,NCDHW,NCDHW",
                                               unknownshape_format=format_unknow)
        input6 = util_select_op_base.gen_param(classify="input6",
                                               name="batch_variance",
                                               datatype="float,float,float,float,float,float,float,float,\
                                                         float,float,float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NCDHW,NCDHW,NCDHW",
                                               unknownshape_format=format_unknow)
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype="float16,float,float16,float,float16,float,float16,float,\
                                                          bfloat16,bfloat16,bfloat16,bfloat16,float16,float,bfloat16",
                                                format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                        NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                        NCDHW,NCDHW,NCDHW",
                                                unknownshape_format=format_unknow)

    param_list = [input0, input1, input2, input3, input4, input5, input6, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def check_supported(grads,
                    x,
                    diff_scale,
                    diff_offset,
                    scale,
                    batch_mean,
                    batch_variance,
                    y,
                    epsilon,
                    before_split_ori_shape=None,
                    before_split_ori_format=None,
                    kernel_name="bn_training_reduce_grad"):
    """
    check supported
    """
    if util_common.is_unknown([grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance]):
        return True, ""

    # static shape only support special soc
    if check_special_soc():
        return True, ""

    return False, ""


def get_op_support_info(grads,
                        x,
                        diff_scale,
                        diff_offset,
                        scale,
                        batch_mean,
                        batch_variance,
                        y,
                        epsilon,
                        before_split_ori_shape=None,
                        before_split_ori_format=None,
                        kernel_name="bn_training_reduce_grad"):
    """
    get_op_support_info
    """
    format_grads = grads.get("format").upper()
    if format_grads == "NC1HWC0":
        axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]], [1, [0], [-1], [-1]]), SplitOutput([0, [0]])]]

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _check_format(data_format, origin_foramt):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_foramt: str
        origin format of data

    Returns
    -------
    None
    """
    if data_format.upper() not in ("NC1HWC0", "NCHW", "NDC1HWC0", "NHWC", "NCDHW"):
        error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad",
                                                      "The data format only supports NC1HWC0,NDC1HWC0,and NCHW.")
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad",
                                                          "The origin format only supports NCHW when format is NCHW")


def support_ub_fusion():
    """
    check ub fusion support
        can not support ub fusion when format is nchw
    """
    inputs = tbe_context.op_context.get_context().get_op_info()[0].inputs
    storage_format = inputs[0].get("format").upper()
    if storage_format == "NCHW" or storage_format == "NCDHW":
        return False

    return only_static_support()


@register_operator_compute("BNTrainingReduceGrad", op_mode="dynamic", support_fusion=support_ub_fusion)
def bn_training_reduce_grad_compute(grads,
                                    x,
                                    diff_scale,
                                    diff_offset,
                                    scale,
                                    batch_mean,
                                    batch_variance,
                                    y,
                                    epsilon,
                                    before_split_ori_shape=None,
                                    before_split_ori_format=None,
                                    kernel_name="bn_training_reduce_grad"):
    """
    Compute for batch_norm_grad
    y:(grads*scale*np.power((batch_variance + epsilon), (-0.5)))+
      np.sum(grads*scale*(-0.5)*x_norm*np.power((batch_variance+epsilon),(-1))))
      *(2/m)+np.sum(grads*scale*(-1)*
      np.power((batch_variance+epsilon),(-0.5)))*(1/m)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads.
        Must be one of the following type: `float16`, `float32`, `bfloat16`.
    x: TVM tensor 5D
        the placeholder of x.
        Must be one of the following type: `float32`, 'float16`, `bfloat16`.
    diff_scale: TVM tensor 5D
        the placeholder of diff_scale.
        Must be one of the following type: `float32`.
    diff_offset: TVM tensor 5D
         the placeholder of diff_offset.
         Must be one of the following types: `float32`.
    scale: TVM tensor 5D
        the placeholder of scale.
        Must be one of the following types: `float32`.
    batch_mean: dict 5D
        the placeholder of batch_mean.
        Must be one of the following types: `float32`.
    batch_variance: dict 5D
        the placeholder of batch_variance.
        Must be one of the following types: `float32`.
    y: dict
        dict of y, include keys(shape and dtype).
    epsilon: float
        A small float number added to the variance of x.
    before_split_ori_shape: list_list_int
        ori_shape for input list, only valid in ffts, default is None.
    before_split_ori_format: list_int
        ori_format for input list, only valid in ffts, default is None.
    kernel_name: str
        kernel name, default value is "bn_training_reduce_grad".

    Returns
    -------
    res: TVM tensor
    """
    epsilon = get_attr_by_cls(epsilon, OpAttr(0, "epsilon", "Float", 0.0000001), "float32")
    is_cast = False
    ori_dtype = grads.dtype
    if grads.dtype in ("float16", "bfloat16"):
        is_cast = True
        grads = tbe.cast_to(grads, "float32")
        grads.op.attrs["mem_unique"] = True

    if x.dtype in ("float16", "bfloat16"):
        is_cast = True
        x = tbe.cast_to(x, "float32")
        x.op.attrs["mem_unique"] = True

    if not util_common.is_unknown([y]):
        # charge ffts
        if is_ffts_for_bn(before_split_ori_shape, before_split_ori_format):
            data_format = trans_format_for_bn(before_split_ori_format[0])
            shape_y = trans_shape_for_bn(before_split_ori_shape[0], data_format)
        else:
            data_format = y.get("format").upper()
            shape_y = y.get("shape")

        if data_format in ("NC1HWC0", "NCHW"):
            reduce_dims = [shape_y[0], shape_y[2], shape_y[3]]
        elif data_format in ("NHWC",):
            reduce_dims = [shape_y[0], shape_y[1], shape_y[2]]
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
        neg_num_rec = tvm.const(-num_bw, dtype="float32")
    else:
        num_rec = tbe.var("num_rec", dtype="float32")
        neg_num_rec = tbe.var("neg_num_rec", dtype="float32")

    data_sqrt = tbe.vsqrt(tbe.vadds(batch_variance, epsilon))
    scale_inv = tbe.vmuls(diff_scale, num_rec)
    scale_inv_reverse = tbe.vmuls(diff_scale, neg_num_rec)
    offset_inv_reverse = tbe.vmuls(diff_offset, neg_num_rec)

    multiplier = tbe.vdiv(scale_inv_reverse, data_sqrt)
    multiplier.op.attrs["tensor_volume"] = 4096
    addend_div = tbe.vdiv(batch_mean, data_sqrt)
    addend_mul = tbe.vmul(addend_div, scale_inv)
    addend = tbe.vadd(addend_mul, offset_inv_reverse)
    addend.op.attrs["tensor_volume"] = 4096

    multiplier_broadcast = tbe.broadcast(multiplier, grads.shape)
    addend_broadcast = tbe.broadcast(addend, grads.shape)

    coef_mul = tbe.vmul(multiplier_broadcast, x)
    coef_add = tbe.vadd(grads, coef_mul)
    coef = tbe.vadd(coef_add, addend_broadcast)

    mul_scale = tbe.vdiv(scale, data_sqrt)
    mul_scale.op.attrs["tensor_volume"] = 4096
    mul_scale_broadcast = tbe.broadcast(mul_scale, grads.shape)

    res = tbe.vmul(coef, mul_scale_broadcast)

    if is_cast and ori_dtype == "float16":
        res = tbe.cast_to(res, "float16")
    if is_cast and ori_dtype == "bfloat16":
        res = tbe.round(res, "bfloat16")
    return res


@register_operator("BNTrainingReduceGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_LIST_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def bn_training_reduce_grad(grads,
                            x,
                            diff_scale,
                            diff_offset,
                            scale,
                            batch_mean,
                            batch_variance,
                            y,
                            epsilon,
                            before_split_ori_shape=None,
                            before_split_ori_format=None,
                            kernel_name="bn_training_reduce_grad"):
    """
    algorithm: batch_norm_grad
    bn_training_reduce_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
        source data type, support "float32", "float16", `bfloat16`.
    x: dict
        dict of s, A 5D Tensor for input x.
        source data type, support "float32", "float16", `bfloat16`.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for input diff_scale.
        The output of bn_training_update_grad.
        source data type, support "float32".
    diff_offset: dict
        dict of diff_offset, A 5HD Tensor for input diff_offset.
        The output of bn_training_update_grad.
        source data type, support "float32".
    scale: dict
        dict of scale, A 5HD Tensor for input scale.
        source data type, support "float32".
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
        source data type, support "float32".
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
        source data type, support "float32".
    y: dict
        dict of output, A `Tensor`. Has the same type as `grads`.
    epsilon: float
        A small float number added to the variance of x.
    before_split_ori_shape: list_list_int
        ori_shape for input list, only valid in ffts, default is None.
    before_split_ori_format: list_int
        ori_format for input list, only valid in ffts, default is None.
    kernel_name: str
        kernel name, default value is "bn_training_reduce_grad"

    Returns
    -------
    None
    """
    grads_dtype = grads.get("dtype").lower()
    x_dtype = x.get("dtype").lower()
    diff_scale_dtype = diff_scale.get("dtype").lower()
    diff_offset_dtype = diff_offset.get("dtype").lower()
    scale_dtype = scale.get("dtype").lower()
    batch_mean_dtype = batch_mean.get("dtype").lower()
    batch_variance_dtype = batch_variance.get("dtype").lower()

    para_check.check_dtype(grads_dtype, ("float32", "float16", "bfloat16"), param_name="grads")
    para_check.check_dtype(x_dtype, ("float32", "float16", "bfloat16"), param_name="x")
    para_check.check_dtype(diff_scale_dtype, ("float32",), param_name="diff_scale")
    para_check.check_dtype(diff_offset_dtype, ("float32",), param_name="diff_offset")
    para_check.check_dtype(scale_dtype, ("float32",), param_name="scale")
    para_check.check_dtype(batch_mean_dtype, ("float32",), param_name="batch_mean")
    para_check.check_dtype(batch_variance_dtype, ("float32",), param_name="batch_variance")

    ori_format = grads.get("ori_format").upper()
    data_format = grads.get("format").upper()
    _check_format(data_format, ori_format)

    if is_unknown_rank_input((grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance)) or epsilon is None:
        if data_format == "NC1HWC0":
            grads["shape"] = [-1, -1, -1, -1, 16]
            grads["range"] = [(1, None), (1, None), (1, None), (1, None), (16, 16)]
            x["shape"] = [-1, -1, -1, -1, 16]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (16, 16)]
            dynamic_shape = [1, -1, 1, 1, 16]
            dynamic_range = [(1, 1), (1, None), (1, 1), (1, 1), (16, 16)]
        elif data_format in ["NCHW", "NHWC"]:
            dynamic_shape = [-1, -1, -1, -1]
            dynamic_range = [(1, None), (1, None), (1, None), (1, None)]
            grads["shape"] = dynamic_shape
            grads["range"] = dynamic_range
            x["shape"] = dynamic_shape
            x["range"] = dynamic_range
        elif data_format in ["NCDHW", ]:
            dynamic_shape = [-1, -1, -1, -1, -1]
            dynamic_range = [(1, None), (1, None), (1, None), (1, None), (1, None)]
            grads["shape"] = dynamic_shape
            grads["range"] = dynamic_range
            x["shape"] = dynamic_shape
            x["range"] = dynamic_range
        else:
            grads["shape"] = [-1, -1, -1, -1, -1, 16]
            grads["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None), (16, 16)]
            x["shape"] = [-1, -1, -1, -1, -1, 16]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None), (16, 16)]
            dynamic_shape = [1, 1, -1, 1, 1, 16]
            dynamic_range = [(1, 1), (1, 1), (1, None), (1, 1), (1, 1), (16, 16)]

        for input_dict in (diff_scale, diff_offset, scale, batch_mean, batch_variance):
            input_dict["shape"] = dynamic_shape
            input_dict["range"] = dynamic_range

    if data_format in ("NCHW", "NHWC", "NCDHW"):
        shape_grads = grads.get("shape")
        shape_list = [1, 1, 1, 1]
        if data_format == "NCHW":
            shape_list[1] = shape_grads[1]
        elif data_format == "NHWC":
            shape_list[3] = shape_grads[3]
        else:
            shape_list = [1, 1, 1, 1, 1]
            shape_list[1] = shape_grads[1]
        range_list = util_common.gen_range(shape_list)
        diff_scale["shape"] = shape_list
        diff_scale["range"] = range_list
        diff_offset["shape"] = shape_list
        diff_offset["range"] = range_list
        scale["shape"] = shape_list
        scale["range"] = range_list
        batch_mean["shape"] = shape_list
        batch_mean["range"] = range_list
        batch_variance["shape"] = shape_list
        batch_variance["range"] = range_list

    tail_clean_info = get_tail_clean_info([grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance])
    extra_params = {"same_input_shape_group": [[0, 1], [2, 3, 4, 5, 6]]}
    ins = classify([grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance],
                   OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []

    for (_grads, _x, _diff_scale, _diff_offset, _scale, _batch_mean, _batch_variance) in ins:
        with tbe.compute():
            _shape_grads, _, _, _, _shape_scale, _, _ = shape_util.variable_shape(
                [_grads, _x, _diff_scale, _diff_offset, _scale, _batch_mean, _batch_variance])

            grads_input = tvm.placeholder(_shape_grads, name="grads_input", dtype=grads_dtype)
            x_input = tvm.placeholder(_shape_grads, name="x_input", dtype=x_dtype)
            diff_scale_input = tvm.placeholder(_shape_scale,
                                               name="diff_scale_input",
                                               dtype=diff_scale_dtype,
                                               attrs=tail_clean_info)
            diff_offset_input = tvm.placeholder(_shape_scale,
                                                name="diff_offset_input",
                                                dtype=diff_offset_dtype,
                                                attrs=tail_clean_info)
            scale_input = tvm.placeholder(_shape_scale, name="scale_input", dtype=scale_dtype, attrs=tail_clean_info)
            batch_mean_input = tvm.placeholder(_shape_scale,
                                               name="batch_mean_input",
                                               dtype=batch_mean_dtype,
                                               attrs=tail_clean_info)
            batch_variance_input = tvm.placeholder(_shape_scale,
                                                   name="batch_variance_input",
                                                   dtype=batch_variance_dtype,
                                                   attrs=tail_clean_info)

            res = bn_training_reduce_grad_compute(grads_input,
                                                  x_input,
                                                  diff_scale_input,
                                                  diff_offset_input,
                                                  scale_input,
                                                  batch_mean_input,
                                                  batch_variance_input,
                                                  y,
                                                  epsilon,
                                                  before_split_ori_shape,
                                                  before_split_ori_format,
                                                  kernel_name=kernel_name)

            tensor_list = [
                grads_input, x_input, diff_scale_input, diff_offset_input, scale_input, batch_mean_input,
                batch_variance_input, res
            ]
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
            schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
