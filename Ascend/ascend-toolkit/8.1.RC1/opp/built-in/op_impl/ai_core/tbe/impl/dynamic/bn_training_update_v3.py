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
dynamic bn_training_update_v3
"""
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_compute import only_static_support
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls
from impl.dynamic.bn_training_update import is_ffts_for_bn
from impl.dynamic.bn_training_update import trans_format_for_bn
from impl.dynamic.bn_training_update import trans_shape_for_bn


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-arguments,too-many-locals,redefined-builtin
def op_select_format(x,
                     sum,
                     square_sum,
                     scale,
                     offset,
                     y,
                     batch_mean,
                     batch_variance,
                     reserve_1,
                     reserve_2,
                     epsilon,
                     before_split_ori_shape=None,
                     before_split_ori_format=None,
                     kernel_name="bn_training_update_v3"):
    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype="float16,float,float16,float,float16,float,float16,float,\
                                                     bfloat16,bfloat16,bfloat16,bfloat16",
                                           format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                   NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                           unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                NCHW,NHWC,NC1HWC0,NDC1HWC0")
    input1 = util_select_op_base.gen_param(classify="input1",
                                           name="sum",
                                           datatype="float,float,float,float,float,float,float,float,\
                                                     float,float,float,float",
                                           format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                   NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                           unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                NCHW,NHWC,NC1HWC0,NDC1HWC0")
    input2 = util_select_op_base.gen_param(classify="input2",
                                           name="square_sum",
                                           datatype="float,float,float,float,float,float,float,float,\
                                                     float,float,float,float",
                                           format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                   NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                           unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                NCHW,NHWC,NC1HWC0,NDC1HWC0")
    input3 = util_select_op_base.gen_param(classify="input3",
                                           name="scale",
                                           datatype="float,float,float,float,float,float,float,float,\
                                                     float,float,float,float",
                                           format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                   NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                           unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                NCHW,NHWC,NC1HWC0,NDC1HWC0")
    input4 = util_select_op_base.gen_param(classify="input4",
                                           name="offset",
                                           datatype="float,float,float,float,float,float,float,float,\
                                                     float,float,float,float",
                                           format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                   NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                           unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                NCHW,NHWC,NC1HWC0,NDC1HWC0")
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype="float16,float,float16,float,float16,float,float16,float,\
                                                      bfloat16,bfloat16,bfloat16,bfloat16",
                                            format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                    NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                            unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                 NCHW,NHWC,NC1HWC0,NDC1HWC0")
    output1 = util_select_op_base.gen_param(classify="output1",
                                            name="batch_mean",
                                            datatype="float,float,float,float,float,float,float,float,\
                                                      float,float,float,float",
                                            format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                    NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                            unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                 NCHW,NHWC,NC1HWC0,NDC1HWC0")
    output2 = util_select_op_base.gen_param(classify="output2",
                                            name="batch_variance",
                                            datatype="float,float,float,float,float,float,float,float,\
                                                      float,float,float,float",
                                            format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                    NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                            unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                 NCHW,NHWC,NC1HWC0,NDC1HWC0")
    output3 = util_select_op_base.gen_param(classify="output3",
                                            name="reserve_1",
                                            datatype="float,float,float,float,float,float,float,float,\
                                                      float,float,float,float",
                                            format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                    NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                            unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                 NCHW,NHWC,NC1HWC0,NDC1HWC0")
    output4 = util_select_op_base.gen_param(classify="output4",
                                            name="reserve_2",
                                            datatype="float,float,float,float,float,float,float,float,\
                                                      float,float,float,float",
                                            format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                    NCHW,NHWC,NC1HWC0,NDC1HWC0",
                                            unknownshape_format="NCHW,NCHW,NHWC,NHWC,NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,\
                                                                 NCHW,NHWC,NC1HWC0,NDC1HWC0")
    param_list = [input0, input1, input2, input3, input4,
                  output0, output1, output2, output3, output4]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-arguments,too-many-locals,redefined-builtin
def check_special_soc():
    if tbe_platform.api_check_support("tik.vcopy"):
        return True
    else:
        return False


def check_supported(x,
                    sum,
                    square_sum,
                    scale,
                    offset,
                    y,
                    batch_mean,
                    batch_variance,
                    reserve_1,
                    reserve_2,
                    epsilon,
                    before_split_ori_shape=None,
                    before_split_ori_format=None,
                    kernel_name="bn_training_update_v3"):
    if is_unknown_rank_input((x, sum, square_sum, scale, offset)) or epsilon is None:
        return True, ""

    if util_common.is_unknown([x, sum, square_sum, scale, offset]):
        return True, ""

    # static shape
    if check_special_soc():
        return True, ""
    else:
        return False, ""


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


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
@register_operator_compute("BNTrainingUpdateV3", op_mode="dynamic", support_fusion=only_static_support,
                           support_bfp16=True)
def bn_training_update_v3_compute(x,
                                  sum,
                                  square_sum,
                                  scale,
                                  offset,
                                  y,
                                  batch_mean,
                                  batch_variance,
                                  reserve_1,
                                  reserve_2,
                                  epsilon,
                                  before_split_ori_shape=None,
                                  before_split_ori_format=None,
                                  kernel_name="bn_training_update_v3",
                                  reduce_shape=None):
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
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"
    reduce_shape: list
        reduce shape of input shape

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v3 compute
    """
    epsilon = get_attr_by_cls(epsilon, OpAttr(0, "epsilon", "Float", 0.0000001), "float32")

    if not util_common.is_unknown([y]):
        # charge ffts
        if is_ffts_for_bn(before_split_ori_shape, before_split_ori_format):
            data_format = trans_format_for_bn(before_split_ori_format[0])
            shape_y = trans_shape_for_bn(before_split_ori_shape[0], data_format)
        else:
            data_format = y.get("format").upper()
            shape_y = y.get("shape")

        if data_format in ("NHWC",):
            reduce_dims = [shape_y[0], shape_y[1], shape_y[2]]
        elif data_format in ("NC1HWC0", "NCHW"):
            reduce_dims = [shape_y[0], shape_y[2], shape_y[3]]
        elif data_format in ("NDC1HWC0",):
            reduce_dims = [shape_y[0], shape_y[1], shape_y[3], shape_y[4]]
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
    multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)
    multiplier = tbe.broadcast(multiplier_div, x.shape)

    addend_mul = tbe.vmul(multiplier_div, save_mean_reduce)
    addend_sub = tbe.vsub(offset, addend_mul)
    addend = tbe.broadcast(addend_sub, x.shape)

    # compute the batch normalization of x
    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")
        res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)
        res_y = tbe.cast_to(res_y, "float16")
    else:
        res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)

    # compute batch_mean and batch_var
    res_batch_mean = tbe.vmuls(sum, num_rec)
    res_batch_var = tbe.vmuls(save_variance_reduce, batch_var_scalar)

    res = [res_y, res_batch_mean, res_batch_var, save_mean_reduce, save_variance_reduce]

    return res


@register_operator("BNTrainingUpdateV3")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_LIST_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.KERNEL_NAME)
def bn_training_update_v3(x,
                          sum,
                          square_sum,
                          scale,
                          offset,
                          y,
                          batch_mean,
                          batch_variance,
                          reserve_1,
                          reserve_2,
                          epsilon,
                          before_split_ori_shape=None,
                          before_split_ori_format=None,
                          kernel_name="bn_training_update_v3"):
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
        dict of scale, A 5HD Tensor for mean.
    offset: dict
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    dtype_sum = sum.get("dtype").lower()
    dtype_sqrsum = square_sum.get("dtype").lower()
    dtype_scale = scale.get("dtype").lower()
    dtype_offset = offset.get("dtype").lower()

    data_format = x.get("format").upper()
    # handle -2
    if is_unknown_rank_input((x, sum, square_sum, scale, offset)) or epsilon is None:
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
        for input_dict in (sum, square_sum, scale, offset):
            input_dict["shape"] = dynamic_shape
            input_dict["range"] = dynamic_range

    shape_x = x.get("shape")
    reduce_shape = None
    dyn_flag = util_common.is_unknown([x, sum, square_sum, scale, offset])

    for _ins in (sum, square_sum, scale, offset, batch_mean, batch_variance, reserve_1, reserve_2):
        if len(_ins.get("shape")) == 1:
            c_dim = _ins.get("shape")[0]
            if data_format in ("NCHW",):
                _ins["shape"] = [1, c_dim, 1, 1]
            elif data_format in ("NHWC",):
                _ins["shape"] = [1, 1, 1, c_dim]
    # support fuzzy compile
    if dyn_flag:
        for _ins in (sum, square_sum, scale, offset):
            simplify_shape(_ins, data_format)
    # handle static shape
    if not dyn_flag and data_format in ("NHWC", ):
        reduce_shape = [shape_x[0], shape_x[1], shape_x[2]]
    elif not dyn_flag and data_format in ("NC1HWC0", "NCHW"):
        reduce_shape = [shape_x[0], shape_x[2], shape_x[3]]
    elif not dyn_flag and data_format in ("NDC1HWC0",):
        reduce_shape = [shape_x[0], shape_x[1], shape_x[3], shape_x[4]]

    ins_list = [x, sum, square_sum, scale, offset]
    ins = classify(ins_list, OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedules = []
    tensors = []

    for (ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset) in ins:
        with tbe.compute():
            _shape_x, _shape_sum, _shape_sqrsum, _shape_scale, _shape_offset = shape_util.variable_shape(
                [ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset])

            in_x = tvm.placeholder(_shape_x, name="x", dtype=dtype_x)
            in_sum = tvm.placeholder(_shape_sum, name="sum", dtype=dtype_sum)
            in_sqrsum = tvm.placeholder(_shape_sqrsum, name="sqrsum", dtype=dtype_sum)
            in_scale = tvm.placeholder(_shape_scale, name="scale", dtype=dtype_sum)
            in_offset = tvm.placeholder(_shape_offset, name="offset", dtype=dtype_sum)
            res = bn_training_update_v3_compute(in_x,
                                                in_sum,
                                                in_sqrsum,
                                                in_scale,
                                                in_offset,
                                                y,
                                                batch_mean,
                                                batch_variance,
                                                reserve_1,
                                                reserve_2,
                                                epsilon,
                                                before_split_ori_shape,
                                                before_split_ori_format,
                                                kernel_name=kernel_name,
                                                reduce_shape=reduce_shape)
            tensors.append([in_x, in_sum, in_sqrsum, in_scale, in_offset] + res)

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
