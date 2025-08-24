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
dynamic bn_training_update_v2
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
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_compute import only_static_support
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-arguments
def op_select_format(x,
                     sum,
                     square_sum,
                     scale,
                     offset,
                     y,
                     batch_mean,
                     batch_variance,
                     epsilon,
                     kernel_name="bn_training_update_v2"):
    """
    1. when input(x)'s ori_shape is [1, ? ,1, ?] and the format is NCHW
    the Op BNTrainingUpdateV2 can support NCHW.
    > for example:
    > x : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > sum : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > square_sum : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > scale : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > offset : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > the Op BNTrainingUpdateV2 can process with NC1HWC0:
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > sum : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > square_sum : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > scale : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > offset : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    """
    origin_format = x.get("ori_format").upper()
    origin_shape = x.get("ori_shape")

    if origin_format == "NCHW" and len(origin_shape) == 4 and origin_shape[0] == 1 and origin_shape[2] == 1:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x",
                                               datatype="float16,float,float16,float,bfloat16,bfloat16",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NCHW,NC1HWC0")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="sum",
                                               datatype="float,float,float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NCHW,NC1HWC0")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="square_sum",
                                               datatype="float,float,float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NCHW,NC1HWC0")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="scale",
                                               datatype="float,float,float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NCHW,NC1HWC0")
        input4 = util_select_op_base.gen_param(classify="input4",
                                               name="offset",
                                               datatype="float,float,float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NCHW,NC1HWC0")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype="float16,float,float16,float,bfloat16,bfloat16",
                                                format="NCHW,NCHW,NC1HWC0,NC1HWC0,NCHW,NC1HWC0")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="batch_mean",
                                                datatype="float,float,float,float,float,float",
                                                format="NCHW,NCHW,NC1HWC0,NC1HWC0,NCHW,NC1HWC0")
        output2 = util_select_op_base.gen_param(classify="output2",
                                                name="batch_variance",
                                                datatype="float,float,float,float,float,float",
                                                format="NCHW,NCHW,NC1HWC0,NC1HWC0,NCHW,NC1HWC0")
    else:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x",
                                               datatype="float16,float,float16,float,bfloat16,bfloat16, \
                                                float16,float,bfloat16,float16,float,bfloat16",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NDC1HWC0, \
                                                NCDHW,NCDHW,NCDHW,NCHW,NCHW,NCHW")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="sum",
                                               datatype="float,float,float,float,float,float, \
                                                float,float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NDC1HWC0, \
                                                NCDHW,NCDHW,NCDHW,NCHW,NCHW,NCHW")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="square_sum",
                                               datatype="float,float,float,float,float,float, \
                                                float,float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NDC1HWC0, \
                                                NCDHW,NCDHW,NCDHW,NCHW,NCHW,NCHW")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="scale",
                                               datatype="float,float,float,float,float,float, \
                                                float,float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NDC1HWC0, \
                                                NCDHW,NCDHW,NCDHW,NCHW,NCHW,NCHW")
        input4 = util_select_op_base.gen_param(classify="input4",
                                               name="offset",
                                               datatype="float,float,float,float,float,float, \
                                                float,float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NDC1HWC0, \
                                                NCDHW,NCDHW,NCDHW,NCHW,NCHW,NCHW")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype="float16,float,float16,float,bfloat16,bfloat16, \
                                                    float16,float,bfloat16,float16,float,bfloat16",
                                                format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NDC1HWC0, \
                                                    NCDHW,NCDHW,NCDHW,NCHW,NCHW,NCHW")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="batch_mean",
                                                datatype="float,float,float,float,float,float, \
                                                    float,float,float,float,float,float",
                                                format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NDC1HWC0, \
                                                    NCDHW,NCDHW,NCDHW,NCHW,NCHW,NCHW")
        output2 = util_select_op_base.gen_param(classify="output2",
                                                name="batch_variance",
                                                datatype="float,float,float,float,float,float, \
                                                    float,float,float,float,float,float",
                                                format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NDC1HWC0, \
                                                    NCDHW,NCDHW,NCDHW,NCHW,NCHW,NCHW")

    param_list = [input0, input1, input2, input3, input4, output0, output1, output2]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


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
    elif data_format in ("NCDHW",):
        dim_c = ins.get("shape")[1]
        ins["shape"] = [1, dim_c, 1, 1, 1]
        c_range = (1, None) if dim_c == -1 else (dim_c, dim_c)
        ins["range"] = [(1, 1), c_range, (1, 1), (1, 1), (1, 1)]
    else:
        dim_c1 = ins.get("shape")[2]
        ins["shape"] = [1, 1, dim_c1, 1, 1, 16]
        c1_range = (1, None) if dim_c1 == -1 else (dim_c1, dim_c1)
        c0_range = (16, 16)
        ins["range"] = [(1, 1), (1, 1), c1_range, (1, 1), (1, 1), c0_range]


def _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale, dtype_offset):
    """check input dtype"""
    para_check.check_dtype(dtype_x, ("float16", "float32", "bfloat16"))
    para_check.check_dtype(dtype_sum, ("float32",))
    para_check.check_dtype(dtype_square_sum, ("float32",))
    para_check.check_dtype(dtype_scale, ("float32",))
    para_check.check_dtype(dtype_offset, ("float32",))


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
@register_operator_compute("BNTrainingUpdateV2", op_mode="dynamic", support_fusion=only_static_support,
                           support_bfp16=True)
def bn_training_update_v2_compute(x,
                                  sum,
                                  square_sum,
                                  scale,
                                  offset,
                                  y,
                                  batch_mean,
                                  batch_variance,
                                  epsilon,
                                  kernel_name="bn_training_update_v2"):
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
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v2 compute
    """
    shape_x = shape_util.shape_to_list(x.shape)

    if not util_common.is_unknown([y]):
        data_format = y.get("format").upper()
        shape_y = y.get("shape")
        if data_format in ("NC1HWC0", "NCHW"):
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
    else:
        num_rec = tbe.var("num_rec", dtype="float32")

    epsilon = get_attr_by_cls(epsilon,
                              OpAttr(0, "epsilon", "Float", 0.0000001),
                              "float32")

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
    multiplier = tbe.broadcast(multiplier_div, shape_x)

    addend_mul = tbe.vmul(multiplier_div, save_mean_reduce)
    addend_sub = tbe.vsub(offset, addend_mul)
    addend = tbe.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")
        res_y = tbe.vmla(multiplier, x, addend)
        res_y = tbe.cast_to(res_y, "float16")
    else:
        res_y = tbe.vmla(multiplier, x, addend)

    res = [res_y, save_mean_reduce, save_variance_reduce]

    return res


@register_operator("BNTrainingUpdateV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def bn_training_update_v2(x,
                          sum,
                          square_sum,
                          scale,
                          offset,
                          y,
                          batch_mean,
                          batch_variance,
                          epsilon,
                          kernel_name="bn_training_update_v2"):
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
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    dtype_sum = sum.get("dtype").lower()
    dtype_sqrsum = square_sum.get("dtype").lower()
    dtype_scale = scale.get("dtype").lower()
    dtype_offset = offset.get("dtype").lower()
    _check_dtype(dtype_x, dtype_sum, dtype_sqrsum, dtype_scale, dtype_offset)

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
        elif data_format == "NCDHW":
            x["shape"] = [-1, -1, -1, -1, -1]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None)]
            dynamic_shape = [1, -1, 1, 1, 1]
            dynamic_range = [(1, 1), (1, None), (1, 1), (1, 1), (1, 1)]
        else:
            x["shape"] = [-1, -1, -1, -1, -1, 16]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None), (16, 16)]
            dynamic_shape = [1, 1, -1, 1, 1, 16]
            dynamic_range = [(1, 1), (1, 1), (1, None), (1, 1), (1, 1), (16, 16)]
        for input_dict in (sum, square_sum, scale, offset):
            input_dict["shape"] = dynamic_shape
            input_dict["range"] = dynamic_range

    dyn_flag = util_common.is_unknown([x, sum, square_sum, scale, offset])

    for _ins in (sum, square_sum, scale, offset):
        if len(_ins.get("shape")) == 1:
            c_dim = _ins.get("shape")[0]
            if data_format in ("NCHW",):
                _ins["shape"] = [1, c_dim, 1, 1]
            elif data_format in ("NHWC",):
                _ins["shape"] = [1, 1, 1, c_dim]
            elif data_format in ("NCDHW",):
                _ins["shape"] = [1, c_dim, 1, 1, 1]

    # support fuzzy compile
    if dyn_flag:
        for _ins in (sum, square_sum, scale, offset):
            simplify_shape(_ins, data_format)

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
            in_sqrsum = tvm.placeholder(_shape_sqrsum, name="sqrsum", dtype=dtype_sqrsum)
            in_scale = tvm.placeholder(_shape_scale, name="scale", dtype=dtype_scale)
            in_offset = tvm.placeholder(_shape_offset, name="offset", dtype=dtype_offset)
            res = bn_training_update_v2_compute(in_x,
                                                in_sum,
                                                in_sqrsum,
                                                in_scale,
                                                in_offset,
                                                y,
                                                batch_mean,
                                                batch_variance,
                                                epsilon,
                                                kernel_name=kernel_name)
            tensors.append([in_x, in_sum, in_sqrsum, in_scale, in_offset] + res)

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
