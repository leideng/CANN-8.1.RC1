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
dynamic softmaxgrad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.norm_pattern_adapter import NormPattern
from impl.util import util_common
from impl.util import util_frac_z as fz
from impl.util import util_select_op_base


def check_is_dynamic(in_tensor):
    """
    check_is_dynamic
    """
    in_shape = in_tensor["shape"]
    ori_shape = in_tensor["ori_shape"]
    for idx, _ in enumerate(in_shape):
        if in_shape[idx] == -1:
            return True
    for idx, _ in enumerate(ori_shape):
        if ori_shape[idx] == -1:
            return True
    return False


# 'pylint: disable=unused-argument
def op_select_format(softmax, grad_softmax, grad_x, axes, kernel_name="softmax_grad"):
    """
    1.when the lengths of x's shape and y's shape are the same and equal to 2,
    the formats of x and y are the same and are one of [FRACTAL_NZ,NC1HWC0,ND].

        example:
        original:
        softmax's Tensor(shape=(16, 16, 16, 16, 16), "FRACTAL_NZ")
        grad_softmax's Tensor(shape=(16, 16, 16, 16, 16), "FRACTAL_NZ")
        grad_x's Tensor(shape=(16, 16, 16, 16, 16), "FRACTAL_NZ")
    """
    shape_x_ori = shape_util.scalar2tensor_one(softmax.get("ori_shape"))
    length_x_ori = len(shape_x_ori)
    if util_common.is_unknown([softmax, grad_softmax]):
        input0 = util_select_op_base.gen_param(classify="input0", name="softmax",
                                               datatype="bfloat16,float16,float32",
                                               format="ND,ND,ND",
                                               unknownshape_format="ND,ND,ND")
        input1 = util_select_op_base.gen_param(classify="input1", name="grad_softmax",
                                               datatype="bfloat16,float16,float32",
                                               format="ND,ND,ND",
                                               unknownshape_format="ND,ND,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="grad_x",
                                                datatype="bfloat16,float16,float32",
                                                format="ND,ND,ND",
                                                unknownshape_format="ND,ND,ND")
    else:
        if length_x_ori == 2 and shape_x_ori[0] == 4096 and shape_x_ori[1] == 4096:
            input0 = util_select_op_base.gen_param(classify="input0", name="softmax",
                                                   datatype="bfloat16,float16,float16,float",
                                                   format="ND, ND, ND, ND")
            input1 = util_select_op_base.gen_param(classify="input1", name="grad_softmax",
                                                   datatype="bfloat16,float16,float16,float",
                                                   format="ND, ND, ND, ND")
            output0 = util_select_op_base.gen_param(classify="output0", name="grad_x",
                                                    datatype="bfloat16,float,float16,float",
                                                    format="ND, ND, ND, ND")
        else:
            input0 = util_select_op_base.gen_param(classify="input0", name="softmax",
                                                   datatype="float16,float16,float16,float16,float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16",
                                                   format="NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ,NC1HWC0,ND,NDC1HWC0,"
                                                          "FRACTAL_NZ,NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ")
            input1 = util_select_op_base.gen_param(classify="input1", name="grad_softmax",
                                                   datatype="float16,float16,float16,float16,float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16",
                                                   format="NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ,NC1HWC0,ND,NDC1HWC0,"
                                                          "FRACTAL_NZ,NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ")
            output0 = util_select_op_base.gen_param(classify="output0", name="grad_x",
                                                    datatype="float16,float16,float16,float16,\
                                                            float,float,float,float,\
                                                            bfloat16,bfloat16,bfloat16,bfloat16",
                                                    format="NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ,\
                                                            NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ,\
                                                            NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ")

    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_is_axes_with_last(shape, axes):
    """
    check_is_axes_with_last
    """
    if len(axes) > 1:
        for i, _ in enumerate(axes):
            if axes[i] == len(shape) - 1:
                return True
    return False


# 'pylint: disable=locally-disabled,unused-argument,too-many-arguments
# 'pylint: disable=unused-variable,disable=too-many-lines,disable=too-many-locals
@register_operator_compute("SoftmaxGrad", op_mode="dynamic", support_fusion=False)
def softmax_grad_compute(softmax, grad_softmax, grad_x, axes, kernel_name="softmax_grad", impl_mode="high_precision"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: TVM tensor
        the placeholder of first input data
    grad_softmax: TVM tensor
        the placeholder of second input data
    grad_x: dict
        the dict of output data
    axes: int, list or tuple .
        the first axes to reduce, may be negative to index from the end
        (e.g., -1 for the last axes).
        axes may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        cce kernel name, default value is "softmax_grad"

    Returns
    -------
    res: TVM tensor
        the result of softmax_grad_compute
    """
    dtype = softmax.dtype
    shape = shape_util.shape_to_list(grad_softmax.shape)
    list_axis = list(axes)

    attributes = softmax.op.attrs
    disable_fuse_axes = attributes["disable_fuse_axes"]
    ori_shape = shape_util.shape_to_list(attributes["ori_shape"])
    ori_format = attributes["ori_format"]
    input_format = attributes["format"]
    has_improve_precision = False
    is_use_value = False

    if dtype == "bfloat16":
        grad_softmax = tbe.cast_to(grad_softmax, "float32")
        softmax = tbe.cast_to(softmax, "float32")

    if len(list_axis) == 2:
        if input_format in ("NC1HWC0", "NDC1HWC0"):
            is_use_value = True
            idc_list = shape_util.shape_to_list(disable_fuse_axes)
            idx_c0 = idc_list[1]
            ori_format = ori_format.upper()
            c = ori_shape[ori_format.find('C')]
            c = tbe.var('c') if c == -1 else c
            pad_c = tvm.floormod(c - 1, shape[idx_c0]) + 1
        if input_format in ("FRACTAL_NZ",):
            is_use_value = True
            idc_list = shape_util.shape_to_list(disable_fuse_axes)
            idx_c1 = idc_list[0]
            idx_c0 = idc_list[1]
            c = -1
            if (idx_c0 - idx_c1) == 2:
                c = ori_shape[-1]
            else:
                c = ori_shape[-2]
            c = tbe.var('c') if c == -1 else c
            pad_c = tvm.floormod(c - 1, shape[idx_c0]) + 1

    if is_use_value:
        softmax = tbe.set_value(softmax, lambda *i: tvm.all(i[list_axis[0]] > shape[list_axis[0]] - 2, \
                                                            i[list_axis[1]] > pad_c - 1), 0)

    if impl_mode == "high_performance" and dtype == "float16" and \
        tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        grad_softmax_fp32 = tbe.cast_to(grad_softmax, "float32")
        softmax_fp32 = tbe.cast_to(softmax, "float32")
        data_vmul = tbe.vmul(softmax_fp32, grad_softmax_fp32)
        data_sum = tbe.reduce_sum(data_vmul, axis=axes, keepdims=True)
        data_sum = tbe.cast_to(data_sum, "float16")

        if check_is_axes_with_last(shape, axes):
            tmp_shape = list(data_sum.shape[:-1]) + [shape[-1]]
            data_sum_tmp = tbe.broadcast(data_sum, tmp_shape)
            data_sum_tmp = tbe.broadcast(data_sum_tmp, shape)
        else:
            data_sum_tmp = tbe.broadcast(data_sum, shape)
        data_sub = tbe.vsub(grad_softmax, data_sum_tmp)
        res = tbe.vmul(softmax, data_sub)
    else:
        if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
            grad_softmax = tbe.cast_to(grad_softmax, "float32")
            softmax = tbe.cast_to(softmax, "float32")
            has_improve_precision = True
        data_vmul = tbe.vmul(softmax, grad_softmax)
        data_sum = tbe.reduce_sum(data_vmul, axis=axes, keepdims=True)

        if check_is_axes_with_last(shape, axes):
            tmp_shape = list(data_sum.shape[:-1]) + [shape[-1]]
            data_sum_tmp = tbe.broadcast(data_sum, tmp_shape)
            data_sum_tmp = tbe.broadcast(data_sum_tmp, shape)
        else:
            data_sum_tmp = tbe.broadcast(data_sum, shape)
        data_sub = tbe.vsub(grad_softmax, data_sum_tmp)
        res = tbe.vmul(softmax, data_sub)
        if has_improve_precision and dtype == "float16":
            res = tbe.cast_to(res, "float16")
        elif dtype == "bfloat16":
            res = tbe.round(res, "bfloat16")

    return res


def update_5hd_axis(origin_format, list_axis, input_format):
    """
    update the axes of 5hd format
    data using for compute and schedule
    """
    if hasattr(list_axis, 'index'):
        list_axis = list_axis[0]

    axis_str = origin_format[list_axis]
    offset_6hd = 1 if input_format == "NDC1HWC0" else 0

    dict_format_axis = {
        "N": [0, ],
        "C": [1 + offset_6hd, 4 + offset_6hd],
        "H": [2 + offset_6hd, ],
        "W": [3 + offset_6hd, ],
        "D": [1, ]
    }

    return dict_format_axis.get(axis_str)


def check_axis_is_int(axes):
    """
    check axes wherther int
    """
    if not isinstance(axes, int):
        axes = list(axes)
    return axes


# 'pylint:disable=too-many-locals,invalid-name,too-many-arguments
@register_operator("SoftmaxGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def softmax_grad(softmax, grad_softmax, grad_x, axes=-1, kernel_name="softmax_grad", impl_mode="high_precision"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: dict
        shape and dtype of first input, only support float16, float32, bfloat16
    grad_softmax: dict
        shape and dtype of second input, only support float16, float32, bfloat16
    grad_x: dict
        shape and dtype of output data, should be same shape and type as input
    axes: int, list or tuple .
        the first axes to reduce, may be negative to index from the end
        (e.g., -1 for the last axes).
        axes may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        kernel name, default value is "softmax_grad"

    Returns
    -------
    None
    """

    shape = softmax.get("shape")
    grad_shape = grad_softmax.get("shape")
    dtype = softmax.get("dtype").lower()
    input_format = softmax.get("format")
    ori_format = softmax.get("ori_format")
    ori_shape = softmax.get("ori_shape")

    para_check.check_shape(shape, param_name="softmax")
    para_check.check_shape(grad_shape, param_name="grad_softmax")
    para_check.check_dtype(dtype, ("bfloat16", "float16", "float32"), param_name="softmax")

    if input_format == "NC1HWC0":
        if len(ori_shape) == 2:
            new_ori_shape = [1, ori_shape[0], ori_shape[1], 1]
            softmax["ori_shape"] = new_ori_shape
            grad_softmax["ori_shape"] = new_ori_shape
            axes = check_axis_is_int(axes)
            if not hasattr(axes, 'index'):
                axes = axes + 1 if axes >= 0 else axes - 1
            else:
                axes[0] = axes[0] + 1 if axes[0] >= 0 else axes[0] - 1
        if len(ori_shape) == 3:
            new_ori_shape = [1, ori_shape[0], ori_shape[1], ori_shape[2]]
            softmax["ori_shape"] = new_ori_shape
            grad_softmax["ori_shape"] = new_ori_shape
            axes = check_axis_is_int(axes)
            if not hasattr(axes, 'index'):
                if axes >= 0:
                    axes = axes + 1
            else:
                if axes[0] >= 0:
                    axes[0] = axes[0] + 1
        ori_shape = softmax.get("ori_shape")

    extra_params = dict()
    if axes is None:
        # when axes is None, it is binary case, go unknown axes schedule
        list_axis = NormPattern.REDUCE_UNKNOWN_MODE
        extra_params.update(NormPattern.REDUCE_SINGLE_TYPE)
        operation.add_compile_info(NormPattern.REDUCE_ATTR_IDX, 0)
        operation.add_compile_info(NormPattern.REDUCE_ATTR_NAME, "axes")
        operation.add_compile_info(NormPattern.REDUCE_ATTR_DTYPE, "ListInt")
    elif not isinstance(axes, int):
        list_axis = list(axes)
    else:
        list_axis = [axes]

    if not util_common.is_unknown(softmax):
        if input_format in ("NC1HWC0", "NDC1HWC0"):
            list_axis = update_5hd_axis(ori_format, list_axis, input_format)

        if fz.is_frac_z(softmax):
            list_axis = fz.to_frac_z_axis(ori_shape, list_axis)

        if input_format in ("NC1HWC0", "NDC1HWC0", "FRACTAL_NZ") and len(list_axis) == 2:
            extra_params.update({"disable_fuse_axes": [list_axis[0], list_axis[1]]})

    tensors = []
    schedules = []
    ins = classify([softmax, grad_softmax, list_axis], OpPatternMode.NORM, extra_params)

    for idx, (x, grad, reduce_axis) in enumerate(ins):
        with tbe.compute():
            disable_fuse_axes = []
            if "disable_fuse_axes" in extra_params:
                disable_fuse_axes = extra_params.get("disable_fuse_axes")[idx]
            shape_var_new, grad_shape_var_new = shape_util.variable_shape([x, grad], op_mode="norm")
            softmax = tvm.placeholder(shape_var_new, dtype=dtype, name="softmax",
                                      attrs={"ori_shape": ori_shape, "ori_format": ori_format,
                                             "format": input_format, "disable_fuse_axes": disable_fuse_axes})
            grad_softmax = tvm.placeholder(grad_shape_var_new, dtype=dtype, name="grad_softmax")
            output = softmax_grad_compute(softmax, grad_softmax, grad_x, reduce_axis, kernel_name, impl_mode)
            tensors.append([softmax, grad_softmax, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
