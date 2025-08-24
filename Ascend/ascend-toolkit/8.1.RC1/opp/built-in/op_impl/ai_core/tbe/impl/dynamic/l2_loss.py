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
l2_loss
"""
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.reduce_pattern_adapter import ReducePattern
from impl.util.platform_adapter import register_operator_compute
import tbe.dsl as dsl


# 'pylint: disable=invalid-name,unused-argument,unused-variable,too-many-locals,too-many-statements
def op_select_format(x, y, kernel_name="l2_loss"):
    """
    select format dynamically
    op_select_format support desc:
    1. current support all format
        ND/C1HWNCoC0/NC1HWC0/FRACTAL_NZ/FRACTAL_Z/NDC1HWC0/FRACTAL_Z_3D -> ND
    2. supported fp16 -> fp16 when tik.set_atomic_add support dtype of float16,
       otherwise support fp16 -> fp32
    """
    input_ori_shape = x.get("ori_shape")
    input_ori_shape = shape_util.scalar2tensor_one(input_ori_shape)

    # current support all foramt
    is_support_hd = True
    is_support_fz = True
    is_support_nz = True

    base_data_type = ["float", "float16"]
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    if not tbe_platform.api_check_support("tik.vadd", "float32"):
        base_data_type.remove("float")
    elif bfp16_support:
        base_data_type.append("bfloat16")

    dtype_base_out = list(base_data_type)
    format_base_out = ["ND"] * len(base_data_type) + ["C1HWNCoC0"] * len(base_data_type)
    dtype_base_out = dtype_base_out + base_data_type
    if is_support_hd:
        other_format = "NC1HWC0" if len(input_ori_shape) != 5 else "NDC1HWC0"
        dtype_base_out = dtype_base_out + base_data_type
        format_base_out = format_base_out + [other_format] * len(base_data_type)
    if is_support_nz:
        other_format = "FRACTAL_NZ"
        dtype_base_out = dtype_base_out + base_data_type
        format_base_out = format_base_out + [other_format] * len(base_data_type)
    if is_support_fz:
        other_format = "FRACTAL_Z" if len(input_ori_shape) != 5 else "FRACTAL_Z_3D"
        dtype_base_out = dtype_base_out + base_data_type
        format_base_out = format_base_out + [other_format] * len(base_data_type)

    dtype_base_in = list(dtype_base_out)

    is_atomic_fp16 = tbe_platform.api_check_support("tik.set_atomic_add", "float16")
    is_atomic_fp32 = tbe_platform.api_check_support("tik.set_atomic_add", "float32")

    if not bfp16_support and is_atomic_fp32 and not is_atomic_fp16:
        dtype_base_out = ["float"] * len(dtype_base_out)
    elif bfp16_support and is_atomic_fp32 and not is_atomic_fp16:
        dtype_base_out = ["float", "bfloat16"] * len(dtype_base_out)

    dtype_in_str = ','.join(dtype_base_in)
    dtype_out_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)
    nd_format_str = ','.join(["ND"] * len(dtype_base_out))
    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype=dtype_in_str,
                                           format=format_str,
                                           unknownshape_format=format_str)
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype=dtype_out_str,
                                            format=nd_format_str,
                                            unknownshape_format=nd_format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("L2Loss", op_mode="dynamic", support_fusion=False, support_bfp16=True)
def l2_loss_compute(x, axes, y, kernel_name="l2_loss"):
    """
    l2_loss compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    axes: int, list, tuple or NONETYPE
        the axes for reduce.
    y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "l2_loss".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same type as input tensor.
    """
    coeff_dtype = x.dtype
    y_dtype = y.get("dtype")
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    if cce_product not in ("Ascend310P",) and tbe_platform.api_check_support("tik.set_atomic_add") \
        and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        coeff_dtype = "float32"
    else:
        coeff_dtype = y_dtype


    if x.dtype != coeff_dtype:
        x = tbe.cast_to(x, coeff_dtype)

    coeff_sqrt = tvm.const(1.0 / (2**0.5), dtype=coeff_dtype)
    data_mul = tbe.vmuls(x, coeff_sqrt)
    data_sqr = tbe.vmul(data_mul, data_mul)

    res = tbe.reduce_sum(data_sqr, axis=axes)
    if y_dtype == "float16":
        res = tbe.cast_to(res, y_dtype)
    elif y_dtype == "bfloat16":
        res = dsl.round(res, y_dtype)

    return res


# 'pylint: disable=too-many-locals,invalid-name
@register_operator("L2Loss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def l2_loss(x, y, kernel_name="l2_loss"):
    """
    reduce a tensor on a certain axes.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "l2_loss".

    Returns
    -------
    None
    """

    dtype_x = x.get("dtype")
    dtype_lower_x = dtype_x.lower()
    check_list_x = ("bfloat16", "float16", "float32")
    para_check.check_dtype(dtype_lower_x, check_list_x, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    # gen reduce axis input dict
    input_axis = {"shape": [-1], "value": [], "rel_pos_to_reduce": "axis"}

    # gen extra_params for reduce pattern
    extra_params = dict()
    # set KEEP_DIMS flag
    extra_params.update(ReducePattern.KEEP_DIMS_FALSE)
    # set all reduce pattern
    extra_params.update(ReducePattern.REDUCE_MODE_REDUCE_ALL)
    schedules = []
    tensors = []
    ins = classify([x, input_axis], OpPatternMode.REDUCE, extra_params)
    for (_x, axes) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x, axes], op_mode="reduce")[0]
            data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_lower_x)

            res = l2_loss_compute(data_input_x, axes.get("value"), y)
            tensors.append([data_input_x, res])
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
