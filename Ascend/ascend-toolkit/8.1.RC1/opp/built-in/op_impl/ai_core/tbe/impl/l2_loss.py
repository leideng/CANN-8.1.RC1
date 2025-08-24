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
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_common
from impl.util import util_select_op_base


# 'pylint: disable=invalid-name,unused-argument,unused-variable,too-many-locals,too-many-statements
def op_select_format(x, y, kernel_name="l2_loss"):
    """
    select format dynamically
    """
    input_ori_shape = x.get("ori_shape")
    input_ori_format = x.get("ori_format")
    input_ori_shape = shape_util.scalar2tensor_one(input_ori_shape)
    align_len = 16
    # charge whether support 5HD 5HD
    hd_support_format = \
        util_common.get_fused_format_str(["N", "C", "H", "W"]) + \
        util_common.get_fused_format_str(["N", "D", "C", "H", "W"])
    # current support all foramt
    is_support_hd = True
    # current support all foramt
    is_support_fz = True
    if len(input_ori_format) == len(input_ori_shape) and input_ori_format in hd_support_format:
        is_shape_c_align = input_ori_shape[input_ori_format.index("C")] % align_len == 0
        is_shape_n_align = input_ori_shape[input_ori_format.index("N")] % align_len == 0
        if is_shape_c_align:
            is_support_hd = True
        if is_shape_n_align and is_shape_c_align and util_common.is_support_fractal_z_input(x):
            is_support_fz = True
    # charge whether support FRACTAL_NZ
    # current support all foramt
    is_support_nz = True
    if len(input_ori_shape) >= 2:
        is_neg_one_dim_align = input_ori_shape[-1] % align_len == 0
        is_neg_two_dim_align = input_ori_shape[-2] % align_len == 0
        if is_neg_one_dim_align and is_neg_two_dim_align:
            is_support_nz = True

    base_data_type = ["float", "float16"]
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        base_data_type.remove("float")

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
    if tbe_platform.api_check_support("tik.set_atomic_add") \
            and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        dtype_base_out = ["float"] * len(dtype_base_out)
    if util_common.is_dynamic_input(x):
        dtype_base_out = dtype_base_in
    dtype_in_str = ','.join(dtype_base_in)
    dtype_out_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)
    nd_format_str = ','.join(["ND"] * len(dtype_base_out))
    input0 = util_select_op_base.gen_param(
        classify="input0", name="x", datatype=dtype_in_str,
        format=format_str, unknownshape_format=format_str)
    output0 = util_select_op_base.gen_param(
        classify="output0", name="y", datatype=dtype_out_str,
        format=nd_format_str, unknownshape_format=nd_format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@register_operator_compute("l2_loss", op_mode="static", support_fusion=True)
def l2_loss_compute(x, y, kernel_name="l2_loss"):
    """
     Reduce a tensor on a certain axis, and scale output with coeff

     Parameters
     ----------
     shape : shape of data
     dtype : source data type, only support float16, float32
     kernel_name : kernel name, default value is "l2_loss"

     Returns
     -------
     res: TVM tensor
        the result of compute
     """
    coeff_dtype = x.dtype
    _, axis = shape_util.simplify_axis_shape(x.shape, range(len(x.shape)))
    if x.dtype == "float16" and tbe_platform.api_check_support("tik.set_atomic_add")\
            and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        x = tbe.cast_to(x, "float32")
        coeff_dtype = "float32"

    coeff_sqrt = tvm.const(1.0 / (2 ** (0.5)), dtype=coeff_dtype)

    data_mul = tbe.vmuls(x, coeff_sqrt)
    data_sqr = tbe.vmul(data_mul, data_mul)
    res = tbe.sum(data_sqr, axis)

    return res


# 'pylint: disable=invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def l2_loss(x, y, kernel_name="l2_loss"):
    """
    Reduce a tensor on a certain axis, and scale output with coeff

    Parameters
    ----------
    shape : shape of data
    dtype : source data type, only support float16, float32
    kernel_name : kernel name, default value is "l2_loss"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")

    para_check.check_shape(shape, param_name="x")

    check_list = ("float16", "float32")
    if not dtype.lower() in check_list:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, "x", \
                                                                 "float16,float32", dtype.lower())

    shape, axis = shape_util.simplify_axis_shape(shape, range(len(shape)))

    inp_dtype = dtype.lower()
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)

    res = l2_loss_compute(data_input, y, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    build(sch, config)
