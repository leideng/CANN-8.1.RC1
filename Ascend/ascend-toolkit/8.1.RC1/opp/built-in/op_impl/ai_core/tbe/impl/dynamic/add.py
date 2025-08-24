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
dynamic add
"""
import functools
import copy
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import in_record
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.util_compute import check_fc_fuse
from impl.util.util_compute import batchmatmul_elem_nd2nz
from impl.util.util_compute import batchmatmul_elem_reshape
from impl.util.util_compute import check_batchmatmul_fuse
from impl.util.util_compute import fetch_batchmatmul_fuse_tensor
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_tensor_dict import get_format_for_broardcast
from impl.util.util_tensor_dict import TensorClass
from impl.util.platform_adapter import tbe_context
from tbe.dsl.base import operation


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    Constant in this class
    """
    GENERAL_INPUT_LENGTH = 5
    FC_LENGTH_MIN = 2
    FC_LENGTH_MAX = 4


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,not-use-list-comprehension
# 'pylint: disable=invalid-name,too-many-locals,too-many-branches,unused-variable,too-many-nested-blocks
# 'pylint: disable=too-many-statements,too-many-boolean-expressions,consider-using-enumerate,too-many-return-values
def op_select_format(input_x, input_y, output_z, kernel_name="add"):
    """
    select format dynamically \n
    op_select_format support desc:

    1.when input x's ori_shape is 4, and bias's shape is not 1. \n
    The Op Bias can support
    ND + ND = ND,
    NC1HWC0 + NC1HWC0 = NC1HWC0.

        for example:
        inputs:
            x        ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"
            bias     ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"
        outputs:
            y        ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"

    2.In other scenes, all input(x, bias) only support ND.

        for example:
        inputs:
            x        ori shape = [2] ori_format = "ND"
            bias     ori shape = [2] ori_format = "ND"
        outputs:
            y        ori shape = [2] ori_format = "ND"

    """
    # do this scene like: input_x shape is [2,3,4] input_y shape is [1,]
    vadd_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vadd", "float32")
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    vadd_support_complex32 = tbe_platform.api_check_support("tbe.dsl.vadd", "complex32")
    vadd_support_complex64 = tbe_platform.api_check_support("tbe.dsl.vadd", "complex64")

    # filter dtype of add
    dtype_list = ["float16", "int32"]
    if vadd_support_fp32:
        dtype_list.append("float")
        # If the platform does not support float32 data type,
        # neither of uint8 and int8 is supported at the same time
        dtype_list.append("uint8")
        dtype_list.append("int8")
        # bfloat16 will cast to fp32
        if bfp16_support:
            dtype_list.append("bfloat16")

    if cce_product in ("Ascend310B", "AS31XM1", "Ascend310P", "Ascend910", "Ascend910B", "Ascend910_93"):
        # These versions support int64 and bool dtype calculation with aicore
        dtype_list.append("int64")
        dtype_list.append("bool")

    if vadd_support_complex32:
        dtype_list.append("complex32")
    if vadd_support_complex64:
        dtype_list.append("complex64")

    dtype_list_input0 = dtype_list.copy()
    dtype_list_input1 = dtype_list.copy()
    dtype_list_output = dtype_list.copy()

    if vadd_support_fp32:
        # float16 * float32 = float32
        dtype_list_input0.append("float16")
        dtype_list_input1.append("float32")
        dtype_list_output.append("float32")
        # float32 * float16 = float32
        dtype_list_input0.append("float32")
        dtype_list_input1.append("float16")
        dtype_list_output.append("float32")
        if bfp16_support:
            # bfloat16 * float32 = float32
            dtype_list_input0.append("bfloat16")
            dtype_list_input1.append("float32")
            dtype_list_output.append("float32")
            # float32 * bfloat16 = float32
            dtype_list_input0.append("float32")
            dtype_list_input1.append("bfloat16")
            dtype_list_output.append("float32")

    if util_common.is_unknown([input_x, input_y]):
        tensor_cls_1 = TensorClass(input_x)
        tensor_cls_2 = TensorClass(input_y)
        format_op_select_res = get_format_for_broardcast([tensor_cls_1, tensor_cls_2])

        format_input1, format_input2, format_output = format_op_select_res
        support_format_num = len(format_input1)

        dtype_result_input0 = []
        dtype_result_input1 = []
        dtype_result_output = []
        for i, _ in enumerate(dtype_list_input0):
            dtype_result_input0 += [dtype_list_input0[i]] * support_format_num
            dtype_result_input1 += [dtype_list_input1[i]] * support_format_num
            dtype_result_output += [dtype_list_output[i]] * support_format_num

        format_input1 = format_input1 * len(dtype_list_input0)
        format_input2 = format_input2 * len(dtype_list_input1)
        format_output = format_output * len(dtype_list_output)

        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x1",
                                               datatype=",".join(dtype_result_input0),
                                               format=",".join(format_input1),
                                               unknownshape_format=",".join(format_input1))
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x2",
                                               datatype=",".join(dtype_result_input1),
                                               format=",".join(format_input2),
                                               unknownshape_format=",".join(format_input2))
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype=",".join(dtype_result_output),
                                                format=",".join(format_output),
                                                unknownshape_format=",".join(format_output))

        param_list = [input0, input1, output0]
        param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

        return param_dynamic_in_json

    param_dynamic_in_json = op_sub_select_format(input_x, input_y, output_z, kernel_name)
    if param_dynamic_in_json != 'None':
        return param_dynamic_in_json

    shape_x = input_x.get("ori_shape")
    shape_y = input_y.get("ori_shape")

    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    format_5d_list = ["NDHWC", "DHWCN", "NCDHW"]
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_list = ["float16", "int32", "int8", "uint8"]
    else:
        dtype_list = ["float32", "float16", "int32", "int8", "uint8"]
        if tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322bf16f"):
            dtype_list.append("bfloat16")
        if cce_product in ("Ascend310B", "AS31XM1", "Ascend910", "Ascend910B", "Ascend910_93", "Ascend310P"):
            dtype_list.append("int64")
            dtype_list.append("bool")
        if tbe_platform.api_check_support("tbe.dsl.vadd", "complex32"):
            dtype_list.append("complex32")
        if tbe_platform.api_check_support("tbe.dsl.vadd", "complex64"):
            dtype_list.append("complex64")

    format_x = input_x.get("ori_format")
    format_y = input_y.get("ori_format")

    dtype_total = []
    format_nd = ["ND"]
    format_nz = ["FRACTAL_NZ"]
    list_input = [input_x, input_y]
    # 2dims add (3|4)dims,fe regards 2dims as HW, actually is WC
    format_5hd_flag = [len(shape_x), len(shape_y)] not in [[2, 4], [2, 3], [3, 2], [4, 2]]
    format_5hd = ["NC1HWC0"] if format_5hd_flag else []

    len_format_list = len(dtype_list)
    add_nd_nz = all([len(shape_x) == 1, len(shape_y) >= 2, shape_x[-1] == shape_y[-1], shape_x[0] % 16 == 0])
    add_nz_nd = all([len(shape_y) == 1, len(shape_x) >= 2, shape_x[-1] == shape_y[-1], shape_y[0] % 16 == 0])

    x_flag = {
        "5d": len(shape_x) == 5 and format_x in format_5d_list,
        "4d": len(shape_x) == 4 and format_x in format_4d_list,
        "Scalar": len(shape_x) == 1 and shape_x[0] == 1
    }
    y_flag = {
        "5d": len(shape_y) == 5 and format_y in format_5d_list,
        "4d": len(shape_y) == 4 and format_y in format_4d_list,
        "Scalar": len(shape_y) == 1 and shape_y[0] == 1
    }

    common_flag = {
        "half_16_div_flg": (_can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y)) or
                           (not _can_division_sixteen(shape_x) and _can_division_sixteen(shape_y))
    }
    if x_flag.get("5d") or x_flag.get("4d"):
        x_cdim = shape_x[format_x.index("C")]
        x_ndim = shape_x[format_x.index("N")]
    if y_flag.get("5d") or y_flag.get("4d"):
        y_cdim = shape_y[format_y.index("C")]
        y_ndim = shape_y[format_y.index("N")]

    format_flag = {
        "NDC1HWC0":
            x_flag.get("5d") and y_flag.get("5d") and x_cdim == y_cdim,
        "FRACTAL_Z_3D":
            x_flag.get("5d") and y_flag.get("5d") and x_cdim == y_cdim and x_ndim == y_ndim,
        "FRACTAL_NZ":
            len(shape_x) >= 2 and len(shape_y) >= 2 and shape_x[-2:] == shape_y[-2:],
        "NC1HWC0":
            x_flag.get("4d") and y_flag.get("4d") and
            ((format_y == format_x and ((x_cdim % 16 == 0 and y_cdim % 16 == 0) or x_cdim == y_cdim) and _can_broad([
                shape_x[format_x.index(format_x[0])], shape_x[format_x.index(format_x[1])], format_x[0] != "C",
                format_x[1] != "C"
            ], [
                shape_y[format_y.index(format_y[0])], shape_y[format_y.index(format_y[1])], format_y[0] != "C",
                format_y[1] != "C"
            ])) or (list(shape_x) == list(shape_y) and -1 not in shape_x) or
             (common_flag.get("half_16_div_flg") and x_cdim % 16 == 0 and y_cdim % 16 == 0 and
              (x_cdim == y_cdim or x_cdim == 16 or y_cdim == 16))),
        "FRACTAL_Z":
            x_flag.get("4d") and y_flag.get("4d") and format_x == format_y and
            ((all(i % 16 == 0
                  for i in [x_cdim, y_cdim, x_ndim, y_ndim]) and util_common.is_support_fractal_z_inputs(list_input) and
              ((list(shape_x) == list(shape_y) and format_x.upper() in ("NCHW", "NHWC")) or
               (format_x.upper() == "HWCN" and shape_x[0] * shape_x[1] == shape_y[0] * shape_y[1]))) or
             (list(shape_x) == list(shape_y) and util_common.is_same_group(list_input)) or
             (list(shape_x) == list(shape_y) and util_common.is_same_group(list_input))),
        "ND":
            True
    }
    format_flag["NC1HWC0"] = format_flag.get("NC1HWC0") or (
        x_flag.get("4d") and y_flag.get("Scalar") and
        x_cdim % 16 == 0) or (x_flag.get("Scalar") and y_flag.get("4d") and
                              y_cdim % 16 == 0) or (len(shape_x) == 1 and len(shape_y) == 1 and shape_x[0] % 16 == 0 and
                                                    shape_y[0] % 16 == 0)
    format_flag["FRACTAL_Z"] = format_flag.get("FRACTAL_Z") or (
        util_common.is_support_fractal_z_inputs(list_input) and
        (x_flag.get("4d") and y_flag.get("Scalar") and x_cdim % 16 == 0 and x_ndim % 16 == 0) or
        (x_flag.get("Scalar") and y_flag.get("4d") and y_cdim % 16 == 0 and y_ndim % 16 == 0))

    format_flag["NC1HWC0"] = format_5hd_flag and format_flag.get("NC1HWC0")

    format_flag["NC1HWC0"] = format_flag.get("NC1HWC0") or \
                             (len(shape_x) == len(shape_y) == 1 and format_x == format_y and
                              format_y in ("NHWC",) and shape_x[0] == 1) or \
                             (len(shape_x) == len(shape_y) == 1 and format_x == format_y and
                              format_x in ("NHWC",) and shape_y[0] == 1)

    format_list = [i for i in format_flag if format_flag.get(i)]
    for dtype in dtype_list:
        dtype_total = dtype_total + [dtype] * len(format_list)
    format_list = format_list * len_format_list
    # ND+ND NZ+NZ 5HD+5HD FZ+FZ
    if len(shape_x) >= 2 and len(shape_y) >= 2 and shape_x[-2:] == shape_y[-2:]:
        format_list_input0 = format_list
        format_list_input1 = format_list
        format_list_output = format_list
        unknownshape_format_list = ["ND"] * len(dtype_total)

    # NZ+ND,ND+ND,5HD+5HD,FZ+FZ,ND+NZ
    elif len(shape_x) >= 2 and len(shape_y) >= 2 and (
        (_can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y)) or
        (not _can_division_sixteen(shape_x) and _can_division_sixteen(shape_y))):

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * 1
        format_list0 = format_list + format_nz * len_format_list
        format_list1 = format_list + format_nd * len_format_list
        if _can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y):
            format_list_input0 = format_list0
            format_list_input1 = format_list1
            format_list_output = format_list0
        else:
            format_list_input0 = format_list1
            format_list_input1 = format_list0
            format_list_output = format_list0
        unknownshape_format_list = ["ND"] * len(dtype_total)

    elif add_nd_nz or add_nz_nd:
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * 1
        format_list0 = format_list + format_nz * len_format_list
        format_list1 = format_list + format_nd * len_format_list
        if len(shape_y) == 1 and x_flag.get("4d") and shape_y[0] % 16 == 0 and x_cdim % 16 == 0:
            format_list0 = format_list + format_5hd * len_format_list
            format_list1 = format_list + format_5hd * len_format_list
        if add_nz_nd:
            format_list_input0 = format_list0
            format_list_input1 = format_list1
            format_list_output = format_list0
        else:
            format_list_input0 = format_list1
            format_list_input1 = format_list0
            format_list_output = format_list0
        unknownshape_format_list = ["ND"] * len(dtype_total)

    # 5HD+scalar,ND+ND,FZ+scalar
    elif len(shape_x) >= 2 and len(shape_y) == 1 and shape_y[0] == 1:
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * 1
        format_list0 = format_list + format_nd * len_format_list
        format_list1 = format_nd * len(format_list) + format_nd * len_format_list
        format_list_input0 = format_list0
        format_list_input1 = format_list1
        format_list_output = format_list0
        unknownshape_format_list = ["ND"] * len(dtype_total)

    # ND+ND,scalar+5HD,scalar+FZ
    elif len(shape_y) >= 2 and len(shape_x) == 1 and shape_x[0] == 1:
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * 1
        format_list0 = format_list + format_nd * len_format_list
        format_list1 = format_nd * len(format_list) + format_nd * len_format_list
        format_list_input0 = format_list1
        format_list_input1 = format_list0
        format_list_output = format_list0
        unknownshape_format_list = ["ND"] * len(dtype_total)
    # ND+ND,5HD+5HD
    else:
        format_list_input0 = format_list
        format_list_input1 = format_list
        format_list_output = format_list
        unknownshape_format_list = ["ND"] * len(dtype_total)

    if _can_broadcast(shape_x, shape_y) and len(shape_x) != len(shape_y):
        y_format = input_y.get("ori_format")
        x_format = input_x.get("ori_format")
        if x_format == "NHWC" or y_format == "NHWC":
            if (len(shape_x) > 4 or len(shape_y) > 4) or para_check.is_scalar(shape_x) or para_check.is_scalar(shape_y):
                formats = ["ND"]
            else:
                formats = format_5hd
            for item in formats:
                dtype_total = dtype_total + dtype_list
                format_list_input0 = format_list_input0 + [item] * len(dtype_list)
                format_list_input1 = format_list_input1 + [item] * len(dtype_list)
                format_list_output = format_list_output + [item] * len(dtype_list)
            unknownshape_format_list = ["ND"] * len(dtype_total)

    # dynamic shape, 5HD + 5HD
    if -1 in shape_x or -1 in shape_y:
        if x_flag.get("4d") and x_cdim > 0 and format_x == format_y and y_flag.get("4d") and x_cdim == y_cdim:
            unknownshape_format_list = [item if item == "NC1HWC0" else "ND" for item in format_list] + \
                                       ["ND"] * (len(dtype_total) - len(format_list))

    param_list = _gen_para(dtype_total, format_list_input0, format_list_input1, format_list_output,
                           unknownshape_format_list, shape_x, shape_y)
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def op_sub_select_format(x1, x2, y, kernel_name="add"):
    """
    Dynamic matching format

    Parameters
    ----------
    x1 : dict
        shape and dtype of input0
    x2 : dict
        shape and dtype of input1
    y : dict
        shape and dtype of output, should be same type as input0

    kernel_name : str
        kernel name, default value is "add"

    Returns
    -------
    None
    """
    shape_x1 = x1.get("ori_shape")
    shape_x2 = x2.get("ori_shape")

    shape_x1 = shape_util.scalar2tensor_one(shape_x1)
    shape_x2 = shape_util.scalar2tensor_one(shape_x2)

    enum_x1 = functools.reduce(lambda x, y: x * y, shape_x1)
    enum_x2 = functools.reduce(lambda x, y: x * y, shape_x2)

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_list = ["float16", "int32", "int8", "uint8"]
    else:
        dtype_list = ["float32", "float16", "int32", "int8", "uint8"]
        if tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322bf16f"):
            dtype_list.append("bfloat16")
        if cce_product in ("Ascend310B", "AS31XM1", "Ascend910", "Ascend910B", "Ascend910_93", "Ascend310P"):
            dtype_list.append("int64")
            dtype_list.append("bool")

    # 5HD + scalar
    if len(shape_x2) == 1 and enum_x2 == 1:
        format_list = ("ND", "NCHW", "NHWC", "FRACTAL_NZ", "NC1HWC0", "FRACTAL_Z", "C1HWNCoC0")
        dtype_list_total = functools.reduce(lambda x, y: x + y, [[ele] * len(format_list) for ele in dtype_list])
        format_list_for_non_one = format_list * len(dtype_list)
        unknownshape_format_list = ["ND"] * len(format_list) * len(dtype_list)
        format_list_for_one = [x2.get("format")] * len(format_list) * len(dtype_list)
    else:
        return 'None'

    if -1 in shape_x1 or -1 in shape_x2:
        input0 = gen_param(classify="input0",
                           name="x1",
                           datatype=",".join(dtype_list_total),
                           format=",".join(format_list_for_non_one),
                           unknownshape_format=",".join(unknownshape_format_list))
        input1 = gen_param(classify="input1",
                           name="x2",
                           datatype=",".join(dtype_list_total),
                           format=",".join(format_list_for_one),
                           unknownshape_format=",".join(unknownshape_format_list))
        output0 = gen_param(classify="output0",
                            name="y",
                            datatype=",".join(dtype_list_total),
                            format=",".join(format_list_for_non_one),
                            unknownshape_format=",".join(unknownshape_format_list))
    else:
        input0 = gen_param(classify="input0",
                           name="x1",
                           datatype=",".join(dtype_list_total),
                           format=",".join(format_list_for_non_one))
        input1 = gen_param(classify="input1",
                           name="x2",
                           datatype=",".join(dtype_list_total),
                           format=",".join(format_list_for_one))
        output0 = gen_param(classify="output0",
                            name="y",
                            datatype=",".join(dtype_list_total),
                            format=",".join(format_list_for_non_one))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _can_division_sixteen(shape):
    """
    check whether divided by 16.

    Parameters
    ----------
    shape: list or tuple

    Returns:
    -------
    None
    """
    if len(shape) < 2:
        if shape[-1] == 0:
            expected_value = "equal to 0"
            real_value = "not equal to 0"
            error_manager_vector.raise_err_input_value_invalid("add", "value of shape", expected_value, real_value)
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        expected_value = "not equal to 0"
        real_value = "equal to 0"
        error_manager_vector.raise_err_input_value_invalid("add", "shape[-1] and shape[-2]", expected_value, real_value)

    if shape[-1] % 16 == 0 and shape[-2] % 16 == 0:
        return True

    return False


def _can_broadcast(shape1, shape2):
    """
    check whether can broadcast or no.

    Parameters
    ----------
    shape1: list or tuple.
    shape2: list or tuple.

    Returns:
    -------
    None
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1

    _dv = len(shape1) - len(shape2)
    shape2 = [1] * _dv + shape2

    for shape1_i, shape2_i in zip(shape1, shape2):
        if not shape1_i == shape2_i and shape1_i != 1 and shape2_i != 1:
            return False

    return True


def _can_broad(x, y):
    if x[2]:
        x[0] *= 16
        y[0] *= 16
    if x[3]:
        x[1] *= 16
        y[1] *= 16
    return (x[0] == y[0] and (x[1] == 16 or y[1] == 16 or x[1] == y[1])) or (
        x[1] == y[1] and
        (x[0] == 16 or
         y[0] == 16)) or x[0] == y[1] == 16 or x[0] == x[1] == 16 or x[1] == y[0] == 16 or y[0] == y[1] == 16


def _gen_para(dtype_total, format_list0, format_list1, format_list2, unknownshape_format_list, shape_x, shape_y):
    """
    generate paras list
    :param dtype_total:
    :param format_list0:
    :param format_list1:
    :param format_list2:
    :param unknownshape_format_list:
    :param shape_x:
    :param shape_y:
    :return:
    """
    dtype_list_input0 = dtype_total.copy()
    dtype_list_input1 = dtype_total.copy()
    dtype_list_output = dtype_total.copy()
    format_list_0 = format_list0.copy()
    format_list_1 = format_list1.copy()
    format_list_2 = format_list2.copy()

    if -1 in shape_x or -1 in shape_y:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x1",
                                               datatype=",".join(dtype_list_input0),
                                               format=",".join(format_list_0),
                                               unknownshape_format=",".join(unknownshape_format_list))
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x2",
                                               datatype=",".join(dtype_list_input1),
                                               format=",".join(format_list_1),
                                               unknownshape_format=",".join(unknownshape_format_list))
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype=",".join(dtype_list_output),
                                                format=",".join(format_list_2),
                                                unknownshape_format=",".join(unknownshape_format_list))
    else:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x1",
                                               datatype=",".join(dtype_list_input0),
                                               format=",".join(format_list_0))
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x2",
                                               datatype=",".join(dtype_list_input1),
                                               format=",".join(format_list_1))
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype=",".join(dtype_list_output),
                                                format=",".join(format_list_2))
    return [input0, input1, output0]


def _add_check_format(x, y):
    """
    funtion to check format

    Parameters
    ----------
    x: dict
        dict of x, include keys(shape and dtype).
    y: dict
        dict of x, include keys(shape and dtype).

    Returns:
    -------
    format_pattern: int
    """
    format_pattern = 0
    shape1 = x.get("shape")
    shape2 = y.get("shape")
    list_format = [x.get("format"), y.get("format")]
    shape1 = shape_util.scalar2tensor_one(shape1)
    shape2 = shape_util.scalar2tensor_one(shape2)
    check_list = [["FRACTAL_NZ", "ND"], ["ND", "FRACTAL_NZ"], ["FRACTAL_NZ", "NHWC"], ["NHWC", "FRACTAL_NZ"],
                  ["FRACTAL_NZ", "NCHW"], ["NCHW", "FRACTAL_NZ"], ["HWCN", "FRACTAL_NZ"]]
    if list_format == check_list[0] and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[1] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[6] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[2] and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[3] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[4] and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[5] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2

    return format_pattern


def _infer_shape(format_pattern, x, y):
    """
    funtion to infer shape

    Parameters
    ----------
    format_pattern: int
    x: dict
        dict of x, include keys(shape and dtype).
    y: dict
        dict of x, include keys(shape and dtype).

    Returns:
    -------
    shape_x: shape of x
    shape_y: shape of y
    """
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    ori_shape_x = x.get("ori_shape")
    ori_shape_y = y.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)

    if format_pattern == 1:
        ori_shape_x, shape_y, shape_max = shape_util.broadcast_shapes(ori_shape_x,
                                                                      shape_y,
                                                                      param_name_input1="input_x",
                                                                      param_name_input2="input_y")
        if shape_y[-2] == 1 and shape_y[-1] == ori_shape_x[-1]:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-3] = 1
            shape_y[-1] = shape_x[-1]
            shape_y[-4] = shape_x[-4]

        elif shape_y[-2] == ori_shape_x[-2] and shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-4] = 1
            shape_y[-2] = shape_x[-2]
            shape_y[-3] = shape_x[-3]

        elif shape_y[-2] == shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)

    elif format_pattern == 2:
        shape_x, ori_shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                      ori_shape_y,
                                                                      param_name_input1="input_x",
                                                                      param_name_input2="input_y")
        if shape_x[-2] == 1 and shape_x[-1] == ori_shape_y[-1]:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-3] = 1
            shape_x[-1] = shape_y[-1]
            shape_x[-4] = shape_y[-4]

        elif shape_x[-2] == ori_shape_y[-2] and shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-4] = 1
            shape_x[-2] = shape_y[-2]
            shape_x[-3] = shape_y[-3]

        elif shape_x[-2] == shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)

    return shape_x, shape_y


def static_reshape(input_x, input_y):
    """static reshape"""
    # check shape
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)
    if shape_y[0] == 1 and len(shape_y) == 1:
        broadcast_flag = True
        is_scene_1d = True

        # check x2 is 1D or not
        if para_check.is_scalar(shape_y):
            broadcast_flag = False
            shape_y = tuple([1] * (len(shape_x) - len(shape_y))) + tuple(shape_y)
    else:
        is_scene_1d = False
        broadcast_flag = True

        # format_pattern value means
        # 1: Nz and vector
        # 2: vector and Nz
        # 0:  Nz scalar  Nz Nz  ND ND
        format_pattern = _add_check_format(input_x, input_y)
        shape_x, shape_y = _infer_shape(format_pattern, input_x, input_y)
        shape_x = shape_util.scalar2tensor_one(shape_x)
        shape_y = shape_util.scalar2tensor_one(shape_y)
        para_check.check_shape(shape_x, param_name="input_x")
        para_check.check_shape(shape_y, param_name="input_y")

        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                  shape_y,
                                                                  param_name_input1="input_x",
                                                                  param_name_input2="input_y")
        if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
            shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
            shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
            shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]
    return [shape_x, shape_y, broadcast_flag, is_scene_1d]


def calc_input_tensor(input_x, input_y):
    """compute with batchmatmul"""
    if in_record():
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(input_x.shape, input_y.shape,
                                                                  param_name_input1="input_x",
                                                                  param_name_input2="input_y")
        input_x = tbe.broadcast(input_x, shape_max)
        input_y = tbe.broadcast(input_y, shape_max)
        return False, input_x, input_y

    batchmatmul_flag = False
    shape_x = input_x.shape
    shape_y = input_y.shape
    fc_fusion_right_flag = len(shape_x) == Constant.GENERAL_INPUT_LENGTH and check_fc_fuse(input_y)
    fc_fusion_left_flag = len(shape_y) == Constant.GENERAL_INPUT_LENGTH and check_fc_fuse(input_x)
    if fc_fusion_left_flag or fc_fusion_right_flag:
        if len(shape_x) == Constant.FC_LENGTH_MIN:
            input_y = tvm.compute(
                shape_x, lambda x, y: input_y(x % shape_y[0], (y % (shape_y[1] * shape_y[-1])) // shape_y[-1], 0, 0,
                                              (y % (shape_y[1] * shape_y[-1])) % shape_y[-1]))
        elif len(shape_x) == Constant.FC_LENGTH_MAX:
            input_y = tvm.compute(shape_x, lambda x1, y1, y0, x0: input_y(0, x1 % shape_y[1], 0, 0, x0 % shape_y[-1]))
        if len(shape_y) == Constant.FC_LENGTH_MIN:
            input_x = tvm.compute(
                shape_y, lambda x, y: input_x(x % shape_x[0], (y % (shape_x[1] * shape_x[-1])) // shape_x[-1], 0, 0,
                                              (y % (shape_x[1] * shape_x[-1])) % shape_x[-1]))
        elif len(shape_y) == Constant.FC_LENGTH_MAX:
            input_x = tvm.compute(shape_y, lambda x1, y1, y0, x0: input_x(0, x1 % shape_x[1], 0, 0, x0 % shape_x[-1]))
    else:
        batch_matmul_flag_lhs = check_batchmatmul_fuse(input_x)
        batch_matmul_flag_rhs = check_batchmatmul_fuse(input_y)
        if batch_matmul_flag_lhs or batch_matmul_flag_rhs:
            batchmatmul_flag = True
            if batch_matmul_flag_rhs:
                input_x, input_y = input_y, input_x
        else:
            shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                      shape_y,
                                                                      param_name_input1="input_x",
                                                                      param_name_input2="input_y")
            input_x = tbe.broadcast(input_x, shape_max)
            input_y = tbe.broadcast(input_y, shape_max)

    return batchmatmul_flag, input_x, input_y


def add_compute_for_batchmatmul(lhs_tensor, rhs_tensor):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    lhs_tensor: TVM tensor
        the placeholder of first input data
    rhs_tensor: TVM tensor
        the placeholder of second input data
    Returns
    -------
    res : output of the lhs_tensor + rhs_tensor
    """
    if "para_name" in lhs_tensor.op.attrs:
        para_name = lhs_tensor.op.attrs["para_name"]
        para_name += "_add"
    else:
        para_name = "add"
    batch_matmul_tensor = fetch_batchmatmul_fuse_tensor(lhs_tensor)
    if batch_matmul_tensor is None:
        error_manager_vector.raise_err_specific_reson("add", "ub fusion with bmm, can't fetch batchmatmul tensor.")

    batch_shape = shape_util.shape_to_list(batch_matmul_tensor.op.attrs["batch_shape"])
    para_dict = {"format_elem": rhs_tensor.op.attrs["format"], "batch_shape": batch_shape}
    rhs_tensor, shape_max = batchmatmul_elem_nd2nz(batch_matmul_tensor, rhs_tensor, para_dict, para_name)
    rhs_tensor = tbe.broadcast(rhs_tensor, shape_max)
    rhs_tensor = batchmatmul_elem_reshape(batch_matmul_tensor, rhs_tensor, batch_shape, para_name)
    res = tbe.vadd(lhs_tensor, rhs_tensor)
    res.op.attrs["batch_shape"] = batch_shape
    res.op.attrs["para_name"] = para_name

    return res


def check_add_compute_ub_fusion():
    """
    [NZ, ND(except scalar)] do not support ub_fusion
    "Broadcast_Nz" is for matmul fusion
    """
    inputs = tbe_context.op_context.get_context().get_op_info()[0].inputs
    if not util_common.is_unknown([inputs[0], inputs[1]]):
        format_pattern = _add_check_format(inputs[0], inputs[1])
        # shape will modify by add entry function
        input0_shape = inputs[0].get("ori_shape")
        input1_shape = inputs[1].get("ori_shape")
        if format_pattern == 1 and len(input1_shape) > 2:
            operation.get_context().add("_special_pattern", "Broadcast_Nz")
        if format_pattern == 2 and len(input0_shape) > 2:
            operation.get_context().add("_special_pattern", "Broadcast_Nz")

    return True


@register_operator_compute("Add", op_mode="dynamic", support_fusion=check_add_compute_ub_fusion)
def add_compute(input_x, input_y, output_z, kernel_name="add"):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x:
    left input, may be dict or tensor

    input_y:
    left input, may be dict or tensor

    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : output of the data's add
    """
    x_dtype = input_x.dtype.lower()
    y_dtype = input_y.dtype.lower()
    is_mix_dtype = x_dtype != y_dtype

    if x_dtype in ("uint8", "int8", "bool"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    if x_dtype in ("bfloat16",) or is_mix_dtype:
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")

    batchmatmul_flag, input_x, input_y = calc_input_tensor(input_x, input_y)
    if batchmatmul_flag:
        return add_compute_for_batchmatmul(input_x, input_y)

    res = tbe.vadd(input_x, input_y)

    if x_dtype in ("uint8", "int8", "bool"):
        res = util_common.uint8_int8_overflow_proc(res, x_dtype)

    output_dtype = output_z.get("dtype")
    if res.dtype != output_dtype:
        if output_dtype == "bfloat16":
            res = tbe.round(res, "bfloat16")
        else:
            res = tbe.cast_to(res, output_dtype)

    return res


# 'pylint: disable=too-many-locals
@register_operator("Add")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def add(input_x, input_y, output_z, kernel_name="add"):
    """
    algorithm: add
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x : dict
       including shape, dtype and range, only support bfloat16, float16, float32, int32, uint8, int8, bool
    input_y : dict
       including shape, dtype and range, only support bfloat16, float16, float32, int32, uint8, int8, bool
    output_z: dict
       shape should be broadcast shape of input, and type equals to input
    kernel_name : str
       cce kernel name, default value is add

    Returns
    -------
    None
    """
    input_x = copy.deepcopy(input_x)
    input_y = copy.deepcopy(input_y)
    output_z = copy.deepcopy(output_z)
    # check input tensor data_type
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    out_dtype = output_z.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "int8", "bfloat16", "int64", "complex32", "complex64", "bool")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")
    para_check.check_dtype(out_dtype, check_list, param_name="output_z")

    mix_dtype_list = (("float16", "float32", "float32"), ("float32", "float16", "float32"),
                      ("bfloat16", "float32", "float32"), ("float32", "bfloat16", "float32"))
    is_valid_mix_dtpye = (x_dtype, y_dtype, out_dtype) in mix_dtype_list
    if not is_valid_mix_dtpye and x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("add", "input_x", "input_y", str(x_dtype), str(y_dtype))

    # calc for static and dynamic merge
    if not util_common.is_unknown([input_x, input_y]):
        shape_x, shape_y, _, _ = static_reshape(input_x, input_y)
        range_x = util_common.gen_range(shape_x)
        range_y = util_common.gen_range(shape_y)
        input_x["shape"] = shape_x
        input_x["range"] = range_x
        input_y["shape"] = shape_y
        input_y["range"] = range_y

    extra_params = {"ignore_fractal_format": True, "unfold_mode": True}
    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []
    for (_input_x, _input_y) in ins:
        with tbe.compute():
            shape_x, shape_y = shape_util.variable_shape([_input_x, _input_y])
            data_x = tvm.placeholder(shape_x, name="data_1", dtype=x_dtype)
            data_y = tvm.placeholder(shape_y, name="data_2", dtype=y_dtype)
            res = add_compute(data_x, data_y, output_z, kernel_name)

            tensors.append((data_x, data_y, res))
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
