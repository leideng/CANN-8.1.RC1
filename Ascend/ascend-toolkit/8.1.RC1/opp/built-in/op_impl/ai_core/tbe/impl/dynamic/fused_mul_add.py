# Copyright 2019 Huawei Technologies Co., Ltd
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
fused_mul_add
"""
from impl import constant_util
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.platform_adapter import register_operator
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import update_shape_for_other_format
from impl.util.util_compute import batchmatmul_elem_nd2nz
from impl.util.util_compute import batchmatmul_elem_reshape
from impl.util.util_compute import check_batchmatmul_fuse


# 'pylint: disable=unused-argument,too-many-locals,invalid-name,too-many-branches
# 'pylint: disable=arguments-out-of-order,unused-variable,too-many-statements,unused-argument,too-many-locals
def _infer_shape_one(shape_input0, shape_input1, shape_input2, format_pattern):
    """
    shape_input0 : FRACTAL_NZ, [N,...,A,B,16,16]
    last_two_dims : [B*16, A*16]
    """
    if format_pattern == 2:
        shape_input0, shape_input1 = shape_input1, shape_input0
    if format_pattern == 3:
        shape_input0, shape_input2 = shape_input2, shape_input0

    last_two_dims = [shape_input0[-2] * shape_input0[-3], shape_input0[-4] * shape_input0[-1]]
    condition2 = (len(shape_input1) == 1 and shape_input1[0] == 1)
    if not condition2:
        if len(shape_input1) == 1:
            shape_input1.insert(0, 1)
        condition0 = (shape_input1[-1] == last_two_dims[-1])
        condition1 = (shape_input1[-2] == last_two_dims[-2])

    condition5 = (len(shape_input2) == 1 and shape_input2[0] == 1)
    if not condition5:
        if len(shape_input2) == 1:
            shape_input2.insert(0, 1)
        condition3 = (shape_input2[-1] == last_two_dims[-1])
        condition4 = (shape_input2[-2] == last_two_dims[-2])

    if condition2:
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1,
                                        param_name_input1="input0",
                                        param_name_input2="input1")
    elif condition0 and not condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-4] = shape_input0[-4]
        shape_input1[-1] = shape_input0[-1]
        shape_input1[-2] = 1
        shape_input1[-3] = 1
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1,
                                        param_name_input1="input0",
                                        param_name_input2="input1")
    elif not condition0 and condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-2] = shape_input0[-2]
        shape_input1[-3] = shape_input0[-3]
        shape_input1[-4] = 1
        shape_input1[-1] = 1
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1,
                                        param_name_input1="input0",
                                        param_name_input2="input1")
    else:
        error_detail = 'shape of input1 or input0 is illegal'
        error_manager_vector.raise_err_specific_reson("fused_mul_add", error_detail)

    if condition5:
        shape_input2, shape_max_mul, shape_max_add0 = \
            shape_util.broadcast_shapes(shape_input2, shape_max_mul,
                                        param_name_input1="input2",
                                        param_name_input2="shape_max_mul")
    elif condition3 and not condition4:
        shape_input2.append(1)
        shape_input2.append(1)
        shape_input2[-4] = shape_input0[-4]
        shape_input2[-1] = shape_input0[-1]
        shape_input2[-2] = 1
        shape_input2[-3] = 1
        shape_input2, shape_max_mul, shape_max_add0 = \
            shape_util.broadcast_shapes(shape_input2, shape_max_mul,
                                        param_name_input1="input2",
                                        param_name_input2="shape_max_mul")
    elif not condition3 and condition4:
        shape_input2.append(1)
        shape_input2.append(1)
        shape_input2[-2] = shape_input0[-2]
        shape_input2[-3] = shape_input0[-3]
        shape_input2[-4] = 1
        shape_input2[-1] = 1
        shape_input2, shape_max_mul, shape_max_add0 = \
            shape_util.broadcast_shapes(shape_input2, shape_max_mul,
                                        param_name_input1="input2",
                                        param_name_input2="shape_max_mul")
    else:
        error_detail = 'shape of input2 or input0 is illegal'
        error_manager_vector.raise_err_specific_reson("fused_mul_add", error_detail)

    if format_pattern == 2:
        shape_input0, shape_input1 = shape_input1, shape_input0
    if format_pattern == 3:
        shape_input0, shape_input2 = shape_input2, shape_input0

    return shape_input0, shape_input1, shape_input2


def _infer_shape_two(shape_input0, shape_input1, shape_input2, format_pattern):
    """
    shape_input0 : FRACTAL_NZ, [N,...,A,B,16,16]
    last_two_dims : [B*16, A*16]
    """
    # support format_pattern == 4 or 5
    # Nz ND Nz || ND NZ NZ
    last_two_dims = [shape_input0[-2] * shape_input0[-3], shape_input0[-4] * shape_input0[-1]]

    condition2 = (len(shape_input1) == 1 and shape_input1[0] == 1)
    if not condition2:
        if len(shape_input1) == 1:
            shape_input1.insert(0, 1)
        condition0 = (shape_input1[-1] == last_two_dims[-1])
        condition1 = (shape_input1[-2] == last_two_dims[-2])

    if condition2:
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                        param_name_input2="input1")
    elif condition0 and not condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-4] = shape_input0[-4]
        shape_input1[-1] = shape_input0[-1]
        shape_input1[-2] = 1
        shape_input1[-3] = 1
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                        param_name_input2="input1")
    elif not condition0 and condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-2] = shape_input0[-2]
        shape_input1[-3] = shape_input0[-3]
        shape_input1[-4] = 1
        shape_input1[-1] = 1
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                        param_name_input2="input1")
    else:
        raise RuntimeError("shape of input1 or input0 is illegal")

    shape_input2, shape_max_mul, shape_max_add0 = \
        shape_util.broadcast_shapes(shape_input2, shape_max_mul, param_name_input1="input2",
                                    param_name_input2="shape_max_mul")

    return shape_input0, shape_input1, shape_input2


def _division_sixteen(shape):
    """
    check be div by sixteen
    """
    if len(shape) < 2:
        if shape[-1] == 0:
            error_detail = 'value of shape is illegal, shape[-1] == 0'
            error_manager_vector.raise_err_specific_reson("fused_mul_add", error_detail)
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        error_detail = 'value of shape is illegal, shape[-1] == %s, shape[-2] == %s' % (shape[-1], shape[-2])
        error_manager_vector.raise_err_specific_reson("fused_mul_add", error_detail)

    return shape[-1] % constant_util.SIZE_SIXTEEN == 0 and shape[-2] % constant_util.SIZE_SIXTEEN == 0


def split_bind(shape0, shape1):
    """
    check can be split together
    """
    if len(shape0) == 0 or len(shape1) == 0:
        return False
    if shape0[0] == 1 or shape1[0] == 1:
        return False
    if len(shape0) != len(shape1):
        return False
    if shape0[0] == shape1[0]:
        return True
    return False


def get_split_matrix(input0_shape, input1_shape, input2_shape):
    """
    get axis split matrix
    """
    axis_split_matrix = None
    if split_bind(input0_shape, input1_shape):
        input_slice_list = [[0, [0], [-1], [-1]], [1, [0], [-1], [-1]]]
        if split_bind(input1_shape, input2_shape):
            input_slice_list.append([2, [0], [-1], [-1]])
        split_0 = [SplitInput(*input_slice_list), SplitOutput([0, [0]])]
        axis_split_matrix = [split_0]

    elif split_bind(input0_shape, input2_shape):
        input_slice_list = [[0, [0], [-1], [-1]], [2, [0], [-1], [-1]]]
        split_0 = [SplitInput(*input_slice_list), SplitOutput([0, [0]])]
        axis_split_matrix = [split_0]

    elif split_bind(input1_shape, input2_shape):
        input_slice_list = [[0, [0], [-1], [-1]], [1, [0], [-1], [-1]]]
        split_0 = [SplitInput(*input_slice_list), SplitOutput([0, [0]])]
        axis_split_matrix = [split_0]

    return axis_split_matrix


def get_op_support_info(input0, input1, input2, output, kernel_name="fused_mul_add"):
    """
    get_op_support_info
    """
    input0_shape = list(input0.get('shape'))
    input1_shape = list(input1.get('shape'))
    input2_shape = list(input2.get('shape'))

    input_list = [input0, input1, input2]
    input_shape_list = [input0_shape, input1_shape, input2_shape]
    input_len_list = [len(input0_shape), len(input1_shape), len(input2_shape)]
    maxlen_idx = input_len_list.index(max(input_len_list))

    axis_split_matrix = None
    axis_reduce_list = None

    if input_len_list[maxlen_idx] != 0:
        for _idx, _input in enumerate(input_list):
            if _idx != maxlen_idx and input_len_list[_idx] != 0:
                input_shape_list[_idx] = \
                update_shape_for_other_format(_input['shape'],
                                              _input['format'].upper(),
                                              _input['ori_shape'],
                                              input_list[maxlen_idx]['format'].upper())

        axis_split_matrix = get_split_matrix(*input_shape_list)
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def op_select_format(input0, input1, input2, output, kernel_name="fused_mul_add"):
    """
    _division_sixteen : judge whether the last two dimensions are divided by 16
    scalar2tensor_one : convert scalar to tensor
    """
    support_dtype_list = ["float16", "float", "int32"]
    support_format_list = ["NCHW", "NHWC", "ND"]
    inputs = [input0, input1, input2]
    is_static = not util_common.is_unknown(inputs)
    if is_static:
        support_format_list.append("NC1HWC0")

    shape_0 = input0.get("ori_shape")
    shape_1 = input1.get("ori_shape")
    shape_2 = input2.get("ori_shape")

    shape_0 = shape_util.scalar2tensor_one(shape_0)
    shape_1 = shape_util.scalar2tensor_one(shape_1)
    shape_2 = shape_util.scalar2tensor_one(shape_2)

    input_shapes = [shape_0, shape_1, shape_2]
    input_formats = [support_format_list.copy() for i in range(len(input_shapes))]
    output_format = support_format_list.copy()
    if is_static:
        check_support_nz = [_division_sixteen(shape) for shape in input_shapes]
        has_nz = any(check_support_nz)
        if has_nz:
            for i, support_nz in enumerate(check_support_nz):
                if support_nz:
                    input_formats[i].append("FRACTAL_NZ")
                else:
                    input_formats[i].append("ND")
            output_format.append("FRACTAL_NZ")
    input0_format, input1_format, input2_format = input_formats

    dtype_list = [
        dtype
        for dtype in support_dtype_list
        for i in range(len(input0_format))
    ]
    input0_format_list = input0_format * len(support_dtype_list)
    input1_format_list = input1_format * len(support_dtype_list)
    input2_format_list = input2_format * len(support_dtype_list)
    output_format_list = output_format * len(support_dtype_list)

    dtype_list_str = ",".join(dtype_list)
    input0_format_list_str = ",".join(input0_format_list)
    input1_format_list_str = ",".join(input1_format_list)
    input2_format_list_str = ",".join(input2_format_list)
    output_format_list_str = ",".join(output_format_list)
    if is_static:
        input0 = util_select_op_base.gen_param(classify="input0",
                                                name="x1",
                                                datatype=dtype_list_str,
                                                format=input0_format_list_str)
        input1 = util_select_op_base.gen_param(classify="input1",
                                                name="x2",
                                                datatype=dtype_list_str,
                                                format=input1_format_list_str)
        input2 = util_select_op_base.gen_param(classify="input2",
                                                name="x3",
                                                datatype=dtype_list_str,
                                                format=input2_format_list_str)
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype=dtype_list_str,
                                                format=output_format_list_str)
    else:
        input0 = util_select_op_base.gen_param(classify="input0",
                                                name="x1",
                                                datatype=dtype_list_str,
                                                format=input0_format_list_str,
                                                unknownshape_format=input0_format_list_str)
        input1 = util_select_op_base.gen_param(classify="input1",
                                                name="x2",
                                                datatype=dtype_list_str,
                                                format=input1_format_list_str,
                                                unknownshape_format=input1_format_list_str)
        input2 = util_select_op_base.gen_param(classify="input2",
                                                name="x3",
                                                datatype=dtype_list_str,
                                                format=input2_format_list_str,
                                                unknownshape_format=input2_format_list_str)
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype=dtype_list_str,
                                                format=output_format_list_str,
                                                unknownshape_format=output_format_list_str)


    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _shape_broadcast(data_1, data_2):
    """
    broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(data_1.shape,
                                                                  data_2.shape,
                                                                  param_name_input1="data_1",
                                                                  param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)

    return data_1, data_2


def _check_format(format_input0, format_input1, format_input2):
    """
    check the format_list
    """
    list_format = [format_input0, format_input1, format_input2]

    nd_format = {"ND", "NHWC", "NCHW", "HWCN"}
    standard_format = []

    for item in list_format:
        if item in nd_format:
            standard_format.append("ND")
        else:
            standard_format.append(item)

    list_pattern = [["FRACTAL_NZ", "ND", "ND"], ["ND", "FRACTAL_NZ", "ND"], ["ND", "ND", "FRACTAL_NZ"],
                    ["FRACTAL_NZ", "ND", "FRACTAL_NZ"]]
    if standard_format in list_pattern:
        format_pattern = list_pattern.index(standard_format) + 1
    else:
        format_pattern = 0

    return format_pattern


def _infer_shape(shape_input0, shape_input1, shape_input2, format_pattern):
    if format_pattern in [1, 2, 3]:
        shape_input0, shape_input1, shape_input2 = \
            _infer_shape_one(shape_input0, shape_input1,
                             shape_input2, format_pattern)
    elif format_pattern == 4:
        shape_input0, shape_input1, shape_input2 = \
            _infer_shape_two(shape_input0, shape_input1,
                             shape_input2, format_pattern)
    elif format_pattern == 5:
        shape_input1, shape_input0, shape_input2 = \
            _infer_shape_two(shape_input1, shape_input0,
                             shape_input2, format_pattern)
    else:
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                        param_name_input2="input1")
        shape_input2, shape_max_mul, shape_max_add0 = \
            shape_util.broadcast_shapes(shape_input2, shape_max_mul, param_name_input1="input2",
                                        param_name_input2="shape_max_mul")
    return [shape_input0, shape_input1, shape_input2]


@register_operator_compute("FusedMulAdd", op_mode="dynamic", support_fusion=True)
def fusion_mul_add_compute(data_input0, data_input1, data_input2, output, kernel_name="fused_mul_add"):
    """
    mul+add calculation function for ub fusion

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul
    data_input2: TVM tensor
         the input tensor of add
    output: TVM tensor
         the output tensor of add
    kernel_name : str
        kernel name, default value is "fused_mul_add"

    Returns
    -------
    output tensor
    """
    batch_matmul_flag_lhs = check_batchmatmul_fuse(data_input0)
    batch_matmul_flag_rhs = check_batchmatmul_fuse(data_input1)

    if batch_matmul_flag_rhs:
        data_input0, data_input1 = data_input1, data_input0
    if "para_name" in data_input0.op.attrs:
        para_name = data_input0.op.attrs["para_name"]
        para_name += "_muladd"
    else:
        para_name = "muladd"
    batch_shape = shape_util.shape_to_list(data_input0.op.attrs["batch_shape"])
    para_dict_1 = {"format_elem": data_input1.op.attrs["format"], "batch_shape": batch_shape}
    para_dict_2 = {"format_elem": data_input2.op.attrs["format"], "batch_shape": batch_shape}

    if batch_matmul_flag_lhs or batch_matmul_flag_rhs:
        data_input1, shape_max = batchmatmul_elem_nd2nz(data_input0, data_input1, para_dict_1, para_name + "1")
        data_input2, _ = batchmatmul_elem_nd2nz(data_input0, data_input2, para_dict_2, para_name + "2")
        data_input1 = tbe.broadcast(data_input1, shape_max)
        data_input2 = tbe.broadcast(data_input2, shape_max)
        data_input1 = batchmatmul_elem_reshape(data_input0, data_input1, batch_shape, para_name + "1")
        data_input2 = batchmatmul_elem_reshape(data_input0, data_input2, batch_shape, para_name + "2")
        mul_result = tbe.vmul(data_input0, data_input1)
        res = tbe.vadd(mul_result, data_input2)
        res.op.attrs["batch_shape"] = batch_shape
        res.op.attrs["para_name"] = para_name
    else:
        res = fused_mul_add_compute(data_input0, data_input1, data_input2, output, kernel_name)

    return res


def fused_mul_add_compute(data_input0, data_input1, data_input2, output, kernel_name="fused_mul_add"):
    """
    mul+add calculation function

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul
    data_input2: TVM tensor
         the input tensor of add
    output: TVM tensor
         the output tensor of add
    kernel_name : str
        kernel name, default value is "fuesd_mul_add"

    Returns
    -------
    output tensor
    """
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product == "Ascend310P":
        x0_shape = shape_util.shape_to_list(data_input0.shape)
        x1_shape = shape_util.shape_to_list(data_input1.shape)
        x2_shape = shape_util.shape_to_list(data_input2.shape)
        x0_shape, x1_shape, x2_shape, shape_max = shape_util.unify_broadcast_shapes(
            [x0_shape, x1_shape, x2_shape]) 
        data_input0 = tbe.broadcast(data_input0, shape_max)
        data_input1 = tbe.broadcast(data_input1, shape_max)
        data_input2 = tbe.broadcast(data_input2, shape_max)
        mul_result = tbe.vmul(data_input0, data_input1)
        # add
        res = tbe.vadd(mul_result, data_input2)
    else:
        data_input0, data_input1 = _shape_broadcast(data_input0, data_input1)
        mul_result = tbe.vmul(data_input0, data_input1)
        # add
        mul_result, data_input2 = _shape_broadcast(mul_result, data_input2)
        res = tbe.vadd(mul_result, data_input2)
    return res


@register_operator("FusedMulAdd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def fused_mul_add(input0, input1, input2, output, kernel_name="fused_mul_add"):
    """
    function: fused for mul+add

    Parameters
    ----------
    input0: dict
         the dict of input of mul, support float16,float32,int32
    input1: dict
         the dict of input of mul, support float16,float32,int32
    input2: dict
         the dict of input of add, support float16,float32,int32
    output: dict
         the dict of output of add, support float16,float32,int32
    kernel_name: str
        cce kernel name, default value is fused_mul_add

    Returns
    -------
    None
    """
    # check dtype
    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()
    check_dtype_list = ["float32", "float16", "int32"]
    para_check.check_dtype(dtype_input0, check_dtype_list, param_name="input0")
    para_check.check_dtype(dtype_input1, check_dtype_list, param_name="input1")
    para_check.check_dtype(dtype_input2, check_dtype_list, param_name="input2")

    # check format
    format_input0 = input0.get("format").upper()
    format_input1 = input1.get("format").upper()
    format_input2 = input2.get("format").upper()
    format_pattern = _check_format(format_input0, format_input1, format_input2)
    if not util_common.is_unknown([input0, input1, input2]):
        shape_input0 = list(shape_util.scalar2tensor_one(input0.get("shape")))
        shape_input1 = list(shape_util.scalar2tensor_one(input1.get("shape")))
        shape_input2 = list(shape_util.scalar2tensor_one(input2.get("shape")))

        shape0, shape1, shape2 = _infer_shape(shape_input0, shape_input1, shape_input2, format_pattern)
        range0 = util_common.gen_range(shape0)
        range1 = util_common.gen_range(shape1)
        range2 = util_common.gen_range(shape2)
        input0["shape"] = shape0
        input0["range"] = range0
        input1["shape"] = shape1
        input1["range"] = range1
        input2["shape"] = shape2
        input2["range"] = range2

    # classify
    ins = classify([input0, input1, input2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input0, _input1, _input2) in ins:
        with tbe.compute():
            shape_input0, shape_input1, shape_input2 = shape_util.variable_shape([_input0, _input1, _input2])

            data_input0 = tvm.placeholder(shape_input0, name="data_input0", dtype=dtype_input0)
            data_input1 = tvm.placeholder(shape_input1, name="data_input1", dtype=dtype_input1)
            data_input2 = tvm.placeholder(shape_input2, name="data_input2", dtype=dtype_input2)

            res = fused_mul_add_compute(data_input0, data_input1, data_input2, output, kernel_name)

            tensor_list = [data_input0, data_input1, data_input2, res]
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
