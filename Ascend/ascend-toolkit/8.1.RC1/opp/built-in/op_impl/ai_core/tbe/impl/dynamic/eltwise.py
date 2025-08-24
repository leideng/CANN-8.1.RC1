# Copyright 2021 Huawei Technologies Co., Ltd
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
eltwise
"""


from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import get_current_build_config
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_binary import BinaryMatchBase
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


VAR_ATTR_COEFF = [-2.0]
ATTR_COEFF = OpAttr(1, "coeff", "list_float")


# 'pylint: disable=unrecognized-inline-option,invalid-name,too-many-locals,unused-argument
def get_fusion_params(x_tensor, y, x_tensor_num):
    """
    Get L1 fusion_params
    Parameters
    ----------
    x_tensor : tensor of input data
    y : dict of output data
    x_tensor_num: input tensor num
    Returns
    -------
    fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    in_l1_flag_list = []
    in_valid_shape_list = []
    in_slice_offset_list = []
    in_select_read_flag_list = []
    is_l1_depth_fusion = False

    for i in range(0, x_tensor_num):
        l1_fusion_type = -1
        if not get_current_build_config("enable_op_prebuild"):
            l1_fusion_type = \
                x_tensor[i].op.attrs["L1_fusion_type"].value if "L1_fusion_type" in x_tensor[i].op.attrs else -1
            if l1_fusion_type == 1:
                error_manager_vector.raise_err_specific_reson("eltwise", "eltwise does not support l1 width fusion")
        is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
        in_l1_flag = x_tensor[i].op.attrs["addr_type"].value == 1 if "addr_type" in x_tensor[i].op.attrs else False
        in_l1_flag_list.append(in_l1_flag)
        in_valid_shape = x_tensor[i].op.attrs["valid_shape"] if "valid_shape" in x_tensor[i].op.attrs else []
        in_valid_shape_list.append(in_valid_shape)
        in_slice_offset = x_tensor[i].op.attrs["slice_offset"] if "slice_offset" in x_tensor[i].op.attrs else []
        in_slice_offset_list.append(in_slice_offset)
        in_select_read_flag = x_tensor[i].op.tag == "read_select_5d"
        in_select_read_flag_list.append(in_select_read_flag)

    l1_fusion_type = 0 if is_l1_depth_fusion is True else -1
    if l1_fusion_type != -1 and y.get("format").upper() != 'NC1HWC0':
        shape_rule = "the input format must be 5HD when l1 fusion"
        error_manager_vector.raise_err_check_params_rules("eltwise", shape_rule, "x",
                                                          y.get("format").upper())

    out_l1_flag = False
    out_valid_shape = []
    out_slice_offset = []
    out_select_write_flag = False
    if y is not None:
        out_l1_flag = y.get("addr_type", 0) == 1
        out_valid_shape = y.get("valid_shape", [])
        out_slice_offset = y.get("slice_offset", [])
        out_select_write_flag = bool(out_valid_shape)

    fusion_params = {"is_l1fusion": is_l1_depth_fusion,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag_list,
                     "in_select_read_flag": in_select_read_flag_list,
                     "in_valid_shape": in_valid_shape_list,
                     "in_slice_offset": in_slice_offset_list,
                     "out_l1_flag": out_l1_flag,
                     "out_select_write_flag": out_select_write_flag,
                     "out_valid_shape": out_valid_shape,
                     "out_slice_offset": out_slice_offset}
    return fusion_params


def fuzz_all_shape(in_tensor):
    """
    generalize_all_shape
    """
    in_tensor["shape"] = [-2]
    in_tensor["ori_shape"] = [-2]
    in_tensor["format"] = "ND"
    in_tensor["ori_format"] = "ND"
    return in_tensor


# 'pylint: disable=too-many-arguments
@tbe_register.register_param_generalization("Eltwise")
def eltwise_generalization(x, y, mode, coeff, kernel_name="eltwise", generalize_config=None):
    """
    eltwise_generalization
    """
    if generalize_config.get(BinaryMatchBase.GENERALIZATIO_MODE_KEY_NAME,
                             "None") != BinaryMatchBase.GENERALIZATIO_MODE_BINARY:
        return None

    x = list(x)
    for i, input_tensor in enumerate(x):
        x[i] = fuzz_all_shape(input_tensor)
    y = fuzz_all_shape(y)

    if len(coeff) == 0:
        coeff = []
    else:
        coeff = VAR_ATTR_COEFF

    return [[x, y, mode, coeff]]


# 'pylint: disable=unidiomatic-typecheck,too-many-branches,too-many-locals
# 'pylint: disable=no-member,dangerous-default-value,invalid-name,len-as-condition
@register_operator_compute("Eltwise", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def eltwise_compute(x, y, mode=1, coeff=[], kernel_name="eltwise"):
    """
    Compute elementwise operation
    """
    tensor_num_d = len(x)
    inp_dtype_d = x[0].dtype
    data0_tmp_d = x[0]
    is_binary = True if coeff is None else False

    tmp_y = {
        "addr_type": 0,
        "valid_shape": [],
        "slice_offset": []
    }
    fuse_y = tmp_y if y is None else y
    fusion_params = get_fusion_params(x, fuse_y, tensor_num_d)
    case = 0  # depthwise_con2d fusion flag

    if coeff is None:
        actual_dtype = inp_dtype_d + "[{0}]".format(tensor_num_d)
        coeff_var = get_attr_by_cls(coeff, ATTR_COEFF, actual_dtype)
        data0_tmp_d = tbe.vmuls(data0_tmp_d, coeff_var[0])

    if mode == 1 and not is_binary:
        if len(coeff) != 0 and len(coeff) != tensor_num_d:
            error_manager_vector.raise_err_specific_reson("eltwise",
                                                          "the parameter coeff's length not equal to inputs'num")
        if len(coeff) == tensor_num_d:
            if type(coeff[0]) != int and type(coeff[0]) != float:
                error_manager_vector.raise_err_specific_reson("eltwise", "ele of coeff must be a number.")
            if coeff[0] != 1:
                coeff1 = tvm.const(coeff[0], dtype=inp_dtype_d)
                data0_tmp_d = tbe.vmuls(data0_tmp_d, coeff1)

    res_d = None
    if tensor_num_d == 1:
        const_val_0 = tvm.const(0, dtype=inp_dtype_d)
        data0_tmp_d = tbe.vadds(data0_tmp_d, const_val_0)
        res_d = data0_tmp_d
    elif tensor_num_d > 1:
        for i in range(1, tensor_num_d):
            datan_tmp = x[i]
            if mode == 0:
                data0_tmp_d = tbe.vmul(data0_tmp_d, datan_tmp)
                case = "eltwise_case_0"
            elif mode == 2:
                data0_tmp_d = tbe.vmax(data0_tmp_d, datan_tmp)
                case = "eltwise_case_2"
            else:
                if coeff is None:
                    datan_tmp = tbe.vmuls(datan_tmp, coeff_var[i])
                    data0_tmp_d = tbe.vadd(data0_tmp_d, datan_tmp)
                    case = "eltwise_case_1_3"
                elif len(coeff) == 0:
                    data0_tmp_d = tbe.vadd(data0_tmp_d, datan_tmp)
                    case = "eltwise_case_1_1"
                elif coeff[i] == 1:
                    data0_tmp_d = tbe.vadd(data0_tmp_d, datan_tmp)
                    case = "eltwise_case_1_1"
                else:
                    coeff2 = tvm.const(coeff[i], dtype=inp_dtype_d)
                    datan_tmp = tbe.vmuls(datan_tmp, coeff2)
                    data0_tmp_d = tbe.vadd(data0_tmp_d, datan_tmp)
                    case = "eltwise_case_1_2"
        res_d = data0_tmp_d

    res_d.op.attrs["ele_fusion_params"] = fusion_params
    res_d.op.attrs["L1_fusion_type"] = fusion_params.get("l1_fusion_type")
    if case:
        res_d.op.attrs["eltwise_case"] = case

    return res_d


def _eltwise_check_para(x, y, mode=1, coeff=[],
                        kernel_name="eltwise"):
    """
    check para dtype and shape
    """

    shape = x[0].get("shape")
    dtype = x[0].get("dtype").lower()
    para_check.check_shape(shape, param_name="x")

    dtype_check_list = ["float16", "float32", "bfloat16"]
    para_check.check_dtype(dtype, dtype_check_list, param_name="x")

    tensor_num = len(x)
    if tensor_num < 1 or tensor_num > 32:
        error_manager_vector.raise_err_input_param_range_invalid("eltwise", "tensor_num", "32", "1", tensor_num)

    # all input data should be same shape and dtype
    if tensor_num > 1:
        for i in range(1, tensor_num):
            dtype_tmp = x[i].get("dtype").lower()
            if dtype_tmp != dtype:
                error_manager_vector.raise_err_inputs_dtype_not_equal("eltwise", "dtype_tmp", "dtype",
                                                                      str(dtype_tmp), str(dtype))

    dtype_output = y.get("dtype").lower()
    if dtype_output != dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("eltwise", "dtype_output", "dtype",
                                                              str(dtype_output), str(dtype))

    # mode type must be 0, 1 or 2
    op_list = (0, 1, 2)
    if mode not in op_list:
        error_manager_vector.raise_err_check_params_rules("eltwise", "mode only support 0,1,2", "mode", mode)


@register_operator("Eltwise")
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.KERNEL_NAME)
def eltwise(x, y, mode=1, coeff=[], kernel_name="eltwise"):
    """
    Compute elementwise modes, such as 0:PRODUCT, 1:SUM and 2:MAX
    Parameters
    ----------
    x : the list of input data, it's element is dict:{"shape":[], "dtype":""}
    y : the dict of output
    mode : 0:product,1:sum,2:max;default is 1:sum.
    coeff : input_num should be equal with coeff size.
    kernel_name : cce kernel name, default value is "eltwise"
    Returns
    -------
    None
    """
    tensor_num = len(x)
    _eltwise_check_para(x, y, mode=mode,
                        coeff=coeff, kernel_name=kernel_name)
    dtype = x[0].get("dtype").lower()

    redundant_coe = 1 if tensor_num > 2 else 0

    # coeff is [-2.0] and the shape of input is unknown, this is a binary scenario.
    is_var_attr = (list(coeff) == VAR_ATTR_COEFF and is_unknown_rank_input(x))
    coeff = None if is_var_attr else coeff

    ins = classify(x, OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for _x in ins:
        with tbe.compute():
            shape_normlize = shape_util.variable_shape(_x)
            tlist = []
            is_l1_depth_fusion = False
            for (i, input_i), shape_i in zip(enumerate(_x), shape_normlize):
                l1_fusion_type = -1
                if not get_current_build_config("enable_op_prebuild"):
                    l1_fusion_type = input_i.get("L1_fusion_type", -1)
                    if l1_fusion_type == 1:
                        error_manager_vector.raise_err_specific_reson("eltwise",
                                                                      "eltwise does not support l1 width fusion")
                is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
                addr_type = input_i.get("addr_type", 0)
                valid_shape = input_i.get("valid_shape", [])
                slice_offset = input_i.get("slice_offset", [])
                attr_x = {"addr_type": addr_type,
                          "valid_shape": valid_shape,
                          "slice_offset": slice_offset,
                          "L1_fusion_type": l1_fusion_type}
                datan_tmp = tvm.placeholder(shape_i, name="data_%d" % i, dtype=dtype, attrs=attr_x)
                tlist.append(datan_tmp)
            res = eltwise_compute(tlist, y, mode, coeff, kernel_name)
            tensors.append(tlist)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res, {"redundant_coe": redundant_coe})
        schedules.append(sch)

    tlist.append(res)
    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": tensors,
              "l1_fusion_option": is_l1_depth_fusion}
    tbe.build(schedules, config)
