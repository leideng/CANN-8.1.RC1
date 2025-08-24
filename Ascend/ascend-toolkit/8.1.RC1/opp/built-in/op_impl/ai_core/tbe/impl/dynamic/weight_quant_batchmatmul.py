# Copyright 2023 Huawei Technologies Co., Ltd
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
dynamic weight_quant_batchmatmul
"""
import collections
import math
import warnings
from functools import reduce
from itertools import product
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.util_cube_dynamic import ceil_div
from impl.dynamic.batch_matmul_v2 import define_batch_matmul_var
from tbe.dsl.compute.weight_quant_bmm_compute import weight_quant_bmm_compute
from tbe.common.utils.const import ComputeFlow
from tbe.common.platform import platform_info
from impl.util.util_gemm import reset_format
from impl.util import util_gemm
from tbe.dsl.base.operation import get_te_var
from impl.util.platform_adapter import tbe_register

DIAGONAL_MATRIX_SIZE = 32
OP_TAG = "weight_quant_bmm_"
BLOCK_CUBE = 16
DYNAMIC_FLAG = -1
DYNAMIC_FLAG_UNRANK = [-2]
BATCH_ND_LENGTH = 3
ND_LENGTH = 2
MKN_MIN = 1
MARK_VALUE_INT32 = 0X80000000


def weight_quant_bmm_fuse_compute(tensor_a, tensor_b, tensor_diag,
    tensor_q_bias, tensor_deq_scale, tensor_bias, output_z=None,
    adj_x1=False, adj_x2=False, kernel_name="weight_quant_batchmatmul"):
    para_dict = {
        "tensor_q_bias": tensor_q_bias,
        "deq_scale": tensor_deq_scale,
        "bias": tensor_bias,
        "trans_a": adj_x1,
        "trans_b": adj_x2,
        "format_out": output_z.get("format"),
        "shape_out": output_z.get("shape"),
        "dtype_out": output_z.get("dtype").lower(),
        "kernel_name": kernel_name,
        "batch_a": tensor_a.op.attrs["ori_batch_shape"],
        "batch_b": tensor_b.op.attrs["ori_batch_shape"],
        "op_tag": OP_TAG
    }
    return weight_quant_bmm_compute(tensor_a, tensor_b, tensor_diag, para_dict)


def _get_input_x1_range(range_x1: tuple, adj_x1: bool, op_type: str) -> list:
    """Get range of m, k, batch.

    Args:
        range_x1: range of x1
        adj_x1: whether to swap m, k before matrix multiplication
        op_type: just for log

    Returns:
        A list of range, like [range_m, range_k, range_batch]. when batch does not exist, range_batch is [].
    """
    range_len = BATCH_ND_LENGTH
    if len(range_x1) >= range_len - 1:
        m_index = -2
        k_x1_index = -1
        batch_range_x1 = range_x1[:-2]
    else:
        error_manager_vector.raise_err_specific_reson(op_type, "Lenth of x1_range illegal")
    m_range = list(range_x1[m_index])
    k_range_x1 = list(range_x1[k_x1_index])
    if operation.get_op_context():
        operation.get_op_context().add_addition("batch_range_x1", batch_range_x1)
    if adj_x1:
        m_range, k_range_x1 = k_range_x1, m_range
    return [m_range, k_range_x1, batch_range_x1]


def _get_input_x2_range(range_x2: tuple, adj_x2: bool, op_type: str) -> list:
    """Get range of k, n, batch.

    Args:
        range_x2: range of x1
        adj_x2: whether to swap m, k before matrix multiplication
        op_type: just for log

    Returns:
        A list of range, like [range_m, range_k, range_batch]. when batch does not exist, range_batch is [].
    """
    range_len = BATCH_ND_LENGTH
    if len(range_x2) >= range_len - 1:
        k_x2_index = -2
        n_index = -1
        batch_range_x2 = range_x2[:-2]
    else:
        error_manager_vector.raise_err_specific_reson(op_type, "Lenth of x1_range illegal")
    k_range_x2 = list(range_x2[k_x2_index])
    n_range = list(range_x2[n_index])
    if operation.get_op_context():
        operation.get_op_context().add_addition("batch_range_x2", batch_range_x2)
    if adj_x2:
        k_range_x2, n_range = n_range, k_range_x2
    return [k_range_x2, n_range, batch_range_x2]


def _get_input_range(range_x1: tuple, format_x1: str, range_x2: tuple, format_x2: str, range_bias: tuple,
                     adj_x1: bool, adj_x2: bool, op_type: str) -> list:
    """
    get range in batch, m, k, n and check range
    """
    batch_range_x1, batch_range_x2 = None, None
    if range_x1:
        m_range, k_range_x1, batch_range_x1 = _get_input_x1_range(range_x1, adj_x1, op_type)
    else:
        # NOTE range_x1 is empty only in check of fuzzy generalization
        m_range = [1, None]
        k_range_x1 = [1, None]

    if range_x2:
        k_range_x2, n_range, batch_range_x2 = _get_input_x2_range(range_x2, adj_x2, op_type)
    else:
        # NOTE range_x2 is empty only in check of fuzzy generalization
        k_range_x2 = [1, None]
        n_range = [1, None]

    k_range = _get_range_intersection(k_range_x1, k_range_x2, "k_range")
    if range_bias:
        range_bias_n = list(range_bias[0])
        n_range = _get_range_intersection(n_range, range_bias_n, "n_range")

    # in generalization func of fuzzy compile, only need check. Not add_addition
    batch_range = None
    batch_range = _get_batch_range(batch_range_x1, batch_range_x2)

    return [batch_range, m_range, k_range, n_range]


def _get_range_intersection(range1: list, range2: list, param_name: str) -> list:
    """
    get range intersection of two range
    """
    if range1[1] is None:
        return range2
    if range2[1] is None:
        return range1

    range_ins = [max(range1[0], range2[0]), min(range1[1], range2[1])]
    range_ins = [min(range_ins[0], range_ins[1]), max(range_ins[0], range_ins[1])]
    return range_ins


def _get_batch_range(range_x1: tuple, range_x2: tuple) -> list:
    """Get reduce range of batch.

    Args:
        range_x1: batch range of input x1
        range_x2: batch range of input x2

    Returns:
        A list of reduced range of batch. Returns None when both range_x1 and range_x2 are [] or None.
    """
    if not range_x1 and not range_x2:
        return None

    batch_range = [1, 1]
    range_x = []
    if range_x1 and range_x2:
        for range_mem1, range_mem2 in zip(range_x1, range_x2):
            range_ins = _get_range_intersection(range_mem1, range_mem2, "batch_range")
            range_x.append(range_ins)
    elif range_x2:
        range_x = range_x2
    else:
        range_x = range_x1

    for range_mem in range_x:
        if range_mem[1] is None:
            batch_range = [1, None]
            break
        else:
            batch_range[0] = batch_range[0] * range_mem[0]
            batch_range[1] = batch_range[1] * range_mem[1]

    return batch_range


def _check_args(args: tuple, expect_args: list, msg: str) -> None:
    """
    check args
    """
    if args not in expect_args:
        error_manager_vector.raise_err_input_format_invalid(
            "batch_matmul", msg, expect_args, args)


def check_and_config_para(input_x1: dict, input_x2: dict, bias: dict, output_z: dict,
                          adj_x1: bool, adj_x2: bool, kernel_name: str, op_type: str) -> tuple:
    """
    check and config dynamic mode
    """
    # get format and dtype
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    format_out = output_z.get("format")
    dtype_a = input_x1.get("dtype").lower()
    dtype_b = input_x2.get("dtype").lower()
    dtype_out = output_z.get("dtype").lower()

    # check kernel_name dtype and format
    para_check.check_kernel_name(kernel_name)
    expect_input_format_a = ['ND']
    expect_input_format_b = ['ND']
    expect_out_format = ['ND']
    expect_args = list(product(expect_input_format_a, ['float16'],
                               expect_input_format_b, ['int8'],
                               expect_out_format, ['float16']))
    _check_args((format_a, dtype_a, format_b, dtype_b, format_out, dtype_out),
                expect_args, "format_a, dtype_a, format_b, dtype_b, format_out, dtype_out")

    # check bias if bias in not None
    if bias:
        dtype_bias = bias.get("dtype")
        para_check.check_dtype_rule(dtype_bias, ("float16", "float32"), "bias")

    # get range and ori_shape
    shape_input, range_input = _get_dynamic_shape_and_range(input_x1, input_x2, bias, op_type)
    shape_x1, shape_x2 = shape_input
    range_x1, range_x2, range_bias = range_input

    # check dynamic mode
    if op_type in (OP_TAG):
        _check_dynamic_mode_of_batch_matmul(shape_x1, shape_x2)
    else:
        reason = f"not support op_type: {op_type}"
        error_manager_vector.raise_err_specific_reson(op_type, reason)

    # get range in m,k,n
    input_range = _get_input_range(range_x1, format_a,
                                   range_x2, format_b,
                                   range_bias, adj_x1, adj_x2, op_type)

    is_cache_tiling = util_gemm.get_cache_tiling_flag(input_range, bias, dtype_out)
    input_range = [[1, None] for i in input_range]
    ori_input_range = input_range
    return [is_cache_tiling, input_range, ori_input_range, shape_input]


def _get_batch_matmul_shape_and_range(input_x: dict, input_y: dict) -> list:
    """
    Get range in uniform format.

    shape: ((1, None), (1, None), (1, None))
    static shape, empty range: adjust the upper and lower bounds of range to be consistent with dim.
    static dim, upper and lower bounds of range is differ: adjust upper and lower of bounds to be consistent with dim.
    dim is 0 in shape: adjust the range to start at 1.

    Args:
        input_x: dict of input x1
        input_y: dict of input x2

    Returns:
        A list of [shape_x, range_x, shape_y, range_y]
    """
    shape_x = input_x.get("ori_shape")
    shape_y = input_y.get("ori_shape")
    range_x = input_x.get("range")
    range_y = input_y.get("range")
    format_x = input_x.get("format")
    format_y = input_y.get("format")
    if (format_x != "ND") or (format_y != "ND"):
        error_manager_vector.raise_err_specific_reson(
            "weight_quant_bmm", "dynamic input shape must be ND"
        )
    range_nd = ((1, None), (1, None), (1, None))
    if list(shape_x) == DYNAMIC_FLAG_UNRANK:
        shape_x = (-1, -1, -1)
        range_x = range_nd
    if list(shape_y) == DYNAMIC_FLAG_UNRANK:
        shape_y = (-1, -1, -1)
        range_y = range_nd
    range_x = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_x)
    range_y = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_y)
    return [list(shape_x), range_x, list(shape_y), range_y]


def _check_dynamic_mode_of_batch_matmul(shape_x1: tuple, shape_x2: tuple) -> None:
    """
    check dynamic mode
    """
    if len(shape_x1) < BATCH_ND_LENGTH - 1:
        error_manager_vector.raise_err_input_shape_invalid(
            "batch_matmul", "x1", "ori_shape dim must more than 1"
        )

    if len(shape_x2) < BATCH_ND_LENGTH - 1:
        error_manager_vector.raise_err_input_shape_invalid(
            "batch_matmul", "x2", "ori_shape dim must more than 1"
        )

    if all(i != DYNAMIC_FLAG for i in shape_x1) and all(i != DYNAMIC_FLAG for i in shape_x2):
        error_manager_vector.raise_err_specific_reson(
            "batch_matmul", "dynamic must at least one of batch, m, k, n"
        )


def _get_dynamic_shape_and_range(input_x1: dict, input_x2: dict, bias: dict, op_type: str) -> tuple:
    """
    get the shape and range of matmul
    """
    bias_range = None
    if op_type in (OP_TAG):
        shape_x1, range_x1, shape_x2, range_x2 = _get_batch_matmul_shape_and_range(
            input_x1, input_x2)
    else:
        reason = f"not support op_type: {op_type}"
        error_manager_vector.raise_err_specific_reson(op_type, reason)

    if bias:
        bias_range = bias.get("range")

    return [shape_x1, shape_x2], [range_x1, range_x2, bias_range]


def _set_shape(format_a: str, format_b: str) -> tuple:
    """
    Set input shape
    """
    m_var = get_te_var("m").get_tvm_var()
    ka_var = get_te_var("k").get_tvm_var()
    kb_var = get_te_var("k").get_tvm_var()
    n_var = get_te_var("n").get_tvm_var()

    m_ori_var = get_te_var("m_ori").get_tvm_var()
    k_ori_var = get_te_var("k_ori").get_tvm_var()
    n_ori_var = get_te_var("n_ori").get_tvm_var()
    if format_a == "ND":
        m_var = m_ori_var
        ka_var = k_ori_var
    if format_b == "ND":
        n_var = n_ori_var
        kb_var = k_ori_var
    return [ka_var, m_var, n_var, kb_var]


def _get_m_k_index(format_a: str, adj_x1: bool) -> list:
    """
    get the correct m, k position for shape_x1.
    """
    if adj_x1:
        m_index = -1 if format_a == "ND" else -2
        k_index = -2 if format_a == "ND" else -1
    else:
        m_index = -2 if format_a == "ND" else -1
        k_index = -1 if format_a == "ND" else -2
    return [m_index, k_index]


def _get_k_n_index(format_b: str, adj_x2: bool) -> list:
    """
    get the correct k, n position for shape_x2.
    """
    if adj_x2:
        n_index = -2 if format_b in ("ND", "FRACTAL_Z") else -1
        k_index = -1 if format_b in ("ND", "FRACTAL_Z") else -2
    else:
        n_index = -1 if format_b in ("ND", "FRACTAL_Z") else -2
        k_index = -2 if format_b in ("ND", "FRACTAL_Z") else -1
    return [k_index, n_index]


def _construct_shape_x1_x2(info_x1_x2):
    shape_x1, shape_x2 = info_x1_x2.get("shapes")
    format_x1, format_x2 = info_x1_x2.get("formats")
    adj_x1, adj_x2 = info_x1_x2.get("attrs")

    # NOTE Only the batches of a and b are the same or the batch dimensions of a and b are different, and
    #      full broadcast to a or b.
    #      the case1 is (b1, b2, x, x) (b1, b2, x, x)
    #      the case2 is (2, 3, x, x) (x, x)
    if len(shape_x1) >= BATCH_ND_LENGTH:
        shape_x1 = [get_te_var("batch").get_tvm_var(), DYNAMIC_FLAG, DYNAMIC_FLAG]
    if len(shape_x2) >= BATCH_ND_LENGTH:
        shape_x2 = [get_te_var("batch").get_tvm_var(), DYNAMIC_FLAG, DYNAMIC_FLAG]

    m_index, ka_index = _get_m_k_index(format_x1, adj_x1)
    kb_index, n_index = _get_k_n_index(format_x2, adj_x2)
    shape_x1[ka_index], shape_x1[m_index], shape_x2[n_index], shape_x2[kb_index] = _set_shape(format_x1,
        format_x2)
    return shape_x1, shape_x2


def define_cache_tiling_var(input_x1: dict, input_x2: dict) -> None:
    """
    define variables in cache tiling
    """
    def _create_var(list_var):
        for name_var in list_var:
            operation.var(name_var)

    list_var = ("batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim", "m_dim", "k_dim",
                "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor",
                "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch_l1_factor", "batch_ub_l0_time", "batch_cub",
                "out_branch_flag", "bias_flag", "hf32_flag", "datatype_bf16", "al1_db", "bl1_db", "l0c_db",
                "l2_cache_flag", "close_k_shift")
    _create_var(list_var)


def _get_bias_tensor(bias: dict):
    """
    Get Bias Tensor
    """
    if bias:
        var_n = get_te_var("n_ori").get_tvm_var()
        bias_dtype = bias.get("dtype")
        bias_shape = [var_n]
        tensor_bias = tvm.placeholder(
            bias_shape, name="bias", dtype=bias_dtype, attrs={'ori_shape': bias_shape})
    else:
        tensor_bias = None
    return tensor_bias


def _create_placeholder(input_dict, in_shape, name):
    in_dtype = input_dict.get("dtype").lower()
    attrs = {
        'format': input_dict.get("format"),
        'ori_format': input_dict.get("ori_format"),
        'ori_shape': in_shape,
        'const_value': input_dict.get("const_value", [])
    }
    in_tensor = tvm.placeholder(in_shape, name=name,
                                dtype=in_dtype, attrs=attrs)
    return in_tensor


def _reset_shape(input_dict, shape_x, in_shape):
    batch_shape = get_te_var("batch").get_tvm_var()
    if input_dict["format"] == "ND":
        batch_shape = reduce(lambda x, y: x * y, in_shape)
        in_shape = [batch_shape, ] + list(shape_x[-2:])
    return in_shape


def _creat_deq_scale_tensor(deq_scale, pre_conv_mode):
    n = get_te_var("n").get_tvm_var()
    block_n0 = int(platform_info.get_soc_spec("cube_n_size"))
    if pre_conv_mode == "S322F16":
        deq_s_ori_shape = (1, 1, 1, 1)
        deq_s_shape = (1, 1, 1, 1, block_n0)
        tensor_deq_scale = tvm.placeholder(deq_s_shape, name="tensor_deq_scale", dtype=deq_scale.get("dtype"),
                                           attrs={"ori_shape": deq_s_ori_shape})
    else:
        n_ori = get_te_var("n_ori").get_tvm_var()
        n = get_te_var("n").get_tvm_var()
        deq_s_ori_shape = (1, 1, 1, n_ori)
        deq_s_shape = (1, n, 1, 1, block_n0)
        tensor_deq_scale = tvm.placeholder(deq_s_shape, name="tensor_deq_scale", dtype=deq_scale.get("dtype"),
                                           attrs={'ori_shape': deq_s_ori_shape})
    return tensor_deq_scale


def get_all_shape(info_x1_x2, input_x, input_y):
    shape_x1_temp, shape_x2_temp = _construct_shape_x1_x2(info_x1_x2) # [batch, m, k_ori], [batch, k_ori, n]

    shape_diag = [DIAGONAL_MATRIX_SIZE, DIAGONAL_MATRIX_SIZE]
    shape_q_bias = [get_te_var("n_ori").get_tvm_var()]
    shape_bias = [get_te_var("n_ori").get_tvm_var()]

    ori_batch_x1 = [get_te_var("batch_a1").get_tvm_var(), get_te_var("batch_a2").get_tvm_var(),
                    get_te_var("batch_a3").get_tvm_var(), get_te_var("batch_a4").get_tvm_var()]
    ori_batch_x2 = [get_te_var("batch_b1").get_tvm_var(), get_te_var("batch_b2").get_tvm_var(),
                        get_te_var("batch_b3").get_tvm_var(), get_te_var("batch_b4").get_tvm_var()]
    shape_x1 = _reset_shape(input_x, shape_x1_temp, ori_batch_x1)
    shape_x2 = _reset_shape(input_y, shape_x2_temp, ori_batch_x2)
    shapes = (shape_x1, shape_x2, shape_diag, shape_q_bias, shape_bias, ori_batch_x1, ori_batch_x2)
    return shapes


def get_bmm_var_params(input_x, input_y):
    op_type_bmm = "BatchMatMulV2"
    is_cache_tiling = True
    extra_params = {"op_type": "BatchMatMulV2"}
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if support_l0c2out and input_x.get("format") == "ND" and input_y.get("format") == "ND":
        extra_params["nd2nz_type"] = ComputeFlow.on_the_fly.value
    return op_type_bmm, extra_params, is_cache_tiling


def weight_quant_batchmatmul(input_x, input_y, diagonal_matrix,
                             q_bias, deq_scale, bias=None, output_z=None,
                             adj_x1=False, adj_x2=False,
                             kernel_name="weight_quant_batchmatmul"):
    tensor_lists = []
    schs = []
    op_type = OP_TAG
    _, input_range, ori_input_range, shape_input = check_and_config_para(
        input_x, input_y, bias, output_z, adj_x1, adj_x2, kernel_name, op_type)
    op_type_bmm, extra_params, is_cache_tiling = get_bmm_var_params(input_x, input_y)
    define_batch_matmul_var(op_type_bmm, extra_params, is_cache_tiling, input_range, ori_input_range)
    define_cache_tiling_var(input_x, input_y)
    format_a = input_x.get("format")
    format_b = input_y.get("format")
    info_x1_x2 = {"shapes": shape_input, "formats": (format_a, format_b), "attrs": (adj_x1, adj_x2)}
    (shape_x1, shape_x2, shape_diag, shape_q_bias, shape_bias, ori_batch_x1, ori_batch_x2) = get_all_shape(
        info_x1_x2, input_x, input_y)
    tensor_a = _create_placeholder(input_x, shape_x1, "tensor_a")
    tensor_b = _create_placeholder(input_y, shape_x2, "tensor_b")
    tensor_diag = _create_placeholder(diagonal_matrix, shape_diag, "tensor_diag")
    tensor_q_bias = _create_placeholder(q_bias, shape_q_bias, "tensor_q_bias")
    tensor_bias = _get_bias_tensor(bias)
    tensor_a.op.attrs["ori_batch_shape"] = ori_batch_x1
    tensor_b.op.attrs["ori_batch_shape"] = ori_batch_x2
    for pre_conv_mode in ["S322F16", "VS322F16"]:
        tensor_deq_scale = _creat_deq_scale_tensor(deq_scale, pre_conv_mode)
        tensor_list = [tensor_a, tensor_b, tensor_diag, tensor_q_bias, tensor_deq_scale]
        if bias:
            tensor_list.append(tensor_bias)
        with tbe.compute():
            out = weight_quant_bmm_fuse_compute(tensor_a, tensor_b, tensor_diag, tensor_q_bias, tensor_deq_scale,
                                                tensor_bias, output_z, adj_x1, adj_x2, kernel_name)
        with tvm.target.cce():
            sch = tbe.auto_schedule(out)
        tensor_list.append(out)
        tensor_lists.append(tensor_list)
        schs.append(sch)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_lists,
        "build_args": {"constant_realize_extent_in_infer_bound": False,
                       "enable_db_fold":True,
                       "InjectSync":{"sync_opt_for_notail_db": 1,
                                    "sync_opt_for_preload_loop_zero": True}}
    }
    tbe.build(schs, config)


@tbe_register.register_param_generalization("WeightQuantBatchmatmul")
def weight_quant_batch_matmul_generalization(input_x,
                                             input_y,
                                             diagonal_matrix,
                                             q_bias,
                                             deq_scale,
                                             bias=None,
                                             output_z=None,
                                             adj_x1=False,
                                             adj_x2=False,
                                             kernel_name="weight_quant_batchmatmul",
                                             generalize_config=None):
    result = []
    if generalize_config.get("mode") == "all_shape":
        input_x.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        input_y.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        diagonal_matrix.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        q_bias.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        deq_scale.update({"ori_shape": [-2], "ori_format": "NHWC", "shape": [-2], "format": "NC1HWC0"})
        if bias is not None:
            bias.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        output_z.update({"ori_shape": [-2], "ori_format": "ND", "shape": [-2], "format": "ND"})
        result.append([input_x, input_y, diagonal_matrix, q_bias, deq_scale, bias, output_z, adj_x1, adj_x2])

    return result