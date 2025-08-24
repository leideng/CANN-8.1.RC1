# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org//licenses//LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
unsorted_segment_sum

"""
# 'pylint: disable=too-many-lines
from functools import reduce
from mimetypes import init
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util import util_common
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl import constant_util as constant


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    # fp32 select key
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN = 1
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE = 2
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN = 4
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E = 5
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E = 6
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MODIFY = 7
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI = 8
    SELECT_KEY_MODE_FP32_INPUT_NUM_SEGMENT_ONE = 17
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_HP = 18
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_HP = 19
    SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_HP_PAD = 20

    DTYPE_FP32 = "float32"
    DTYPE_INT32 = "int32"
    DTYPE_UINT32 = "uint32"
    DTYPE_INT64 = "int64"
    TILING_PARAM_DTYPE = DTYPE_INT32

    # max_int32
    MAX_INT32 = 2**31 - 1

    # fp32 byte
    BYTE_FP32 = 4

    # int32 byte
    BYTE_INT32 = 4

    # int64 byte
    BYTE_INT64 = 8

    # full mask for fp32
    MASK_FP32 = 64

    # full mask for fp32
    MASK_8 = 8

    # full mask for int32
    MASK_INT32 = 64

    # byte of one block
    BYTE_BLOCK = 32

    # byte of one repeat block
    BYTE_REPEAT_BLOCK = 256

    # max repeat time of vector instruction
    MAX_REPEAT_TIME = 255

    # min ids nums in data move
    MIN_IDS_NUMS = BYTE_BLOCK // BYTE_FP32

    # cloud block num
    CLOUD_CORE_NUM = 32

    # min_tensor_ele_num
    MIN_TENSOR_ELE_NUM = 32

    # tiling params num
    TILING_PARAMS_NUM = 128

    # fp32 ele num one ub block
    ELE_NUM_ONE_BLOCK_FP32 = BYTE_BLOCK // BYTE_FP32

    # modify last axis one, multi times
    MULTI = 4

    # cache ids simplifing number
    CACHE_ACT_SIMPLING_NUM = 32

    # cache ids simplifing buffer number
    CACHE_ACT_SIMPLING_BUFF_NUM = 64

    # optimization method tpye
    INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM = 1
    INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_SIMPLING = 2

    # one row max cache size
    INPUT_LAST_AXIS_ONE_ROW_MAX_CACHE_SIZE = 1024

    # num fp32
    ONE = 1.0
    NEG_ONE = -1.0
    ZERO = 0.0

    BYTE_DTYPE = {"float32": 4, "float16": 2, "int32": 4, "int64": 8, "int8": 1, "uint8": 1}
    IMPL_MODE = {'high_performance': 1}


# 'pylint: disable=invalid-name,too-many-instance-attributes,too-many-arguments,too-many-statements
# 'pylint: disable=too-many-locals,too-few-public-methods,unused-argument
def _ceil_div(val, block):
    """
    compute ceil div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    ceil value
    """
    return (val + block - 1) // block


def _floor(val, block):
    """
    compute floor div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    floor value
    """
    return val // block * block


def _div(val, block):
    """
    compute front part and last part in ceil div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    front_part_num: front part in ceil div
    last_part: last part in ceil div
    """
    front_part_num = val // block
    last_part = val - front_part_num * block
    return front_part_num, last_part


# 'pylint: disable=too-many-arguments
def op_select_format(x,
                     segment_ids,
                     num_segments,
                     y,
                     kernel_name="unsorted_segment_sum",
                     impl_mode=OpImplMode.HIGH_PRECISION):
    """
    select format dynamically
    """
    segment_ids_shape = segment_ids.get("shape")
    check_vbi_supported = tbe_platform.api_check_support("tik.vbi")
    move_pad = tbe_platform.api_check_support("tik.data_move_pad")
    is_unknown_rank = util_common.is_unknown_rank_input([x, segment_ids, num_segments, y])
    if not check_vbi_supported:
        input0_dtype = "float32,int32,float32,int32,float32,int32,float32,int32"
        input0_format = "ND,ND,ND,ND,ND,ND,ND,ND"
        input1_dtype = "int32,int32,int32,int32,int64,int64,int64,int64"
        input1_format = "ND,ND,ND,ND,ND,ND,ND,ND"
        input2_dtype = "int32,int32,int64,int64,int32,int32,int64,int64"
        input2_format = "ND,ND,ND,ND,ND,ND,ND,ND"
    elif check_vbi_supported and not move_pad:
        input0_dtype = "float32,int32,float16,float32,int32,float16,float32,int32,float16,float32,int32,float16"
        input0_format = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
        input1_dtype = "int32,int32,int32,int64,int64,int64,int64,int64,int64,int32,int32,int32"
        input1_format = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
        input2_dtype = "int32,int32,int32,int64,int64,int64,int32,int32,int32,int64,int64,int64"
        input2_format = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
    else:
        input0_dtype = "float32,int32,float16,float32,int32,float16,float32,int32,float16,float32,int32,float16,"\
                       "bfloat16,bfloat16,bfloat16,bfloat16"
        input0_format = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
        input1_dtype = "int32,int32,int32,int64,int64,int64,int64,int64,int64,int32,int32,int32,"\
                       "int32,int64,int32,int64"
        input1_format = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
        input2_dtype = "int32,int32,int32,int64,int64,int64,int32,int32,int32,int64,int64,int64,"\
                       "int32,int64,int64,int32"
        input2_format = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"

    input0 = gen_param(classify="input0",
                       name="x",
                       datatype=input0_dtype,
                       format=input0_format,
                       unknownshape_format=input0_format)
    input1 = gen_param(classify="input1",
                       name="segment_ids",
                       datatype=input1_dtype,
                       format=input1_format,
                       unknownshape_format=input1_format)
    input2 = gen_param(classify="input2",
                       name="num_segments",
                       datatype=input2_dtype,
                       format=input2_format,
                       unknownshape_format=input2_format)
    output0 = gen_param(classify="output0",
                        name="y",
                        datatype=input0_dtype,
                        format=input0_format,
                        unknownshape_format=input0_format)

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=too-many-return-statements,too-many-branches,too-many-arguments
def check_supported(x,
                    segment_ids,
                    num_segments,
                    y,
                    kernel_name="unsorted_segment_sum",
                    impl_mode=OpImplMode.HIGH_PRECISION):
    """
    dynamic -2 not support
    dynamic -1 support
    segment_ids int64 not support
    static shape x_shape ends with 1 or lens equals 1 not support
    temporary support x_dtype of "float32" in compilestatic process
    """
    id_dtype = segment_ids.get("dtype").lower()
    x_dtype = x.get("dtype").lower()
    num_segment_dtype = num_segments.get("dtype").lower()

    if id_dtype not in ("int32", "int64") or num_segment_dtype not in ("int32", "int64"):
        reason = "the segment_ids's or num_segments's dytpe not equeal int32 or int64, segment_ids_dtype=%s, "\
        "num_segment_dtype=%s" % (id_dtype, num_segment_dtype)
        return False, reason

    if x_dtype not in ("float32", "float16", "int32", "bfloat16"):
        reason = "not support this x_dtype, x_dtype=%s" % x_dtype
        return False, reason
    return True, ""


class UnsortedSegmentSum():
    """
        Function: use to store concat base parameters
        Modify : 2020-12-9
    """

    def __init__(self,
                 x_dict,
                 segment_ids_dict,
                 num_segments_dict,
                 y_dict,
                 kernel_name,
                 opname="unsort_segment_sum",
                 impl_mode=OpImplMode.HIGH_PRECISION):
        """
        constructor of class UnsortedSegmentSum

        Parameters
        ----------
        x_dict: dict
            shape and dtype of x
        segment_ids_dict: dict
            shape and dtype of segment_ids
        num_segments_dict: dict
            shape and dtype of num_segments
        y_dict: dict
            shape and dtype of y
        kernel_name: str
            kernel_name, default value is "UnsortedSegmentSum"

        Returns
        -------
        None
        """
        # get dtype
        self.input_dtype = x_dict.get("dtype", None)
        self.input_dtype = self.input_dtype.lower()
        self.ids_dtype = segment_ids_dict.get("dtype", None)
        self.ids_dtype = self.ids_dtype.lower()
        self.num_segments_dtype = num_segments_dict.get("dtype", None)
        self.num_segments_dtype = self.num_segments_dtype.lower()
        self.output_dtype = self.input_dtype
        self.fp32_ele_num_one_block = Constant.ELE_NUM_ONE_BLOCK_FP32
        self.is_double_buffer = False
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.ub_size = _tik_get_ub_size(self.is_double_buffer)
        self.core_num = _tik_get_core_num()
        self.opname = opname
        self.impl_mode = impl_mode
        if self.input_dtype == Constant.DTYPE_FP32:
            self.ub_tensor_num = 3

        self.shape_x = x_dict.get("ori_shape")
        self.shape_segment_ids = segment_ids_dict.get("ori_shape")
        if self.opname == "unsorted_segment_sum":
            self.shape_num_segments = num_segments_dict.get("ori_shape")
        else:
            self.shape_num_segments = (1,)
        self.shape_y = y_dict.get("ori_shape")

        self.is_unknown_shape = util_common.is_unknown([x_dict, segment_ids_dict, num_segments_dict, y_dict])
        self.e_size, self.e_size_align8 = self.compute_e_size()
        self.is_high_performance = self.check_high_performance()
        self.input_ub_tensor_size, self.ids_ub_tensor_size = self.compute_ub_tensor_size()
        self.output_0_ub_tensor_size = self.e_size_align8

        class GmTensor():
            """
                Function: use to store concat base parameters
                Modify : 2020-12-9
            """

            def __init__(self, tik_instance, input_dtype, ids_dtype, num_segments_dtype, opname):
                """
                constructor of class GmTensor

                Parameters
                ----------
                tik_instance: tik_instance
                input_dtype: x dtype
                ids_dtype: ids dtype
                num_segments_dtype: num_segments dtype

                Returns
                -------
                None
                """
                self.input_gm = tik_instance.Tensor(input_dtype, (Constant.MAX_INT32,),
                                                    name="input_gm",
                                                    scope=tik.scope_gm)
                self.ids_gm = tik_instance.Tensor(ids_dtype, (Constant.MAX_INT32,), name="ids_gm", scope=tik.scope_gm)
                if opname == "unsort_segment_sum":
                    self.num_segments_gm = tik_instance.Tensor(num_segments_dtype, (Constant.MIN_TENSOR_ELE_NUM,),
                                                               name="num_segments",
                                                               scope=tik.scope_gm)
                if input_dtype == Constant.DTYPE_FP32:
                    self.output_gm = tik_instance.Tensor(input_dtype, (Constant.MAX_INT32,),
                                                         name="output_gm",
                                                         scope=tik.scope_gm,
                                                         is_atomic_add=True)
                else:
                    self.output_gm = tik_instance.Tensor(input_dtype, (Constant.MAX_INT32,),
                                                         name="output_gm",
                                                         scope=tik.scope_gm)
                self.tiling_gm = tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                     name="tiling_gm",
                                                     scope=tik.scope_gm)

        class UbTensor():
            """
            Function: use to store concat base parameters
            Modify : 2020-12-9
            """

            def __init__(self, opname, impl_mode):
                """
                constructor of class UbTensor

                Parameters
                ----------
                None

                Returns
                -------
                None
                """
                self.input_ub = None
                self.ids_ub = None
                self.output_ub = None
                if opname == "unsort_segment_sum":
                    self.num_segments_ub = None
                if impl_mode == OpImplMode.HIGH_PERFORMANCE:
                    self.output_0_ub = None
                    self.output_1_ub = None
                    self.input1_ub = None
                    self.duiqi_ub = None
                    self.cached_ub = None
                    self.indices_index_float_ub = None
                    self.indices_simpling_ub = None
                    self.indices_temp_int_ub = None
                    self.indices_temp_float_ub = None

        # scalar of tiling params
        class CommonScalar():
            """
                Function: use to store concat base parameters
                Modify : 2020-12-9
            """

            def __init__(self, tik_instance, num_segments_dtype, ids_dtype, input_dtype, core_num):
                """
                constructor of class CommonScalar

                Parameters
                ----------
                tik_instance: tik_instance
                num_segments_dtype: num_segments dtype
                ids_dtype: ids dtype
                core_num: core_num

                Returns
                -------
                None
                """
                self.num_segments_scalar = tik_instance.Scalar(dtype=num_segments_dtype, name="num_segments_scalar")
                self.id_val_scalar = tik_instance.Scalar(dtype=ids_dtype, name="id_val_scalar")
                self.select_key = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="select_key")
                self.need_core_num = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="need_core_num")
                self.num_segments_front_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="num_segments_front_core")
                self.num_segments_last_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="num_segments_last_core")
                self.num_segments = tik_instance.Scalar(dtype=num_segments_dtype, name="num_segments")
                self.cache_mode = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                      name="cache_mode",
                                                      init_value=Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM)
                self.simpling_step = tik_instance.Scalar(dtype="int32", name="simpling_step")
                self.input_dsize = Constant.BYTE_DTYPE.get(input_dtype)
                self.ids_dsize = Constant.BYTE_DTYPE.get(ids_dtype)
                self.ids_type_is_int32 = 0
                if ids_dtype == Constant.DTYPE_INT32:
                    self.ids_type_is_int32 = 1
                self.cache_simpling_data_arr = None
                self.ori_ids_base_offset = None
                self.row_num_once_ub = None

                self.core_num_var = tik_instance.Scalar(name="core_num_var", init_value=core_num)

            def set_running_core_num(self, tiling_core_num):
                self.core_num_var.set_as(tiling_core_num)

        class Fp32InputDataInputScalar():
            """
                Function: use to store concat base parameters
            """

            def __init__(self, tik_instance):
                """
                constructor of class Fp32InputDataInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                # front core
                self.ele_num_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="ele_num_front_core")
                # front part front core
                self.mov_times_gm2ub_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_front_part_front_core")
                self.front_burst_len_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_burst_len_front_part_front_core")
                self.last_burst_len_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_burst_len_front_part_front_core")
                self.front_ele_num_ub_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_ele_num_ub_front_part_front_core")
                self.last_ele_num_ub_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_ele_num_ub_front_part_front_core")
                self.front_rows_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_rows_front_part_front_core")
                self.last_rows_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_rows_front_part_front_core")
                # last part front core
                self.mov_times_gm2ub_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_last_part_front_core")
                self.front_burst_len_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_burst_len_last_part_front_core")
                self.last_burst_len_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_burst_len_last_part_front_core")
                self.front_ele_num_ub_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_ele_num_ub_last_part_front_core")
                self.last_ele_num_ub_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_ele_num_ub_last_part_front_core")
                self.front_rows_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_rows_last_part_front_core")
                self.last_rows_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_rows_last_part_front_core")

                # last core
                self.ele_num_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="ele_num_last_core")
                # front part last core
                self.mov_times_gm2ub_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_front_part_last_core")
                self.front_burst_len_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_burst_len_front_part_last_core")
                self.last_burst_len_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_burst_len_front_part_last_core")
                self.front_ele_num_ub_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_ele_num_ub_front_part_last_core")
                self.last_ele_num_ub_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_ele_num_ub_front_part_last_core")
                self.front_rows_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_rows_front_part_last_core")
                self.last_rows_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_rows_front_part_last_core")
                # last part last core
                self.mov_times_gm2ub_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_last_part_last_core")
                self.front_burst_len_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_burst_len_last_part_last_core")
                self.last_burst_len_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_burst_len_last_part_last_core")
                self.front_ele_num_ub_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_ele_num_ub_last_part_last_core")
                self.last_ele_num_ub_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_ele_num_ub_last_part_last_core")
                self.front_rows_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_rows_last_part_last_core")
                self.last_rows_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_rows_last_part_last_core")

        class Fp32ENumInputScalar():
            """
                Function: use to store concat base parameters
            """

            def __init__(self, tik_instance):
                """
                constructor of class Fp32ENumInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.e_num = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="e_num")
                self.e_mov_times_gm2ub = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                             name="e_mov_times_gm2ub")
                self.e_ub2gm_front_burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="e_ub2gm_front_burst_len")
                self.e_num_front_part = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="e_num_front_part")
                self.e_ub2gm_last_burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="e_ub2gm_last_burst_len")
                self.e_gm2ub_last_burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="e_gm2ub_last_burst_len")
                self.e_num_last_part = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="e_num_last_part")
                self.repeat_front_front_part_front_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                              name="repeat_front_front_part_front_core")
                self.col_sub_block_front_front_part_front_core = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="col_sub_block_front_front_part_front_core")
                self.repeat_last_front_part_front_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                             name="repeat_last_front_part_front_core")
                self.col_sub_block_last_front_part_front_core = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="col_sub_block_last_front_part_front_core")
                self.repeat_front_last_part_front_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                             name="repeat_front_last_part_front_core")
                self.col_sub_block_front_last_part_front_core = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="col_sub_block_front_last_part_front_core")
                self.repeat_last_last_part_front_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                            name="repeat_last_last_part_front_core")
                self.col_sub_block_last_last_part_front_core = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="col_sub_block_last_last_part_front_core")
                self.repeat_front_front_part_last_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                             name="repeat_front_front_part_last_core")
                self.col_sub_block_front_front_part_last_core = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="col_sub_block_front_front_part_last_core")
                self.repeat_last_front_part_last_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                            name="repeat_last_front_part_last_core")
                self.col_sub_block_last_front_part_last_core = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="col_sub_block_last_front_part_last_core")
                self.repeat_front_last_part_last_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                            name="repeat_front_last_part_last_core")
                self.col_sub_block_front_last_part_last_core = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="col_sub_block_front_last_part_last_core")
                self.repeat_last_last_part_last_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                           name="repeat_last_last_part_last_core")
                self.col_sub_block_last_last_part_last_core = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="col_sub_block_last_last_part_last_core")
                self.e_num_sub = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="e_num_sub")
                self.vadd_repeat_255 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="vadd_repeat_255")
                self.vadd_repeat_64 = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="vadd_repeat_64")
                self.vadd_repeat_last = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="vadd_repeat_last")
                self.move_pad = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="move_pad")
                self.repeat_remove_pad = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                             name="repeat_remove_pad")
                self.col_block_remove_pad = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                name="col_block_remove_pad")
                self.cache_num_block = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="cache_num_block")

        class Fp32IdsInputScalar():
            """
                Function: use to store concat base parameters
            """

            def __init__(self, tik_instance):
                """
                constructor of class Fp32IdsInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.size = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="size")
                self.ele_num_front_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                              name="ele_num_front_core")
                self.mov_times_gm2ub_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_front_core")
                self.front_burst_len_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_burst_len_front_core")
                self.last_burst_len_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_burst_len_front_core")
                self.ele_num_ub_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="ele_num_ub_front_part_front_core")
                self.ele_num_ub_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="ele_num_ub_last_part_front_core")
                self.ele_num_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="ele_num_last_core")
                self.mov_times_gm2ub_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_last_core")
                self.front_burst_len_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="front_burst_len_last_core")
                self.last_burst_len_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_burst_len_last_core")
                self.ele_num_ub_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="ele_num_ub_front_part_last_core")
                self.ele_num_ub_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="ele_num_ub_last_part_last_core")

        class Fp32OutputInitInputScalar():
            """
            Function: use to store concat base parameters
            """

            def __init__(self, tik_instance):
                """
                constructor of class Fp32OutputInitInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self. \
                    last_repeat_time_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_repeat_time_front_part_front_core")
                self.init_times_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="init_times_front_part_front_core")
                self. \
                    last_repeat_time_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_repeat_time_last_part_front_core")
                self.init_times_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="init_times_last_part_front_core")
                self. \
                    last_repeat_time_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_repeat_time_front_part_last_core")
                self.init_times_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="init_times_front_part_last_core")
                self. \
                    last_repeat_time_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_repeat_time_last_part_last_core")
                self.init_times_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="init_times_last_part_last_core")
                self.last_axis_align_front_part = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_axis_align_front_part")
                self.last_axis_align_floor = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_axis_align_floor")
                self.last_part_vadd_mask = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_part_vadd_mask")
                self.last_repeat_time_last_row_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_repeat_time_last_row_front_part_front_core")
                self.init_times_last_row_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="init_times_last_row_front_part_front_core")
                self.last_repeat_time_last_row_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_repeat_time_last_row_last_part_front_core")
                self.init_times_last_row_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="init_times_last_row_last_part_front_core")
                self.last_repeat_time_last_row_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_repeat_time_last_row_front_part_last_core")
                self.init_times_last_row_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="init_times_last_row_front_part_last_core")
                self.last_repeat_time_last_row_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="last_repeat_time_last_row_last_part_last_core")
                self.init_times_last_row_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="init_times_last_row_last_part_last_core")
                self.max_cache_n_num = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="max_cache_n_num")

        self.obj_gm_tensor = GmTensor(self.tik_instance, self.input_dtype, self.ids_dtype, self.num_segments_dtype,
                                      self.opname)
        self.obj_ub_tensor = UbTensor(self.opname, self.impl_mode)
        self.obj_common_scalar = CommonScalar(self.tik_instance, self.num_segments_dtype, self.ids_dtype,
                                              self.input_dtype, self.core_num)
        self.obj_fp32_input_data_input_scalar = Fp32InputDataInputScalar(self.tik_instance)
        self.obj_fp32_e_num_input_scalar = Fp32ENumInputScalar(self.tik_instance)
        self.obj_fp32_ids_input_scalar = Fp32IdsInputScalar(self.tik_instance)
        self.obj_fp32_output_init_input_scalar = Fp32OutputInitInputScalar(self.tik_instance)

        if self.opname == "unsorted_segment_sum":
            with self.tik_instance.new_stmt_scope():
                self.obj_ub_tensor.num_segments_ub = self.tik_instance.Tensor(self.num_segments_dtype,
                                                                              (Constant.MIN_TENSOR_ELE_NUM,),
                                                                              name="num_segments_ub",
                                                                              scope=tik.scope_ubuf)
                self.tik_instance.data_move(self.obj_ub_tensor.num_segments_ub, self.obj_gm_tensor.num_segments_gm, 0,
                                            1, 1, 0, 0)
                self.obj_common_scalar.num_segments_scalar.set_as(self.obj_ub_tensor.num_segments_ub[1])

        with self.tik_instance.new_stmt_scope():
            self.obj_ub_tensor.tiling_ub = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE,
                                                                    (Constant.TILING_PARAMS_NUM,),
                                                                    name="tiling_ub",
                                                                    scope=tik.scope_ubuf)
            # mov tiling params from gm to ub
            self.tik_instance.data_move(self.obj_ub_tensor.tiling_ub, self.obj_gm_tensor.tiling_gm, 0, 1,
                                        Constant.TILING_PARAMS_NUM * Constant.BYTE_INT32 // Constant.BYTE_BLOCK, 0, 0)
            # input scalar in flowtable
            input_scalar_index = 0
            # common params
            self.obj_common_scalar.select_key.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_common_scalar.need_core_num.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1

            # input data params
            # front core
            self. \
                obj_fp32_input_data_input_scalar.ele_num_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # front part front core
            self. \
                obj_fp32_input_data_input_scalar. \
                mov_times_gm2ub_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_burst_len_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_burst_len_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_ele_num_ub_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_ele_num_ub_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_rows_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_rows_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # last part front core
            self.obj_fp32_input_data_input_scalar. \
                mov_times_gm2ub_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_burst_len_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_burst_len_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_ele_num_ub_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_ele_num_ub_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_rows_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_rows_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # last core
            self.obj_fp32_input_data_input_scalar. \
                ele_num_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # front part last core
            self.obj_fp32_input_data_input_scalar. \
                mov_times_gm2ub_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_burst_len_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_burst_len_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_ele_num_ub_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_ele_num_ub_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_rows_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_rows_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # last part last core
            self.obj_fp32_input_data_input_scalar. \
                mov_times_gm2ub_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_burst_len_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_burst_len_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_ele_num_ub_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_ele_num_ub_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                front_rows_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar. \
                last_rows_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1

            # e num params
            self.obj_fp32_e_num_input_scalar.e_num.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.e_mov_times_gm2ub. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar. \
                e_ub2gm_front_burst_len. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.e_num_front_part. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar. \
                e_ub2gm_last_burst_len. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.e_num_last_part. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1

            # ids params
            self.obj_fp32_ids_input_scalar.size.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                ele_num_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                mov_times_gm2ub_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                front_burst_len_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                last_burst_len_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                ele_num_ub_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                ele_num_ub_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.ele_num_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                mov_times_gm2ub_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                front_burst_len_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                last_burst_len_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                ele_num_ub_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar. \
                ele_num_ub_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1

            # output init params
            self.obj_fp32_output_init_input_scalar. \
                last_repeat_time_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                init_times_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_repeat_time_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                init_times_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_repeat_time_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                init_times_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_repeat_time_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                init_times_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_axis_align_front_part. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_axis_align_floor. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_part_vadd_mask. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.e_gm2ub_last_burst_len. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_repeat_time_last_row_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                init_times_last_row_front_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_repeat_time_last_row_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                init_times_last_row_last_part_front_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_repeat_time_last_row_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                init_times_last_row_front_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                last_repeat_time_last_row_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar. \
                init_times_last_row_last_part_last_core. \
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            if self.opname == "segment_sum":
                self.obj_common_scalar.num_segments. \
                    set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])

            input_scalar_index = input_scalar_index + 1
            self.obj_common_scalar.set_running_core_num( \
                self.obj_ub_tensor.tiling_ub[input_scalar_index])

            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.repeat_front_front_part_front_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.col_sub_block_front_front_part_front_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.repeat_last_front_part_front_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.col_sub_block_last_front_part_front_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.repeat_front_last_part_front_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.col_sub_block_front_last_part_front_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.repeat_last_last_part_front_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.col_sub_block_last_last_part_front_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.repeat_front_front_part_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.col_sub_block_front_front_part_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])

            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.repeat_last_front_part_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.col_sub_block_last_front_part_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.repeat_front_last_part_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.col_sub_block_front_last_part_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.repeat_last_last_part_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.col_sub_block_last_last_part_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.e_num_sub.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.vadd_repeat_255.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.vadd_repeat_64.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.vadd_repeat_last.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.move_pad.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.max_cache_n_num.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.repeat_remove_pad.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.col_block_remove_pad.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.cache_num_block.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])

    def unsorted_segment_sum(self):
        """
        main process of unsorted_segment_sum

        Parameters
        ----------
        None

        Returns:
        -------
        None
        """
        _enable_atomic_add(self.tik_instance)
        with self.tik_instance.for_range(0,
                                         self.obj_common_scalar.core_num_var,
                                         block_num=self.obj_common_scalar.core_num_var) as block_index:
            with self.tik_instance.if_scope(block_index < self.obj_common_scalar.need_core_num):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.obj_common_scalar.select_key == 0):
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype, (64,),
                                                                                name="output_ub",
                                                                                scope=tik.scope_ubuf)
                        self.tik_instance.vector_dup(Constant.MASK_FP32, self.obj_ub_tensor.output_ub[0], 0, 1, 1, 8)
                        self.tik_instance.data_move(self.obj_gm_tensor.output_gm[0], self.obj_ub_tensor.output_ub[0], 0,
                                                    1, 1, 0, 0)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == Constant.SELECT_KEY_MODE_FP32_INPUT_NUM_SEGMENT_ONE):
                        # fp32 last axis 32B align
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype,
                                                                               (self.ub_size // Constant.BYTE_FP32,),
                                                                               name="input_ub",
                                                                               scope=tik.scope_ubuf)
                        _tik_atomic_add_num_segment_one(block_index, self.tik_instance, self.obj_gm_tensor,
                                                        self.obj_ub_tensor, self.obj_common_scalar,
                                                        self.obj_fp32_input_data_input_scalar,
                                                        self.obj_fp32_e_num_input_scalar,
                                                        self.obj_fp32_ids_input_scalar)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN):
                        # fp32 last axis 32B align
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (self.ub_size // 2 // Constant.BYTE_FP32,),
                            name="input_ub",
                            scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 2 // Constant.BYTE_DTYPE.get(self.ids_dtype),),
                            name="ids_ub",
                            scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_align_small_e(block_index, self.tik_instance, self.obj_gm_tensor,
                                                                self.obj_ub_tensor, self.obj_common_scalar,
                                                                self.obj_fp32_input_data_input_scalar,
                                                                self.obj_fp32_e_num_input_scalar,
                                                                self.obj_fp32_ids_input_scalar)
                with self.tik_instance.new_stmt_scope():
                    if self.ids_dtype != Constant.DTYPE_INT64:
                        with self.tik_instance.if_scope(
                                self.obj_common_scalar.select_key == Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE):
                            # fp32 last axis is 1
                            def _compute_input_ub_row():
                                one_row_size = Constant.BYTE_FP32 + Constant.BYTE_INT32 + \
                                               Constant.BYTE_FP32 * \
                                               self.fp32_ele_num_one_block
                                return _floor(self.ub_size // one_row_size, self.fp32_ele_num_one_block)

                            self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype,
                                                                                   (_compute_input_ub_row(),),
                                                                                   name="input_ub",
                                                                                   scope=tik.scope_ubuf)
                            self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype,
                                                                                 (_compute_input_ub_row(),),
                                                                                 name="ids_ub",
                                                                                 scope=tik.scope_ubuf)
                            self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(
                                self.output_dtype, (_compute_input_ub_row(), self.fp32_ele_num_one_block),
                                name="output_ub",
                                scope=tik.scope_ubuf)
                            _tik_atomic_add_last_axis_one(
                                block_index, self.tik_instance, self.obj_gm_tensor, self.obj_ub_tensor,
                                self.obj_common_scalar, self.obj_fp32_input_data_input_scalar,
                                self.obj_fp32_e_num_input_scalar, self.obj_fp32_ids_input_scalar,
                                self.obj_fp32_output_init_input_scalar, self.fp32_ele_num_one_block)
                with self.tik_instance.new_stmt_scope():
                    if self.ids_dtype != Constant.DTYPE_INT64:
                        with self.tik_instance.if_scope(self.obj_common_scalar.select_key ==
                                                        Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MODIFY):
                            # fp32 last axis is 1 modify
                            def _compute_input_ub_row1():
                                one_row_size = Constant.BYTE_FP32 + Constant.BYTE_INT32
                                return _floor(self.ub_size // one_row_size, Constant.MASK_FP32)

                            self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype,
                                                                                   (_compute_input_ub_row1(),),
                                                                                   name="input_ub",
                                                                                   scope=tik.scope_ubuf)
                            self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype,
                                                                                 (_compute_input_ub_row1(),),
                                                                                 name="ids_ub",
                                                                                 scope=tik.scope_ubuf)
                            _tik_atomic_add_last_axis_one_modify(block_index, self.tik_instance, self.obj_gm_tensor,
                                                                 self.obj_ub_tensor, self.obj_common_scalar,
                                                                 self.obj_fp32_input_data_input_scalar,
                                                                 self.obj_fp32_ids_input_scalar,
                                                                 self.obj_fp32_output_init_input_scalar)
                with self.tik_instance.new_stmt_scope():
                    if self.ids_dtype != Constant.DTYPE_INT64:
                        with self.tik_instance.if_scope(self.obj_common_scalar.select_key ==
                                                        Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI):
                            # fp32 last axis is 1 multi 64
                            def _compute_input_ub_row2():
                                one_row_size = Constant.BYTE_FP32 + Constant.BYTE_INT32
                                return _floor(self.ub_size // one_row_size, 16 * Constant.MASK_FP32)

                            self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype,
                                                                                   (_compute_input_ub_row2(),),
                                                                                   name="input_ub",
                                                                                   scope=tik.scope_ubuf)
                            self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype,
                                                                                 (_compute_input_ub_row2(),),
                                                                                 name="ids_ub",
                                                                                 scope=tik.scope_ubuf)
                            _tik_atomic_add_last_axis_one_multi(block_index, self.tik_instance, self.obj_gm_tensor,
                                                                self.obj_ub_tensor, self.obj_common_scalar,
                                                                self.obj_fp32_input_data_input_scalar,
                                                                self.obj_fp32_ids_input_scalar,
                                                                self.obj_fp32_output_init_input_scalar)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.obj_common_scalar.select_key ==
                                                    Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN):
                        # fp32 last axis 32B not align
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (self.ub_size // 3 // Constant.BYTE_FP32,),
                            name="input_ub",
                            scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 3 // Constant.BYTE_DTYPE.get(self.ids_dtype),),
                            name="ids_ub",
                            scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(
                            self.output_dtype, (self.ub_size // 3 // Constant.BYTE_FP32,),
                            name="output_ub",
                            scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_not_align_small_e(
                            block_index, self.tik_instance, self.obj_gm_tensor, self.obj_ub_tensor,
                            self.obj_common_scalar, self.obj_fp32_input_data_input_scalar,
                            self.obj_fp32_e_num_input_scalar, self.obj_fp32_ids_input_scalar,
                            self.obj_fp32_output_init_input_scalar, self.fp32_ele_num_one_block)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.obj_common_scalar.select_key ==
                                                    Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E):
                        # fp32 last axis 32B align and big e
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (self.ub_size // 2 // Constant.BYTE_FP32,),
                            name="input_ub",
                            scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 2 // Constant.BYTE_DTYPE.get(self.ids_dtype),),
                            name="ids_ub",
                            scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_align_big_e(block_index, self.tik_instance, self.obj_gm_tensor,
                                                              self.obj_ub_tensor, self.obj_common_scalar,
                                                              self.obj_fp32_input_data_input_scalar,
                                                              self.obj_fp32_e_num_input_scalar,
                                                              self.obj_fp32_ids_input_scalar)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.obj_common_scalar.select_key ==
                                                    Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E):
                        # fp32 last axis 32B not align and big e
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (self.ub_size // 3 // Constant.BYTE_FP32,),
                            name="input_ub",
                            scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 3 // Constant.BYTE_DTYPE.get(self.ids_dtype),),
                            name="ids_ub",
                            scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(
                            self.output_dtype, (self.ub_size // 3 // Constant.BYTE_FP32,),
                            name="output_ub",
                            scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_not_align_big_e(block_index, self.tik_instance, self.obj_gm_tensor,
                                                                  self.obj_ub_tensor, self.obj_common_scalar,
                                                                  self.obj_fp32_input_data_input_scalar,
                                                                  self.obj_fp32_e_num_input_scalar,
                                                                  self.obj_fp32_ids_input_scalar,
                                                                  self.obj_fp32_output_init_input_scalar)
                with self.tik_instance.new_stmt_scope():
                    if self.is_high_performance:
                        with self.tik_instance.if_scope(self.obj_common_scalar.select_key ==
                                                        Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_HP):
                            # fp32 last axis 32B align
                            ub_tensor_size = _floor(self.ub_size // 6, Constant.BYTE_BLOCK)
                            output_num = ub_tensor_size // Constant.BYTE_FP32
                            self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                                self.input_dtype, (ub_tensor_size // Constant.BYTE_FP32,),
                                name="input_ub",
                                scope=tik.scope_ubuf)
                            self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                                self.ids_dtype, (ub_tensor_size // Constant.BYTE_DTYPE.get(self.ids_dtype) + 8, ),
                                name="ids_ub",
                                scope=tik.scope_ubuf)
                            self.obj_ub_tensor.output_0_ub = self.tik_instance.Tensor(self.input_dtype, (output_num,),
                                                                                      name="output_0_ub",
                                                                                      scope=tik.scope_ubuf)
                            self.obj_ub_tensor.cached_ub = self.tik_instance.Tensor(
                                self.input_dtype, (ub_tensor_size // self.obj_common_scalar.input_dsize,),
                                name="cached_ub",
                                scope=tik.scope_ubuf)
                            self.obj_ub_tensor.indices_index_float_ub = self.tik_instance.Tensor(
                                Constant.DTYPE_FP32, (ub_tensor_size // self.obj_common_scalar.ids_dsize + 8, ),
                                name="indices_index_float_ub",
                                scope=tik.scope_ubuf)
                            self.obj_ub_tensor.indices_simpling_ub = self.tik_instance.Tensor(
                                Constant.DTYPE_FP32, (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                name="indices_simpling_ub",
                                scope=tik.scope_ubuf)
                            self.obj_common_scalar.cache_simpling_data_arr = self.tik_instance.ScalarArray(
                                Constant.DTYPE_INT32, Constant.CACHE_ACT_SIMPLING_NUM, "cache_simpling_data_arr")
                            self.obj_common_scalar.ori_ids_base_offset = self.tik_instance.Scalar(
                                dtype=Constant.TILING_PARAM_DTYPE, name="ori_ids_base_offset", init_value=0)
                            self.obj_common_scalar.row_num_once_ub = self.tik_instance.Scalar(
                                dtype=Constant.TILING_PARAM_DTYPE, name="row_num_once_ub", init_value=0)
                            self.obj_ub_tensor.indices_temp_int_ub = self.tik_instance.Tensor(
                                Constant.DTYPE_INT32, (output_num // 2, ),
                                name="indices_temp_int_ub",
                                scope=tik.scope_ubuf)
                            self.obj_ub_tensor.indices_temp_float_ub = self.tik_instance.Tensor(
                                Constant.DTYPE_FP32, (output_num // 2, ),
                                name="indices_temp_float_ub",
                                scope=tik.scope_ubuf)
                            repeat_time_255_a, left_part_a = _div(output_num // 2,
                                    Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
                            repeat_time_64_a, last_part_mask_a = _div(left_part_a, Constant.MASK_FP32)
                            args_init = (self.tik_instance, self.obj_ub_tensor.indices_temp_float_ub,
                                    last_part_mask_a, repeat_time_64_a, repeat_time_255_a)
                            _tik_init_ub_tensor_hp(args_init)
                            args_init_align = (self.tik_instance, self.obj_ub_tensor.indices_temp_int_ub,
                                    last_part_mask_a, repeat_time_64_a, repeat_time_255_a)
                            _tik_init_ub_tensor_hp(args_init_align)
                            _tik_atomic_add_last_axis_align_small_e_hp(
                                block_index, self.tik_instance, self.obj_gm_tensor, self.obj_ub_tensor,
                                self.obj_common_scalar, self.obj_fp32_input_data_input_scalar,
                                self.obj_fp32_e_num_input_scalar, self.obj_fp32_ids_input_scalar,
                                self.obj_fp32_output_init_input_scalar, output_num)
                with self.tik_instance.new_stmt_scope():
                    if self.is_high_performance:
                        with self.tik_instance.if_scope(self.obj_common_scalar.select_key ==
                                                        Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_HP):
                            # fp32 last axis 32B not align
                            ub_tensor_size = _floor(self.ub_size // 3, Constant.BYTE_BLOCK)
                            output_num = ub_tensor_size // Constant.BYTE_FP32
                            self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype, (output_num,),
                                                                                   name="input_ub",
                                                                                   scope=tik.scope_ubuf)
                            self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype, (output_num,),
                                                                                 name="ids_ub",
                                                                                 scope=tik.scope_ubuf)
                            self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype,
                                                                                    (Constant.MASK_8,),
                                                                                    name="output_ub",
                                                                                    scope=tik.scope_ubuf)
                            self.obj_ub_tensor.output_0_ub = self.tik_instance.Tensor(self.output_dtype, (output_num,),
                                                                                      name="output_0_ub",
                                                                                      scope=tik.scope_ubuf)
                            _tik_atomic_add_last_axis_not_align_small_e_hp(
                                block_index, self.tik_instance, self.obj_gm_tensor, self.obj_ub_tensor,
                                self.obj_common_scalar, self.obj_fp32_input_data_input_scalar,
                                self.obj_fp32_e_num_input_scalar, self.obj_fp32_ids_input_scalar,
                                self.obj_fp32_output_init_input_scalar, output_num)
                with self.tik_instance.new_stmt_scope():
                    if self.is_high_performance:
                        with self.tik_instance.if_scope(self.obj_common_scalar.select_key ==
                                                        Constant.SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_HP_PAD):
                            # fp32 last axis 32B not align
                            ub_tensor_size = _floor(self.ub_size // 6, Constant.BYTE_BLOCK)
                            output_num = ub_tensor_size // Constant.BYTE_FP32
                            self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype, (output_num,),
                                                                                   name="input_ub",
                                                                                   scope=tik.scope_ubuf)
                            self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype, (output_num,),
                                                                                 name="ids_ub",
                                                                                 scope=tik.scope_ubuf)
                            self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype,
                                                                                    (Constant.MASK_8,),
                                                                                    name="output_ub",
                                                                                    scope=tik.scope_ubuf)
                            self.obj_ub_tensor.output_0_ub = self.tik_instance.Tensor(self.output_dtype, (output_num,),
                                                                                      name="output_0_ub",
                                                                                      scope=tik.scope_ubuf)

                            self.obj_ub_tensor.cached_ub = self.tik_instance.Tensor(self.input_dtype, (output_num,),
                                                                                    name="cached_ub",
                                                                                    scope=tik.scope_ubuf)

                            self.obj_common_scalar.cache_simpling_data_arr = self.tik_instance.ScalarArray(
                                Constant.DTYPE_INT32, Constant.CACHE_ACT_SIMPLING_NUM, "cache_simpling_data_arr")

                            self.obj_common_scalar.ori_ids_base_offset = self.tik_instance.Scalar(
                                dtype=Constant.TILING_PARAM_DTYPE, name="ori_ids_base_offset", init_value=0)

                            self.obj_common_scalar.row_num_once_ub = self.tik_instance.Scalar(
                                dtype=Constant.TILING_PARAM_DTYPE, name="row_num_once_ub", init_value=0)
                            _tik_atomic_add_last_axis_not_align_small_e_hp_pad(
                                block_index, self.tik_instance, self.obj_gm_tensor, self.obj_ub_tensor,
                                self.obj_common_scalar, self.obj_fp32_input_data_input_scalar,
                                self.obj_fp32_e_num_input_scalar, self.obj_fp32_ids_input_scalar,
                                self.obj_fp32_output_init_input_scalar, output_num)
        _disable_atomic_add(self.tik_instance)
        # add compile info
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size,
                "core_num": self.core_num,
                "dtype": self.obj_gm_tensor.input_gm.dtype,
                "ub_tensor_num": self.ub_tensor_num,
                "impl_mode": Constant.IMPL_MODE.get(self.impl_mode, 0)
            })
        tbe_context.get_context().add_compile_info("is_tik", True)
        opt_config = {"enable_const_fold": True}
        if self.opname == "segment_sum":
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=[self.obj_gm_tensor.input_gm, self.obj_gm_tensor.ids_gm],
                                       outputs=[self.obj_gm_tensor.output_gm],
                                       flowtable=[self.obj_gm_tensor.tiling_gm],
                                       config=opt_config)
        else:
            self.tik_instance.BuildCCE(
                kernel_name=self.kernel_name,
                inputs=[self.obj_gm_tensor.input_gm, self.obj_gm_tensor.ids_gm, self.obj_gm_tensor.num_segments_gm],
                outputs=[self.obj_gm_tensor.output_gm],
                flowtable=[self.obj_gm_tensor.tiling_gm],
                config=opt_config)

    def check_high_performance(self):
        input_ub_tensor_size = _floor(self.ub_size // 2, Constant.BYTE_BLOCK)
        if self.impl_mode == OpImplMode.HIGH_PERFORMANCE and \
                (self.e_size_align8 * 2 + 8) * Constant.BYTE_FP32 < input_ub_tensor_size:
            return True
        return False

    def compute_e_size(self):
        e_size, e_size_align8 = -1, -1
        if not self.is_unknown_shape:
            x_ele_num = reduce(lambda x, y: x * y, self.shape_x)
            ids_ele_num = reduce(lambda x, y: x * y, self.shape_segment_ids)
            e_size = x_ele_num // ids_ele_num
            e_size_align8 = util_common.div_align_scalar(e_size, Constant.MASK_8, div_mode="ceil")
        return e_size, e_size_align8

    def compute_ub_tensor_size(self):
        input_ub_tensor_size, ids_ub_tensor_size = -1, -1
        if self.is_unknown_shape:
            return input_ub_tensor_size, ids_ub_tensor_size

        num_segments = self.shape_y[0]
        if num_segments > 1:
            if self.e_size == 1:
                one_row_size = Constant.BYTE_FP32 + Constant.BYTE_INT32 + Constant.ELE_NUM_ONE_BLOCK_FP32
                ids_ub_tensor_size = util_common.div_align_scalar(self.ub_size // one_row_size,
                                                                  Constant.BYTE_BLOCK,
                                                                  div_mode="ceil")
                input_ub_tensor_size = ids_ub_tensor_size
            elif self.e_size > 1:
                if self.e_size % Constant.ELE_NUM_ONE_BLOCK_FP32 == 0:
                    ids_ub_tensor_size = _floor(self.ub_size // 2, Constant.BYTE_BLOCK)
                    input_ub_tensor_size = ids_ub_tensor_size
                    if self.is_high_performance:
                        input_ub_tensor_size = input_ub_tensor_size - self.e_size_align8 * Constant.BYTE_FP32
                elif self.e_size % Constant.ELE_NUM_ONE_BLOCK_FP32 > 0:
                    ids_ub_tensor_size = _floor(self.ub_size // 3, Constant.BYTE_BLOCK)
                    input_ub_tensor_size = ids_ub_tensor_size
                    if self.is_high_performance:
                        ids_ub_tensor_size = _floor(self.ub_size // 2, Constant.BYTE_BLOCK)
                        input_ub_tensor_size = ids_ub_tensor_size - (self.e_size_align8 + 8) * Constant.BYTE_FP32
        else:
            ids_ub_tensor_size = _floor(self.ub_size, Constant.BYTE_BLOCK)
            input_ub_tensor_size = ids_ub_tensor_size

        return input_ub_tensor_size, ids_ub_tensor_size


def _enable_atomic_add(tik_inst):
    """
    enable atomic add

    Parameters
    ----------
    tik_inst: tik instance

    Returns
    -------
    None
    """
    if tbe_platform.api_check_support("tik.set_atomic_add"):
        tik_inst.set_atomic_add(1)


def _disable_atomic_add(tik_inst):
    """
    disable atomic add

    Parameters
    ----------
    tik_inst: tik instance

    Returns
    -------
    None
    """
    if tbe_platform.api_check_support("tik.set_atomic_add"):
        tik_inst.set_atomic_add(0)


def _tik_get_ub_size(is_double_buffer=True):
    """
    get ub size

    Parameters
    ----------
    is_double_buffer: is_double_buffer flag

    Returns
    -------
    ub_size
    """
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 256 * 10
    if is_double_buffer:
        return ub_size // 2
    return ub_size


def _tik_get_core_num():
    """
    get core num

    Parameters
    ----------
    None

    Returns
    -------
    core num
    """
    return tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)


def _tik_init_ub_tensor(tik_inst, ub_tensor, init_last_repeat_time, init_times):
    """
    init ub tensor

    Parameters
    ----------
    tik_inst: tik instance
    ub_tensor: ub_tensor
    init_last_repeat_time: last repeat time
    init_times: init times

    Returns
    -------
    None
    """
    with tik_inst.for_range(0, init_times) as init_index:
        with tik_inst.if_scope(init_index == init_times - 1):
            tik_inst.vector_dup(Constant.MASK_FP32,
                                ub_tensor[init_index * Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME], 0,
                                init_last_repeat_time, 1, 8)
        with tik_inst.else_scope():
            tik_inst.vector_dup(Constant.MASK_FP32,
                                ub_tensor[init_index * Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME], 0,
                                Constant.MAX_REPEAT_TIME, 1, 8)


def _tik_init_ub_tensor_hp(args):
    """
    init ub tensor

    Parameters
    ----------
    args: args
    Returns
    -------
    None
    """
    (tik_inst, ub_tensor, last_part_mask, repeat_time_64, repeat_time_255x64) = args
    with tik_inst.for_range(0, repeat_time_255x64) as init_index:
        tik_inst.vector_dup(Constant.MASK_FP32, ub_tensor[init_index * Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME],
                            0, Constant.MAX_REPEAT_TIME, 1, 8)
    if repeat_time_64 > 0:
        tik_inst.vector_dup(Constant.MASK_FP32,
                            ub_tensor[repeat_time_255x64 * Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME], 0,
                            repeat_time_64, 1, 8)
    if last_part_mask > 0:
        tik_inst.vector_dup(
            last_part_mask, ub_tensor[repeat_time_255x64 * Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME +
                                      repeat_time_64 * Constant.MASK_FP32], 0, 1, 1, 8)


def _tik_vadd_ub_tensor_hp(args):
    """
    init ub tensor

    Parameters
    ----------
    args: args

    Returns
    -------
    None
    """
    (tik_inst, output_ub, input_ub, output_0_ub_offset, input_offset_ub, last_part_mask, repeat_time_64,
     repeat_time_255x64) = args
    max_repeat_e_num_once = Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME
    with tik_inst.for_range(0, repeat_time_255x64) as init_index:
        _tik_vadd(tik_inst, input_ub[input_offset_ub + init_index * max_repeat_e_num_once],
                  output_ub[output_0_ub_offset + init_index * max_repeat_e_num_once], Constant.MAX_REPEAT_TIME,
                  Constant.MASK_FP32)
    with tik_inst.if_scope(repeat_time_64 > 0):
        _tik_vadd(tik_inst, input_ub[input_offset_ub + repeat_time_255x64 * max_repeat_e_num_once],
                  output_ub[output_0_ub_offset + repeat_time_255x64 * max_repeat_e_num_once], repeat_time_64,
                  Constant.MASK_FP32)
    with tik_inst.if_scope(last_part_mask > 0):
        _tik_vadd(
            tik_inst, input_ub[input_offset_ub + repeat_time_255x64 * max_repeat_e_num_once +
                               repeat_time_64 * Constant.MASK_FP32],
            output_ub[output_0_ub_offset + repeat_time_255x64 * max_repeat_e_num_once +
                      repeat_time_64 * Constant.MASK_FP32], 1, last_part_mask)


def _tik_vadd_ub_tensor_hp_align(args, index, e_num):
    """
    init ub tensor

    Parameters
    ----------
    args: args

    Returns
    -------
    None
    """
    (tik_inst, output_ub, input_ub, input_offset_ub, last_part_mask, repeat_time_64, repeat_time_255x64) = args
    max_repeat_e_num_once = Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME
    with tik_inst.for_range(0, repeat_time_255x64) as init_index:
        _tik_vadd(tik_inst, input_ub[input_offset_ub + init_index * max_repeat_e_num_once],
                  output_ub[index * e_num + init_index * max_repeat_e_num_once], Constant.MAX_REPEAT_TIME,
                  Constant.MASK_FP32)
    with tik_inst.if_scope(repeat_time_64 > 0):
        _tik_vadd(tik_inst, input_ub[input_offset_ub + repeat_time_255x64 * max_repeat_e_num_once],
                  output_ub[index * e_num + repeat_time_255x64 * max_repeat_e_num_once], repeat_time_64,
                  Constant.MASK_FP32)
    with tik_inst.if_scope(last_part_mask > 0):
        _tik_vadd(
            tik_inst, input_ub[input_offset_ub + repeat_time_255x64 * max_repeat_e_num_once +
                               repeat_time_64 * Constant.MASK_FP32],
            output_ub[index * e_num + repeat_time_255x64 * max_repeat_e_num_once + repeat_time_64 * Constant.MASK_FP32],
            1, last_part_mask)


def _tik_init_ub_tensor_once(tik_inst, ub_tensor, repeat_time, mask):
    """
    init ub tensor once

    Parameters
    ----------
    tik_inst: tik instance
    ub_tensor: ub_tensor
    repeat_time: repeat time
    mask: mask

    Returns
    -------
    None
    """
    tik_inst.vector_dup(mask, ub_tensor, 0, repeat_time, 1, 8)


def _tik_vadd(tik_inst, input_ub, output_ub, repeat_time, mask):
    """
    tik_vadd

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input ub tensor
    output_ub: output ub tensor
    repeat_time: repeat time
    mask: mask

    Returns
    -------
    None
    """
    tik_inst.vadd(mask, output_ub, output_ub, input_ub, repeat_time, 1, 1, 1, 8, 8, 8)


def _tik_mov_output_ub2gm_continue(tik_inst, output_gm, output_ub, output_offset_gm, output_offset_ub, output_n_burst,
                                   output_burst_len):
    """
    tik_mov_output_ub2gm_continue

    Parameters
    ----------
    tik_inst: tik instance
    output_gm: output gm tensor
    output_ub: output ub tensor
    output_offset_gm: output offset gm
    output_offset_ub: output offset ub
    output_n_burst: n_burst
    output_burst_len: burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(output_gm[output_offset_gm], output_ub[output_offset_ub], 0, output_n_burst, output_burst_len, 0,
                       0)


def _tik_mov_input_gm2ub_continue(tik_inst, input_gm, input_ub, input_offset_gm, input_offset_ub, input_n_burst,
                                  input_burst_len):
    """
    tik_mov_input_gm2ub_continue

    Parameters
    ----------
    tik_inst: tik instance
    input_gm: input gm tensor
    input_ub: input ub tensor
    input_offset_gm: input offset gm
    input_offset_ub: input offset ub
    input_n_burst: n_burst
    input_burst_len: burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(input_ub[input_offset_ub], input_gm[input_offset_gm], 0, input_n_burst, input_burst_len, 0, 0)


def _tik_mov_input_gm2ub_discrete(tik_inst, input_gm, input_ub, input_offset_gm, input_offset_ub, input_n_burst,
                                  input_burst_len, input_mov_times, input_ele_num_one_row,
                                  input_ele_num_one_row_align_32b):
    """
    tik_mov_input_gm2ub_discrete

    Parameters
    ----------
    tik_inst: tik instance
    input_gm: input gm tensor
    input_ub: input ub tensor
    input_offset_gm: input offset gm
    input_offset_ub: input offset ub
    input_n_burst: n_burst
    input_burst_len: burst_len
    input_mov_times: mov times
    input_ele_num_one_row: input ele num one row
    input_ele_num_one_row_align_32b: input ele num one row align 32b

    Returns
    -------
    None
    """
    with tik_inst.for_range(0, input_mov_times) as input_mov_index:
        tik_inst.data_move(input_ub[input_offset_ub + input_mov_index * input_ele_num_one_row_align_32b],
                           input_gm[input_offset_gm + input_mov_index * input_ele_num_one_row], 0, input_n_burst,
                           input_burst_len, 0, 0)


def _tik_mov_ids_gm2ub(tik_inst, ids_gm, ids_ub, ids_offset_gm, ids_offset_ub, ids_n_burst, ids_burst_len):
    """
    tik_mov_ids_gm2ub

    Parameters
    ----------
    tik_inst: tik instance
    ids_gm: ids_gm tensor
    ids_ub: ids_ub tensor
    ids_offset_gm: ids_offset_gm
    ids_offset_ub: ids_offset_ub
    ids_n_burst: ids_n_burst
    ids_burst_len: ids_burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(ids_ub[ids_offset_ub], ids_gm[ids_offset_gm], 0, ids_n_burst, ids_burst_len, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_one(tik_inst, input_ub, ids_ub, output_ub, output_gm, ub2gm_burst_len,
                                              ids_num, output_ele_num_one_row, id_val_scalar):
    """
    tik_atomic_add_ub2gm_by_id_last_axis_one

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    ids_ub: ids_ub tensor
    output_ub: output_ub tensor
    output_gm: output_gm tensor
    ub2gm_burst_len: ub2gm_burst_len
    ids_num: ids_num
    output_ele_num_one_row: output_ele_num_one_row
    id_val_scalar: id_val_scalar

    Returns
    -------
    None
    """
    input_ele_scalar = tik_inst.Scalar(dtype="float32", name="input_ele_scalar")
    with tik_inst.for_range(0, ids_num) as ids_index:
        input_ele_scalar.set_as(input_ub[ids_index])
        output_ub[ids_index * output_ele_num_one_row].set_as(input_ele_scalar)
        id_val_scalar.set_as(ids_ub[ids_index])
        tik_inst.data_move(output_gm[id_val_scalar], output_ub[ids_index * output_ele_num_one_row], 0, 1,
                           ub2gm_burst_len, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(tik_inst, input_ub, ids_ub, times_by_mask, output_gm,
                                                     id_val_scalar):
    """
    modify float32 atomic add when last axis of input is one

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    ids_ub: ids_ub tensor
    output_gm: output_gm tensor
    id_val_scalar: id_val_scalar

    Returns
    -------
    None
    """
    id_val_fp32 = tik_inst.Scalar(Constant.DTYPE_FP32, "id_val_fp32")
    input_val = tik_inst.Scalar(Constant.DTYPE_FP32, "input_val")
    neg_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "neg_ub")
    tik_inst.vector_dup(Constant.MASK_FP32, neg_ub[0], Constant.NEG_ONE, 1, 1, 8)
    with tik_inst.for_range(0, times_by_mask) as index:
        # times divided by mask
        conv_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "conv_ub")
        tik_inst.vconv(Constant.MASK_FP32, "", conv_ub[0], ids_ub[index * Constant.MASK_FP32], 1, 1, 1, 8, 8)
        with tik_inst.for_range(0, Constant.MASK_FP32) as ids_index:
            # traversal ids
            id_val_fp32.set_as(conv_ub[ids_index])
            with tik_inst.if_scope(id_val_fp32 >= Constant.ZERO):
                # new id
                zero_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "zero_ub")
                tik_inst.vector_dup(Constant.MASK_FP32, zero_ub[0], Constant.ZERO, 1, 1, 8)
                dup_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "dup_ub")
                tik_inst.vector_dup(Constant.MASK_FP32, dup_ub[0], id_val_fp32, 1, 1, 8)
                cmpmask = tik_inst.vcmp_eq(Constant.MASK_FP32, dup_ub[0], conv_ub[0], 1, 1)
                tik_inst.vsel(Constant.MASK_FP32, 0, conv_ub[0], cmpmask, neg_ub[0], conv_ub[0], 1, 1, 1, 1, 8, 8, 8)
                sel_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "sel_ub")
                tik_inst.vsel(Constant.MASK_FP32, 0, sel_ub[0], cmpmask, input_ub[index * Constant.MASK_FP32],
                              zero_ub[0], 1, 1, 1, 1, 8, 8, 8)
                cadd_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "cadd_ub")
                tik_inst.vcadd(Constant.MASK_FP32, cadd_ub[0], sel_ub[0], 1, 1, 1, 8)
                input_val.set_as(cadd_ub[0])
                zero_ub[0].set_as(input_val)
                id_val_scalar.set_as(ids_ub[index * Constant.MASK_FP32 + ids_index])
                tik_inst.data_move(output_gm[id_val_scalar], zero_ub[0], 0, 1, 1, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_one_modify_last_part(tik_inst, input_ub, ids_ub, last_mask, output_gm,
                                                               id_val_scalar, offset_last_part):
    """
    modify float32 atomic add last part when last axis of input is one

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    ids_ub: ids_ub tensor
    last_mask: last part ele num
    output_gm: output_gm tensor
    id_val_scalar: id_val_scalar
    offset_last_part: offset to last part

    Returns
    -------
    None
    """
    id_val_fp32 = tik_inst.Scalar(Constant.DTYPE_FP32, "id_val_fp32")
    input_val = tik_inst.Scalar(Constant.DTYPE_FP32, "input_val")
    neg_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "neg_ub")
    tik_inst.vector_dup(Constant.MASK_FP32, neg_ub[0], Constant.NEG_ONE, 1, 1, 8)
    conv_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "conv_ub")
    tik_inst.vector_dup(Constant.MASK_FP32, conv_ub[0], Constant.NEG_ONE, 1, 1, 8)
    tik_inst.vconv(last_mask, "", conv_ub[0], ids_ub[offset_last_part], 1, 1, 1, 8, 8)
    with tik_inst.for_range(0, last_mask) as ids_index:
        # traversal ids
        id_val_fp32.set_as(conv_ub[ids_index])
        with tik_inst.if_scope(id_val_fp32 >= Constant.ZERO):
            # new id
            zero_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "zero_ub")
            tik_inst.vector_dup(Constant.MASK_FP32, zero_ub[0], Constant.ZERO, 1, 1, 8)
            dup_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "dup_ub")
            tik_inst.vector_dup(Constant.MASK_FP32, dup_ub[0], id_val_fp32, 1, 1, 8)
            cmpmask = tik_inst.vcmp_eq(Constant.MASK_FP32, dup_ub[0], conv_ub[0], 1, 1)
            tik_inst.vsel(Constant.MASK_FP32, 0, conv_ub[0], cmpmask, neg_ub[0], conv_ub[0], 1, 1, 1, 1, 8, 8, 8)
            sel_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "sel_ub")
            tik_inst.vsel(Constant.MASK_FP32, 0, sel_ub[0], cmpmask, input_ub[offset_last_part], zero_ub[0], 1, 1, 1, 1,
                          8, 8, 8)
            cadd_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "cadd_ub")
            tik_inst.vcadd(Constant.MASK_FP32, cadd_ub[0], sel_ub[0], 1, 1, 1, 8)
            input_val.set_as(cadd_ub[0])
            zero_ub[0].set_as(input_val)
            id_val_scalar.set_as(ids_ub[offset_last_part + ids_index])
            tik_inst.data_move(output_gm[id_val_scalar], zero_ub[0], 0, 1, 1, 0, 0)


def last_axis_one_modify_multi(tik_inst, input_ub, ids_ub, times_by_multi, output_gm, id_val_scalar):
    """
    last_axis_one_modify_multi
    """
    id_val_fp32 = tik_inst.Scalar(Constant.DTYPE_FP32, "id_val_fp32")
    neg_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "neg_ub")
    tik_inst.vector_dup(Constant.MASK_FP32, neg_ub[0], Constant.NEG_ONE, 1, 1, 8)
    multi = 4
    with tik_inst.for_range(0, times_by_multi) as index:
        # times divided by mask
        conv_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32 * multi,), tik.scope_ubuf, "conv_ub")
        tik_inst.vconv(Constant.MASK_FP32, "", conv_ub[0], ids_ub[index * multi * Constant.MASK_FP32], multi, 1, 1, 8,
                       8)
        with tik_inst.for_range(0, multi) as multi_index:
            with tik_inst.for_range(0, Constant.MASK_FP32) as ids_index:
                # traversal ids
                id_val_fp32.set_as(conv_ub[multi_index * Constant.MASK_FP32 + ids_index])
                with tik_inst.if_scope(id_val_fp32 >= Constant.ZERO):
                    output_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "output_ub")
                    tik_inst.vector_dup(Constant.MASK_FP32, output_ub[0], Constant.ZERO, 1, 1, 8)
                    # new id
                    zero_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "zero_ub")
                    tik_inst.vector_dup(Constant.MASK_FP32, zero_ub[0], Constant.ZERO, 1, 1, 8)
                    dup_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "dup_ub")
                    tik_inst.vector_dup(Constant.MASK_FP32, dup_ub[0], id_val_fp32, 1, 1, 8)
                    with tik_inst.for_range(multi_index, multi) as cmp_index:
                        cmpmask = tik_inst.vcmp_eq(Constant.MASK_FP32, dup_ub[0],
                                                   conv_ub[cmp_index * Constant.MASK_FP32], 1, 1)
                        tik_inst.vsel(Constant.MASK_FP32, 0, conv_ub[cmp_index * Constant.MASK_FP32], cmpmask,
                                      neg_ub[0], conv_ub[cmp_index * Constant.MASK_FP32], 1, 1, 1, 1, 8, 8, 8)
                        sel_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "sel_ub")
                        tik_inst.vsel(Constant.MASK_FP32, 0, sel_ub[0], cmpmask,
                                      input_ub[index * multi * Constant.MASK_FP32 + cmp_index * Constant.MASK_FP32],
                                      zero_ub[0], 1, 1, 1, 1, 8, 8, 8)
                        cadd_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "cadd_ub")
                        tik_inst.vector_dup(Constant.MASK_FP32, cadd_ub[0], Constant.ZERO, 1, 1, 8)
                        tik_inst.vcadd(Constant.MASK_FP32, cadd_ub[0], sel_ub[0], 1, 1, 1, 8)
                        tik_inst.vadd(Constant.MASK_FP32, output_ub[0], output_ub[0], cadd_ub[0], 1, 1, 1, 1, 8, 8, 8)
                    id_val_scalar.set_as(ids_ub[index * multi * Constant.MASK_FP32 + \
                                                multi_index * Constant.MASK_FP32 + ids_index])
                    tik_inst.data_move(output_gm[id_val_scalar], output_ub[0], 0, 1, 1, 0, 0)


def last_axis_one_modify_single(tik_inst, input_ub, ids_ub, times_by_mask, output_gm, id_val_scalar, offset):
    """
    last_axis_one_modify_single
    """
    id_val_fp32 = tik_inst.Scalar(Constant.DTYPE_FP32, "id_val_fp32")
    input_val = tik_inst.Scalar(Constant.DTYPE_FP32, "input_val")
    neg_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "neg_ub")
    tik_inst.vector_dup(Constant.MASK_FP32, neg_ub[0], Constant.NEG_ONE, 1, 1, 8)
    with tik_inst.for_range(0, times_by_mask) as index:
        # times divided by mask
        conv_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "conv_ub")
        tik_inst.vconv(Constant.MASK_FP32, "", conv_ub[0], ids_ub[offset + index * Constant.MASK_FP32], 1, 1, 1, 8, 8)
        with tik_inst.for_range(0, Constant.MASK_FP32) as ids_index:
            # traversal ids
            id_val_fp32.set_as(conv_ub[ids_index])
            with tik_inst.if_scope(id_val_fp32 >= Constant.ZERO):
                # new id
                zero_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "zero_ub")
                tik_inst.vector_dup(Constant.MASK_FP32, zero_ub[0], Constant.ZERO, 1, 1, 8)
                dup_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "dup_ub")
                tik_inst.vector_dup(Constant.MASK_FP32, dup_ub[0], id_val_fp32, 1, 1, 8)
                cmpmask = tik_inst.vcmp_eq(Constant.MASK_FP32, dup_ub[0], conv_ub[0], 1, 1)
                tik_inst.vsel(Constant.MASK_FP32, 0, conv_ub[0], cmpmask, neg_ub[0], conv_ub[0], 1, 1, 1, 1, 8, 8, 8)
                sel_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "sel_ub")
                tik_inst.vsel(Constant.MASK_FP32, 0, sel_ub[0], cmpmask, input_ub[offset + index * Constant.MASK_FP32],
                              zero_ub[0], 1, 1, 1, 1, 8, 8, 8)
                cadd_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.MASK_FP32,), tik.scope_ubuf, "cadd_ub")
                tik_inst.vcadd(Constant.MASK_FP32, cadd_ub[0], sel_ub[0], 1, 1, 1, 8)
                input_val.set_as(cadd_ub[0])
                zero_ub[0].set_as(input_val)
                id_val_scalar.set_as(ids_ub[offset + index * Constant.MASK_FP32 + ids_index])
                tik_inst.data_move(output_gm[id_val_scalar], zero_ub[0], 0, 1, 1, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, input_ub, output_gm, ub2gm_burst_len, input_ub_offset,
                                                output_gm_offset):
    """
    tik_atomic_add_ub2gm_by_id_last_axis_align

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    output_gm: output_gm tensor
    ub2gm_burst_len: ub2gm_burst_len
    input_ub_offset: input_ub_offset
    output_gm_offset: output_gm_offset

    Returns
    -------
    None
    """
    tik_inst.data_move(output_gm[output_gm_offset], input_ub[input_ub_offset], 0, 1, ub2gm_burst_len, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, input_ub, output_ub, output_gm, input_ub_offset,
                                                    output_ub_offset, output_gm_offset, vadd_mask):
    """
    tik_atomic_add_ub2gm_by_id_last_axis_not_align

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    output_ub: output_ub tensor
    output_gm: output_gm tensor
    input_ub_offset: input_ub_offset
    output_ub_offset: output_ub_offset
    output_gm_offset: output_gm_offset
    vadd_mask: vadd_mask

    Returns
    -------
    None
    """
    tik_inst.vadd(vadd_mask, output_ub[output_ub_offset], input_ub[input_ub_offset], output_ub[output_ub_offset], 1, 1,
                  1, 1, 8, 8, 8)
    tik_inst.data_move(output_gm[output_gm_offset], output_ub[output_ub_offset], 0, 1, 1, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(tik_inst, input_ub, output_ub, output_gm,
                                                                   input_ub_offset, output_gm_offset, vadd_mask):
    """
    tik_atomic_add_ub2gm_by_id_last_axis_not_align

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    output_ub: output_ub tensor
    output_gm: output_gm tensor
    input_ub_offset: input_ub_offset
    output_ub_offset: output_ub_offset
    output_gm_offset: output_gm_offset
    vadd_mask: vadd_mask

    Returns
    -------
    None
    """
    tik_inst.vector_dup(Constant.MASK_8, output_ub, 0, 1, 1, 8)
    tik_inst.vadd(vadd_mask, output_ub, input_ub[input_ub_offset], output_ub, 1, 1, 1, 1, 8, 8, 8)
    tik_inst.data_move(output_gm[output_gm_offset], output_ub, 0, 1, 1, 0, 0)


def _tik_atomic_add_last_axis_align_small_e(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                            obj_fp32_input_data_input_scalar, obj_fp32_e_num_input_scalar,
                                            obj_fp32_ids_input_scalar):
    """
    _tik_atomic_add_last_axis_align_small_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0, obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # front part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_front_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_front_core +
                                                     rows_index])
                            output_gm_offset = id_val_scalar * obj_fp32_e_num_input_scalar.e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # last part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.last_burst_len_front_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.last_rows_front_part_front_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_front_core +
                                                     rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # front part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_last_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                              obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_front_core +
                                                     rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar. \
                                                   e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # last part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar. \
                                              e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar. \
                            last_burst_len_last_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.last_rows_last_part_front_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                              obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_front_core +
                                                     rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar. \
                                                   e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # front part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar. \
                            front_burst_len_front_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = \
                                rows_index * \
                                obj_fp32_e_num_input_scalar. \
                                    e_num
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_last_core +
                                                     rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar. \
                                                   e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # last part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar. \
                            last_burst_len_front_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.last_rows_front_part_last_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                              obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_last_core +
                                                     rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        # front part ids last part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_last_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_last_part_last_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                              obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_last_core +
                                                     rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar. \
                                                   e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        # last part ids last part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.last_burst_len_last_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.last_rows_last_part_last_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                              obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_last_core +
                                                     rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)


def _tik_atomic_add_last_axis_one(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                  obj_fp32_input_data_input_scalar, obj_fp32_e_num_input_scalar,
                                  obj_fp32_ids_input_scalar, obj_fp32_output_init_input_scalar, fp32_ele_num_one_block):
    """
    _tik_atomic_add_last_axis_one

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_fp32_output_init_input_scalar: obj_fp32_output_init_input_scalar
    fp32_ele_num_one_block: fp32_ele_num_one_block

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input front part front core
                input_offset_gm = block_index * \
                                  obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_fp32_input_data_input_scalar. \
                    front_burst_len_front_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # init output
                _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub,
                                    obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                                    obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                # ub2gm by id
                _tik_atomic_add_ub2gm_by_id_last_axis_one(
                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub, obj_ub_tensor.output_ub,
                    obj_gm_tensor.output_gm, obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                    obj_fp32_input_data_input_scalar.front_rows_front_part_front_core, fp32_ele_num_one_block,
                    id_val_scalar)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input front part front core
                input_offset_gm = block_index * \
                                  obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_fp32_input_data_input_scalar. \
                    front_burst_len_last_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # init output
                _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub,
                                    obj_fp32_output_init_input_scalar.last_repeat_time_last_part_front_core,
                                    obj_fp32_output_init_input_scalar.init_times_last_part_front_core)
                # ub2gm by id
                _tik_atomic_add_ub2gm_by_id_last_axis_one(
                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub, obj_ub_tensor.output_ub,
                    obj_gm_tensor.output_gm, obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                    obj_fp32_input_data_input_scalar.front_rows_last_part_front_core, fp32_ele_num_one_block,
                    id_val_scalar)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input front part last core
                input_offset_gm = block_index * \
                                  obj_fp32_input_data_input_scalar. \
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar. \
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_fp32_input_data_input_scalar. \
                    front_burst_len_front_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # init output
                _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub,
                                    obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                    obj_fp32_output_init_input_scalar.init_times_front_part_last_core)
                # ub2gm by id
                _tik_atomic_add_ub2gm_by_id_last_axis_one(
                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub, obj_ub_tensor.output_ub,
                    obj_gm_tensor.output_gm, obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                    obj_fp32_input_data_input_scalar.front_rows_front_part_last_core, fp32_ele_num_one_block,
                    id_val_scalar)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input last part last core
                input_offset_gm = block_index * \
                                  obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_fp32_input_data_input_scalar. \
                    front_burst_len_last_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # init output
                _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub,
                                    obj_fp32_output_init_input_scalar.last_repeat_time_last_part_last_core,
                                    obj_fp32_output_init_input_scalar.init_times_last_part_last_core)
                # ub2gm by id
                _tik_atomic_add_ub2gm_by_id_last_axis_one(
                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub, obj_ub_tensor.output_ub,
                    obj_gm_tensor.output_gm, obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                    obj_fp32_input_data_input_scalar.front_rows_last_part_last_core, fp32_ele_num_one_block,
                    id_val_scalar)


def _tik_atomic_add_last_axis_one_modify(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                         obj_input_data_scalar, obj_fp32_ids_input_scalar, obj_output_init_scalar):
    """
    modify float32 atomic add when last axis of input is one

    Parameters
    ----------
    block_index: block index
    tik_inst: tik instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_input_data_scalar: obj_input_data_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_output_init_scalar: obj_output_init_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input front part front core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar. \
                                      ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar. \
                                      ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar. \
                    front_burst_len_front_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # cadd front part front core
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(
                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.init_times_front_part_front_core, obj_gm_tensor.output_gm, id_val_scalar)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input last part front core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar. \
                                      ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar. \
                                      ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar. \
                    front_burst_len_last_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # cadd last part front core
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub,
                                                                 obj_output_init_scalar.init_times_last_part_front_core,
                                                                 obj_gm_tensor.output_gm, id_val_scalar)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            # ids tiling by ub last core
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input front part last core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar. \
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar. \
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar. \
                    front_burst_len_front_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # cadd front part last core
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub,
                                                                 obj_output_init_scalar.init_times_front_part_last_core,
                                                                 obj_gm_tensor.output_gm, id_val_scalar)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input last part last core
                input_offset_gm = block_index * \
                                  obj_fp32_ids_input_scalar. \
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar. \
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.front_burst_len_last_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # cadd last part last core
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(
                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.init_times_last_part_last_core - 1, obj_gm_tensor.output_gm, id_val_scalar)
                # last part
                offset_last_part = (obj_output_init_scalar.
                                    init_times_last_part_last_core - 1) * \
                                   Constant.MASK_FP32
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify_last_part(tik_inst, obj_ub_tensor.input_ub,
                                                                           obj_ub_tensor.ids_ub,
                                                                           obj_output_init_scalar.last_part_vadd_mask,
                                                                           obj_gm_tensor.output_gm, id_val_scalar,
                                                                           offset_last_part)


def _tik_atomic_add_last_axis_one_multi(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                        obj_input_data_scalar, obj_fp32_ids_input_scalar, obj_output_init_scalar):
    """
    _tik_atomic_add_last_axis_one_multi
    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input front part front core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar. \
                                      ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar. \
                                      ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar. \
                    front_burst_len_front_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # cadd front part front core
                # multi 64 part
                with tik_inst.if_scope(obj_output_init_scalar.init_times_front_part_front_core > 0):
                    last_axis_one_modify_multi(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub,
                                               obj_output_init_scalar.init_times_front_part_front_core,
                                               obj_gm_tensor.output_gm, id_val_scalar)
                # single 64 part
                offset_multi = obj_output_init_scalar.init_times_front_part_front_core * \
                               Constant.MULTI * Constant.MASK_FP32
                times_by_mask = obj_output_init_scalar.last_repeat_time_front_part_front_core
                with tik_inst.if_scope(times_by_mask > 0):
                    last_axis_one_modify_single(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub, times_by_mask,
                                                obj_gm_tensor.output_gm, id_val_scalar, offset_multi)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input last part front core
                input_offset_gm = block_index * obj_input_data_scalar.ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.front_burst_len_last_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # cadd last part front core
                # multi 64 part
                with tik_inst.if_scope(obj_output_init_scalar.init_times_last_part_front_core > 0):
                    last_axis_one_modify_multi(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub,
                                               obj_output_init_scalar.init_times_last_part_front_core,
                                               obj_gm_tensor.output_gm, id_val_scalar)
                # single 64 part
                offset_multi = obj_output_init_scalar.init_times_last_part_front_core * \
                               Constant.MULTI * Constant.MASK_FP32
                times_by_mask = obj_output_init_scalar.last_repeat_time_last_part_front_core
                with tik_inst.if_scope(times_by_mask > 0):
                    last_axis_one_modify_single(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub, times_by_mask,
                                                obj_gm_tensor.output_gm, id_val_scalar, offset_multi)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0, obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            # ids tiling by ub last core
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids front part last core
                ids_offset_gm = block_index * obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input front part last core
                input_offset_gm = block_index * obj_input_data_scalar.ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.front_burst_len_front_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # cadd front part last core
                # multi 64 part
                with tik_inst.if_scope(obj_output_init_scalar.init_times_front_part_last_core > 0):
                    last_axis_one_modify_multi(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub,
                                               obj_output_init_scalar.init_times_front_part_last_core,
                                               obj_gm_tensor.output_gm, id_val_scalar)
                # single 64 part
                offset_multi = obj_output_init_scalar.init_times_front_part_last_core * \
                               Constant.MULTI * Constant.MASK_FP32
                times_by_mask = obj_output_init_scalar.last_repeat_time_front_part_last_core
                with tik_inst.if_scope(times_by_mask > 0):
                    last_axis_one_modify_single(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub, times_by_mask,
                                                obj_gm_tensor.output_gm, id_val_scalar, offset_multi)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids last part last core
                ids_offset_gm = block_index * obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                # input last part last core
                input_offset_gm = block_index * \
                                  obj_fp32_ids_input_scalar. \
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar. \
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.front_burst_len_last_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, input_offset_gm,
                                              input_offset_ub, input_n_burst, input_burst_len)
                # cadd last part last core
                # multi 64 part
                with tik_inst.if_scope(obj_output_init_scalar.init_times_last_part_last_core > 0):
                    last_axis_one_modify_multi(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub,
                                               obj_output_init_scalar.init_times_last_part_last_core,
                                               obj_gm_tensor.output_gm, id_val_scalar)
                # single 64 part
                offset_multi = obj_output_init_scalar.init_times_last_part_last_core * \
                               Constant.MULTI * Constant.MASK_FP32
                times_by_mask = obj_output_init_scalar.last_repeat_time_last_part_last_core
                with tik_inst.if_scope(times_by_mask > 0):
                    last_axis_one_modify_single(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub, times_by_mask,
                                                obj_gm_tensor.output_gm, id_val_scalar, offset_multi)
                # last mask part
                with tik_inst.if_scope(obj_output_init_scalar.last_part_vadd_mask > 0):
                    offset_last_part = offset_multi + times_by_mask * Constant.MASK_FP32
                    _tik_atomic_add_ub2gm_by_id_last_axis_one_modify_last_part(
                        tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.ids_ub,
                        obj_output_init_scalar.last_part_vadd_mask, obj_gm_tensor.output_gm, id_val_scalar,
                        offset_last_part)


def _tik_atomic_add_last_axis_not_align_small_e(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                                obj_fp32_input_data_input_scalar, obj_fp32_e_num_input_scalar,
                                                obj_fp32_ids_input_scalar, obj_fp32_output_init_input_scalar,
                                                fp32_ele_num_one_block):
    """
    _tik_atomic_add_last_axis_not_align_small_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_fp32_output_init_input_scalar: obj_fp32_output_init_input_scalar
    fp32_ele_num_one_block: fp32_ele_num_one_block

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # front part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.front_rows_front_part_front_core
                        input_ele_num_one_row = obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub,
                                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                                            obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_front_core +
                                                     rows_index])
                            # align part
                            with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                              obj_fp32_output_init_input_scalar.\
                                              last_axis_align_floor + \
                                              obj_fp32_output_init_input_scalar.\
                                              last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               obj_fp32_output_init_input_scalar.\
                                               last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                            obj_ub_tensor.output_ub,
                                                                            obj_gm_tensor.output_gm, input_ub_offset,
                                                                            output_ub_offset, output_gm_offset,
                                                                            vadd_mask)
                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # last part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.last_rows_front_part_front_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        _tik_init_ub_tensor(
                            tik_inst, obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.last_repeat_time_last_row_front_part_front_core,
                            obj_fp32_output_init_input_scalar.init_times_last_row_front_part_front_core)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.last_rows_front_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_front_core +
                                                     rows_index])
                            # align part
                            with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                              obj_fp32_output_init_input_scalar. \
                                                  last_axis_align_floor + \
                                              obj_fp32_output_init_input_scalar. \
                                                  last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               obj_fp32_output_init_input_scalar.\
                                               last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                            obj_ub_tensor.output_ub,
                                                                            obj_gm_tensor.output_gm, input_ub_offset,
                                                                            output_ub_offset, output_gm_offset,
                                                                            vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # front part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.front_rows_last_part_front_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub,
                                            obj_fp32_output_init_input_scalar.last_repeat_time_last_part_front_core,
                                            obj_fp32_output_init_input_scalar.init_times_last_part_front_core)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_front_core +
                                                     rows_index])
                            # align part
                            with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                              obj_fp32_output_init_input_scalar. \
                                                  last_axis_align_floor + \
                                              obj_fp32_output_init_input_scalar. \
                                                  last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               obj_fp32_output_init_input_scalar. \
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar. \
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                            obj_ub_tensor.output_ub,
                                                                            obj_gm_tensor.output_gm, input_ub_offset,
                                                                            output_ub_offset, output_gm_offset,
                                                                            vadd_mask)
                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # last part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.last_rows_last_part_front_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        _tik_init_ub_tensor(
                            tik_inst, obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.last_repeat_time_last_row_last_part_front_core,
                            obj_fp32_output_init_input_scalar.init_times_last_row_last_part_front_core)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.last_rows_last_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_front_core +
                                                     rows_index])
                            # align part
                            with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                              obj_fp32_output_init_input_scalar.last_axis_align_floor + \
                                              obj_fp32_output_init_input_scalar.\
                                              last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               obj_fp32_output_init_input_scalar.\
                                               last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                            obj_ub_tensor.output_ub,
                                                                            obj_gm_tensor.output_gm, input_ub_offset,
                                                                            output_ub_offset, output_gm_offset,
                                                                            vadd_mask)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # front part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar. \
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.front_rows_front_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub,
                                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                            obj_fp32_output_init_input_scalar.init_times_front_part_last_core)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_last_core +
                                                     rows_index])
                            # align part
                            with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                              obj_fp32_output_init_input_scalar.last_axis_align_floor + \
                                              obj_fp32_output_init_input_scalar.last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               obj_fp32_output_init_input_scalar.last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                            obj_ub_tensor.output_ub,
                                                                            obj_gm_tensor.output_gm, input_ub_offset,
                                                                            output_ub_offset, output_gm_offset,
                                                                            vadd_mask)
                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # last part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.last_rows_front_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        _tik_init_ub_tensor(
                            tik_inst, obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.last_repeat_time_last_row_front_part_last_core,
                            obj_fp32_output_init_input_scalar.init_times_last_row_front_part_last_core)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.last_rows_front_part_last_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_last_core +
                                                     rows_index])
                            # align part
                            with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                              obj_fp32_output_init_input_scalar.last_axis_align_floor + \
                                              obj_fp32_output_init_input_scalar.last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               obj_fp32_output_init_input_scalar. \
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar. \
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                            obj_ub_tensor.output_ub,
                                                                            obj_gm_tensor.output_gm, input_ub_offset,
                                                                            output_ub_offset, output_gm_offset,
                                                                            vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        # front part ids last part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar. \
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar. \
                                              e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar. \
                            front_rows_last_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar. \
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub,
                                            obj_fp32_output_init_input_scalar.last_repeat_time_last_part_last_core,
                                            obj_fp32_output_init_input_scalar.init_times_last_part_last_core)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_last_part_last_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_last_core +
                                                     rows_index])
                            # align part
                            with tik_inst.if_scope(
                                    obj_fp32_e_num_input_scalar. \
                                            e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                              obj_fp32_output_init_input_scalar. \
                                                  last_axis_align_floor + \
                                              obj_fp32_output_init_input_scalar. \
                                                  last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               obj_fp32_output_init_input_scalar. \
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar. \
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                            obj_ub_tensor.output_ub,
                                                                            obj_gm_tensor.output_gm, input_ub_offset,
                                                                            output_ub_offset, output_gm_offset,
                                                                            vadd_mask)
                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = \
                            obj_fp32_e_num_input_scalar. \
                                e_ub2gm_front_burst_len + \
                            obj_fp32_e_num_input_scalar. \
                                e_ub2gm_last_burst_len
                        input_mov_times = \
                            obj_fp32_input_data_input_scalar. \
                                last_rows_last_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar. \
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        _tik_init_ub_tensor(
                            tik_inst, obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.last_repeat_time_last_row_last_part_last_core,
                            obj_fp32_output_init_input_scalar.init_times_last_row_last_part_last_core)
                        with tik_inst.for_range(
                                0, obj_fp32_input_data_input_scalar.last_rows_last_part_last_core) as rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_last_core +
                                                     rows_index])
                            # align part
                            with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                              obj_fp32_output_init_input_scalar. \
                                                  last_axis_align_floor + \
                                              obj_fp32_output_init_input_scalar. \
                                                  last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               obj_fp32_output_init_input_scalar. \
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar. \
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                            obj_ub_tensor.output_ub,
                                                                            obj_gm_tensor.output_gm, input_ub_offset,
                                                                            output_ub_offset, output_gm_offset,
                                                                            vadd_mask)


def _tik_atomic_add_last_axis_align_big_e(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                          obj_fp32_input_data_input_scalar, obj_fp32_e_num_input_scalar,
                                          obj_fp32_ids_input_scalar):
    """
    _tik_atomic_add_last_axis_align_big_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                                            obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar. \
                                                  ele_num_front_core + \
                                              input_mov_tims_front_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                        with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar. \
                                                  ele_num_front_core + \
                                              input_mov_tims_front_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar. \
                                                  e_num + \
                                              e_mov_index * obj_fp32_e_num_input_scalar. \
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar. \
                                e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar. \
                                                   e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len, input_ub_offset, output_gm_offset)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                                            obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar. \
                                                  ele_num_front_core + \
                                              input_mov_tims_last_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar. \
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar. \
                                    e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                        with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_last_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * obj_fp32_e_num_input_scalar. \
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar. \
                                e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len, input_ub_offset, output_gm_offset)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                                            obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_front_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                        with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar. \
                                                  ele_num_front_core + \
                                              input_mov_tims_front_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar. \
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len, input_ub_offset, output_gm_offset)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                                            obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_last_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar. \
                                                   e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                        with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_last_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar. \
                                e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len, input_ub_offset, output_gm_offset)


def _tik_atomic_add_num_segment_one(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                    obj_fp32_input_data_input_scalar, obj_fp32_e_num_input_scalar,
                                    obj_fp32_ids_input_scalar):
    """
    _tik_atomic_add_num_segment_one

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar

    Returns
    -------
    None
    """
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0, obj_fp32_ids_input_scalar.ele_num_front_core) as i:
            with tik_inst.for_range(0, obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as e_mov_index:
                with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                    input_offset_gm = (block_index * obj_fp32_ids_input_scalar.ele_num_front_core +
                                       i) * obj_fp32_e_num_input_scalar.e_num + e_mov_index * \
                                      obj_fp32_e_num_input_scalar.e_num_front_part
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    output_gm_offset = e_mov_index * obj_fp32_e_num_input_scalar.e_num_front_part
                    _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.input_ub,
                                                                obj_gm_tensor.output_gm,
                                                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, 0,
                                                                output_gm_offset)
                with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                    input_offset_gm = (block_index * obj_fp32_ids_input_scalar.ele_num_front_core +
                                       i) * obj_fp32_e_num_input_scalar.e_num + e_mov_index * \
                                      obj_fp32_e_num_input_scalar.e_num_front_part
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    output_gm_offset = e_mov_index * obj_fp32_e_num_input_scalar.e_num_front_part
                    _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.input_ub,
                                                                obj_gm_tensor.output_gm, input_burst_len, 0,
                                                                output_gm_offset)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0, obj_fp32_ids_input_scalar.ele_num_last_core) as i:
            with tik_inst.for_range(0, obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as e_mov_index:
                with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                    input_offset_gm = (block_index * obj_fp32_ids_input_scalar.ele_num_front_core +
                                       i) * obj_fp32_e_num_input_scalar.e_num + e_mov_index * \
                                      obj_fp32_e_num_input_scalar.e_num_front_part
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    output_gm_offset = e_mov_index * obj_fp32_e_num_input_scalar.e_num_front_part
                    _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.input_ub,
                                                                obj_gm_tensor.output_gm,
                                                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, 0,
                                                                output_gm_offset)
                with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                    input_offset_gm = (block_index * obj_fp32_ids_input_scalar.ele_num_front_core +
                                       i) * obj_fp32_e_num_input_scalar.e_num + e_mov_index * \
                                      obj_fp32_e_num_input_scalar.e_num_front_part
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    output_gm_offset = e_mov_index * obj_fp32_e_num_input_scalar.e_num_front_part
                    _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.input_ub,
                                                                obj_gm_tensor.output_gm, input_burst_len, 0,
                                                                output_gm_offset)


def _tik_atomic_add_last_axis_not_align_big_e(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                              obj_fp32_input_data_input_scalar, obj_fp32_e_num_input_scalar,
                                              obj_fp32_ids_input_scalar, obj_fp32_output_init_input_scalar):
    """
    _tik_atomic_add_last_axis_not_align_big_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_fp32_output_init_input_scalar: obj_fp32_output_init_input_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                                            obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_front_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar. \
                                                  e_num + \
                                              e_mov_index * obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar. \
                                                   e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                        with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar. \
                                                  ele_num_front_core + \
                                              input_mov_tims_front_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar. \
                                                  e_num + \
                                              e_mov_index * obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.e_gm2ub_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len, input_ub_offset, output_gm_offset)
                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                            with tik_inst.if_scope(vadd_mask > 0):
                                _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub, 1, 1)
                                input_ub_offset = \
                                    obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                output_ub_offset = 0
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   e_mov_index * \
                                                   obj_fp32_e_num_input_scalar.e_num_front_part + input_ub_offset

                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                                obj_ub_tensor.output_ub,
                                                                                obj_gm_tensor.output_gm,
                                                                                input_ub_offset, output_ub_offset,
                                                                                output_gm_offset, vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                                            obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_last_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar. \
                                                   e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                        with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar. \
                                                  ele_num_front_core + \
                                              input_mov_tims_last_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.e_gm2ub_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len, input_ub_offset, output_gm_offset)
                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                            with tik_inst.if_scope(vadd_mask > 0):
                                _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub, 1, 1)
                                input_ub_offset = \
                                    obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                output_ub_offset = 0
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   e_mov_index * \
                                                   obj_fp32_e_num_input_scalar.e_num_front_part + input_ub_offset

                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                                obj_ub_tensor.output_ub,
                                                                                obj_gm_tensor.output_gm,
                                                                                input_ub_offset, output_ub_offset,
                                                                                output_gm_offset, vadd_mask)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = \
                    obj_fp32_ids_input_scalar.front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                                            obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_front_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                        with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_front_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.e_gm2ub_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len, input_ub_offset, output_gm_offset)
                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                            with tik_inst.if_scope(vadd_mask > 0):
                                _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub, 1, 1)
                                input_ub_offset = \
                                    obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                output_ub_offset = 0
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   e_mov_index * \
                                                   obj_fp32_e_num_input_scalar.e_num_front_part + input_ub_offset

                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                                obj_ub_tensor.output_ub,
                                                                                obj_gm_tensor.output_gm,
                                                                                input_ub_offset, output_ub_offset,
                                                                                output_gm_offset, vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                                            obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(e_mov_index < obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_last_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset, output_gm_offset)
                        with tik_inst.if_scope(e_mov_index == obj_fp32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                              obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                              input_mov_tims_last_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.e_num + \
                                              e_mov_index * \
                                              obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.e_gm2ub_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.e_num + \
                                               e_mov_index * \
                                               obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len, input_ub_offset, output_gm_offset)
                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                            with tik_inst.if_scope(vadd_mask > 0):
                                _tik_init_ub_tensor(tik_inst, obj_ub_tensor.output_ub, 1, 1)
                                input_ub_offset = \
                                    obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                output_ub_offset = 0
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   e_mov_index * \
                                                   obj_fp32_e_num_input_scalar.e_num_front_part + input_ub_offset

                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst, obj_ub_tensor.input_ub,
                                                                                obj_ub_tensor.output_ub,
                                                                                obj_gm_tensor.output_gm,
                                                                                input_ub_offset, output_ub_offset,
                                                                                output_gm_offset, vadd_mask)


# 'pylint: disable=too-many-arguments
def _tik_atomic_add_last_axis_align_small_e_hp_copy_slice_data(tik_inst, dst_ub, src_ub, ids_base_offset, row_num,
                                                               ids_dsize, back_offset):
    e_num_pre_block = 8
    head_align_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="head_align_offset")
    head_align_offset.set_as(ids_base_offset // e_num_pre_block * e_num_pre_block)
    back_offset.set_as(ids_base_offset - head_align_offset)
    copy_block_num = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="copy_block_num")
    copy_block_num.set_as(((row_num + back_offset) * ids_dsize + constant.BLOCK_SIZE - 1) // constant.BLOCK_SIZE)
    tik_inst.data_move(dst_ub, src_ub[head_align_offset], 0, 1, copy_block_num, 0, 0)


# 'pylint: disable=too-many-arguments
def _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(tik_inst, obj_ub_tensor, obj_common_scalar, row_num,
                                                             prama_ids_base_offset, back_offset):
    """
    compute for cache data
    """
    indices_value_dup_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                           name="indice_dup_ub",
                                           scope=tik.scope_ubuf)
    idx_value_dup_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                       name="indice_dup_ub",
                                       scope=tik.scope_ubuf)
    is_eq_ub = tik_inst.Tensor(Constant.DTYPE_UINT32, (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                               name="is_eq_ub",
                               scope=tik.scope_ubuf)
    zero_dup_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                  name="zero_dup_ub",
                                  scope=tik.scope_ubuf)
    invalid_dup_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                     name="invalid_dup_ub",
                                     scope=tik.scope_ubuf)
    tik_inst.vec_dup(constant.MASK64, zero_dup_ub, 0.0, 1, 8)
    tik_inst.vec_dup(constant.MASK64, invalid_dup_ub, Constant.NEG_ONE, 1, 8)
    indices_value_fp32 = tik_inst.Scalar(dtype=Constant.DTYPE_FP32, name="indices_value_fp32")
    ids_base_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="ids_base_offset")
    ids_base_offset.set_as(prama_ids_base_offset)
    with tik_inst.if_scope(obj_common_scalar.simpling_step != 0):
        _tik_atomic_add_last_axis_align_small_e_hp_copy_slice_data(tik_inst, obj_ub_tensor.indices_temp_float_ub,
                                                                   obj_ub_tensor.indices_index_float_ub,
                                                                   ids_base_offset, row_num,
                                                                   obj_common_scalar.ids_dsize, back_offset)
        repeat_times = tik_inst.Scalar(Constant.DTYPE_INT32,
                                       name="repeat_times",
                                       init_value=(row_num + back_offset + constant.MASK64 - 1) // constant.MASK64)
        idx_num_cur_batch_tail = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="idx_num_cur_batch_tail")
        idx_num_cur_batch_tail.set_as((row_num + back_offset) -
                                      (row_num + back_offset) // constant.MASK64 * constant.MASK64)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            with tik_inst.for_range(0, Constant.CACHE_ACT_SIMPLING_NUM) as idx_simpling_i:
                indices_value_fp32.set_as(obj_ub_tensor.indices_simpling_ub[idx_simpling_i])
                with tik_inst.if_scope(indices_value_fp32 != Constant.NEG_ONE):
                    tik_inst.vec_dup(constant.MASK64, indices_value_dup_ub, indices_value_fp32, 1, 8)
                    tik_inst.vec_cmpv_eq(is_eq_ub, indices_value_dup_ub, obj_ub_tensor.indices_simpling_ub, 1, 8, 8)
                    tik_inst.vec_sel(Constant.CACHE_ACT_SIMPLING_NUM, 0, obj_ub_tensor.indices_simpling_ub, \
                                     is_eq_ub, invalid_dup_ub, obj_ub_tensor.indices_simpling_ub, 1, 8, 8, 8)
                    idx_value = tik_inst.Scalar(dtype=Constant.DTYPE_FP32,
                                                name="idx_value",
                                                init_value=-(idx_simpling_i + 1))
                    tik_inst.vec_dup(constant.MASK64, idx_value_dup_ub, idx_value, 1, 8)
                    with tik_inst.if_scope((repeat_times - 1) != 0):
                        tik_inst.vec_cmpv_eq(is_eq_ub, indices_value_dup_ub,
                                             obj_ub_tensor.indices_temp_float_ub[0], repeat_times - 1, 0, 8)
                        tik_inst.vec_sel(constant.MASK64, 2,
                                         obj_ub_tensor.indices_temp_float_ub[0], is_eq_ub,
                                         idx_value_dup_ub,
                                         obj_ub_tensor.indices_temp_float_ub[0], repeat_times - 1, 8, 0, 8)
                    with tik_inst.if_scope(idx_num_cur_batch_tail != 0):
                        tik_inst.vec_cmpv_eq(is_eq_ub, indices_value_dup_ub,
                                             obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * (repeat_times - 1)],
                                             1, 8, 8)
                        tik_inst.vec_sel(idx_num_cur_batch_tail, 2,
                                         obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * (repeat_times - 1)],
                                         is_eq_ub, idx_value_dup_ub,
                                         obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * (repeat_times - 1)],
                                         1, 8, 8, 8)
            #cmp
            tik_inst.vec_cmpv_gt(is_eq_ub, obj_ub_tensor.indices_temp_float_ub[0],
                                 zero_dup_ub, repeat_times, 8, 0)
            tik_inst.vec_sel(constant.MASK64, 2, obj_ub_tensor.indices_temp_float_ub[0],
                             is_eq_ub, zero_dup_ub, obj_ub_tensor.indices_temp_float_ub[0],
                             repeat_times, 8, 0, 8)
            tik_inst.vec_abs(constant.MASK64, obj_ub_tensor.indices_temp_float_ub, obj_ub_tensor.indices_temp_float_ub,
                             repeat_times, 8, 8)
            tik_inst.h_cast(obj_ub_tensor.indices_temp_int_ub, obj_ub_tensor.indices_temp_float_ub, 'round')
        else:
            with tik_inst.for_range(0, Constant.CACHE_ACT_SIMPLING_NUM) as idx_simpling_i:
                indices_value_fp32.set_as(obj_ub_tensor.indices_simpling_ub[idx_simpling_i])
                with tik_inst.if_scope(indices_value_fp32 != Constant.NEG_ONE):
                    tik_inst.vec_dup(constant.MASK64, indices_value_dup_ub, indices_value_fp32, 1, 8)
                    tik_inst.vec_cmpv_eq(is_eq_ub, indices_value_dup_ub, obj_ub_tensor.indices_simpling_ub, 1, 8, 8)
                    tik_inst.vec_sel(Constant.CACHE_ACT_SIMPLING_NUM, 0, obj_ub_tensor.indices_simpling_ub, \
                                     is_eq_ub, invalid_dup_ub, obj_ub_tensor.indices_simpling_ub, 1, 8, 8, 8)
                    idx_value = tik_inst.Scalar(dtype=Constant.DTYPE_FP32,
                                                name="idx_value",
                                                init_value=-(idx_simpling_i + 1))
                    tik_inst.vec_dup(constant.MASK64, idx_value_dup_ub, idx_value, 1, 8)
                    with tik_inst.for_range(0, repeat_times) as idx_repeat:
                        tik_inst.vec_cmpv_eq(is_eq_ub, indices_value_dup_ub,
                                             obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * idx_repeat],
                                             1, 8, 8)
                        with tik_inst.if_scope(tik.all(idx_num_cur_batch_tail != 0, idx_repeat == repeat_times - 1)):
                            tik_inst.vec_sel(idx_num_cur_batch_tail, 0,
                                             obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * idx_repeat],
                                             is_eq_ub, idx_value_dup_ub,
                                             obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * idx_repeat],
                                             1, 8, 8, 8)
                        with tik_inst.else_scope():
                            tik_inst.vec_sel(constant.MASK64, 0,
                                             obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * idx_repeat],
                                             is_eq_ub,
                                             idx_value_dup_ub,
                                             obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * idx_repeat],
                                             1, 8, 8, 8)
            #cmp
            with tik_inst.for_range(0, repeat_times) as idx_repeat:
                tik_inst.vec_cmpv_gt(is_eq_ub, obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * idx_repeat],
                                     zero_dup_ub, 1, 8, 8)
                tik_inst.vec_sel(constant.MASK64, 0,
                                 obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * idx_repeat],
                                 is_eq_ub, zero_dup_ub,
                                 obj_ub_tensor.indices_temp_float_ub[constant.MASK64 * idx_repeat],
                                 1, 8, 8, 8)
            tik_inst.vec_abs(constant.MASK64, obj_ub_tensor.indices_temp_float_ub,
                             obj_ub_tensor.indices_temp_float_ub,
                             repeat_times, 8, 8)
            tik_inst.h_cast(obj_ub_tensor.indices_temp_int_ub, obj_ub_tensor.indices_temp_float_ub, 'round')


# 'pylint: disable=too-many-arguments
def _tik_atomic_add_last_axis_align_small_e_hp_calc_output_data(tik_inst, obj_ub_tensor, obj_gm_tensor,
                                                                obj_fp32_output_init_input_scalar, obj_common_scalar,
                                                                obj_fp32_e_num_input_scalar, row_num, back_offset):
    id_val_scalar = obj_common_scalar.id_val_scalar
    mask_num = tik_inst.Scalar(dtype=Constant.DTYPE_INT32,
                               name="mask_num",
                               init_value=Constant.BYTE_REPEAT_BLOCK // obj_common_scalar.input_dsize)
    tik_inst.vec_dup(mask_num, obj_ub_tensor.cached_ub, 0.0,
                     _ceil_div((Constant.CACHE_ACT_SIMPLING_NUM + 1) * obj_fp32_e_num_input_scalar.e_num, mask_num), 8)
    with tik_inst.for_range(0, row_num) as rows_index:
        # visit cachee ids
        input_ub_offset = rows_index * obj_fp32_e_num_input_scalar.e_num
        id_val_scalar.set_as(obj_ub_tensor.indices_temp_int_ub[back_offset + rows_index])
        with tik_inst.if_scope(id_val_scalar > 0):
            args_add = (tik_inst, obj_ub_tensor.cached_ub, obj_ub_tensor.input_ub, input_ub_offset,
                        obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                        obj_fp32_output_init_input_scalar.last_part_vadd_mask,
                        obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
            _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar, obj_fp32_e_num_input_scalar.e_num)
        with tik_inst.else_scope():
            out_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32,
                                         name="out_offset",
                                         init_value=obj_common_scalar.ori_ids_base_offset)
            out_offset.set_as(obj_ub_tensor.ids_ub[out_offset + rows_index])
            output_gm_offset = out_offset * obj_fp32_e_num_input_scalar.e_num
            _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                        input_ub_offset, output_gm_offset)

    with tik_inst.for_range(0, Constant.CACHE_ACT_SIMPLING_NUM) as sample_idx:
        out_put_cache_ub_offset = (sample_idx + 1) * obj_fp32_e_num_input_scalar.e_num
        output_gm_offset = obj_common_scalar.cache_simpling_data_arr[sample_idx] * obj_fp32_e_num_input_scalar.e_num
        _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.cached_ub, obj_gm_tensor.output_gm,
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                    out_put_cache_ub_offset, output_gm_offset)


# 'pylint: disable=too-many-arguments
def _tik_atomic_add_last_axis_not_align_small_e_hp_output_data(tik_inst, obj_ub_tensor, obj_gm_tensor,
                                                               obj_fp32_output_init_input_scalar, obj_common_scalar,
                                                               obj_fp32_e_num_input_scalar, row_num, back_offset):
    move_pad = tbe_platform.api_check_support("tik.data_move_pad")
    id_val_scalar = obj_common_scalar.id_val_scalar
    mask_num = 64
    tik_inst.vec_dup(
        mask_num, obj_ub_tensor.cached_ub, 0.0,
        _ceil_div((Constant.CACHE_ACT_SIMPLING_NUM + 1) * obj_fp32_output_init_input_scalar.last_axis_align_floor,
                  mask_num), 8)
    with tik_inst.for_range(0, row_num) as rows_index:
        # visit cachee ids
        input_ub_offset = rows_index * obj_fp32_output_init_input_scalar.last_axis_align_floor
        id_val_scalar.set_as(obj_ub_tensor.indices_temp_int_ub[back_offset + rows_index])
        with tik_inst.if_scope(id_val_scalar > 0):
            args_add = (
                tik_inst,
                obj_ub_tensor.cached_ub,
                obj_ub_tensor.input_ub,
                input_ub_offset,
                obj_fp32_e_num_input_scalar.vadd_repeat_last,
                obj_fp32_e_num_input_scalar.vadd_repeat_64,
                obj_fp32_e_num_input_scalar.vadd_repeat_255)
            _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar,
                                         obj_fp32_output_init_input_scalar.last_axis_align_floor)
        with tik_inst.else_scope():
            out_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32,
                                         name="out_offset",
                                         init_value=obj_common_scalar.ori_ids_base_offset)
            out_offset.set_as(obj_ub_tensor.ids_ub[out_offset + rows_index])
            output_gm_offset = out_offset * obj_fp32_e_num_input_scalar.e_num
            if not move_pad:
                with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                    _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.input_ub,
                                                                obj_gm_tensor.output_gm,
                                                                obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                                input_ub_offset, output_gm_offset)
                    # last part
                vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                input_ub_offset1 = input_ub_offset + obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len * 8
                output_gm_offset1 = output_gm_offset + obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len * 8
                _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(tik_inst, obj_ub_tensor.input_ub,
                                                                               obj_ub_tensor.output_ub,
                                                                               obj_gm_tensor.output_gm,
                                                                               input_ub_offset1, output_gm_offset1,
                                                                               vadd_mask)
            else:
                tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                       obj_ub_tensor.input_ub[input_ub_offset], 1, obj_fp32_e_num_input_scalar.move_pad,
                                       0, 0)

    if not move_pad:
        with tik_inst.for_range(0, Constant.CACHE_ACT_SIMPLING_NUM) as sample_idx:
            out_put_cache_ub_offset = (sample_idx + 1) * obj_fp32_output_init_input_scalar.last_axis_align_floor
            output_gm_offset = obj_common_scalar.cache_simpling_data_arr[sample_idx] * obj_fp32_e_num_input_scalar.e_num
            with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.cached_ub, obj_gm_tensor.output_gm,
                                                            obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                            out_put_cache_ub_offset, output_gm_offset)

            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
            input_ub_offset1 = out_put_cache_ub_offset + obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len * 8
            output_gm_offset1 = output_gm_offset + obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len * 8
            _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(tik_inst, obj_ub_tensor.cached_ub,
                                                                           obj_ub_tensor.output_ub,
                                                                           obj_gm_tensor.output_gm, input_ub_offset1,
                                                                           output_gm_offset1, vadd_mask)
    else:
        with tik_inst.for_range(0, Constant.CACHE_ACT_SIMPLING_NUM) as sample_idx:
            out_put_cache_ub_offset = (sample_idx + 1) * obj_fp32_output_init_input_scalar.last_axis_align_floor
            output_gm_offset = obj_common_scalar.cache_simpling_data_arr[sample_idx] * obj_fp32_e_num_input_scalar.e_num
            tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                   obj_ub_tensor.cached_ub[out_put_cache_ub_offset], 1,
                                   obj_fp32_e_num_input_scalar.move_pad, 0, 0)


def _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar,
                                                           obj_common_scalar, row_num, max_cache_num, ids_base_offset):
    """
    compute for optimization mode simpling or cache n number
    """
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        obj_common_scalar.cache_mode.set_as(Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM)
    else:
        with tik_inst.if_scope(
                tik.any(obj_common_scalar.ids_type_is_int32 == 0, max_cache_num <
                        (Constant.CACHE_ACT_SIMPLING_NUM + 1))):
            obj_common_scalar.cache_mode.set_as(Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM)
        with tik_inst.else_scope():
            tik_inst.h_cast(obj_ub_tensor.indices_index_float_ub, obj_ub_tensor.ids_ub, 'none')
            one_row_size = tik_inst.Scalar(dtype="int32", name="one_row_size")
            one_row_size.set_as(obj_fp32_e_num_input_scalar.e_num * obj_common_scalar.input_dsize)
            # calc simpling step
            idx_row_base_offset = tik_inst.Scalar(dtype="int32", name="idx_row_base_offset")
            idx_row_base_offset.set_as(ids_base_offset)
            obj_common_scalar.simpling_step.set_as(row_num // Constant.CACHE_ACT_SIMPLING_NUM)

            with tik_inst.if_scope(one_row_size > Constant.INPUT_LAST_AXIS_ONE_ROW_MAX_CACHE_SIZE):
                obj_common_scalar.cache_mode.set_as(Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM)

            with tik_inst.elif_scope(obj_common_scalar.simpling_step != 0):
                with tik_inst.for_range(0, Constant.CACHE_ACT_SIMPLING_NUM) as idx:
                    obj_common_scalar.cache_simpling_data_arr[idx].set_as(
                        obj_ub_tensor.ids_ub[idx_row_base_offset + idx * obj_common_scalar.simpling_step])

                    obj_ub_tensor.indices_simpling_ub[idx].set_as(
                        obj_ub_tensor.indices_index_float_ub[idx_row_base_offset +
                                                             idx * obj_common_scalar.simpling_step])

                max_dup_data = tik_inst.Tensor(Constant.DTYPE_FP32, (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                               name="max_dup_data",
                                               scope=tik.scope_ubuf)
                sel_gt_ub = tik_inst.Tensor(Constant.DTYPE_UINT32, (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                            name="sel_gt_ub",
                                            scope=tik.scope_ubuf)
                sel_gt_ub[0].set_as(0)
                max_cache_num_fp32 = tik_inst.Scalar(dtype=Constant.DTYPE_FP32,
                                                     name="max_cache_num_fp32",
                                                     init_value=max_cache_num)
                tik_inst.vec_dup(Constant.CACHE_ACT_SIMPLING_NUM, max_dup_data, max_cache_num_fp32, 1, 8)
                tik_inst.vec_cmpv_gt(sel_gt_ub, obj_ub_tensor.indices_simpling_ub, max_dup_data, 1, 8, 8)

                sel_gt_flag = tik_inst.Scalar(dtype=Constant.DTYPE_UINT32, name="sel_gt_flag", init_value=sel_gt_ub[0])
                with tik_inst.if_scope(sel_gt_flag == 0):
                    obj_common_scalar.cache_mode.set_as(Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM)
                with tik_inst.else_scope():
                    obj_common_scalar.cache_mode.set_as(Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_SIMPLING)

            with tik_inst.else_scope():
                obj_common_scalar.cache_mode.set_as(Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM)


def _tik_atomic_add_last_axis_align_small_e_hp(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                               obj_fp32_input_data_input_scalar, obj_fp32_e_num_input_scalar,
                                               obj_fp32_ids_input_scalar, obj_fp32_output_init_input_scalar,
                                               output_num):
    """
    _tik_atomic_add_last_axis_align_small_e_hp

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    e_num: e_num

    Returns
    -------
    None
    """
    repeat_time_255x64, left_part = _div(output_num, Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
    repeat_time_64, last_part_mask = _div(left_part, Constant.MASK_FP32)
    args_init = (tik_inst, obj_ub_tensor.output_0_ub, last_part_mask, repeat_time_64, repeat_time_255x64)
    _tik_init_ub_tensor_hp(args_init)

    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0, obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # front part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_front_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                            tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                            obj_fp32_input_data_input_scalar.front_rows_front_part_front_core,
                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                            input_mov_tims_front_part_front_core_index *
                            obj_fp32_input_data_input_scalar.front_rows_front_part_front_core)
                        with tik_inst.if_scope(
                                obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                            with tik_inst.for_range(0,
                                                    obj_fp32_input_data_input_scalar.\
                                                    front_rows_front_part_front_core) as \
                                    rows_index:
                                # visit ids
                                input_ub_offset = rows_index * obj_fp32_e_num_input_scalar.e_num
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                                         input_mov_tims_front_part_front_core_index * \
                                                         obj_fp32_input_data_input_scalar.\
                                                         front_rows_front_part_front_core + rows_index])
                                with tik_inst.if_scope(id_val_scalar <
                                                       obj_fp32_output_init_input_scalar. \
                                                               last_repeat_time_front_part_front_core):
                                    args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                input_ub_offset,
                                                obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                                obj_fp32_output_init_input_scalar.last_part_vadd_mask,
                                                obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                                    _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar,
                                                                 obj_fp32_e_num_input_scalar.e_num)
                                with tik_inst.else_scope():
                                    output_gm_offset = id_val_scalar * obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                        with tik_inst.else_scope():
                            back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                            obj_common_scalar.ori_ids_base_offset.set_as(
                                input_mov_tims_front_part_front_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core)
                            _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                tik_inst, obj_ub_tensor, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core,
                                obj_common_scalar.ori_ids_base_offset, back_offset)
                            _tik_atomic_add_last_axis_align_small_e_hp_calc_output_data(
                                tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                obj_common_scalar, obj_fp32_e_num_input_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core, back_offset)
                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # last part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.last_burst_len_front_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        # get the cached opt mode
                        _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                            tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                            obj_fp32_input_data_input_scalar.last_rows_front_part_front_core,
                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                            input_mov_tims_front_part_front_core_index *
                            obj_fp32_input_data_input_scalar.front_rows_front_part_front_core)
                        #with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.if_scope(
                                obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                            with tik_inst.for_range(0,
                                                    obj_fp32_input_data_input_scalar.\
                                                    last_rows_front_part_front_core) as \
                                    rows_index:
                                # visit ids
                                input_ub_offset = rows_index * obj_fp32_e_num_input_scalar.e_num
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                    input_mov_tims_front_part_front_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_front_core + rows_index])
                                with tik_inst.if_scope(id_val_scalar <
                                                       obj_fp32_output_init_input_scalar. \
                                                               last_repeat_time_front_part_front_core):
                                    args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                input_ub_offset,
                                                obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                                obj_fp32_output_init_input_scalar.last_part_vadd_mask,
                                                obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                                    _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar,
                                                                 obj_fp32_e_num_input_scalar.e_num)
                                with tik_inst.else_scope():
                                    output_gm_offset = id_val_scalar * obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                        with tik_inst.else_scope():
                            back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                            obj_common_scalar.ori_ids_base_offset.set_as(
                                input_mov_tims_front_part_front_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core)
                            _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                tik_inst, obj_ub_tensor, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_front_part_front_core,
                                obj_common_scalar.ori_ids_base_offset, back_offset)
                            _tik_atomic_add_last_axis_align_small_e_hp_calc_output_data(
                                tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                obj_common_scalar, obj_fp32_e_num_input_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_front_part_front_core, back_offset)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # front part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_last_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        # get the cached opt mode
                        _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                            tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                            obj_fp32_input_data_input_scalar.front_rows_last_part_front_core,
                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                            input_mov_tims_last_part_front_core_index *
                            obj_fp32_input_data_input_scalar.front_rows_last_part_front_core)
                        #with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.if_scope(
                                obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                            #with tik_inst.new_stmt_scope(disable_sync=True):
                            with tik_inst.for_range(0,
                                                    obj_fp32_input_data_input_scalar.\
                                                    front_rows_last_part_front_core) as \
                                    rows_index:
                                # visit ids
                                input_ub_offset = rows_index * \
                                                  obj_fp32_e_num_input_scalar.e_num
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                    input_mov_tims_last_part_front_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_front_core + rows_index])
                                with tik_inst.if_scope(id_val_scalar <
                                                       obj_fp32_output_init_input_scalar. \
                                                               last_repeat_time_front_part_front_core):
                                    args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                input_ub_offset,
                                                obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                                obj_fp32_output_init_input_scalar.last_part_vadd_mask,
                                                obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                                    _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar,
                                                                 obj_fp32_e_num_input_scalar.e_num)
                                with tik_inst.else_scope():
                                    output_gm_offset = id_val_scalar * obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                        with tik_inst.else_scope():
                            back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                            obj_common_scalar.ori_ids_base_offset.set_as(
                                input_mov_tims_last_part_front_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core)
                            _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                tik_inst, obj_ub_tensor, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core,
                                obj_common_scalar.ori_ids_base_offset, back_offset)
                            _tik_atomic_add_last_axis_align_small_e_hp_calc_output_data(
                                tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                obj_common_scalar, obj_fp32_e_num_input_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core, back_offset)

                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # last part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar. \
                                              e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar. \
                            last_burst_len_last_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        # get the cached opt mode
                        _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                            tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                            obj_fp32_input_data_input_scalar.last_rows_last_part_front_core,
                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                            input_mov_tims_last_part_front_core_index *
                            obj_fp32_input_data_input_scalar.front_rows_last_part_front_core)
                        #with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.if_scope(
                                obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                            #with tik_inst.new_stmt_scope(disable_sync=True):
                            with tik_inst.for_range(0,
                                                    obj_fp32_input_data_input_scalar.\
                                                    last_rows_last_part_front_core) as \
                                    rows_index:
                                # visit ids
                                input_ub_offset = rows_index * \
                                                  obj_fp32_e_num_input_scalar.e_num
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                    input_mov_tims_last_part_front_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_front_core + rows_index])
                                with tik_inst.if_scope(id_val_scalar < obj_fp32_output_init_input_scalar.
                                                       last_repeat_time_front_part_front_core):
                                    args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                input_ub_offset,
                                                obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                                obj_fp32_output_init_input_scalar.last_part_vadd_mask,
                                                obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                                    _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar,
                                                                 obj_fp32_e_num_input_scalar.e_num)
                                with tik_inst.else_scope():
                                    output_gm_offset = id_val_scalar * obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                        with tik_inst.else_scope():
                            back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                            obj_common_scalar.ori_ids_base_offset.set_as(
                                input_mov_tims_last_part_front_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core)
                            _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                tik_inst, obj_ub_tensor, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_last_part_front_core,
                                obj_common_scalar.ori_ids_base_offset, back_offset)
                            _tik_atomic_add_last_axis_align_small_e_hp_calc_output_data(
                                tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                obj_common_scalar, obj_fp32_e_num_input_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_last_part_front_core, back_offset)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # front part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar. \
                            front_burst_len_front_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        # get the cached opt mode
                        _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                            tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                            obj_fp32_input_data_input_scalar.front_rows_front_part_last_core,
                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                            input_mov_tims_front_part_last_core_index *
                            obj_fp32_input_data_input_scalar.front_rows_front_part_last_core)
                        #with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.if_scope(
                                obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                            #with tik_inst.new_stmt_scope(disable_sync=True):
                            with tik_inst.for_range(0,
                                                    obj_fp32_input_data_input_scalar.\
                                                    front_rows_front_part_last_core) as \
                                    rows_index:
                                # visit ids
                                input_ub_offset = \
                                    rows_index * \
                                    obj_fp32_e_num_input_scalar. \
                                        e_num
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                    input_mov_tims_front_part_last_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_last_core + rows_index])
                                with tik_inst.if_scope(id_val_scalar <
                                                       obj_fp32_output_init_input_scalar. \
                                                               last_repeat_time_front_part_front_core):
                                    args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                input_ub_offset,
                                                obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                                obj_fp32_output_init_input_scalar.last_part_vadd_mask,
                                                obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                                    _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar,
                                                                 obj_fp32_e_num_input_scalar.e_num)
                                with tik_inst.else_scope():
                                    output_gm_offset = id_val_scalar * obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                        with tik_inst.else_scope():
                            back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                            obj_common_scalar.ori_ids_base_offset.set_as(
                                input_mov_tims_front_part_last_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core)
                            _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                tik_inst, obj_ub_tensor, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core,
                                obj_common_scalar.ori_ids_base_offset, back_offset)
                            _tik_atomic_add_last_axis_align_small_e_hp_calc_output_data(
                                tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                obj_common_scalar, obj_fp32_e_num_input_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core, back_offset)

                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # last part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar. \
                            last_burst_len_front_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        # get the cached opt mode
                        _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                            tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                            obj_fp32_input_data_input_scalar.last_rows_front_part_last_core,
                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                            input_mov_tims_front_part_last_core_index *
                            obj_fp32_input_data_input_scalar.front_rows_front_part_last_core)
                        #with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.if_scope(
                                obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                            # with tik_inst.new_stmt_scope(disable_sync=True):
                            with tik_inst.for_range(0,
                                                    obj_fp32_input_data_input_scalar.\
                                                    last_rows_front_part_last_core) as \
                                    rows_index:
                                # visit ids
                                input_ub_offset = rows_index * \
                                                  obj_fp32_e_num_input_scalar.e_num
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                    input_mov_tims_front_part_last_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_last_core + rows_index])
                                with tik_inst.if_scope(id_val_scalar <
                                                       obj_fp32_output_init_input_scalar. \
                                                               last_repeat_time_front_part_front_core):
                                    args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                input_ub_offset,
                                                obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                                obj_fp32_output_init_input_scalar.last_part_vadd_mask,
                                                obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                                    _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar,
                                                                 obj_fp32_e_num_input_scalar.e_num)
                                with tik_inst.else_scope():
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                        with tik_inst.else_scope():
                            back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                            obj_common_scalar.ori_ids_base_offset.set_as(
                                input_mov_tims_front_part_last_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core)
                            _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                tik_inst, obj_ub_tensor, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_front_part_last_core,
                                obj_common_scalar.ori_ids_base_offset, back_offset)
                            _tik_atomic_add_last_axis_align_small_e_hp_calc_output_data(
                                tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                obj_common_scalar, obj_fp32_e_num_input_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_front_part_last_core, back_offset)

            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar. \
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        # front part ids last part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_last_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        # get the cached opt mode
                        _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                            tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                            obj_fp32_input_data_input_scalar.front_rows_last_part_last_core,
                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                            input_mov_tims_last_part_last_core_index *
                            obj_fp32_input_data_input_scalar.front_rows_last_part_last_core)
                        #with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.if_scope(
                                obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                            #with tik_inst.new_stmt_scope(disable_sync=True):
                            with tik_inst.for_range(0,
                                                    obj_fp32_input_data_input_scalar.\
                                                    front_rows_last_part_last_core) as \
                                    rows_index:
                                # visit ids
                                input_ub_offset = rows_index * \
                                                  obj_fp32_e_num_input_scalar.e_num
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index *
                                                         obj_fp32_input_data_input_scalar.front_rows_last_part_last_core
                                                         + rows_index])
                                with tik_inst.if_scope(id_val_scalar <
                                                       obj_fp32_output_init_input_scalar. \
                                                               last_repeat_time_front_part_front_core):
                                    args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                input_ub_offset,
                                                obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                                obj_fp32_output_init_input_scalar.last_part_vadd_mask,
                                                obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                                    _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar,
                                                                 obj_fp32_e_num_input_scalar.e_num)
                                with tik_inst.else_scope():
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar. \
                                                           e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                        with tik_inst.else_scope():
                            back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                            obj_common_scalar.ori_ids_base_offset.set_as(
                                input_mov_tims_last_part_last_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_last_part_last_core)
                            _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                tik_inst, obj_ub_tensor, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_last_part_last_core,
                                obj_common_scalar.ori_ids_base_offset, back_offset)
                            _tik_atomic_add_last_axis_align_small_e_hp_calc_output_data(
                                tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                obj_common_scalar, obj_fp32_e_num_input_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_last_part_last_core, back_offset)

                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        # last part ids last part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.last_burst_len_last_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        # get the cached opt mode
                        _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                            tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                            obj_fp32_input_data_input_scalar.last_rows_last_part_last_core,
                            obj_fp32_output_init_input_scalar.last_repeat_time_front_part_front_core,
                            input_mov_tims_last_part_last_core_index *
                            obj_fp32_input_data_input_scalar.front_rows_last_part_last_core)
                        #with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.if_scope(
                                obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                            #with tik_inst.new_stmt_scope(disable_sync=True):
                            with tik_inst.for_range(0,
                                                    obj_fp32_input_data_input_scalar.last_rows_last_part_last_core) as \
                                    rows_index:
                                # visit ids
                                input_ub_offset = rows_index * \
                                                  obj_fp32_e_num_input_scalar.e_num
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index *
                                                         obj_fp32_input_data_input_scalar.front_rows_last_part_last_core
                                                         + rows_index])
                                with tik_inst.if_scope(id_val_scalar <
                                                       obj_fp32_output_init_input_scalar. \
                                                               last_repeat_time_front_part_front_core):
                                    args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                input_ub_offset,
                                                obj_fp32_output_init_input_scalar.last_repeat_time_front_part_last_core,
                                                obj_fp32_output_init_input_scalar.last_part_vadd_mask,
                                                obj_fp32_output_init_input_scalar.init_times_front_part_front_core)
                                    _tik_vadd_ub_tensor_hp_align(args_add, id_val_scalar,
                                                                 obj_fp32_e_num_input_scalar.e_num)
                                with tik_inst.else_scope():
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                        with tik_inst.else_scope():
                            back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                            obj_common_scalar.ori_ids_base_offset.set_as(
                                input_mov_tims_last_part_last_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_last_part_last_core)
                            _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                tik_inst, obj_ub_tensor, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_last_part_last_core,
                                obj_common_scalar.ori_ids_base_offset, back_offset)
                            _tik_atomic_add_last_axis_align_small_e_hp_calc_output_data(
                                tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                obj_common_scalar, obj_fp32_e_num_input_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_last_part_last_core, back_offset)
    _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.output_0_ub, obj_gm_tensor.output_gm,
                                                obj_fp32_output_init_input_scalar.init_times_front_part_last_core, 0, 0)


def _tik_remove_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align, output_num):
    with tik_inst.new_stmt_scope():
        repeat = obj_fp32_e_num_input_scalar.repeat_remove_pad
        src_input1_ub = obj_ub_tensor.output_0_ub.reinterpret_cast_to("float16")
        obj_ub_tensor.output_1_ub = tik_inst.Tensor("float32", (output_num,), name="output_1_ub", scope=tik.scope_ubuf)
        obj_ub_tensor.duiqi_ub = tik_inst.Tensor("float32", (output_num,), name="duiqi_ub", scope=tik.scope_ubuf)
        src1_duiqi = obj_ub_tensor.duiqi_ub.reinterpret_cast_to("float16")
        dst_output = obj_ub_tensor.output_1_ub.reinterpret_cast_to("float16")
        e_size = obj_fp32_e_num_input_scalar.e_num
        col_sub_block = obj_fp32_e_num_input_scalar.col_block_remove_pad
        sub = obj_fp32_e_num_input_scalar.e_num_sub
        with tik_inst.if_scope(tik.all(col_sub_block > 0, repeat != 0)):
            dst_list = [dst_output[16 * i] for i in range(16)]
            src_list = [src_input1_ub[16 * e_num_align * repeat * i] for i in range(16)]
            tik_inst.vnchwconv(False, False, dst_list, src_list, e_num_align * repeat, 16, 1)
            with tik_inst.for_range(0, repeat) as index:
                tik_inst.data_move(obj_ub_tensor.duiqi_ub[index * 128 * e_size],
                                   obj_ub_tensor.output_1_ub[index * 128 * e_num_align], 0, 8, e_size * 2, sub * 2, 0)

            dst_list2 = [src_input1_ub[16 * e_size * repeat * i] for i in range(16)]
            src_list2 = [src1_duiqi[16 * i] for i in range(16)]
            with tik_inst.if_scope(e_size * repeat == 1):
                tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_size * repeat, 0, 0)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_size * repeat, 1, 16)
            # last part
            tik_inst.data_move(obj_ub_tensor.duiqi_ub[0], obj_ub_tensor.output_0_ub[128 * repeat * e_num_align], 0, 1,
                               col_sub_block, 0, 0)
            dst_list = [dst_output[16 * i] for i in range(16)]
            src_list = [src1_duiqi[16 * e_num_align * i] for i in range(16)]
            tik_inst.vnchwconv(False, False, dst_list, src_list, e_num_align, 16, 1)
            tik_inst.data_move(obj_ub_tensor.duiqi_ub[0], obj_ub_tensor.output_1_ub[0], 0, 8, e_size * 2, sub * 2, 0)

            dst_list2 = [dst_output[16 * e_size * i] for i in range(16)]
            src_list2 = [src1_duiqi[16 * i] for i in range(16)]
            with tik_inst.if_scope(e_size == 1):
                tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_size, 0, 0)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_size, 1, 16)
            tik_inst.data_move(obj_ub_tensor.output_0_ub[repeat * 128 * e_size], obj_ub_tensor.output_1_ub[0], 0, 1,
                               col_sub_block, 0, 0)
        with tik_inst.if_scope(tik.all(col_sub_block == 0, repeat != 0)):
            dst_list = [dst_output[16 * i] for i in range(16)]
            src_list = [src_input1_ub[16 * e_num_align * repeat * i] for i in range(16)]
            tik_inst.vnchwconv(False, False, dst_list, src_list, e_num_align * repeat, 16, 1)
            with tik_inst.for_range(0, repeat) as index:
                tik_inst.data_move(obj_ub_tensor.duiqi_ub[index * 128 * e_size],
                                   obj_ub_tensor.output_1_ub[index * 128 * e_num_align], 0, 8, e_size * 2, sub * 2, 0)
            dst_list2 = [src_input1_ub[16 * e_size * repeat * i] for i in range(16)]
            src_list2 = [src1_duiqi[16 * i] for i in range(16)]
            with tik_inst.if_scope(e_size * repeat == 1):
                tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_size * repeat, 0, 0)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_size * repeat, 1, 16)

        with tik_inst.if_scope(repeat == 0):
            dst_list = [dst_output[16 * i] for i in range(16)]
            src_list = [src_input1_ub[16 * e_num_align * i] for i in range(16)]
            tik_inst.vnchwconv(False, False, dst_list, src_list, e_num_align, 16, 1)
            tik_inst.data_move(obj_ub_tensor.duiqi_ub[0], obj_ub_tensor.output_1_ub[0], 0, 8, e_size * 2, sub * 2, 0)

            dst_list2 = [src_input1_ub[16 * e_size * i] for i in range(16)]
            src_list2 = [src1_duiqi[16 * i] for i in range(16)]
            with tik_inst.if_scope(e_size == 1):
                tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_size, 0, 0)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_size, 1, 16)


def _tik_align_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align, output_num, front):
    with tik_inst.new_stmt_scope():
        src_input1_ub = obj_ub_tensor.cached_ub.reinterpret_cast_to("float16")
        dst1_input_ub = obj_ub_tensor.input_ub.reinterpret_cast_to("float16")
        obj_ub_tensor.output_1_ub = tik_inst.Tensor("float32", (output_num,), name="output_1_ub", scope=tik.scope_ubuf)
        obj_ub_tensor.duiqi_ub = tik_inst.Tensor("float32", (output_num,), name="duiqi_ub", scope=tik.scope_ubuf)
        src1_duiqi = obj_ub_tensor.duiqi_ub.reinterpret_cast_to("float16")
        dst_output = obj_ub_tensor.output_1_ub.reinterpret_cast_to("float16")
        e_size = obj_fp32_e_num_input_scalar.e_num
        if front == 0:
            repeat = obj_fp32_e_num_input_scalar.repeat_front_front_part_front_core
            col_sub_block = obj_fp32_e_num_input_scalar.col_sub_block_front_front_part_front_core
        elif front == 1:
            repeat = obj_fp32_e_num_input_scalar.repeat_last_front_part_front_core
            col_sub_block = obj_fp32_e_num_input_scalar.col_sub_block_last_front_part_front_core
        elif front == 2:
            repeat = obj_fp32_e_num_input_scalar.repeat_front_last_part_front_core
            col_sub_block = obj_fp32_e_num_input_scalar.col_sub_block_front_last_part_front_core
        elif front == 3:
            repeat = obj_fp32_e_num_input_scalar.repeat_last_last_part_front_core
            col_sub_block = obj_fp32_e_num_input_scalar.col_sub_block_last_last_part_front_core
        elif front == 4:
            repeat = obj_fp32_e_num_input_scalar.repeat_front_front_part_last_core
            col_sub_block = obj_fp32_e_num_input_scalar.col_sub_block_front_front_part_last_core
        elif front == 5:
            repeat = obj_fp32_e_num_input_scalar.repeat_last_front_part_last_core
            col_sub_block = obj_fp32_e_num_input_scalar.col_sub_block_last_front_part_last_core
        elif front == 6:
            repeat = obj_fp32_e_num_input_scalar.repeat_front_last_part_last_core
            col_sub_block = obj_fp32_e_num_input_scalar.col_sub_block_front_last_part_last_core
        elif front == 7:
            repeat = obj_fp32_e_num_input_scalar.repeat_last_last_part_last_core
            col_sub_block = obj_fp32_e_num_input_scalar.col_sub_block_last_last_part_last_core
        sub = obj_fp32_e_num_input_scalar.e_num_sub

        with tik_inst.if_scope(tik.all(col_sub_block > 0, repeat != 0)):
            dst_list = [dst_output[16 * i] for i in range(16)]
            src_list = [src_input1_ub[16 * e_size * repeat * i] for i in range(16)]
            with tik_inst.if_scope(e_size * repeat == 1):
                tik_inst.vnchwconv(False, False, dst_list, src_list, e_size * repeat, 0, 0)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list, src_list, e_size * repeat, 16, 1)
            with tik_inst.for_range(0, repeat) as index:
                tik_inst.data_move(obj_ub_tensor.duiqi_ub[index * 128 * e_num_align],
                                   obj_ub_tensor.output_1_ub[index * 128 * e_size], 0, 8, e_size * 2, 0,
                                   sub * 2)

            dst_list2 = [dst1_input_ub[16 * e_num_align * repeat * i] for i in range(16)]
            src_list2 = [src1_duiqi[16 * i] for i in range(16)]
            tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_num_align * repeat, 1, 16)

            #weikuai
            tik_inst.data_move(obj_ub_tensor.duiqi_ub[0], obj_ub_tensor.cached_ub[128 * repeat * e_size], 0, 1,
                               col_sub_block, 0, 0)
            dst_list = [dst_output[16 * i] for i in range(16)]
            src_list = [src1_duiqi[16 * e_size * i] for i in range(16)]
            with tik_inst.if_scope(e_size == 1):
                tik_inst.vnchwconv(False, False, dst_list, src_list, e_size, 0, 0)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list, src_list, e_size, 16, 1)
            tik_inst.data_move(obj_ub_tensor.duiqi_ub[0], obj_ub_tensor.output_1_ub[0], 0, 8, e_size * 2, 0, sub * 2)

            dst_list2 = [dst_output[16 * e_num_align * i] for i in range(16)]
            src_list2 = [src1_duiqi[16 * i] for i in range(16)]
            tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_num_align, 1, 16)
            tik_inst.data_move(obj_ub_tensor.input_ub[repeat * 128 * e_num_align], obj_ub_tensor.output_1_ub[0], 0, 1,
                               col_sub_block, 0, 0)
        with tik_inst.if_scope(tik.all(col_sub_block == 0, repeat != 0)):
            dst_list = [dst_output[16 * i] for i in range(16)]
            src_list = [src_input1_ub[16 * e_size * repeat * i] for i in range(16)]
            with tik_inst.if_scope(e_size * repeat == 1):
                tik_inst.vnchwconv(False, False, dst_list, src_list, e_size * repeat, 0, 0)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list, src_list, e_size * repeat, 16, 1)
            with tik_inst.for_range(0, repeat) as index:
                tik_inst.data_move(obj_ub_tensor.duiqi_ub[index * 128 * e_num_align],
                                   obj_ub_tensor.output_1_ub[index * 128 * e_size], 0, 8, e_size * 2, 0,
                                   sub * 2)

            dst_list2 = [dst1_input_ub[16 * e_num_align * repeat * i] for i in range(16)]
            src_list2 = [src1_duiqi[16 * i] for i in range(16)]
            tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_num_align * repeat, 1, 16)
        with tik_inst.if_scope(repeat == 0):
            dst_list = [dst_output[16 * i] for i in range(16)]
            src_list = [src_input1_ub[16 * e_size * i] for i in range(16)]
            with tik_inst.if_scope(e_size == 1):
                tik_inst.vnchwconv(False, False, dst_list, src_list, e_size, 0, 0)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list, src_list, e_size, 16, 1)
            tik_inst.data_move(obj_ub_tensor.duiqi_ub[0], obj_ub_tensor.output_1_ub[0], 0, 8, e_size * 2, 0, sub * 2)

            dst_list2 = [dst1_input_ub[16 * e_num_align * i] for i in range(16)]
            src_list2 = [src1_duiqi[16 * i] for i in range(16)]
            tik_inst.vnchwconv(False, False, dst_list2, src_list2, e_num_align, 1, 16)


def _tik_atomic_add_last_axis_not_align_small_e_hp_pad(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor,
                                                       obj_common_scalar, obj_fp32_input_data_input_scalar,
                                                       obj_fp32_e_num_input_scalar, obj_fp32_ids_input_scalar,
                                                       obj_fp32_output_init_input_scalar, output_num):
    """
    _tik_atomic_add_last_axis_not_align_small_e_hp

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_fp32_output_init_input_scalar: obj_fp32_output_init_input_scalar
    output_num: output_num

    Returns
    -------
    None
    """
    repeat_time_255x64, left_part = _div(output_num, Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
    repeat_time_64, last_part_mask = _div(left_part, Constant.MASK_FP32)
    args_init = (tik_inst, obj_ub_tensor.output_0_ub, last_part_mask, repeat_time_64, repeat_time_255x64)
    _tik_init_ub_tensor_hp(args_init)
    move_pad = tbe_platform.api_check_support("tik.data_move_pad")

    repeat_time_255x64, left_part = _div(obj_fp32_e_num_input_scalar.e_num,
                                         Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
    repeat_time_64, last_part_mask = _div(left_part, Constant.MASK_FP32)

    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # front part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_front_part_front_core
                        
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.cached_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst,
                                                      input_burst_len)
                        e_num_align = obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_align_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align,
                                       output_num, 0)
                        with tik_inst.new_stmt_scope():
                            obj_ub_tensor.indices_index_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (output_num,),
                                                                                   name="indices_index_float_ub",
                                                                                   scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_int_ub = tik_inst.Tensor(Constant.DTYPE_INT32,
                                                                                (output_num // 2,),
                                                                                name="indices_temp_int_ub",
                                                                                scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                  (output_num // 2,),
                                                                                  name="indices_temp_float_ub",
                                                                                  scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_simpling_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                                                                name="indices_simpling_ub",
                                                                                scope=tik.scope_ubuf)
                            repeat_time_255, left_part_n = _div(output_num // 2,
                                    Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
                            repeat_time_64_n, last_part_mask_n = _div(left_part_n, Constant.MASK_FP32)
                            args_init = (tik_inst, obj_ub_tensor.indices_temp_float_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init)
                            args_init_ub = (tik_inst, obj_ub_tensor.indices_temp_int_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init_ub)
                            _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                                tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core,
                                obj_fp32_output_init_input_scalar.max_cache_n_num,
                                input_mov_tims_front_part_front_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core)
                            with tik_inst.if_scope(
                                    obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                                with tik_inst.for_range(
                                        0, obj_fp32_input_data_input_scalar.front_rows_front_part_front_core
                                ) as rows_index:
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                        input_mov_tims_front_part_front_core_index *
                                        obj_fp32_input_data_input_scalar.front_rows_front_part_front_core + rows_index])
                                    with tik_inst.if_scope(
                                            id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_0_ub_offset = id_val_scalar * \
                                                             obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                    output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                                    repeat_time_255x64)
                                        _tik_vadd_ub_tensor_hp(args_add)
                                    with tik_inst.else_scope():
                                        # align part
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_gm_offset = id_val_scalar * \
                                                           obj_fp32_e_num_input_scalar.e_num
                                        if not move_pad:
                                            with tik_inst.if_scope(
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):

                                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                    input_ub_offset, output_gm_offset)
                                            # last part
                                            input_ub_offset_new = rows_index * \
                                                                  obj_fp32_output_init_input_scalar.\
                                                                  last_axis_align_floor + \
                                                                  obj_fp32_output_init_input_scalar.\
                                                                  last_axis_align_front_part
                                            output_gm_offset_new = id_val_scalar * \
                                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                                   obj_fp32_output_init_input_scalar.\
                                                                   last_axis_align_front_part
                                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                                tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                                obj_gm_tensor.output_gm, input_ub_offset_new, output_gm_offset_new,
                                                vadd_mask)
                                        else:
                                            tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                                                   obj_ub_tensor.input_ub[input_ub_offset], 1,
                                                                   obj_fp32_e_num_input_scalar.move_pad, 0, 0)
                            with tik_inst.else_scope():
                                back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                                obj_common_scalar.ori_ids_base_offset.set_as(
                                    input_mov_tims_front_part_front_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_front_core)
                                _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                    tik_inst, obj_ub_tensor, obj_common_scalar,
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_front_core,
                                    obj_common_scalar.ori_ids_base_offset, back_offset)
                                _tik_atomic_add_last_axis_not_align_small_e_hp_output_data(
                                    tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                    obj_common_scalar, obj_fp32_e_num_input_scalar,
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_front_core, back_offset)

                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # last part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.last_burst_len_front_part_front_core
                      
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.cached_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst,
                                                      input_burst_len)
                        e_num_align = obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_align_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align,
                                       output_num, 1)
                        with tik_inst.new_stmt_scope():
                            obj_ub_tensor.indices_index_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (output_num,),
                                                                                   name="indices_index_float_ub",
                                                                                   scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_int_ub = tik_inst.Tensor(Constant.DTYPE_INT32,
                                                                                (output_num // 2,),
                                                                                name="indices_temp_int_ub",
                                                                                scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                  (output_num // 2,),
                                                                                  name="indices_temp_float_ub",
                                                                                  scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_simpling_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                                                                name="indices_simpling_ub",
                                                                                scope=tik.scope_ubuf)
                            repeat_time_255, left_part_n = _div(output_num // 2,
                                    Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
                            repeat_time_64_n, last_part_mask_n = _div(left_part_n, Constant.MASK_FP32)
                            args_init = (tik_inst, obj_ub_tensor.indices_temp_float_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init)
                            args_init_ub = (tik_inst, obj_ub_tensor.indices_temp_int_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init_ub)
                            # get the cached opt mode
                            _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                                tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_front_part_front_core,
                                obj_fp32_output_init_input_scalar.max_cache_n_num,
                                input_mov_tims_front_part_front_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core)
                            with tik_inst.if_scope(
                                    obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                                with tik_inst.for_range(0,
                                                        obj_fp32_input_data_input_scalar.\
                                                        last_rows_front_part_front_core) as \
                                        rows_index:
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                        input_mov_tims_front_part_front_core_index *
                                        obj_fp32_input_data_input_scalar.front_rows_front_part_front_core + rows_index])
                                    with tik_inst.if_scope(
                                            id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_0_ub_offset = id_val_scalar * \
                                                             obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                    output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                                    repeat_time_255x64)
                                        _tik_vadd_ub_tensor_hp(args_add)
                                    with tik_inst.else_scope():
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_gm_offset = id_val_scalar * \
                                                           obj_fp32_e_num_input_scalar.e_num
                                        # align part
                                        if not move_pad:
                                            with tik_inst.if_scope(
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):

                                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                    input_ub_offset, output_gm_offset)
                                            # last part
                                            input_ub_offset_new = rows_index * \
                                                                  obj_fp32_output_init_input_scalar. \
                                                                      last_axis_align_floor + \
                                                                  obj_fp32_output_init_input_scalar. \
                                                                      last_axis_align_front_part
                                            output_gm_offset_new = id_val_scalar * \
                                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                                   obj_fp32_output_init_input_scalar.\
                                                                   last_axis_align_front_part
                                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                                tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                                obj_gm_tensor.output_gm, input_ub_offset_new, output_gm_offset_new,
                                                vadd_mask)
                                        else:
                                            tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                                                   obj_ub_tensor.input_ub[input_ub_offset], 1,
                                                                   obj_fp32_e_num_input_scalar.move_pad, 0, 0)
                            with tik_inst.else_scope():
                                back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                                obj_common_scalar.ori_ids_base_offset.set_as(
                                    input_mov_tims_front_part_front_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_front_core)
                                _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                    tik_inst, obj_ub_tensor, obj_common_scalar,
                                    obj_fp32_input_data_input_scalar.last_rows_front_part_front_core,
                                    obj_common_scalar.ori_ids_base_offset, back_offset)
                                _tik_atomic_add_last_axis_not_align_small_e_hp_output_data(
                                    tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                    obj_common_scalar, obj_fp32_e_num_input_scalar,
                                    obj_fp32_input_data_input_scalar.last_rows_front_part_front_core, back_offset)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # front part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_last_part_front_core
                        
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.cached_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst,
                                                      input_burst_len)
                        e_num_align = obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_align_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align,
                                       output_num, 2)
                        with tik_inst.new_stmt_scope():
                            obj_ub_tensor.indices_index_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (output_num,),
                                                                                   name="indices_index_float_ub",
                                                                                   scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_int_ub = tik_inst.Tensor(Constant.DTYPE_INT32,
                                                                                (output_num // 2,),
                                                                                name="indices_temp_int_ub",
                                                                                scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                  (output_num // 2,),
                                                                                  name="indices_temp_float_ub",
                                                                                  scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_simpling_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                                                                name="indices_simpling_ub",
                                                                                scope=tik.scope_ubuf)
                            repeat_time_255, left_part_n = _div(output_num // 2,
                                    Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
                            repeat_time_64_n, last_part_mask_n = _div(left_part_n, Constant.MASK_FP32)
                            args_init = (tik_inst, obj_ub_tensor.indices_temp_float_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init)
                            args_init_ub = (tik_inst, obj_ub_tensor.indices_temp_int_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init_ub)
                            # get the cached opt mode
                            _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                                tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core,
                                obj_fp32_output_init_input_scalar.max_cache_n_num,
                                input_mov_tims_last_part_front_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core)
                            with tik_inst.if_scope(
                                    obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                                with tik_inst.for_range(0,
                                                        obj_fp32_input_data_input_scalar.\
                                                        front_rows_last_part_front_core) as \
                                        rows_index:
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                        input_mov_tims_last_part_front_core_index *
                                        obj_fp32_input_data_input_scalar.front_rows_last_part_front_core + rows_index])
                                    with tik_inst.if_scope(
                                            id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_0_ub_offset = id_val_scalar * \
                                                             obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                    output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                                    repeat_time_255x64)
                                        _tik_vadd_ub_tensor_hp(args_add)
                                    with tik_inst.else_scope():
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar. \
                                                              last_axis_align_floor
                                        output_gm_offset = id_val_scalar * \
                                                           obj_fp32_e_num_input_scalar.e_num
                                        if not move_pad:
                                            # align part
                                            with tik_inst.if_scope(
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):

                                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                    input_ub_offset, output_gm_offset)
                                            # last part
                                            input_ub_offset_new = rows_index * \
                                                                  obj_fp32_output_init_input_scalar. \
                                                                      last_axis_align_floor + \
                                                                  obj_fp32_output_init_input_scalar. \
                                                                      last_axis_align_front_part
                                            output_gm_offset_new = id_val_scalar * \
                                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                                   obj_fp32_output_init_input_scalar. \
                                                                       last_axis_align_front_part
                                            vadd_mask = obj_fp32_output_init_input_scalar. \
                                                last_part_vadd_mask
                                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                                tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                                obj_gm_tensor.output_gm, input_ub_offset_new, output_gm_offset_new,
                                                vadd_mask)
                                        else:
                                            tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                                                   obj_ub_tensor.input_ub[input_ub_offset], 1,
                                                                   obj_fp32_e_num_input_scalar.move_pad, 0, 0)
                            with tik_inst.else_scope():
                                back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                                obj_common_scalar.ori_ids_base_offset.set_as(
                                    input_mov_tims_last_part_front_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_front_core)
                                _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                    tik_inst, obj_ub_tensor, obj_common_scalar,
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_front_core,
                                    obj_common_scalar.ori_ids_base_offset, back_offset)
                                _tik_atomic_add_last_axis_not_align_small_e_hp_output_data(
                                    tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                    obj_common_scalar, obj_fp32_e_num_input_scalar,
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_front_core, back_offset)
                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # last part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.last_burst_len_last_part_front_core
             
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.cached_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst,
                                                      input_burst_len)
                        e_num_align = obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_align_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align,
                                       output_num, 3)
                        with tik_inst.new_stmt_scope():
                            obj_ub_tensor.indices_index_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (output_num,),
                                                                                   name="indices_index_float_ub",
                                                                                   scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_int_ub = tik_inst.Tensor(Constant.DTYPE_INT32,
                                                                                (output_num // 2,),
                                                                                name="indices_temp_int_ub",
                                                                                scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                  (output_num // 2,),
                                                                                  name="indices_temp_float_ub",
                                                                                  scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_simpling_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                                                                name="indices_simpling_ub",
                                                                                scope=tik.scope_ubuf)
                            repeat_time_255, left_part_n = _div(output_num // 2,
                                    Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
                            repeat_time_64_n, last_part_mask_n = _div(left_part_n, Constant.MASK_FP32)
                            args_init = (tik_inst, obj_ub_tensor.indices_temp_float_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init)
                            args_init_ub = (tik_inst, obj_ub_tensor.indices_temp_int_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init_ub)
                            # get the cached opt mode
                            _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                                tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_last_part_front_core,
                                obj_fp32_output_init_input_scalar.max_cache_n_num,
                                input_mov_tims_last_part_front_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core)
                            with tik_inst.if_scope(
                                    obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                                with tik_inst.for_range(0,
                                                        obj_fp32_input_data_input_scalar.\
                                                        last_rows_last_part_front_core) as \
                                        rows_index:
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                        input_mov_tims_last_part_front_core_index *
                                        obj_fp32_input_data_input_scalar.front_rows_last_part_front_core + rows_index])
                                    with tik_inst.if_scope(
                                            id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_0_ub_offset = id_val_scalar * \
                                                             obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                    output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                                    repeat_time_255x64)
                                        _tik_vadd_ub_tensor_hp(args_add)
                                    with tik_inst.else_scope():
                                        # align part
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar. \
                                                              last_axis_align_floor
                                        output_gm_offset = id_val_scalar * \
                                                           obj_fp32_e_num_input_scalar.e_num
                                        if not move_pad:
                                            with tik_inst.if_scope(
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):

                                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                    input_ub_offset, output_gm_offset)
                                            # last part
                                            input_ub_offset_new = rows_index * \
                                                                  obj_fp32_output_init_input_scalar.\
                                                                  last_axis_align_floor + \
                                                                  obj_fp32_output_init_input_scalar.\
                                                                  last_axis_align_front_part
                                            output_gm_offset_new = id_val_scalar * \
                                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                                   obj_fp32_output_init_input_scalar.\
                                                                   last_axis_align_front_part
                                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                                tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                                obj_gm_tensor.output_gm, input_ub_offset_new, output_gm_offset_new,
                                                vadd_mask)
                                        else:
                                            tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                                                   obj_ub_tensor.input_ub[input_ub_offset], 1,
                                                                   obj_fp32_e_num_input_scalar.move_pad, 0, 0)
                            with tik_inst.else_scope():
                                back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                                obj_common_scalar.ori_ids_base_offset.set_as(
                                    input_mov_tims_last_part_front_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_front_core)
                                _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                    tik_inst, obj_ub_tensor, obj_common_scalar,
                                    obj_fp32_input_data_input_scalar.last_rows_last_part_front_core,
                                    obj_common_scalar.ori_ids_base_offset, back_offset)
                                _tik_atomic_add_last_axis_not_align_small_e_hp_output_data(
                                    tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                    obj_common_scalar, obj_fp32_e_num_input_scalar,
                                    obj_fp32_input_data_input_scalar.last_rows_last_part_front_core, back_offset)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # front part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_front_part_last_core

                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.cached_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst,
                                                      input_burst_len)
                        e_num_align = obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_align_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align,
                                       output_num, 4)
                        with tik_inst.new_stmt_scope():
                            obj_ub_tensor.indices_index_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (output_num,),
                                                                                   name="indices_index_float_ub",
                                                                                   scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_int_ub = tik_inst.Tensor(Constant.DTYPE_INT32,
                                                                                (output_num // 2,),
                                                                                name="indices_temp_int_ub",
                                                                                scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                  (output_num // 2,),
                                                                                  name="indices_temp_float_ub",
                                                                                  scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_simpling_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                                                                name="indices_simpling_ub",
                                                                                scope=tik.scope_ubuf)
                            repeat_time_255, left_part_n = _div(output_num // 2, 
                                    Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
                            repeat_time_64_n, last_part_mask_n = _div(left_part_n, Constant.MASK_FP32)
                            args_init = (tik_inst, obj_ub_tensor.indices_temp_float_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init)
                            args_init_ub = (tik_inst, obj_ub_tensor.indices_temp_int_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init_ub)
                            # get the cached opt mode
                            _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                                tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core,
                                obj_fp32_output_init_input_scalar.max_cache_n_num,
                                input_mov_tims_front_part_last_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core)
                            with tik_inst.if_scope(
                                    obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                                with tik_inst.for_range(0,
                                                        obj_fp32_input_data_input_scalar.\
                                                        front_rows_front_part_last_core) as \
                                        rows_index:
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                        input_mov_tims_front_part_last_core_index *
                                        obj_fp32_input_data_input_scalar.front_rows_front_part_last_core + rows_index])
                                    with tik_inst.if_scope(
                                            id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_0_ub_offset = id_val_scalar * \
                                                             obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                    output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                                    repeat_time_255x64)
                                        _tik_vadd_ub_tensor_hp(args_add)
                                    with tik_inst.else_scope():
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_gm_offset = id_val_scalar * \
                                                           obj_fp32_e_num_input_scalar.e_num
                                        if not move_pad:
                                            # align part
                                            with tik_inst.if_scope(
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):

                                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                    input_ub_offset, output_gm_offset)
                                            # last part
                                            input_ub_offset_new = rows_index * \
                                                                  obj_fp32_output_init_input_scalar.\
                                                                  last_axis_align_floor + \
                                                                  obj_fp32_output_init_input_scalar.\
                                                                  last_axis_align_front_part
                                            output_gm_offset_new = id_val_scalar * \
                                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                                   obj_fp32_output_init_input_scalar.\
                                                                   last_axis_align_front_part
                                            vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                                tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                                obj_gm_tensor.output_gm, input_ub_offset_new, output_gm_offset_new,
                                                vadd_mask)
                                        else:
                                            tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                                                   obj_ub_tensor.input_ub[input_ub_offset], 1,
                                                                   obj_fp32_e_num_input_scalar.move_pad, 0, 0)
                            with tik_inst.else_scope():
                                back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                                obj_common_scalar.ori_ids_base_offset.set_as(
                                    input_mov_tims_front_part_last_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_last_core)
                                _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                    tik_inst, obj_ub_tensor, obj_common_scalar,
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_last_core,
                                    obj_common_scalar.ori_ids_base_offset, back_offset)
                                _tik_atomic_add_last_axis_not_align_small_e_hp_output_data(
                                    tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                    obj_common_scalar, obj_fp32_e_num_input_scalar,
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_last_core, back_offset)
                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # last part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        
                        input_burst_len = obj_fp32_input_data_input_scalar.last_burst_len_front_part_last_core

                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.cached_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst,
                                                      input_burst_len)
                        e_num_align = obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_align_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align,
                                       output_num, 5)
                        with tik_inst.new_stmt_scope():
                            obj_ub_tensor.indices_index_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (output_num,),
                                                                                   name="indices_index_float_ub",
                                                                                   scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_int_ub = tik_inst.Tensor(Constant.DTYPE_INT32,
                                                                                (output_num // 2,),
                                                                                name="indices_temp_int_ub",
                                                                                scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                  (output_num // 2,),
                                                                                  name="indices_temp_float_ub",
                                                                                  scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_simpling_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                                                                name="indices_simpling_ub",
                                                                                scope=tik.scope_ubuf)
                            repeat_time_255, left_part_n = _div(output_num // 2,
                                    Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
                            repeat_time_64_n, last_part_mask_n = _div(left_part_n, Constant.MASK_FP32)
                            args_init = (tik_inst, obj_ub_tensor.indices_temp_float_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init)
                            args_init_ub = (tik_inst, obj_ub_tensor.indices_temp_int_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init_ub)
                            # get the cached opt mode
                            _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                                tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_front_part_last_core,
                                obj_fp32_output_init_input_scalar.max_cache_n_num,
                                input_mov_tims_front_part_last_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core)
                            with tik_inst.if_scope(
                                    obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                                with tik_inst.for_range(0,
                                                        obj_fp32_input_data_input_scalar.\
                                                        last_rows_front_part_last_core) as \
                                        rows_index:
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                        input_mov_tims_front_part_last_core_index *
                                        obj_fp32_input_data_input_scalar.front_rows_front_part_last_core + rows_index])
                                    with tik_inst.if_scope(
                                            id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_0_ub_offset = id_val_scalar * \
                                                             obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                    output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                                    repeat_time_255x64)
                                        _tik_vadd_ub_tensor_hp(args_add)
                                    with tik_inst.else_scope():
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.\
                                                          last_axis_align_floor
                                        output_gm_offset = id_val_scalar * \
                                                           obj_fp32_e_num_input_scalar.e_num
                                        if not move_pad:
                                        # align part
                                            with tik_inst.if_scope(
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):

                                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                    input_ub_offset, output_gm_offset)
                                            # last part
                                            input_ub_offset_new = rows_index * \
                                                                  obj_fp32_output_init_input_scalar.\
                                                                  last_axis_align_floor + \
                                                                  obj_fp32_output_init_input_scalar.\
                                                                  last_axis_align_front_part
                                            output_gm_offset_new = id_val_scalar * \
                                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                                   obj_fp32_output_init_input_scalar. \
                                                                       last_axis_align_front_part
                                            vadd_mask = obj_fp32_output_init_input_scalar. \
                                                last_part_vadd_mask
                                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                                tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                                obj_gm_tensor.output_gm, input_ub_offset_new, output_gm_offset_new,
                                                vadd_mask)
                                        else:
                                            tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                                                   obj_ub_tensor.input_ub[input_ub_offset], 1,
                                                                   obj_fp32_e_num_input_scalar.move_pad, 0, 0)
                            with tik_inst.else_scope():
                                back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                                obj_common_scalar.ori_ids_base_offset.set_as(
                                    input_mov_tims_front_part_last_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_front_part_last_core)
                                _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                    tik_inst, obj_ub_tensor, obj_common_scalar,
                                    obj_fp32_input_data_input_scalar.last_rows_front_part_last_core,
                                    obj_common_scalar.ori_ids_base_offset, back_offset)
                                _tik_atomic_add_last_axis_not_align_small_e_hp_output_data(
                                    tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                    obj_common_scalar, obj_fp32_e_num_input_scalar,
                                    obj_fp32_input_data_input_scalar.last_rows_front_part_last_core, back_offset)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        # front part ids last part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
             
                        input_burst_len = obj_fp32_input_data_input_scalar.front_burst_len_last_part_last_core

                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.cached_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst,
                                                      input_burst_len)
                        e_num_align = obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_align_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align,
                                       output_num, 6)
                        with tik_inst.new_stmt_scope():
                            obj_ub_tensor.indices_index_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (output_num,),
                                                                                   name="indices_index_float_ub",
                                                                                   scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_int_ub = tik_inst.Tensor(Constant.DTYPE_INT32,
                                                                                (output_num // 2,),
                                                                                name="indices_temp_int_ub",
                                                                                scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                  (output_num // 2,),
                                                                                  name="indices_temp_float_ub",
                                                                                  scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_simpling_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                                                                name="indices_simpling_ub",
                                                                                scope=tik.scope_ubuf)
                            repeat_time_255, left_part_n = _div(output_num // 2,
                                    Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
                            repeat_time_64_n, last_part_mask_n = _div(left_part_n, Constant.MASK_FP32)
                            args_init = (tik_inst, obj_ub_tensor.indices_temp_float_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init)
                            args_init_ub = (tik_inst, obj_ub_tensor.indices_temp_int_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init_ub)
                            # get the cached opt mode
                            _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                                tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.front_rows_last_part_last_core,
                                obj_fp32_output_init_input_scalar.max_cache_n_num,
                                input_mov_tims_last_part_last_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_last_part_last_core)
                            with tik_inst.if_scope(
                                    obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                                with tik_inst.for_range(0,
                                                        obj_fp32_input_data_input_scalar.\
                                                        front_rows_last_part_last_core) as \
                                        rows_index:
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                        input_mov_tims_last_part_last_core_index *
                                        obj_fp32_input_data_input_scalar.front_rows_last_part_last_core + rows_index])
                                    with tik_inst.if_scope(
                                            id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_0_ub_offset = id_val_scalar * \
                                                             obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                    output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                                    repeat_time_255x64)
                                        _tik_vadd_ub_tensor_hp(args_add)
                                    with tik_inst.else_scope():
                                        # align part
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar. \
                                                              last_axis_align_floor
                                        output_gm_offset = id_val_scalar * \
                                                           obj_fp32_e_num_input_scalar.e_num
                                        if not move_pad:
                                            with tik_inst.if_scope(
                                                    obj_fp32_e_num_input_scalar. \
                                                            e_ub2gm_front_burst_len > 0):

                                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                    input_ub_offset, output_gm_offset)
                                            # last part
                                            input_ub_offset_new = rows_index * \
                                                                  obj_fp32_output_init_input_scalar. \
                                                                      last_axis_align_floor + \
                                                                  obj_fp32_output_init_input_scalar. \
                                                                      last_axis_align_front_part
                                            output_gm_offset_new = id_val_scalar * \
                                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                                   obj_fp32_output_init_input_scalar. \
                                                                       last_axis_align_front_part
                                            vadd_mask = obj_fp32_output_init_input_scalar. \
                                                last_part_vadd_mask
                                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                                tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                                obj_gm_tensor.output_gm, input_ub_offset_new, output_gm_offset_new,
                                                vadd_mask)
                                        else:
                                            tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                                                   obj_ub_tensor.input_ub[input_ub_offset], 1,
                                                                   obj_fp32_e_num_input_scalar.move_pad, 0, 0)
                            with tik_inst.else_scope():
                                back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                                obj_common_scalar.ori_ids_base_offset.set_as(
                                    input_mov_tims_last_part_last_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_last_core)
                                _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                    tik_inst, obj_ub_tensor, obj_common_scalar,
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_last_core,
                                    obj_common_scalar.ori_ids_base_offset, back_offset)
                                _tik_atomic_add_last_axis_not_align_small_e_hp_output_data(
                                    tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                    obj_common_scalar, obj_fp32_e_num_input_scalar,
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_last_core, back_offset)
                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                  
                        input_burst_len = obj_fp32_input_data_input_scalar.last_burst_len_last_part_last_core

                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.cached_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst,
                                                      input_burst_len)
                        e_num_align = obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_align_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align,
                                       output_num, 7)
                        with tik_inst.new_stmt_scope():
                            obj_ub_tensor.indices_index_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32, (output_num,),
                                                                                   name="indices_index_float_ub",
                                                                                   scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_int_ub = tik_inst.Tensor(Constant.DTYPE_INT32,
                                                                                (output_num // 2,),
                                                                                name="indices_temp_int_ub",
                                                                                scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_temp_float_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                  (output_num // 2,),
                                                                                  name="indices_temp_float_ub",
                                                                                  scope=tik.scope_ubuf)

                            obj_ub_tensor.indices_simpling_ub = tik_inst.Tensor(Constant.DTYPE_FP32,
                                                                                (Constant.CACHE_ACT_SIMPLING_BUFF_NUM,),
                                                                                name="indices_simpling_ub",
                                                                                scope=tik.scope_ubuf)
                            repeat_time_255, left_part_n = _div(output_num // 2,
                                    Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
                            repeat_time_64_n, last_part_mask_n = _div(left_part_n, Constant.MASK_FP32)
                            args_init = (tik_inst, obj_ub_tensor.indices_temp_float_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init)
                            args_init_ub = (tik_inst, obj_ub_tensor.indices_temp_int_ub,
                                    last_part_mask_n, repeat_time_64_n, repeat_time_255)
                            _tik_init_ub_tensor_hp(args_init_ub)
                            # get the cached opt mode
                            _tik_atomic_add_last_axis_align_small_e_hp_get_opt_mod(
                                tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, obj_common_scalar,
                                obj_fp32_input_data_input_scalar.last_rows_last_part_last_core,
                                obj_fp32_output_init_input_scalar.max_cache_n_num,
                                input_mov_tims_last_part_last_core_index *
                                obj_fp32_input_data_input_scalar.front_rows_last_part_last_core)
                            with tik_inst.if_scope(
                                    obj_common_scalar.cache_mode == Constant.INPUT_LAST_AXIS_ALIGN_HP_OPT_CACHE_N_NUM):
                                with tik_inst.for_range(
                                        0,
                                        obj_fp32_input_data_input_scalar.last_rows_last_part_last_core) as rows_index:
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                        input_mov_tims_last_part_last_core_index *
                                        obj_fp32_input_data_input_scalar.front_rows_last_part_last_core + rows_index])
                                    with tik_inst.if_scope(
                                            id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        output_0_ub_offset = id_val_scalar * \
                                                             obj_fp32_output_init_input_scalar.last_axis_align_floor
                                        args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                                    output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                                    repeat_time_255x64)
                                        _tik_vadd_ub_tensor_hp(args_add)
                                    with tik_inst.else_scope():
                                        # align part
                                        input_ub_offset = rows_index * \
                                                          obj_fp32_output_init_input_scalar. \
                                                              last_axis_align_floor
                                        output_gm_offset = id_val_scalar * \
                                                           obj_fp32_e_num_input_scalar.e_num
                                        if not move_pad:
                                            with tik_inst.if_scope(
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):

                                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                                    tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                    input_ub_offset, output_gm_offset)
                                            # last part
                                            input_ub_offset_new = rows_index * \
                                                                  obj_fp32_output_init_input_scalar. \
                                                                      last_axis_align_floor + \
                                                                  obj_fp32_output_init_input_scalar. \
                                                                      last_axis_align_front_part
                                            output_gm_offset_new = id_val_scalar * \
                                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                                   obj_fp32_output_init_input_scalar. \
                                                                       last_axis_align_front_part
                                            vadd_mask = obj_fp32_output_init_input_scalar. \
                                                last_part_vadd_mask
                                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                                tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                                obj_gm_tensor.output_gm, input_ub_offset_new, output_gm_offset_new,
                                                vadd_mask)
                                        else:
                                            tik_inst.data_move_pad(obj_gm_tensor.output_gm[output_gm_offset],
                                                                   obj_ub_tensor.input_ub[input_ub_offset], 1,
                                                                   obj_fp32_e_num_input_scalar.move_pad, 0, 0)
                            with tik_inst.else_scope():
                                back_offset = tik_inst.Scalar(dtype=Constant.DTYPE_INT32, name="back_offset")
                                obj_common_scalar.ori_ids_base_offset.set_as(
                                    input_mov_tims_last_part_last_core_index *
                                    obj_fp32_input_data_input_scalar.front_rows_last_part_last_core)
                                _tik_atomic_add_last_axis_align_small_e_hp_opt_cache_mod(
                                    tik_inst, obj_ub_tensor, obj_common_scalar,
                                    obj_fp32_input_data_input_scalar.last_rows_last_part_last_core,
                                    obj_common_scalar.ori_ids_base_offset, back_offset)
                                _tik_atomic_add_last_axis_not_align_small_e_hp_output_data(
                                    tik_inst, obj_ub_tensor, obj_gm_tensor, obj_fp32_output_init_input_scalar,
                                    obj_common_scalar, obj_fp32_e_num_input_scalar,
                                    obj_fp32_input_data_input_scalar.last_rows_last_part_last_core, back_offset)
   
    e_num_align = obj_fp32_output_init_input_scalar.last_axis_align_floor
    _tik_remove_pad(tik_inst, obj_ub_tensor, obj_fp32_e_num_input_scalar, e_num_align, output_num)
    with tik_inst.for_range(0, obj_fp32_e_num_input_scalar.e_num_sub) as dex:
        max_dex = obj_fp32_output_init_input_scalar.max_cache_n_num * obj_fp32_e_num_input_scalar.e_num
        obj_ub_tensor.output_0_ub[max_dex + dex].set_as(0)
    input_burst_len = obj_fp32_e_num_input_scalar.cache_num_block
    _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.output_0_ub,
                                                obj_gm_tensor.output_gm, input_burst_len, 0, 0)


def _tik_atomic_add_last_axis_not_align_small_e_hp(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor,
                                                   obj_common_scalar, obj_fp32_input_data_input_scalar,
                                                   obj_fp32_e_num_input_scalar, obj_fp32_ids_input_scalar,
                                                   obj_fp32_output_init_input_scalar, output_num):
    """
    _tik_atomic_add_last_axis_not_align_small_e_hp

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_fp32_output_init_input_scalar: obj_fp32_output_init_input_scalar
    output_num: output_num

    Returns
    -------
    None
    """
    repeat_time_255x64, left_part = _div(output_num, Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
    repeat_time_64, last_part_mask = _div(left_part, Constant.MASK_FP32)
    args_init = (tik_inst, obj_ub_tensor.output_0_ub, last_part_mask, repeat_time_64, repeat_time_255x64)
    _tik_init_ub_tensor_hp(args_init)

    repeat_time_255x64, left_part = _div(obj_fp32_e_num_input_scalar.e_num,
                                         Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME)
    repeat_time_64, last_part_mask = _div(left_part, Constant.MASK_FP32)

    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # front part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.front_rows_front_part_front_core
                        input_ele_num_one_row = obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_front_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_front_core +
                                                     rows_index])
                            with tik_inst.if_scope(id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_0_ub_offset = id_val_scalar * \
                                                     obj_fp32_output_init_input_scalar.last_axis_align_floor
                                args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                            output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                            repeat_time_255x64)
                                _tik_vadd_ub_tensor_hp(args_add)
                            with tik_inst.else_scope():
                                # align part
                                with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                    input_ub_offset = rows_index * \
                                                      obj_fp32_output_init_input_scalar.last_axis_align_floor
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                                # last part
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.\
                                                  last_axis_align_floor + \
                                                  obj_fp32_output_init_input_scalar.\
                                                  last_axis_align_front_part
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, obj_gm_tensor.output_gm,
                                    input_ub_offset, output_gm_offset, vadd_mask)
                    with tik_inst.if_scope(input_mov_tims_front_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_front_core - 1):
                        # last part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.last_rows_front_part_front_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.last_rows_front_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_front_core +
                                                     rows_index])
                            with tik_inst.if_scope(id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_0_ub_offset = id_val_scalar * \
                                                     obj_fp32_output_init_input_scalar.last_axis_align_floor
                                args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                            output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                            repeat_time_255x64)
                                _tik_vadd_ub_tensor_hp(args_add)
                            with tik_inst.else_scope():
                                # align part
                                with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                    input_ub_offset = rows_index * \
                                                      obj_fp32_output_init_input_scalar.last_axis_align_floor
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                                # last part
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_floor + \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_front_part
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, obj_gm_tensor.output_gm,
                                    input_ub_offset, output_gm_offset, vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # front part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.front_rows_last_part_front_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_last_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_front_core +
                                                     rows_index])
                            with tik_inst.if_scope(id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_0_ub_offset = id_val_scalar * \
                                                     obj_fp32_output_init_input_scalar.last_axis_align_floor
                                args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                            output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                            repeat_time_255x64)
                                _tik_vadd_ub_tensor_hp(args_add)
                            with tik_inst.else_scope():
                                # align part
                                with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                    input_ub_offset = rows_index * \
                                                      obj_fp32_output_init_input_scalar. \
                                                          last_axis_align_floor
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                                # last part
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_floor + \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_front_part
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   obj_fp32_output_init_input_scalar. \
                                                       last_axis_align_front_part
                                vadd_mask = obj_fp32_output_init_input_scalar. \
                                    last_part_vadd_mask
                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, obj_gm_tensor.output_gm,
                                    input_ub_offset, output_gm_offset, vadd_mask)
                    with tik_inst.if_scope(input_mov_tims_last_part_front_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_front_core - 1):
                        # last part ids last part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.last_rows_last_part_front_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.last_rows_last_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_front_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_front_core +
                                                     rows_index])
                            with tik_inst.if_scope(id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_0_ub_offset = id_val_scalar * \
                                                     obj_fp32_output_init_input_scalar.last_axis_align_floor
                                args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                            output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                            repeat_time_255x64)
                                _tik_vadd_ub_tensor_hp(args_add)
                            with tik_inst.else_scope():
                                # align part
                                with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                    input_ub_offset = rows_index * \
                                                      obj_fp32_output_init_input_scalar. \
                                                          last_axis_align_floor
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                                # last part
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor + \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, obj_gm_tensor.output_gm,
                                    input_ub_offset, output_gm_offset, vadd_mask)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index < obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # front part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar.front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar. \
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.front_rows_front_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.front_rows_front_part_last_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_last_core +
                                                     rows_index])
                            with tik_inst.if_scope(id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_0_ub_offset = id_val_scalar * \
                                                     obj_fp32_output_init_input_scalar.last_axis_align_floor
                                args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                            output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                            repeat_time_255x64)
                                _tik_vadd_ub_tensor_hp(args_add)
                            with tik_inst.else_scope():
                                # align part
                                with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                    input_ub_offset = rows_index * \
                                                      obj_fp32_output_init_input_scalar.last_axis_align_floor
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                                # last part
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor + \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, obj_gm_tensor.output_gm,
                                    input_ub_offset, output_gm_offset, vadd_mask)
                    with tik_inst.if_scope(input_mov_tims_front_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_front_part_last_core - 1):
                        # last part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_front_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.last_rows_front_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar.last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.\
                                                last_rows_front_part_last_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_front_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_front_part_last_core +
                                                     rows_index])
                            with tik_inst.if_scope(id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_0_ub_offset = id_val_scalar * \
                                                     obj_fp32_output_init_input_scalar.last_axis_align_floor
                                args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                            output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                            repeat_time_255x64)
                                _tik_vadd_ub_tensor_hp(args_add)
                            with tik_inst.else_scope():
                                # align part
                                with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                    input_ub_offset = rows_index * \
                                                      obj_fp32_output_init_input_scalar.last_axis_align_floor
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                                # last part
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor + \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_front_part
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   obj_fp32_output_init_input_scalar. \
                                                       last_axis_align_front_part
                                vadd_mask = obj_fp32_output_init_input_scalar. \
                                    last_part_vadd_mask
                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, obj_gm_tensor.output_gm,
                                    input_ub_offset, output_gm_offset, vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index == obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar. \
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                                   ids_n_burst, ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index <
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        # front part ids last part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar. \
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar. \
                                              e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar. \
                            front_rows_last_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar. \
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        with tik_inst.for_range(0,
                                                obj_fp32_input_data_input_scalar.\
                                                front_rows_last_part_last_core) as \
                                rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_last_core +
                                                     rows_index])
                            with tik_inst.if_scope(id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_0_ub_offset = id_val_scalar * \
                                                     obj_fp32_output_init_input_scalar.last_axis_align_floor
                                args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                            output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                            repeat_time_255x64)
                                _tik_vadd_ub_tensor_hp(args_add)
                            with tik_inst.else_scope():
                                # align part
                                with tik_inst.if_scope(
                                        obj_fp32_e_num_input_scalar. \
                                                e_ub2gm_front_burst_len > 0):
                                    input_ub_offset = rows_index * \
                                                      obj_fp32_output_init_input_scalar. \
                                                          last_axis_align_floor
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                                # last part
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_floor + \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_front_part
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   obj_fp32_output_init_input_scalar. \
                                                       last_axis_align_front_part
                                vadd_mask = obj_fp32_output_init_input_scalar. \
                                    last_part_vadd_mask
                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, obj_gm_tensor.output_gm,
                                    input_ub_offset, output_gm_offset, vadd_mask)
                    with tik_inst.if_scope(input_mov_tims_last_part_last_core_index ==
                                           obj_fp32_input_data_input_scalar.mov_times_gm2ub_last_part_last_core - 1):
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar. \
                                              ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                          input_mov_tims_last_part_last_core_index * \
                                          obj_fp32_input_data_input_scalar. \
                                              front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = \
                            obj_fp32_e_num_input_scalar. \
                                e_ub2gm_front_burst_len + \
                            obj_fp32_e_num_input_scalar. \
                                e_ub2gm_last_burst_len
                        input_mov_times = \
                            obj_fp32_input_data_input_scalar. \
                                last_rows_last_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32b = \
                            obj_fp32_output_init_input_scalar. \
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len,
                                                      input_mov_times, input_ele_num_one_row,
                                                      input_ele_num_one_row_align_32b)
                        with tik_inst.for_range(
                                0, obj_fp32_input_data_input_scalar.last_rows_last_part_last_core) as rows_index:
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[input_mov_tims_last_part_last_core_index *
                                                     obj_fp32_input_data_input_scalar.front_rows_last_part_last_core +
                                                     rows_index])
                            with tik_inst.if_scope(id_val_scalar < obj_fp32_output_init_input_scalar.max_cache_n_num):
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar.last_axis_align_floor
                                output_0_ub_offset = id_val_scalar * \
                                                     obj_fp32_output_init_input_scalar.last_axis_align_floor
                                args_add = (tik_inst, obj_ub_tensor.output_0_ub, obj_ub_tensor.input_ub,
                                            output_0_ub_offset, input_ub_offset, last_part_mask, repeat_time_64,
                                            repeat_time_255x64)
                                _tik_vadd_ub_tensor_hp(args_add)
                            with tik_inst.else_scope():
                                # align part
                                with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
                                    input_ub_offset = rows_index * \
                                                      obj_fp32_output_init_input_scalar. \
                                                          last_axis_align_floor
                                    output_gm_offset = id_val_scalar * \
                                                       obj_fp32_e_num_input_scalar.e_num
                                    _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                        tik_inst, obj_ub_tensor.input_ub, obj_gm_tensor.output_gm,
                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len, input_ub_offset,
                                        output_gm_offset)
                                # last part
                                input_ub_offset = rows_index * \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_floor + \
                                                  obj_fp32_output_init_input_scalar. \
                                                      last_axis_align_front_part
                                output_gm_offset = id_val_scalar * \
                                                   obj_fp32_e_num_input_scalar.e_num + \
                                                   obj_fp32_output_init_input_scalar. \
                                                       last_axis_align_front_part
                                vadd_mask = obj_fp32_output_init_input_scalar. \
                                    last_part_vadd_mask
                                _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(
                                    tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, obj_gm_tensor.output_gm,
                                    input_ub_offset, output_gm_offset, vadd_mask)

    with tik_inst.for_range(0, obj_fp32_output_init_input_scalar.max_cache_n_num) as cache_index:
        with tik_inst.if_scope(obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len > 0):
            input_ub_offset = cache_index * \
                              obj_fp32_output_init_input_scalar. \
                                  last_axis_align_floor
            output_gm_offset = cache_index * \
                               obj_fp32_e_num_input_scalar.e_num
            _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst, obj_ub_tensor.output_0_ub, obj_gm_tensor.output_gm,
                                                        obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                                                        input_ub_offset, output_gm_offset)
        # last part
        input_ub_offset = cache_index * \
                          obj_fp32_output_init_input_scalar. \
                              last_axis_align_floor + \
                          obj_fp32_output_init_input_scalar. \
                              last_axis_align_front_part
        output_gm_offset = cache_index * \
                           obj_fp32_e_num_input_scalar.e_num + \
                           obj_fp32_output_init_input_scalar. \
                               last_axis_align_front_part
        vadd_mask = obj_fp32_output_init_input_scalar.last_part_vadd_mask
        _tik_atomic_add_ub2gm_by_id_last_axis_not_align_fp32_one_block(tik_inst, obj_ub_tensor.output_0_ub,
                                                                       obj_ub_tensor.output_ub, obj_gm_tensor.output_gm,
                                                                       input_ub_offset, output_gm_offset, vadd_mask)


def unsorted_segment_sum_compute(x, segment_ids, var_num_segments, y, check_ids=False,
                                 kernel_name="UnsortedSegmentSum", impl_mode=OpImplMode.HIGH_PRECISION):
    if impl_mode == OpImplMode.SUPPORT_OUT_OF_BOUND_INDEX:
        check_ids = True
    res = tbe.segment(x, segment_ids, var_num_segments, 0, "segmentensor_sum", check_ids)
    return res


def unsorted_segment_sum_dsl(x,
                             segment_ids,
                             num_segments,
                             y_dict,
                             check_ids=False,
                             kernel_name="UnsortedSegmentSum",
                             impl_mode=OpImplMode.HIGH_PRECISION):
    dtype_x = x.get("dtype").lower()
    id_dtype = segment_ids.get("dtype").lower()
    dtype_num_segments = num_segments.get("dtype").lower()
    ins = classify([x, segment_ids, num_segments], OpPatternMode.SEGMENT, {"impl_mode": impl_mode})
    schedules, tensors = [], []
    for (input1, input2, input3) in ins:
        with tbe.compute():
            shape_x1, shape_x2, var_segments = \
                shape_util.variable_shape([input1, input2, input3], op_mode="segment")
            x_tensor = tvm.placeholder(shape_x1, name="var", dtype=dtype_x)
            ids_tensor = tvm.placeholder(shape_x2, name="segment_ids", dtype=id_dtype)
            segments_tensor = tvm.placeholder([1], name="num_segments", dtype=dtype_num_segments)
            res = unsorted_segment_sum_compute(x_tensor, ids_tensor, var_segments, y_dict,
                                               check_ids, kernel_name, impl_mode)
            tensors.append([x_tensor, ids_tensor, segments_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


def unsorted_segment_sum_tik(x,
                             segment_ids,
                             num_segments,
                             y_dict,
                             kernel_name="UnsortedSegmentSum",
                             impl_mode=OpImplMode.HIGH_PRECISION):
    obj = UnsortedSegmentSum(x, segment_ids, num_segments, y_dict, kernel_name, impl_mode=impl_mode)
    return obj.unsorted_segment_sum()


@register_operator("UnsortedSegmentSum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def unsorted_segment_sum(x_dict,
                         segment_ids_dict,
                         num_segments,
                         y_dict,
                         check_ids=False,
                         kernel_name="UnsortedSegmentSum",
                         impl_mode=OpImplMode.HIGH_PRECISION):
    """
    unsorted_segment_sum entry interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    segment_ids_dict: segment_ids shape, dtype and range
    num_segments: num_segments shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of UnsortedSegmentSum op
    impl_mode: impl_mode, only "high_performance" is supported

    Returns
    -------
    compile info
    """
    check_op_impl_mode(impl_mode,
                       [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION, OpImplMode.SUPPORT_OUT_OF_BOUND_INDEX],
                       kernel_name)
    dtype_x = x_dict.get("dtype").lower()
    dtype_x_check_list = ("float32", "float16", "int32", "bfloat16")
    para_check.check_dtype(dtype_x, dtype_x_check_list, param_name="x_dict")

    segment_ids_dtype = segment_ids_dict.get("dtype").lower()
    para_check.check_dtype(segment_ids_dtype, ("int32", "int64"), param_name="segment_ids_dict")

    unsorted_segment_sum_dsl(x_dict, segment_ids_dict, num_segments, y_dict, check_ids,
                             kernel_name, impl_mode)
