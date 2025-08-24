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
top_k_d
"""
# 'pylint: disable=too-many-lines
import functools
from enum import Enum
from enum import unique

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import OpPatternMode
from tbe.common.platform import get_bit_len
from impl.util.util_common import is_unknown_rank_input
from tbe.dsl.base import operation


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    MAX_INT32 = 2**31 - 1
    INDICES_NUM = MAX_INT32
    DTYPE_INT32 = "int32"
    TILING_PARAMS_NUM = 8
    MAX_SHAPE_SIZE = MAX_INT32
    TILING_PARAM_DTYPE = DTYPE_INT32
    # byte of one block
    BYTE_BLOCK = 32
    FULL_MASK_FP16 = 128
    FULL_MASK_INT32 = 64


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class SortParam:
    """
    The class for SortParam
    """
    SUPPORT_VBITSORT32 = False
    SORT_REGION_BYTE = 0
    SORT_ONCE_NUM = 0


def _get_dtype_byte(dtype):
    return get_bit_len(dtype) // 8


def _get_dtype_min_val(dtype):
    """
    get dtype min val
    """
    val_list = {'float16': -65536, 'float32': -(2 - 2**-23) * 2**127 + 1, 'int32': -(2**31), 'uint32': 0}
    return val_list.get(dtype)


def _ceil_fill(value, block):
    """
    Fill the input value by block.
    """
    return (value + block - 1) // block * block


def _ceil_div(value, block):
    """
    compute ceil div

    Parameters
    ----------
    value: num
    block: factor

    Returns
    -------
    ceil value
    """
    return (value + block - 1) // block


def _exec_front_last(tik_instance, num, part, fun):
    front = num // part
    last = num - front * part
    with tik_instance.for_range(0, front, name="front") as i:
        fun(i * part, part)
    with tik_instance.if_scope(last > 0):
        fun(front * part, last)


def isinstance_if(tik_instance, condition, fun):
    """
    isinstance_if
    """
    if isinstance(condition, int):
        if condition:
            fun()
    else:
        with tik_instance.if_scope(condition):
            fun()


# 'pylint: disable=invalid-name
@unique
class Mode(Enum):
    """Mode for Region proposal"""
    X1 = 0
    Y1 = 1
    X2 = 2
    Y2 = 3
    Score = 4


# 'pylint: disable=too-many-instance-attributes,too-many-public-methods
# 'pylint: disable=attribute-defined-outside-init
class GlobalVarGM:
    """GlobalVarGM Class Defination"""

    def __init__(self, tik_instance):
        """"
        __init__
        """
        self.data_gm = None
        self.data_gm_out = None
        self.indices_gm = None
        self.indices_gm_out = None
        self.tiling_gm = tik_instance.Tensor(Constant.DTYPE_INT32, (Constant.TILING_PARAMS_NUM,),
                                             name="tiling_gm",
                                             scope=tik.scope_gm)

    def set_data_gm(self, data_gm):
        """"
        set_data_gm
        """
        self.data_gm = data_gm

    def get_data_gm(self):
        """"
        get_data_gm
        """
        return self.data_gm

    def set_data_gm_out(self, data_gm_out):
        """"
        set_data_gm_out
        """
        self.data_gm_out = data_gm_out

    def get_data_gm_out(self):
        """"
        data_gm_out
        """
        return self.data_gm_out

    def set_indices_gm(self, indices_gm):
        """"
        set_indices_gm
        """
        self.indices_gm = indices_gm

    def get_indices_gm(self):
        """"
        get_indices_gm
        """
        return self.indices_gm

    def set_indices_gm_out(self, indices_gm_out):
        """"
        set_indices_gm_out
        """
        self.indices_gm_out = indices_gm_out

    def get_indices_gm_out(self):
        """"
        get_indices_gm_out
        """
        return self.indices_gm_out

    def set_offset_gm(self, offset_gm):
        """
        set_offset_gm
        """
        self.offset_gm = offset_gm

    def get_offset_gm(self):
        """
        get_offset_gm
        """
        return self.offset_gm


class GlobalVarUB:
    """GlobalVarUB Class Defination"""

    def __init__(self):
        """"
        __init__
        """
        self.indices_ub = None
        self.indices_out_fp16_ub = None
        self.indices_out_int32_ub = None
        self.data_tail_block_ub = None
        self.indices_tail_block_ub = None
        self.region_k_ub = None
        self.region_k2_ub = None
        self.data_ub = None
        self.region_ub = None
        self.region_sorted_ub = None

    def set_indices_ub(self, indices_ub):
        """"
        set_indices_ub
        """
        self.indices_ub = indices_ub

    def get_indices_ub(self):
        """"
        get_indices_ub
        """
        return self.indices_ub

    def set_indices_out_fp16_ub(self, indices_out_fp16_ub):
        """"
        set_indices_out_fp16_ub
        """
        self.indices_out_fp16_ub = indices_out_fp16_ub

    def get_indices_out_fp16_ub(self):
        """"
        get_indices_out_fp16_ub
        """
        return self.indices_out_fp16_ub

    def set_indices_out_int32_ub(self, indices_out_int32_ub):
        """"
        set_indices_out_int32_ub
        """
        self.indices_out_int32_ub = indices_out_int32_ub

    def get_indices_out_int32_ub(self):
        """"
        get_indices_out_int32_ub
        """
        return self.indices_out_int32_ub

    def set_data_tail_block_ub(self, data_tail_block_ub):
        """"
        set_data_tail_block_ub
        """
        self.data_tail_block_ub = data_tail_block_ub

    def get_data_tail_block_ub(self):
        """"
        get_data_tail_block_ub
        """
        return self.data_tail_block_ub

    def set_indices_tail_block_ub(self, indices_tail_block_ub):
        """"
        set_indices_tail_block_ub
        """
        self.indices_tail_block_ub = indices_tail_block_ub

    def get_indices_tail_block_ub(self):
        """"
        get_indices_tail_block_ub
        """
        return self.indices_tail_block_ub

    def set_region_k2_ub(self, region_k2_ub):
        """"
        set_region_k2_ub
        """
        self.region_k2_ub = region_k2_ub

    def get_region_k2_ub(self):
        """"
        get_region_k2_ub
        """
        return self.region_k2_ub

    def set_data_ub(self, data_ub):
        """"
        set_data_ub
        """
        self.data_ub = data_ub

    def get_data_ub(self):
        """"
        get_data_ub
        """
        return self.data_ub

    def set_region_ub(self, region_ub):
        """"
        set_region_ub
        """
        self.region_ub = region_ub

    def get_region_ub(self):
        """"
        get_region_ub
        """
        return self.region_ub

    def set_region_sorted_ub(self, region_sorted_ub):
        """"
        set_region_sorted_ub
        """
        self.region_sorted_ub = region_sorted_ub

    def get_region_sorted_ub(self):
        """"
        get_region_sorted_ub
        """
        return self.region_sorted_ub

    def set_region_k_ub(self, region_k_ub):
        """"
        set_region_k_ub
        """
        self.region_k_ub = region_k_ub

    def get_region_k_ub(self):
        """"
        get_region_k_ub
        """
        return self.region_k_ub

    def set_indices_out_final_ub(self, indices_out_final_ub):
        """
        set_indices_out_final_ub
        """
        self.indices_out_final_ub = indices_out_final_ub

    def get_indices_out_final_ub(self):
        """
        get_indices_out_final_ub
        """
        return self.indices_out_final_ub

    def set_offset_ub(self, offset_ub):
        """
        set_offset_ub
        """
        self.offset_ub = offset_ub

    def get_offset_ub(self):
        """
        get_offset_ub
        """
        return self.offset_ub

    def set_offset_fp16_ub(self, offset_fp16_ub):
        """
        set_offset_fp16_ub
        """
        self.offset_fp16_ub = offset_fp16_ub

    def get_offset_fp16_ub(self):
        """
        get_offset_fp16_ub
        """
        return self.offset_fp16_ub

    def set_offset_int32_ub(self, offset_int32_ub):
        """
        set_offset_int32_ub
        """
        self.offset_int32_ub = offset_int32_ub

    def get_offset_int32_ub(self):
        """
        get_offset_int32_ub
        """
        return self.offset_int32_ub


class GlobalVarTilingScalar:
    """GlobalVarTilingScalar Class Defination"""

    # 'pylint:disable=too-many-arguments
    def __init__(self, tik_instance, tiling_gm, mode, k, input_shape):
        """
        constructor of class CommonScalar

        Parameters
        ----------
        tik_instance: tik_instance
        Returns
        -------
        None
        """

        profile = tik.Dprofile()
        self.ub_size = profile.get_unified_buffer_size()
        self.core_num = profile.get_aicore_num()

        # indices_per_part is used to avoid Memory trampling
        self.indices_per_part = 1024
        if SortParam.SUPPORT_VBITSORT32:
            # there are 40*batch_cols_padding ub in set_tensor_less_4096
            self.batch_cols_padding = (self.ub_size - 1024) // 40
            # there are 120*cols_per_part ub in set_tensor_more_4096
            self.cols_per_part = (self.ub_size - 1024) // 120 // \
                                 tbe_platform.VECTOR_INST_BLOCK_WIDTH * tbe_platform.VECTOR_INST_BLOCK_WIDTH
        else:
            # there are 54*batch_cols_padding ub in set_tensor_less_4096
            self.batch_cols_padding = (self.ub_size - 1024) // 54
            # there are 240*cols_per_part ub in set_tensor_more_4096
            self.cols_per_part = (self.ub_size - 1024) // 240 // \
                                 tbe_platform.VECTOR_INST_BLOCK_WIDTH * tbe_platform.VECTOR_INST_BLOCK_WIDTH
        self.max_region_len = 5 * self.cols_per_part

        self.mode_threshold = self.batch_cols_padding // 1024 * 1024

        self._update_tiling_value(mode, k, tik_instance, tiling_gm, input_shape)

    def _update_tiling_value(self, mode, k, tik_instance, tiling_gm, input_shape):
        if mode == "dynamic":
            self.need_core_num_input_scalar = tik_instance.Scalar(dtype="int32", name="need_core_num_input_scalar")
            self.num_rows_scalar = tik_instance.Scalar(dtype="int32", name="num_rows_scalar")
            self.num_cols_scalar = tik_instance.Scalar(dtype="int32", name="num_cols_scalar")
            self.num_rows_cores_scalar = tik_instance.Scalar(dtype="int32", name="num_rows_cores_scalar")
            self.num_turn_scalar = tik_instance.Scalar(dtype="int32", name="num_turn_scalar")
            self.num_batch_scalar = tik_instance.Scalar(dtype="int32", name="num_batch_scalar")
            self.num_k_scalar = tik_instance.Scalar(dtype="int32", name="num_k_scalar")
            self.loop_times_scalar = tik_instance.Scalar(dtype="int32", name="loop_times_scalar")

            self.tiling_ub = tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            # mov tiling params from gm to ub
            tik_instance.data_move(self.tiling_ub, tiling_gm, 0, 1,
                Constant.TILING_PARAMS_NUM * _get_dtype_byte(self.tiling_ub.dtype) // Constant.BYTE_BLOCK, 0, 0)
            # input scalar in flowtable
            input_scalar_index = 0
            self.need_core_num_input_scalar.set_as(self.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.num_rows_scalar.set_as(self.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.num_cols_scalar.set_as(self.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.num_k_scalar.set_as(self.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.loop_times_scalar.set_as(self.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.num_batch_scalar.set_as(self.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.num_rows_cores_scalar.set_as(self.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.num_turn_scalar.set_as(self.tiling_ub[input_scalar_index])
        else:
            need_core, row, col, loops, batch, rows_per_core, turning = self.topk_tiling(k, input_shape)
            self.need_core_num_input_scalar = need_core
            self.num_rows_scalar = row
            self.num_cols_scalar = col
            self.num_k_scalar = k
            self.loop_times_scalar = loops
            self.num_batch_scalar = batch
            self.num_rows_cores_scalar = rows_per_core
            self.num_turn_scalar = turning

    @staticmethod
    def _get_loop_times(cols):
        level = 0
        regions = _ceil_div(cols, SortParam.SORT_ONCE_NUM)
        if regions <= 1:
            return level

        while True:
            level += 1
            regions = _ceil_div(regions, 4)
            if regions <= 1:
                break
        return _ceil_fill(level, 2)

    # 'pylint:disable=too-many-return-values
    def topk_tiling(self, k_num, input_shape):
        """
        topk_tiling
        """
        input_dims = len(input_shape)
        row = 1
        for i in range(input_dims - 1):
            row *= input_shape[i]

        col = input_shape[input_dims - 1]

        if row <= self.core_num:
            rows_per_core = 1
            need_core = row
            batch = 1
            turning = self.core_num
        else:
            need_core = self.core_num
            cols_padding = _ceil_fill(col, SortParam.SORT_ONCE_NUM)
            remain = row % self.core_num
            # need +1 in op for mode2
            rows_per_core = _ceil_div(row, self.core_num)
            if col <= self.batch_cols_padding // 1024 * 1024:
                batch = self.batch_cols_padding // cols_padding
            else:
                batch = 1
            turning = remain
            if remain == 0:
                turning = self.core_num
        if 0 < k_num < SortParam.SORT_ONCE_NUM:
            need_core = 1
            rows_per_core = row
        loops = self._get_loop_times(col)
        return need_core, row, col, loops, batch, rows_per_core, turning

    def get_loop_times(self):
        """
        get_k_num
        """
        return self.loop_times_scalar

    def get_k_num(self):
        """
        get_k_num
        """
        return self.num_k_scalar

    def get_core_num(self):
        """
        get_rows_num
        """
        return self.need_core_num_input_scalar

    def get_rows_num(self):
        """
        get_rows_num
        """
        return self.num_rows_scalar

    def get_cols_num(self):
        """
        get_cols_num
        """
        return self.num_cols_scalar

    def get_rows_cores(self):
        """
        get_cols_num
        """
        return self.num_rows_cores_scalar

    def get_turn_num(self):
        """
        get_cols_num
        """
        return self.num_turn_scalar

    def get_batch_num(self):
        """
        get_cols_num
        """
        return self.num_batch_scalar


# 'pylint: disable=too-many-locals,too-many-arguments
def set_tensor_more_4096(tik_instance, obj_tiling, obj_gm, obj_ub, ins, outs):
    """
    Set UB when tensor bigger than 4096
    """
    input_a = ins[0]
    indices = ins[-1]
    output = outs[0]
    indices_out = outs[1]
    cols_per_part = obj_tiling.cols_per_part
    max_region_len = obj_tiling.max_region_len
    data_ub = tik_instance.Tensor("float16", (cols_per_part,), name="data_ub", scope=tik.scope_ubuf)
    indices_ub = tik_instance.Tensor("float16", (cols_per_part,), name="indices_ub", scope=tik.scope_ubuf)
    indices_out_fp16_ub = indices_ub
    indices_out_int32_ub = tik_instance.Tensor("int32", (1, max_region_len),
                                               name="indices_out_int32_ub",
                                               scope=tik.scope_ubuf)
    indices_out_final_ub = indices_out_int32_ub
    offset_ub = tik_instance.Tensor("float16", (cols_per_part * 2,), name="offset_ub", scope=tik.scope_ubuf)
    offset_fp16_ub = offset_ub
    offset_int32_ub = tik_instance.Tensor("int32", (1, max_region_len), name="offset_int32_ub", scope=tik.scope_ubuf)
    region_ub = tik_instance.Tensor("float16", (1, cols_per_part * 8), name="region_ub", scope=tik.scope_ubuf)
    region_sorted_ub = tik_instance.Tensor("float16", (1, cols_per_part * 8),
                                           name="region_sorted_ub",
                                           scope=tik.scope_ubuf)
    region_k_ub = tik_instance.Tensor("float16", (1, max_region_len * 8), name="region_k_ub", scope=tik.scope_ubuf)
    region_k2_ub = tik_instance.Tensor("float16", (1, max_region_len * 8), name="region_k2_ub", scope=tik.scope_ubuf)
    data_tail_block_ub = tik_instance.Tensor("float16", (SortParam.SORT_ONCE_NUM,),
                                             name="data_tail_block_ub",
                                             scope=tik.scope_ubuf)
    indices_tail_block_ub = tik_instance.Tensor("int32", (SortParam.SORT_ONCE_NUM,),
                                                name="indices_tail_block_ub",
                                                scope=tik.scope_ubuf)
    obj_gm.set_data_gm_out(output)
    obj_ub.set_data_ub(data_ub)
    obj_ub.set_region_ub(region_ub)
    obj_ub.set_region_sorted_ub(region_sorted_ub)
    obj_ub.set_region_k_ub(region_k_ub)
    obj_ub.set_indices_ub(indices_ub)
    obj_ub.set_indices_out_fp16_ub(indices_out_fp16_ub)
    obj_ub.set_indices_out_int32_ub(indices_out_int32_ub)
    obj_gm.set_indices_gm_out(indices_out)
    obj_gm.set_data_gm(input_a)
    obj_gm.set_indices_gm(indices)
    obj_ub.set_region_k2_ub(region_k2_ub)
    obj_ub.set_offset_ub(offset_ub)
    obj_ub.set_offset_fp16_ub(offset_fp16_ub)
    obj_ub.set_offset_int32_ub(offset_int32_ub)
    obj_ub.set_indices_out_final_ub(indices_out_final_ub)
    obj_ub.set_data_tail_block_ub(data_tail_block_ub)
    obj_ub.set_indices_tail_block_ub(indices_tail_block_ub)


# 'pylint: disable=too-many-locals,too-many-arguments
def set_tensor_more_4096_a100(tik_instance, obj_tiling, obj_gm, obj_ub, ins, outs, dtype):
    """
    Set UB when tensor bigger than 4096
    """
    input_a = ins[0]
    indices = ins[-1]
    output = outs[0]
    indices_out = outs[1]
    cols_per_part = obj_tiling.cols_per_part
    max_region_len = obj_tiling.max_region_len
    data_ub = tik_instance.Tensor(dtype, (cols_per_part,), name="data_ub", scope=tik.scope_ubuf)
    indices_ub = tik_instance.Tensor(dtype, (cols_per_part,), name="indices_ub", scope=tik.scope_ubuf)
    indices_out_int32_ub = tik_instance.Tensor("int32", (1, max_region_len),
                                               name="indices_out_int32_ub",
                                               scope=tik.scope_ubuf)
    indices_out_final_ub = indices_out_int32_ub
    region_ub = tik_instance.Tensor("uint32", (cols_per_part * 2,), name="region_ub", scope=tik.scope_ubuf)
    region_sorted_ub = tik_instance.Tensor("uint32", (1, cols_per_part * 2),
                                           name="region_sorted_ub",
                                           scope=tik.scope_ubuf)
    region_k_ub = tik_instance.Tensor("uint32", (1, max_region_len * 2), name="region_k_ub", scope=tik.scope_ubuf)
    region_k2_ub = tik_instance.Tensor("uint32", (1, max_region_len * 2), name="region_k2_ub", scope=tik.scope_ubuf)
    data_tail_block_ub = tik_instance.Tensor(dtype, (SortParam.SORT_ONCE_NUM,), name="data_tail_block_ub",
                                             scope=tik.scope_ubuf)
    indices_tail_block_ub = tik_instance.Tensor("int32", (SortParam.SORT_ONCE_NUM,),
                                                name="indices_tail_block_ub",
                                                scope=tik.scope_ubuf)
    obj_gm.set_data_gm_out(output)
    obj_ub.set_data_ub(data_ub)
    obj_ub.set_region_ub(region_ub)
    obj_ub.set_region_sorted_ub(region_sorted_ub)
    obj_ub.set_region_k_ub(region_k_ub)
    obj_ub.set_indices_ub(indices_ub)
    obj_gm.set_indices_gm_out(indices_out)
    obj_gm.set_data_gm(input_a)
    obj_gm.set_indices_gm(indices)
    obj_ub.set_region_k2_ub(region_k2_ub)
    obj_ub.set_indices_out_final_ub(indices_out_final_ub)
    obj_ub.set_data_tail_block_ub(data_tail_block_ub)
    obj_ub.set_indices_tail_block_ub(indices_tail_block_ub)


# 'pylint: disable=too-many-arguments,too-many-locals
def set_tensor_less_4096(tik_instance, obj_tiling, obj_gm, obj_ub, ins, outs):
    """
    Set UB when tensor less than 4096
    """
    input_a = ins[0]
    indices = ins[-1]
    output = outs[0]
    indices_out = outs[1]
    batch_cols_padding = obj_tiling.batch_cols_padding
    data_ub = tik_instance.Tensor("float16", (batch_cols_padding,), name="data_ub", scope=tik.scope_ubuf)
    indices_ub = tik_instance.Tensor("float16", (batch_cols_padding,), name="indices_ub", scope=tik.scope_ubuf)
    indices_out_fp16_ub = tik_instance.Tensor("float16", (batch_cols_padding,),
                                              name="indices_out_fp16_ub",
                                              scope=tik.scope_ubuf)
    indices_out_int32_ub = tik_instance.Tensor("int32", (batch_cols_padding,),
                                               name="indices_out_int32_ub",
                                               scope=tik.scope_ubuf)
    indices_out_final_ub = tik_instance.Tensor("int32", (batch_cols_padding,),
                                               name="indices_out_final_ub",
                                               scope=tik.scope_ubuf)
    offset_ub = tik_instance.Tensor("float16", (batch_cols_padding,), name="offset_ub", scope=tik.scope_ubuf)
    offset_fp16_ub = tik_instance.Tensor("float16", (batch_cols_padding,), name="offset_fp16_ub", scope=tik.scope_ubuf)
    offset_int32_ub = tik_instance.Tensor("int32", (batch_cols_padding,), name="offset_int32_ub", scope=tik.scope_ubuf)
    region_ub = tik_instance.Tensor("float16", (batch_cols_padding * 8,), name="region_ub", scope=tik.scope_ubuf)
    region_sorted_ub = tik_instance.Tensor("float16", (batch_cols_padding * 8,),
                                           name="region_sorted_ub",
                                           scope=tik.scope_ubuf)
    data_tail_block_ub = tik_instance.Tensor("float16", (SortParam.SORT_ONCE_NUM,),
                                             name="data_tail_block_ub",
                                             scope=tik.scope_ubuf)
    indices_tail_block_ub = tik_instance.Tensor("int32", (SortParam.SORT_ONCE_NUM,),
                                                name="indices_tail_block_ub",
                                                scope=tik.scope_ubuf)
    obj_gm.set_data_gm_out(output)
    obj_ub.set_data_ub(data_ub)
    obj_ub.set_region_ub(region_ub)
    obj_ub.set_region_sorted_ub(region_sorted_ub)
    obj_ub.set_indices_ub(indices_ub)
    obj_ub.set_indices_out_fp16_ub(indices_out_fp16_ub)
    obj_ub.set_indices_out_int32_ub(indices_out_int32_ub)
    obj_gm.set_indices_gm_out(indices_out)
    obj_gm.set_data_gm(input_a)
    obj_gm.set_indices_gm(indices)
    obj_ub.set_offset_ub(offset_ub)
    obj_ub.set_offset_fp16_ub(offset_fp16_ub)
    obj_ub.set_offset_int32_ub(offset_int32_ub)
    obj_ub.set_indices_out_final_ub(indices_out_final_ub)
    obj_ub.set_data_tail_block_ub(data_tail_block_ub)
    obj_ub.set_indices_tail_block_ub(indices_tail_block_ub)


# 'pylint: disable=too-many-arguments,too-many-locals
def set_tensor_less_4096_a100(tik_instance, obj_tiling, obj_gm, obj_ub, ins, outs, dtype):
    """
    Set UB when tensor less than 4096
    """
    input_a = ins[0]
    indices = ins[-1]
    output = outs[0]
    indices_out = outs[1]
    batch_cols_padding = obj_tiling.batch_cols_padding
    data_ub = tik_instance.Tensor(dtype, (batch_cols_padding,), name="data_ub", scope=tik.scope_ubuf)
    indices_ub = tik_instance.Tensor(dtype, (batch_cols_padding,), name="indices_ub", scope=tik.scope_ubuf)
    indices_out_final_ub = tik_instance.Tensor("int32", (batch_cols_padding,),
                                               name="indices_out_final_ub",
                                               scope=tik.scope_ubuf)
    region_ub = tik_instance.Tensor("uint32", (batch_cols_padding * 4,), name="region_ub", scope=tik.scope_ubuf)
    region_sorted_ub = tik_instance.Tensor("uint32", (batch_cols_padding * 4,),
                                           name="region_sorted_ub",
                                           scope=tik.scope_ubuf)
    data_tail_block_ub = tik_instance.Tensor(dtype, (SortParam.SORT_ONCE_NUM,), name="data_tail_block_ub",
                                             scope=tik.scope_ubuf)
    indices_tail_block_ub = tik_instance.Tensor("int32", (SortParam.SORT_ONCE_NUM,),
                                                name="indices_tail_block_ub",
                                                scope=tik.scope_ubuf)
    obj_gm.set_data_gm_out(output)
    obj_ub.set_data_ub(data_ub)
    obj_ub.set_region_ub(region_ub)
    obj_ub.set_region_sorted_ub(region_sorted_ub)
    obj_ub.set_indices_ub(indices_ub)
    obj_gm.set_indices_gm_out(indices_out)
    obj_gm.set_data_gm(input_a)
    obj_gm.set_indices_gm(indices)
    obj_ub.set_indices_out_final_ub(indices_out_final_ub)
    obj_ub.set_data_tail_block_ub(data_tail_block_ub)
    obj_ub.set_indices_tail_block_ub(indices_tail_block_ub)


class GlobalVarFunction:
    """GlobalVarFunction Class Defination"""

    def __init__(self, obj_gm, obj_tiling, obj_ub):
        """
        constructor of class Function

        Parameters
        ----------
        obj_gm: obj_gm
        obj_tiling: obj_tiling
        obj_ub: obj_ub
        Returns
        -------
        None
        """
        self.func_obj_gm = obj_gm
        self.func_obj_tiling = obj_tiling
        self.func_obj_ub = obj_ub
        self.data_gm = obj_gm.get_data_gm()
        self.data_ub = obj_ub.get_data_ub()
        self.data_gm_out = obj_gm.get_data_gm_out()
        self.indices_gm = obj_gm.get_indices_gm()
        self.indices_gm_out = obj_gm.get_indices_gm_out()
        self.indices_ub = obj_ub.get_indices_ub()
        if not SortParam.SUPPORT_VBITSORT32:
            self.indices_out_fp16_ub = obj_ub.get_indices_out_fp16_ub()
            self.indices_out_int32_ub = obj_ub.get_indices_out_int32_ub()
        self.region_ub = obj_ub.get_region_ub()
        self.region_sorted_ub = obj_ub.get_region_sorted_ub()
        self.region_k_ub = obj_ub.get_region_k_ub()
        self.region_k2_ub = obj_ub.get_region_k2_ub()
        self.data_tail_block_ub = obj_ub.get_data_tail_block_ub()
        self.indices_tail_block_ub = obj_ub.get_indices_tail_block_ub()
        if not SortParam.SUPPORT_VBITSORT32:
            self.offset_ub = obj_ub.get_offset_ub()
            self.offset_fp16_ub = obj_ub.get_offset_fp16_ub()
            self.offset_int32_ub = obj_ub.get_offset_int32_ub()
        self.indices_out_final_ub = obj_ub.get_indices_out_final_ub()
        self.rows = obj_tiling.get_rows_num()
        self.cols = obj_tiling.get_cols_num()
        self.cols_padding = _ceil_fill(self.cols, SortParam.SORT_ONCE_NUM)
        self.k = obj_tiling.get_k_num()
        self.core_num = obj_tiling.core_num
        self.loop_times = obj_tiling.get_loop_times()
        self.rows_per_core = obj_tiling.get_rows_cores()
        self.batch_num = obj_tiling.get_batch_num()
        self.turn_block_idx = obj_tiling.get_turn_num()
        self.num_per_ele = SortParam.SORT_REGION_BYTE // _get_dtype_byte(self.data_ub.dtype)

    # 'pylint: disable=locally-disabled,too-many-locals,too-many-arguments
    def kernel_ir(self, tik_instance, largest, by_part, block_idx, block_dim, k):
        """
        Funtion for common process in top_k op
        """
        cols = self.cols
        multi_core = tik_instance.Scalar(init_value=1)
        with tik_instance.if_scope(block_dim <= 1):
            multi_core.set_as(0)

        turn = self.turn_block_idx
        batch = self.batch_num
        loops = tik_instance.Scalar("int32")
        remain = tik_instance.Scalar("int32")
        core_rows_start_scalar = tik_instance.Scalar("int32")
        with tik_instance.if_scope(block_idx < turn):
            loops.set_as(self.rows_per_core // batch)
            remain.set_as(self.rows_per_core - loops * batch)
            core_rows_start_scalar.set_as(self.rows_per_core * block_idx)
        with tik_instance.else_scope():
            loops.set_as((self.rows_per_core - 1) // batch)
            remain.set_as((self.rows_per_core - 1) - loops * batch)
            core_rows_start_scalar.set_as(self.rows_per_core * block_idx - (block_idx - turn))

        if by_part:
            with tik_instance.for_range(0, loops, name='i0') as i:
                self.topk_a_row_by_part(tik_instance,
                                        row_start_in_core=i,
                                        cols=cols,
                                        k=k,
                                        core_rows_start=core_rows_start_scalar,
                                        multi_core=multi_core,
                                        largest=largest)
        else:
            with tik_instance.for_range(0, loops, name='i0') as i:
                self.topk_rows(tik_instance,
                               row_start_in_core=i * batch,
                               rows=batch,
                               cols=cols,
                               k=k,
                               core_rows_start=core_rows_start_scalar,
                               multi_core=multi_core,
                               largest=largest)
            with tik_instance.if_scope(remain > 0):
                self.topk_rows(tik_instance,
                               row_start_in_core=loops * batch,
                               rows=remain,
                               cols=cols,
                               k=k,
                               core_rows_start=core_rows_start_scalar,
                               multi_core=multi_core,
                               largest=largest)

    # 'pylint: disable=too-many-arguments,too-many-statements,too-many-branches
    def topk_a_row_by_part(self, tik_instance, row_start_in_core, cols, k, core_rows_start, multi_core, largest):
        """
        topk_a_row_by_part
        """
        data_gm = self.data_gm
        data_ub = self.data_ub
        data_gm_out = self.data_gm_out
        indices_gm = self.indices_gm
        indices_gm_out = self.indices_gm_out
        indices_ub = self.indices_ub
        region_ub = self.region_ub
        if SortParam.SUPPORT_VBITSORT32:
            region_sorted_ub = self.region_sorted_ub.reinterpret_cast_to(data_ub.dtype)
            region_k_ub = self.region_k_ub.reinterpret_cast_to(data_ub.dtype)
            region_k2_ub = self.region_k2_ub.reinterpret_cast_to(data_ub.dtype)
        else:
            region_sorted_ub = self.region_sorted_ub
            region_k_ub = self.region_k_ub
            region_k2_ub = self.region_k2_ub
            offset_ub = self.offset_ub
            offset_int32_ub = self.offset_int32_ub
        data_tail_block_ub = self.data_tail_block_ub
        indices_tail_block_ub = self.indices_tail_block_ub
        indices_out_final_ub = self.indices_out_final_ub
        cols_per_part = self.func_obj_tiling.cols_per_part
        indices_per_part = self.func_obj_tiling.indices_per_part
        k_padding = _ceil_fill(k, SortParam.SORT_ONCE_NUM)
        cols_padding = _ceil_fill(cols, SortParam.SORT_ONCE_NUM)
        part_cnt = _ceil_div(cols, cols_per_part)
        last_part_cols = cols - ((part_cnt - 1) * cols_per_part)
        last_part_cols_padding = _ceil_fill(last_part_cols, SortParam.SORT_ONCE_NUM)
        gm_offset = row_start_in_core * cols + core_rows_start * cols
        multiplier_scalar = tik_instance.Scalar(data_ub.dtype)

        self.copy_gm_to_ubuf_func(tik_instance,
                                  data_ub,
                                  data_gm,
                                  num_rows=1,
                                  cols=cols_per_part,
                                  col_start=0,
                                  gm_offset=gm_offset,
                                  largest=largest)

        vadds_len = tbe_platform.VECTOR_INST_BLOCK_WIDTH // _get_dtype_byte(indices_out_final_ub.dtype)
        if SortParam.SUPPORT_VBITSORT32:
            indices_num = min(indices_per_part, cols_per_part)
            indices_block_num = indices_num * _get_dtype_byte(indices_gm.dtype) // tbe_platform.BLOCK_REDUCE_INT8
            tik_instance.data_move(indices_ub, indices_gm, 0, 1, indices_block_num, 0, 0)
            self.conv_fp162s32(tik_instance, indices_out_final_ub, 0, indices_ub.reinterpret_cast_to(indices_gm.dtype),
                               0, indices_num)
            for i in range(1, _ceil_div(cols_per_part, indices_per_part)):
                tik_instance.vadds(vadds_len, self.indices_out_final_ub[indices_per_part * i],
                                   self.indices_out_final_ub, indices_per_part * i, indices_per_part // vadds_len, 1, 1,
                                   8, 8)
            self.sort_region_a100(tik_instance, region_sorted_ub, data_ub, indices_out_final_ub, 1, cols_per_part)
        else:
            # indices_ub is used to store multiplier
            indices_block_num = indices_ub.buffer_size * _get_dtype_byte(
                indices_gm.dtype) // tbe_platform.BLOCK_REDUCE_INT8
            tik_instance.data_move(offset_ub, indices_gm, 0, 1, indices_block_num, 0, 0)
            tik_instance.vector_dup(indices_block_num, indices_ub, 0.0,
                                    indices_ub.buffer_size // tbe_platform.VECTOR_INST_BLOCK_WIDTH, 1, 8)
            self.emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Score.value, cnt=cols_per_part)
            self.emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Y2.value, cnt=cols_per_part)
            self.emit_vconcat(tik_instance, region_ub, indices_ub, mode=Mode.X1.value, cnt=cols_per_part)
            self.emit_vconcat(tik_instance, region_ub, offset_ub, mode=Mode.Y1.value, cnt=cols_per_part)

            self.sort_region(tik_instance, region_sorted_ub, region_ub, 1, cols_per_part)
        result_ub = region_sorted_ub
        self.copy_region(tik_instance, dst=region_k_ub, src=result_ub, num=cols_per_part)
        if isinstance(part_cnt, int):
            part_cnt = max(part_cnt, 2)
        with tik_instance.for_range(0, part_cnt - 2, name='topk_i0') as i:
            self.copy_gm_to_ubuf_func(tik_instance,
                                      data_ub,
                                      data_gm,
                                      num_rows=1,
                                      cols=cols_per_part,
                                      col_start=cols_per_part * (i + 1),
                                      gm_offset=gm_offset,
                                      largest=largest)

            if SortParam.SUPPORT_VBITSORT32:
                tik_instance.vadds(vadds_len, indices_out_final_ub, indices_out_final_ub, cols_per_part,
                                   cols_per_part // vadds_len, 1, 1, 8, 8)
                self.sort_region_a100(tik_instance, region_sorted_ub, data_ub, indices_out_final_ub, 1, cols_per_part)
            else:
                multiplier_scalar.set_as(offset_ub[i + 1])
                tik_instance.vector_dup(indices_block_num, indices_ub, multiplier_scalar,
                                        indices_ub.buffer_size // tbe_platform.VECTOR_INST_BLOCK_WIDTH, 1, 8)
                self.emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Score.value, cnt=cols_per_part)
                self.emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Y2.value, cnt=cols_per_part)
                self.emit_vconcat(tik_instance, region_ub, indices_ub, mode=Mode.X1.value, cnt=cols_per_part)
                self.emit_vconcat(tik_instance, region_ub, offset_ub, mode=Mode.Y1.value, cnt=cols_per_part)

                self.sort_region(tik_instance, region_sorted_ub, region_ub, 1, cols_per_part)
            result_ub = region_sorted_ub
            with tik_instance.if_scope(i == 0):
                self.merge_two_sorted_region(tik_instance,
                                             dst=region_k2_ub,
                                             src_region_k=region_k_ub,
                                             src_region_sorted=result_ub,
                                             len_region_k=cols_per_part,
                                             len_region_sorted=cols_per_part)
                self.copy_region(tik_instance, dst=region_k_ub, src=region_k2_ub, num=cols_per_part * 2)
            with tik_instance.if_scope(i == 1):
                self.merge_two_sorted_region(tik_instance,
                                             dst=region_k2_ub,
                                             src_region_k=region_k_ub,
                                             src_region_sorted=result_ub,
                                             len_region_k=cols_per_part * 2,
                                             len_region_sorted=cols_per_part)
                self.copy_region(tik_instance, dst=region_k_ub, src=region_k2_ub, num=cols_per_part * 3)
            with tik_instance.if_scope(i == 2):
                self.merge_two_sorted_region(tik_instance,
                                             dst=region_k2_ub,
                                             src_region_k=region_k_ub,
                                             src_region_sorted=result_ub,
                                             len_region_k=cols_per_part * 3,
                                             len_region_sorted=cols_per_part)
                self.copy_region(tik_instance, dst=region_k_ub, src=region_k2_ub, num=cols_per_part * 4)
            with tik_instance.if_scope(i >= 3):
                self.merge_two_sorted_region(tik_instance,
                                             dst=region_k2_ub,
                                             src_region_k=region_k_ub,
                                             src_region_sorted=result_ub,
                                             len_region_k=cols_per_part * 4,
                                             len_region_sorted=cols_per_part)
                self.copy_region(tik_instance, dst=region_k_ub, src=region_k2_ub, num=cols_per_part * 5)

        self.copy_gm_to_ubuf_func(tik_instance,
                                  data_ub,
                                  data_gm,
                                  num_rows=1,
                                  cols=last_part_cols,
                                  col_start=(part_cnt - 1) * cols_per_part,
                                  gm_offset=gm_offset,
                                  largest=largest)

        if SortParam.SUPPORT_VBITSORT32:
            tik_instance.vadds(vadds_len, indices_out_final_ub, indices_out_final_ub, cols_per_part,
                               cols_per_part // vadds_len, 1, 1, 8, 8)
            self.sort_region_a100(tik_instance, region_sorted_ub, data_ub, indices_out_final_ub, 1,
                                  last_part_cols_padding)
        else:
            multiplier_scalar.set_as(offset_ub[part_cnt - 1])
            tik_instance.vector_dup(indices_block_num, indices_ub, multiplier_scalar,
                                    indices_ub.buffer_size // tbe_platform.VECTOR_INST_BLOCK_WIDTH, 1, 8)

            self.emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Score.value, cnt=last_part_cols_padding)
            self.emit_vconcat(tik_instance, region_ub, data_ub, mode=Mode.Y2.value, cnt=last_part_cols_padding)
            self.emit_vconcat(tik_instance, region_ub, indices_ub, mode=Mode.X1.value, cnt=cols_per_part)
            self.emit_vconcat(tik_instance, region_ub, offset_ub, mode=Mode.Y1.value, cnt=last_part_cols_padding)

            self.sort_region(tik_instance, region_sorted_ub, region_ub, 1, last_part_cols_padding)
        result_ub = region_sorted_ub
        self.merge_two_sorted_region(tik_instance,
                                     dst=region_k2_ub,
                                     src_region_k=region_k_ub,
                                     src_region_sorted=result_ub,
                                     len_region_k=cols_per_part * 4,
                                     len_region_sorted=last_part_cols_padding)

        if SortParam.SUPPORT_VBITSORT32:
            tik_instance.vreduce(k_padding * SortParam.SORT_REGION_BYTE // _get_dtype_byte(data_ub.dtype),
                                 region_k_ub,
                                 region_k2_ub,
                                 src1_pattern=3,
                                 repeat_times=1,
                                 src0_blk_stride=1,
                                 src0_rep_stride=8,
                                 src1_rep_stride=0,
                                 stride_unit=0,
                                 rsvd_scalar=None,
                                 mask_mode="counter")
            tik_instance.vreduce(k_padding * SortParam.SORT_REGION_BYTE // _get_dtype_byte(indices_out_final_ub.dtype),
                                 indices_out_final_ub,
                                 region_k2_ub.reinterpret_cast_to(indices_out_final_ub.dtype),
                                 src1_pattern=2,
                                 repeat_times=1,
                                 src0_blk_stride=1,
                                 src0_rep_stride=8,
                                 src1_rep_stride=0,
                                 stride_unit=0,
                                 rsvd_scalar=None,
                                 mask_mode="counter")
        else:
            self.emit_vextract(tik_instance, region_k_ub, region_k2_ub, Mode.Y2.value, k_padding)
            self.emit_vextract(tik_instance,
                               region_k_ub,
                               region_k2_ub,
                               Mode.Y1.value,
                               k_padding,
                               dst_offset=k_padding * 2)
            self.emit_vextract(tik_instance, region_k_ub, region_k2_ub, Mode.X1.value, k_padding, dst_offset=k_padding)

            if not largest:
                self.emit_vmuls(tik_instance, region_k_ub, region_k_ub, cnt=k_padding)

            # get multiplier
            self.conv_fp162s32(tik_instance, offset_int32_ub, 0, region_k_ub, k_padding, k_padding)
            tik_instance.vector_dup(8, indices_tail_block_ub, cols_per_part, 1, 0, 0)
            # multiplier * cols_per_part
            repeat_times = 4 * cols_per_part // vadds_len
            tik_instance.vmul(vadds_len, indices_out_final_ub, offset_int32_ub, indices_tail_block_ub, repeat_times, 1,
                              1, 0, 8, 8, 0)

            # multiplier * cols_per_part + offset
            self.conv_fp162s32(tik_instance, offset_int32_ub, 0, region_k_ub, k_padding * 2, k_padding)
            tik_instance.vadd(vadds_len, indices_out_final_ub, offset_int32_ub, indices_out_final_ub, repeat_times, 1,
                              1, 1, 8, 8, 8)

        self.copy_ubuf_to_gm(tik_instance,
                             data_ub.dtype,
                             data_gm_out,
                             region_k_ub,
                             num_rows=1,
                             cols_padding=cols_padding,
                             k=k,
                             tail_block_ub=data_tail_block_ub,
                             gm_offset=row_start_in_core * k + core_rows_start * k,
                             multi_core=multi_core)
        self.copy_ubuf_to_gm(tik_instance,
                             'int32',
                             indices_gm_out,
                             indices_out_final_ub,
                             1,
                             cols_padding,
                             k,
                             tail_block_ub=indices_tail_block_ub,
                             gm_offset=row_start_in_core * k + core_rows_start * k,
                             multi_core=multi_core)

    # 'pylint: disable=too-many-arguments,no-self-use
    def emit_vextract(self, tik_instance, dst, src, mode, cnt, dst_offset=0, src_offset=0):
        """
        emit_vextract
        """
        _exec_front_last(
            tik_instance, cnt, SortParam.SORT_ONCE_NUM * 255,
            lambda offset, part_len: tik_instance.vextract(dst[dst_offset + offset], src[
                src_offset + offset * self.num_per_ele], _ceil_div(part_len, SortParam.SORT_ONCE_NUM), mode))

    def _extract(self, tik_instance, sorted_ub, rows):
        self.emit_vextract(tik_instance, self.data_ub, sorted_ub, Mode.Y2.value, rows * self.cols_padding)
        with tik_instance.for_range(0, rows, name='i0') as i:
            self.emit_vextract(tik_instance,
                               self.indices_out_fp16_ub,
                               sorted_ub,
                               Mode.X1.value,
                               self.cols_padding,
                               dst_offset=i * self.cols_padding,
                               src_offset=i * self.cols_padding * self.num_per_ele)
            self.emit_vextract(tik_instance,
                               self.offset_fp16_ub,
                               sorted_ub,
                               Mode.Y1.value,
                               self.cols_padding,
                               dst_offset=i * self.cols_padding,
                               src_offset=i * self.cols_padding * self.num_per_ele)

    # 'pylint: disable=too-many-arguments
    def topk_rows(self, tik_instance, row_start_in_core, rows, cols, k, core_rows_start, multi_core, largest):
        """
        topk_rows do topk action muilti rows
        """
        self.copy_gm_to_ubuf_func(tik_instance,
                                  self.data_ub,
                                  self.data_gm,
                                  num_rows=rows,
                                  cols=cols,
                                  col_start=0,
                                  gm_offset=row_start_in_core * cols + core_rows_start * cols,
                                  largest=largest)

        self.copy_gm_to_ubuf(tik_instance,
                             self.indices_ub,
                             self.indices_gm,
                             num_rows=1,
                             cols=cols,
                             col_start=0,
                             gm_offset=0)

        if SortParam.SUPPORT_VBITSORT32:
            indices_per_part = self.func_obj_tiling.indices_per_part
            with tik_instance.if_scope(indices_per_part < cols):
                indices_num = indices_per_part
            with tik_instance.else_scope():
                indices_num = cols
            self.conv_fp162s32(tik_instance, self.indices_out_final_ub, 0,
                               self.indices_ub.reinterpret_cast_to(self.indices_gm.dtype), 0, indices_num)
            vadds_len = tbe_platform.VECTOR_INST_BLOCK_WIDTH // _get_dtype_byte(self.indices_out_final_ub.dtype)
            with tik_instance.for_range(1, _ceil_div(self.cols_padding, indices_per_part), name='i0') as i:
                tik_instance.vadds(vadds_len, self.indices_out_final_ub[indices_per_part * i],
                                   self.indices_out_final_ub, indices_per_part * i, indices_per_part // vadds_len, 1, 1,
                                   8, 8)

            with tik_instance.for_range(1, rows, name='i0') as i:
                indices_out_final_ub_c0 = tbe_platform.BLOCK_REDUCE_INT8 // _get_dtype_byte(
                    self.indices_out_final_ub.dtype)

                _exec_front_last(
                    tik_instance, self.cols_padding, 255 * indices_out_final_ub_c0,
                    lambda offset, part_len: tik_instance.data_move(
                        self.indices_out_final_ub[i * self.cols_padding + offset], self.indices_out_final_ub[offset], 0,
                        1, part_len // indices_out_final_ub_c0, 0, 0))
            self.sort_region_a100(tik_instance, self.region_sorted_ub.reinterpret_cast_to(self.data_ub.dtype),
                                  self.data_ub, self.indices_out_final_ub, rows, self.cols_padding)
            tik_instance.vreduce(rows * self.cols_padding * \
                                        SortParam.SORT_REGION_BYTE // _get_dtype_byte(self.data_ub.dtype),
                                 self.data_ub,
                                 self.region_sorted_ub.reinterpret_cast_to(self.data_ub.dtype),
                                 src1_pattern=3,
                                 repeat_times=1,
                                 src0_blk_stride=1,
                                 src0_rep_stride=8,
                                 src1_rep_stride=0,
                                 stride_unit=0,
                                 rsvd_scalar=None,
                                 mask_mode="counter")
            tik_instance.vreduce(rows * self.cols_padding * SortParam.SORT_REGION_BYTE //
                                 _get_dtype_byte(self.indices_out_final_ub.dtype),
                                 self.indices_out_final_ub,
                                 self.region_sorted_ub.reinterpret_cast_to(self.indices_out_final_ub.dtype),
                                 src1_pattern=2,
                                 repeat_times=1,
                                 src0_blk_stride=1,
                                 src0_rep_stride=8,
                                 src1_rep_stride=0,
                                 stride_unit=0,
                                 rsvd_scalar=None,
                                 mask_mode="counter")
        else:
            self.copy_gm_to_ubuf(tik_instance,
                                 self.offset_ub,
                                 self.indices_gm,
                                 num_rows=1,
                                 cols=cols,
                                 col_start=4096,
                                 gm_offset=0)

            self.emit_vconcat(tik_instance,
                              self.region_ub,
                              self.data_ub,
                              mode=Mode.Score.value,
                              cnt=rows * self.cols_padding)
            self.emit_vconcat(tik_instance,
                              self.region_ub,
                              self.data_ub,
                              mode=Mode.Y2.value,
                              cnt=rows * self.cols_padding)
            with tik_instance.for_range(0, rows, name='i0') as i:
                self.emit_vconcat(tik_instance,
                                  self.region_ub,
                                  self.indices_ub,
                                  mode=Mode.X1.value,
                                  cnt=self.cols_padding,
                                  dst_offset=i * self.cols_padding * self.num_per_ele,
                                  src_offset=0)

                self.emit_vconcat(tik_instance,
                                  self.region_ub,
                                  self.offset_ub,
                                  mode=Mode.Y1.value,
                                  cnt=self.cols_padding,
                                  dst_offset=i * self.cols_padding * self.num_per_ele,
                                  src_offset=0)

            self.sort_region(tik_instance, self.region_sorted_ub, self.region_ub, rows, self.cols_padding)
            with tik_instance.if_scope(self.loop_times % 2 == 0):
                self._extract(tik_instance, self.region_sorted_ub, rows)
            with tik_instance.else_scope():
                self._extract(tik_instance, self.region_ub, rows)

            if not largest:
                self.emit_vmuls(tik_instance, self.data_ub, self.data_ub, cnt=rows * self.cols_padding)
            with tik_instance.for_range(0, rows, name='i0') as i:
                self.conv_fp162s32(tik_instance, self.indices_out_int32_ub, i * self.cols_padding,
                                   self.indices_out_fp16_ub, i * self.cols_padding, self.cols_padding)
                self.conv_fp162s32(tik_instance, self.offset_int32_ub, i * self.cols_padding, self.offset_fp16_ub,
                                   i * self.cols_padding, self.cols_padding)
            self._add(tik_instance, self.indices_out_final_ub, self.indices_out_int32_ub, self.offset_int32_ub, rows,
                      self.cols_padding)

        self.copy_ubuf_to_gm(tik_instance,
                             self.data_ub.dtype,
                             self.data_gm_out,
                             self.data_ub,
                             rows,
                             self.cols_padding,
                             k,
                             tail_block_ub=self.data_tail_block_ub,
                             gm_offset=row_start_in_core * k + core_rows_start * k,
                             multi_core=multi_core)
        self.copy_ubuf_to_gm(tik_instance,
                             'int32',
                             self.indices_gm_out,
                             self.indices_out_final_ub,
                             rows,
                             self.cols_padding,
                             k,
                             tail_block_ub=self.indices_tail_block_ub,
                             gm_offset=row_start_in_core * k + core_rows_start * k,
                             multi_core=multi_core)

    # 'pylint: disable=no-self-use,too-many-arguments
    def merge_two_sorted_region(self, tik_instance, dst, src_region_k, src_region_sorted, len_region_k,
                                len_region_sorted):
        """
        merge_two_sorted_region
        """
        if SortParam.SUPPORT_VBITSORT32:
            if len_region_k < 4 * self.func_obj_tiling.cols_per_part:
                merge_n0 = len_region_k
                merge_n1_merge_two_reg = tik_instance.Scalar(init_value=len_region_sorted,
                                                             name="merge_n1_merge_two_reg")
                src_list = [src_region_k[0], src_region_sorted[0]]
                tik_instance.vmrgsort(dst, src_list, (merge_n0, merge_n1_merge_two_reg), False, 1)
            elif len_region_k >= 4 * self.func_obj_tiling.cols_per_part:
                merge_n0 = 2 * self.func_obj_tiling.cols_per_part
                merge_n1 = 2 * self.func_obj_tiling.cols_per_part
                merge_n2_merge_two_reg = tik_instance.Scalar(init_value=len_region_sorted,
                                                             name="merge_n2_merge_two_reg")
                src_list = [
                    src_region_k[0], src_region_k[2 * self.func_obj_tiling.cols_per_part * self.num_per_ele],
                    src_region_sorted[0]
                ]
                tik_instance.vmrgsort(dst, src_list, (merge_n0, merge_n1, merge_n2_merge_two_reg), False, 1)
        else:
            if len_region_k < 4 * self.func_obj_tiling.cols_per_part:
                merge_n0 = len_region_k
                merge_n1_merge_two_reg = tik_instance.Scalar(init_value=len_region_sorted,
                                                             name="merge_n1_merge_two_reg")
                src_list = [src_region_k[0], src_region_sorted[0], src_region_k[16], src_region_k[16]]
                tik_instance.vmrgsort4(dst, src_list, (merge_n0, merge_n1_merge_two_reg, 16, 16), False, 3, 1)
            elif len_region_k >= 4 * self.func_obj_tiling.cols_per_part:
                merge_n0 = 2 * self.func_obj_tiling.cols_per_part
                merge_n1 = 2 * self.func_obj_tiling.cols_per_part
                merge_n2_merge_two_reg = tik_instance.Scalar(init_value=len_region_sorted,
                                                             name="merge_n2_merge_two_reg")
                src_list = [
                    src_region_k[0], src_region_k[2 * self.func_obj_tiling.cols_per_part * self.num_per_ele],
                    src_region_sorted[0], src_region_k[16]
                ]
                tik_instance.vmrgsort4(dst, src_list, (merge_n0, merge_n1, merge_n2_merge_two_reg, 16), False, 7, 1)

    # 'pylint: disable=no-self-use,too-many-arguments
    @staticmethod
    def copy_region(tik_instance, dst, src, num, dst_offset=0):
        """
        copy_region
        """
        burstlen = _ceil_div(num * SortParam.SORT_REGION_BYTE, tbe_platform.BLOCK_REDUCE_INT8)
        tik_instance.data_move(dst[dst_offset], src, 0, 1, burstlen, 0, 0)

    # 'pylint: disable=no-self-use,too-many-arguments
    @staticmethod
    def _add(tik_instance, dst, src1, src2, rows, cols_padding):
        # process 256B data per repeat for vsub
        vadd_len = 64
        repeat = (rows * cols_padding) // vadd_len
        remain = (rows * cols_padding) - repeat * vadd_len
        isinstance_if(tik_instance, repeat > 0,
                      lambda: tik_instance.vadd(Constant.FULL_MASK_INT32, dst, src1, src2, repeat, 1, 1, 1, 8, 8, 8))
        isinstance_if(
            tik_instance, remain > 0, lambda: tik_instance.vadd(remain, dst[repeat * vadd_len], src1[repeat * vadd_len],
                                                                src2[repeat * vadd_len], 1, 1, 1, 1, 8, 8, 8))

    # 'pylint: disable=too-many-arguments,no-self-use
    @staticmethod
    def conv_fp162s32(tik_instance, s32ub, s32ub_offset, fp16ub, fp16ub_offset, num):
        """
        fp16 to int32
        """
        repeat = num // 64
        remain = num - repeat * 64
        isinstance_if(
            tik_instance, repeat > 0,
            lambda: tik_instance.vconv(64, "round", s32ub[s32ub_offset], fp16ub[fp16ub_offset], repeat, 1, 1, 8, 4))
        isinstance_if(
            tik_instance, remain > 0, lambda: tik_instance.vconv(remain, "round", s32ub[s32ub_offset + repeat * 64],
                                                                 fp16ub[fp16ub_offset + repeat * 64], 1, 1, 1, 8, 4))

    def sort_region(self, tik_instance, dst, src, rows, cols):
        """
        sort_region
        """
        _exec_front_last(
            tik_instance, rows * cols, SortParam.SORT_ONCE_NUM * 255, lambda offset, part_len: tik_instance.vrpsort16(
                dst[offset * self.num_per_ele], src[offset * self.num_per_ele],
                _ceil_div(part_len, SortParam.SORT_ONCE_NUM)))

        with tik_instance.if_scope(cols > SortParam.SORT_ONCE_NUM):
            self.merge_region(tik_instance, dst=src, src=dst, rows=rows, cols=cols)

    def sort_region_a100(self, tik_instance, dst, score, index, rows, cols):
        """
        sort_region
        """
        _exec_front_last(
            tik_instance, rows * cols, SortParam.SORT_ONCE_NUM * 255, lambda offset, part_len: tik_instance.vsort32(
                dst[offset * self.num_per_ele], score[offset],
                index.reinterpret_cast_to("uint32")[offset], _ceil_div(part_len, SortParam.SORT_ONCE_NUM)))

        with tik_instance.if_scope(cols > SortParam.SORT_ONCE_NUM):
            self.merge_region(tik_instance,
                              dst=self.region_ub.reinterpret_cast_to(score.dtype),
                              src=dst,
                              rows=rows,
                              cols=cols)

    def merge_region(self, tik_instance, dst, src, rows, cols):
        """
        merge_region
        """
        cols_padding = _ceil_fill(cols, SortParam.SORT_ONCE_NUM)
        with tik_instance.for_range(0, rows, name='merge_i0') as i:
            self._merge_loop(tik_instance,
                             src,
                             dst,
                             cols,
                             _ceil_div(cols, SortParam.SORT_ONCE_NUM),
                             region_offset=i * cols_padding * self.num_per_ele)

    def _merge_loop(self, tik_instance, src_ub, dst_ub, last_dim, total_region_list, region_offset=0):
        region_list_reg = tik_instance.Scalar(init_value=total_region_list, dtype="int32", name="region_list_reg")
        with tik_instance.for_range(0, self.loop_times) as i:
            with tik_instance.if_scope(i % 2 == 0):
                self._merge(tik_instance, src_ub, dst_ub, last_dim, region_list_reg, i, region_offset)

            with tik_instance.else_scope():
                self._merge(tik_instance, dst_ub, src_ub, last_dim, region_list_reg, i, region_offset)

    # 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,no-self-use
    def _merge(self, tik_instance, src_ub, dst_ub, last_dim, total_region_list, level, region_offset=0):
        """
        _merge_recur
        merge multi sorted region proposal list to one sorted region proposal list
        """
        max_merge_tensors = 4
        mid_var = total_region_list // max_merge_tensors
        loops = tik_instance.Scalar(init_value=(mid_var), dtype="int32", name="loops")
        remain = tik_instance.Scalar(init_value=(total_region_list - loops * max_merge_tensors),
                                     dtype="int32",
                                     name="remain")

        level_reg = tik_instance.Scalar(init_value=level, dtype="int32", name="level_reg")

        merge_n0_reg = tik_instance.Scalar(init_value=1, name="merge_n0_reg")
        merge_n0_reg.set_as((merge_n0_reg << (2 * level_reg)) * SortParam.SORT_ONCE_NUM)
        merge_repeat = tik_instance.Scalar(init_value=loops, name="merge_repeat")

        need_tail_process = tik_instance.Scalar(init_value=0, name="need_tail_process")
        merge_last = tik_instance.Scalar(init_value=0, name="merge_last")

        with tik_instance.if_scope(tik.all(loops > 0, remain == 0)):
            with tik_instance.if_scope(merge_n0_reg * max_merge_tensors * loops > last_dim):
                merge_repeat.set_as(loops - 1)
                need_tail_process.set_as(1)

        with tik_instance.if_scope(merge_repeat > 0):
            src_list = [
                src_ub[region_offset], src_ub[region_offset + merge_n0_reg * self.num_per_ele],
                src_ub[region_offset + merge_n0_reg * self.num_per_ele * 2],
                src_ub[region_offset + merge_n0_reg * self.num_per_ele * 3]
            ]
            if SortParam.SUPPORT_VBITSORT32:
                tik_instance.vmrgsort(dst_ub[region_offset], src_list,
                                      (merge_n0_reg, merge_n0_reg, merge_n0_reg, merge_n0_reg), False, merge_repeat)
            else:
                tik_instance.vmrgsort4(dst_ub[region_offset], src_list,
                                       (merge_n0_reg, merge_n0_reg, merge_n0_reg, merge_n0_reg), False, 15,
                                       merge_repeat)

        offset_reg = tik_instance.Scalar(dtype="int32",
                                         init_value=merge_repeat * merge_n0_reg * max_merge_tensors,
                                         name="offset_reg")
        tail_offset = offset_reg * self.num_per_ele
        with tik_instance.if_scope(need_tail_process == 1):
            merge_last.set_as(last_dim - (offset_reg + merge_n0_reg * 3))
            src_list = [
                src_ub[region_offset + tail_offset],
                src_ub[region_offset + tail_offset + merge_n0_reg * self.num_per_ele],
                src_ub[region_offset + tail_offset + merge_n0_reg * self.num_per_ele * 2],
                src_ub[region_offset + tail_offset + merge_n0_reg * self.num_per_ele * 3]
            ]
            if SortParam.SUPPORT_VBITSORT32:
                tik_instance.vmrgsort(dst_ub[region_offset + tail_offset], src_list,
                                      (merge_n0_reg, merge_n0_reg, merge_n0_reg, merge_last), False, 1)
            else:
                tik_instance.vmrgsort4(dst_ub[region_offset + tail_offset], src_list,
                                       (merge_n0_reg, merge_n0_reg, merge_n0_reg, merge_last), False, 15, 1)

        with tik_instance.if_scope(remain == 3):
            merge_last.set_as(last_dim - (offset_reg + merge_n0_reg * 2))

            src_list = [
                src_ub[region_offset + tail_offset],
                src_ub[region_offset + tail_offset + merge_n0_reg * self.num_per_ele],
                src_ub[region_offset + tail_offset + merge_n0_reg * self.num_per_ele * 2]
            ]
            if SortParam.SUPPORT_VBITSORT32:
                tik_instance.vmrgsort(dst_ub[region_offset + tail_offset], src_list,
                                      (merge_n0_reg, merge_n0_reg, merge_last), False, 1)
            else:
                tik_instance.vmrgsort4(dst_ub[region_offset + tail_offset], src_list + [src_ub[0]],
                                       (merge_n0_reg, merge_n0_reg, merge_last, 16), False, 7, 1)
        with tik_instance.if_scope(remain == 2):
            merge_last.set_as(last_dim - (offset_reg + merge_n0_reg))
            src_list = [
                src_ub[region_offset + tail_offset],
                src_ub[region_offset + tail_offset + merge_n0_reg * self.num_per_ele]
            ]
            if SortParam.SUPPORT_VBITSORT32:
                tik_instance.vmrgsort(dst_ub[region_offset + tail_offset], src_list, (merge_n0_reg, merge_last), False,
                                      1)
            else:
                tik_instance.vmrgsort4(dst_ub[region_offset + tail_offset], src_list + [src_ub[0], src_ub[0]],
                                       (merge_n0_reg, merge_last, 16, 16), False, 3, 1)
        with tik_instance.if_scope(remain == 1):
            merge_last.set_as(last_dim - offset_reg)
            num_blocks_write_reg = tik_instance.Scalar(init_value=_ceil_div(merge_last * SortParam.SORT_REGION_BYTE,
                                                                            tbe_platform.BLOCK_REDUCE_INT8),
                                                       name="num_blocks_write_reg")
            tik_instance.data_move(dst_ub[region_offset + tail_offset], src_ub[region_offset + tail_offset], 0, 1,
                                   num_blocks_write_reg, 0, 0)
        total_region_list.set_as(_ceil_div(total_region_list, max_merge_tensors))

    # 'pylint: disable=too-many-arguments,no-self-use
    def emit_vconcat(self, tik_instance, dst, src, mode, cnt, dst_offset=0, src_offset=0):
        """
        emit_vconcat
        """
        _exec_front_last(
            tik_instance, cnt, SortParam.SORT_ONCE_NUM * 255,
            lambda offset, part_len: tik_instance.vconcat(dst[dst_offset + offset * self.num_per_ele], src[
                src_offset + offset], _ceil_div(part_len, SortParam.SORT_ONCE_NUM), mode))

    def copy_gm_to_ubuf_func(self, tik_instance, dst, src, num_rows, cols, col_start, gm_offset, largest):
        """
        copy_gm_to_ubuf copy data from gm to ubuf
        """
        cols_padding = _ceil_fill(cols, SortParam.SORT_ONCE_NUM)
        burstlen = _ceil_div(cols * _get_dtype_byte(dst.dtype), tbe_platform.BLOCK_REDUCE_INT8)
        # 'pylint: disable=invalid-name
        cols_32b_align = cols % SortParam.SORT_ONCE_NUM
        reg_min_number = tik_instance.Scalar(dtype=dst.dtype,
                                             init_value=_get_dtype_min_val(dst.dtype),
                                             name='reg_min_number')

        with tik_instance.if_scope(cols_32b_align == 0):
            tik_instance.data_move(dst[0], src[col_start + gm_offset], 0, 1, burstlen * num_rows, 0, 0)
            if not largest:
                self.emit_vmuls(tik_instance, dst, dst, cnt=num_rows * cols_padding)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, num_rows, name='gm2ub_i0') as i:
                self.emit_copy_gm_to_ubuf(tik_instance,
                                          dst,
                                          src,
                                          1,
                                          burstlen,
                                          0,
                                          0,
                                          dst_offset=cols_padding * i,
                                          src_offset=cols * i + col_start + gm_offset)
            if not largest:
                self.emit_vmuls(tik_instance, dst, dst, cnt=num_rows * cols_padding)
            with tik_instance.for_range(0, num_rows, name='gm2ub_i0') as i:
                with tik_instance.for_range(0, cols_padding - cols) as j:
                    dst[cols_padding * i + cols + j].set_as(reg_min_number)

    # 'pylint: disable=no-self-use
    @staticmethod
    def emit_vmuls(tik_instance, dst, src, cnt):
        """
        emit_vmuls
        """
        mid_var = cnt // 128
        repeat_255_scalar = tik_instance.Scalar(init_value=(mid_var))
        repeat_remain_scalar = tik_instance.Scalar(init_value=(cnt - repeat_255_scalar * 128))

        times_scalar = tik_instance.Scalar(init_value=((repeat_255_scalar + 254) // 255))

        with tik_instance.if_scope(repeat_255_scalar > 0):
            with tik_instance.for_range(0, times_scalar, name='vmuls_i0') as i:
                src0_scalar = tik_instance.Scalar(dtype="int64",
                                                  name='src0_scalar',
                                                  init_value=repeat_255_scalar - i * 255)
                src1_scalar = tik_instance.Scalar(dtype="int64", name='src1_scalar', init_value=255)
                times_len = tik_instance.Scalar(dtype="int64", name='dst_scalar')
                tik_instance.scalar_min(times_len, src0_scalar, src1_scalar)
                tik_instance.vmuls(Constant.FULL_MASK_FP16, dst[i * 128 * 255], src[i * 128 * 255], -1, times_len, 1, 1,
                                   8, 8)

        with tik_instance.if_scope(repeat_remain_scalar > 0):
            tik_instance.vmuls(repeat_remain_scalar, dst[repeat_255_scalar * 128], src[repeat_255_scalar * 128], -1, 1,
                               1, 1, 8, 8)

    # 'pylint: disable=invalid-name
    def copy_ubuf_to_gm(self,
                        tik_instance,
                        dtype,
                        dst,
                        src,
                        num_rows,
                        cols_padding,
                        k,
                        tail_block_ub,
                        gm_offset=0,
                        multi_core=0):
        """
        copy_ubuf_to_gm
        """
        burstlen = _ceil_div(k * _get_dtype_byte(dtype), tbe_platform.BLOCK_REDUCE_INT8)
        blocklen = tbe_platform.BLOCK_REDUCE_INT8 // _get_dtype_byte(dtype)
        k_32b_align = self.k % blocklen
        cols_32b_align = self.cols % blocklen
        dst_offset = tik_instance.Scalar(dtype="int32", init_value=gm_offset)
        src_offset = tik_instance.Scalar(dtype="int32", init_value=0)
        with tik_instance.if_scope(tik.all(cols_32b_align == 0, k_32b_align == 0)):
            src_stride = cols_padding // blocklen - burstlen
            tik_instance.data_move(dst[dst_offset], src[0], 0, num_rows, burstlen, src_stride, 0)

        with tik_instance.else_scope():
            with tik_instance.for_range(0, num_rows - 1, name='ub2gmi0') as i:
                self.emit_copy_ubuf_to_gm(tik_instance, dst, src, 1, burstlen, 0, 0,
                                          dst_offset=dst_offset, src_offset=src_offset)
                dst_offset.set_as(dst_offset + k)
                src_offset.set_as(src_offset + cols_padding)

            with tik_instance.if_scope(tik.all(multi_core == 1, k > blocklen)):
                self.emit_copy_ubuf_to_gm(tik_instance, dst, src, 1, burstlen - 1, 0, 0,
                                          dst_offset=k * (num_rows - 1) + gm_offset,
                                          src_offset=cols_padding * (num_rows - 1))
                for i in range(blocklen):
                    tail_block_ub[i].set_as(src[cols_padding * (num_rows - 1) + k - blocklen + i])

                self.emit_copy_ubuf_to_gm(tik_instance, dst, tail_block_ub, 1, 1, 0, 0,
                                          dst_offset=k * (num_rows - 1) + gm_offset + k - blocklen,
                                          src_offset=0)
            with tik_instance.else_scope():
                self.emit_copy_ubuf_to_gm(tik_instance, dst, src, 1, burstlen, 0, 0,
                                          dst_offset=k * (num_rows - 1) + gm_offset,
                                          src_offset=cols_padding * (num_rows - 1))

    def copy_gm_to_ubuf(self, tik_instance, dst, src, num_rows, cols, col_start, gm_offset):
        """
        copy_gm_to_ubuf copy data from gm to ubuf
        """
        cols_padding = _ceil_fill(cols, SortParam.SORT_ONCE_NUM)
        burstlen = _ceil_div(cols * _get_dtype_byte(dst.dtype), tbe_platform.BLOCK_REDUCE_INT8)
        # 'pylint: disable=invalid-name
        cols_32b_align = cols % (tbe_platform.BLOCK_REDUCE_INT8 // _get_dtype_byte(dst.dtype))
        with tik_instance.if_scope(cols_32b_align == 0):
            tik_instance.data_move(dst[0], src[col_start + gm_offset], 0, num_rows, burstlen, 0, 0)

        with tik_instance.else_scope():
            with tik_instance.for_range(0, num_rows, name='gm2ub_i0') as i:
                self.emit_copy_gm_to_ubuf(tik_instance,
                                          dst,
                                          src,
                                          1,
                                          burstlen,
                                          0,
                                          0,
                                          dst_offset=cols_padding * i,
                                          src_offset=cols * i + col_start + gm_offset)

    # 'pylint: disable=too-many-arguments,no-self-use
    @staticmethod
    def emit_copy_gm_to_ubuf(tik_instance,
                             dst,
                             src,
                             nburst,
                             burstlen,
                             srcstride,
                             dststride,
                             dst_offset=0,
                             src_offset=0):
        """
        emit_copy_gm_to_ubuf
        """
        tik_instance.data_move(dst[dst_offset], src[src_offset], 0, nburst, burstlen, srcstride, dststride)

    # 'pylint: disable=too-many-arguments,no-self-use
    @staticmethod
    def emit_copy_ubuf_to_gm(tik_instance,
                             dst,
                             src,
                             nburst,
                             burstlen,
                             srcstride,
                             dststride,
                             dst_offset=0,
                             src_offset=0):
        """
        emit_copy_ubuf_to_gm
        """
        tik_instance.data_move(dst[dst_offset], src[src_offset], 0, nburst, burstlen, srcstride, dststride)


def add_compile_info_and_build(tik_instance, obj_gm, obj_tiling, ins, outs, kernel_name, mode, k):
    build_config = {"out_of_bound_sync_check": True}
    if mode == "dynamic":
        tbe_context.get_context().add_compile_info("is_tik", True)
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": obj_tiling.core_num,
                "k_num": k,
                "ub_size": obj_tiling.ub_size,
                "batch_cols_padding": obj_tiling.batch_cols_padding,
                "max_k": 4 * obj_tiling.cols_per_part
            })
        tik_instance.BuildCCE(kernel_name=kernel_name,
                              inputs=ins,
                              outputs=outs,
                              flowtable=(obj_gm.tiling_gm,),
                              enable_l2=True,
                              config=build_config)
    else:
        tik_instance.BuildCCE(kernel_name=kernel_name, inputs=ins, outputs=outs, enable_l2=True, config=build_config)


# 'pylint: disable=too-many-arguments,too-many-locals,unused-argument
def top_k_compute(tik_instance, obj_gm, obj_tiling, obj_ub, profile, dtype, indices_dtype, largest, k, kernel_name,
                  mode):
    """
    compute of top_k

    Parameters
    ----------
    None

    Returns
    -------
    compile info
    """
    x_shape = (Constant.MAX_SHAPE_SIZE,)
    indices_shape = (Constant.INDICES_NUM,)
    res_shape = (Constant.MAX_SHAPE_SIZE,)
    indices_out_shape = (Constant.MAX_SHAPE_SIZE,)
    data_input = tik_instance.Tensor(dtype.lower(), x_shape, name='data_a', scope=tik.scope_gm)
    indices = tik_instance.Tensor(indices_dtype.lower(), indices_shape, name='indices', scope=tik.scope_gm)
    res = tik_instance.Tensor(dtype.lower(), res_shape, name='res', scope=tik.scope_gm)
    indices_out = tik_instance.Tensor("int32", indices_out_shape, name='indices_out', scope=tik.scope_gm)
    outs = [res, indices_out]
    cols = obj_tiling.get_cols_num()
    block_dim = obj_tiling.get_core_num()

    if k > 0:
        k_scalar = tik_instance.Scalar(dtype="int32", name="k_scalar", init_value=k)
        ins = [data_input, indices]
    else:
        k_input = tik_instance.Tensor('int32', (1,), name='k_gm', scope=tik.scope_gm)
        k_ub = tik_instance.Tensor('int32', (1,), name='k_ub', scope=tik.scope_ubuf)
        ins = [data_input, k_input, indices]
        k_scalar = tik_instance.Scalar(dtype="int32", name="k_scalar")
        tik_instance.data_move(k_ub, k_input, 0, 1, 1, 0, 0)
        k_scalar.set_as(k_ub[0])
        with tik_instance.if_scope(k_scalar < SortParam.SORT_ONCE_NUM):
            block_dim.set_as(1)
            obj_tiling.num_rows_cores_scalar.set_as(obj_tiling.num_rows_scalar)

    with tik_instance.for_range(0, obj_tiling.core_num, block_num=obj_tiling.core_num) as block_idx:
        with tik_instance.if_scope(block_idx < block_dim):
            with tik_instance.if_scope(cols > obj_tiling.mode_threshold):
                if (isinstance(cols, int) and cols > obj_tiling.mode_threshold) or not isinstance(cols, int):
                    if SortParam.SUPPORT_VBITSORT32:
                        set_tensor_more_4096_a100(tik_instance, obj_tiling, obj_gm, obj_ub, ins, outs, dtype)
                    else:
                        set_tensor_more_4096(tik_instance, obj_tiling, obj_gm, obj_ub, ins, outs)
                    obj_func = GlobalVarFunction(obj_gm, obj_tiling, obj_ub)
                    obj_func.kernel_ir(tik_instance, largest, True, block_idx, block_dim, k_scalar)
            with tik_instance.else_scope():
                if (isinstance(cols, int) and cols <= obj_tiling.mode_threshold) or not isinstance(cols, int):
                    if SortParam.SUPPORT_VBITSORT32:
                        set_tensor_less_4096_a100(tik_instance, obj_tiling, obj_gm, obj_ub, ins, outs, dtype)
                    else:
                        set_tensor_less_4096(tik_instance, obj_tiling, obj_gm, obj_ub, ins, outs)
                    obj_func = GlobalVarFunction(obj_gm, obj_tiling, obj_ub)
                    obj_func.kernel_ir(tik_instance, largest, False, block_idx, block_dim, k_scalar)
    add_compile_info_and_build(tik_instance, obj_gm, obj_tiling, ins, outs, kernel_name, mode, k)
    

# 'pylint: disable=unused-argument,redefined-builtin
def check_supported(input_tensor,
                    indices_tensor,
                    out_tensor,
                    out_indices_tensor,
                    k,
                    sorted=True,
                    dim=-1,
                    largest=True,
                    kernel_name='top_k'):
    """
    check whether ai_core is supported
    max last dim should exist and max last dim of input_tensor should <= 1024 * 2048 and k
    should <= 4096
    """
    unknown_shape = input_tensor.get('shape')
    unknwon_dim_status = unknown_shape[0]
    if unknwon_dim_status == -2:
        return True, ""

    shape = input_tensor.get("ori_shape")
    sorted_axis = dim
    if sorted_axis < 0:
        sorted_axis = sorted_axis + len(shape)

    shape_range = input_tensor.get("range")
    if shape_range:
        max_last_dim = shape_range[sorted_axis][-1]
        if not max_last_dim:
            return "Unknown"
    else:
        max_last_dim = shape[sorted_axis]

    input_size = functools.reduce(lambda x, y: x * y, shape)
    soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    if input_size > 32768 and k > 0 and k < 8 and soc_version != "Ascend910B":
        reason = "Input_size is too big(> 32768), and k is in (0-8), input_size:%s, k:%s" \
                 % (input_size, k)
        return False, reason

    if soc_version == "Ascend310":
        if k > 4096:
            reason = "k is too big(> 4096), k:%s" % k
            return False, reason
        if max_last_dim > 1024 * 2048:
            reason = "input_tensor is too big(> 1024 * 2048), max_last_dim:%s" % max_last_dim
            return False, reason
    else:
        if max_last_dim == 1:
            reason = "data of sort axis is 1"
            return False, reason
    return True, ""


def topk_dsl(input_tensor, indices_tensor, out_tensor, out_indices_tensor, k, sorted, dim, largest, kernel_name):
    op_mode = "topk"
    if is_unknown_rank_input(input_tensor):
        op_mode = "topkv2"
        operation.get_context().add("_is_unknown_shape", True)
    ins = classify([input_tensor, dim, k], OpPatternMode.SORT, {"op_mode": op_mode})
    schedules, tensors = [], []
    if op_mode == "topkv2":
        for (_x, _k) in ins:
            with tbe.compute():
                x_shape, k_var = shape_util.variable_shape([_x, _k], "sort")
                x_input = tvm.placeholder(x_shape, name="data_input", dtype=input_tensor["dtype"])
                indices_input = tvm.placeholder(indices_tensor["shape"], name="indices_input",
                                                dtype=indices_tensor["dtype"])
                direction = "descend" if largest else "ascend"
                if input_tensor["dtype"] == "bfloat16":
                    x_input_fp32 = tbe.cast_to(x_input, "float32")
                    value, indices = tbe.topk(x_input_fp32, k_var[0], sort_axis=-1, direction=direction,
                                              return_type="both", indices_dtype=out_indices_tensor["dtype"],
                                              need_cast=True)
                else:
                    value, indices = tbe.topk(x_input, k_var[0], sort_axis=-1, direction=direction, return_type="both",
                                              indices_dtype=out_indices_tensor["dtype"])
                tensors.append([x_input, indices_input, value, indices])
            with tvm.target.cce():
                sch = tbe.auto_schedule([value, indices])
            schedules.append(sch)
    else:
        for (_x, ) in ins:
            with tbe.compute():
                x_shape = shape_util.variable_shape([_x], "sort")
                x_input = tvm.placeholder(x_shape, name="data_input", dtype=input_tensor["dtype"])
                indices_input = tvm.placeholder(indices_tensor["shape"], name="indices_input",
                                                dtype=indices_tensor["dtype"])
                direction = "descend" if largest else "ascend"
                if input_tensor["dtype"] == "bfloat16":
                    x_input_fp32 = tbe.cast_to(x_input, "float32")
                    value, indices = tbe.topk(x_input_fp32, k, sort_axis=-1, direction=direction, return_type="both",
                                              indices_dtype=out_indices_tensor["dtype"], need_cast=True)
                else:
                    value, indices = tbe.topk(x_input, k, sort_axis=-1, direction=direction, return_type="both",
                                          indices_dtype=out_indices_tensor["dtype"])
                tensors.append([x_input, indices_input, value, indices])
            with tvm.target.cce():
                sch = tbe.auto_schedule([value, indices])
            schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


def topk_tik(input_tensor, indices_tensor, out_tensor, out_indices_tensor, k, sorted, dim, largest, kernel_name, mode):
    SortParam.SUPPORT_VBITSORT32 = tbe_platform.api_check_support("tik.vbitsort32")
    if SortParam.SUPPORT_VBITSORT32:
        SortParam.SORT_REGION_BYTE = 8
    else:
        SortParam.SORT_REGION_BYTE = 16
    SortParam.SORT_ONCE_NUM = tbe_platform.VECTOR_INST_BLOCK_WIDTH // SortParam.SORT_REGION_BYTE

    dtype = input_tensor.get("dtype")
    indices_dtype = indices_tensor.get("dtype")
    out_dtype = out_tensor.get("dtype")
    out_indices_dtype = out_indices_tensor.get("dtype")
    check_list = ("float16")
    out_indices_check_list = ("int32")
    para_check.check_dtype(dtype, check_list, param_name="input_tensor")
    para_check.check_dtype(indices_dtype, check_list, param_name="indices_tensor")
    para_check.check_dtype(out_dtype, check_list, param_name="out_tensor")
    para_check.check_dtype(out_indices_dtype, out_indices_check_list, param_name="out_indices_tensor")
    profile = tik.Dprofile()
    tik_instance = tik.Tik(profile)
    obj_gm = GlobalVarGM(tik_instance)
    obj_ub = GlobalVarUB()
    obj_tiling_gm = obj_gm.tiling_gm
    obj_tiling = GlobalVarTilingScalar(tik_instance, obj_tiling_gm, mode, k, input_tensor.get("shape"))
    if k > 4 * obj_tiling.cols_per_part or (k < 1 and k != -1):
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'k', 1, 4 * obj_tiling.cols_per_part, k)

    return top_k_compute(tik_instance, obj_gm, obj_tiling, obj_ub, profile, dtype, indices_dtype, largest, k,
                         kernel_name, mode)


# 'pylint: disable=too-many-arguments,too-many-locals,unused-argument,global-statement,redefined-builtin
@register_operator("TopKD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def top_k_d(input_tensor,
            indices_tensor,
            out_tensor,
            out_indices_tensor,
            k,
            sorted=True,
            dim=-1,
            largest=True,
            kernel_name='top_k',
            mode="dynamic"):
    """
    top_k interface

    Parameters
    ----------
    input_tensor: dict. input params shape, dtype and range
    indices_tensor: dict. input indices shape, dtype and range
    out_tensor: dict. output shape, dtype and range
    out_indices_tensor: dict. output index shape, dtype and range
    k: int. Number of largest elements to be select
    sorted : bool. if is sorted
    largest : bool. if is sorted by largest
    kernel_name: kernel name of top_k op
    """
    if dim is None:
        dim = -1

    sort_one = (mode == "static" and input_tensor.get("shape")[-1] == 1)
    if tbe_platform.api_check_support("tbe.dsl.topk", "float16") and not sort_one:
        topk_dsl(input_tensor, indices_tensor, out_tensor, out_indices_tensor, k, sorted, dim, largest, kernel_name)
    else:
        topk_tik(input_tensor, indices_tensor, out_tensor, out_indices_tensor, k, sorted, dim, largest, kernel_name,
                 mode)
