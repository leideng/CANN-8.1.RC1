#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
gather_elements
"""

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform as tbe_platform_adapter
from tbe.tik.common.tik_get_soc_name import get_soc_name


# 'pylint: disable=too-few-public-methods
class Constant(object):
    """
    The class for constant
    """
    PARAMS_SIZE = 2 ** 31 - 1
    INDICE_NUM = 2 ** 31 - 1
    TILING_ARG_NUM = 43
    MAX_DIMS = 8
    # reserved ub size
    RESERVED_UB_SIZE = 2 * 1024
    CONVERT_TO_AICPU_UB = 3000 * 1024
    INDICES_NUM_THRESHOULD = 2048
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    BLOCK_SIZE = 32
    TILING_MODE_X_LARGE_INDICES_LARGE = 1
    TILING_MODE_X_SMALL_INDICES_LARGE = 2
    TILING_MODE_X_SLICE_INDICES_LARGE = 3
    TILING_MODE_X_LARGE_INDICES_LARGE_DIFF_SHAPE = 4
    TILING_MODE_X_SMALL_INDICES_LARGE_DIFF_SHAPE = 5
    TILING_MODE_X_SLICE_INDICES_LARGE_DIFF_SHAPE = 6
    # tiling mode when params and indices are so large that both are cut into slices
    TILING_MODE_FOR_LAST_AXIS = 7
    TILING_MODE_FOR_LAST_AXIS_VGATHER = 8
    TILING_MODE_FOR_LAST_AXIS_DIFF_SHAPE = 9
    TILING_MODE_FOR_LAST_AXIS_CUT_VGATHER = 10
    TILING_MODE_FOR_LAST_AXIS_VGATHER_310P = 11
    DATA_MOVE_BURST_THRESHOLD = 4095
    MAX_MASK = 256
    MAX_MASK_BLOCK = 8
    HALF_MASK_BLOCK = 4
    INDICE_DSIZE_INT64 = 8
    INDICE_DSIZE_INT32 = 4
    INT_MAX = 2147483647
    HALF = 2

    DIM_0 = 0
    DIM_1 = 1
    DIM_2 = 2
    DIM_3 = 3
    DIM_4 = 4
    DIM_5 = 5
    DIM_6 = 6
    DIM_7 = 7
    TRUE_ = 1
    FALSE_ = 0

    MAX_REPEAT_TIME = 255
    ONE_REPEAT_BYTES = 256
    FULL_MASK_16BITS = 128
    FULL_MASK_32BITS = 64
    FULL_MASK_64BITS = 32

    TYPE_LEN_DICT = {"float16": 2, "float32": 4, "int8": 1, "uint8": 1,
                     "int16": 2, "uint16": 2, "int32": 4, "uint32": 4,
                     "int64": 8, "uint64": 8, "float": 4, "bfloat16": 2}
    VGATHER_DTYPES = ["float16", "bfloat16"]


def if_same_dim_value_except_axis(x_shape, index_shape, axis):
    """
    check whether the shape of x and index is the same except the axis
    """
    for dim, value in enumerate(x_shape):
        if (dim != axis) and (value != index_shape[dim]):
            return False
    return True


def is_last_axis_support(x_dict, indices_dict, axis, large_num_per_block):
    """
    check whether the cases of last axis are supported
    """
    x_shape = x_dict.get("shape")
    x_dtype = x_dict.get("dtype").lower()
    x_dsize = Constant.TYPE_LEN_DICT.get(x_dtype)
    index_shape = indices_dict.get("shape")
    indices_dtype = indices_dict.get("dtype").lower()
    index_dsize = Constant.TYPE_LEN_DICT.get(indices_dtype)

    dims = len(x_shape)
    axis = (axis + dims) % dims
    x_axis = x_shape[axis]
    index_axis = index_shape[axis]
    repeat_per_core = 1
    same_dim_value_except_axis_flag = if_same_dim_value_except_axis(x_shape, index_shape, axis)
    ub_size = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)
    available_ub_size = ub_size - Constant.RESERVED_UB_SIZE
    is_last_axis = (axis == dims - 1)

    # Branch judgment for 910b vgather
    vgather_flag = tbe_platform_adapter.api_check_support("tik.vgather") and\
                   x_dtype in Constant.VGATHER_DTYPES
    api_support_flag = vgather_flag and tbe_platform_adapter.api_check_support("tik.vconv", "s642s32") and\
                       tbe_platform_adapter.api_check_support("tik.vshr", "int32")
                   
    if is_last_axis and same_dim_value_except_axis_flag and api_support_flag:
        all_data_size = x_axis * x_dsize + index_axis * (x_dsize + index_dsize * 2)
        if all_data_size < available_ub_size:
            return True
    
    # Branch judgment for cutting indices into slices
    last_dim_size = x_axis * x_dsize + index_axis * (x_dsize + index_dsize)
    cutting_into_slices_flag = index_axis % large_num_per_block == 0 and\
                               last_dim_size >= available_ub_size and\
                               x_axis * x_dsize <= available_ub_size / 2 and\
                               same_dim_value_except_axis_flag
    if is_last_axis and cutting_into_slices_flag:
        return True

    # Normal branches
    if is_last_axis and same_dim_value_except_axis_flag:
        if index_axis % large_num_per_block:
            while repeat_per_core * index_axis % large_num_per_block != 0:
                repeat_per_core += 1
    elif is_last_axis and index_axis < large_num_per_block:
        return False
    elif not is_last_axis:
        return False
    all_data_size = repeat_per_core * x_axis * x_dsize + repeat_per_core * index_axis * (x_dsize + index_dsize)
    
    if all_data_size >= available_ub_size:
        return False
    return True


# 'pylint: disable=unused-argument,unused-variable
# 'pylint: disable=too-many-arguments,too-many-locals,too-many-return-statements
def check_supported(x_dict, indices_dict, y_dict, dim=0, kernel_name="GatherElements"):
    """
    check whether ai_core is supported
    """
    x_shape = x_dict.get("shape")
    if int(-1) in x_shape or int(-2) in x_shape:
        return "Unknown"
    
    x_dtype = x_dict.get("dtype").lower()
    indices_dtype = indices_dict.get("dtype").lower()
    indices_shape = indices_dict.get("shape")
    params_total = 1
    indices_total = 1
    for x_shape_range_i in x_shape:
        params_total *= x_shape_range_i
    if x_shape[dim] > Constant.INT_MAX / Constant.HALF:
        return False, "shape range of x axis is larger than the threshold."
    for indices_shape_range_i in indices_shape:
        indices_total *= indices_shape_range_i
    params_dsize = Constant.TYPE_LEN_DICT.get(x_dtype)
    indices_dsize = Constant.TYPE_LEN_DICT.get(indices_dtype)
    params_blocknum = Constant.BLOCK_SIZE // params_dsize
    indices_blocknum = Constant.BLOCK_SIZE // indices_dsize
    large_num_per_block = max(params_blocknum, indices_blocknum)
    params_total_ceil = ceil_value(params_total, params_blocknum) * params_blocknum * params_dsize
    if params_total_ceil > Constant.INT_MAX:
        reason = "params num is too large."
        return False, reason
    if is_last_axis_support(x_dict, indices_dict, dim, large_num_per_block):
        return True, "" 
    if (params_total_ceil >= Constant.CONVERT_TO_AICPU_UB and \
        indices_total >= Constant.INDICES_NUM_THRESHOULD):
        reason = "params num is larger than convert_to_aicpu_ub and indices num is larger than 2k."
        return False, reason
    return True, ""


def ceil_value(value, factor):
    """
    if not divide exactly then plus 1
    """
    return (value + factor - 1) // factor


class GatherElements(object):
    """
        Function: use to store concat base parameters
    """
    # 'pylint: disable=too-many-public-methods,invalid-name,too-many-arguments,too-many-locals
    # 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements,unused-argument
    def __init__(self, params_dict, indices_dict, y_dict, axis, kernel_name):
        """
        constructor of GatherElements

        Parameters
        ----------
        params_dict: dict
            shape and dtype of input params
        indices_dict: dict
            shape and dtype of input indices
        axis: int
            which axis to gather on
        y_dict: dict
            shape and dtype of output, should be same dtype as input
        kernel_name: str
            kernel name, default value is "GatherElements"

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.params_dtype = params_dict.get("dtype").lower()
        self.indices_dtype = indices_dict.get("dtype").lower()
        self.y_dtype = y_dict.get("dtype").lower()
        if self.params_dtype == "bfloat16":
            self.params_dtype = "float16"
            self.y_dtype = "float16"
        self.tiling_dtype = Constant.INT64
        
        # shape total
        self.x_shape = (Constant.PARAMS_SIZE,)
        self.indices_shape = (Constant.INDICE_NUM,)
        self.y_shape = (Constant.INDICE_NUM,)

        self.check_params(kernel_name)

        self.ub_size = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)
        self.core_num = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.CORE_NUM)
        
        self.kernel_name = kernel_name
        self.params_dsize = Constant.TYPE_LEN_DICT.get(self.params_dtype)
        self.indices_dsize = Constant.TYPE_LEN_DICT.get(self.indices_dtype)
        self.param_smaller_than_indices = None
        if self.indices_dsize > self.params_dsize:
            self.param_smaller_than_indices = self.indices_dsize // self.params_dsize
        else:
            self.param_smaller_than_indices = 1
        self.params_num_each_block = Constant.BLOCK_SIZE // self.params_dsize
        self.indices_num_each_block = Constant.BLOCK_SIZE // self.indices_dsize
        self.larger_num_each_block = max(self.params_num_each_block, self.indices_num_each_block)

        self.vgather_flag = tbe_platform_adapter.api_check_support("tik.vgather") and\
                       self.params_dtype in Constant.VGATHER_DTYPES
        api_support_flag = self.vgather_flag and tbe_platform_adapter.api_check_support("tik.vconv", "s642s32") and\
                           tbe_platform_adapter.api_check_support("tik.vshr", "int32")
        if api_support_flag:
            self.support_vgather = Constant.TRUE_
        else:
            self.support_vgather = Constant.FALSE_
        infer_vgather_flag = self.vgather_flag and self.indices_dtype == "int32" and\
                             not tbe_platform_adapter.api_check_support("tik.vconv", "s642s32")
        if infer_vgather_flag:
            self.infer_vgather = Constant.TRUE_
        else:
            self.infer_vgather = Constant.FALSE_

        self.support_data_move_pad = tbe_platform_adapter.api_check_support("tik.data_move_pad")
        self.special_check = get_soc_name() == "Ascend310P" and self.indices_dtype == "int32" and \
                             self.params_dtype in Constant.VGATHER_DTYPES

        self.x = None
        self.indices = None
        self.y = None
        self.tiling_gm = None
        self.x_ub = None
        self.indices_ub = None
        self.res_ub = None

        # tiling parameters of x
        self.axis = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="axis")
        self.params_pre = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_pre")
        self.params_axis = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_axis")
        self.params_row = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_row")
        self.params_total = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_total")

        # tiling parameters of indices
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.indices_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num")
        self.indices_axis = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_axis")
        self.indices_num_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_each_core")
        self.indices_num_remaining = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_remaining")
        self.indices_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_loop_num")
        self.indices_row_num_once = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_once")
        self.indices_row_num_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_last")
        self.remaining_block_remain = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="remaining_block_remain")
        self.remaining_block_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="remaining_block_num")

        # x cut into slices
        self.slice_thickness_once = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="slice_thickness_once")
        self.slice_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="slice_num")
        self.slice_thickness_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="slice_thickness_last")

        # indices cut into slices
        self.indices_slice_thickness_dim1 = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                                     name="indices_slice_thickness_dim1")
        self.indices_slice_thickness_dim1_last = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                                          name="indices_slice_thickness_dim1_last")
        self.indices_slice_num_dim1 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_slice_num_dim1")

        self.indices_shape_range_tensor = self.tik_instance.Tensor(Constant.INT64, (Constant.MAX_DIMS,),
                                                                   name="indices_shape_range_tensor",
                                                                   scope=tik.scope_ubuf)
        self.params_shape_range_tensor = self.tik_instance.Tensor(Constant.INT64, (Constant.MAX_DIMS,),
                                                                  name="params_shape_range_tensor",
                                                                  scope=tik.scope_ubuf)
        self.dims = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dims")

        self.repeat_per_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="repeat_per_core")
        self.rounds = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="rounds")
        self.rounds_tail = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="rounds_tail")
        self.db_flag = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="db_flag", init_value=0)
        self.vgather_310p_flag = self.tik_instance.Scalar(dtype=Constant.INT64, name="vgather_310p_flag", init_value=0)

        self.vgather_mask = Constant.ONE_REPEAT_BYTES // self.params_dsize
        
        self.x_block_num = None
        self.x_align_num = None
        self.indices_axis_block_num = None
        self.indices_axis_out_block_num = None
        self.rev_params_axis = None
        self.params_axis_int32 = None       

    def output_indices_index(self, indices_offset):
        """
        compute an index tensor of indices offset

        Parameters
        ----------
        indices_offset: offset of an indice

        Returns
        -------
        index_tensor: all indexes of the indice
        """
        index_tensor = self.tik_instance.Tensor(Constant.INT64, (Constant.MAX_DIMS,), name="index_tensor", 
                                                scope=tik.scope_ubuf)
        acc_temp = self.tik_instance.Scalar(Constant.INT64, init_value=self.indices_num)
        temp_range = self.tik_instance.Scalar(Constant.INT64, init_value=1)
        with self.tik_instance.for_range(0, self.dims) as dims_i:
            temp_range.set_as(self.indices_shape_range_tensor[dims_i])
            acc_temp.set_as(acc_temp // temp_range)
            index_tensor[dims_i].set_as(indices_offset // acc_temp % temp_range)
        return index_tensor

    def output_param_offset(self, index_tensor):
        """
        compute param offset according to the index tensor

        Parameters
        ----------
        index_tensor: all indexes of the indice

        Returns
        -------
        param_offset: offset of a param
        """
        param_offset = self.tik_instance.Scalar(Constant.INT64, init_value=0)
        acc_temp = self.tik_instance.Scalar(Constant.INT64, init_value=self.params_total)
        temp_range = self.tik_instance.Scalar(Constant.INT64, init_value=1)
        temp_index = self.tik_instance.Scalar(Constant.INT64, init_value=0)
        with self.tik_instance.for_range(0, self.dims) as dims_i:
            temp_range.set_as(self.params_shape_range_tensor[dims_i])
            acc_temp.set_as(acc_temp // temp_range)
            temp_index.set_as(index_tensor[dims_i])
            param_offset.set_as(temp_index * acc_temp + param_offset)
        return param_offset

    def check_params(self, kernel_name):
        """
        check params

        Parameters
        ----------
        kernel_name

        Returns
        -------
        None
        """
        dtype_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                      "uint32", "uint64", "float16", "float32", "float")
        indices_support_dtype_list = ("int32", "int64")
        para_check.check_dtype(self.params_dtype, dtype_list, param_name="x")
        para_check.check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        if self.y_dtype != self.params_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name,
                                                                  "y", "x", self.y_dtype, self.params_dtype)

    def get_tiling_args(self, tiling_ub):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from gather_elements tiling

        Returns
        -------
        None
        """ 
        self.axis.set_as(tiling_ub[1])
        self.params_pre.set_as(tiling_ub[2])
        self.params_axis.set_as(tiling_ub[3])
        self.params_row.set_as(tiling_ub[4])
        self.params_total.set_as(tiling_ub[5])
        self.need_core_num.set_as(tiling_ub[6])
        self.indices_num.set_as(tiling_ub[7])
        self.indices_axis.set_as(tiling_ub[8])
        self.indices_num_each_core.set_as(tiling_ub[9])
        self.indices_num_remaining.set_as(tiling_ub[10])
        self.indices_loop_num.set_as(tiling_ub[11])
        self.indices_row_num_once.set_as(tiling_ub[12])
        self.indices_row_num_last.set_as(tiling_ub[13])
        self.remaining_block_remain.set_as(tiling_ub[14])
        self.remaining_block_num.set_as(tiling_ub[15])
        self.slice_thickness_once.set_as(tiling_ub[16])
        self.slice_num.set_as(tiling_ub[17])
        self.slice_thickness_last.set_as(tiling_ub[18])
        self.indices_slice_thickness_dim1.set_as(tiling_ub[19])
        self.indices_slice_thickness_dim1_last.set_as(tiling_ub[20])
        self.indices_slice_num_dim1.set_as(tiling_ub[21])
        params_shape_0 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_shape_0")
        params_shape_0.set_as(tiling_ub[22])
        self.params_shape_range_tensor[Constant.DIM_0].set_as(params_shape_0)
        params_shape_1 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_shape_1")
        params_shape_1.set_as(tiling_ub[23])
        self.params_shape_range_tensor[Constant.DIM_1].set_as(params_shape_1)
        params_shape_2 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_shape_2")
        params_shape_2.set_as(tiling_ub[24])
        self.params_shape_range_tensor[Constant.DIM_2].set_as(params_shape_2)
        params_shape_3 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_shape_3")
        params_shape_3.set_as(tiling_ub[25])
        self.params_shape_range_tensor[Constant.DIM_3].set_as(params_shape_3)
        params_shape_4 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_shape_4")
        params_shape_4.set_as(tiling_ub[26])
        self.params_shape_range_tensor[Constant.DIM_4].set_as(params_shape_4)
        params_shape_5 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_shape_5")
        params_shape_5.set_as(tiling_ub[27])
        self.params_shape_range_tensor[Constant.DIM_5].set_as(params_shape_5)
        params_shape_6 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_shape_6")
        params_shape_6.set_as(tiling_ub[28])
        self.params_shape_range_tensor[Constant.DIM_6].set_as(params_shape_6)
        params_shape_7 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_shape_7")
        params_shape_7.set_as(tiling_ub[29])
        self.params_shape_range_tensor[Constant.DIM_7].set_as(params_shape_7)
        indices_shape_0 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_shape_0")
        indices_shape_0.set_as(tiling_ub[30])
        self.indices_shape_range_tensor[Constant.DIM_0].set_as(indices_shape_0)
        indices_shape_1 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_shape_1")
        indices_shape_1.set_as(tiling_ub[31])
        self.indices_shape_range_tensor[Constant.DIM_1].set_as(indices_shape_1)
        indices_shape_2 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_shape_2")
        indices_shape_2.set_as(tiling_ub[32])
        self.indices_shape_range_tensor[Constant.DIM_2].set_as(indices_shape_2)
        indices_shape_3 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_shape_3")
        indices_shape_3.set_as(tiling_ub[33])
        self.indices_shape_range_tensor[Constant.DIM_3].set_as(indices_shape_3)
        indices_shape_4 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_shape_4")
        indices_shape_4.set_as(tiling_ub[34])
        self.indices_shape_range_tensor[Constant.DIM_4].set_as(indices_shape_4)
        indices_shape_5 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_shape_5")
        indices_shape_5.set_as(tiling_ub[35])
        self.indices_shape_range_tensor[Constant.DIM_5].set_as(indices_shape_5)
        indices_shape_6 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_shape_6")
        indices_shape_6.set_as(tiling_ub[36])
        self.indices_shape_range_tensor[Constant.DIM_6].set_as(indices_shape_6)
        indices_shape_7 = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_shape_7")
        indices_shape_7.set_as(tiling_ub[37])
        self.indices_shape_range_tensor[Constant.DIM_7].set_as(indices_shape_7)
        
        self.dims.set_as(tiling_ub[38])
        self.repeat_per_core.set_as(tiling_ub[39])
        self.rounds.set_as(tiling_ub[40])
        self.rounds_tail.set_as(tiling_ub[41])
        self.db_flag.set_as(tiling_ub[42])
        
        self.x_block_num = ceil_value(self.params_axis, self.params_num_each_block)
        self.x_align_num = self.x_block_num * self.params_num_each_block
        self.indices_axis_block_num = self.indices_axis // self.indices_num_each_block
        self.indices_axis_out_block_num = self.indices_axis // self.params_num_each_block

        if self.vgather_flag:
            self.rev_params_axis = self.tik_instance.Scalar(dtype="float32", name="rev_params_axis",
                                                            init_value=self.params_axis)
            self.params_axis_int32 = self.tik_instance.Scalar(dtype="int32", name="params_axis_int32",
                                                              init_value=self.params_axis)
            self.rev_params_axis.set_as(1.0 / self.rev_params_axis)

    def compute_indices_res_ubsize(self, x_ub_size):
        """
        Compute indices_ub and res_ub according to ub size of AIcore and ub size of x

        Parameters
        ----------
        tiling_ub: tensor, runtime params from gather_elements tiling

        Returns
        -------
        None
        """
        indices_num = (self.ub_size - x_ub_size - Constant.RESERVED_UB_SIZE) // (self.indices_dsize + self.params_dsize)
        res_num = indices_num // self.params_num_each_block * self.params_num_each_block
        indices_num = indices_num // self.params_num_each_block * self.params_num_each_block
        res_num = min(indices_num, res_num)
        indices_ub_size = res_num * self.indices_dsize
        res_ub_size = res_num * self.params_dsize
        return indices_ub_size, res_ub_size

    def gather_elements_compute_tiling(self):
        """
        Main process of gather_elements

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
        tiling_size = ceil_value(Constant.TILING_ARG_NUM * Constant.TYPE_LEN_DICT.get(self.tiling_dtype),
                                 Constant.BLOCK_SIZE)

        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, tiling_size, 0, 0)
        tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
        tiling_mode.set_as(tiling_ub[0])

        self.get_tiling_args(tiling_ub)

        with self.tik_instance.for_range(0, self.need_core_num, block_num=self.need_core_num) as block_id:
            with self.tik_instance.if_scope(tiling_mode == Constant.TILING_MODE_X_LARGE_INDICES_LARGE):
                self.compute_mode_x_large_indices_large(block_id, compute_func=self.compute_x_larger_cache,
                                                        get_output_gm_offset_func=self.get_output_gm_offset_same)
            with self.tik_instance.elif_scope(tiling_mode == Constant.TILING_MODE_X_SMALL_INDICES_LARGE):
                self.compute_mode_x_small_indices_large(block_id, compute_func=self.compute_x_less_cache,
                                                        get_output_gm_offset_func=self.get_output_gm_offset_same)
            with self.tik_instance.elif_scope(tiling_mode == Constant.TILING_MODE_X_SLICE_INDICES_LARGE):
                self.compute_mode_x_slice_indices_large(block_id, compute_func=self.compute_x_slice_less_cache,
                                                        get_output_gm_offset_func=self.get_output_gm_offset_same)

            with self.tik_instance.elif_scope(tiling_mode == Constant.TILING_MODE_X_LARGE_INDICES_LARGE_DIFF_SHAPE):
                self.compute_mode_x_large_indices_large(block_id, compute_func=self.compute_x_larger_cache,
                                                        get_output_gm_offset_func=self.get_output_gm_offset_dif)
            with self.tik_instance.elif_scope(tiling_mode == Constant.TILING_MODE_X_SMALL_INDICES_LARGE_DIFF_SHAPE):
                self.compute_mode_x_small_indices_large(block_id, compute_func=self.compute_x_less_cache,
                                                        get_output_gm_offset_func=self.get_output_gm_offset_dif)
            with self.tik_instance.elif_scope(tiling_mode == Constant.TILING_MODE_X_SLICE_INDICES_LARGE_DIFF_SHAPE):
                self.compute_mode_x_slice_indices_large(block_id, compute_func=self.compute_x_slice_less_cache,
                                                        get_output_gm_offset_func=self.get_output_gm_offset_dif)
            with self.tik_instance.else_scope():   
                self.compute_pre_for_last_axis(block_id, tiling_mode)

    def compute_pre_for_last_axis(self, block_id, tiling_mode):
        """
            computational process of cores for last axis
        """
        with self.tik_instance.if_scope(self.db_flag > 0):
            with self.tik_instance.for_range(0, self.indices_loop_num, thread_num=2) as k:
                task_id = block_id + k * self.need_core_num
                self.compute_core_for_last_axis(task_id, tiling_mode)

            with self.tik_instance.if_scope(block_id < self.indices_row_num_last):
                task_id = block_id + self.indices_loop_num * self.need_core_num
                self.compute_core_for_last_axis(task_id, tiling_mode)

        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.indices_loop_num) as k:
                task_id = block_id + k * self.need_core_num
                self.compute_core_for_last_axis(task_id, tiling_mode)

            with self.tik_instance.if_scope(block_id < self.indices_row_num_last):
                task_id = block_id + self.indices_loop_num * self.need_core_num
                self.compute_core_for_last_axis(task_id, tiling_mode)

    def compute_core_for_last_axis(self, task_id, tiling_mode):
        """
            computational process of single core for last axis
        """
        if self.infer_vgather:
            with self.tik_instance.if_scope(tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS_VGATHER_310P):
                with self.tik_instance.if_scope(tik.all(self.rounds_tail != 0, (task_id + 1) == self.rounds)):
                    self.process_last_axis_aligned_infer_vgather(task_id, self.rounds_tail)
                with self.tik_instance.else_scope():
                    self.process_last_axis_aligned_infer_vgather(task_id, self.repeat_per_core)
        
        with self.tik_instance.if_scope(tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS_CUT_VGATHER):
            self.process_last_axis_vgather_cut_slice(task_id)

        with self.tik_instance.elif_scope(tik.all(self.indices_axis % self.larger_num_each_block == 0,
                                            tik.any(tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS,
                                                    tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS_VGATHER,
                                                    tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS_DIFF_SHAPE))):
            if self.support_vgather:
                with self.tik_instance.if_scope(tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS_VGATHER):
                    self.process_last_axis_aligned_vgather(task_id, tiling_mode)
                with self.tik_instance.else_scope():
                    self.process_last_axis_aligned(task_id, tiling_mode)
            else:
                self.process_last_axis_aligned(task_id, tiling_mode)

        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(tik.any(tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS,
                tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS_VGATHER)):
                with self.tik_instance.if_scope(tik.all(self.rounds_tail != 0, (task_id + 1) == self.rounds)):
                    self.process_last_axis_unaligned_entrance(task_id, self.rounds_tail, tiling_mode) # tail_block
                with self.tik_instance.else_scope():
                    self.process_last_axis_unaligned_entrance(task_id, self.repeat_per_core, tiling_mode)
            with self.tik_instance.if_scope(tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS_DIFF_SHAPE):
                self.process_last_axis_unaligned_diff_shape(task_id)

    def get_loop_args(self, compute_mask, compute_num, max_repeat_times):
        """
            needed args for tik vec calculating
        """
        vconv_loop = compute_num // compute_mask // max_repeat_times
        repeat_tail = compute_num % (compute_mask * max_repeat_times) // compute_mask
        last_tail = compute_num % (compute_mask * max_repeat_times) % compute_mask
        return vconv_loop, repeat_tail, last_tail

    def vconv_int64_to_int32(self, dst, src, compute_num):
        """
            convert int64 to int32
        """
        vconv_loop, repeat_tail, last_tail = self.get_loop_args(Constant.FULL_MASK_64BITS, compute_num,
            Constant.MAX_REPEAT_TIME)
        with self.tik_instance.for_range(0, vconv_loop) as vconv_loop_id:
            loop_offset = Constant.FULL_MASK_64BITS * Constant.MAX_REPEAT_TIME * vconv_loop_id
            self.tik_instance.vconv(Constant.FULL_MASK_64BITS, "", dst[loop_offset], src[loop_offset],
                Constant.MAX_REPEAT_TIME, 1, 1, 4, 8)
        with self.tik_instance.if_scope(repeat_tail > 0):
            loop_offset = Constant.FULL_MASK_64BITS * Constant.MAX_REPEAT_TIME * vconv_loop
            self.tik_instance.vconv(Constant.FULL_MASK_64BITS, "", dst[loop_offset], src[loop_offset], repeat_tail,
                1, 1, 4, 8)
        with self.tik_instance.if_scope(last_tail):
            loop_offset = Constant.FULL_MASK_64BITS * Constant.MAX_REPEAT_TIME * vconv_loop +\
                Constant.FULL_MASK_64BITS * repeat_tail
            self.tik_instance.vconv(last_tail, "", dst[loop_offset], src[loop_offset], 1, 1, 1, 4, 8)

    def alloc_indices_ub(self, compute_num, task_offset):
        """
            alloc indices ub & move data from gm to ub
        """
        self.indices_ub = self.tik_instance.Tensor(Constant.INT32, (compute_num,), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        if self.indices_dtype == Constant.INT64:
            indices_ub_int64 = self.tik_instance.Tensor(Constant.INT64, (compute_num,),
                name="indices_ub_int64", scope=tik.scope_ubuf)
            indices_ub_int64 = indices_ub_int64.reinterpret_cast_to("int32")
            self.indices = self.indices.reinterpret_cast_to("int32")
            self.tik_instance.data_move_pad(indices_ub_int64, self.indices[task_offset * 2], 1,
                                            self.indices_axis * self.indices_dsize, 0, 0)
            indices_ub_int64 = indices_ub_int64.reinterpret_cast_to("int64")
            self.indices = self.indices.reinterpret_cast_to("int64")
            self.vconv_int64_to_int32(self.indices_ub, indices_ub_int64, compute_num)
        else:
            self.tik_instance.data_move_pad(self.indices_ub, self.indices[task_offset], 1,
                                            self.indices_axis * self.indices_dsize, 0, 0)

    def do_negative_indices(self, dst, compute_num):
        """
            when indices ub have some negative elems, the func could add self.params_axis when < 0 (add 0 when >= 0)
        """
        vsh_loop, repeat_tail, last_tail = self.get_loop_args(Constant.FULL_MASK_32BITS, compute_num,
            Constant.MAX_REPEAT_TIME)
        indices_div = self.tik_instance.Tensor(Constant.INT32, (compute_num,), name="indices_div",
            scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, vsh_loop) as vsh_loop_id:
            loop_offset = Constant.FULL_MASK_32BITS * Constant.MAX_REPEAT_TIME * vsh_loop_id
            self.tik_instance.vshr(Constant.FULL_MASK_32BITS, indices_div[loop_offset], dst[loop_offset], 31,
                Constant.MAX_REPEAT_TIME, 1, 1, 8, 8)
            self.tik_instance.vmuls(Constant.FULL_MASK_32BITS, indices_div[loop_offset], indices_div[loop_offset],
                self.params_axis_int32, Constant.MAX_REPEAT_TIME, 1, 1, 8, 8)
            self.tik_instance.vsub(Constant.FULL_MASK_32BITS, dst[loop_offset], dst[loop_offset],
                indices_div[loop_offset], Constant.MAX_REPEAT_TIME, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vmuls(Constant.FULL_MASK_32BITS, dst[loop_offset], dst[loop_offset], self.params_dsize,
                Constant.MAX_REPEAT_TIME, 1, 1, 8, 8)
        with self.tik_instance.if_scope(repeat_tail > 0):
            loop_offset = Constant.FULL_MASK_32BITS * Constant.MAX_REPEAT_TIME * vsh_loop
            self.tik_instance.vshr(Constant.FULL_MASK_32BITS, indices_div[loop_offset], dst[loop_offset], 31,
                repeat_tail, 1, 1, 8, 8)
            self.tik_instance.vmuls(Constant.FULL_MASK_32BITS, indices_div[loop_offset], indices_div[loop_offset],
                self.params_axis_int32, repeat_tail, 1, 1, 8, 8)
            self.tik_instance.vsub(Constant.FULL_MASK_32BITS, dst[loop_offset], dst[loop_offset],
                indices_div[loop_offset], repeat_tail, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vmuls(Constant.FULL_MASK_32BITS, dst[loop_offset], dst[loop_offset], self.params_dsize,
                repeat_tail, 1, 1, 8, 8)
        with self.tik_instance.if_scope(last_tail > 0):
            loop_offset = Constant.FULL_MASK_32BITS * Constant.MAX_REPEAT_TIME * vsh_loop +\
                Constant.FULL_MASK_32BITS * repeat_tail
            self.tik_instance.vshr(last_tail, indices_div[loop_offset], dst[loop_offset], 31, 1, 1, 1, 8, 8)
            self.tik_instance.vmuls(last_tail, indices_div[loop_offset], indices_div[loop_offset],
                self.params_axis_int32, 1, 1, 1, 8, 8)
            self.tik_instance.vsub(last_tail, dst[loop_offset], dst[loop_offset], indices_div[loop_offset], 1, 1, 1, 1,
                8, 8, 8)
            self.tik_instance.vmuls(last_tail, dst[loop_offset], dst[loop_offset], self.params_dsize, 1, 1, 1, 8, 8)

    def do_vgather(self, dst, src, src_off, compute_num):
        """
            tik vgather
        """
        vgather_loop, repeat_tail, last_tail = self.get_loop_args(self.vgather_mask,
                                                                  compute_num,
                                                                  Constant.MAX_REPEAT_TIME)
        with self.tik_instance.for_range(0, vgather_loop) as vgather_loop_id:
            loop_offset = self.vgather_mask * Constant.MAX_REPEAT_TIME * vgather_loop_id
            self.tik_instance.vgather(self.vgather_mask, dst[loop_offset], src, src_off[loop_offset],
                                      Constant.MAX_REPEAT_TIME, 8)
        with self.tik_instance.if_scope(repeat_tail > 0):
            loop_offset = self.vgather_mask * Constant.MAX_REPEAT_TIME * vgather_loop
            self.tik_instance.vgather(self.vgather_mask, dst[loop_offset], src, src_off[loop_offset], repeat_tail, 8)
        with self.tik_instance.if_scope(last_tail > 0):
            loop_offset = self.vgather_mask * Constant.MAX_REPEAT_TIME * vgather_loop + repeat_tail * self.vgather_mask
            self.tik_instance.vgather(last_tail, dst[loop_offset], src, src_off[loop_offset], 1, 8)

    def process_last_axis_vgather_cut_slice(self, task_id):
        """
            computational process for last axis 32bytes aligned, 310P_vgather
            same shape version
        """
        x_col = self.tik_instance.Scalar(dtype=Constant.INT64, name="x_col", init_value=task_id)
        self.x_ub = self.tik_instance.Tensor(self.params_dtype, (self.x_align_num,),
                                             name="x_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.x_ub, self.x[x_col * self.params_axis], 0, 1, self.x_block_num, 0, 0)

        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (self.slice_thickness_once,), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        self.res_ub = self.tik_instance.Tensor(self.params_dtype, (self.slice_thickness_once,),
                                               name="res_ub", scope=tik.scope_ubuf)
        
        cur_slice_thickness = self.tik_instance.Scalar(dtype=Constant.INT64, name="cur_slice_thickness", init_value=0)
        with self.tik_instance.for_range(0, self.slice_num) as slice_id:
            with self.tik_instance.if_scope(slice_id == self.slice_num - 1):
                cur_slice_thickness.set_as(self.slice_thickness_last)
            with self.tik_instance.else_scope():
                cur_slice_thickness.set_as(self.slice_thickness_once)
            self.tik_instance.data_move(
                self.indices_ub,
                self.indices[self.indices_axis * task_id + slice_id * self.slice_thickness_once],
                0, 1, ceil_value(cur_slice_thickness, self.indices_num_each_block), 0, 0)
            
            if self.special_check:
                all_data_size = (Constant.INDICE_DSIZE_INT32 * 3 + self.params_dsize) *\
                                cur_slice_thickness + self.params_axis * self.params_dsize
                with self.tik_instance.if_scope(all_data_size < (self.ub_size - Constant.RESERVED_UB_SIZE)):
                    self.do_negative_indices_infer(self.indices_ub, cur_slice_thickness)
                    self.do_vgather(self.res_ub, self.x_ub, self.indices_ub, cur_slice_thickness)
                with self.tik_instance.else_scope():
                    self.gather_by_scalar(cur_slice_thickness)
            else:
                self.gather_by_scalar(cur_slice_thickness)
            self.tik_instance.data_move(self.y[self.indices_axis * task_id + slice_id * self.slice_thickness_once],
                                        self.res_ub, 0, 1,
                                        ceil_value(cur_slice_thickness, self.params_num_each_block), 0, 0)
    
    def choose_move_method(self, dst, src, data_move_pad_burst, data_move_burst=1):
        """
            when the data type is integer, choose different handling methods
        """
        ori_dtype = dst.dtype
        if tbe_platform_adapter.api_check_support("tik.data_move_pad", ori_dtype):
            self.tik_instance.data_move_pad(dst, src, 1, data_move_pad_burst, 0, 0)
        elif self.support_data_move_pad: # int64 or uint64
            dst = dst.reinterpret_cast_to(Constant.INT8)
            src = src.reinterpret_cast_to(Constant.INT8)
            self.tik_instance.data_move_pad(dst, src, 1, data_move_pad_burst, 0, 0)
            dst = dst.reinterpret_cast_to(ori_dtype)
            src = src.reinterpret_cast_to(ori_dtype)
        else:
            self.tik_instance.data_move(dst, src, 0, 1, data_move_burst, 0, 0)

    def process_last_axis_aligned(self, task_id, tiling_mode):
        """
            computational process for last axis 32bytes aligned
            same shape + diff shape version
        """
        x_col = self.tik_instance.Scalar(dtype=Constant.INT64, name="x_col", init_value=task_id)
        with self.tik_instance.if_scope(tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS_DIFF_SHAPE):
            self.cal_data_col_offset(task_id, x_col)

        self.x_ub = self.tik_instance.Tensor(self.params_dtype, (self.x_align_num,),
                                             name="x_ub", scope=tik.scope_ubuf)  
        self.choose_move_method(self.x_ub, self.x[x_col * self.params_axis],
                                self.params_axis * self.params_dsize, self.x_block_num)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (self.indices_axis,), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        self.choose_move_method(self.indices_ub, self.indices[self.indices_axis * task_id],
                                self.indices_axis * self.indices_dsize,
                                ceil_value(self.indices_axis, self.indices_num_each_block))
        self.res_ub = self.tik_instance.Tensor(self.params_dtype, (self.indices_axis,),
                                               name="res_ub", scope=tik.scope_ubuf)
        self.gather_by_scalar(self.indices_axis)
        self.choose_move_method(self.y[self.indices_axis * task_id], self.res_ub,
                                self.indices_axis * self.params_dsize,
                                ceil_value(self.indices_axis, self.params_num_each_block))
        
    def gather_by_scalar(self, slice_thickness):
        """
            perform a gather operation by scalar
        """
        indices_value = self.tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
        with self.tik_instance.for_range(0, slice_thickness) as index_i:
            indices_value.set_as(self.indices_ub[index_i])
            indices_value = (indices_value + self.params_axis) % self.params_axis
            self.res_ub[index_i].set_as(self.x_ub[indices_value])

    def do_negative_indices_infer(self, indices_ub_int32, slice_thickness):
        """
            when indices ub have some negative elems, the func could add self.params_axis when < 0 on inference soc
        """
        temp_indices_ub_pre = self.tik_instance.Tensor("float32", (slice_thickness,), name="temp_indices_ub_pre",
                                                        scope=tik.scope_ubuf)
        temp_indices_ub_pro = self.tik_instance.Tensor("int32", (slice_thickness,), name="temp_indices_ub_pro",
                                                        scope=tik.scope_ubuf)
        
        def indice_vector_calculate(mask, offset, repeat):
            """
                process part of one line per time
            """
            self.tik_instance.vec_adds(mask, indices_ub_int32[offset], indices_ub_int32[offset], self.params_axis_int32,
                                       repeat, Constant.MAX_MASK_BLOCK, Constant.MAX_MASK_BLOCK)
            self.tik_instance.vec_conv(mask, 'none', temp_indices_ub_pre[offset], indices_ub_int32[offset],
                                       repeat, Constant.MAX_MASK_BLOCK, Constant.MAX_MASK_BLOCK)
            self.tik_instance.vec_muls(mask, temp_indices_ub_pre[offset], temp_indices_ub_pre[offset],
                                       self.rev_params_axis, repeat,
                                       Constant.MAX_MASK_BLOCK, Constant.MAX_MASK_BLOCK)
            self.tik_instance.vec_conv(mask, 'floor', temp_indices_ub_pro[offset], temp_indices_ub_pre[offset], 
                                       repeat, Constant.MAX_MASK_BLOCK, Constant.MAX_MASK_BLOCK)
            self.tik_instance.vec_muls(mask, temp_indices_ub_pro[offset], temp_indices_ub_pro[offset],
                                       self.params_axis_int32, repeat, Constant.MAX_MASK_BLOCK, Constant.MAX_MASK_BLOCK)
            self.tik_instance.vec_sub(mask, indices_ub_int32[offset], indices_ub_int32[offset],
                                      temp_indices_ub_pro[offset], repeat, Constant.MAX_MASK_BLOCK,
                                      Constant.MAX_MASK_BLOCK, Constant.MAX_MASK_BLOCK)
            self.tik_instance.vec_muls(mask, indices_ub_int32[offset], indices_ub_int32[offset], self.params_dsize,
                                       repeat, Constant.MAX_MASK_BLOCK, Constant.MAX_MASK_BLOCK)

        repeat_loop, repeat_tail, last_tail = self.get_loop_args(Constant.FULL_MASK_32BITS,
                                                                 slice_thickness,
                                                                 Constant.MAX_REPEAT_TIME)
        with self.tik_instance.for_range(0, repeat_loop) as loop_id:
            loop_offset = Constant.FULL_MASK_32BITS * Constant.MAX_REPEAT_TIME * loop_id
            indice_vector_calculate(Constant.FULL_MASK_32BITS, loop_offset, Constant.MAX_REPEAT_TIME)
        with self.tik_instance.if_scope(repeat_tail > 0):
            loop_offset = Constant.FULL_MASK_32BITS * Constant.MAX_REPEAT_TIME * repeat_loop
            indice_vector_calculate(Constant.FULL_MASK_32BITS, loop_offset, repeat_tail)
        with self.tik_instance.if_scope(last_tail):
            loop_offset = slice_thickness - last_tail
            indice_vector_calculate(last_tail, loop_offset, 1)         

    def process_last_axis_aligned_infer_vgather(self, task_id, repeat_cur):
        """
            computational process for last axis 32bytes aligned, where vgather is used on inference soc
            same shape version
        """
        x_col = self.tik_instance.Scalar(dtype=Constant.INT64, name="x_col", init_value=task_id * self.repeat_per_core)
        self.x_ub = self.tik_instance.Tensor(self.params_dtype, (self.x_align_num * repeat_cur,),
                                             name="x_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, repeat_cur) as repeat_id:
            self.tik_instance.data_move(self.x_ub[repeat_id * self.x_align_num],
                                        self.x[x_col * self.params_axis + repeat_id * self.params_axis],
                                        0, 1, self.x_block_num, 0, 0)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (self.indices_axis * repeat_cur,),
                                                   name="indices_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.indices_ub, self.indices[self.indices_axis * self.repeat_per_core * task_id],
                                    0, 1, repeat_cur * self.indices_axis_block_num, 0, 0)
        self.res_ub = self.tik_instance.Tensor(self.params_dtype, (self.indices_axis * repeat_cur,),
                                               name="res_ub", scope=tik.scope_ubuf)
        
        self.do_negative_indices_infer(self.indices_ub, self.indices_axis * repeat_cur)
        with self.tik_instance.for_range(0, repeat_cur) as repeat_id:
            self.do_vgather(self.res_ub[self.indices_axis * repeat_id], self.x_ub[repeat_id * self.x_align_num],
                            self.indices_ub[self.indices_axis * repeat_id], self.indices_axis)
        self.tik_instance.data_move(self.y[self.indices_axis * x_col], self.res_ub, 0, 1,
                                    repeat_cur * self.indices_axis_out_block_num, 0, 0)
    

    def process_last_axis_aligned_vgather(self, task_id, tiling_mode):
        """
            computational process for last axis 32bytes aligned
            same shape + diff shape version
        """
        x_col = self.tik_instance.Scalar(dtype=Constant.INT64, name="x_col", init_value=task_id)
        with self.tik_instance.if_scope(tiling_mode == Constant.TILING_MODE_FOR_LAST_AXIS_DIFF_SHAPE):
            self.cal_data_col_offset(task_id, x_col)

        self.x_ub = self.tik_instance.Tensor(self.params_dtype, (self.x_block_num * self.params_num_each_block,),
                                             name="x_ub", scope=tik.scope_ubuf)  
        self.tik_instance.data_move_pad(self.x_ub, self.x[task_id * self.params_axis], 1,
                                        self.params_axis * self.params_dsize, 0, 0)
        self.res_ub = self.tik_instance.Tensor(self.params_dtype, (self.indices_axis,),
                                          name="res_ub", scope=tik.scope_ubuf)

        self.alloc_indices_ub(self.indices_axis, self.indices_axis * task_id)
        self.do_negative_indices(self.indices_ub, self.indices_axis)
        self.do_vgather(self.res_ub, self.x_ub, self.indices_ub, self.indices_axis)

        self.tik_instance.data_move(self.y[self.indices_axis * task_id], self.res_ub, 0, 1,
                                    ceil_value(self.indices_axis, self.params_num_each_block), 0, 0)

    def process_last_axis_unaligned_entrance(self, task_id, data_repeat_per_block, tiling_mode):
        """
            computational process for last axis 32bytes unaligned
            same shape version
        """
        not_support_data_move_pad_and_drift_addr = tik.all((self.indices_axis < self.larger_num_each_block),
            (self.support_data_move_pad == False))
        if self.support_vgather:
            with self.tik_instance.if_scope(tik.all(not_support_data_move_pad_and_drift_addr,
                                                    tiling_mode != Constant.TILING_MODE_FOR_LAST_AXIS_VGATHER)):
                self.process_last_axis_unaligned(task_id, data_repeat_per_block)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, data_repeat_per_block) as index_i:
                    self.process_last_axis_unaligned_vgather(task_id, index_i)
        else:
            self.process_last_axis_unaligned(task_id, data_repeat_per_block)

    def process_last_axis_unaligned(self, task_id, data_repeat_per_block):
        """
            computational process for last axis 32bytes unaligned
            same shape version
        """
        x_block_num = ceil_value(self.params_axis * data_repeat_per_block, self.params_num_each_block)
        self.x_ub = self.tik_instance.Tensor(self.params_dtype, (x_block_num * self.params_num_each_block,),
                                             name="x_ub", scope=tik.scope_ubuf)

        self.choose_move_method(self.x_ub, self.x[task_id * self.params_axis * self.repeat_per_core],
                                self.params_axis * data_repeat_per_block * self.params_dsize,
                                x_block_num)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (self.repeat_per_core * self.indices_axis,),
                                                   name="indices_ub", scope=tik.scope_ubuf)
        self.choose_move_method(self.indices_ub, self.indices[self.repeat_per_core * self.indices_axis * task_id],
                                data_repeat_per_block * self.indices_axis * self.indices_dsize,
                                ceil_value(data_repeat_per_block * self.indices_axis, self.indices_num_each_block))
        self.res_ub = self.tik_instance.Tensor(self.params_dtype, (self.repeat_per_core * self.indices_axis,),
                                               name="res_ub", scope=tik.scope_ubuf)
        indices_value = self.tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
        with self.tik_instance.for_range(0, self.indices_axis * data_repeat_per_block) as index_i:
            indices_value.set_as(self.indices_ub[index_i])
            indices_value = (indices_value + self.params_axis) % self.params_axis
            pre = index_i // self.indices_axis
            self.res_ub[index_i].set_as(self.x_ub[indices_value + pre * self.params_axis])
        
        self.choose_move_method(self.y[self.repeat_per_core * self.indices_axis * task_id], self.res_ub,
                                data_repeat_per_block * self.indices_axis * self.params_dsize,
                                ceil_value(data_repeat_per_block * self.indices_axis, self.params_num_each_block))

    def process_last_axis_unaligned_vgather(self, task_id, index_i):
        """
            computational process for last axis 32bytes unaligned
            same shape version
            branch1: if support data_move_pad
            branch2: else drift the addr
        """
        aligned_num = ceil_value(self.indices_axis, self.larger_num_each_block) * self.larger_num_each_block
        x_aligned_num = ceil_value(self.params_axis, self.params_num_each_block) * self.params_num_each_block
        with self.tik_instance.new_stmt_scope():
            self.x_ub = self.tik_instance.Tensor(self.params_dtype, (x_aligned_num,), name="x_ub",
                scope=tik.scope_ubuf)
            self.tik_instance.data_move_pad(self.x_ub, self.x[task_id * self.params_axis * self.repeat_per_core +
                self.params_axis * index_i], 1, self.params_axis * self.params_dsize, 0, 0)
            self.res_ub = self.tik_instance.Tensor(self.params_dtype, (aligned_num,), name="res_ub",
                scope=tik.scope_ubuf)

            self.alloc_indices_ub(aligned_num, task_id * self.indices_axis * self.repeat_per_core +
                self.indices_axis * index_i)
            with self.tik_instance.for_range(self.indices_axis, aligned_num) as i:
                self.indices_ub[i].set_as(0)

            self.do_negative_indices(self.indices_ub, aligned_num)
            self.do_vgather(self.res_ub, self.x_ub, self.indices_ub, aligned_num)
            y_offset = self.repeat_per_core * self.indices_axis * task_id + index_i * self.indices_axis

            self.tik_instance.data_move_pad(self.y[y_offset], self.res_ub, 1, self.indices_axis *
                self.params_dsize, 0, 0)

    def process_last_axis_unaligned_diff_shape(self, task_id):
        """
            computational process for last axis unaligned
            diff shape version
        """
        indices_block = self.indices_axis // self.larger_num_each_block
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_block * self.larger_num_each_block,),
                                                   name="indices_ub", scope=tik.scope_ubuf)
        indices_tail_ub = self.tik_instance.Tensor(self.indices_dtype, (self.larger_num_each_block,),
                                                   name="indices_tail_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.indices_ub, self.indices[self.indices_axis * task_id], 0, 1,
                                    indices_block * self.larger_num_each_block // self.indices_num_each_block, 0, 0)
        self.tik_instance.data_move(indices_tail_ub,
                                    self.indices[self.indices_axis * (task_id + 1) - self.larger_num_each_block], 
                                    0, 1,
                                    self.larger_num_each_block // self.indices_num_each_block,
                                    0, 0)

        x_block = ceil_value(self.params_axis, self.params_num_each_block)
        self.x_ub = self.tik_instance.Tensor(self.params_dtype, (x_block * self.params_num_each_block,), name="x_ub",
                                             scope=tik.scope_ubuf)
        x_col = self.tik_instance.Scalar(dtype=Constant.INT64, name="x_col")
        self.cal_data_col_offset(task_id, x_col)
        self.tik_instance.data_move(self.x_ub, self.x[x_col * self.params_axis], 0, 1, x_block, 0, 0)
        self.res_ub = self.tik_instance.Tensor(self.params_dtype, (self.indices_axis,), name="res_ub",
                                               scope=tik.scope_ubuf)
        res_tail_ub = self.tik_instance.Tensor(self.params_dtype, (self.larger_num_each_block,), name="res_tail_ub",
                                               scope=tik.scope_ubuf)

        indices_value = self.tik_instance.Scalar(dtype=Constant.INT64, name="indices_value", init_value=0)
        with self.tik_instance.for_range(0, self.indices_axis - self.larger_num_each_block) as index_i:
            indices_value.set_as(self.indices_ub[index_i])
            indices_value = (indices_value + self.params_axis) % self.params_axis
            self.res_ub[index_i].set_as(self.x_ub[indices_value])

        indices_value_tail = self.tik_instance.Scalar(dtype=Constant.INT64, name="indices_value_tail", init_value=0)
        with self.tik_instance.for_range(0, self.larger_num_each_block) as index_i:
            indices_value_tail.set_as(indices_tail_ub[index_i])
            indices_value_tail = (indices_value_tail + self.params_axis) % self.params_axis
            res_tail_ub[index_i].set_as(self.x_ub[indices_value_tail])
        self.tik_instance.data_move(self.y[self.indices_axis * task_id], self.res_ub, 0, 1,
                                    self.indices_axis // self.params_num_each_block, 0, 0)
        self.tik_instance.data_move(self.y[self.indices_axis * (task_id + 1) - self.larger_num_each_block], 
                                    res_tail_ub, 0, 1, self.larger_num_each_block // self.params_num_each_block, 0, 0)
    
    def cal_data_col_offset(self, task_id, x_col):
        """
            calculate the current data column with each task_id
        """
        length_acc_cur = self.tik_instance.Scalar(dtype=Constant.INT64, name="length_acc_cur",
                                                  init_value=self.indices_shape_range_tensor[0])
        shape_indices_cur = self.tik_instance.Scalar(dtype=Constant.INT64, name="shape_indices_cur")
        with self.tik_instance.for_range(1, self.dims - 1) as dim:  # self.dims_indices >= 3 for diffshape
            shape_indices_cur.set_as(self.indices_shape_range_tensor[dim])
            length_acc_cur.set_as(length_acc_cur * shape_indices_cur)
        shape_data_next = self.tik_instance.Scalar(dtype=Constant.INT64, name="shape_x_next")
        x_col_next = self.tik_instance.Scalar(dtype=Constant.INT64, name="x_col_next")
        length_ordinate = self.tik_instance.Tensor(Constant.INT64, (self.dims - 1,), name="length_ordinate",
                                                   scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.dims - 1) as dim:
            shape_indices_cur.set_as(self.indices_shape_range_tensor[dim])
            length_acc_cur.set_as(length_acc_cur / shape_indices_cur)
            length_ordinate[dim].set_as(task_id // length_acc_cur % shape_indices_cur)
        x_col.set_as(length_ordinate[0])
        with self.tik_instance.for_range(0, self.dims - 2) as j:
            shape_data_next.set_as(self.params_shape_range_tensor[j + 1])
            x_col_next.set_as(length_ordinate[j + 1])
            x_col.set_as(x_col * shape_data_next + x_col_next)

    def process_all_indices(self, block_id, run_outer):
        """
            process all indices in a specific order
        """
        with self.tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
            run_outer(indices_num_offset, self.indices_row_num_once)

        with self.tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + self.indices_loop_num * \
                                 self.indices_row_num_once
            run_outer(indices_num_offset, self.indices_row_num_last)

        with self.tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.remaining_block_num)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + \
                                 block_id * self.indices_num_each_block * self.param_smaller_than_indices
            run_outer(indices_num_offset, self.indices_num_each_block * self.param_smaller_than_indices)

        with self.tik_instance.if_scope(tik.all(self.indices_num_remaining > 0,
                                                block_id == self.remaining_block_num, self.remaining_block_remain > 0)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + \
                                 block_id * self.indices_num_each_block * self.param_smaller_than_indices
            run_outer(indices_num_offset, self.remaining_block_remain)

    def compute_mode_x_large_indices_large(self, block_id, compute_func, get_output_gm_offset_func):
        """
        compute for large x and large indices
        params larger than carry_block_ub
        indices larger than the number contained in one block of each core
        same shape + diff shape version

        Parameters
        ----------
        block_id: id of ai core
        compute_func: a function which processes some of offsets
        get_output_gm_offset_func: a function which get output offsets

        Returns
        -------
        None
        """
        indices_ub_size, res_ub_size = self.compute_indices_res_ubsize(0)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_ub_size // self.indices_dsize,),
                                                   name="indices_ub", scope=tik.scope_ubuf)
        self.res_ub = self.tik_instance.Tensor(self.params_dtype, (res_ub_size // self.params_dsize,),
                                               name="res_ub", scope=tik.scope_ubuf)
        def run_outer(indices_num_offset, indices_num):
            self.tik_instance.data_move(self.indices_ub, self.indices[indices_num_offset], 0, 1,
                                        ceil_value(indices_num, self.indices_num_each_block), 0, 0)
            compute_func(self.y[indices_num_offset], indices_num_offset, indices_num, get_output_gm_offset_func)
        self.process_all_indices(block_id, run_outer)

    def compute_mode_x_small_indices_large(self, block_id, compute_func, get_output_gm_offset_func):
        """
        compute for small x large indices
        params less than cache_ub
        indices larger than the number contained in one block of each core
        same shape + diff shape version

        Parameters
        ----------
        block_id: id of ai core
        compute_func: a function which processes some of offsets
        get_output_gm_offset_func: a function which get output offsets
        Returns
        -------
        None
        """
        x_ub_size = self.params_total * self.params_dsize
        indices_ub_size, res_ub_size = self.compute_indices_res_ubsize(x_ub_size)
        self.x_ub = self.tik_instance.Tensor(self.params_dtype, (x_ub_size // self.params_dsize,),
                                             name="x_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.x_ub, self.x, 0, 1,
                                    ceil_value(self.params_total, self.params_num_each_block), 0, 0)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_ub_size // self.indices_dsize,),
                                              name="indices_ub", scope=tik.scope_ubuf)
        self.res_ub = self.tik_instance.Tensor(self.params_dtype, (res_ub_size // self.params_dsize,),
                                          name="res_ub", scope=tik.scope_ubuf)
        def run_outer(indices_num_offset, indices_num):
            self.tik_instance.data_move(self.indices_ub, self.indices[indices_num_offset], 0, 1,
                                        ceil_value(indices_num, self.indices_num_each_block),
                                        0, 0)
            compute_func(self.y[indices_num_offset], indices_num_offset, indices_num, get_output_gm_offset_func)

        self.process_all_indices(block_id, run_outer)

    def compute_mode_x_slice_indices_large(self, block_id, compute_func, get_output_gm_offset_func):
        """
        compute for x_slice and large indices
        params less than carry_block_ub
        params larger than cache_ub
        indices larger than the number contained in one block of each core
        same shape + diff shape version

        Parameters
        ----------
        block_id: id of ai core
        compute_func: a function which processes a lot of offsets
        get_output_gm_offset_func: a function which get output offsets
        Returns
        -------
        None
        """
        x_slice_ub_size = self.slice_thickness_once * self.params_dsize
        indices_ub_size, res_ub_size = self.compute_indices_res_ubsize(x_slice_ub_size)
        self.x_ub = self.tik_instance.Tensor(self.params_dtype, (x_slice_ub_size // self.params_dsize,),
                                             name="x_slice_ub", scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_ub_size // self.indices_dsize,),
                                                   name="indices_ub", scope=tik.scope_ubuf)
        self.res_ub = self.tik_instance.Tensor(self.params_dtype, (res_ub_size // self.params_dsize,),
                                               name="res_ub", scope=tik.scope_ubuf)
        cur_slice_thickness = self.tik_instance.Scalar(Constant.INT32, name="cur_slice_thickness")

        def run_outer(indices_num_offset, indices_num):
            with self.tik_instance.for_range(0, self.slice_num) as x_slice_id:
                with self.tik_instance.if_scope(x_slice_id < self.slice_num - 1):
                    cur_slice_thickness.set_as(self.slice_thickness_once)
                with self.tik_instance.elif_scope(x_slice_id == self.slice_num - 1):
                    cur_slice_thickness.set_as(self.slice_thickness_last)
                x_offset = x_slice_id * self.slice_thickness_once
                block_num = ceil_value(cur_slice_thickness, self.params_num_each_block)
                self.tik_instance.data_move(self.x_ub, self.x[x_offset], 0, 1, block_num, 0, 0)
                self.tik_instance.data_move(self.indices_ub, self.indices[indices_num_offset], 0, 1,
                                            ceil_value(indices_num, self.indices_num_each_block), 0, 0)
                compute_func(indices_num_offset, x_slice_id,
                             cur_slice_thickness, indices_num, get_output_gm_offset_func)
            self.tik_instance.data_move(self.y[indices_num_offset], self.res_ub, 0, 1,
                                        ceil_value(indices_num, self.params_num_each_block), 0, 0)

        self.process_all_indices(block_id, run_outer)

    # 'pylint: disable=too-many-arguments
    def compute_x_slice_less_cache(self, offset, x_slice_id, cur_slice_thickness, process_num,
                                   get_output_gm_offset_func):
        """
        move data x slice when shapes of indices and params are the same except the axis
        same shape + diff shape version

        Parameters
        ----------
        offset: offset of index
        x_slice_id: id of x_slice
        cur_slice_thickness: size of current param slice
        process_num: num of index
        get_output_gm_offset_func: a function which get output offsets

        Returns
        -------
        None
        """
        x_slice_offset = self.tik_instance.Scalar(Constant.INT64, name="x_slice_offset")
        indices_value = self.tik_instance.Scalar(self.indices_dtype, name="indices_value", init_value=0)
        with self.tik_instance.for_range(0, process_num) as index_i:
            with self.tik_instance.new_stmt_scope():
                indices_value.set_as(self.indices_ub[index_i])
                indices_value = (indices_value + self.params_axis) % self.params_axis
                gm_offset = get_output_gm_offset_func(index_i + offset, indices_value)
                with self.tik_instance.if_scope(
                        tik.all(gm_offset < (x_slice_id * self.slice_thickness_once + cur_slice_thickness),
                                gm_offset >= x_slice_id * self.slice_thickness_once)):
                    x_slice_offset.set_as(gm_offset - x_slice_id * self.slice_thickness_once)
                    self.res_ub[index_i].set_as(self.x_ub[x_slice_offset])

    def get_output_gm_offset_same(self, index, indices_value):
        """
        get x_gm offset according to the index and value of indices
        same shape version
        """
        tail_row = index % self.params_row
        loop_pre = index // (self.params_row * self.indices_axis)
        gm_offset = (loop_pre * self.params_axis + indices_value) * self.params_row + tail_row
        return gm_offset
    
    def get_output_gm_offset_dif(self, index, indices_value):
        """
        get x_gm offset according to the index and value of indices
        diff shape version
        """
        index_tensor = self.output_indices_index(index)
        index_tensor[self.axis].set_as(indices_value)
        gm_offset = self.output_param_offset(index_tensor)
        return gm_offset

    def compute_x_larger_cache(self, y_dst, offset, process_num, get_output_gm_offset_func):
        """
        move data when x larger than carry_block_ub
        same shape + diff shape version

        Parameters
        ----------
        y_dst: output
        offset: offset of index
        process_num: num of index
        get_output_gm_offset_func: a function which get output offsets

        Returns
        -------
        None
        """
        indices_value = self.tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
        with self.tik_instance.for_range(0, process_num) as index_i:
            with self.tik_instance.new_stmt_scope():
                indices_value.set_as(self.indices_ub[index_i])
                indices_value = (indices_value + self.params_axis) % self.params_axis
                gm_offset = get_output_gm_offset_func(index_i + offset, indices_value)
                block_ub = self.tik_instance.Tensor(self.params_dtype, (self.params_num_each_block,),
                                                    name="block_ub", scope=tik.scope_ubuf)
                with self.tik_instance.if_scope(
                        gm_offset + self.params_num_each_block > self.params_total):
                    self.tik_instance.data_move(block_ub,
                                                self.x[1 + gm_offset - self.params_num_each_block],
                                                0, 1, 1, 0, 0)
                    self.res_ub[index_i].set_as(block_ub[self.params_num_each_block - 1])
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(block_ub, self.x[gm_offset], 0, 1, 1, 0, 0)
                    self.res_ub[index_i].set_as(block_ub)
        self.tik_instance.data_move(y_dst, self.res_ub, 0, 1,
                                    ceil_value(process_num, self.params_num_each_block), 0, 0)
    
    def compute_x_less_cache(self, y_dst, offset, process_num, get_output_gm_offset_func):
        """
        move data x less than cache
        same shape + diff shape version

        Parameters
        ----------
        y_dst: output
        offset: offset of index
        process_num: num of index
        get_output_gm_offset_func: a function which get output offsets

        Returns
        -------
        None
        """
        indices_value = self.tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
        with self.tik_instance.for_range(0, process_num) as index_i:  
            with self.tik_instance.new_stmt_scope():
                indices_value.set_as(self.indices_ub[index_i])
                indices_value = (indices_value + self.params_axis) % self.params_axis
                gm_offset = get_output_gm_offset_func(index_i + offset, indices_value)
                self.res_ub[index_i].set_as(self.x_ub[gm_offset])
        self.tik_instance.data_move(y_dst, self.res_ub, 0, 1,
                                    ceil_value(process_num, self.params_num_each_block), 0, 0)

    def gather_elements_compute(self):
        """
        compute of gather_elements

        Parameters
        ----------
        None

        Returns
        -------
        compile info
        """
        self.x = self.tik_instance.Tensor(self.params_dtype, self.x_shape, name="x", scope=tik.scope_gm)
        self.indices = self.tik_instance.Tensor(self.indices_dtype,
                                                self.indices_shape, name="indices", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype,
                                                  (Constant.TILING_ARG_NUM, ), name="tiling_gm", scope=tik.scope_gm)
        self.y = self.tik_instance.Tensor(self.y_dtype, self.y_shape, name="y", scope=tik.scope_gm)
        self.gather_elements_compute_tiling()

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                                            "ub_size": self.ub_size,
                                                            "params_dsize": self.params_dsize,
                                                            "support_vgather": self.support_vgather,
                                                            "infer_vgather": self.infer_vgather,
                                                            "indices_dsize": self.indices_dsize})

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x, self.indices),
                                   outputs=(self.y,),
                                   flowtable=(self.tiling_gm,), enable_l2=True, config=opt_config)
        return self.tik_instance


@register_operator("GatherElements")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def gather_elements(x_dict, indices_dict, y_dict, dim=0, kernel_name="GatherElements"):
    """
    gather_elements inferface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    indices_dict: input indices shape, dtype and range
    y_dict: output shape, dtype and range
    dim: which dim to gather on, attr
    kernel_name: kernel name of gather_elements op

    Returns
    -------
    compile info
    """
    obj = GatherElements(x_dict, indices_dict, y_dict, dim, kernel_name)
    return obj.gather_elements_compute()