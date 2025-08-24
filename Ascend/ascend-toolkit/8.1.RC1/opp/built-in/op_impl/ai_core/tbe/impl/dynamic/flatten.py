# Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
flatten
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_tik_comm_func import ceil_div


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max int32
    MAX_INT32 = 2**31 - 1
    # tiling param num
    TILING_ARG_NUM = 16


# 'pylint: disable=unused-argument,invalid-name
def get_op_support_info(x, y, axis=1, kernel_name="flatten"):
    """
    get_op_support_info
    """
    axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=too-many-instance-attributes
class Flatten:
    """Performs flatten on input tensor
    """

    def __init__(self, src, dst, kernel_name):
        """Init flatten parameters
        """
        # reserved ub size
        reserved_ub_size = 8 * 1024
        self.tik_instance = tik.Tik(block_size=tbe_platform.get_block_size())
        self.src_dtype = src.get("dtype").lower()
        self.dst_dtype = dst.get("dtype").lower()
        self.src_dtype = "int8" if self.src_dtype == "bool" else self.src_dtype
        self.dst_dtype = "int8" if self.dst_dtype == "bool" else self.dst_dtype
        # check dtype
        para_check.check_dtype(self.src_dtype, ("float16", "float32", "int8", "int32", "int64", "uint8", "int16",
                                                "uint16", "uint32", "uint64", "bfloat16"),
                               param_name="src")
        para_check.check_dtype(self.dst_dtype, ("float16", "float32", "int8", "int32", "int64", "uint8", "int16",
                                                "uint16", "uint32", "uint64", "bfloat16"),
                               param_name="dst")
        if self.src_dtype != self.dst_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal("Flatten", "src", "dst", self.src_dtype,
                                                                  self.dst_dtype)
        self.kernel_name = kernel_name
        # `get dtype size, float16 size = 2 byte / float32 size = 4 byte`
        self.dtype_size = tbe_platform.get_bit_len(self.src_dtype) // 8

        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.data_len_one_block = tbe_platform.get_block_size() // self.dtype_size
        self.ub_availble = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - reserved_ub_size
        self.ub_max_data = self.ub_availble // self.dtype_size
        self.copy_segment = self.ub_max_data // 2
        self.copy_segment = (ceil_div(self.copy_segment, self.data_len_one_block) - 1) * self.data_len_one_block

        # input and output tensor in gm
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.src_gm = self.tik_instance.Tensor(self.src_dtype, (Constant.MAX_INT32,), name="src_gm", scope=tik.scope_gm)
        self.dst_gm = self.tik_instance.Tensor(self.dst_dtype, (Constant.MAX_INT32,), name="dst_gm", scope=tik.scope_gm)
        self.tiling_ub = None

        # tiling args
        self.core_data = self.tik_instance.Scalar("int64", name="core_data")
        self.core_used = self.tik_instance.Scalar("int64", name="core_used")
        self.copy_loop = self.tik_instance.Scalar("int64", name="copy_loop")
        self.copy_tail = self.tik_instance.Scalar("int64", name="copy_tail")
        self.last_copy_loop = self.tik_instance.Scalar("int64", name="last_copy_loop")
        self.last_copy_tail = self.tik_instance.Scalar("int64", name="last_copy_tail")
        self.new_core_num = self.tik_instance.Scalar("int64", name="new_core_num", init_value=self.core_num)

        self.support_move_pad = tbe_platform.api_check_support("tik.data_move_pad", self.src_dtype)

    def copy_only(self, core_index, loop_num, tail_num):
        """Only execute move in and move out
        """
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.copy_in_and_out(core_index, loop_idx, self.copy_segment)
        with self.tik_instance.if_scope(tail_num > 0):
            self.copy_in_and_out(core_index, loop_num, tail_num)

    def copy_in_and_out(self, core_index, loop_idx, loop_len):
        """Copy in and out
        """
        offset = core_index * self.core_data + loop_idx * self.copy_segment
        bust_len = self._get_ceil_int(loop_len, self.data_len_one_block)
        bust_len_pad = loop_len * self.dtype_size
        data_ub = self.tik_instance.Tensor(self.dst_dtype, [self.copy_segment], name="data_ub", scope=tik.scope_ubuf)
        if self.support_move_pad:
            self.tik_instance.data_move_pad(data_ub, self.src_gm[offset], 1, bust_len_pad, 0, 0)
            self.tik_instance.data_move_pad(self.dst_gm[offset], data_ub, 1, bust_len_pad, 0, 0)
        elif tbe_platform.api_check_support("tik.data_move_pad") and self.src_dtype in ("int64", "uint64"):
            s8_src_gm = self.src_gm.reinterpret_cast_to("int8")
            s8_data_ub = data_ub.reinterpret_cast_to("int8")
            s8_dst_gm = self.dst_gm.reinterpret_cast_to("int8")
            self.tik_instance.data_move_pad(s8_data_ub, s8_src_gm[offset * 8], 1, bust_len_pad, 0, 0)
            self.tik_instance.data_move_pad(s8_dst_gm[offset * 8], s8_data_ub, 1, bust_len_pad, 0, 0)
        else:
            self.tik_instance.data_move(data_ub, self.src_gm[offset], 0, 1, bust_len, 0, 0)
            self.tik_instance.data_move(self.dst_gm[offset], data_ub, 0, 1, bust_len, 0, 0)

    def flatten_compute(self):
        """Flatten compute
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
        self._tiling_args()
        # core process
        loop_num = self.tik_instance.Scalar("int64", name="loop_num")
        tail_num = self.tik_instance.Scalar("int64", name="tail_num")
        with self.tik_instance.for_range(0, self.new_core_num, block_num=self.new_core_num) as core_index:
            with self.tik_instance.if_scope(core_index < (self.core_used - 1)):
                loop_num.set_as(self.copy_loop)
                tail_num.set_as(self.copy_tail)
                self.copy_only(core_index, loop_num, tail_num)
            with self.tik_instance.elif_scope(core_index == (self.core_used - 1)):
                loop_num.set_as(self.last_copy_loop)
                tail_num.set_as(self.last_copy_tail)
                self.copy_only(core_index, loop_num, tail_num)

        # add compile info
        tbe_context.get_context().add_compile_info("vars", {
            "ub_size": self.copy_segment,
            "core_num": self.core_num,
            "block_size": self.data_len_one_block
        })

        # build cce
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.src_gm],
                                   outputs=[self.dst_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    def _tiling_args(self):
        """Get runtime tiling parameters from tiling
        """
        # read tiling int64 scalar
        self.core_data.set_as(self.tiling_ub[0])
        self.core_used.set_as(self.tiling_ub[1])
        self.copy_loop.set_as(self.tiling_ub[2])
        self.copy_tail.set_as(self.tiling_ub[3])
        self.last_copy_loop.set_as(self.tiling_ub[4])
        self.last_copy_tail.set_as(self.tiling_ub[5])
        self.new_core_num.set_as(self.tiling_ub[6])

    # 'pylint: disable = invalid-name,unused-argument,too-many-instance-attributes
    def _get_ceil_int(self, int1, int2):
        """Get ceil
        """
        result = self.tik_instance.Scalar("int64", name="result")
        with self.tik_instance.if_scope(int1 % int2 == 0):
            result.set_as(int1 // int2)
        with self.tik_instance.else_scope():
            result.set_as(int1 // int2 + 1)
        return result


# 'pylint: disable=unused-argument,invalid-name
@register_operator("Flatten")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def flatten(x, y, axis=1, kernel_name="flatten"):
    """return a copy of the tensor collapsed into one dimension.

    Parameters
    ----------
    x : dict
        shape and dtype of input.
    y : dict
        shape and dtype of output.
    kernel_name : str
        kernel name, default value is "flatten"

    Returns
    -------
    None
    """
    obj = Flatten(x, y, kernel_name)
    obj.flatten_compute()
