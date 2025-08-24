
# Copyright 2022 Huawei Technologies Co., Ltd
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
dynamic rotary_mul_grad
"""
from functools import reduce as functools_reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl import constant_util as constant


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    RESERVED_UB = 20480
    INT8_BLOCK = 32
    INT32_BLOCK = 8
    INT64_BLOCK = 4
    BYTE_SIZE = 8
    BLOCK_NUM = 16
    # the number of blocks skipped per repeat
    STRIDE_EIGHT = 8
    # the number of blocks skipped per repeat
    STRIDE_FOUR = 4
    # the number of blocks skipped per repeat
    STRIDE_ONE = 1
    # the number of tenosr for fp32
    LIST_NUMBER = 4
    # the number of tenosr for fp16
    LIST_NUMBER_LARGE = 8
    # the number of transposes per repeat
    NUMBER_TWO = 2
    # max int32
    MAX_INT32 = 2 ** 31 - 1
    # ting param num
    TILING_ARG_NUM = 19
    # MAX loop number
    MAX_LOOP = 2000
    # MAXNUM for fp32
    MASK_64 = 64


# 'pylint: disable=useless-object-inheritance,too-many-instance-attributes
class RotaryBackward(object):

    # 'pylint: disable=too-many-arguments,invalid-name
    def __init__(self, xq, x1, x2, grad, xq_grad, x1_grad, x2_grad, need_backward, kernel_name, impl_mode):
        """
        Init RotaryBackward base parameters

        Parameters
        ----------
        x1 : dict
            shape and dtype of input x1
        xq : dict
            shape and dtype of xq
        x2 : dict
            shape and dtype of x2
        grad : dict
            shape and dtype of grad
        x1_grad : dict
            shape and dtype of input x1_grad
        xq_grad : dict
            shape and dtype of xq_grad
        x2_grad : dict
            shape and dtype of x2_grad
        kernel_name : str
            kernel name, default value is "rotary_mul_grad"

        Returns
        -------
        None
        """
        byte_size = 8
        block_number_fp16 = 32
        self.tik_instance = tik.Tik()
        self.x1_dtype = x1.get("dtype").lower()
        self.xq_dtype = xq.get("dtype").lower()
        self.x2_dtype = x2.get("dtype").lower()
        self.grad_dtype = grad.get("dtype").lower()
        self.kernel_name = kernel_name
        self.need_backward = need_backward
        self.x_dtype_bytes_size = tbe_platform.get_bit_len(self.x1_dtype) // byte_size
        self.x_data_each_block = constant.BLOCK_SIZE // self.x_dtype_bytes_size
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.each_repeat_block_number = block_number_fp16
        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.mode = impl_mode
        if (self.x1_dtype == "float16" and self.mode == "high_precision") or self.x1_dtype == "bfloat16":
            self.ub_max_size = (self.total_ub - Constant.RESERVED_UB) // Constant.LIST_NUMBER_LARGE // Constant.NUMBER_TWO
        elif self.x1_dtype == "float32":
            self.ub_max_size = (self.total_ub - Constant.RESERVED_UB) // Constant.LIST_NUMBER_LARGE
        else:
            self.ub_max_size = (self.total_ub - Constant.RESERVED_UB) // Constant.LIST_NUMBER
        self.available_ub_size = self.tik_instance.Scalar("int32", name="available_ub_size", init_value=2048)
        self.mask_num = 64 if self.x1_dtype == "float32" else 128
        self.cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        self.round_mode_fp16 = '' if self.cce_product == "Ascend310P" else 'round'
        # init scalar data
        self.xq_src_stride_chunk = self.tik_instance.Scalar(dtype="int32", name="xq_src_stride_chunk")
        self.x2_src_stride_chunk = self.tik_instance.Scalar(dtype="int32", name="x2_src_stride_chunk")
        self.x1_nburst = self.tik_instance.Scalar(dtype="int32", name="x1_nburst")
        self.nburst = self.tik_instance.Scalar(dtype="int32", name="nburst")
        self.x1_burst = self.tik_instance.Scalar(dtype="int32", name="x1_burst")
        self.burst = self.tik_instance.Scalar(dtype="int32", name="burst")
        self.burst_chunk = self.tik_instance.Scalar(dtype="int32", name="burst_chunk")
        self.nburst_chunk = self.tik_instance.Scalar(dtype="int32", name="nburst_chunk")
        self.src_stride = self.tik_instance.Scalar(dtype="int32", name="src_stride")
        self.first_reduce = self.tik_instance.Scalar(dtype="int32", name="first_reduce")
        self.second_reduce = self.tik_instance.Scalar(dtype="int32", name="second_reduce")
        self.ub_loop_gap = self.tik_instance.Scalar(dtype="int32", name="ub_loop_gap")

        # init tiling data
        self.x1_num = self.tik_instance.Scalar("int32", name="x1_num")
        self.core_data = self.tik_instance.Scalar("int32", name="core_data")
        self.core_used = self.tik_instance.Scalar("int32", name="core_used")
        self.copy_loop = self.tik_instance.Scalar("int32", name="copy_loop")
        self.copy_tail = self.tik_instance.Scalar("int32", name="copy_tail")
        self.last_copy_loop = self.tik_instance.Scalar("int32", name="last_copy_loop")
        self.last_copy_tail = self.tik_instance.Scalar("int32", name="last_copy_tail")
        self.dim_0 = self.tik_instance.Scalar("int32", name="dim_0")
        self.dim_1 = self.tik_instance.Scalar("int32", name="dim_1")
        self.dim_2 = self.tik_instance.Scalar("int32", name="dim_2")
        self.dim_3 = self.tik_instance.Scalar("int32", name="dim_3")
        self.dim_4 = self.tik_instance.Scalar("int32", name="dim_4")
        self.dim_5 = self.tik_instance.Scalar("int32", name="dim_5")
        self.dim_6 = self.tik_instance.Scalar("int32", name="dim_6")
        self.dim_7 = self.tik_instance.Scalar("int32", name="dim_7")
        self.reserver_number = self.tik_instance.Scalar("int32", name="reserver_number")
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode", init_value=0)
        self.tiling_core_num = self.tik_instance.Scalar(dtype="int32", name="tiling_core_num")
        self.ub_per_reser_number = self.tik_instance.Scalar(dtype="int32", name="ub_per_reser_number")

        # init gm data
        self.xq_gm = self.tik_instance.Tensor(self.xq_dtype, [Constant.MAX_INT32],
                                              name="xq_gm", scope=tik.scope_gm)
        self.x1_gm = self.tik_instance.Tensor(self.x1_dtype, [Constant.MAX_INT32],
                                              name="x1_gm", scope=tik.scope_gm)
        self.x2_gm = self.tik_instance.Tensor(self.x2_dtype, [Constant.MAX_INT32],
                                              name="x2_gm", scope=tik.scope_gm)
        self.grad_gm = self.tik_instance.Tensor(self.grad_dtype, [Constant.MAX_INT32],
                                                name="grad_gm", scope=tik.scope_gm)
        self.xq_grad_gm = self.tik_instance.Tensor(self.xq_dtype, [Constant.MAX_INT32], name="xq_grad_gm",
                                                   scope=tik.scope_gm, is_atomic_add=True)
        self.x1_grad_gm = self.tik_instance.Tensor(self.x1_dtype, [Constant.MAX_INT32], name="x1_grad_gm",
                                                   scope=tik.scope_gm, is_atomic_add=True)
        self.x2_grad_gm = self.tik_instance.Tensor(self.x2_dtype, [Constant.MAX_INT32], name="x2_grad_gm",
                                                   scope=tik.scope_gm, is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
    
    def get_tiling_args(self, tiling_ub):
        """
        get runtime tiling params from tiling

        Parameters
        ----------

        Returns
        -------
        None
        """
        # read tiling int32 scalar
        self.x1_num.set_as(tiling_ub[0])
        self.core_data.set_as(tiling_ub[1])
        self.core_used.set_as(tiling_ub[2])
        self.copy_loop.set_as(tiling_ub[3])
        self.copy_tail.set_as(tiling_ub[4])
        self.last_copy_loop.set_as(tiling_ub[5])
        self.last_copy_tail.set_as(tiling_ub[6])
        self.dim_0.set_as(tiling_ub[7])
        self.dim_1.set_as(tiling_ub[8])
        self.dim_2.set_as(tiling_ub[9])
        self.dim_3.set_as(tiling_ub[10])
        self.dim_4.set_as(tiling_ub[11])
        self.dim_5.set_as(tiling_ub[12])
        self.dim_6.set_as(tiling_ub[13])
        self.dim_7.set_as(tiling_ub[14])
        self.reserver_number.set_as(tiling_ub[15])
        self.available_ub_size.set_as((tiling_ub[16]))
        self.ub_per_reser_number.set_as((tiling_ub[17]))
        self.tiling_mode.set_as((tiling_ub[18]))

    def calculte_burst(self):
        with self.tik_instance.if_scope(self.tiling_mode == 0):
            self.x1_nburst.set_as(1)
            self.x1_burst.set_as((self.ub_per_reser_number * self.dim_3 +\
                                 self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst.set_as(
                (self.ub_per_reser_number * self.dim_3 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.nburst.set_as(1)
            self.nburst_chunk.set_as(self.ub_per_reser_number)
            self.src_stride.set_as(0)
            self.xq_src_stride_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.x2_src_stride_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.first_reduce.set_as(1)
            self.second_reduce.set_as(1)
            self.ub_loop_gap.set_as(self.dim_3)
        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.x1_nburst.set_as(1)
            self.x1_burst.set_as((self.ub_per_reser_number * self.dim_3 +\
                                 self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst.set_as((self.dim_3 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.nburst.set_as(self.ub_per_reser_number)
            self.nburst_chunk.set_as(self.ub_per_reser_number)
            self.src_stride.set_as(((self.dim_2 - 1) * self.dim_3 +\
                                   self.x_data_each_block - 1) // self.x_data_each_block)
            self.xq_src_stride_chunk.set_as(((self.dim_2 - 1) * self.dim_3 + self.dim_3//2 +\
                                            self.x_data_each_block - 1) // self.x_data_each_block)
            self.x2_src_stride_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.first_reduce.set_as(self.dim_0)
            self.second_reduce.set_as(self.dim_2)
            self.ub_loop_gap.set_as(self.dim_1 * self.dim_2 * self.dim_3)
        with self.tik_instance.elif_scope(self.tiling_mode == 2):
            self.x1_nburst.set_as(1)
            self.x1_burst.set_as((self.ub_per_reser_number * self.dim_3 +\
                                 self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst.set_as((self.ub_per_reser_number*self.dim_3 +\
                              self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.nburst.set_as(1)
            self.nburst_chunk.set_as(self.ub_per_reser_number)
            self.src_stride.set_as(0)
            self.xq_src_stride_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.x2_src_stride_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.first_reduce.set_as(self.dim_0 * self.dim_1)
            self.second_reduce.set_as(1)
            self.ub_loop_gap.set_as(self.dim_2 * self.dim_3)
        with self.tik_instance.elif_scope(self.tiling_mode == 3):
            self.x1_nburst.set_as(1)
            self.x1_burst.set_as((self.ub_per_reser_number*self.dim_3 +\
                                 self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst.set_as((self.dim_3 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.nburst.set_as(self.ub_per_reser_number)
            self.nburst_chunk.set_as(self.ub_per_reser_number)
            self.src_stride.set_as(((self.dim_2*self.dim_1 - 1) * self.dim_3 +\
                                    self.x_data_each_block - 1) // self.x_data_each_block)
            self.xq_src_stride_chunk.set_as(((self.dim_2*self.dim_1 - 1) * self.dim_3 + self.dim_3//2 +\
                                            self.x_data_each_block - 1) // self.x_data_each_block)
            self.x2_src_stride_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.first_reduce.set_as(1)
            self.second_reduce.set_as(self.dim_1 * self.dim_2)
            self.ub_loop_gap.set_as(1)
        with self.tik_instance.elif_scope(self.tiling_mode == 4):
            self.x1_nburst.set_as(1)
            self.x1_burst.set_as((self.ub_per_reser_number*self.dim_3 +\
                                 self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst.set_as((self.dim_3 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.burst_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.nburst.set_as(self.ub_per_reser_number)
            self.nburst_chunk.set_as(self.ub_per_reser_number)
            self.src_stride.set_as(((self.dim_2 - 1) * self.dim_3 +\
                                   self.x_data_each_block - 1) // self.x_data_each_block)
            self.xq_src_stride_chunk.set_as(((self.dim_2 - 1) * self.dim_3 + self.dim_3//2 +\
                                            self.x_data_each_block - 1) // self.x_data_each_block)
            self.x2_src_stride_chunk.set_as((self.dim_3 // 2 + self.x_data_each_block - 1) // self.x_data_each_block)
            self.first_reduce.set_as(1)
            self.second_reduce.set_as(self.dim_2)
            self.ub_loop_gap.set_as(1)

    def rotary_compute_fp16(self, loop_input, loop_input_x1, burst_normal, cal_num):
        self.tik_instance.set_atomic_add(self.x1_dtype)
        self.ub_per_reser_number.set_as(cal_num//self.dim_3)
        x1_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_per_reser_number, self.dim_3),
                                              name="x1_ub_fp32", scope=tik.scope_ubuf)
        xq_grad_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_per_reser_number, self.dim_3),
                                                   name="xq_grad_ub_fp32", scope=tik.scope_ubuf)
        xq_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_per_reser_number, self.dim_3,),
                                              name="xq_ub_fp32", scope=tik.scope_ubuf)
        xq_ub2_fp32 = self.tik_instance.Tensor("float32", (self.ub_per_reser_number, self.dim_3,),
                                               name="xq_ub2_fp32", scope=tik.scope_ubuf)
        ub_temp_fp32 = self.tik_instance.Tensor("float32", (self.ub_per_reser_number, self.dim_3),
                                                name="ub_temp_fp32", scope=tik.scope_ubuf)
        ub_temp2_fp32 = self.tik_instance.Tensor("float32", (self.ub_per_reser_number, self.dim_3),
                                                name="ub_temp2_fp32", scope=tik.scope_ubuf)
        x2_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_per_reser_number, self.dim_3),
                                              name="x2_ub_fp32", scope=tik.scope_ubuf)
        grad_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_per_reser_number, self.dim_3),
                                                name="grad_ub_fp32", scope=tik.scope_ubuf)
        x1_ub = self.tik_instance.Tensor(self.x1_dtype, (self.ub_per_reser_number, self.dim_3),
                                         name="x1_ub", scope=tik.scope_ubuf)
        xq_grad_ub = self.tik_instance.Tensor(self.xq_dtype, (self.ub_per_reser_number, self.dim_3),
                                              name="xq_grad_ub", scope=tik.scope_ubuf)
        xq_ub = self.tik_instance.Tensor(self.xq_dtype, (self.ub_per_reser_number, self.dim_3,),
                                         name="xq_ub", scope=tik.scope_ubuf)
        x2_ub = self.tik_instance.Tensor(self.x1_dtype, (self.ub_per_reser_number, self.dim_3),
                                         name="x2_ub", scope=tik.scope_ubuf)
        grad_ub = self.tik_instance.Tensor(self.x1_dtype, (self.ub_per_reser_number, self.dim_3),
                                           name="grad_ub", scope=tik.scope_ubuf)
        ub_temp = self.tik_instance.Tensor(self.x1_dtype, (self.ub_per_reser_number, self.dim_3),
                                           name="ub_temp", scope=tik.scope_ubuf)
        self.calculte_burst()
        repeat_time_conv = (self.ub_per_reser_number * self.dim_3) // Constant.MASK_64
        burst_chunk_conv = (self.dim_3 // 2 + 8 - 1) // 8
        self.tik_instance.data_move(x1_ub, self.x1_gm[loop_input_x1],
                                    constant.SID, 1,
                                    self.x1_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        self.tik_instance.data_move(x2_ub, self.x2_gm[loop_input_x1],
                                    constant.SID, 1,
                                    self.x1_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        self.tik_instance.vec_conv(Constant.MASK_64, '', x1_ub_fp32, x1_ub,
                                   repeat_time_conv, Constant.STRIDE_EIGHT, Constant.STRIDE_FOUR)
        self.tik_instance.vec_conv(Constant.MASK_64, '', x2_ub_fp32, x2_ub,
                                   repeat_time_conv, Constant.STRIDE_EIGHT, Constant.STRIDE_FOUR)
        self.tik_instance.vec_dup(64, xq_ub_fp32, 0, repeat_time_conv, Constant.STRIDE_EIGHT)
        self.tik_instance.vec_dup(64, xq_ub2_fp32, 0, repeat_time_conv, Constant.STRIDE_EIGHT)
        with self.tik_instance.for_range(0, self.first_reduce, thread_num=2) as ub_loop_idx:
            ub_loop_addr = ub_loop_idx * self.ub_loop_gap + loop_input
            with self.tik_instance.for_range(0, self.second_reduce) as reduce_add_loop_idx:
                ub_loop_addr = ub_loop_addr + reduce_add_loop_idx * self.dim_3
                self.tik_instance.data_move(grad_ub, self.grad_gm[ub_loop_addr],
                                            constant.SID, self.nburst,
                                            self.burst, self.src_stride, constant.STRIDE_ZERO)
                self.tik_instance.vec_conv(Constant.MASK_64, '', grad_ub_fp32, grad_ub,
                                           repeat_time_conv, Constant.STRIDE_EIGHT, Constant.STRIDE_FOUR)
                if self.need_backward is True:
                    self.tik_instance.data_move(ub_temp, self.xq_gm[ub_loop_addr],
                                                constant.SID, self.nburst,
                                                self.burst, self.src_stride, constant.STRIDE_ZERO)
                    self.tik_instance.vec_conv(Constant.MASK_64, '', ub_temp_fp32, ub_temp,
                                               repeat_time_conv, Constant.STRIDE_EIGHT, Constant.STRIDE_FOUR)
                    
                    self.tik_instance.data_move(ub_temp2_fp32[self.dim_3//2], ub_temp_fp32,
                                                constant.SID, self.nburst_chunk,
                                                burst_chunk_conv, burst_chunk_conv, burst_chunk_conv)
                    self.tik_instance.vmuls(Constant.MASK_64, ub_temp_fp32, ub_temp_fp32,
                                            -1, repeat_time_conv, 1, 1, 8, 8)
                    self.tik_instance.data_move(ub_temp2_fp32, ub_temp_fp32[self.dim_3//2],
                                                constant.SID, self.nburst_chunk,
                                                burst_chunk_conv, burst_chunk_conv, burst_chunk_conv)
                    # reduce_add x2_grad
                    self.tik_instance.vec_mul(Constant.MASK_64, ub_temp2_fp32, grad_ub_fp32,
                                              ub_temp2_fp32, repeat_time_conv, 8, 8, 8)
                    self.tik_instance.vec_add(Constant.MASK_64, xq_ub2_fp32, ub_temp2_fp32,
                                              xq_ub2_fp32, repeat_time_conv, 8, 8, 8)
                    # x1_grad = xq * grad
                    self.tik_instance.vmuls(Constant.MASK_64, ub_temp_fp32, ub_temp_fp32,
                                            -1, repeat_time_conv, 1, 1, 8, 8)
                    self.tik_instance.vec_mul(Constant.MASK_64, ub_temp_fp32, grad_ub_fp32,
                                              ub_temp_fp32, repeat_time_conv, 8, 8, 8)
                    self.tik_instance.vec_add(Constant.MASK_64, xq_ub_fp32, ub_temp_fp32,
                                              xq_ub_fp32, repeat_time_conv, 8, 8, 8)

                # xq_grad = x1 * grad + [1,-1]*x2*grad
                # move x1*grad
                self.tik_instance.vec_mul(Constant.MASK_64, xq_grad_ub_fp32, grad_ub_fp32,
                                          x1_ub_fp32, repeat_time_conv, 8, 8, 8)
                # conv fp32 to bfloat16 or float16
                self.tik_instance.vec_conv(Constant.MASK_64, self.round_mode_fp16, xq_grad_ub,
                                           xq_grad_ub_fp32, repeat_time_conv, Constant.STRIDE_FOUR, 8)
                self.tik_instance.data_move(self.xq_grad_gm[ub_loop_addr], xq_grad_ub,
                                            constant.SID, self.nburst,
                                            self.burst, constant.STRIDE_ZERO, self.src_stride)
                # move [-1,1]*x2*grad
                self.tik_instance.vec_mul(Constant.MASK_64, xq_grad_ub_fp32, grad_ub_fp32,
                                          x2_ub_fp32, repeat_time_conv, 8, 8, 8)
                self.tik_instance.vec_conv(Constant.MASK_64, self.round_mode_fp16, xq_grad_ub, xq_grad_ub_fp32,
                                           repeat_time_conv, Constant.STRIDE_FOUR, 8)
                self.tik_instance.data_move(self.xq_grad_gm[ub_loop_addr], xq_grad_ub[self.dim_3 // 2],
                                            constant.SID, self.nburst_chunk,
                                            self.burst_chunk, self.x2_src_stride_chunk, self.xq_src_stride_chunk)
                self.tik_instance.vmuls(Constant.MASK_64, xq_grad_ub_fp32,
                                        xq_grad_ub_fp32, -1, repeat_time_conv, 1, 1, 8, 8)
                self.tik_instance.vec_conv(Constant.MASK_64, self.round_mode_fp16, xq_grad_ub, xq_grad_ub_fp32,
                                           repeat_time_conv, Constant.STRIDE_FOUR, 8)
                self.tik_instance.data_move(self.xq_grad_gm[ub_loop_addr + self.dim_3 // 2], xq_grad_ub,
                                            constant.SID, self.nburst_chunk,
                                            self.burst_chunk, self.x2_src_stride_chunk, self.xq_src_stride_chunk)
        if self.need_backward is True:
            self.tik_instance.vec_conv(Constant.MASK_64, self.round_mode_fp16, xq_ub, xq_ub_fp32, repeat_time_conv,
                                    Constant.STRIDE_FOUR, 8)
            self.tik_instance.data_move(self.x1_grad_gm[loop_input_x1], xq_ub,
                                        constant.SID, constant.DEFAULT_NBURST,
                                        self.x1_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)

            # x2_grad = [-1,1]*xq*grad
            self.tik_instance.vec_conv(Constant.MASK_64, self.round_mode_fp16, xq_ub, xq_ub2_fp32, repeat_time_conv,
                                        Constant.STRIDE_FOUR, 8)
            self.tik_instance.data_move(self.x2_grad_gm[loop_input_x1], xq_ub,
                                        constant.SID, constant.DEFAULT_NBURST,
                                        self.x1_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)      

    def rotary_need_broadcast(self, loop_input, loop_input_x1, burst_normal, cal_num):
        self.tik_instance.set_atomic_add(self.x1_dtype)
        self.ub_per_reser_number.set_as(cal_num//self.dim_3)
        self.calculte_burst()
        x1_ub = self.tik_instance.Tensor(self.x1_dtype, (self.ub_per_reser_number, self.dim_3),
                                         name="x1_ub", scope=tik.scope_ubuf)
        xq_grad_ub = self.tik_instance.Tensor(self.xq_dtype, (self.ub_per_reser_number, self.dim_3),
                                              name="xq_grad_ub", scope=tik.scope_ubuf)
        xq_ub = self.tik_instance.Tensor(self.xq_dtype, (self.ub_per_reser_number, self.dim_3,),
                                         name="xq_ub", scope=tik.scope_ubuf)
        ub_temp = self.tik_instance.Tensor(self.xq_dtype, (self.ub_per_reser_number, self.dim_3),
                                           name="ub_temp", scope=tik.scope_ubuf)
        ub_temp2 = self.tik_instance.Tensor(self.xq_dtype, (self.ub_per_reser_number, self.dim_3),
                                           name="ub_temp2", scope=tik.scope_ubuf)
        xq_ub2 = self.tik_instance.Tensor(self.xq_dtype, (self.ub_per_reser_number, self.dim_3),
                                           name="xq_ub2", scope=tik.scope_ubuf)
        x2_ub = self.tik_instance.Tensor(self.x1_dtype, (self.ub_per_reser_number, self.dim_3),
                                         name="x2_ub", scope=tik.scope_ubuf)
        grad_ub = self.tik_instance.Tensor(self.x1_dtype, (self.ub_per_reser_number, self.dim_3),
                                           name="grad_ub", scope=tik.scope_ubuf)

        repeat_time = (self.ub_per_reser_number * self.dim_3) // self.mask_num
        self.tik_instance.data_move(x1_ub, self.x1_gm[loop_input_x1],
                                    constant.SID, 1,
                                    self.x1_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        self.tik_instance.data_move(x2_ub, self.x2_gm[loop_input_x1],
                                    constant.SID, 1,
                                    self.x1_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        self.tik_instance.vec_dup(self.mask_num, xq_ub, 0, repeat_time, 8)
        self.tik_instance.vec_dup(self.mask_num, xq_ub2, 0, repeat_time, 8)
        with self.tik_instance.for_range(0, self.first_reduce, thread_num=2) as ub_loop_idx:
            ub_loop_addr = ub_loop_idx * self.ub_loop_gap + loop_input
            with self.tik_instance.for_range(0, self.second_reduce) as reduce_add_loop_idx:
                ub_loop_addr = ub_loop_addr + reduce_add_loop_idx * self.dim_3
                # x1_grad = xq * grad
                self.tik_instance.data_move(ub_temp, self.xq_gm[ub_loop_addr],
                                            constant.SID, self.nburst,
                                            self.burst, self.src_stride, constant.STRIDE_ZERO)
                self.tik_instance.data_move(grad_ub, self.grad_gm[ub_loop_addr],
                                            constant.SID, self.nburst,
                                            self.burst, self.src_stride, constant.STRIDE_ZERO)
                if self.need_backward is True:
                    self.tik_instance.data_move(ub_temp2[self.dim_3//2], ub_temp,
                                                constant.SID, self.nburst_chunk,
                                                self.burst_chunk, self.burst_chunk, self.burst_chunk)
                    self.tik_instance.vmuls(self.mask_num, ub_temp, ub_temp, -1, repeat_time, 1, 1, 8, 8)
                    self.tik_instance.data_move(ub_temp2, ub_temp[self.dim_3//2],
                                                constant.SID, self.nburst_chunk,
                                                self.burst_chunk, self.burst_chunk, self.burst_chunk)
                    # reduce_add xq_ub2
                    self.tik_instance.vec_mul(self.mask_num, ub_temp2, grad_ub, ub_temp2, repeat_time, 8, 8, 8)
                    self.tik_instance.vec_add(self.mask_num, xq_ub2, ub_temp2, xq_ub2, repeat_time, 8, 8, 8)
                    
                    # reduce_add xq_ub
                    self.tik_instance.vmuls(self.mask_num, ub_temp, ub_temp, -1, repeat_time, 1, 1, 8, 8)
                    self.tik_instance.vec_mul(self.mask_num, ub_temp, grad_ub, ub_temp, repeat_time, 8, 8, 8)
                    self.tik_instance.vec_add(self.mask_num, xq_ub, ub_temp, xq_ub, repeat_time, 8, 8, 8)
                    
                # xq_grad = x1 * grad + [1,-1]*x2*grad
                self.tik_instance.vec_mul(self.mask_num, xq_grad_ub, grad_ub, x1_ub, repeat_time, 8, 8, 8)
                self.tik_instance.vec_mul(self.mask_num, ub_temp, grad_ub, x2_ub, repeat_time, 8, 8, 8)
                # move x1*grad
                self.tik_instance.data_move(self.xq_grad_gm[ub_loop_addr], xq_grad_ub,
                                            constant.SID, self.nburst,
                                            self.burst, constant.STRIDE_ZERO, self.src_stride)
                # move [-1,1]*x2*grad
                self.tik_instance.data_move(self.xq_grad_gm[ub_loop_addr], ub_temp[self.dim_3 // 2],
                                            constant.SID, self.nburst_chunk,
                                            self.burst_chunk, self.x2_src_stride_chunk, self.xq_src_stride_chunk)
                self.tik_instance.vmuls(self.mask_num, ub_temp, ub_temp, -1, repeat_time, 1, 1, 8, 8)
                self.tik_instance.data_move(self.xq_grad_gm[ub_loop_addr + self.dim_3 // 2], ub_temp,
                                            constant.SID, self.nburst_chunk,
                                            self.burst_chunk, self.x2_src_stride_chunk, self.xq_src_stride_chunk)
                
        if self.need_backward is True:
            self.tik_instance.data_move(self.x1_grad_gm[loop_input_x1], xq_ub,
                                        constant.SID, constant.DEFAULT_NBURST,
                                        self.x1_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            # # x2_grad = ([-1,1]*xq)*grad
            self.tik_instance.data_move(self.x2_grad_gm[loop_input_x1], xq_ub2,
                                        constant.SID, constant.DEFAULT_NBURST,
                                        self.x1_burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def caculation_process_normal(self, core_index, loop_idx, burst_normal, cal_num):
        loop_input_x1 = core_index * self.core_data + loop_idx * self.available_ub_size
        with self.tik_instance.if_scope(self.tiling_mode == 0):
            loop_input_xq = core_index * self.core_data + loop_idx * self.available_ub_size
            self.rotary_need_broadcast(loop_input_xq, loop_input_x1, burst_normal, cal_num)
        with self.tik_instance.if_scope(self.tiling_mode == 1):
            loop_input_xq = core_index * self.core_data * self.dim_2 + loop_idx * self.available_ub_size * self.dim_2
            self.rotary_need_broadcast(loop_input_xq, loop_input_x1, burst_normal, cal_num)
        with self.tik_instance.elif_scope(self.tiling_mode == 2):
            self.rotary_need_broadcast(loop_input_x1, loop_input_x1, burst_normal, cal_num)
        with self.tik_instance.elif_scope(self.tiling_mode == 3):
            loop_input_xq = core_index * self.core_data * self.dim_2*self.dim_1 +\
                            loop_idx * self.available_ub_size * self.dim_2 * self.dim_1
            self.rotary_need_broadcast(loop_input_xq, loop_input_x1, burst_normal, cal_num)
        with self.tik_instance.elif_scope(self.tiling_mode == 4):
            loop_input_xq = core_index * self.core_data * self.dim_2 + loop_idx * self.available_ub_size * self.dim_2
            self.rotary_need_broadcast(loop_input_xq, loop_input_x1, burst_normal, cal_num)

    def caculation_process_fp16(self, core_index, loop_idx, burst_normal, cal_num):
        loop_input_x1 = core_index * self.core_data + loop_idx * self.available_ub_size
        with self.tik_instance.if_scope(self.tiling_mode == 0):
            loop_input_xq = core_index * self.core_data + loop_idx * self.available_ub_size
            self.rotary_compute_fp16(loop_input_xq, loop_input_x1, burst_normal, cal_num)
        with self.tik_instance.if_scope(self.tiling_mode == 1):
            loop_input_xq = core_index * self.core_data * self.dim_2 + loop_idx * self.available_ub_size * self.dim_2
            self.rotary_compute_fp16(loop_input_xq, loop_input_x1, burst_normal, cal_num)
        with self.tik_instance.elif_scope(self.tiling_mode == 2):
            self.rotary_compute_fp16(loop_input_x1, loop_input_x1, burst_normal, cal_num)
        with self.tik_instance.elif_scope(self.tiling_mode == 3):
            loop_input_xq = core_index * self.core_data * self.dim_2*self.dim_1 +\
                            loop_idx * self.available_ub_size * self.dim_2 * self.dim_1
            self.rotary_compute_fp16(loop_input_xq, loop_input_x1, burst_normal, cal_num)
        with self.tik_instance.elif_scope(self.tiling_mode == 4):
            loop_input_xq = core_index * self.core_data * self.dim_2 + loop_idx * self.available_ub_size * self.dim_2
            self.rotary_compute_fp16(loop_input_xq, loop_input_x1, burst_normal, cal_num)
    
    def copy_only(self, core_index, loop_num, tail_num):
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            burst_normal = (self.available_ub_size + self.x_data_each_block - 1) // self.x_data_each_block
            if (self.x1_dtype == "float16" and self.mode == "high_precision") or self.x1_dtype == "bfloat16":
                self.caculation_process_fp16(core_index, loop_idx, burst_normal, self.available_ub_size)
            else:
                self.caculation_process_normal(core_index, loop_idx, burst_normal, self.available_ub_size)
      
        with self.tik_instance.if_scope(tail_num > 0):
            burst_normal = (tail_num + self.x_data_each_block - 1) // self.x_data_each_block
            if (self.x1_dtype == "float16" and self.mode == "high_precision") or self.x1_dtype == "bfloat16":
                self.caculation_process_fp16(core_index, loop_num, burst_normal, tail_num)
            else:
                self.caculation_process_normal(core_index, loop_num, burst_normal, tail_num)

    def tik_instance_function(self):
        """
        the entry of rotary_backward calculation

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            burst_len = (Constant.TILING_ARG_NUM - 1) // Constant.INT32_BLOCK + 1
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, burst_len, 0, 0)
            self.get_tiling_args(tiling_ub)
        self.tik_instance.scope_attr(None, "pragma_atomic_intra_core_only", 1)
        # core process
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_index:
            with self.tik_instance.if_scope(core_index < (self.core_used - 1)):
                self.copy_only(core_index, self.copy_loop, self.copy_tail)
            with self.tik_instance.elif_scope(core_index == (self.core_used - 1)):
                self.copy_only(core_index, self.last_copy_loop, self.last_copy_tail)

        opt_config = {}
        tbe_context.get_context().add_compile_info("vars",
                                                   {"core_num": self.core_num,
                                                    "x_data_each_block": self.x_data_each_block,
                                                    "each_repeat_block_number": self.each_repeat_block_number,
                                                    "ub_max_size": self.ub_max_size})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.xq_gm, self.x1_gm, self.x2_gm, self.grad_gm],
                                   outputs=[self.xq_grad_gm, self.x1_grad_gm, self.x2_grad_gm],
                                   flowtable=[self.tiling_gm], config=opt_config)
        return self.tik_instance


# 'pylint: disable=unused-argument, too-many-locals, too-many-lines, too-many-arguments
@register_operator("RotaryMulGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def rotary_mul_grad(x, r1, r2, dy, dx, dr1, dr2, need_backward,
                    kernel_name="rotary_mul_grad", impl_mode="high_precision"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input x1
    r1 : dict
        shape and dtype of xq
    r2 : dict
        shape and dtype of x2
    dy : dict
        shape and dtype of grad
    dx : dict
        shape and dtype of input x1_grad
    dr1 : dict
        shape and dtype of xq_grad
    dr2 : dict
        shape and dtype of x2_grad
    need_backward : bool
        r1, r2 whether need compute grad
    kernel_name : str
        kernel name, default value is "rotary_mul_grad"

    Returns
    -------
    Nonevim i
    """
    result_instance = RotaryBackward(x, r1, r2, dy, dx, dr1, dr2, need_backward, kernel_name, impl_mode)
    instance = result_instance.tik_instance_function()
    return instance