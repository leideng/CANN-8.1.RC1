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

from functools import reduce as functools_reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context


class Constant:
    """
    class of constant
    """
    MAX_INT64 = 2**63 - 1
    BYTES_PER_BLOCK = 32
    BYTES_PER_KB = 1024
    UB_PRESERVED = 16 * BYTES_PER_KB
    UB_SLICES = 10
    MAX_VEC_PROCESS_NUM = 64
    TASK_ALIGN = 4096
    TILING_ARG_NUM = 4
    TILING_ARGUMENT_DTYPE = 'int64'
    BOOL_DTYPE = 'int8'
    DTYPE_BYTES_DICT = {"float16": 2, "float32": 4, "int64": 8, "int32": 4, 
                        "uint8": 1, "int8": 1, "bool": 1, "int16": 2}
    
    
# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
def check_supported(x, mask, updates, y, kernel_name="masked_scatter"):
    """
    check the op support situation
    """
    x_shape = x.get("shape")
    mask_shape = mask.get("shape")
    if int(-1) in x_shape or int(-2) in x_shape:
        return "Unknown"
    if x_shape != mask_shape:
        return False
    return True


# 'pylint: disable=too-many-arguments, too-many-instance-attributes，unused-argument, too-many-locals, too-many-lines
class MaskedScatter:
    """
    Function: replaces elements in a tensor based on a boolean mask. 
    For each True value in the mask, the corresponding element 
    in the tensor is replaced by a value from a source tensor.
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, x, mask, updates, y, kernel_name):
        """
        initialize the masked_scatter function
        """
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.used_aicore_num = tik.Dprofile().get_aicore_num()

        self.dtype = y.get("dtype").lower()
        if self.dtype == "bfloat16":
            self.dtype = "float16"
        
        # number of elements of x, mask and updates
        self.num_elem_x = self.tik_instance.Scalar("int64")
        self.num_elem_mask = self.tik_instance.Scalar("int64")
        self.num_elem_updates = self.tik_instance.Scalar("int64")
        self.tiling_core_num = self.tik_instance.Scalar("int64")
        
        # request gm space for inputs
        self.x_gm = self.tik_instance.Tensor(
            self.dtype, [Constant.MAX_INT64], name="x_gm", scope=tik.scope_gm)
        self.mask_gm = self.tik_instance.Tensor(
            Constant.BOOL_DTYPE, [Constant.MAX_INT64], name="mask_gm", scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(
            self.dtype, [Constant.MAX_INT64], name="updates_gm", scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(
            self.dtype, [Constant.MAX_INT64], name="y_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_ARGUMENT_DTYPE, [Constant.TILING_ARG_NUM], 
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.get_tiling_args()

        # get the number of elements in one blok according to 'dtype'
        self.dtype_bytes_size = Constant.DTYPE_BYTES_DICT.get(self.dtype, 8)
        self.num_elem_per_block = Constant.BYTES_PER_BLOCK // self.dtype_bytes_size

        # 32B aligned elem_per_core
        self.aligned_elem_per_core = self.tik_instance.Scalar("int64", init_value=self.get_task_block_size())
        self.logi_task_num = self.tik_instance.Scalar("int64")
        self.logi_task_num_per_aicore = self.tik_instance.Scalar("int64")
        self.logi_task_tail = self.tik_instance.Scalar("int64")

        self.logi_task_num.set_as(self.ceil_div(self.num_elem_x, self.aligned_elem_per_core))
        self.logi_task_num_per_aicore.set_as(self.logi_task_num // self.tiling_core_num)
        self.logi_task_tail.set_as(self.logi_task_num % self.tiling_core_num)

    @staticmethod
    def ceil_div(val_x, val_y):
        """
        ceiling division
        """
        return ((val_x + val_y - 1) // val_y)

    def align_div(self, val_x, val_y):
        result = self.tik_instance.Scalar("int64", init_value=0)
        with self.tik_instance.if_scope(val_y != 0):
            result.set_as(((val_x + val_y - 1) / val_y) * val_y)
        return result
    
    def get_task_block_size(self):
        block_size_aligned = self.tik_instance.Scalar("int64", init_value=0)
        with self.tik_instance.if_scope(self.tiling_core_num != 0):
            block_size = self.tik_instance.Scalar("int64", init_value=self.num_elem_x / self.tiling_core_num)
            block_size_aligned.set_as(self.align_div(block_size, Constant.MAX_VEC_PROCESS_NUM))
        with self.tik_instance.if_scope(self.num_elem_x >= self.tiling_core_num * Constant.TASK_ALIGN):
            block_size_aligned.set_as(Constant.TASK_ALIGN)
        with self.tik_instance.if_scope(self.num_elem_x <= Constant.MAX_VEC_PROCESS_NUM):
            block_size_aligned.set_as(Constant.MAX_VEC_PROCESS_NUM)
        return block_size_aligned
        
    def get_tiling_args(self):
        """
        get_tiling_args
        """
        tiling_ub = self.tik_instance.Tensor("int64", [Constant.TILING_ARG_NUM], 
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        self.num_elem_x.set_as(tiling_ub[0])
        self.num_elem_mask.set_as(tiling_ub[1])
        self.num_elem_updates.set_as(tiling_ub[2])
        self.tiling_core_num.set_as(tiling_ub[3])

    def task_schedule(self):
        """multi-core task scheduling"""    
        with self.tik_instance.for_range(0, self.tiling_core_num, block_num =self.tiling_core_num) as aicore_idx:
            core_task_num = self.tik_instance.Scalar("int64", init_value=self.logi_task_num_per_aicore)
            with self.tik_instance.if_scope(aicore_idx < self.logi_task_tail):
                core_task_num.set_as(core_task_num + 1)

            pre_core_task_num = self.tik_instance.Scalar("int64", init_value=(self.logi_task_num_per_aicore + 1) *
                                                         aicore_idx)
            with self.tik_instance.if_scope(aicore_idx > self.logi_task_tail):
                pre_core_task_num.set_as((self.logi_task_num_per_aicore + 1) * self.logi_task_tail +
                                         (aicore_idx - self.logi_task_tail) * self.logi_task_num_per_aicore)
            task_updates_start = self.tik_instance.Scalar("int64", init_value=0)
            with self.tik_instance.if_scope(core_task_num > 0):
                task_updates_start.set_as(self.calc_updates_start(pre_core_task_num * self.aligned_elem_per_core))

            with self.tik_instance.for_range(0, core_task_num) as task_idx:
                task_updates_start.set_as(self.compute((pre_core_task_num + task_idx) *
                                                       self.aligned_elem_per_core, task_updates_start))

        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.used_aicore_num
            })

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.x_gm, self.mask_gm, self.updates_gm],
            outputs=[self.y_gm],
            flowtable=[self.tiling_gm]
        )

        return self.tik_instance

    def compute(self, input_offset, updates_start):
        """
        compute "num_elem_per_input" numbers of elements from "input_offset"

        Parameters
        ----------
        input_offset : int
            the number of elements will be calculated
        num_elem_per_input : int, optional
            offset of the starting element in gm
        Returns
        -------
        """
        num_elem_per_input = self.tik_instance.Scalar("int64", init_value=self.aligned_elem_per_core)
        burst_mask = self.tik_instance.Scalar("int64", init_value=self.aligned_elem_per_core
                                              // Constant.BYTES_PER_BLOCK)
                    
        with self.tik_instance.if_scope(input_offset + self.aligned_elem_per_core > self.num_elem_x):
            num_elem_per_input.set_as(self.num_elem_x - input_offset)
            burst_mask.set_as(self.ceil_div(num_elem_per_input, Constant.BYTES_PER_BLOCK))

        burst_x = self.tik_instance.Scalar("int64", init_value=self.ceil_div(num_elem_per_input, 
                                                                             self.num_elem_per_block))

        num_remain_updates = self.tik_instance.Scalar("int64", init_value=self.num_elem_updates - updates_start)
        with self.tik_instance.if_scope(num_remain_updates > num_elem_per_input):
            num_remain_updates.set_as(num_elem_per_input)

        burst_updates = self.tik_instance.Scalar("int64", init_value=self.ceil_div(num_remain_updates, 
                                                                                   self.num_elem_per_block))

        x_ub = self.tik_instance.Tensor(self.dtype, [self.aligned_elem_per_core], name="x_ub", scope=tik.scope_ubuf)
        mask_ub = self.tik_instance.Tensor(Constant.BOOL_DTYPE, [self.aligned_elem_per_core], 
                                           name="mask_ub", scope=tik.scope_ubuf)
        updates_ub = self.tik_instance.Tensor(self.dtype, [self.aligned_elem_per_core], 
                                              name="updates_ub", scope=tik.scope_ubuf)

        self.tik_instance.data_move(x_ub, self.x_gm[input_offset], 0, 1, burst_x, 0, 0)
        self.tik_instance.data_move(mask_ub, self.mask_gm[input_offset], 0, 1, burst_mask, 0, 0)

        updates_offset_ub = self.tik_instance.Scalar("int64", init_value=0)
        with self.tik_instance.if_scope(num_remain_updates > 0):
            # prevent oom, address rollback
            updates_back_offset = self.tik_instance.Scalar("int64", init_value=0)
            aligned_num_elem_updates = self.tik_instance.Scalar("int64", init_value=self.ceil_div(
                self.num_elem_updates, self.num_elem_per_block) * self.num_elem_per_block)
            with self.tik_instance.if_scope(updates_start + burst_updates * self.num_elem_per_block >
                                            aligned_num_elem_updates):
                updates_back_offset.set_as(updates_start + burst_updates * self.num_elem_per_block -
                                           aligned_num_elem_updates)
            self.tik_instance.data_move(updates_ub, self.updates_gm[updates_start - updates_back_offset], 0, 1,
                                        burst_updates, 0, 0)
            with self.tik_instance.for_range(0, num_elem_per_input) as offset:
                with self.tik_instance.if_scope(tik.all(mask_ub[offset] != 0)):
                    x_ub[offset].set_as(updates_ub[updates_offset_ub + updates_back_offset])
                    updates_offset_ub.set_as(updates_offset_ub + 1)
        updates_start.set_as(updates_start + updates_offset_ub)
        self.tik_instance.data_move(self.y_gm[input_offset], x_ub, 0, 1, burst_x, 0, 0)
        return updates_start

    def calc_updates_start(self, num_elem_mask):
        """
        compute number of "true" for the first "num_elem_mask" elements in mask.

        Parameters
        ----------
        num_elem_mask : int
            the number of elements will be calculated in mask
        Returns
        -------
        """
        sum_res_int64 = self.tik_instance.Scalar("int64", init_value=0)
        
        with self.tik_instance.new_stmt_scope():
            # in case of ub overflow, several iters will be needed when the input is large
            data_move_length = self.tik_instance.Scalar("int64", init_value=Constant.TASK_ALIGN)
            with self.tik_instance.if_scope(tik.all(num_elem_mask < Constant.TASK_ALIGN, num_elem_mask > 0)):
                data_move_length.set_as(num_elem_mask)
                
            iters = self.ceil_div(num_elem_mask, data_move_length)

            bool_ub = self.tik_instance.Tensor(Constant.BOOL_DTYPE, [data_move_length],
                                               name="bool_ub", scope=tik.scope_ubuf)
            fp16_ub = self.tik_instance.Tensor("float16", [data_move_length],
                                               name="fp16_ub", scope=tik.scope_ubuf)
            fp32_ub = self.tik_instance.Tensor("float32", [data_move_length],
                                               name="fp32_ub", scope=tik.scope_ubuf)
            work_tensor_ub = self.tik_instance.Tensor("float32", [data_move_length],
                                                      name="work_tensor_ub", scope=tik.scope_ubuf)
            sum_fp32_ub = self.tik_instance.Tensor("float32", [data_move_length],
                                                   name="sum_fp32_ub", scope=tik.scope_ubuf)

            # number of mask elements will be calculated per iteration
            burst = self.tik_instance.Scalar("int32", init_value=0)
            repeat_times = self.tik_instance.Scalar("int32", init_value=0)
            sum_res_fp32 = self.tik_instance.Scalar("float32", init_value=0)
            temp_res_int32 = self.tik_instance.Scalar("int32", init_value=0)
            
            stride_int8 = (Constant.MAX_VEC_PROCESS_NUM * Constant.DTYPE_BYTES_DICT.get("bool", 1) //
                           Constant.BYTES_PER_BLOCK)
            stride_fp16 = (Constant.MAX_VEC_PROCESS_NUM * Constant.DTYPE_BYTES_DICT.get("float16", 2) //
                           Constant.BYTES_PER_BLOCK)
            stride_fp32 = (Constant.MAX_VEC_PROCESS_NUM * Constant.DTYPE_BYTES_DICT.get("float32", 4) //
                           Constant.BYTES_PER_BLOCK)
            burst.set_as(self.ceil_div(data_move_length * Constant.DTYPE_BYTES_DICT.get("bool", 1),
                                       Constant.BYTES_PER_BLOCK))
            repeat_times.set_as(self.ceil_div(data_move_length, Constant.MAX_VEC_PROCESS_NUM))
            remain_length = self.tik_instance.Scalar("int64", init_value=num_elem_mask -
                                                     (iters - 1) * data_move_length)
            with self.tik_instance.for_range(0, iters) as ub_idx:
                offset = (self.ceil_div(data_move_length * Constant.DTYPE_BYTES_DICT.get("bool", 1),
                                        Constant.BYTES_PER_BLOCK) * Constant.BYTES_PER_BLOCK * ub_idx)
                with self.tik_instance.if_scope(tik.all(ub_idx == iters - 1, remain_length > 0)):
                    burst.set_as(self.ceil_div(remain_length * Constant.DTYPE_BYTES_DICT.get("bool", 1),
                                        Constant.BYTES_PER_BLOCK))
                    repeat_times.set_as(self.ceil_div(remain_length, Constant.MAX_VEC_PROCESS_NUM))
                self.tik_instance.data_move(bool_ub, self.mask_gm[offset], 0, 1, burst, 0, 0)
                self.tik_instance.vec_conv(Constant.MAX_VEC_PROCESS_NUM, "none", fp16_ub, bool_ub,
                                        repeat_times, stride_fp16, stride_int8)
                self.tik_instance.vec_conv(Constant.MAX_VEC_PROCESS_NUM, "none", fp32_ub, fp16_ub,
                                        repeat_times, stride_fp32, stride_fp16)
                self.tik_instance.vec_reduce_add(Constant.MAX_VEC_PROCESS_NUM, sum_fp32_ub, fp32_ub,
                                                work_tensor_ub, repeat_times, stride_fp32)
                sum_res_fp32.set_as(sum_fp32_ub[0])
                self.tik_instance.scalar_conv("round", temp_res_int32, sum_res_fp32)
                sum_res_int64.set_as(sum_res_int64 + temp_res_int32)
            return sum_res_int64


@ register_operator("masked_scatter")
@ para_check.check_op_params(para_check.REQUIRED_INPUT,
                             para_check.REQUIRED_INPUT,
                             para_check.REQUIRED_INPUT,
                             para_check.REQUIRED_OUTPUT,
                             para_check.KERNEL_NAME)
def masked_scatter(x, mask, updates, y, kernel_name="masked_scatter"):
    """
    calculating data

    Parameters
    ----------
    x : tensor
        input x
    mask : tensor
        boolean mask tensor
    updates : tensor
        replace elements in tensor x
    y : tensor
        output tensor
    kernel_name : str
        kernel name, default value is "masked_scatter"

    Returns
    -------
    Tik instance
    """
    op = MaskedScatter(x, mask, updates, y, kernel_name)
    return op.task_schedule()
