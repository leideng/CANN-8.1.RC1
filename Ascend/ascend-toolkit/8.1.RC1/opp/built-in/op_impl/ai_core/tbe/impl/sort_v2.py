# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
sort_v2
"""

# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
from functools import reduce as functools_reduce

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # proposal struct contains 8 elements
    PROPOSAL_NUM = 8
    # use this idx in proposal struct for val
    VAL_IDX = 4
    # use this idx in proposal struct for idx's merchant part
    INT_IDX = 0
    # use this idx in proposal struct for idx's remainder part
    REM_IDX = 1
    # mask for vadds
    MASK = 128
    # sorting limit for data volume
    DATA_LIMITE = 100000
    # api arg
    REPEAT_MAX = 255
    # num per block
    NUM_PER_BLOCK = 8


def op_select_format(x, y, axis=-1, descending=False, kernel_name="sort_v2"):
    """
    select format depend on the soc version.
    """
    is_old_version = True if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310", "Ascend910") else False

    op_dtype = "float16" if is_old_version else "float16,float32,bfloat16"
    op_format = "ND" if is_old_version else "ND,ND,ND"

    input0 = gen_param(classify="input0", name="x", datatype=op_dtype, format=op_format)
    output0 = gen_param(classify="output0", name="y", datatype=op_dtype, format=op_format)

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
class SortSimple(object):
    def __init__(self, x, y, axis, descending, kernel_name):
        """ __init__ """
        self.cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        self.is_old_version = True if not tbe_platform.api_check_support("tik.vreducev2") else False
        self.shape, self.dtype, self.num_per_task = self.check(x, y, axis, kernel_name)
        self.kernel_name = kernel_name
        self.descending = descending
        self.task_num = functools_reduce(lambda x, y: x * y, self.shape) // self.num_per_task
        self.tik_instance = tik.Tik()
        self.align = 4 if self.dtype == "float16" else 2
        self.block_size = 16 if self.dtype == "float16" else 8
        self.num_offset = Constant.PROPOSAL_NUM if self.is_old_version else self.align
        self.min = -65504 if self.dtype == "float16" else -3.4e38
        self.max = 65504 if self.dtype == "float16" else 3.4e38
        self.sort_size = 16 if self.is_old_version else 32
        self.struce_len = 16 if self.is_old_version else 8
        self.mask = 128 if self.dtype == "float16" else 64
        self.num_block = 2048 if self.is_old_version else 4096

        self.num_align = (self.num_per_task + self.num_block - 1) // self.num_block * self.num_block
        self.batch_per_task = self.num_align // self.num_block
        self.move_out_block = (self.num_per_task + self.block_size - 1) // self.block_size
        self.num_per_batch_align = self.move_out_block * self.block_size
        self.num_mini_align = (self.num_per_task + self.sort_size - 1) // self.sort_size * self.sort_size
        self.num_per_batch = self.num_mini_align if self.num_per_task <= self.num_block else self.num_block

        self.input_gm = self.tik_instance.Tensor(self.dtype, self.shape, name="input_gm", scope=tbe_platform.scope_gm)
        self.data_out = self.tik_instance.Tensor(self.dtype, self.shape, name="data_out", scope=tbe_platform.scope_gm)

        self.available_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.used_core_num = self.available_core_num if self.task_num > self.available_core_num else self.task_num
        self.batch_num_per_core_process = self.task_num // self.used_core_num
        self.batch_tail = self.task_num % self.used_core_num

        self.tmp_workspace = None
        if self.batch_per_task == 1:
            self.data_out_align = self.tik_instance.Tensor(self.dtype, [self.task_num * self.num_per_batch],
                                                           name="data_out_align",
                                                           scope=tbe_platform.scope_gm, is_workspace=True)

        else:
            self.data_out_align = self.tik_instance.Tensor(self.dtype, [self.task_num * self.num_align],
                                                           name="data_out_align",
                                                           scope=tbe_platform.scope_gm, is_workspace=True)

            self.tmp_workspace = self.tik_instance.Tensor(
                self.dtype, [self.used_core_num * self.num_align * self.num_offset],
                name="tmp_workspace", scope=tbe_platform.scope_gm, is_workspace=True)

    def check(self, x, y, axis, kernel_name):
        """
        Function: Check parameters (eg: shape dtype etc).
        """
        para_check.check_kernel_name(kernel_name)

        shape = y.get("shape")
        dtype = y.get("dtype").lower()
        para_check.check_dtype_rule(dtype, ("float16", "float32"))
        para_check.check_shape_rule(shape)

        shape = x.get("shape")
        dtype = x.get("dtype").lower()
        para_check.check_dtype_rule(dtype, ("float16", "float32"))
        para_check.check_shape_rule(shape)

        if axis != -1 and axis != len(shape) - 1:
            raise RuntimeError("Dim should take the last one.")

        num_per_task = shape[axis]

        return shape, dtype, num_per_task

    def sort_compute(self):
        """
        Function: sort compute.
        """
        with self.tik_instance.for_range(0, self.used_core_num, block_num=self.used_core_num) as i_idx:
            with self.tik_instance.for_range(0, self.batch_num_per_core_process) as j_idx:
                self.task_schedule(i_idx + j_idx * self.used_core_num)
            with self.tik_instance.if_scope(i_idx < self.batch_tail):
                self.task_schedule(self.batch_num_per_core_process * self.used_core_num + i_idx)

            self.tune()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_gm],
                                   outputs=[self.data_out])

        return self.tik_instance

    def task_schedule(self, task_idx):
        """
        Function: sort compute task schedule, new for 910B & 310B.
        """
        if self.is_old_version:
            if self.num_per_task <= self.num_block:
                self.sort_mini_num(task_idx)
            else:
                self.sort_large_num(task_idx)
        else:
            if self.num_per_task <= self.num_block:
                self.sort_mini_num_new(task_idx)
            else:
                self.sort_large_num_new(task_idx)

    def sort_mini_num(self, task_idx):
        """
        Function: fix num_per_task less than 2048.
        """
        input_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch * Constant.PROPOSAL_NUM * 2],
                                            name="input_ub", scope=tbe_platform.scope_ubuf)

        offset_in = task_idx * self.num_per_task
        offset_out = task_idx * self.num_per_batch
        dest_pos_ub = self.num_per_batch * Constant.PROPOSAL_NUM
        n_repeat_total = self.num_per_batch // self.block_size
        # 1. Move data from OUT to UB
        self.tik_instance.data_move(input_ub[dest_pos_ub], self.input_gm[offset_in], 0, 1, n_repeat_total, 0, 0)
        max_num = self.tik_instance.Scalar(self.dtype, init_value=self.max)
        min_num = self.tik_instance.Scalar(self.dtype, init_value=self.min)
        # Add ineffective object for 16 alignment
        if self.descending:
            with self.tik_instance.for_range(0, self.num_per_batch - self.num_per_task) as i:
                input_ub[(self.num_per_task + i) + dest_pos_ub].set_as(min_num)
        else:
            with self.tik_instance.for_range(0, self.num_per_batch - self.num_per_task) as i:
                input_ub[(self.num_per_task + i) + dest_pos_ub].set_as(max_num)

        self.tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], n_repeat_total, Constant.VAL_IDX)

        # 2. vbs16
        self.tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=n_repeat_total)
        # 3. vms4
        input_ub, dest_pos_ub = self.vms4(input_ub, dest_pos_ub)
        # 4. Move Data from UB to OUT
        self.moveout_mini_num(offset_out, input_ub, dest_pos_ub)

    def sort_large_num(self, task_idx):
        """
        Function: fix num_per_task more than 2048.
        """
        offset = (task_idx % self.used_core_num) * self.batch_per_task * self.num_block * Constant.PROPOSAL_NUM
        ori_offset = task_idx * self.num_per_task
        input_ub = self.tik_instance.Tensor(self.dtype, [self.num_block * 2 * Constant.PROPOSAL_NUM * 2],
                                            name="input_ub", scope=tbe_platform.scope_ubuf)

        # SORT IN UB
        for i in range(self.batch_per_task):
            self.sort_in_ub(input_ub, i, ori_offset, offset)

        # SORT IN GM
        self.sort_in_gm(input_ub, offset)

        # Pick Data from GM to GM
        self.moveout_large_num(offset, task_idx, input_ub)

    def sort_mini_num_new(self, task_idx):
        """
        Function: fix num_per_task less than 4096, for 910B & 310B.
        """
        # offset in self.input_gm
        in_offset = task_idx * self.num_per_task

        input_ub_tmp = self.tik_instance.Tensor(self.dtype, [self.num_per_batch * self.num_offset * 2],
                                                name="input_ub_tmp", scope=tbe_platform.scope_ubuf)
        val_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch], name="val_ub",
                                          scope=tbe_platform.scope_ubuf)
        idx_ub = self.tik_instance.Tensor("uint32", [self.num_per_batch], name="src1_ub", scope=tbe_platform.scope_ubuf)

        # dest position in UB
        repeat_times = self.num_per_batch // self.block_size
        # 1. Move data from OUT to UB

        self.tik_instance.data_move(val_ub, self.input_gm[in_offset], 0, 1, repeat_times, 0, 0)
        dest_pos_ub = self.num_per_batch * self.num_offset

        # aline for k
        if self.descending:
            with self.tik_instance.for_range(0, self.num_per_batch - self.num_per_task) as i:
                val_ub[self.num_per_task + i].set_as(self.min)
        else:
            with self.tik_instance.for_range(0, self.num_per_batch - self.num_per_task) as i:
                val_ub[self.num_per_task + i].set_as(self.max)

        # 2. vbs32
        self.tik_instance.vsort32(input_ub_tmp[dest_pos_ub], val_ub, idx_ub, self.num_per_batch // self.sort_size)

        # 3. vms4
        input_ub_tmp, dest_pos_ub = self.vms4(input_ub_tmp, dest_pos_ub)

        src_pos_ub = self.num_per_batch * self.num_offset - dest_pos_ub

        self.moveout_mini_num_new(task_idx, input_ub_tmp, src_pos_ub, repeat_times, dest_pos_ub)

    def sort_large_num_new(self, task_idx):
        """
        Function: fix num_per_task more than 4096, for 910B & 310B.
        """
        # offset in self.input_gm
        in_offset = task_idx * self.num_per_task

        # offset in self.tmp_workspace
        ws_offset = (task_idx % self.used_core_num) * self.num_align * self.num_offset
        # SORT IN UB
        with self.tik_instance.for_range(0, self.batch_per_task) as batch_idx:
            self.sort_in_ub_new(batch_idx, in_offset, ws_offset)

        # SORT IN GM
        self.sort_in_gm_new(ws_offset)

        # Pick Data from GM to GM
        self.moveout_large_num_new(ws_offset, task_idx)

    def vms4(self, input_ub, dest_pos_ub):
        """
        Function: Merge all lists into one.
        """
        # record the lists info
        length = self.num_per_batch // self.sort_size
        num_list = [self.sort_size] * length
        # Let them merge evenly, to avoid exceeding the limit of the number of single list.
        src_pos_ub = 0
        while len(num_list) > 1:
            src_pos_ub, dest_pos_ub = dest_pos_ub, src_pos_ub
            index = 0
            offset = 0
            while True:
                res = len(num_list) - index
                if res > 3:
                    num_list, input_ub, offset = self.merge4(num_list, input_ub, offset, src_pos_ub, index,
                                                             dest_pos_ub)
                elif res == 3:
                    num_list, input_ub, offset = self.merge3(num_list, input_ub, offset, src_pos_ub, index,
                                                             dest_pos_ub)
                elif res == 2:
                    num_list, input_ub, offset = self.merge2(num_list, input_ub, offset, src_pos_ub, index,
                                                             dest_pos_ub)
                elif res == 1:
                    self.tik_instance.data_move(input_ub[dest_pos_ub + offset * self.num_offset],
                                                input_ub[src_pos_ub + offset * self.num_offset], 0, 1,
                                                num_list[index] * self.struce_len // 32, 0, 0)
                else:
                    break
                index += 1

        return input_ub, dest_pos_ub

    def merge4(self, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
        """
        Function: Merge 4 lists in UB.
        """
        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index]) * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * self.num_offset],
                    input_ub[
                        src_pos_ub + (
                                offset + num_list[index] + num_list[index + 1] + num_list[
                            index + 2]) * self.num_offset]]

        src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], num_list[index + 3]]
        # merge 4 lists
        if self.is_old_version:
            self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                        if_exhausted_suspension=False, valid_bit="1111", repeat_times=1)
        else:
            self.tik_instance.vmrgsort(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                       if_exhausted_suspension=False, repeat_times=1)
        # update the lists info : Merge the four element values and record them in a(num_list)
        num_list[index] = sum(num_list[index:index + 4])
        a = num_list[:index + 1:]
        b = num_list[index + 4::]
        a.extend(b)
        offset += a[index]

        return a, input_ub, offset

    def merge3(self, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
        """
        Function: Merge 3 lists in UB.
        """
        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index]) * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * self.num_offset],
                    input_ub[0]]
        src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], 0]
        # merge 3 lists
        if self.is_old_version:
            self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                        if_exhausted_suspension=False, valid_bit="0111", repeat_times=1)
        else:
            self.tik_instance.vmrgsort(input_ub[dest_pos_ub + offset * self.num_offset], src_list[0:3],
                                       src_list_lengths[0:3], if_exhausted_suspension=False, repeat_times=1)
        # update the lists info : Merge the three element values and record them in a num_list
        num_list[index] = sum(num_list[index:index + 3])
        a = num_list[:index + 1:]
        b = num_list[index + 3::]
        a.extend(b)
        offset += a[index]

        return a, input_ub, offset

    def merge2(self, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
        """
        Function: Merge 2 lists in UB.
        """
        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index]) * self.num_offset], input_ub[0], input_ub[0]]

        src_list_lengths = [num_list[index], num_list[index + 1], 0, 0]
        # merge 2 lists
        if self.is_old_version:
            self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                        if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)
        else:
            self.tik_instance.vmrgsort(input_ub[dest_pos_ub + offset * self.num_offset], src_list[0:2],
                                       src_list_lengths[0:2], if_exhausted_suspension=False, repeat_times=1)
        # update the lists info : Merge the two element values and record them in num_list
        num_list[index] += num_list[index + 1]
        del num_list[index + 1]
        offset += num_list[index]

        return num_list, input_ub, offset

    def sort_in_ub(self, input_ub, i, ori_offset, offset):
        """
        Function: sort in ub.
        """
        # dest position in UB
        dest_pos_ub = self.num_block * Constant.PROPOSAL_NUM
        repeat_times = self.num_block // self.block_size
        # 1. Move data from OUT to UB
        self.tik_instance.data_move(input_ub[dest_pos_ub], self.input_gm[ori_offset + i * self.num_block], 0, 1,
                                    repeat_times, 0, 0)

        if self.num_per_task < (i + 1) * self.num_block:
            # aline for NUM_BLOCK
            aline = self.num_block - self.num_per_task % self.num_block
            if self.descending:
                tmp = self.tik_instance.Scalar('float16', init_value=self.min)
            # descend
            else:
                tmp = self.tik_instance.Scalar('float16', init_value=self.max)
            # Add ineffective object for 16 alignment
            for j in range(aline % self.block_size):
                input_ub[dest_pos_ub + self.num_per_task % self.num_block + j].set_as(tmp)
            # Add ineffective object for NUM_BLOCK alignment
            if aline > self.block_size - 1:
                self.tik_instance.vec_dup(self.block_size, input_ub[
                    dest_pos_ub + self.num_per_task % self.num_block + aline % self.block_size],
                                          tmp, aline // self.block_size, 1)

        self.tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], repeat_times, Constant.VAL_IDX)
        # 2. vrpsort16
        self.tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=repeat_times)
        # 3. vms4
        input_ub, dest_pos_ub = self.vms4(input_ub, dest_pos_ub)
        # 4. Move Data from UB to OUT
        self.tik_instance.data_move(self.tmp_workspace[offset + i * self.num_block * Constant.PROPOSAL_NUM],
                                    input_ub[dest_pos_ub], 0, 1,
                                    self.num_block * Constant.PROPOSAL_NUM // self.block_size, 0, 0)

    def sort_in_gm(self, input_ub, offset):
        """
        Function: sort in gm.
        ----------
        """
        src_pos_ub = self.tik_instance.Scalar("int32")
        dest_pos_ub = self.tik_instance.Scalar("int32")

        with self.tik_instance.for_range(0, self.batch_per_task - 1) as tail:
            src_pos_ub.set_as(0)
            dest_pos_ub.set_as(self.num_block * 2 * Constant.PROPOSAL_NUM)

            self.tik_instance.data_move(input_ub[src_pos_ub + self.num_block * Constant.PROPOSAL_NUM],
                                        self.tmp_workspace[offset], 0,
                                        1, (self.num_block * Constant.PROPOSAL_NUM) // self.block_size, 0, 0)
            with self.tik_instance.for_range(1, self.batch_per_task - tail) as i:
                self.tik_instance.data_move(input_ub[src_pos_ub],
                                            self.tmp_workspace[
                                                offset + self.num_block * i * Constant.PROPOSAL_NUM],
                                            0, 1, (self.num_block * Constant.PROPOSAL_NUM) // self.block_size, 0, 0)

                self.tik_instance.vmrgsort4(input_ub[dest_pos_ub],
                                            [input_ub[src_pos_ub],
                                             input_ub[src_pos_ub + self.num_block * Constant.PROPOSAL_NUM],
                                             input_ub[0], input_ub[0]],
                                            [self.num_block, self.num_block, 0, 0],
                                            if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)

                self.tik_instance.data_move(
                    self.tmp_workspace[offset + self.num_block * (i - 1) * Constant.PROPOSAL_NUM],
                    input_ub[dest_pos_ub], 0, 1, (self.num_block * Constant.PROPOSAL_NUM) // self.block_size, 0, 0)

                dest_pos_ub.set_as(src_pos_ub)
                src_pos_ub.set_as(self.num_block * 2 * Constant.PROPOSAL_NUM - dest_pos_ub)

            # Move Data from UB to GM
            self.tik_instance.data_move(
                self.tmp_workspace[
                    offset + self.num_block * (self.batch_per_task - tail - 1) * Constant.PROPOSAL_NUM],
                input_ub[src_pos_ub + self.num_block * Constant.PROPOSAL_NUM], 0, 1,
                (self.num_block * Constant.PROPOSAL_NUM) // self.block_size, 0, 0)

    def sort_in_ub_new(self, batch_idx, in_offset, ws_offset):
        """
        Function: sort in ub, for 910B & 310B.
        """
        with self.tik_instance.new_stmt_scope():
            input_ub_tmp = self.tik_instance.Tensor(self.dtype, [self.num_per_batch * self.num_offset * 2],
                                                    name="input_ub_tmp", scope=tbe_platform.scope_ubuf)
            val_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch], name="val_ub",
                                              scope=tbe_platform.scope_ubuf)
            idx_ub = self.tik_instance.Tensor("uint32", [self.num_per_batch], name="src1_ub",
                                              scope=tbe_platform.scope_ubuf)

            # dest position in UB
            repeat_times = self.num_per_batch // self.block_size
            # 1. Move data from OUT to UB
            self.tik_instance.data_move(val_ub, self.input_gm[in_offset + batch_idx * self.num_per_batch],
                                        0, 1, repeat_times, 0, 0)
            dest_pos_ub = self.num_per_batch * self.num_offset
            with self.tik_instance.if_scope(self.num_per_task < (batch_idx + 1) * self.num_per_batch):
                # aline for k
                aline = self.num_per_batch - self.num_per_task % self.num_per_batch
                tmp_val = self.tik_instance.Scalar(self.dtype)
                if self.descending:
                    tmp_val.set_as(self.min)
                else:
                    tmp_val.set_as(self.max)

                repeat_num_max = self.block_size * Constant.REPEAT_MAX
                # Add ineffective object for 16 alignment
                with self.tik_instance.for_range(0, aline % self.block_size) as j:
                    val_ub[self.num_per_task % self.num_per_batch + j].set_as(tmp_val)
                # Add ineffective object for k alignment
                if aline >= self.block_size:
                    if aline > repeat_num_max:
                        with self.tik_instance.for_range(0, aline // repeat_num_max) as repeat_idx:
                            self.tik_instance.vec_dup(self.block_size,
                                                      val_ub[self.num_per_task % self.num_per_batch +
                                                             aline % self.block_size +
                                                             repeat_idx * repeat_num_max],
                                                      tmp_val, Constant.REPEAT_MAX, 1)
                        self.tik_instance.vec_dup(self.block_size,
                                                  val_ub[self.num_per_task % self.num_per_batch +
                                                         aline % self.block_size +
                                                         aline // repeat_num_max * repeat_num_max],
                                                  tmp_val, aline // self.block_size -
                                                  aline // repeat_num_max * Constant.REPEAT_MAX, 1)
                    else:
                        self.tik_instance.vec_dup(self.block_size,
                                                  val_ub[self.num_per_task % self.num_per_batch +
                                                         aline % self.block_size],
                                                  tmp_val, aline // self.block_size, 1)

            # 2. vbs32
            self.tik_instance.vsort32(input_ub_tmp[dest_pos_ub], val_ub, idx_ub, self.num_per_batch // self.sort_size)

            # 3. vms4
            input_ub_tmp, dest_pos_ub = self.vms4(input_ub_tmp, dest_pos_ub)

            # 4. Move Data from UB to OUT
            self.tik_instance.data_move(
                self.tmp_workspace[ws_offset + batch_idx * self.num_per_batch * self.num_offset],
                input_ub_tmp[dest_pos_ub], 0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)

    def sort_in_gm_new(self, ws_offset):
        """
        Function: sort in gm, for 910B & 310B.
        """
        with self.tik_instance.new_stmt_scope():
            input_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch * self.num_offset * 2],
                                                name="input_ub", scope=tbe_platform.scope_ubuf)
            src_pos_ub = self.tik_instance.Scalar("int32")
            dest_pos_ub = self.tik_instance.Scalar("int32")

            with self.tik_instance.for_range(0, self.batch_per_task - 1) as tail:
                src_pos_ub.set_as(0)
                dest_pos_ub.set_as(self.num_per_batch * self.num_offset)

                self.tik_instance.data_move(input_ub[src_pos_ub + self.num_per_batch * self.num_offset],
                                            self.tmp_workspace[ws_offset], 0, 1,
                                            self.num_per_batch * self.struce_len // 32, 0, 0)
                with self.tik_instance.for_range(1, self.batch_per_task - tail) as i:
                    self.tik_instance.data_move(input_ub[src_pos_ub],
                                                self.tmp_workspace[
                                                    ws_offset + self.num_per_batch * i * self.num_offset],
                                                0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)
                    self.tik_instance.vmrgsort(input_ub[dest_pos_ub],
                                               [input_ub[src_pos_ub],
                                                input_ub[src_pos_ub + self.num_per_batch * self.num_offset]],
                                               [self.num_per_batch, self.num_per_batch],
                                               if_exhausted_suspension=False, repeat_times=1)
                    self.tik_instance.data_move(
                        self.tmp_workspace[ws_offset + self.num_per_batch * (i - 1) * self.num_offset],
                        input_ub[dest_pos_ub], 0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)

                    dest_pos_ub.set_as(src_pos_ub)
                    src_pos_ub.set_as(self.num_per_batch * self.num_offset - dest_pos_ub)

                # Move Data from UB to GM
                self.tik_instance.data_move(
                    self.tmp_workspace[
                        ws_offset + self.num_per_batch * (self.batch_per_task - tail - 1) * self.num_offset],
                    input_ub[src_pos_ub + self.num_per_batch * self.num_offset], 0, 1,
                    self.num_per_batch * self.struce_len // 32, 0, 0)

    def moveout_mini_num(self, offset_out, input_ub, dest_pos_ub):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32.
        """
        src_pos_ub = self.num_per_batch * Constant.PROPOSAL_NUM if dest_pos_ub == 0 else 0
        # ascend
        if self.descending is False:
            # data is continuous in GM & gather scattered data together
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                input_ub[i2 + src_pos_ub].set_as(
                    input_ub[(self.num_per_batch - 1 - i2) * Constant.PROPOSAL_NUM + Constant.VAL_IDX + dest_pos_ub])

        # descend
        else:
            # data is continuous in GM & gather scattered data together
            if self.cce_product == tbe_platform.ASCEND_310:
                with self.tik_instance.for_range(0, self.num_per_task) as i2:
                    input_ub[i2 + src_pos_ub].set_as(
                        input_ub[i2 * Constant.PROPOSAL_NUM + Constant.VAL_IDX + dest_pos_ub])
            else:
                self.tik_instance.vextract(input_ub[src_pos_ub], input_ub[dest_pos_ub],
                                           self.num_per_batch // self.block_size, Constant.VAL_IDX)

        # move output (float16) from UB to GM
        self.tik_instance.data_move(self.data_out_align[offset_out], input_ub[src_pos_ub], 0, 1,
                                    self.num_per_batch // self.block_size, 0, 0)

    def moveout_large_num(self, offset, task_idx, input_ub):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32.
        """
        # dest position in UB
        dest_pos_ub = self.num_block * Constant.PROPOSAL_NUM
        repeat_times = self.num_block // self.block_size

        with self.tik_instance.for_range(0, self.batch_per_task) as i:
            self.tik_instance.data_move(input_ub[0],
                                        self.tmp_workspace[offset + self.num_block * i * Constant.PROPOSAL_NUM],
                                        0, 1, (self.num_block * Constant.PROPOSAL_NUM) // self.block_size, 0, 0)

            # data is continuous in GM & gather scattered data together
            if self.cce_product == tbe_platform.ASCEND_310:
                with self.tik_instance.for_range(0, self.num_block) as i2:
                    input_ub[dest_pos_ub + i2].set_as(input_ub[i2 * Constant.PROPOSAL_NUM + Constant.VAL_IDX])
            else:
                self.tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, Constant.VAL_IDX)
            # move output (float16) from UB to GM
            # ascend
            if self.descending is False:
                with self.tik_instance.for_range(0, self.num_block) as i2:
                    input_ub[dest_pos_ub + self.num_block + i2].set_as(
                        input_ub[dest_pos_ub + self.num_block - i2 - 1])

                self.tik_instance.data_move(
                    self.data_out_align[task_idx * self.num_align + self.num_block * (self.batch_per_task - i - 1)],
                    input_ub[dest_pos_ub + self.num_block], 0, 1, repeat_times, 0, 0)
            # descend
            else:
                self.tik_instance.data_move(self.data_out_align[task_idx * self.num_align + self.num_block * i],
                                            input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)

    def moveout_mini_num_new(self, task_idx, input_ub_tmp, src_pos_ub, repeat_times, dest_pos_ub):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32, for 910B & 310B.
        """
        # move output from UB to GM
        if self.dtype == "float16":
            self.tik_instance.vreducev2(None, input_ub_tmp[src_pos_ub], input_ub_tmp[dest_pos_ub], 3,
                                        self.num_per_batch // 32, 1, 8, 0)
        else:
            self.tik_instance.vreducev2(None, input_ub_tmp[src_pos_ub], input_ub_tmp[dest_pos_ub], 1,
                                        self.num_per_batch // 32, 1, 8, 0)

        if self.descending:
            # move output (float16) from UB to GM
            self.tik_instance.data_move(self.data_out_align[task_idx * self.num_per_batch],
                                        input_ub_tmp[src_pos_ub], 0, 1, repeat_times, 0, 0)
        # ascend
        else:
            # data is continuous in GM & gather scattered data together
            with self.tik_instance.for_range(0, self.num_per_batch) as i2:
                input_ub_tmp[i2 + dest_pos_ub].set_as(
                    input_ub_tmp[src_pos_ub + (self.num_per_batch - 1 - i2)])

            # move output (float16) from UB to GM
            self.tik_instance.data_move(self.data_out_align[task_idx * self.num_per_batch],
                                        input_ub_tmp[dest_pos_ub], 0, 1, repeat_times, 0, 0)

    def moveout_large_num_new(self, ws_offset, task_idx):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32, for 910B & 310B.
        """
        with self.tik_instance.new_stmt_scope():
            # dest position in UB
            dest_pos_ub = self.num_per_batch * self.num_offset
            repeat_times = self.num_per_batch // self.block_size
            input_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch * self.num_offset * 2],
                                                name="input_ub", scope=tbe_platform.scope_ubuf)

            with self.tik_instance.for_range(0, self.batch_per_task) as i:
                self.tik_instance.data_move(input_ub[0],
                                            self.tmp_workspace[ws_offset + self.num_per_batch * i * self.num_offset],
                                            0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)
                # ascend
                if self.descending is False:
                    # data is continuous in GM & gather scattered data together
                    with self.tik_instance.for_range(0, self.num_per_batch) as i2:
                        input_ub[i2 + dest_pos_ub].set_as(input_ub[(self.num_per_batch - 1 - i2) * self.align])

                    # move output (float16) from UB to GM
                    self.tik_instance.data_move(self.data_out_align[
                                                    task_idx * self.num_align + self.num_per_batch * (
                                                            self.batch_per_task - 1 - i)],
                                                input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)

                # descend
                else:
                    # move output from UB to GM
                    if self.dtype == "float16":
                        self.tik_instance.vreducev2(None, input_ub[dest_pos_ub], input_ub[0], 3,
                                                    self.num_per_batch // 32, 1, 8, 0)
                    else:
                        self.tik_instance.vreducev2(None, input_ub[dest_pos_ub], input_ub[0], 1,
                                                    self.num_per_batch // 32, 1, 8, 0)

                    self.tik_instance.data_move(
                        self.data_out_align[task_idx * self.num_align + self.num_per_batch * i],
                        input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)

    def tune(self):
        """
        Function: remove min.
        """
        if self.num_per_task <= self.num_block:
            repeat_times = self.num_per_batch // self.block_size
            float_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch], name="float_ub",
                                                scope=tbe_platform.scope_ubuf)
            with self.tik_instance.for_range(0, self.task_num) as i:
                self.tik_instance.data_move(float_ub[0], self.data_out_align[i * self.num_per_batch], 0, 1,
                                            repeat_times, 0, 0)
                self.tik_instance.data_move(self.data_out[i * self.num_per_task], float_ub[0], 0, 1, repeat_times,
                                            0, 0)

        else:
            repeat_times = self.num_block // self.block_size
            float_ub = self.tik_instance.Tensor(self.dtype, [self.num_block], name="float_ub",
                                                scope=tbe_platform.scope_ubuf)
            with self.tik_instance.for_range(0, self.task_num) as i:
                with self.tik_instance.for_range(0, self.batch_per_task) as j:
                    self.tik_instance.data_move(float_ub[0],
                                                self.data_out_align[i * self.num_align + j * self.num_block],
                                                0, 1, repeat_times, 0, 0)
                    self.tik_instance.data_move(self.data_out[i * self.num_per_task + j * self.num_block],
                                                float_ub[0], 0, 1, repeat_times, 0, 0)


# 'pylint: disable=too-few-public-methods
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sort_v2(x, y, axis=-1, descending=False, kernel_name="sort_v2"):
    """
    Function: Sorts the elements of the input tensor along a given dimension in ascending order by value.

    Init base parameters
    Parameters
    ----------
    input(x): dict
        data of input
    output(y): dict
        data of output
    dim(axis): int
    descending: bool
    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = SortSimple(x, y, axis, descending, kernel_name)
    return op_obj.sort_compute()
