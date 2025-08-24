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
sort
"""

from impl.sort_v2 import Constant
from impl.sort_v2 import SortSimple
from impl.util.platform_adapter import para_check
# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


def check_supported(x, y1, y2, axis, descending, stable, kernel_name="sort"):
    """
    check the op support situation.
    Go to AICPU when the date in sort axis is over 100K.
    """
    input_shape = x.get("shape")
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ["Ascend310", "Ascend310B", "Ascend031"]:
        if input_shape[-1] > Constant.DATA_LIMITE:
            reason = "The data in sort axis is over 100K."
            return False, reason
    else:
        if input_shape[-1] == 1:
            reason = "The data of sort axis is 1."
            return False, reason

    return True, ""


def op_select_format(x, y1, y2, axis=-1, descending=False, stable=False, kernel_name="sort"):
    """
    select format depend on the soc version.
    """
    is_old_version = True if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310", "Ascend910") else False

    op_dtype = "float16" if is_old_version else "float16,float32,bfloat16"
    op_dtype_int = "int32" if is_old_version else "int32,int32,int32"
    op_format = "ND" if is_old_version else "ND,ND,ND"

    input0 = gen_param(classify="input0", name="x", datatype=op_dtype, format=op_format,
                       unknownshape_format=op_format)
    output0 = gen_param(classify="output0", name="y1", datatype=op_dtype, format=op_format,
                        unknownshape_format=op_format)
    output1 = gen_param(classify="output1", name="y2", datatype=op_dtype_int, format=op_format,
                        unknownshape_format=op_format)

    param_list = [input0, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
class Sort(SortSimple):
    def __init__(self, x, y1, y2, axis, descending, kernel_name):
        """__init__"""
        super().__init__(x, y1, axis, descending, kernel_name)
        self.data_indices = self.tik_instance.Tensor("int32", self.shape, name="data_indices", scope=tik.scope_gm)

        if self.batch_per_task == 1:
            self.data_indices_align = self.tik_instance.Tensor("int32", [self.task_num * self.num_per_batch],
                                                               name="data_indices_align",
                                                               scope=tik.scope_gm, is_workspace=True)
        else:
            self.data_indices_align = self.tik_instance.Tensor("int32", [self.task_num * self.num_align],
                                                               name="data_indices_align",
                                                               scope=tik.scope_gm, is_workspace=True)
        if not self.is_old_version:
            if self.dtype == "float16":
                self.data_a_list = self.tik_instance.Tensor(self.dtype, [self.task_num * self.num_per_batch * 4],
                                                            name="data_a_list", scope=tik.scope_gm, is_workspace=True)
            else:
                self.data_a_list = self.tik_instance.Tensor(self.dtype, [self.task_num * self.num_per_batch * 2],
                                                            name="data_a_list", scope=tik.scope_gm, is_workspace=True)

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
                                   outputs=[self.data_out, self.data_indices])

        return self.tik_instance

    def sort_mini_num(self, task_idx):
        """
        Function: fix num_per_task less than 2048.
        """
        idx_ub = self.tik_instance.Tensor(self.dtype, [self.num_block], name="idx_ub", scope=tik.scope_ubuf)
        input_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch * Constant.PROPOSAL_NUM * 2],
                                            name="input_ub", scope=tik.scope_ubuf)

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

        if self.cce_product == tbe_platform.ASCEND_310:
            data_out_ub_ = self.tik_instance.Tensor(self.dtype, [self.block_size], name="data_out_ub_",
                                                    scope=tik.scope_ubuf)
            data_indices_ub_int_ = self.tik_instance.Tensor("int32", [self.block_size], name="data_indices_ub_int_",
                                                            scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                data_indices_ub_int_.set_as(self.num_per_task - 1 - i2)
                self.tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                idx_ub[(self.num_per_task - 1 - i2)].set_as(data_out_ub_[0])
        else:
            idx = self.tik_instance.Scalar(dtype="float32", init_value=self.num_per_task)
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                idx.set_as(idx - 1)
                idx_ub[(self.num_per_task - 1 - i2)].set_as(idx)
        self.tik_instance.vconcat(input_ub[0], idx_ub[0], n_repeat_total, Constant.INT_IDX)

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
        idx_ub = self.tik_instance.Tensor(self.dtype, [self.num_block], name="idx_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_instance.Tensor(self.dtype, [self.num_block], name="tmp_ub", scope=tik.scope_ubuf)

        if self.cce_product == tbe_platform.ASCEND_310:
            data_out_ub_ = self.tik_instance.Tensor(self.dtype, [self.block_size], name="data_out_ub_",
                                                    scope=tik.scope_ubuf)
            data_indices_ub_int_ = self.tik_instance.Tensor("int32", [self.block_size], name="data_indices_ub_int_",
                                                            scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.num_block // 2) as i2:
                data_indices_ub_int_.set_as(i2)
                self.tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                idx_ub[i2].set_as(data_out_ub_[0])
        else:
            idx = self.tik_instance.Scalar(dtype="float32", init_value=0)
            with self.tik_instance.for_range(0, self.num_block // 2) as i2:
                idx_ub[i2].set_as(idx)
                idx.set_as(idx + 1)

        self.tik_instance.vec_adds(self.mask, idx_ub[self.num_block // 2], idx_ub, 1024.0,
                                   self.num_block // 2 // self.mask, 8, 8)

        offset = (task_idx % self.used_core_num) * self.batch_per_task * self.num_block * Constant.PROPOSAL_NUM
        ori_offset = task_idx * self.num_per_task
        input_ub = self.tik_instance.Tensor(self.dtype, [self.num_block * 2 * Constant.PROPOSAL_NUM * 2],
                                            name="input_ub", scope=tik.scope_ubuf)

        # SORT IN UB
        for i in range(self.batch_per_task):
            self.sort_in_ub(input_ub, idx_ub, tmp_ub, i, ori_offset, offset)

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
                                                name="input_ub_tmp", scope=tik.scope_ubuf)
        val_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch], name="val_ub",
                                          scope=tik.scope_ubuf)
        idx_ub = self.tik_instance.Tensor("uint32", [self.num_per_batch], name="src1_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.num_per_task) as i2:
            idx_ub[i2].set_as(i2)

        # 1. Move data from OUT to UB
        self.tik_instance.data_move(val_ub, self.input_gm[in_offset], 0, 1, self.num_per_batch // self.block_size, 0, 0)
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

        self.moveout_mini_num_new(task_idx, input_ub_tmp, dest_pos_ub)

    def sort_in_ub(self, input_ub, idx_ub, tmp_ub, i, ori_offset, offset):
        """
        Function: sort in ub.
        """
        # dest position in UB
        dest_pos_ub = self.num_block * Constant.PROPOSAL_NUM
        repeat_times = self.num_block // self.block_size
        # 1. Move data from OUT to UB
        self.tik_instance.data_move(input_ub[dest_pos_ub], self.input_gm[ori_offset + i * self.num_block], 0, 1,
                                    repeat_times, 0, 0)

        self.tik_instance.vector_dup(self.block_size, tmp_ub[0], i, repeat_times, 1, 1)
        self.tik_instance.vconcat(input_ub[0], tmp_ub[0], repeat_times, Constant.INT_IDX)
        self.tik_instance.vconcat(input_ub[0], idx_ub[0], repeat_times, Constant.REM_IDX)

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

    def sort_in_ub_new(self, batch_idx, in_offset, ws_offset):
        """
        Function: sort in ub, for 910B & 310B.
        """
        with self.tik_instance.new_stmt_scope():
            input_ub_tmp = self.tik_instance.Tensor(self.dtype, [self.num_per_batch * self.num_offset * 2],
                                                    name="input_ub_tmp", scope=tik.scope_ubuf)
            val_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch], name="val_ub", scope=tik.scope_ubuf)
            idx_ub = self.tik_instance.Tensor("uint32", [self.num_per_batch], name="src1_ub", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, self.num_per_batch) as i2:
                idx_ub[i2].set_as(i2 + batch_idx * self.num_per_batch)

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

    def moveout_mini_num(self, offset_out, input_ub, dest_pos_ub):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32.
        """
        int_list = self.tik_instance.Tensor("int32", [self.num_per_batch], name="int_list", scope=tik.scope_ubuf)
        src_pos_ub = self.num_per_batch * Constant.PROPOSAL_NUM if dest_pos_ub == 0 else 0
        # ascend
        if self.descending is False:
            # data is continuous in GM & gather scattered data together
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                input_ub[i2 + src_pos_ub].set_as(
                    input_ub[(self.num_per_batch - 1 - i2) * Constant.PROPOSAL_NUM + Constant.VAL_IDX + dest_pos_ub])
                input_ub[i2 + src_pos_ub + self.num_per_batch].set_as(
                    input_ub[(self.num_per_batch - 1 - i2) * Constant.PROPOSAL_NUM + dest_pos_ub])

        # descend
        else:
            # data is continuous in GM & gather scattered data together
            if self.cce_product == tbe_platform.ASCEND_310:
                with self.tik_instance.for_range(0, self.num_per_task) as i2:
                    input_ub[i2 + src_pos_ub].set_as(
                        input_ub[i2 * Constant.PROPOSAL_NUM + Constant.VAL_IDX + dest_pos_ub])
                    input_ub[i2 + src_pos_ub + self.num_per_batch].set_as(
                        input_ub[i2 * Constant.PROPOSAL_NUM + dest_pos_ub])
            else:
                self.tik_instance.vextract(input_ub[src_pos_ub], input_ub[dest_pos_ub],
                                           self.num_per_batch // self.block_size, Constant.VAL_IDX)
                self.tik_instance.vextract(input_ub[src_pos_ub + self.num_per_batch], input_ub[dest_pos_ub],
                                           self.num_per_batch // self.block_size, Constant.INT_IDX)

        # conv indices (float16->int32) , and move from UB to GM
        self.tik_instance.vec_conv(self.block_size, "round", int_list, input_ub[src_pos_ub + self.num_per_batch],
                                   self.num_per_batch // self.block_size, 2, 1)

        # move output (float16) from UB to GM
        self.tik_instance.data_move(self.data_out_align[offset_out], input_ub[src_pos_ub], 0, 1,
                                    self.num_per_batch // self.block_size, 0, 0)
        self.tik_instance.data_move(self.data_indices_align[offset_out], int_list, 0, 1,
                                    2 * self.num_per_batch // self.block_size, 0, 0)

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
            int_list_1 = self.tik_instance.Tensor("int32", [self.num_block], name="int_list_1",
                                                  scope=tik.scope_ubuf)
            int_list_2 = self.tik_instance.Tensor("int32", [self.num_block], name="int_list_2",
                                                  scope=tik.scope_ubuf)
            int_list_3 = self.tik_instance.Tensor("int32", [self.num_block], name="int_list_3",
                                                  scope=tik.scope_ubuf)
            int_list_4 = self.tik_instance.Tensor("int32", [self.num_block], name="int_list_4",
                                                  scope=tik.scope_ubuf)

            self.tik_instance.vector_dup(self.block_size, int_list_4, self.num_block, repeat_times, 1, 2)

            self.tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, Constant.INT_IDX)
            self.tik_instance.vextract(input_ub[dest_pos_ub + self.num_block], input_ub[0], repeat_times,
                                       Constant.REM_IDX)
            self.tik_instance.vec_conv(self.block_size, "round", int_list_1, input_ub[dest_pos_ub], repeat_times, 2, 1)
            self.tik_instance.vec_conv(self.block_size, "round", int_list_2, input_ub[dest_pos_ub + self.num_block],
                                       repeat_times, 2, 1)

            self.tik_instance.vec_mul(self.block_size, int_list_3, int_list_1, int_list_4, repeat_times, 2, 2, 2)
            self.tik_instance.vec_add(self.block_size, int_list_1, int_list_2, int_list_3, repeat_times, 2, 2, 2)

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
                    int_list_2[i2].set_as(int_list_1[self.num_block - i2 - 1])
                    input_ub[dest_pos_ub + self.num_block + i2].set_as(
                        input_ub[dest_pos_ub + self.num_block - i2 - 1])
                self.tik_instance.data_move(
                    self.data_indices_align[
                        task_idx * self.num_align + self.num_block * (self.batch_per_task - i - 1)],
                    int_list_2, 0, 1, self.num_block // 8, 0, 0)
                self.tik_instance.data_move(
                    self.data_out_align[task_idx * self.num_align + self.num_block * (self.batch_per_task - i - 1)],
                    input_ub[dest_pos_ub + self.num_block], 0, 1, repeat_times, 0, 0)
            # descend
            else:
                self.tik_instance.data_move(self.data_indices_align[task_idx * self.num_align + self.num_block * i],
                                            int_list_1, 0, 1, self.num_block // 8, 0, 0)
                self.tik_instance.data_move(self.data_out_align[task_idx * self.num_align + self.num_block * i],
                                            input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)

    def moveout_mini_num_new(self, task_idx, input_ub_tmp, dest_pos_ub):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32, for 910B & 310B.
        """
        repeat_times = self.num_per_batch // self.block_size
        src_pos_ub = self.num_per_batch * self.num_offset - dest_pos_ub

        uint32_ub = self.tik_instance.Tensor("uint32", [self.num_per_batch * 3], name="uint32_ub", scope=tik.scope_ubuf)

        if self.dtype == "float16":
            self.tik_instance.data_move(self.data_a_list[task_idx * self.num_per_batch * 4:],
                                        input_ub_tmp[dest_pos_ub], 0, 1, self.num_per_batch * self.struce_len // 32,
                                        0, 0)
            data_b = self.data_a_list[task_idx * self.num_per_batch * 4:].reinterpret_cast_to("uint32")
            self.tik_instance.data_move(uint32_ub, data_b, 0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)
        else:
            self.tik_instance.data_move(self.data_a_list[task_idx * self.num_per_batch * 2:],
                                        input_ub_tmp[dest_pos_ub], 0, 1, self.num_per_batch * self.struce_len // 32,
                                        0, 0)
            data_b = self.data_a_list[task_idx * self.num_per_batch * 2:].reinterpret_cast_to("uint32")
            self.tik_instance.data_move(uint32_ub, data_b, 0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)

        tmp = self.tik_instance.Scalar("uint32")

        # move output from UB to GM
        if self.dtype == "float16":
            self.tik_instance.vreducev2(None, input_ub_tmp[src_pos_ub], input_ub_tmp[dest_pos_ub], 3,
                                        self.num_per_batch // 32, 1, 8, 0)
        else:
            self.tik_instance.vreducev2(None, input_ub_tmp[src_pos_ub], input_ub_tmp[dest_pos_ub], 1,
                                        self.num_per_batch // 32, 1, 8, 0)

        if self.descending:
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                tmp.set_as(uint32_ub[2 * i2 + 1])
                uint32_ub[self.num_per_batch * 2 + i2].set_as(tmp)

            # move output (float16) from UB to GM
            self.tik_instance.data_move(self.data_out_align[task_idx * self.num_per_batch],
                                        input_ub_tmp[src_pos_ub], 0, 1, self.move_out_block, 0, 0)
        # ascend
        else:
            # data is continuous in GM & gather scattered data together
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                tmp.set_as(uint32_ub[2 * (self.num_per_batch - 1 - i2) + 1])
                uint32_ub[self.num_per_batch * 2 + i2].set_as(tmp)

                input_ub_tmp[i2 + dest_pos_ub].set_as(input_ub_tmp[src_pos_ub + (self.num_per_batch - 1 - i2)])

            # move output (float16) from UB to GM
            self.tik_instance.data_move(self.data_out_align[task_idx * self.num_per_batch],
                                        input_ub_tmp[dest_pos_ub], 0, 1, self.move_out_block, 0, 0)

        self.tik_instance.data_move(self.data_indices_align[task_idx * self.num_per_batch],
                                    uint32_ub[self.num_per_batch * 2], 0, 1,
                                    self.num_per_batch_align // Constant.NUM_PER_BLOCK, 0, 0)

    def moveout_large_num_new(self, ws_offset, task_idx):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32, for 910B & 310B.
        """
        # dest position in UB
        dest_pos_ub = self.num_per_batch * self.num_offset
        repeat_times = self.num_per_batch // self.block_size
        input_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch * self.num_offset * 2],
                                            name="input_ub", scope=tik.scope_ubuf)
        uint32_ub = self.tik_instance.Tensor("uint32", [self.num_per_batch * 3], name="uint32_ub",
                                             scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.batch_per_task) as i:
            self.tik_instance.data_move(input_ub[0],
                                        self.tmp_workspace[ws_offset + self.num_per_batch * i * self.num_offset],
                                        0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)

            if self.dtype == "float16":
                self.tik_instance.data_move(self.data_a_list[task_idx * self.num_per_batch * 4], input_ub[dest_pos_ub],
                                            0, 1, self.num_per_batch * self.struce_len // 32,
                                            0, 0)
                data_b = self.data_a_list[task_idx * self.num_per_batch * 4:
                                          (task_idx + 1) * self.num_per_batch * 4].reinterpret_cast_to("uint32")
            else:
                self.tik_instance.data_move(self.data_a_list[task_idx * self.num_per_batch * 2], input_ub[dest_pos_ub],
                                            0, 1, self.num_per_batch * self.struce_len // 32,
                                            0, 0)
                data_b = self.data_a_list[task_idx * self.num_per_batch * 2:
                                          (task_idx + 1) * self.num_per_batch * 2].reinterpret_cast_to("uint32")

            tmp = self.tik_instance.Scalar("uint32")

            self.tik_instance.data_move(uint32_ub, data_b, 0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)

            # move output from UB to GM
            if self.dtype == "float16":
                self.tik_instance.vreducev2(None, input_ub[dest_pos_ub], input_ub[0], 3,
                                            self.num_per_batch // 32, 1, 8, 0)
            else:
                self.tik_instance.vreducev2(None, input_ub[dest_pos_ub], input_ub[0], 1,
                                            self.num_per_batch // 32, 1, 8, 0)

            if self.descending:
                self.tik_instance.data_move(
                    self.data_out_align[task_idx * self.num_align + self.num_per_batch * i],
                    input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)
                # data is continuous in GM & gather scattered data together
                with self.tik_instance.for_range(0, self.num_per_batch) as i2:
                    tmp.set_as(uint32_ub[2 * i2 + 1])
                    uint32_ub[self.num_per_batch * 2 + i2].set_as(tmp)

                self.tik_instance.data_move(self.data_indices_align[
                                                task_idx * self.num_align + self.num_per_batch * i],
                                            uint32_ub[self.num_per_batch * 2], 0, 1, self.num_per_batch // 8, 0, 0)

            # ascend
            else:
                # data is continuous in GM & gather scattered data together
                with self.tik_instance.for_range(0, self.num_per_batch) as i2:
                    input_ub[i2].set_as(input_ub[dest_pos_ub + (self.num_per_batch - 1 - i2)])
                    tmp.set_as(uint32_ub[2 * (self.num_per_batch - 1 - i2) + 1])
                    uint32_ub[self.num_per_batch * 2 + i2].set_as(tmp)

                # move output (float16) from UB to GM
                self.tik_instance.data_move(self.data_out_align[
                                                task_idx * self.num_align + self.num_per_batch * (
                                                        self.batch_per_task - 1 - i)],
                                            input_ub, 0, 1, repeat_times, 0, 0)

                self.tik_instance.data_move(self.data_indices_align[
                                                task_idx * self.num_align + self.num_per_batch * (
                                                        self.batch_per_task - 1 - i)],
                                            uint32_ub[self.num_per_batch * 2], 0, 1, self.num_per_batch // 8, 0, 0)

    def tune(self):
        """
        Function: remove min.
        """
        float_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch], name="float_ub",
                                            scope=tik.scope_ubuf)
        int_ub = self.tik_instance.Tensor("int32", [self.num_per_batch], name="int_ub", scope=tik.scope_ubuf)
        if self.num_per_task <= self.num_block:
            with self.tik_instance.for_range(0, self.task_num) as i:
                self.tik_instance.data_move(float_ub, self.data_out_align[i * self.num_per_batch], 0, 1,
                                            self.move_out_block, 0, 0)
                self.tik_instance.data_move(self.data_out[i * self.num_per_task], float_ub, 0, 1, self.move_out_block,
                                            0, 0)
            with self.tik_instance.for_range(0, self.task_num) as i:
                self.tik_instance.data_move(int_ub, self.data_indices_align[i * self.num_per_batch], 0, 1,
                                            self.num_per_batch_align // Constant.NUM_PER_BLOCK, 0, 0)
                self.tik_instance.data_move(self.data_indices[i * self.num_per_task], int_ub, 0, 1,
                                            self.num_per_batch_align // Constant.NUM_PER_BLOCK, 0, 0)
        else:
            with self.tik_instance.for_range(0, self.task_num) as i:
                with self.tik_instance.for_range(0, self.batch_per_task) as j:
                    self.tik_instance.data_move(float_ub,
                                                self.data_out_align[i * self.num_align + j * self.num_per_batch],
                                                0, 1, self.move_out_block, 0, 0)
                    self.tik_instance.data_move(self.data_out[i * self.num_per_task + j * self.num_per_batch],
                                                float_ub, 0, 1, self.move_out_block, 0, 0)

                with self.tik_instance.for_range(0, self.batch_per_task) as j:
                    self.tik_instance.data_move(int_ub,
                                                self.data_indices_align[i * self.num_align + j * self.num_per_batch],
                                                0, 1, self.num_per_batch_align // Constant.NUM_PER_BLOCK, 0, 0)
                    self.tik_instance.data_move(self.data_indices[i * self.num_per_task + j * self.num_per_batch],
                                                int_ub, 0, 1, self.num_per_batch_align // Constant.NUM_PER_BLOCK, 0, 0)


# 'pylint: disable=too-few-public-methods
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sort(x, y1, y2, axis=-1, descending=False, stable=False, kernel_name="sort"):
    """
    Function: Sorts the elements of the input tensor along a given dimension in ascending order by value.

    Init base parameters
    Parameters
    ----------
    input(x): dict
        data of input
    output(y1): dict
        data of output
    indices(y2): dict
        data of indices
    dim(axis): int
    descending: bool
    stable: bool
    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = Sort(x, y1, y2, axis, descending, kernel_name)
    return op_obj.sort_compute()
