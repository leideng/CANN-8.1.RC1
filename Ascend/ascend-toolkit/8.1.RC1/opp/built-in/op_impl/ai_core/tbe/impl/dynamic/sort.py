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
# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import OpPatternMode


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
    # api arg
    REPEAT_MAX = 255
    # tiling args
    TILING_ALIGN = 8
    # MAX_INT32
    WORKSPACE_SIZE = 2 ** 31 - 1
    # for 32Byte align
    BLOCK = 8


# pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
class Sort(object):
    def __init__(self, x, descending, kernel_name):
        """__init__"""
        self.kernel_name = kernel_name
        self.cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        self.is_old_version = True if not tbe_platform.api_check_support("tik.vreducev2") else False
        self.dtype = self.check(x, kernel_name)
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.align = 4 if self.dtype == "float16" else 2
        self.bytes = 2 if self.dtype == "float16" else 4
        self.block_size = 16 if self.dtype == "float16" else 8
        self.num_offset = Constant.PROPOSAL_NUM if self.is_old_version else self.align
        self.min = -65504 if self.dtype == "float16" else -3.4e38
        self.max = 65504 if self.dtype == "float16" else 3.4e38
        self.sort_size = 16 if self.is_old_version else 32
        self.struce_len = 16 if self.is_old_version else 8
        self.mask = 128 if self.dtype == "float16" else 64
        self.num_block = 2048 if self.is_old_version else 4096

        self.descending = descending
        self.available_core_num = tik.Dprofile().get_aicore_num()

        self.input_gm = self.tik_instance.Tensor(self.dtype, [Constant.WORKSPACE_SIZE], name="input_gm",
                                                 scope=tik.scope_gm)
        self.data_out = self.tik_instance.Tensor(self.dtype, [Constant.WORKSPACE_SIZE], name="data_out",
                                                 scope=tik.scope_gm)
        self.data_indices = self.tik_instance.Tensor("int32", [Constant.WORKSPACE_SIZE], name="data_indices",
                                                     scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int32", [Constant.TILING_ALIGN], name="tiling_gm",
                                                  scope=tik.scope_gm)

        self.data_out_align = self.tik_instance.Tensor(self.dtype, [Constant.WORKSPACE_SIZE],
                                                       name="data_out_align",
                                                       scope=tik.scope_gm, is_workspace=True)
        self.data_indices_align = self.tik_instance.Tensor("int32", [Constant.WORKSPACE_SIZE],
                                                           name="data_indices_align",
                                                           scope=tik.scope_gm, is_workspace=True)
        self.tmp_workspace = self.tik_instance.Tensor(self.dtype, [Constant.WORKSPACE_SIZE],
                                                      name="tmp_workspace",
                                                      scope=tik.scope_gm, is_workspace=True)
        tiling_ub = self.tik_instance.Tensor("int32", [Constant.TILING_ALIGN],
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)

        self.task_num = self.tik_instance.Scalar("int32", name="task_num")
        self.num_per_task = self.tik_instance.Scalar("int32", name="num_per_task")

        self.num_align = self.tik_instance.Scalar("int32", name="num_align")
        self.batch_per_task = self.tik_instance.Scalar("int32", name="batch_per_task")
        self.num_mini_align = self.tik_instance.Scalar("int32", name="num_mini_align")
        self.num_per_batch = self.tik_instance.Scalar("int32", name="num_per_batch", init_value=self.num_block)

        self.task_num.set_as(tiling_ub[0])
        self.num_per_task.set_as(tiling_ub[1])

        self.num_align.set_as((self.num_per_task + self.num_block - 1) // self.num_block * self.num_block)
        self.batch_per_task.set_as(self.num_align // self.num_block)
        self.num_mini_align.set_as((self.num_per_task + self.sort_size - 1) // self.sort_size * self.sort_size)

        with self.tik_instance.if_scope(self.num_per_task <= self.num_block):
            self.num_per_batch.set_as(self.num_mini_align)

        self.data_a_list = self.tik_instance.Tensor(self.dtype, [Constant.WORKSPACE_SIZE],
                                                    name="data_a_list", scope=tik.scope_gm, is_workspace=True)

    def check(self, x, kernel_name):
        """
        Function: Check parameters (eg: shape dtype etc).
        """
        para_check.check_kernel_name(kernel_name)

        dtype = x.get("dtype").lower()
        para_check.check_dtype_rule(dtype, ("float16", "float32"))

        return dtype

    def sort_compute(self):
        """
        Function: sort compute.
        """
        batch_num_per_aicore = self.tik_instance.Scalar("int32", init_value=self.task_num // self.available_core_num)
        batch_tail = self.tik_instance.Scalar("int32", init_value=self.task_num % self.available_core_num)

        with self.tik_instance.for_range(0, self.available_core_num, block_num=self.available_core_num) as i_idx:
            with self.tik_instance.for_range(0, batch_num_per_aicore) as j_idx:
                self.task_schedule(i_idx + j_idx * self.available_core_num)
            with self.tik_instance.if_scope(i_idx < batch_tail):
                self.task_schedule(batch_num_per_aicore * self.available_core_num + i_idx)
            if self.is_old_version:
                with self.tik_instance.if_scope(tik.any(self.num_per_task > self.num_block, self.task_num > 1)):
                    self.tune()
            else:
                self.tune()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_gm],
                                   outputs=[self.data_out, self.data_indices], flowtable=[self.tiling_gm],
                                   config=opt_config)

        # distinguish between tik and dsl
        tbe_context.get_context().add_compile_info("is_tik", True)
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.available_core_num,
            "num_block": self.num_block,
            "num_offset": self.num_offset,
            "float_bytes": self.bytes
        })
        return self.tik_instance

    def task_schedule(self, task_idx):
        """
        Function: sort compute task schedule, new for 910B & 310B.
        """
        if self.is_old_version:
            with self.tik_instance.if_scope(self.num_per_task <= self.num_block):
                self.sort_mini_num(task_idx)
            with self.tik_instance.else_scope():
                self.sort_large_num(task_idx)
        else:
            with self.tik_instance.if_scope(self.num_per_task <= self.num_block):
                self.sort_mini_num_new(task_idx)
            with self.tik_instance.else_scope():
                self.sort_large_num_new(task_idx)

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

        self.tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], self.num_per_batch // self.sort_size,
                                  Constant.VAL_IDX)

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
        self.tik_instance.vconcat(input_ub[0], idx_ub[0], self.num_per_batch // self.sort_size, Constant.INT_IDX)

        # 2. vbs16
        self.tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0],
                                    repeat_times=self.num_per_batch // self.sort_size)
        # 3. vms4
        input_ub = self.vms4(input_ub, dest_pos_ub)
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

        ws_offset = (task_idx % self.available_core_num) * self.batch_per_task * self.num_block * Constant.PROPOSAL_NUM
        ori_offset = task_idx * self.num_per_task

        # SORT IN UB
        with self.tik_instance.for_range(0, self.batch_per_task) as batch_idx:
            self.sort_in_ub(idx_ub, tmp_ub, batch_idx, ori_offset, ws_offset)

        # SORT IN GM
        input_ub = self.tik_instance.Tensor(self.dtype, [self.num_block * 2 * Constant.PROPOSAL_NUM * 2],
                                            name="input_ub", scope=tik.scope_ubuf)
        self.sort_in_gm(input_ub, ws_offset)

        # Pick Data from GM to GM
        self.moveout_large_num(ws_offset, task_idx, input_ub)

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
        idx_ub = self.tik_instance.Tensor("uint32", [self.num_per_batch], name="idx_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.num_per_batch) as i2:
            idx_ub[i2].set_as(i2)

        # dest position in UB
        repeat_times = self.num_per_batch // self.block_size
        # 1. Move data from OUT to UB

        self.tik_instance.data_move(val_ub, self.input_gm[in_offset], 0, 1, repeat_times, 0, 0)
        dest_pos_ub = self.num_per_batch * self.num_offset

        # aline for k
        aline = self.num_per_batch - self.num_per_task % self.num_per_batch
        tmp_val = self.tik_instance.Scalar(self.dtype)
        if self.descending:
            tmp_val.set_as(self.min)
        else:
            tmp_val.set_as(self.max)
        # Add ineffective object for 16 alignment
        with self.tik_instance.for_range(0, aline) as j:
            val_ub[self.num_per_task % self.num_per_batch + j].set_as(tmp_val)

        # 2. vbs32
        self.tik_instance.vsort32(input_ub_tmp[dest_pos_ub], val_ub, idx_ub, self.num_per_batch // self.sort_size)

        # 3. vms4
        input_ub_tmp = self.vms4(input_ub_tmp, dest_pos_ub)

        self.moveout_mini_num_new(task_idx, input_ub_tmp, repeat_times, dest_pos_ub)

    def sort_large_num_new(self, task_idx):
        """
        Function: fix num_per_task more than 4096, for 910B & 310B.
        """
        # offset in self.input_gm
        in_offset = task_idx * self.num_per_task

        # offset in self.tmp_workspace
        ws_offset = (task_idx % self.available_core_num) * self.num_align * self.num_offset
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
        # list_num 2048//16 = 128
        list_num = self.tik_instance.Scalar(dtype="int32", init_value=self.num_per_batch // self.sort_size)

        # level 0: list_num == 1
        with self.tik_instance.if_scope(tik.all(list_num > 1, list_num <= 128)):
            # level 1: 2-4
            with self.tik_instance.if_scope(list_num <= 4):
                input_ub = self.vms4_level_1(input_ub, dest_pos_ub, list_num)
            with self.tik_instance.else_scope():
                # level 2: 5-16
                with self.tik_instance.if_scope(list_num <= 16):
                    input_ub = self.vms4_level_2(input_ub, dest_pos_ub, list_num)
                with self.tik_instance.else_scope():
                    # level 2: 17-64
                    with self.tik_instance.if_scope(list_num <= 64):
                        input_ub = self.vms4_level_3(input_ub, dest_pos_ub, list_num)
                    # level 3: 65-128
                    with self.tik_instance.else_scope():
                        input_ub = self.vms4_level_4(input_ub, dest_pos_ub, list_num)

        return input_ub

    def vms4_level_1(self, input_ub, dest_pos_ub, list_num):
        """
        Function: Merge 2-4 lists in UB.
        """
        # part 1 64 align
        src_pos_ub = dest_pos_ub
        dest_pos_ub = 0
        offset = 0
        num_per_list = self.sort_size

        with self.tik_instance.if_scope(list_num == 2):
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub], input_ub[src_pos_ub]]
            src_list_lengths = [num_per_list, num_per_list, 0, 0]

            input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)
        with self.tik_instance.if_scope(list_num == 3):
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub]]
            src_list_lengths = [num_per_list, num_per_list, num_per_list, 0]

            input_ub = self.merge3(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)
        with self.tik_instance.if_scope(list_num == 4):
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]
            src_list_lengths = [num_per_list, num_per_list, num_per_list, num_per_list]

            input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        self.tik_instance.data_move(input_ub[src_pos_ub], input_ub[dest_pos_ub], 0, 1,
                                    self.num_per_batch * self.struce_len // 32, 0, 0)

        return input_ub

    def vms4_level_2(self, input_ub, dest_pos_ub, list_num):
        """
        Function: Merge 5-16 lists in UB.
        """
        # part 1 64 align
        src_pos_ub = dest_pos_ub
        dest_pos_ub = 0
        offset = 0
        num_per_list = self.sort_size
        with self.tik_instance.for_range(0, list_num // 4) as loop_idx:
            offset = loop_idx * 4 * num_per_list
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]
            src_list_lengths = [num_per_list, num_per_list, num_per_list, num_per_list]

            input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        res_num = list_num % 4
        with self.tik_instance.if_scope(res_num > 0):
            offset = (list_num - res_num) * self.sort_size
            with self.tik_instance.if_scope(res_num == 1):
                self.tik_instance.data_move(input_ub[dest_pos_ub + offset * self.num_offset],
                                            input_ub[src_pos_ub + offset * self.num_offset], 0, 1,
                                            num_per_list * self.struce_len // 32, 0, 0)

            with self.tik_instance.if_scope(res_num == 2):
                src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                            input_ub[src_pos_ub], input_ub[src_pos_ub]]
                src_list_lengths = [num_per_list, num_per_list, 0, 0]

                input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

            with self.tik_instance.if_scope(res_num == 3):
                src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                            input_ub[src_pos_ub]]
                src_list_lengths = [num_per_list, num_per_list, num_per_list, 0]

                input_ub = self.merge3(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        # part 2 256 align
        dest_pos_ub = src_pos_ub
        src_pos_ub = 0

        offset = 0
        num_per_list = self.sort_size * 4
        res_list = (list_num + 3) // 4

        with self.tik_instance.if_scope(res_list == 2):
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub], input_ub[src_pos_ub]]
            src_list_lengths = [num_per_list, (list_num - 4) * self.sort_size, 0, 0]

            input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        with self.tik_instance.if_scope(res_list == 3):
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub]]
            src_list_lengths = [num_per_list, num_per_list, (list_num - 8) * self.sort_size, 0]

            input_ub = self.merge3(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        with self.tik_instance.if_scope(res_list == 4):
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]
            src_list_lengths = [num_per_list, num_per_list, num_per_list, (list_num - 12) * self.sort_size]

            input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        return input_ub

    def vms4_level_3(self, input_ub, dest_pos_ub, list_num):
        """
        Function: Merge 17-64 lists in UB.
        """
        # part 1 64 align
        src_pos_ub = dest_pos_ub
        dest_pos_ub = 0
        offset = 0
        num_per_list = self.sort_size
        with self.tik_instance.for_range(0, list_num // 4) as loop_idx:
            offset = loop_idx * 4 * num_per_list
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]
            src_list_lengths = [num_per_list, num_per_list, num_per_list, num_per_list]

            input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        res_num = list_num % 4
        with self.tik_instance.if_scope(res_num > 0):
            offset = (list_num - res_num) * self.sort_size
            with self.tik_instance.if_scope(res_num == 1):
                self.tik_instance.data_move(input_ub[dest_pos_ub + offset * self.num_offset],
                                            input_ub[src_pos_ub + offset * self.num_offset], 0, 1,
                                            num_per_list * self.struce_len // 32, 0, 0)
            with self.tik_instance.else_scope():
                src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                            input_ub[src_pos_ub]]
                src_list_lengths = [num_per_list, num_per_list, num_per_list, 0]
                with self.tik_instance.if_scope(res_num == 2):
                    input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)
                with self.tik_instance.else_scope():
                    input_ub = self.merge3(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        # part 2 256 align
        dest_pos_ub = src_pos_ub
        src_pos_ub = 0
        offset = 0
        num_per_list = self.sort_size * 4

        with self.tik_instance.for_range(0, list_num // 16) as loop_idx:
            offset = loop_idx * 4 * num_per_list
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]
            src_list_lengths = [num_per_list, num_per_list, num_per_list, num_per_list]

            input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        res_num = list_num % 16

        with self.tik_instance.if_scope(res_num > 0):
            offset = (list_num - res_num) * self.sort_size

            with self.tik_instance.if_scope(res_num <= 4):
                self.tik_instance.data_move(input_ub[dest_pos_ub + offset * self.num_offset],
                                            input_ub[src_pos_ub + offset * self.num_offset], 0, 1,
                                            self.sort_size * res_num * self.struce_len // 32, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(res_num <= 8):
                    src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                                input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                                input_ub[src_pos_ub], input_ub[src_pos_ub]]

                    src_list_lengths = [num_per_list, self.sort_size * (res_num - 4), 0, 0]

                    input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(res_num <= 12):
                        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                                    input_ub[src_pos_ub]]

                        src_list_lengths = [num_per_list, num_per_list, self.sort_size * (res_num - 8), 0]
                        input_ub = self.merge3(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)
                    with self.tik_instance.else_scope():
                        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]

                        src_list_lengths = [num_per_list, num_per_list, num_per_list, self.sort_size * (res_num - 12)]
                        input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        # part 3 1024 align
        src_pos_ub = dest_pos_ub
        dest_pos_ub = 0
        offset = 0
        num_per_list = self.sort_size * 16
        res_list = (list_num + 15) // 16

        with self.tik_instance.if_scope(res_list == 2):
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub], input_ub[src_pos_ub]]
            src_list_lengths = [num_per_list, (list_num - 16) * self.sort_size, 0, 0]

            input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        with self.tik_instance.if_scope(res_list == 3):
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub]]
            src_list_lengths = [num_per_list, num_per_list, (list_num - 32) * self.sort_size, 0]

            input_ub = self.merge3(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        with self.tik_instance.if_scope(res_list == 4):
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]
            src_list_lengths = [num_per_list, num_per_list, num_per_list, (list_num - 48) * self.sort_size]

            input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        self.tik_instance.data_move(input_ub[src_pos_ub], input_ub[dest_pos_ub], 0, 1,
                                    self.num_per_batch * self.struce_len // 32, 0, 0)

        return input_ub

    def vms4_level_4(self, input_ub, dest_pos_ub, list_num):
        """
        Function: Merge 65-128 lists in UB.
        """
        # part 1 64 align
        src_pos_ub = dest_pos_ub
        dest_pos_ub = 0
        offset = 0
        num_per_list = self.sort_size

        with self.tik_instance.for_range(0, list_num // 4) as loop_idx:
            offset = loop_idx * 4 * num_per_list
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]
            src_list_lengths = [num_per_list, num_per_list, num_per_list, num_per_list]

            input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        res_num = list_num % 4
        with self.tik_instance.if_scope(res_num > 0):
            offset = (list_num - res_num) * self.sort_size
            with self.tik_instance.if_scope(res_num == 1):
                self.tik_instance.data_move(input_ub[dest_pos_ub + offset * self.num_offset],
                                            input_ub[src_pos_ub + offset * self.num_offset], 0, 1,
                                            num_per_list * self.struce_len // 32, 0, 0)
            with self.tik_instance.else_scope():
                src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                            input_ub[src_pos_ub]]
                src_list_lengths = [num_per_list, num_per_list, num_per_list, 0]
                with self.tik_instance.if_scope(res_num == 2):
                    input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)
                with self.tik_instance.else_scope():
                    input_ub = self.merge3(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        # part 2 256 align
        dest_pos_ub = src_pos_ub
        src_pos_ub = 0
        offset = 0
        num_per_list = self.sort_size * 4

        with self.tik_instance.for_range(0, list_num // 16) as loop_idx:
            offset = loop_idx * 4 * num_per_list
            src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                        input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]
            src_list_lengths = [num_per_list, num_per_list, num_per_list, num_per_list]

            input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        res_num = list_num % 16

        with self.tik_instance.if_scope(res_num > 0):
            offset = (list_num - res_num) * self.sort_size

            with self.tik_instance.if_scope(res_num <= 4):
                self.tik_instance.data_move(input_ub[dest_pos_ub + offset * self.num_offset],
                                            input_ub[src_pos_ub + offset * self.num_offset], 0, 1,
                                            self.sort_size * res_num * self.struce_len // 32, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(res_num <= 8):
                    src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                                input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                                input_ub[src_pos_ub], input_ub[src_pos_ub]]

                    src_list_lengths = [num_per_list, self.sort_size * (res_num - 4), 0, 0]

                    input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(res_num <= 12):
                        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                                    input_ub[src_pos_ub]]

                        src_list_lengths = [num_per_list, num_per_list, self.sort_size * (res_num - 8), 0]
                        input_ub = self.merge3(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)
                    with self.tik_instance.else_scope():
                        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                                    input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]

                        src_list_lengths = [num_per_list, num_per_list, num_per_list, self.sort_size * (res_num - 12)]
                        input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        # part 3 1024 align
        src_pos_ub = dest_pos_ub
        dest_pos_ub = 0
        offset = 0
        num_per_list = self.sort_size * 16

        src_list = [input_ub[src_pos_ub],
                    input_ub[src_pos_ub + num_per_list * self.num_offset],
                    input_ub[src_pos_ub + num_per_list * 2 * self.num_offset],
                    input_ub[src_pos_ub + num_per_list * 3 * self.num_offset]]
        src_list_lengths = [num_per_list, num_per_list, num_per_list, num_per_list]

        input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        res_num = list_num % 64
        with self.tik_instance.if_scope(res_num > 0):
            offset = (list_num - res_num) * self.sort_size

            with self.tik_instance.if_scope(res_num <= 16):
                self.tik_instance.data_move(input_ub[dest_pos_ub + offset * self.num_offset],
                                            input_ub[src_pos_ub + offset * self.num_offset], 0, 1,
                                            self.sort_size * res_num * self.struce_len // 32, 0, 0)

            with self.tik_instance.if_scope(tik.all(res_num > 16, res_num <= 32)):
                src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                            input_ub[src_pos_ub], input_ub[src_pos_ub]]
                src_list_lengths = [num_per_list, self.sort_size * (res_num - 16), 0, 0]

                input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

            with self.tik_instance.if_scope(tik.all(res_num > 32, res_num <= 48)):
                src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                            input_ub[src_pos_ub]]
                src_list_lengths = [num_per_list, num_per_list, self.sort_size * (res_num - 32), 0]

                input_ub = self.merge3(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

            with self.tik_instance.if_scope(tik.all(res_num > 48, res_num < 64)):
                src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list * 2) * self.num_offset],
                            input_ub[src_pos_ub + (offset + num_per_list * 3) * self.num_offset]]
                src_list_lengths = [num_per_list, num_per_list, num_per_list, self.sort_size * (res_num - 48)]

                input_ub = self.merge4(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        # part 4 align
        dest_pos_ub = src_pos_ub
        src_pos_ub = 0
        offset = 0
        num_per_list = self.sort_size * 64

        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_per_list) * self.num_offset],
                    input_ub[src_pos_ub], input_ub[src_pos_ub]]
        src_list_lengths = [num_per_list, (list_num - 64) * self.sort_size, 0, 0]

        input_ub = self.merge2(input_ub, offset, src_list, src_list_lengths, dest_pos_ub)

        return input_ub

    def merge4(self, input_ub, offset, src_list, src_list_lengths, dest_pos_ub):
        """
        Function: Merge 4 lists in UB.
        """
        # merge 4 lists
        if self.is_old_version:
            self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                        if_exhausted_suspension=False, valid_bit="1111", repeat_times=1)
        else:
            self.tik_instance.vmrgsort(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                       if_exhausted_suspension=False, repeat_times=1)

        return input_ub

    def merge3(self, input_ub, offset, src_list, src_list_lengths, dest_pos_ub):
        """
        Function: Merge 3 lists in UB.
        """
        # merge 3 lists
        if self.is_old_version:
            self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                        if_exhausted_suspension=False, valid_bit="0111", repeat_times=1)
        else:
            self.tik_instance.vmrgsort(input_ub[dest_pos_ub + offset * self.num_offset], src_list[0:3],
                                       src_list_lengths[0:3], if_exhausted_suspension=False, repeat_times=1)

        return input_ub

    def merge2(self, input_ub, offset, src_list, src_list_lengths, dest_pos_ub):
        """
        Function: Merge 2 lists in UB.
        """

        # merge 2 lists
        if self.is_old_version:
            self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                        if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)
        else:
            self.tik_instance.vmrgsort(input_ub[dest_pos_ub + offset * self.num_offset], src_list[0:2],
                                       src_list_lengths[0:2], if_exhausted_suspension=False, repeat_times=1)

        return input_ub

    def sort_in_ub(self, idx_ub, tmp_ub, batch_idx, ori_offset, ws_offset):
        """
        Function: sort in ub.
        """
        input_ub = self.tik_instance.Tensor(self.dtype, [self.num_block * Constant.PROPOSAL_NUM * 2],
                                            name="input_ub", scope=tik.scope_ubuf)
        # dest position in UB
        dest_pos_ub = self.num_block * Constant.PROPOSAL_NUM
        repeat_times = self.num_block // self.block_size
        # 1. Move data from OUT to UB
        self.tik_instance.data_move(input_ub[dest_pos_ub], self.input_gm[ori_offset + batch_idx * self.num_block],
                                    0, 1, repeat_times, 0, 0)

        temp_float16_ub = self.tik_instance.Tensor("float16", [Constant.BLOCK], name="temp_float16_ub",
                                                   scope=tik.scope_ubuf)
        temp_ub_int = self.tik_instance.Tensor("int32", [Constant.BLOCK], name="temp_ub_int", scope=tik.scope_ubuf)
        temp_ub_int[0].set_as(batch_idx)
        self.tik_instance.vec_conv(1, "none", temp_float16_ub, temp_ub_int, 1, 0, 0, deqscale=1.0)
        temp_scalar = self.tik_instance.Scalar("float16", name="temp_scalar")
        temp_scalar.set_as(temp_float16_ub[0])

        self.tik_instance.vec_dup(self.block_size, tmp_ub, temp_scalar, repeat_times, 1)
        self.tik_instance.vconcat(input_ub, tmp_ub, repeat_times, Constant.INT_IDX)
        self.tik_instance.vconcat(input_ub, idx_ub, repeat_times, Constant.REM_IDX)

        with self.tik_instance.if_scope(self.num_per_task < (batch_idx + 1) * self.num_block):
            # aline for NUM_BLOCK
            aline = self.num_block - self.num_per_task % self.num_block
            tmp = self.tik_instance.Scalar('float16', init_value=self.min)
            # Add ineffective object for 16 alignment
            with self.tik_instance.for_range(0, aline % self.block_size) as j:
                input_ub[dest_pos_ub + self.num_per_task % self.num_block + j].set_as(tmp)
            # Add ineffective object for NUM_BLOCK alignment
            with self.tik_instance.if_scope(aline > self.block_size - 1):
                self.tik_instance.vec_dup(self.block_size, input_ub[
                    dest_pos_ub + self.num_per_task % self.num_block + aline % self.block_size],
                                          tmp, aline // self.block_size, 1)

        self.tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], self.num_block // self.sort_size,
                                  Constant.VAL_IDX)
        # 2. vrpsort16
        self.tik_instance.vrpsort16(input_ub[dest_pos_ub], input_ub[0], self.num_block // self.sort_size)
        # 3. vms4
        input_ub = self.vms4_level_4(input_ub, dest_pos_ub, 128)
        # 4. Move Data from UB to OUT
        self.tik_instance.data_move(self.tmp_workspace[ws_offset + batch_idx * self.num_block * Constant.PROPOSAL_NUM],
                                    input_ub[dest_pos_ub], 0, 1,
                                    self.num_block * Constant.PROPOSAL_NUM // self.block_size, 0, 0)

    def sort_in_gm(self, input_ub, offset):
        """
        Function: sort in gm.
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
                with self.tik_instance.if_scope(aline >= self.block_size):
                    with self.tik_instance.if_scope(aline >= repeat_num_max):
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
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_dup(self.block_size,
                                                  val_ub[self.num_per_task % self.num_per_batch +
                                                         aline % self.block_size],
                                                  tmp_val, aline // self.block_size, 1)

            # 2. vbs32
            self.tik_instance.vsort32(input_ub_tmp[dest_pos_ub], val_ub, idx_ub, self.num_per_batch // self.sort_size)

            # 3. vms4
            input_ub_tmp = self.vms4_level_4(input_ub_tmp, dest_pos_ub, 128)

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
                                                name="input_ub", scope=tik.scope_ubuf)
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
        int_list = self.tik_instance.Tensor("int32", [self.num_per_batch], name="int_list", scope=tik.scope_ubuf)
        src_pos_ub = self.num_per_batch * Constant.PROPOSAL_NUM - dest_pos_ub
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
        with self.tik_instance.if_scope(self.task_num > 1):
            self.tik_instance.data_move(self.data_out_align[offset_out], input_ub[src_pos_ub], 0, 1,
                                        self.num_per_batch // self.block_size, 0, 0)
            self.tik_instance.data_move(self.data_indices_align[offset_out], int_list, 0, 1,
                                        2 * self.num_per_batch // self.block_size, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.data_out[offset_out], input_ub[src_pos_ub], 0, 1,
                                        self.num_per_batch // self.block_size, 0, 0)
            self.tik_instance.data_move(self.data_indices[offset_out], int_list, 0, 1,
                                        2 * self.num_per_batch // self.block_size, 0, 0)

    def moveout_large_num(self, offset, task_idx, input_ub):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32.
        """
        # dest position in UB
        dest_pos_ub = self.num_block * Constant.PROPOSAL_NUM
        repeat_times = self.num_block // self.block_size

        ascend_tail_num = self.tik_instance.Scalar("int32", name="ascend_tail_num")
        ascend_tail_num.set_as(self.num_per_task - self.num_align + self.num_block)
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

            self.tik_instance.vec_dup(self.block_size, int_list_4, self.num_block, repeat_times,
                                      self.block_size // Constant.BLOCK)

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
                with self.tik_instance.if_scope(i < self.batch_per_task - 1):
                    with self.tik_instance.for_range(0, self.num_block) as i2:
                        int_list_2[i2].set_as(int_list_1[self.num_block - i2 - 1])
                        input_ub[dest_pos_ub + self.num_block + i2].set_as(
                            input_ub[dest_pos_ub + self.num_block - i2 - 1])
                    self.tik_instance.data_move(
                        self.data_indices_align[
                            task_idx * self.num_align + self.num_block * (
                                        self.batch_per_task - i - 1) - self.num_align + self.num_per_task],
                        int_list_2, 0, 1, 2 * repeat_times, 0, 0)
                    self.tik_instance.data_move(
                        self.data_out_align[task_idx * self.num_align + self.num_block * (
                                    self.batch_per_task - i - 1) - self.num_align + self.num_per_task],
                        input_ub[dest_pos_ub + self.num_block], 0, 1, repeat_times, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        int_list_2, self.data_indices_align[task_idx * self.num_align], 0, 1, 2 * repeat_times, 0, 0)
                    self.tik_instance.data_move(
                        input_ub[dest_pos_ub + self.num_block], self.data_out_align[task_idx * self.num_align], 0, 1,
                        repeat_times, 0, 0)
                    ascend_tail_num.set_as(self.num_per_task - self.num_align + self.num_block)
                    with self.tik_instance.for_range(0, ascend_tail_num) as i2:
                        int_list_2[i2].set_as(int_list_1[ascend_tail_num - i2 - 1])
                        input_ub[dest_pos_ub + self.num_block + i2].set_as(
                            input_ub[dest_pos_ub + ascend_tail_num - i2 - 1])
                    self.tik_instance.data_move(
                        self.data_indices_align[
                            task_idx * self.num_align], int_list_2, 0, 1, 2 * repeat_times, 0, 0)
                    self.tik_instance.data_move(
                        self.data_out_align[task_idx * self.num_align], input_ub[dest_pos_ub + self.num_block], 0, 1,
                        repeat_times, 0, 0)
            # descend
            else:
                self.tik_instance.data_move(self.data_indices_align[task_idx * self.num_align + self.num_block * i],
                                            int_list_1, 0, 1, 2 * repeat_times, 0, 0)
                self.tik_instance.data_move(self.data_out_align[task_idx * self.num_align + self.num_block * i],
                                            input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)

    def moveout_mini_num_new(self, task_idx, input_ub_tmp, repeat_times, dest_pos_ub):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32, for 910B & 310B.
        """
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
                                        input_ub_tmp[src_pos_ub], 0, 1, repeat_times, 0, 0)
        # ascend
        else:
            # data is continuous in GM & gather scattered data together
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                tmp.set_as(uint32_ub[2 * (self.num_per_batch - 1 - i2) + 1])
                uint32_ub[self.num_per_batch * 2 + i2].set_as(tmp)

                input_ub_tmp[i2 + dest_pos_ub].set_as(input_ub_tmp[src_pos_ub + (self.num_per_batch - 1 - i2)])

            # move output (float16) from UB to GM
            self.tik_instance.data_move(self.data_out_align[task_idx * self.num_per_batch],
                                        input_ub_tmp[dest_pos_ub], 0, 1, repeat_times, 0, 0)

        self.tik_instance.data_move(self.data_indices_align[task_idx * self.num_per_batch],
                                    uint32_ub[self.num_per_batch * 2], 0, 1, self.num_per_batch // 8, 0, 0)

    def moveout_large_num_new(self, ws_offset, task_idx):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32, for 910B & 310B.
        """
        with self.tik_instance.new_stmt_scope():
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
                    self.tik_instance.data_move(self.data_a_list[task_idx * self.num_per_batch * 4],
                                                input_ub[dest_pos_ub],
                                                0, 1, self.num_per_batch * self.struce_len // 32,
                                                0, 0)
                    data_b = self.data_a_list[task_idx * self.num_per_batch * 4:
                                              (task_idx + 1) * self.num_per_batch * 4].reinterpret_cast_to("uint32")
                else:
                    self.tik_instance.data_move(self.data_a_list[task_idx * self.num_per_batch * 2],
                                                input_ub[dest_pos_ub],
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
        repeat_times = self.num_per_batch // self.block_size
        float_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch], name="float_ub",
                                            scope=tik.scope_ubuf)
        int_ub = self.tik_instance.Tensor("int32", [self.num_per_batch], name="int_ub", scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(self.num_per_task <= self.num_block):
            with self.tik_instance.for_range(0, self.task_num) as i:
                self.tik_instance.data_move(float_ub, self.data_out_align[i * self.num_per_batch], 0, 1,
                                            repeat_times, 0, 0)
                self.tik_instance.data_move(self.data_out[i * self.num_per_task], float_ub, 0, 1, repeat_times,
                                            0, 0)
            with self.tik_instance.for_range(0, self.task_num) as i:
                self.tik_instance.data_move(int_ub, self.data_indices_align[i * self.num_per_batch], 0, 1,
                                            self.num_per_batch // 8, 0, 0)
                self.tik_instance.data_move(self.data_indices[i * self.num_per_task], int_ub, 0, 1,
                                            self.num_per_batch // 8, 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.task_num) as i:
                with self.tik_instance.for_range(0, self.batch_per_task) as j:
                    self.tik_instance.data_move(float_ub,
                                                self.data_out_align[i * self.num_align + j * self.num_per_batch],
                                                0, 1, repeat_times, 0, 0)
                    self.tik_instance.data_move(self.data_out[i * self.num_per_task + j * self.num_per_batch],
                                                float_ub, 0, 1, repeat_times, 0, 0)
                with self.tik_instance.for_range(0, self.batch_per_task) as j:
                    self.tik_instance.data_move(int_ub,
                                                self.data_indices_align[i * self.num_align + j * self.num_per_batch],
                                                0, 1, self.num_per_batch // 8, 0, 0)
                    self.tik_instance.data_move(self.data_indices[i * self.num_per_task + j * self.num_per_batch],
                                                int_ub, 0, 1, self.num_per_batch // 8, 0, 0)


def sort_dsl(x, y1, y2, axis, descending, kernel_name):
    ins = classify([x, axis], OpPatternMode.SORT, {"op_mode": "sort"})
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x], "sort")
            x_input = tvm.placeholder(x_shape, name="data_input", dtype=x["dtype"])
            direction = "descend" if descending else "ascend"
            if x["dtype"] == "bfloat16":
                x_input_fp32 = tbe.cast_to(x_input, "float32")
                value, indices = tbe.sort(x_input_fp32, sort_axis=-1, direction=direction, return_type="both",
                                      indices_dtype=y2["dtype"], need_cast=True)
            else:
                value, indices = tbe.sort(x_input, sort_axis=-1, direction=direction, return_type="both",
                                        indices_dtype=y2["dtype"])
            tensors.append([x_input, value, indices])
        with tvm.target.cce():
            sch = tbe.auto_schedule([value, indices])
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=too-few-public-methods
@register_operator("Sort")
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
    kernel_name: str
        the name of the operator
    ----------
    """
    if axis is None:
        axis = -1
    if tbe_platform.api_check_support("tbe.dsl.sort", "float16"):
        sort_dsl(x, y1, y2, axis, descending, kernel_name)
    else:
        op_obj = Sort(x, descending, kernel_name)
        return op_obj.sort_compute()
