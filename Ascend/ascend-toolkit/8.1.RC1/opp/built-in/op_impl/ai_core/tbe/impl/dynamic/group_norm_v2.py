# Copyright (C) Huawei Technologies Co., Ltd 2023-2023. All rights reserved.
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
group_norm_v2
"""

from impl import common_util
from impl.util.platform_adapter import tik
import tbe.common.platform as tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.platform_adapter import para_check


MAX_INT32 = 2 ** 31 - 1
TILING_NUM = 16
BLOCK_32 = 8
BLOCK_16 = 16
BYTES_PER_BLOCK = 32
MASK_32 = 64
MASK_16 = 128
MAX_REPEAT = 255


def op_select_format(x, scale, offset, y, mean, rstd, num_groups, data_format="NCHW", epsilon=1e-5,
                     is_training=True, kernel_name="group_norm_v2"):
    """
    op_select format func for dynamic format
    """
    dtype_list = ["float16", "float32"]
    format_list0 = ["ND", "ND"]
    format_list1 = ["ND", "ND"]

    soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if soc_version == tbe_platform.ASCEND_910:
        dtype_list = ["float32"]
        format_list0 = ["ND"]
        format_list1 = ["ND"]

    input0 = gen_param(classify="input0", name="x", datatype=",".join(dtype_list), format=",".join(format_list0))
    input1 = gen_param(classify="input1", name="gamma", datatype=",".join(dtype_list), format=",".join(format_list1))
    input2 = gen_param(classify="input2", name="beta", datatype=",".join(dtype_list), format=",".join(format_list1))
    output0 = gen_param(classify="output0", name="y", datatype=",".join(dtype_list), format=",".join(format_list0))
    output1 = gen_param(classify="output1", name="mean", datatype=",".join(dtype_list), format=",".join(format_list1))
    output2 = gen_param(classify="output2", name="rstd", datatype=",".join(dtype_list),
                        format=",".join(format_list1))

    param_dynamic_in_json = get_dynamic_param_in_json([input0, input1, input2, output0, output1, output2])
    return param_dynamic_in_json


# 'pylint: disable=unused-argument,too-many-locals
class GroupNormV2(object):
    """
    object of GroupNormV2
    """
    def __init__(self, x, scale, offset, y, mean, rstd, num_groups, data_format="NCHW", epsilon=1e-5,
                 is_training=True, kernel_name="group_norm_v2"):
        self.tik_instance = tik.Tik()
        self.dtype = x.get("dtype").lower()
        self.kernel_name = kernel_name
        self.tiling_param_dtype = 'int32'
        self.block = BLOCK_32 if self.dtype == "float32" else BLOCK_16
        self.mask = MASK_32 if self.dtype == "float32" else MASK_16
        self.atomic_num = 1 if self.dtype == "float32" else 2
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)) // BYTES_PER_BLOCK * BYTES_PER_BLOCK

        self.input_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="input_gm")
        self.scale_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="scale_gm")
        self.offset_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="offset_gm")
        self.output_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="output_gm",
                                                is_atomic_add=True)
        self.mean_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="mean_gm",
                                                is_atomic_add=True)
        self.rstd_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="rstd_gm",
                                               is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor("int32", [TILING_NUM], scope=tik.scope_gm, name="tiling_gm")

        self.n = self.tik_instance.Scalar(self.tiling_param_dtype, name='n')
        self.num_groups = self.tik_instance.Scalar(self.tiling_param_dtype, name='num_groups')
        self.elem_num = self.tik_instance.Scalar(self.tiling_param_dtype, name='elem_num')
        self.hw_num = self.tik_instance.Scalar(self.tiling_param_dtype, name='hw_num')
        self.group_c = self.tik_instance.Scalar(self.tiling_param_dtype, name='group_c')
        self.used_aicore_num = self.tik_instance.Scalar(self.tiling_param_dtype, name='used_aicore_num')
        self.avg_ng = self.tik_instance.Scalar(self.tiling_param_dtype, name='avg_ng')
        self.tail_ng = self.tik_instance.Scalar(self.tiling_param_dtype, name='tail_ng')
        self.c = self.tik_instance.Scalar(self.tiling_param_dtype, name='c')
        self.x_repeat = self.tik_instance.Scalar(self.tiling_param_dtype, name='x_repeat')
        self.x_block = self.tik_instance.Scalar(self.tiling_param_dtype, name='x_block')
        self.loop_per_group = self.tik_instance.Scalar(self.tiling_param_dtype, name='loop_per_group')
        self.x_tail_repeat = self.tik_instance.Scalar(self.tiling_param_dtype, name='x_tail_repeat')
        self.loop_per_line = self.tik_instance.Scalar(self.tiling_param_dtype, name='loop_per_line')
        self.x_tail_repeat_per_line = self.tik_instance.Scalar(self.tiling_param_dtype, name='x_tail_repeat_per_line')
        self.epsilon = self.tik_instance.Scalar("float32", name='epsilon')
        self.tiling_ub = self.tik_instance.Tensor(self.tiling_param_dtype, (TILING_NUM,),
                                                  name='tiling_ub', scope=tik.scope_ubuf)

    def get_tiling_params(self):
        """
        get runtime params from tiling
        """
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1,
                                    TILING_NUM // BLOCK_32, 0, 0)
        self.n.set_as(self.tiling_ub[0])
        self.num_groups.set_as(self.tiling_ub[1])
        self.elem_num.set_as(self.tiling_ub[2])
        self.hw_num.set_as(self.tiling_ub[3])
        self.group_c.set_as(self.tiling_ub[4])
        self.used_aicore_num.set_as(self.tiling_ub[5])
        self.avg_ng.set_as(self.tiling_ub[6])
        self.tail_ng.set_as(self.tiling_ub[7])
        self.c.set_as(self.tiling_ub[8])
        self.x_repeat.set_as(self.tiling_ub[9])
        self.x_block.set_as(self.tiling_ub[10])
        self.loop_per_group.set_as(self.tiling_ub[11])
        self.x_tail_repeat.set_as(self.tiling_ub[12])
        self.loop_per_line.set_as(self.tiling_ub[13])
        self.x_tail_repeat_per_line.set_as(self.tiling_ub[14])
        self.epsilon.set_as(self.tiling_ub[15])

    def get_mean_rstd(self, ng_idx, mean, rstd):
        """
            Calculate mean and rstd for current group
        """
        with self.tik_instance.new_stmt_scope():
            x_ub = self.tik_instance.Tensor(self.dtype, [self.x_block], scope=tik.scope_ubuf, name="x_ub")
            mean_ub = self.tik_instance.Tensor(self.dtype, [self.block], scope=tik.scope_ubuf, name="mean_ub")
            rstd_ub = self.tik_instance.Tensor(self.dtype, [self.block], scope=tik.scope_ubuf, name="rstd_ub")
            temp_ub = self.tik_instance.Tensor(self.dtype, [self.block], scope=tik.scope_ubuf, name="temp_ub")
            mean_temp = self.tik_instance.Scalar(self.dtype, init_value=0)
            rstd_temp = self.tik_instance.Scalar(self.dtype, init_value=0)
            block_len = self.tik_instance.Scalar("int32")
            total_repeat = self.tik_instance.Scalar("int32")
            repeat_times = self.tik_instance.Scalar("int32")
            repeat_tail = self.tik_instance.Scalar("int32")
            x_cur = self.tik_instance.Scalar("int32")

            # Calculate mean for current group
            with self.tik_instance.for_range(0, self.loop_per_group) as cur_loop:
                with self.tik_instance.if_scope(cur_loop != self.loop_per_group - 1):
                    self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + cur_loop * self.x_block],
                                                0, 1, self.x_repeat, 0, 0)
                    block_len.set_as(self.x_block)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + cur_loop * self.x_block],
                                                0, 1, self.x_tail_repeat, 0, 0)
                    block_len.set_as(self.elem_num % self.x_block)
                    # Set self.mask number of elements after block_len in x_ub to 0 for vcadd
                    self.tik_instance.vec_dup(self.mask, x_ub[block_len], 0, 1, 8)

                with self.tik_instance.for_range(0, ((block_len + self.mask - 1) // self.mask)) as i:
                    self.tik_instance.vcadd(self.mask, mean_ub, x_ub[i * self.mask], 1, 1, 1, 8)
                    mean_temp.set_as(mean_ub[0])
                    mean.set_as(mean + mean_temp)

            mean.set_as(mean / self.elem_num)
            mean_ub[0].set_as(mean)
            with self.tik_instance.for_range(1, self.block) as i:
                mean_ub[i].set_as(0)
            self.tik_instance.set_atomic_add(self.atomic_num)
            self.tik_instance.data_move(self.mean_gm[ng_idx], mean_ub, 0, 1, 1, 0, 0)
            self.tik_instance.set_atomic_add(0)
            mean.set_as(mean * (-1))

            # Calculate rstd for current group
            with self.tik_instance.for_range(0, self.loop_per_group) as cur_loop:
                with self.tik_instance.if_scope(cur_loop != self.loop_per_group - 1):
                    self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + cur_loop * self.x_block],
                                                0, 1, self.x_repeat, 0, 0)
                    block_len.set_as(self.x_block)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + cur_loop * self.x_block],
                                                0, 1, self.x_tail_repeat, 0, 0)
                    block_len.set_as(self.elem_num % self.x_block)

                total_repeat.set_as((block_len + self.mask - 1) // self.mask)
                repeat_times.set_as(total_repeat // MAX_REPEAT)
                repeat_tail.set_as(total_repeat % MAX_REPEAT)
                with self.tik_instance.for_range(0, repeat_times) as i:
                    x_cur.set_as(i * self.mask * MAX_REPEAT)
                    self.tik_instance.vec_adds(self.mask, x_ub[x_cur], x_ub[x_cur], mean, MAX_REPEAT, 8, 8)
                    self.tik_instance.vec_mul(self.mask, x_ub[x_cur], x_ub[x_cur], x_ub[x_cur], MAX_REPEAT, 8, 8, 8)
                with self.tik_instance.if_scope(repeat_tail != 0):
                    x_cur.set_as(repeat_times * self.mask * MAX_REPEAT)
                    self.tik_instance.vec_adds(self.mask, x_ub[x_cur], x_ub[x_cur], mean, repeat_tail, 8, 8)
                    self.tik_instance.vec_mul(self.mask, x_ub[x_cur], x_ub[x_cur], x_ub[x_cur], repeat_tail, 8, 8, 8)

                with self.tik_instance.if_scope(cur_loop == self.loop_per_group - 1):
                    # Set self.mask number of elements after block_len in x_ub to 0 for vcadd
                    self.tik_instance.vec_dup(self.mask, x_ub[block_len], 0, 1, 8)
                with self.tik_instance.for_range(0, total_repeat) as i:
                    self.tik_instance.vcadd(self.mask, temp_ub, x_ub[i * self.mask], 1, 1, 1, 8)
                    rstd_temp.set_as(temp_ub[0])
                    rstd.set_as(rstd + rstd_temp)

            rstd.set_as(rstd / self.elem_num)
            rstd.set_as(rstd + self.epsilon)
            self.tik_instance.scalar_sqrt(rstd, rstd)
            rstd.set_as(1.0 / rstd)
            rstd_ub[0].set_as(rstd)
            with self.tik_instance.for_range(1, self.mask) as i:
                rstd_ub[i].set_as(0)
            self.tik_instance.set_atomic_add(self.atomic_num)
            self.tik_instance.data_move(self.rstd_gm[ng_idx], rstd_ub, 0, 1, 1, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def get_mean_rstd_fp16(self, ng_idx, mean, rstd):
        """
            Calculate mean and rstd for current group
        """
        with self.tik_instance.new_stmt_scope():
            x_ub = self.tik_instance.Tensor(self.dtype, [self.x_block], scope=tik.scope_ubuf, name="x_ub")
            mean_ub = self.tik_instance.Tensor(self.dtype, [self.block], scope=tik.scope_ubuf, name="mean_ub")
            rstd_ub = self.tik_instance.Tensor(self.dtype, [self.block], scope=tik.scope_ubuf, name="rstd_ub")
            temp_ub = self.tik_instance.Tensor(self.dtype, [self.block], scope=tik.scope_ubuf, name="temp_ub")
            mean_temp = self.tik_instance.Scalar(self.dtype, init_value=0)
            rstd_temp = self.tik_instance.Scalar(self.dtype, init_value=0)
            block_len = self.tik_instance.Scalar("int32")
            total_repeat = self.tik_instance.Scalar("int32")
            repeat_times = self.tik_instance.Scalar("int32")
            repeat_tail = self.tik_instance.Scalar("int32")
            x_cur = self.tik_instance.Scalar("int32")
            mean_temp_fp32 = self.tik_instance.Scalar("float32", init_value=0)
            rstd_temp_fp32 = self.tik_instance.Scalar("float32", init_value=0)
            mean_fp32 = self.tik_instance.Scalar("float32", init_value=0)
            rstd_fp32 = self.tik_instance.Scalar("float32", init_value=0)

            # Calculate mean for current group
            with self.tik_instance.for_range(0, self.loop_per_group) as cur_loop:
                with self.tik_instance.if_scope(cur_loop != self.loop_per_group - 1):
                    self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + cur_loop * self.x_block],
                                                0, 1, self.x_repeat, 0, 0)
                    block_len.set_as(self.x_block)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + cur_loop * self.x_block],
                                                0, 1, self.x_tail_repeat, 0, 0)
                    block_len.set_as(self.elem_num % self.x_block)
                    # Set self.mask number of elements after block_len in x_ub to 0 for vcadd
                    self.tik_instance.vec_dup(self.mask, x_ub[block_len], 0, 1, 8)

                with self.tik_instance.for_range(0, ((block_len + self.mask - 1) // self.mask)) as i:
                    self.tik_instance.vcadd(self.mask, mean_ub, x_ub[i * self.mask], 1, 1, 1, 8)
                    mean_temp.set_as(mean_ub[0])
                    self.tik_instance.scalar_conv('', mean_temp_fp32, mean_temp)
                    self.tik_instance.scalar_conv('', mean_fp32, mean)
                    mean_fp32.set_as(mean_fp32 + mean_temp_fp32)
                    self.tik_instance.scalar_conv('', mean, mean_fp32)

            self.tik_instance.scalar_conv('', mean_fp32, mean)
            mean_fp32.set_as(mean_fp32 / self.elem_num)
            self.tik_instance.scalar_conv('', mean, mean_fp32)

            mean_ub[0].set_as(mean)
            with self.tik_instance.for_range(1, self.block) as i:
                mean_ub[i].set_as(0)
            self.tik_instance.set_atomic_add(self.atomic_num)
            self.tik_instance.data_move(self.mean_gm[ng_idx], mean_ub, 0, 1, 1, 0, 0)
            self.tik_instance.set_atomic_add(0)

            self.tik_instance.scalar_conv('', mean_fp32, mean)
            mean_fp32.set_as(mean_fp32 * (-1))
            self.tik_instance.scalar_conv('', mean, mean_fp32)

            # Calculate rstd for current group
            with self.tik_instance.for_range(0, self.loop_per_group) as cur_loop:
                with self.tik_instance.if_scope(cur_loop != self.loop_per_group - 1):
                    self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + cur_loop * self.x_block],
                                                0, 1, self.x_repeat, 0, 0)
                    block_len.set_as(self.x_block)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + cur_loop * self.x_block],
                                                0, 1, self.x_tail_repeat, 0, 0)
                    block_len.set_as(self.elem_num % self.x_block)

                total_repeat.set_as((block_len + self.mask - 1) // self.mask)
                repeat_times.set_as(total_repeat // MAX_REPEAT)
                repeat_tail.set_as(total_repeat % MAX_REPEAT)
                with self.tik_instance.for_range(0, repeat_times) as i:
                    x_cur.set_as(i * self.mask * MAX_REPEAT)
                    self.tik_instance.vec_adds(self.mask, x_ub[x_cur], x_ub[x_cur], mean, MAX_REPEAT, 8, 8)
                    self.tik_instance.vec_mul(self.mask, x_ub[x_cur], x_ub[x_cur], x_ub[x_cur], MAX_REPEAT, 8, 8, 8)
                with self.tik_instance.if_scope(repeat_tail != 0):
                    x_cur.set_as(repeat_times * self.mask * MAX_REPEAT)
                    self.tik_instance.vec_adds(self.mask, x_ub[x_cur], x_ub[x_cur], mean, repeat_tail, 8, 8)
                    self.tik_instance.vec_mul(self.mask, x_ub[x_cur], x_ub[x_cur], x_ub[x_cur], repeat_tail, 8, 8, 8)

                with self.tik_instance.if_scope(cur_loop == self.loop_per_group - 1):
                    # Set self.mask number of elements after block_len in x_ub to 0 for vcadd
                    self.tik_instance.vec_dup(self.mask, x_ub[block_len], 0, 1, 8)
                with self.tik_instance.for_range(0, total_repeat) as i:
                    self.tik_instance.vcadd(self.mask, temp_ub, x_ub[i * self.mask], 1, 1, 1, 8)
                    rstd_temp.set_as(temp_ub[0])
                    self.tik_instance.scalar_conv('', rstd_temp_fp32, rstd_temp)
                    self.tik_instance.scalar_conv('', rstd_fp32, rstd)
                    rstd_fp32.set_as(rstd_fp32 + rstd_temp_fp32)
                    self.tik_instance.scalar_conv('', rstd, rstd_fp32)

            self.tik_instance.scalar_conv('', rstd_fp32, rstd)
            rstd_fp32.set_as(rstd_fp32 / self.elem_num)
            rstd_fp32.set_as(rstd_fp32 + self.epsilon)
            self.tik_instance.scalar_sqrt(rstd_fp32, rstd_fp32)
            rstd_fp32.set_as(1.0 / rstd_fp32)
            self.tik_instance.scalar_conv('', rstd, rstd_fp32)

            rstd_ub[0].set_as(rstd)
            with self.tik_instance.for_range(1, self.mask) as i:
                rstd_ub[i].set_as(0)
            self.tik_instance.set_atomic_add(self.atomic_num)
            self.tik_instance.data_move(self.rstd_gm[ng_idx], rstd_ub, 0, 1, 1, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def calc_out(self, ng_idx, mean, rstd, scale_scalar, offset_scalar, group_idx):
        """
            Calculate y
        """
        x_ub = self.tik_instance.Tensor(self.dtype, [self.x_block], scope=tik.scope_ubuf, name="x_ub")
        block_len = self.tik_instance.Scalar("int32")
        total_repeat = self.tik_instance.Scalar("int32")
        repeat_times = self.tik_instance.Scalar("int32")
        repeat_tail = self.tik_instance.Scalar("int32")
        x_cur = self.tik_instance.Scalar("int32")
        with self.tik_instance.for_range(0, self.loop_per_line) as cur_loop:
            with self.tik_instance.if_scope(cur_loop != self.loop_per_line - 1):
                self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + group_idx * self.hw_num +
                                                                cur_loop * self.x_block], 0, 1, self.x_repeat, 0, 0)
                block_len.set_as(self.x_block)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(x_ub, self.input_gm[ng_idx * self.elem_num + group_idx * self.hw_num +
                                                                cur_loop * self.x_block], 0, 1,
                                            self.x_tail_repeat_per_line, 0, 0)
                block_len.set_as(self.hw_num % self.x_block)

            total_repeat.set_as((block_len + self.mask - 1) // self.mask)
            repeat_times.set_as(total_repeat // MAX_REPEAT)
            repeat_tail.set_as(total_repeat % MAX_REPEAT)
            with self.tik_instance.for_range(0, repeat_times) as i:
                x_cur.set_as(i * self.mask * MAX_REPEAT)
                self.tik_instance.vec_adds(self.mask, x_ub[x_cur], x_ub[x_cur], mean, MAX_REPEAT, 8, 8)
                self.tik_instance.vec_muls(self.mask, x_ub[x_cur], x_ub[x_cur], rstd, MAX_REPEAT, 8, 8)
                self.tik_instance.vec_muls(self.mask, x_ub[x_cur], x_ub[x_cur], scale_scalar, MAX_REPEAT, 8, 8)
                self.tik_instance.vec_adds(self.mask, x_ub[x_cur], x_ub[x_cur], offset_scalar, MAX_REPEAT, 8, 8)
            with self.tik_instance.if_scope(repeat_tail != 0):
                x_cur.set_as(repeat_times * self.mask * MAX_REPEAT)
                self.tik_instance.vec_adds(self.mask, x_ub[x_cur], x_ub[x_cur], mean, repeat_tail, 8, 8)
                self.tik_instance.vec_muls(self.mask, x_ub[x_cur], x_ub[x_cur], rstd, repeat_tail, 8, 8)
                self.tik_instance.vec_muls(self.mask, x_ub[x_cur], x_ub[x_cur], scale_scalar, repeat_tail, 8, 8)
                self.tik_instance.vec_adds(self.mask, x_ub[x_cur], x_ub[x_cur], offset_scalar, repeat_tail, 8, 8)

            with self.tik_instance.if_scope(cur_loop != self.loop_per_line - 1):
                self.tik_instance.data_move(self.output_gm[ng_idx * self.elem_num + group_idx * self.hw_num +
                                                           cur_loop * self.x_block], x_ub, 0, 1, self.x_repeat, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(self.hw_num % self.x_block, self.x_block) as i:
                    x_ub[i].set_as(0)
                self.tik_instance.set_atomic_add(self.atomic_num)
                self.tik_instance.data_move(self.output_gm[ng_idx * self.elem_num + group_idx * self.hw_num +
                                                           cur_loop * self.x_block], x_ub, 0, 1,
                                            self.x_tail_repeat_per_line, 0, 0)
                self.tik_instance.set_atomic_add(0)

    def compute_per_core(self, block_idx, ng_num, add_ng):
        """
        compute per ai_core
        """
        ng_idx = self.tik_instance.Scalar("int32")
        g_idx = self.tik_instance.Scalar("int32")
        scale_repeat = self.tik_instance.Scalar("int32")
        offset = self.tik_instance.Scalar("int32")
        mean = self.tik_instance.Scalar(self.dtype, init_value=0)
        rstd = self.tik_instance.Scalar(self.dtype, init_value=0)
        scale_scalar = self.tik_instance.Scalar(self.dtype)
        offset_scalar = self.tik_instance.Scalar(self.dtype)
        scale_ub = self.tik_instance.Tensor(self.dtype, [self.c], scope=tik.scope_ubuf, name="scale_ub")
        offset_ub = self.tik_instance.Tensor(self.dtype, [self.c], scope=tik.scope_ubuf, name="offset_ub")

        scale_repeat.set_as((self.c + self.block - 1) // self.block)
        self.tik_instance.data_move(scale_ub, self.scale_gm, 0, 1, scale_repeat, 0, 0)
        self.tik_instance.data_move(offset_ub, self.offset_gm, 0, 1, scale_repeat, 0, 0)
        with self.tik_instance.for_range(0, ng_num) as n_idx:
            ng_idx.set_as(block_idx * self.avg_ng + add_ng + n_idx)
            g_idx.set_as(ng_idx % self.num_groups)

            if (self.dtype == "float32"):
                self.get_mean_rstd(ng_idx, mean, rstd)
            else:
                self.get_mean_rstd_fp16(ng_idx, mean, rstd)

            with self.tik_instance.for_range(0, self.group_c) as group_idx:
                offset.set_as(g_idx * self.group_c + group_idx)
                scale_scalar.set_as(scale_ub[offset])
                offset_scalar.set_as(offset_ub[offset])
                self.calc_out(ng_idx, mean, rstd, scale_scalar, offset_scalar, group_idx)

    def compute(self):
        """
        main compute func
        """
        self.get_tiling_params()
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as block_idx:
            ng_num = self.tik_instance.Scalar("int32")
            add_ng = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(block_idx < self.tail_ng):
                ng_num.set_as(self.avg_ng + 1)
                add_ng.set_as(block_idx)
            with self.tik_instance.else_scope():
                ng_num.set_as(self.avg_ng)
                add_ng.set_as(self.tail_ng)
            self.compute_per_core(block_idx, ng_num, add_ng)

        tbe_context.get_context().add_compile_info("vars", {'ub_size_bytes': self.ub_size_bytes})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm, self.scale_gm, self.offset_gm],
                                   outputs=[self.output_gm, self.mean_gm, self.rstd_gm],
                                   flowtable=[self.tiling_gm])
        return self.tik_instance


def check_params(x, scale, offset):
    """
    check params of GroupNormV2
    """
    dtype_x = x.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")

    if dtype_x != dtype_scale or dtype_x != dtype_offset:
        raise RuntimeError("dtype of x, scale, offset must be same")

    if dtype_x not in ("float16", "float32"):
        raise RuntimeError("only support float16 and float32")


# 'pylint: disable=unused-argument,too-many-locals
@register_operator("GroupNormV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def group_norm_v2(x, scale, offset, y, mean, rstd, num_groups, data_format="NCHW", epsilon=1e-5, is_training=True,
               kernel_name="group_norm_v2"):
    """
    :param x: input_data, support ND of float16 or float32
    :param scale: scale_factor
    :param offset: offset_factor
    :param y: The result of GroupNormV2
    :param mean: mean of x
    :param rstd: rstd of x
    :param num_groups: number of groups
    :param data_format: data_format, default to NCHW
    :param epsilon: epsilon avoid divided by zero, default to 1e-5
    :param is_training: is_training
    :param kernel_name: kernel_name, default to group_norm_v2
    :return: instance
    """
    check_params(x, scale, offset)
    instance = GroupNormV2(x, scale, offset, y, mean, rstd, num_groups, data_format, epsilon, is_training,
                               kernel_name)
    return instance.compute()
