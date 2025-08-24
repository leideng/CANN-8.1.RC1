# Copyright (c) Huawei Technologies Co., Ltd. 2022-2025. All rights reserved.
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

from builtins import range as builtins_range
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.util_tik_comm_func import ceil_div


class Constant(object):
    """
    The class for constant
    """
    TRANSFORM_BYTE_TO_BIT = 8
    VECTOR_PROC_MAX_BYTE = 256
    TILING_ARG_NUM = 16
    MAX_SIZE = 2 ** 31 - 1
    BLOCK_SIZE = 32
    INT64_TYPE_SIZE = 8
    CONV_F322BF16_MASK = 64
    CONV_MAX_REPEATS = 255
    DTYPE_BF16 = "bfloat16"
    DTYPE_FP16 = "float16"


# 'pylint: disable=too-many-arguments,attribute-defined-outside-init
class Range(object):
    def __init__(self, start_dtype, limit_dtype, delta_dtype, y_dtype, kernel_name):
        self.tik_inst = tik.Tik()
        self.kernel_name = kernel_name
        self.dtype = y_dtype
        self.dtype_half = y_dtype
        if self.dtype_half == Constant.DTYPE_BF16 or self.dtype_half == Constant.DTYPE_FP16:
            self.dtype = "float32"

        self.dtype_size = tbe_platform.get_bit_len(self.dtype) // Constant.TRANSFORM_BYTE_TO_BIT
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.work_unit_num = Constant.VECTOR_PROC_MAX_BYTE // self.dtype_size
        self.output_ub_size = self.ub_size - Constant.VECTOR_PROC_MAX_BYTE * 2
        if self.dtype_half == Constant.DTYPE_BF16 or self.dtype_half == Constant.DTYPE_FP16:
            self.output_ub_size = self.output_ub_size // 2
        self.output_ub_num = self.output_ub_size // self.dtype_size // self.work_unit_num * self.work_unit_num
        self.init_gm_tensor(start_dtype, limit_dtype, delta_dtype)

    def init_gm_tensor(self, start_dtype, limit_dtype, delta_dtype):
        self.tiling_gm = self.tik_inst.Tensor(self.dtype, (Constant.TILING_ARG_NUM,),
                                              name="tiling_gm", scope=tik.scope_gm)
        self.start_gm = self.tik_inst.Tensor(start_dtype, (Constant.MAX_SIZE,), name="start_gm", scope=tik.scope_gm)
        self.limit_gm = self.tik_inst.Tensor(limit_dtype, (Constant.MAX_SIZE,), name="limit_gm", scope=tik.scope_gm)
        self.delta_gm = self.tik_inst.Tensor(delta_dtype, (Constant.MAX_SIZE,), name="delta_gm", scope=tik.scope_gm)
        if self.dtype_half == Constant.DTYPE_BF16 or self.dtype_half == Constant.DTYPE_FP16:
            self.output_gm = self.tik_inst.Tensor(self.dtype_half, (Constant.MAX_SIZE,),
                                                  name="output_gm", scope=tik.scope_gm)
        else:
            self.output_gm = self.tik_inst.Tensor(self.dtype, (Constant.MAX_SIZE,),
                                                  name="output_gm", scope=tik.scope_gm)

    def get_tiling_args(self):
        self.output_total_num = self.tik_inst.Scalar("int64", "output_total_num")
        self.running_core_num = self.tik_inst.Scalar("int64", "running_core_num")
        self.start = self.tik_inst.Scalar(self.dtype, "start")
        self.delta = self.tik_inst.Scalar(self.dtype, "delta")
        with self.tik_inst.new_stmt_scope():
            tiling_ub = self.tik_inst.Tensor(self.dtype, (Constant.TILING_ARG_NUM,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
            self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
            # output_total_num dtype is int64, tiling ub dtype is int32, so read value by cast to int64
            tiling_ub_int64 = tiling_ub.reinterpret_cast_to("int64")
            self.output_total_num.set_as(tiling_ub_int64[0])
            self.running_core_num.set_as(tiling_ub_int64[1])
            index = 2 * Constant.INT64_TYPE_SIZE // self.dtype_size
            self.start.set_as(tiling_ub[index])
            self.delta.set_as(tiling_ub[index + 1])

    def init_work_unit(self, output_offset):
        offset_value = self.tik_inst.Scalar(self.dtype, "offset_value")
        self.work_unit = self.tik_inst.Tensor(self.dtype, (self.work_unit_num,), name="work_unit", scope=tik.scope_ubuf)
        self.work_step = self.tik_inst.Tensor(self.dtype, (self.work_unit_num,), name="work_step", scope=tik.scope_ubuf)
        offset_value.set_as(self.work_unit_num * self.delta)
        self.tik_inst.vec_dup(self.work_unit_num, self.work_step, offset_value, 1, 8)

        with self.tik_inst.new_stmt_scope():
            work_unit_init = self.tik_inst.Tensor(self.dtype, (self.work_unit_num,), name="work_unit_init",
                                                  scope=tik.scope_ubuf)
            for idx in builtins_range(self.work_unit_num):
                work_unit_init[idx] = idx
            work_unit_delta = self.tik_inst.Tensor(self.dtype, (self.work_unit_num,), name="work_unit_delta",
                                                   scope=tik.scope_ubuf)
            self.tik_inst.vec_dup(self.work_unit_num, work_unit_delta, self.delta, 1, 8)
            work_unit_tmp = self.tik_inst.Tensor(self.dtype, (self.work_unit_num,), name="work_unit_tmp",
                                                 scope=tik.scope_ubuf)
            self.tik_inst.vec_mul(self.work_unit_num, work_unit_tmp, work_unit_init, work_unit_delta, 1, 0, 0, 0)
            work_offset = self.tik_inst.Tensor(self.dtype, (self.work_unit_num,), name="work_offset",
                                               scope=tik.scope_ubuf)
            offset_value.set_as(self.start + output_offset * self.delta)
            self.tik_inst.vec_dup(self.work_unit_num, work_offset, offset_value, 1, 8)
            self.tik_inst.vec_add(self.work_unit_num, self.work_unit, work_unit_tmp, work_offset, 1, 0, 0, 0)

    def range_compute(self):
        self.get_tiling_args()

        # divide core, for example: total 5, 4 core, result is: [2, 2, 1, 0]
        output_full_num_per_core = ceil_div(self.output_total_num, self.running_core_num)
        core_num_for_full = self.output_total_num // output_full_num_per_core
        output_lack_num_per_core = self.output_total_num - (output_full_num_per_core * core_num_for_full)
        with self.tik_inst.for_range(0, self.running_core_num, block_num=self.running_core_num) as core_idx:
            output_offset = core_idx * output_full_num_per_core
            with self.tik_inst.if_scope(core_idx < core_num_for_full):
                self.range_compute_per_core(output_offset, output_full_num_per_core)
            with self.tik_inst.elif_scope(core_idx == core_num_for_full):
                self.range_compute_per_core(output_offset, output_lack_num_per_core)

        tbe_context.get_context().add_compile_info("vars", {
            "core_num": tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        })

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.start_gm, self.limit_gm, self.delta_gm],
                               outputs=[self.output_gm],
                               flowtable=[self.tiling_gm],
                               config={"enable_const_fold": True})

        return self.tik_inst

    def range_compute_per_core(self, output_offset, output_num):
        self.init_work_unit(output_offset)
        self.output_ub = self.tik_inst.Tensor(self.dtype, (self.output_ub_num,), name="output_ub", scope=tik.scope_ubuf)
        if self.dtype_half == Constant.DTYPE_BF16 or self.dtype_half == Constant.DTYPE_FP16:
            self.output_ub_bf16 = self.tik_inst.Tensor(self.dtype_half, (self.output_ub_num,),
                                                       name="output_ub_bf16", scope=tik.scope_ubuf)

        ub_loop = output_num // self.output_ub_num
        with self.tik_inst.for_range(0, ub_loop) as ub_loop_idx:
            ub_loop_output_offset = output_offset + ub_loop_idx * self.output_ub_num
            self.range_compute_per_ub(ub_loop_output_offset, self.output_ub_num)

        ub_loop_output_offset = output_offset + ub_loop * self.output_ub_num
        output_num_last = output_num - ub_loop * self.output_ub_num
        with self.tik_inst.if_scope(output_num_last > 0):
            self.range_compute_per_ub(ub_loop_output_offset, output_num_last)

    def range_data_move_pad(self, output_gm, output_ub, dtype_size, output_offset, output_num):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            dst_gm = output_gm[output_offset].reinterpret_cast_to("int8")
            src_ub = output_ub.reinterpret_cast_to("int8")
            self.tik_inst.data_move_pad(dst_gm, src_ub[0], 1,
                                        output_num * dtype_size, 0, 0, 0, 0, None)
        else:
            move_to_gm_block_size = ceil_div((output_num * dtype_size), Constant.BLOCK_SIZE)
            self.tik_inst.data_move(output_gm[output_offset],
                                    output_ub,
                                    0, 1, move_to_gm_block_size, 0, 0)

    def range_compute_per_ub(self, output_offset, output_num):
        self.tik_inst.data_move(self.output_ub, self.work_unit, 0, 1, 8, 0, 0)

        work_loop = ceil_div(output_num, self.work_unit_num)
        with self.tik_inst.for_range(1, work_loop) as work_loop_idx:
            self.tik_inst.vec_add(self.work_unit_num, self.output_ub[work_loop_idx * self.work_unit_num],
                                  self.output_ub[(work_loop_idx - 1) * self.work_unit_num], self.work_step, 1, 8, 0, 0)
        if self.dtype_half != Constant.DTYPE_BF16 and self.dtype_half != Constant.DTYPE_FP16:
            self.range_data_move_pad(self.output_gm, self.output_ub, self.dtype_size, output_offset, output_num)
        else:
            if tbe_platform.api_check_support("tik.vcopy"):
                round_mode = "round"
            else:
                round_mode = "none"
            mask = Constant.CONV_F322BF16_MASK
            repeat = Constant.CONV_MAX_REPEATS
            loops = output_num // (mask * repeat)
            tails = output_num % (mask * repeat)
            with self.tik_inst.for_range(0, loops) as idx:
                self.tik_inst.vec_conv(mask, round_mode,
                                       self.output_ub_bf16[idx * mask * repeat],
                                       self.output_ub[idx * mask * repeat],
                                       repeat, 4, 8, None, True)
            tail_repeat = ceil_div(tails, mask)
            tail_last = tails % mask
            with self.tik_inst.if_scope(tail_repeat > 0):
                self.tik_inst.vec_conv(mask, round_mode,
                                       self.output_ub_bf16[loops * mask * repeat],
                                       self.output_ub[loops * mask * repeat],
                                       tail_repeat, 4, 8, None, True)
            with self.tik_inst.if_scope(tail_last > 0):
                self.tik_inst.vec_conv(tail_last, round_mode,
                                       self.output_ub_bf16[loops * mask * repeat + mask * tail_repeat],
                                       self.output_ub[loops * mask * repeat + mask * tail_repeat],
                                       1, 4, 8, None, True)

            bf16_size = tbe_platform.get_bit_len(self.dtype_half) // Constant.TRANSFORM_BYTE_TO_BIT
            self.range_data_move_pad(self.output_gm, self.output_ub_bf16, bf16_size, output_offset, output_num)

        # the last one unit, do accumulation, as the next ub first unit
        self.tik_inst.vec_add(self.work_unit_num, self.work_unit,
                              self.output_ub[(work_loop - 1) * self.work_unit_num],
                              self.work_step, 1, 8, 0, 0)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def range(start, limit, delta, y, is_closed=False, kernel_name="range"):
    """
    algorithm: range
    Generates values in an interval
    A sequence of delta evenly-spaced values are generated beginning at start
    so that the last one is exactly limit
    For example:
    range(1, 10.0, 2) => [ 1.0,3.0,5.0,7.0,9.0]
    range(1, 5)=> [1,2,3,4]
    range(5)=> [0,1,2,3,4]

    Parameters
    ----------
    start: dict
      dict of start, which contains int or float
    limit: dict
      dict of limit, which contains int or float
    delta: dict
      dict of delta, which contains int or float
    y: dict
      dict of output, which contains shape and dtype
    kernel_name: str
      kernel name, default value is "range"

    Returns
    -------
    None
    """
    shape_start = start.get("shape")
    shape_limit = limit.get("shape")
    shape_delta = delta.get("shape")
    para_check.check_shape(shape_start, param_name="start")
    para_check.check_shape(shape_limit, param_name="limit")
    para_check.check_shape(shape_delta, param_name="delta")

    dtype_start = start.get("dtype").lower()
    dtype_limit = limit.get("dtype").lower()
    dtype_delta = delta.get("dtype").lower()
    # self.start_gm, self.limit_gm, self.delta_gm仅在buildCCE时占位，且不支持double，故转为int64
    if dtype_start == "double":
        dtype_start = "int64"
    if dtype_limit == "double":
        dtype_limit = "int64"
    if dtype_delta == "double":
        dtype_delta = "int64"

    dtype_y = y.get("dtype").lower()
    para_check.check_dtype(dtype_start, ("float16", "int32", "float32", "bfloat16", "int64"), param_name="start")
    para_check.check_dtype(dtype_limit, ("float16", "int32", "float32", "bfloat16", "int64"), param_name="limit")
    para_check.check_dtype(dtype_delta, ("float16", "int32", "float32", "bfloat16", "int64"), param_name="delta")
    para_check.check_dtype(dtype_y, ("float16", "int32", "float32", "bfloat16", "int64"), param_name="y")

    range_instance = Range(dtype_start, dtype_limit, dtype_delta, dtype_y, kernel_name)
    tik_instance = range_instance.range_compute()

    return tik_instance
