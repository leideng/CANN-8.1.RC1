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
roll
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator

MAX_INT64 = 2 ** 63 - 1
TILING_NUM = 14
BLOCK_INT64_ALIGN = 4
MASK = 64

TILING_MODE0 = 0
TILING_MODE1 = 1
TILING_MODE2 = 2


class Roll():
    """
    Implementation of roll
    """

    def __init__(self, input_x, shifts, dims, kernel_name):
        """
        init of roll
        """
        self.tik_instance = tik.Tik()
        self.input_x_dtype = input_x.get("dtype")
        if self.input_x_dtype == "bfloat16":
            self.input_x_dtype = "float16"
        self.kernel_name = kernel_name
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_bytes_size_x = tbe_platform.get_bit_len(self.input_x_dtype) // 8
        self.data_each_block_x = 32 // self.dtype_bytes_size_x
        self.output_y_dtype = self.input_x_dtype
        self.align_num = 16 if self.input_x_dtype not in ("uint8", "int8") else 32

        self.dtype_bytes_size_y = tbe_platform.get_bit_len(self.output_y_dtype) // 8
        self.data_each_block_y = 32 // self.dtype_bytes_size_y
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 128
        self.ub_tensor_size = self.ub_size // self.dtype_bytes_size_x // 2 // 2 // 32 * 32
        self.align_ub = self.ub_size // self.dtype_bytes_size_x // self.align_num
        self.ub_num = self.align_ub * self.align_num
        self.input_x_gm = self.tik_instance.Tensor(self.input_x_dtype, [MAX_INT64], name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.output_y_dtype, [MAX_INT64], name="output_y_gm",
                                                    scope=tik.scope_gm)

        self.tiling_gm = self.tik_instance.Tensor("int64", [TILING_NUM], scope=tik.scope_gm, name="tiling_gm")

        self.tiling_mode = None
        self.need_core_num = None
        self.num_each_core = None
        self.last_core_num = None
        self.in_num = None
        self.after_num = None
        self.shift = None
        self.n = None
        self.chw_num = None
        self.loop = None
        self.tail_num = None
        self.flag = None

        self.input_x_ub, self.tmp_ub = None, None
        self.offset_this_dim, self.num_this_dim = None, None
        self.begin, self.end = None, None
        self.compute_burse_len_num, self.burse_len = None, None
        self.first_offset, self.last_offset, self.ori_first_offset, self.ori_last_offset = [None] * 4
        self.num_need_make_up, self.num_front, self.num_back, self.burse_num = [None] * 4
        self.loop_first_offset = None

    def get_tiling_params(self):
        """
        get runtime params from tiling
        :return: None
        """
        self.tiling_mode = self.tik_instance.Scalar("int64")
        self.need_core_num = self.tik_instance.Scalar("int64")
        self.num_each_core = self.tik_instance.Scalar("int64")
        self.last_core_num = self.tik_instance.Scalar("int64")
        self.in_num = self.tik_instance.Scalar("int64")
        self.after_num = self.tik_instance.Scalar("int64")
        self.shift = self.tik_instance.Scalar("int64")
        self.n = self.tik_instance.Scalar("int64")
        self.chw_num = self.tik_instance.Scalar("int64")
        self.loop = self.tik_instance.Scalar("int64")
        self.tail_num = self.tik_instance.Scalar("int64")
        self.flag = self.tik_instance.Scalar("int64")

        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", shape=(TILING_NUM,), scope=tik.scope_ubuf, name="tiling_ub")
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                        (TILING_NUM + BLOCK_INT64_ALIGN - 1) // BLOCK_INT64_ALIGN, 0, 0)

            self.tiling_mode.set_as(tiling_ub[0])
            self.need_core_num.set_as(tiling_ub[1])
            self.num_each_core.set_as(tiling_ub[2])
            self.last_core_num.set_as(tiling_ub[3])
            self.in_num.set_as(tiling_ub[4])
            self.after_num.set_as(tiling_ub[5])
            self.shift.set_as(tiling_ub[6])
            self.n.set_as(tiling_ub[7])
            self.chw_num.set_as(tiling_ub[8])
            self.loop.set_as(tiling_ub[9])
            self.tail_num.set_as(tiling_ub[10])
            self.flag.set_as(tiling_ub[11])

    def roll(self):
        """
        Calculate total entrance
        """
        self.roll_compute()

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.ai_core_num, "ub_size": self.ub_size,
                                                            "ub_num": self.ub_num})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_x_gm],
                                   outputs=[self.output_y_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    def roll_compute(self):
        """
        compute entrance
        """
        with self.tik_instance.for_range(0, self.ai_core_num,
                                         block_num=self.ai_core_num) as core_id:
            self.get_tiling_params()
            with self.tik_instance.if_scope(core_id < self.need_core_num):

                with self.tik_instance.if_scope(self.tiling_mode == TILING_MODE0):
                    move_offset = self.tik_instance.Scalar("int64")
                    move_offset.set_as(self.num_each_core * core_id)
                    with self.tik_instance.if_scope(core_id != self.need_core_num - 1):
                        self.roll_compute_each_core(move_offset, self.num_each_core)
                    with self.tik_instance.else_scope():
                        self.roll_compute_each_core(move_offset, self.last_core_num)

                with self.tik_instance.if_scope(self.tiling_mode == TILING_MODE1):
                    n_num = self.tik_instance.Scalar("int64")
                    with self.tik_instance.if_scope(core_id != self.need_core_num - 1):
                        n_num.set_as(self.num_each_core)
                    with self.tik_instance.else_scope():
                        n_num.set_as(self.last_core_num)
                    self.move_in_move_out(n_num, core_id)

    def roll_compute_each_core(self, core_move_offset, core_move_num):
        """
        Compute on each core
        """
        loop_time = self.tik_instance.Scalar("int64")
        last_num = self.tik_instance.Scalar("int64")
        move_offset = self.tik_instance.Scalar("int64")
        loop_time.set_as(core_move_num // self.ub_tensor_size)
        last_num.set_as(core_move_num % self.ub_tensor_size)
        move_offset.set_as(core_move_offset)
        need_db = self.tik_instance.Scalar("int32", init_value=1)

        with self.tik_instance.if_scope(loop_time < 2):
            need_db.set_as(0)
        with self.tik_instance.if_scope(loop_time > 0):
            with self.tik_instance.if_scope(need_db == 1):
                with self.tik_instance.for_range(0, loop_time, thread_num=2) as loop_id:
                    move_offset.set_as(loop_id * self.ub_tensor_size + core_move_offset)
                    self.roll_compute_each_loop(move_offset, self.ub_tensor_size)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, loop_time) as loop_id:
                    move_offset.set_as(loop_id * self.ub_tensor_size + core_move_offset)
                    self.roll_compute_each_loop(move_offset, self.ub_tensor_size)
            move_offset.set_as(loop_time * self.ub_tensor_size + core_move_offset)

        with self.tik_instance.if_scope(last_num > 0):
            self.roll_compute_each_loop(move_offset, last_num)

    def roll_compute_each_loop(self, move_offset, move_num):
        """
        compute each loop
        move_num <= ub_tensor_size
        """
        self.init_ub_tensor_and_scalar()
        self.loop_first_offset.set_as(move_offset)
        burse_len = self.tik_instance.Scalar("int64")
        burse_len.set_as((move_num + self.data_each_block_x - 1) // self.data_each_block_x)
        with self.tik_instance.if_scope(self.flag == 0):
            self.begin.set_as(move_offset // self.after_num)
            with self.tik_instance.if_scope((move_offset + move_num) % self.after_num == 0):
                self.end.set_as((move_offset + move_num) // self.after_num)
            with self.tik_instance.else_scope():
                self.end.set_as((move_offset + move_num) // self.after_num + 1)
            with self.tik_instance.for_range(self.begin, self.end) as i:
                with self.tik_instance.if_scope(i == move_offset // self.after_num):
                    self.offset_this_dim.set_as(move_offset)
                    with self.tik_instance.if_scope((move_offset + move_num) >= (self.after_num * (i + 1))):
                        self.num_this_dim.set_as(self.after_num * (i + 1) - move_offset)
                    with self.tik_instance.else_scope():
                        self.num_this_dim.set_as(move_num)
                with self.tik_instance.else_scope():
                    self.offset_this_dim.set_as(self.after_num * i)
                    with self.tik_instance.if_scope((move_offset + move_num) >= (self.after_num * (i + 1))):
                        self.num_this_dim.set_as(self.after_num)
                    with self.tik_instance.else_scope():
                        self.num_this_dim.set_as((move_offset + move_num) % self.after_num)
                self.roll_each_dim()
            self.tik_instance.data_move(self.output_y_gm[move_offset],
                                        self.input_x_ub,
                                        0, 1, burse_len, 0, 0)
        with self.tik_instance.else_scope():
            self.ori_first_offset.set_as(move_offset - self.shift)
            with self.tik_instance.if_scope(self.ori_first_offset < 0):
                self.ori_first_offset.set_as(self.ori_first_offset + self.in_num)
            self.ori_last_offset.set_as(self.ori_first_offset + move_num)
            # cut off
            with self.tik_instance.if_scope(self.ori_last_offset > self.in_num):
                self.ori_last_offset.set_as(self.ori_last_offset % self.in_num)
                # the front section
                self.num_front.set_as(self.in_num - self.ori_first_offset)
                self.tik_instance.scalar_conv('', self.compute_burse_len_num, self.num_front)
                self.compute_burse_len_num.set_as(self.compute_burse_len_num / self.data_each_block_x)
                self.tik_instance.scalar_conv('ceil', self.burse_len, self.compute_burse_len_num)
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self._move_data_by_bytes(self.tmp_ub, self.input_x_gm,
                                            self.num_front * self.dtype_bytes_size_x, src_offset=self.ori_first_offset)
                else:
                    self.tik_instance.data_move(self.tmp_ub[0], self.input_x_gm[self.ori_first_offset],
                                                0, 1, self.burse_len, 0, 0)
                with self.tik_instance.for_range(0, self.num_front) as n_id:
                    self.input_x_ub[n_id].set_as(self.tmp_ub[n_id])
                # the back section
                self.num_back.set_as(self.ori_last_offset)
                self.tik_instance.scalar_conv('', self.compute_burse_len_num, self.num_back)
                self.compute_burse_len_num.set_as(self.compute_burse_len_num / self.data_each_block_x)
                self.tik_instance.scalar_conv('ceil', self.burse_len, self.compute_burse_len_num)
                self.tik_instance.data_move(self.tmp_ub[0],
                                            self.input_x_gm[0],
                                            0, 1, self.burse_len, 0, 0)
                with self.tik_instance.for_range(0, self.num_back) as n_id:
                    self.input_x_ub[self.num_front + n_id].set_as(self.tmp_ub[n_id])
                self.tik_instance.data_move(self.output_y_gm[move_offset],
                                            self.input_x_ub,
                                            0, 1, burse_len, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.input_x_ub[0],
                                            self.input_x_gm[self.ori_first_offset],
                                            0, 1, burse_len, 0, 0)
                self.tik_instance.data_move(self.output_y_gm[move_offset],
                                            self.input_x_ub[0],
                                            0, 1, burse_len, 0, 0)

    def init_ub_tensor_and_scalar(self):
        """
        init tensor and scalar in ub
        """
        self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                   (self.ub_tensor_size,),
                                                   name="input_x_ub",
                                                   scope=tik.scope_ubuf)
        self.tmp_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                               (self.ub_tensor_size,),
                                               name="tmp_ub",
                                               scope=tik.scope_ubuf)
        self.offset_this_dim = self.tik_instance.Scalar(dtype="int64")
        self.num_this_dim = self.tik_instance.Scalar(dtype="int32")
        self.begin = self.tik_instance.Scalar(dtype="int64")
        self.end = self.tik_instance.Scalar(dtype="int64")
        self.compute_burse_len_num = self.tik_instance.Scalar(dtype="float32")
        self.burse_len = self.tik_instance.Scalar(dtype="int32")
        self.first_offset = self.tik_instance.Scalar(dtype="int64")
        self.last_offset = self.tik_instance.Scalar(dtype="int64")
        self.ori_first_offset = self.tik_instance.Scalar(dtype="int64")
        self.ori_last_offset = self.tik_instance.Scalar(dtype="int64")
        self.num_need_make_up = self.tik_instance.Scalar(dtype="int64")
        self.num_front = self.tik_instance.Scalar(dtype="int32")
        self.num_back = self.tik_instance.Scalar(dtype="int32")
        self.burse_num = self.tik_instance.Scalar(dtype="int32")
        self.loop_first_offset = self.tik_instance.Scalar(dtype="int64")

    def roll_each_dim(self):
        """
        roll compute on each dim
        """
        self.tik_instance.scalar_conv('', self.compute_burse_len_num, self.num_this_dim)
        self.compute_burse_len_num.set_as(self.compute_burse_len_num / self.data_each_block_x)
        self.tik_instance.scalar_conv('ceil', self.burse_len, self.compute_burse_len_num)
        self.ori_first_offset.set_as((self.offset_this_dim - self.shift * self.after_num))
        with self.tik_instance.if_scope(self.ori_first_offset < 0):
            self.ori_first_offset.set_as(self.ori_first_offset + self.in_num)
        with self.tik_instance.else_scope():
            self.ori_first_offset.set_as(self.ori_first_offset)
        self.ori_last_offset.set_as(self.ori_first_offset + self.num_this_dim)

        if tbe_platform.api_check_support("tik.data_move_pad"):
            self._move_data_by_bytes(self.tmp_ub, self.input_x_gm,
                                     self.num_this_dim * self.dtype_bytes_size_x, src_offset=self.ori_first_offset)
        else:
            self.tik_instance.data_move(self.tmp_ub, self.input_x_gm[self.ori_first_offset],
                                        0, 1, self.burse_len, 0, 0)

        with self.tik_instance.for_range(0, self.num_this_dim) as n_id:
            self.input_x_ub[self.offset_this_dim - self.loop_first_offset + n_id].set_as(self.tmp_ub[n_id])

    def move_in_move_out(self, n_num, block_id):
        # move data in and move out according to shifts
        in_n = self.tik_instance.Scalar("int32")
        out_n = self.tik_instance.Scalar("int32")
        in_offset = self.tik_instance.Scalar("int64")
        out_offset = self.tik_instance.Scalar("int64")
        in_offset2 = self.tik_instance.Scalar("int64")
        out_offset2 = self.tik_instance.Scalar("int64")

        move_ub = self.tik_instance.Tensor(self.input_x_dtype, [self.ub_num], scope=tik.scope_ubuf, name="move_ub")
        with self.tik_instance.for_range(0, n_num) as n_id:
            in_n.set_as(self.num_each_core * block_id + n_id)
            out_n.set_as((in_n + self.shift) % self.n)
            in_offset.set_as(in_n * self.chw_num)
            out_offset.set_as(out_n * self.chw_num)

            with self.tik_instance.if_scope(self.loop > 0):
                with self.tik_instance.for_range(0, self.loop) as loop_id:
                    in_offset2.set_as(in_offset + loop_id * self.ub_num)
                    out_offset2.set_as(out_offset + loop_id * self.ub_num)
                    self.data_move(move_ub, self.input_x_gm[in_offset2], num=self.ub_num)
                    self.data_move(self.output_y_gm[out_offset2], move_ub, num=self.ub_num)

                in_offset.set_as(in_offset + self.loop * self.ub_num)
                out_offset.set_as(out_offset + self.loop * self.ub_num)

            with self.tik_instance.if_scope(self.tail_num > 0):
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self._move_data_by_bytes(
                        move_ub, self.input_x_gm, self.tail_num * self.dtype_bytes_size_x, src_offset=in_offset)
                    self._move_data_by_bytes(
                        self.output_y_gm, move_ub, self.tail_num * self.dtype_bytes_size_x, dst_offset=out_offset)
                else:
                    self.data_move(move_ub, self.input_x_gm[in_offset], num=self.tail_num)
                    self.data_move(self.output_y_gm[out_offset], move_ub, num=self.tail_num)

    # 'pylint: disable-msg=too-many-arguments,too-many-locals,too-many-statements
    def data_move(self, dst, src, num, src_stride=0, dst_stride=0):
        """
        move data
        """
        sid = 0
        nburst = 1

        burst_len = (num + self.data_each_block_x - 1) // self.data_each_block_x
        self.tik_instance.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride, dst_stride=dst_stride)

    def _move_data_by_bytes(self, dst, src, size, dst_offset=0, src_offset=0):
        # use tik.data_move_pad to fix global memory out of bounds
        if dst.dtype != "int64":
            self.tik_instance.data_move_pad(dst[dst_offset], src[src_offset], 1, size, 0, 0)
        else:
            # tik.data_move_pad not support int64, reinterpret_cast_to int32
            dst_int32 = dst.reinterpret_cast_to("int32")
            src_int32 = src.reinterpret_cast_to("int32")
            self.tik_instance.data_move_pad(dst_int32[dst_offset * 2], src_int32[src_offset * 2], 1, size, 0, 0)


@register_operator("Roll")
# 'pylint: disable=unused-argument,too-many-branches,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def roll(x, y, shifts, dims, kernel_name="roll"):
    """
    roll the data according to the shifts and dims

    Parameters
    ----------
    x : dict
    shape and dtype of input_x
    y : dict
    shape and dtype of output_y, should be same shape as input, dtype is same as the quantified type
    shifts: list
    the processed shifts
    dims: list
    the processed dim
    kernel_name : str
    kernel name, default value is "roll"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype")
    check_x_tuple = ("float16", "float32", "int32", "uint32", "int8", "uint8", "bfloat16", "int64")

    if dtype_x not in check_x_tuple:
        raise RuntimeError("X only support %s while dtype is %s" %
                           (",".join(check_x_tuple), dtype_x))

    roll_instance = Roll(x, shifts, dims, kernel_name)
    return roll_instance.roll()
