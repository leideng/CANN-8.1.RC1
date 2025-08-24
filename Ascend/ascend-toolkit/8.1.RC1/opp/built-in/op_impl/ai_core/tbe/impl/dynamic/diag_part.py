# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
dynamic diag_part
"""
from impl import common_util
from impl import constant_util as constant
from impl.util import util_tik_comm_func
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # tiling arg num
    TILING_ARG_NUM = 8
    # MAX INT 64
    MAX_INT64 = 2 ** 64 - 1
    # NUM_128
    NUM_128 = 128
    # NUM_64
    NUM_64 = 64
    # BASE_4
    BASE_4 = 4


# 'pylint: disable=unused-argument,invalid-name,too-many-lines,line-too-long,no-self-use
# 'pylint: disable=,too-many-locals,attribute-defined-outside-init,too-many-instance-attributes
class DiagPart():
    """
    Function: class that execute diag_part
    """
    def __init__(self, input_x, kernel_name="diag_part"):
        self.shape_x = input_x.get("shape")
        self.dtype_x = input_x.get("dtype")
        self.dtype_bytes_size = common_util.get_data_size(self.dtype_x)

        # check dtype
        para_check.check_dtype(self.dtype_x, ("float16", "float32", "int32"), param_name="value")

        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()

        self.compute_assist()

        # get ai_core num
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.tiling_line_num = self.tik_instance.Scalar("int64", name="tiling_input_num", init_value=1)
        self.tiling_act_core_num = self.tik_instance.Scalar("int64", name="tiling_act_core_num", init_value=1)
        self.tiling_each_core_line_num = self.tik_instance.Scalar("int64", name="tiling_each_core_line_num", \
                                                                  init_value=1)
        self.tiling_last_core_line_num = self.tik_instance.Scalar("int64", name="tiling_last_core_line_num", \
                                                                  init_value=1)
        self.core_num_scalar = self.tik_instance.Scalar("int64", name="core_num_scalar", init_value=self.ai_core_num)

        # assist space
        dtype_tiling_bytes_size = common_util.get_data_size("int64") * Constant.TILING_ARG_NUM
        dtype_assist_bytes_size = self.dtype_bytes_size * Constant.NUM_128 * Constant.NUM_128

        # get ub size
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - dtype_tiling_bytes_size - \
                             dtype_assist_bytes_size

        # get data_num in one block
        self.data_each_block = constant.BLOCK_SIZE // self.dtype_bytes_size

        # ub for input and output
        self.ub_tensor_size = (self.ub_size_bytes // self.dtype_bytes_size // 2 // self.data_each_block *
                               self.data_each_block)
        # make gm tensor
        self.input_x_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT64,), name="input_x_gm", \
                                                   scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT64,),
                                                  name="output_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,), name="tiling_gm", \
                                                  scope=tik.scope_gm)

        # ub tensor
        self.input_x_ub = None
        self.assist_ub = None

        self.data_move_pad_support = tbe_platform.api_check_support("tik.data_move_pad")

    def compute_assist(self):
        """
        func compute_assist
        """
        assist_data = [[0] * 128 for _ in range(128)]
        if self.dtype_bytes_size == Constant.BASE_4:
            self.assist_type = "uint32"
            assist_value = 0xFFFFFFFF
        else:
            self.assist_type = "uint16"
            assist_value = 0xFFFF
        for i in range(0, 128):
            assist_data[i][i] = assist_value
        self.assist_gm = self.tik_instance.Tensor(self.assist_type, (Constant.NUM_128, Constant.NUM_128),
                                                  name="assist_gm", scope=tik.scope_gm, init_value=assist_data)

    def core_scedule_args(self, core_idx):
        """
        core_scedule_args
        """
        with self.tik_instance.if_scope(core_idx < self.tiling_act_core_num - 1):
            self.each_core_scedule(core_idx, self.tiling_each_core_line_num)
        with self.tik_instance.if_scope(core_idx == self.tiling_act_core_num - 1):
            with self.tik_instance.if_scope(self.tiling_line_num >= Constant.NUM_128):
                self.each_core_scedule(core_idx, self.tiling_last_core_line_num)
            with self.tik_instance.else_scope():
                self.small_shape_scedule()

    def each_core_scedule(self, core_idx, compute_line_num):
        """
        each_core_scedule
        """
        burst_len = Constant.NUM_128 // self.data_each_block
        self.tik_instance.data_move(self.assist_ub,
                                    self.assist_gm,
                                    0, Constant.NUM_128, burst_len, 0, 0)

        move_offset = core_idx * self.tiling_each_core_line_num * self.tiling_line_num + \
                      core_idx * self.tiling_each_core_line_num

        dtype_flag = 1
        nburst = Constant.NUM_128
        need_compute_line_num = Constant.NUM_128
        if self.dtype_x in ("float32", "int32"):
            dtype_flag = 2
            nburst = Constant.NUM_64

        loop_num = compute_line_num // (Constant.NUM_128 // dtype_flag)
        output_offset = core_idx * self.tiling_each_core_line_num
        with self.tik_instance.for_range(0, loop_num) as loop_index:
            x_gm_move_offset = move_offset + \
                               loop_index * Constant.NUM_128 // dtype_flag * self.tiling_line_num + \
                               loop_index * Constant.NUM_128 // dtype_flag

            with self.tik_instance.if_scope(self.tiling_line_num % self.data_each_block == 0):
                src_stride = (self.tiling_line_num - Constant.NUM_128 // dtype_flag) // self.data_each_block
                burst_len = Constant.NUM_128 // dtype_flag // self.data_each_block
                self.tik_instance.data_move(self.input_x_ub,
                                            self.input_x_gm[x_gm_move_offset],
                                            0, nburst, burst_len, src_stride, 0)

            with self.tik_instance.if_scope(self.tiling_line_num % self.data_each_block != 0):
                burst_len = Constant.NUM_128 // dtype_flag // self.data_each_block
                with self.tik_instance.for_range(0, Constant.NUM_128 // dtype_flag) as line_index:
                    move_x_gm_move_offset = x_gm_move_offset + line_index * self.tiling_line_num
                    if dtype_flag == 1:
                        x_ub_offset = line_index * Constant.NUM_128
                    if dtype_flag == 2:
                        x_ub_offset = line_index * Constant.NUM_64
                    self.tik_instance.data_move(self.input_x_ub[x_ub_offset],
                                                self.input_x_gm[move_x_gm_move_offset],
                                                0, 1, burst_len, 0, 0)

            if dtype_flag == 1:
                loop_output_offset = output_offset + loop_index * Constant.NUM_128
            if dtype_flag == 2:
                loop_output_offset = output_offset + loop_index * Constant.NUM_64
                need_compute_line_num = Constant.NUM_64
            self.compute_each_loop(dtype_flag, need_compute_line_num, loop_output_offset)

        last_line_num = compute_line_num - (loop_num * Constant.NUM_128 // dtype_flag)
        with self.tik_instance.if_scope(last_line_num > 0):
            back_offset = Constant.NUM_128 // dtype_flag - (last_line_num % (Constant.NUM_128 // dtype_flag))
            last_x_gm_move_offset = core_idx * self.tiling_each_core_line_num * self.tiling_line_num + \
                                    core_idx * self.tiling_each_core_line_num + \
                                    loop_num * Constant.NUM_128 // dtype_flag * self.tiling_line_num + \
                                    loop_num * Constant.NUM_128 // dtype_flag - \
                                    back_offset * self.tiling_line_num - back_offset

            burst_len = Constant.NUM_128 // dtype_flag // self.data_each_block
            with self.tik_instance.for_range(0, Constant.NUM_128 // dtype_flag) as line_index:
                last_x_gm_move_offset += line_index * self.tiling_line_num
                x_ub_offset = line_index * Constant.NUM_128 // dtype_flag
                if self.data_move_pad_support:
                    self.tik_instance.data_move_pad(self.input_x_ub[x_ub_offset],
                                                    self.input_x_gm[last_x_gm_move_offset],
                                                    1, Constant.NUM_128 // dtype_flag * self.dtype_bytes_size, 0, 0)
                else:
                    self.tik_instance.data_move(self.input_x_ub[x_ub_offset],
                                                self.input_x_gm[last_x_gm_move_offset],
                                                0, 1, burst_len, 0, 0)
            if dtype_flag == 1:
                output_offset += loop_num * Constant.NUM_128 - back_offset
                self.compute_each_loop(dtype_flag, Constant.NUM_128, output_offset)
            if dtype_flag == 2:
                output_offset += loop_num * Constant.NUM_64 - back_offset
                self.compute_each_loop(dtype_flag, Constant.NUM_64, output_offset)

    def small_shape_scedule(self):
        """
        small_shape_scedule
        """
        # MAX BURST LEN
        max_burst_len = 65535
        burst_len = util_tik_comm_func.ceil_div(self.tiling_line_num * self.tiling_line_num, self.data_each_block)
        nburst = util_tik_comm_func.ceil_div(burst_len, max_burst_len)
        burst_len = util_tik_comm_func.ceil_div(burst_len, nburst)
        if self.data_move_pad_support:
            self.tik_instance.data_move_pad(self.input_x_ub,
                                            self.input_x_gm,
                                            nburst, self.tiling_line_num * self.tiling_line_num * self.dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(self.input_x_ub,
                                        self.input_x_gm,
                                        0, nburst, burst_len, 0, 0)
        with self.tik_instance.for_range(0, self.tiling_last_core_line_num) as index:
            ub_out_offset = index * (self.tiling_line_num + 1)
            self.input_x_ub[index].set_as(self.input_x_ub[ub_out_offset])
        burst_len = util_tik_comm_func.ceil_div(self.tiling_line_num, self.data_each_block)
        if self.data_move_pad_support:
            self.tik_instance.data_move_pad(self.output_gm,
                                            self.input_x_ub,
                                            1, self.tiling_line_num * self.dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(self.output_gm,
                                        self.input_x_ub,
                                        0, 1, burst_len, 0, 0)

    def diag_part_compute(self):
        """
        func diag_part_compute
        """
        with self.tik_instance.new_stmt_scope():
            self._diag_part_compute_tiling()
        with self.tik_instance.for_range(0, self.core_num_scalar, block_num=self.core_num_scalar) as index:
            # make ub tensor
            self.input_x_ub = self.tik_instance.Tensor(self.dtype_x, (self.ub_tensor_size,),
                                                       name="input_x_ub", scope=tik.scope_ubuf)
            self.assist_ub = self.tik_instance.Tensor(self.assist_type, (Constant.NUM_128, Constant.NUM_128),
                                                      name="assist_ub", scope=tik.scope_ubuf)
            self.core_scedule_args(index)

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("global_variable_link", True)
        tbe_context.get_context().add_compile_info(
            "vars", {"core_num": self.ai_core_num}
        )
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_x_gm],
            outputs=[self.output_gm],
            flowtable=[self.tiling_gm],
            config=opt_config)

        return self.tik_instance

    def compute_each_loop(self, dtype_flag, compute_num, output_offset):
        """
        compute_each_loop
        """
        com_mask = compute_num * dtype_flag
        assist_uint16 = self.assist_ub.reinterpret_cast_to("uint16")
        ub_input_uint16 = self.input_x_ub.reinterpret_cast_to("uint16")
        rep_stride = util_tik_comm_func.ceil_div(compute_num, self.data_each_block)
        assist_rep_stride = Constant.NUM_128 // self.data_each_block
        self.tik_instance.vand(com_mask,
                               ub_input_uint16,
                               assist_uint16,
                               ub_input_uint16,
                               compute_num, 1, 1, 1,
                               rep_stride, assist_rep_stride, rep_stride)
        with self.tik_instance.for_range(0, compute_num - 1) as compute_index:
            self.tik_instance.vec_or(com_mask,
                                     ub_input_uint16,
                                     ub_input_uint16,
                                     ub_input_uint16[(compute_index + 1) * com_mask],
                                     1, rep_stride, rep_stride, rep_stride)
        burst_len = util_tik_comm_func.ceil_div(compute_num, self.data_each_block)
        self.tik_instance.data_move(self.output_gm[output_offset],
                                    self.input_x_ub,
                                    0, 1, burst_len, 0, 0)

    def _diag_part_compute_tiling(self):
        """
        tiling info:
            tiling_line_num: number of input_x's line
            tiling_act_core_num: need use aicore number
            tiling_each_core_line_num: line number of one each aicore need to compute except last aicore
            tiling_last_core_line_num: line number of one last aicore need to compute
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,), name="tiling_ub", \
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)  
        self.tiling_line_num.set_as(self.tiling_ub[0])
        self.tiling_act_core_num.set_as(self.tiling_ub[1])
        self.tiling_each_core_line_num.set_as(self.tiling_ub[2])
        self.tiling_last_core_line_num.set_as(self.tiling_ub[3])
        self.core_num_scalar.set_as(self.tiling_ub[4])


@register_operator("DiagPart")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def diag_part(x, y, kernel_name="diag_part"):
    """
    algorithm: diag_part
    calculating diag_part(x):
    returns a diag_partonal tensor with a given x values.
    If the shape of x is [D1,...,Dk],the shape of diag_partonal tensor is
    [D1,...,Dk,D1,...,Dk]
    For example:
    x :    [1, 2, 3]
    res :  [[1, 0, 0]
            [0, 2, 0]
            [0, 0, 3]]

    Parameters
    ----------
    x: dict
        dict with keys(shape and dtype) of x, and the dtype of x must
        be in [float16, float32, int32]
    y: dict
        dict with keys(shape and dtype) of y
    kernel_name: str
        kernel name, default value is "diag_part"

    Returns
    -------
    None
    """
    obj = DiagPart(x, kernel_name)
    tik_instance = obj.diag_part_compute()
    return tik_instance
