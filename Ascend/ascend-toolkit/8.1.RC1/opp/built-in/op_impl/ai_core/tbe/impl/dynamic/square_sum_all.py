# Copyright 2021 Huawei Technologies Co., Ltd
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
square_sum_all
"""
import math

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    BYTE_BLOCK = 32
    TILING_PARAMS_NUM = 32
    TILING_PARAM_DTYPE = "int64"

    MAX_INT64 = 2 ** 63 - 1
    MAX_SHAPE_SIZE = MAX_INT64
    USE_DATA_COPY_PAD = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ["Ascend910B", "Ascend910_93"]


# 'pylint: disable=too-many-instance-attributes,too-few-public-methods
class SquareSumAll():
    """
    Function: class that execute square_sum_all
    """

    # 'pylint: disable=too-many-statements
    def __init__(self, input_x, input_y, kernel_name):
        """
        Init square_sum_all base parameters

        Parameters
        ----------
        input_x: dict
            data of input_x
            datatype supports float32
        input_y: dict
            data of input_y
            datatype supports float32
        kernel_name: str
            the name of the operator

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.input_x_dtype = input_x.get("dtype").lower()
        self.input_y_dtype = input_y.get("dtype").lower()
        self._check_param()

        one_block_bytes_size = tbe_platform.VECTOR_INST_BLOCK_WIDTH // tbe_platform.VECTOR_INST_BLOCK_NUM
        self.dtype_bytes_size = get_bit_len(self.input_x_dtype) // 8
        self.data_each_block = one_block_bytes_size // self.dtype_bytes_size
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 1024
        self.vector_mask_max = tbe_platform.VECTOR_INST_BLOCK_NUM * self.data_each_block
        self.kernel_name = kernel_name

        self.input_x_gm = None
        self.input_y_gm = None
        self.output_x_gm = None
        self.output_y_gm = None
        self.input_x_ub = None
        self.input_y_ub = None
        self.every_process_data_num = None
        self.process_times = None
        self.tail_num = None
        self.reduce_sum_loop = None
        self.reduce_sum_loop_tail = None
        self.burst_len = None
        self.burst_len_tail = None

        # tiling data
        self.tiling_ub = None
        self.need_core_num_input_scalar = None
        self.data_num_each_core = None
        self.process_times_per_core = None
        self.process_times_remain_core = None
        self.every_process_data_num_per_core = None
        self.every_process_data_num_remain_core = None
        self.tail_num_per_core = None
        self.tail_num_remain_core = None
        self.reduce_sum_loop_per_core = None
        self.reduce_sum_loop_tail_per_core = None
        self.reduce_sum_loop_remain_core = None
        self.reduce_sum_loop_tail_remain_core = None
        self.burst_len_per_core = None
        self.burst_len_tail_per_core = None
        self.burst_len_remain_core = None
        self.burst_len_tail_remain_core = None
        self.tiling_device_core_num = None
        self.byte_burst_len_per_core = None
        self.byte_burst_len_tail_per_core = None
        self.byte_burst_len_remain_core = None
        self.byte_burst_len_tail_remain_core = None

    def square_sum_all_compute(self):
        """
        SquareSumAll operation

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        tiling_gm = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                             name="tiling_gm", scope=tik.scope_gm)

        # get tiling args from gm
        self.tiling_ub = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                  name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, tiling_gm, 0, 1, 8, 0, 0)

        self._get_tiling_args()

        self.input_x_gm = self.tik_instance.Tensor(self.input_x_dtype, (Constant.MAX_SHAPE_SIZE,), name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.input_y_gm = self.tik_instance.Tensor(self.input_y_dtype, (Constant.MAX_SHAPE_SIZE,), name="input_y_gm",
                                                   scope=tik.scope_gm)
        self.output_x_gm = self.tik_instance.Tensor(self.input_x_dtype, (1,), name="output_x_gm", scope=tik.scope_gm,
                                                    is_atomic_add=True)
        self.output_y_gm = self.tik_instance.Tensor(self.input_y_dtype, (1,), name="output_y_gm", scope=tik.scope_gm,
                                                    is_atomic_add=True)

        self._square_sum_all_compute_tiling()

        # add compile info
        compile_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": compile_core_num,
                "data_each_block": self.data_each_block,
                "dtype_bytes_size": self.dtype_bytes_size
            })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.input_x_gm, self.input_y_gm),
                                   outputs=(self.output_x_gm, self.output_y_gm),
                                   flowtable=(tiling_gm,))

    def _check_param(self):
        """
        Check parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        para_check.check_dtype(self.input_x_dtype, ("float32",), param_name="input_x")
        para_check.check_dtype(self.input_y_dtype, ("float32",), param_name="input_y")

        add_support = tbe_platform.api_check_support("tik.vadd", "float32")

        if self.input_x_dtype != self.input_y_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "input_x", "input_y",
                                                                  self.input_x_dtype, self.input_y_dtype)

        if self.input_x_dtype == "float32" and not add_support:
            error_manager_vector.raise_err_input_dtype_not_supported(self.kernel_name, "input_x", [],
                                                                     self.input_x_dtype)

    def _get_tiling_args(self):
        """
        Get tiling args from tiling_ub

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        self.need_core_num_input_scalar = self.tik_instance.Scalar(dtype="int64", name="need_core_num_input_scalar")
        self.data_num_each_core = self.tik_instance.Scalar(dtype="int64", name="data_num_each_core")
        self.process_times_per_core = self.tik_instance.Scalar(dtype="int64", name="process_times_per_core")
        self.process_times_remain_core = self.tik_instance.Scalar(dtype="int64", name="process_times_remain_core")
        self.every_process_data_num_per_core = self.tik_instance.Scalar(dtype="int64",
                                                                        name="every_process_data_num_per_core")
        self.every_process_data_num_remain_core = self.tik_instance.Scalar(dtype="int64",
                                                                           name="every_process_data_num_remain_core")
        self.tail_num_per_core = self.tik_instance.Scalar(dtype="int64", name="tail_num_per_core")
        self.tail_num_remain_core = self.tik_instance.Scalar(dtype="int64", name="tail_num_remain_core")
        self.reduce_sum_loop_per_core = self.tik_instance.Scalar(dtype="int64", name="reduce_sum_loop_per_core")
        self.reduce_sum_loop_tail_per_core = self.tik_instance.Scalar(dtype="int64",
                                                                      name="reduce_sum_loop_tail_per_core")
        self.reduce_sum_loop_remain_core = self.tik_instance.Scalar(dtype="int64", name="reduce_sum_loop_remain_core")
        self.reduce_sum_loop_tail_remain_core = self.tik_instance.Scalar(dtype="int64",
                                                                         name="reduce_sum_loop_tail_remain_core")
        self.burst_len_per_core = self.tik_instance.Scalar(dtype="int64", name="burst_len_per_core")
        self.burst_len_tail_per_core = self.tik_instance.Scalar(dtype="int64", name="burst_len_tail_per_core")
        self.burst_len_remain_core = self.tik_instance.Scalar(dtype="int64", name="burst_len_remain_core")
        self.burst_len_tail_remain_core = self.tik_instance.Scalar(dtype="int64", name="burst_len_tail_remain_core")
        self.tiling_device_core_num = self.tik_instance.Scalar(dtype="int64", name="tiling_device_core_num")
        self.byte_burst_len_per_core = self.tik_instance.Scalar(dtype="int64", 
                                                                name="byte_burst_len_per_core")
        self.byte_burst_len_tail_per_core = self.tik_instance.Scalar(dtype="int64", 
                                                                     name="byte_burst_len_tail_per_core")
        self.byte_burst_len_remain_core = self.tik_instance.Scalar(dtype="int64", 
                                                                   name="byte_burst_len_remain_core")
        self.byte_burst_len_tail_remain_core = self.tik_instance.Scalar(dtype="int64", 
                                                                        name="byte_burst_len_tail_remain_core")

        # input scalar in flowtable
        input_scalar_index = 0
        self.need_core_num_input_scalar.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.data_num_each_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.process_times_per_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.process_times_remain_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.every_process_data_num_per_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.every_process_data_num_remain_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.tail_num_per_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.tail_num_remain_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.reduce_sum_loop_per_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.reduce_sum_loop_tail_per_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.reduce_sum_loop_remain_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.reduce_sum_loop_tail_remain_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.burst_len_per_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.burst_len_tail_per_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.burst_len_remain_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.burst_len_tail_remain_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.tiling_device_core_num.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.byte_burst_len_per_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.byte_burst_len_tail_per_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.byte_burst_len_remain_core.set_as(self.tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.byte_burst_len_tail_remain_core.set_as(self.tiling_ub[input_scalar_index])

    # 'pylint: disable=too-many-arguments
    def _vector_mul(self, mask, des_offset, src1_offset, src2_offset, repeat_times, calc_target):
        """
        Execute the vector mul calculation

        Parameters
        ----------
        mask: int
            the mask of instruction
        des_offset: int
            destination address offset
        src1_offset: int
            src1 address offset
        src2_offset: int
            src2 address offset
        repeat_times: int
            the repeat times of instruction
        calc_target: string
            which input calculate on
        Returns
        -------
        None
        """
        for _, i in enumerate(calc_target):
            self.tik_instance.vmul(mask, i[des_offset], i[src1_offset], i[src2_offset], repeat_times, 1, 1, 1, 8, 8, 8)

    def _reduce_sum(self, calc_num):
        """
        Execute add calculation

        Parameters
        ----------
        calc_num: int
            the number of tensor elements in add calculation
        Returns
        calc_num: int
            the number of tensor elements in add calculation next time
        """

        # ensured the data address aligned 32b after tensor divided by 2
        align_value = self.data_each_block * 2

        tail_num = self.tik_instance.Scalar(dtype="int32", init_value=0, name="tail_num")
        mid_tail_num = calc_num % align_value
        tail_num.set_as(mid_tail_num)
        calc_num.set_as((calc_num // align_value) * align_value)

        total_sum_num = self.tik_instance.Scalar(dtype="int32", init_value=0, name="total_sum_num")
        total_sum_num.set_as(calc_num)

        calc_num.set_as(calc_num // 2)
        add_loop = self.tik_instance.Scalar(dtype="int32", init_value=0, name="add_loop")
        mid_add_loop = calc_num // (self.vector_mask_max * 255)
        add_loop.set_as(mid_add_loop)
        calc_offset = self.tik_instance.Scalar(dtype="int32", init_value=0, name="calc_offset")

        with self.tik_instance.if_scope(add_loop > 0):
            with self.tik_instance.for_range(0, add_loop, name="add_loop_index") as add_index:
                mid_calc_offset = add_index * self.vector_mask_max * 255
                calc_offset.set_as(mid_calc_offset)
                self._vector_add(self.vector_mask_max, calc_offset, calc_offset, calc_offset + calc_num, 255)
            mid_calc_offset = add_loop * self.vector_mask_max * 255
            calc_offset.set_as(mid_calc_offset)

        repeat_time = self.tik_instance.Scalar(dtype="int32", init_value=0, name="repeat_time")
        mid_repeat_time = (calc_num % (self.vector_mask_max * 255)) // self.vector_mask_max
        repeat_time.set_as(mid_repeat_time)

        with self.tik_instance.if_scope(repeat_time > 0):
            self._vector_add(self.vector_mask_max, calc_offset, calc_offset, calc_offset + calc_num, repeat_time)

        last_num = self.tik_instance.Scalar(dtype="int32", init_value=0, name="last_num")
        mid_last_num = calc_num % self.vector_mask_max
        last_num.set_as(mid_last_num)

        with self.tik_instance.if_scope(last_num > 0):
            mid_calc_offset = calc_offset + repeat_time * self.vector_mask_max
            calc_offset.set_as(mid_calc_offset)
            self._vector_add(last_num, calc_offset, calc_offset, calc_offset + calc_num, 1)

        with self.tik_instance.if_scope(tail_num > 0):
            mid_last_num = tail_num % self.vector_mask_max
            last_num.set_as(mid_last_num)
            self._vector_add(last_num, 0, 0, total_sum_num, 1)

        return calc_num

    # 'pylint: disable=too-many-arguments
    def _vector_add(self, mask, des_offset, src1_offset, src2_offset, repeat_times):
        """
        Execute the vector add calculation

        Parameters
        ----------
        mask: int
            the mask of instruction
        des_offset: int
            destination address offset
        src1_offset: int
            src1 address offset
        src2_offset: int
            src2 address offset
        repeat_times: int
            the repeat times of instruction
        Returns
        -------
        None
        """
        self.tik_instance.vadd(mask, self.input_x_ub[des_offset], self.input_x_ub[src1_offset],
                               self.input_x_ub[src2_offset], repeat_times, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.input_y_ub[des_offset], self.input_y_ub[src1_offset],
                               self.input_y_ub[src2_offset], repeat_times, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _calc_op(self, calc_num, burst_len, reduce_sum_loop, offset):
        """
        every process square_sum

        Parameters
        ----------
        calc_num: int
            the number of every process data
        burst_len: int
            burst_len of data_move instruction
        reduce_sum_loop: int
            the number of loop for reduce sum calculation
        offset: int
            the offset of data address

        Returns
        -------
        None
        """
        if Constant.USE_DATA_COPY_PAD:
            self.tik_instance.data_move_pad(self.input_x_ub, self.input_x_gm[offset],
                                            1, burst_len, dst_gap=0, src_gap=0)
            self.tik_instance.data_move_pad(self.input_y_ub, self.input_y_gm[offset],
                                            1, burst_len, dst_gap=0, src_gap=0)
        else:
            self.tik_instance.data_move(self.input_x_ub, self.input_x_gm[offset], 0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(self.input_y_ub, self.input_y_gm[offset], 0, 1, burst_len, 0, 0)

        calc_loop = self.tik_instance.Scalar(dtype="int32", name="calc_loop")
        loop = calc_num // (self.vector_mask_max * 255)
        calc_loop.set_as(loop)

        calc_offset = self.tik_instance.Scalar(dtype="int32", init_value=0)

        with self.tik_instance.if_scope(calc_loop > 0):
            with self.tik_instance.for_range(0, calc_loop) as add_index:
                mid_calc_offset = add_index * self.vector_mask_max * 255
                calc_offset.set_as(mid_calc_offset)
                self._vector_mul(self.vector_mask_max, calc_offset, calc_offset, calc_offset, 255,
                                 [self.input_x_ub, self.input_y_ub])
            mid_calc_offset = self.vector_mask_max * 255 * calc_loop
            calc_offset.set_as(mid_calc_offset)

        repeat_time = self.tik_instance.Scalar("int32")
        mid_repeat_time = (calc_num % (self.vector_mask_max * 255) // self.vector_mask_max)
        repeat_time.set_as(mid_repeat_time)

        with self.tik_instance.if_scope(repeat_time > 0):
            self._vector_mul(self.vector_mask_max, calc_offset, calc_offset, calc_offset, repeat_time,
                             [self.input_x_ub, self.input_y_ub])

        last_num = self.tik_instance.Scalar("int32")
        mid_last_num = calc_num % self.vector_mask_max
        last_num.set_as(mid_last_num)
        with self.tik_instance.if_scope(last_num > 0):
            calc_offset = calc_offset + repeat_time * self.vector_mask_max
            self._vector_mul(last_num, calc_offset, calc_offset, calc_offset, 1, [self.input_x_ub, self.input_y_ub])

        mid_calc_num = self.tik_instance.Scalar("int32")
        mid_calc_num.set_as(calc_num)
        with self.tik_instance.for_range(0, reduce_sum_loop):
            mid_calc_num = self._reduce_sum(mid_calc_num)

        vcadd_mask = mid_calc_num
        self.tik_instance.vcadd(vcadd_mask, self.input_x_ub, self.input_x_ub, 1, 1, 1, 8)
        self.tik_instance.vcadd(vcadd_mask, self.input_y_ub, self.input_y_ub, 1, 1, 1, 8)

        self.tik_instance.set_atomic_add(1)
        if Constant.USE_DATA_COPY_PAD:
            self.tik_instance.data_move_pad(self.output_x_gm, self.input_x_ub,
                                            1, self.dtype_bytes_size, dst_gap=0, src_gap=0)
            self.tik_instance.data_move_pad(self.output_y_gm, self.input_y_ub,
                                            1, self.dtype_bytes_size, dst_gap=0, src_gap=0)
        else:
            self.tik_instance.data_move(self.output_x_gm, self.input_x_ub, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.output_y_gm, self.input_y_ub, 0, 1, 1, 0, 0)
        self.tik_instance.set_atomic_add(0)

    def _execute_square_sum_all(self, data_offset):
        """
        execute square_sum operation

        Parameters
        ----------
        data_offset: int
            the offset of data address in different core

        Returns
        -------
        None
        """
        temp_offset = data_offset
        with self.tik_instance.for_range(0, self.process_times) as i:
            data_offset = data_offset + i * self.every_process_data_num
            self._calc_op(self.every_process_data_num, self.burst_len, self.reduce_sum_loop, data_offset)

        data_offset = temp_offset + self.every_process_data_num * self.process_times
        with self.tik_instance.if_scope(self.tail_num > 0):
            self._calc_op(self.tail_num, self.burst_len_tail, self.reduce_sum_loop_tail, data_offset)

    def _set_tensor(self, is_remain_core):
        """
        Compute arguments for execute square_sum_all

        Parameters
        ----------
        is_remain_core: bool
            check if arguments are used for remain core calculation

        Returns
        -------
        None
        """
        if is_remain_core:
            self.process_times = self.process_times_remain_core
            self.every_process_data_num = self.every_process_data_num_remain_core
            self.tail_num = self.tail_num_remain_core
            self.reduce_sum_loop = self.reduce_sum_loop_remain_core
            self.reduce_sum_loop_tail = self.reduce_sum_loop_tail_remain_core
            self.burst_len = self.byte_burst_len_remain_core if Constant.USE_DATA_COPY_PAD \
                else self.burst_len_remain_core
            self.burst_len_tail = self.byte_burst_len_tail_remain_core if Constant.USE_DATA_COPY_PAD \
                else self.burst_len_tail_remain_core
        else:
            self.process_times = self.process_times_per_core
            self.every_process_data_num = self.every_process_data_num_per_core
            self.tail_num = self.tail_num_per_core
            self.reduce_sum_loop = self.reduce_sum_loop_per_core
            self.reduce_sum_loop_tail = self.reduce_sum_loop_tail_per_core
            self.burst_len = self.byte_burst_len_per_core if Constant.USE_DATA_COPY_PAD \
                else self.burst_len_per_core
            self.burst_len_tail = self.byte_burst_len_tail_per_core if Constant.USE_DATA_COPY_PAD \
                else self.burst_len_tail_per_core

        # for two inputs
        ub_max_num = self.ub_size_bytes // self.dtype_bytes_size // 2
        flag = self.data_each_block
        assign_ub_shape = (ub_max_num // flag * flag,)

        self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                   assign_ub_shape,
                                                   name="input_x_ub",
                                                   scope=tik.scope_ubuf)
        self.input_y_ub = self.tik_instance.Tensor(self.input_y_dtype,
                                                   assign_ub_shape,
                                                   name="input_y_ub",
                                                   scope=tik.scope_ubuf)

    def _square_sum_all_compute_tiling(self):
        """
        SquareSumAll main process

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        soc_core_num = self.tiling_device_core_num
        process_data_num = self.data_num_each_core
        actual_block_num = self.need_core_num_input_scalar

        with self.tik_instance.for_range(0, soc_core_num, block_num=soc_core_num) as block_idx:
            move_offset = self.tik_instance.Scalar(dtype="int64", init_value=block_idx * process_data_num,
                                                   name="move_offset")
            with self.tik_instance.if_scope(block_idx < actual_block_num - 1):
                self._set_tensor(False)
                self._execute_square_sum_all(move_offset)
            with self.tik_instance.else_scope():
                self._set_tensor(True)
                self._execute_square_sum_all(move_offset)


# 'pylint: disable=unused-argument
@register_operator("SquareSumAll")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def square_sum_all(input_x, input_y, output_x, output_y, kernel_name="square_sum_all"):
    """
    calculating square_sum_all

    Parameters
    ----------
    input_x: dict
        input tensor contains shape and dtype attributes.
        only support float32.
    input_y: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype and shape as 'input_x'.
    output_x: dict
        output tensor contains shape and dtype attributes.
        Must have the same dtype  as 'input_x'.
    output_y: dict
        output tensor contains shape and dtype attributes.
        Must have the same dtype as 'input_x'.
    kernel_name: str
        cce kernel name, default value is `square_sum_all`.

    Returns
    -------
    None
    """

    obj = SquareSumAll(input_x, input_y, kernel_name)
    obj.square_sum_all_compute()
