# Copyright 2020 Huawei Technologies Co., Ltd
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
import functools
import math
import operator

import te.platform as tbe_platform
from te import tik
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.util.util_select_op_base import ReduceInput
from impl.util.util_select_op_base import ReduceOutput
from impl.util.util_select_op_base import get_op_cal_info
from tbe.common.platform import get_bit_len

# Each core processing data num greater than the size we can get better performace from experience
MINIMUM_DATA_NUM_EACH_CORE = 1024


# 'pylint: disable=unused-argument
def get_op_support_info(input_x, input_y, output_x, output_y, kernel_name="square_sum"):
    """
    get unpack slice info
    """
    format_x = input_x.get("format")
    shape_x = input_x.get("shape")
    support_format = ["FRACTAL_Z", "C1HWNCoC0", "NC1HWC0", "ND", "NCHW", "NHWC"]
    reduce_add = 1  # enumerated value
    if format_x in support_format:
        axis_reduce_list = []
        for idx, _ in enumerate(shape_x):
            reduce_info = [
                ReduceInput([0, [idx]], [1, [idx]]),
                ReduceOutput([0, reduce_add, True], [1, reduce_add, True])
            ]
        axis_reduce_list.append(reduce_info)
        axis_split_matrix = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)

    return op_cal_info_in_json


# 'pylint: disable=too-many-instance-attributes,too-few-public-methods
class SquareSumAll():
    """
    Function: use to store square_sum_all base parameters
    """

    # 'pylint: disable=too-many-statements
    def __init__(self, input_x, input_y, kernel_name):
        """
        Init square_sum_all base  parameters

        Parameters
        ----------
        input_x: dict
            data of input_x
            datatype suports float32
        input_y: dict
            data of input_y
            datatype suports float32
        kernel_name: str
            the name of the operator

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.use_data_copy_pad = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ["Ascend910B", "Ascend910_93"]
        self.device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.input_x_shape = input_x.get("shape")
        self.input_x_dtype = input_x.get("dtype").lower()
        self.input_y_shape = input_y.get("shape")
        self.input_y_dtype = input_y.get("dtype").lower()

        self.shape_one_dim = (functools.reduce(operator.mul, self.input_x_shape), )
        self.input_x_num = self.shape_one_dim[0]

        # Reserved 64 Bytes for the two inputs 32B alignment
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 64

        self.kernel_name = kernel_name
        self.dtype_bytes_size = get_bit_len(self.input_x_dtype) // 8
        one_block_bytes_size = tbe_platform.VECTOR_INST_BLOCK_WIDTH // tbe_platform.VECTOR_INST_BLOCK_NUM
        self.data_each_block = one_block_bytes_size // self.dtype_bytes_size

        self.vector_process_bytes = 256

        self.vector_mask_max = tbe_platform.VECTOR_INST_BLOCK_NUM * self.data_each_block

        if self.input_x_num < self.data_each_block:
            self.block_num = 1
        else:
            ai_core_num = self.device_core_num
            temp_num = math.ceil(self.input_x_num / MINIMUM_DATA_NUM_EACH_CORE)
            if temp_num < 32:
                self.block_num = temp_num
            else:
                self.block_num = ai_core_num

        self.data_num_each_core = self.input_x_num // self.block_num
        self.remain_core = self.input_x_num % self.block_num
        self.process_data_num_each_core = self.data_num_each_core

        self._check_param()

        self.input_x_gm = self.tik_instance.Tensor(self.input_x_dtype,
                                                   self.shape_one_dim,
                                                   name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.input_y_gm = self.tik_instance.Tensor(self.input_y_dtype,
                                                   self.shape_one_dim,
                                                   name="input_y_gm",
                                                   scope=tik.scope_gm)
        self.output_x_gm = self.tik_instance.Tensor(self.input_x_dtype, (1, ),
                                                    name="output_x_gm",
                                                    scope=tik.scope_gm,
                                                    is_atomic_add=True)
        self.output_y_gm = self.tik_instance.Tensor(self.input_y_dtype, (1, ),
                                                    name="output_y_gm",
                                                    scope=tik.scope_gm,
                                                    is_atomic_add=True)

        self.input_x_ub = None
        self.input_y_ub = None
        self.every_process_data_num = None
        self.process_times = None
        self.core_tail_num = None

    def _check_param(self):
        """
        Check parameter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        para_check.check_shape(self.input_x_shape, param_name="input_x")
        para_check.check_shape(self.input_y_shape, param_name="input_y")
        para_check.check_dtype(self.input_x_dtype, ("float32", ), param_name="input_x")
        para_check.check_dtype(self.input_y_dtype, ("float32", ), param_name="input_y")

        add_support = tbe_platform.api_check_support("tik.vadd", "float32")

        if self.input_x_dtype != self.input_y_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "input_x", "input_y",
                                                                  self.input_x_dtype, self.input_y_dtype)

        if self.input_x_dtype == "float32" and not add_support:
            error_manager_vector.raise_err_input_dtype_not_supported(self.kernel_name, "input_x", [],
                                                                     self.input_x_dtype)

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
        tail_num = calc_num % (align_value)
        calc_num = (calc_num // align_value) * align_value
        total_sum_num = calc_num

        calc_num = calc_num // 2
        add_loop = calc_num // (self.vector_mask_max * 255)
        calc_offset = 0
        if add_loop > 0:
            with self.tik_instance.for_range(0, add_loop) as add_index:
                calc_offset = add_index * self.vector_mask_max * 255
                self._vector_add(self.vector_mask_max, calc_offset, calc_offset, calc_offset + calc_num, 255)
            calc_offset = add_loop * self.vector_mask_max * 255
        repeat_time = (calc_num % (self.vector_mask_max * 255)) // self.vector_mask_max
        if repeat_time > 0:
            self._vector_add(self.vector_mask_max, calc_offset, calc_offset, calc_offset + calc_num, repeat_time)
        last_num = calc_num % self.vector_mask_max
        if last_num > 0:
            calc_offset += repeat_time * self.vector_mask_max
            self._vector_add(last_num, calc_offset, calc_offset, calc_offset + calc_num, 1)
        if tail_num > 0:
            last_num = tail_num % self.vector_mask_max
            self._vector_add(last_num, 0, 0, total_sum_num, 1)
        return calc_num

    def _init_ub_tensor(self, process_data_num):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        process_data_num: int
            the number of process data each core

        Returns
        -------
        None
        """
        process_data_num_each_core = process_data_num
        every_process_data_num = process_data_num
        ub_max_num = self.ub_size_bytes // self.dtype_bytes_size

        if process_data_num_each_core > ub_max_num // 2:
            every_process_data_num = ub_max_num // 2

        self.every_process_data_num = every_process_data_num
        self.process_times = process_data_num_each_core // every_process_data_num
        self.core_tail_num = process_data_num_each_core % every_process_data_num

        flag = self.data_each_block
        assign_ub_shape = (math.ceil(every_process_data_num / flag) * flag, )

        self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                   assign_ub_shape,
                                                   name="input_x_ub",
                                                   scope=tik.scope_ubuf)
        self.input_y_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                   assign_ub_shape,
                                                   name="input_y_ub",
                                                   scope=tik.scope_ubuf)

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
            self._calc_op(self.every_process_data_num, data_offset)

        data_offset = temp_offset + self.every_process_data_num * self.process_times
        if self.core_tail_num > 0:
            self._calc_op(self.core_tail_num, data_offset)

    # 'pylint: disable=too-many-arguments,
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

    def _calc_op(self, calc_num, offset):
        """
        every process square_sum

        Parameters
        ----------
        calc_num: int
            the number of every process data
        offset: int
            the offset of data address

        Returns
        -------
        None
        """
        if self.use_data_copy_pad:
            byte_burst_len = self.dtype_bytes_size * calc_num
            self.tik_instance.data_move_pad(self.input_x_ub, self.input_x_gm[offset],
                                            1, byte_burst_len, dst_gap=0, src_gap=0)
            self.tik_instance.data_move_pad(self.input_y_ub, self.input_y_gm[offset],
                                            1, byte_burst_len, dst_gap=0, src_gap=0)
        else:
            burst_len = math.ceil(calc_num / self.data_each_block)
            self.tik_instance.data_move(self.input_x_ub, self.input_x_gm[offset], 0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(self.input_y_ub, self.input_y_gm[offset], 0, 1, burst_len, 0, 0)

        calc_loop = calc_num // (self.vector_mask_max * 255)
        calc_offset = 0
        if calc_loop > 0:
            with self.tik_instance.for_range(0, calc_loop) as add_index:
                calc_offset = add_index * self.vector_mask_max * 255
                self._vector_mul(self.vector_mask_max, calc_offset, calc_offset, calc_offset, 255, [self.input_x_ub])
            with self.tik_instance.for_range(0, calc_loop) as add_index:
                calc_offset = add_index * self.vector_mask_max * 255
                self._vector_mul(self.vector_mask_max, calc_offset, calc_offset, calc_offset, 255, [self.input_y_ub])
            calc_offset = self.vector_mask_max * 255 * (calc_loop)

        repeat_time = (calc_num % (self.vector_mask_max * 255) // self.vector_mask_max)

        if repeat_time > 0:
            self._vector_mul(self.vector_mask_max, calc_offset, calc_offset, calc_offset, repeat_time,
                             [self.input_x_ub, self.input_y_ub])
        last_num = calc_num % self.vector_mask_max
        if last_num > 0:
            calc_offset += repeat_time * self.vector_mask_max
            self._vector_mul(last_num, calc_offset, calc_offset, calc_offset, 1, [self.input_x_ub, self.input_y_ub])
        while calc_num > self.vector_process_bytes // self.dtype_bytes_size:
            calc_num = self._reduce_sum(calc_num)
            if calc_num <= self.vector_process_bytes // self.dtype_bytes_size:
                break
        vcadd_mask = calc_num
        self.tik_instance.vcadd(vcadd_mask, self.input_x_ub, self.input_x_ub, 1, 1, 1, 8)
        self.tik_instance.vcadd(vcadd_mask, self.input_y_ub, self.input_y_ub, 1, 1, 1, 8)

        self.tik_instance.set_atomic_add(1)
        if self.use_data_copy_pad:
            self.tik_instance.data_move_pad(self.output_x_gm, self.input_x_ub,
                                            1, self.dtype_bytes_size, dst_gap=0, src_gap=0)
            self.tik_instance.data_move_pad(self.output_y_gm, self.input_y_ub,
                                            1, self.dtype_bytes_size, dst_gap=0, src_gap=0)
        else:
            self.tik_instance.data_move(self.output_x_gm, self.input_x_ub, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.output_y_gm, self.input_y_ub, 0, 1, 1, 0, 0)
        self.tik_instance.set_atomic_add(0)

    def square_sum_all_operator(self):
        """
        SquareSumAll operation

        Parameters
        ----------
        None

        Returns:

        ----------
        tik_instance: tik instance
        """
        if self.block_num > 1:
            process_data_num = self.data_num_each_core
            process_data_extern_num = self.data_num_each_core + self.remain_core
            with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as loop_index:
                move_offset = loop_index * self.data_num_each_core
                with self.tik_instance.if_scope(loop_index == self.block_num - 1):
                    self._init_ub_tensor(process_data_extern_num)
                    self._execute_square_sum_all(move_offset)
                with self.tik_instance.else_scope():
                    self._init_ub_tensor(process_data_num)
                    self._execute_square_sum_all(move_offset)
        else:
            self._init_ub_tensor(self.data_num_each_core)
            move_offset = 0
            self._execute_square_sum_all(move_offset)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.input_x_gm, self.input_y_gm),
                                   outputs=(self.output_x_gm, self.output_y_gm),
                                   enable_l2=False)

        return self.tik_instance


# 'pylint: disable=unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def square_sum_all(input_x, input_y, output_x, output_y, kernel_name="square_sum"):
    """
    calculating square_sum

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

    Returns
    -------
    None
    """

    square_sum_all_res = SquareSumAll(input_x, input_y, kernel_name)
    square_sum_all_res.square_sum_all_operator()
