#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Huawei Technologies Co., Ltd
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
population_count
"""
import functools

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl import common_util
from impl import constant_util
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info

# Considering the efficiency of data parallel processing,
# set the number of multicores to 32
MAX_CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

# The maximum number of float16 type data that can be
# stored in Unified Buffer with pingpang
MAX_UB_ELEMENT_NUMBER = 12800

# The maximum number of uint16/int16 that can be processed
# at one repeat time
MAX_SINGLE_REPEAT_NUM = 8

# The computing mode 0 of vsel
VSEL_MODE = 0

# thread num of pingpang
THREAD_NUM = 2


# 'pylint: disable = unused-argument
def get_op_support_info(x, y, kernel_name="population_count"):
    format_x = x.get("format").upper()
    shape_x_len = len(x.get("shape"))
    if format_x in ("ND", "NHWC", "NCHW"):
        axis_split_matrix = []
        for i in range(0, shape_x_len):
            split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
            axis_split_matrix.append(split_0)
        axis_reduce_list = None

    elif format_x == "NC1HWC0":
        axis_split_matrix = [
            [SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])],
            [SplitInput([0, [2], [-1], [-1]]), SplitOutput([0, [2]])],
            [SplitInput([0, [3], [-1], [-1]]), SplitOutput([0, [3]])]
        ]
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=too-many-instance-attributes, no-self-use, invalid-name
def _check_param(input_x, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error

    Parameters
    ----------
    input_x: dict,shape and datatype
    kernel_name: kernel_name
    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_shape(input_shape, param_name="x")
    para_check.check_dtype(input_dtype, ("int16", "uint16"), param_name="x")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def population_count(x, y, kernel_name="population_count"):
    """
    the main function of population_count

    Parameters
    ----------
    x: dict,shape and data type,data type supports int16,uint16
    y: dict,shape and data type,data type supports uint8
    kernel_name: cce kernel name, default value is "population_count"

    Returns
    ----------
    None
    """
    _check_param(x, kernel_name)
    bitcount = PopulationCount(x, y)
    bitcount.tik_instance_function(kernel_name)


class PopulationCountBase:
    """
    Function: use to store population_count base parameters
    Modify : 2019-11-13
    """
    def __init__(self, input_x, output_y):
        """
        init population_count base parameters

        Parameters
        ----------
        input_x: shape and data type,datatype supports int16,uint16
        output_y: shape and data type,data type supports uint8

        Returns
        -------
        None
        """
        self.input_shape, self.input_dtype = self.get_input_params(input_x)
        self.output_shape, self.output_dtype = self.get_output_params(output_y)
        if self.output_dtype != "uint8":
            self.output_dtype = "uint8"
        self.input_data_size = common_util.get_data_size(self.input_dtype)
        self.output_data_size = common_util.get_data_size(self.output_dtype)
        self.tik_instance = tik.Tik()

    def get_input_params(self, input_x):
        """
        get the shape and data type of input_x

        Parameters
        ----------

        Returns
        -------
        shape: the shape of input_x
        type: data type of input_x
        """
        shape = input_x.get("shape")
        dtype = input_x.get("dtype").lower()
        return shape, dtype

    def get_output_params(self, output_y):
        """
        get the shape and data type of output_y

        Parameters
        ----------

        Returns
        -------
        shape: the shape of output_y
        type: data type of output_y
        """
        shape = output_y.get("shape")
        dtype = output_y.get("dtype").lower()
        return shape, dtype


class PopulationCount(PopulationCountBase):
    """
    Function: use to store population_count compute parameters
    Modify : 2019-11-13
    """
    def __init__(self, input_x, output_y):
        """
        init population_count base parameters

        Parameters
        ----------
        input_x: shape and data type,datatype supports int16,uint16
        output_y: shape and data type,data type supports uint8

        Returns
        ----------
        None
        """
        super(PopulationCount, self).__init__(input_x, output_y)
        self.once_repeat_element_number = (constant_util.VECTOR_BYTE_SIZE
                                           // self.input_data_size)
        self.once_burst_element_number = (constant_util.BLOCK_SIZE
                                          // self.input_data_size)
        self.gm_element_num = self.get_gm_element()
        self.core_num = self.get_core_num()
        self.first_core_num, self.not_first_core_num = self.get_core_param()
        self.loop_cycle = self.get_loop_cycle_num()
        self.not_last_loop_num = constant_util.BLOCK_SIZE
        self.last_loop_num = self.get_first_core_last_loop_param()

        self.input_x_gm = self.tik_instance.Tensor(self.input_dtype,
                                                   self.input_shape,
                                                   name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.output_dtype,
                                                    self.output_shape,
                                                    name="output_y_gm",
                                                    scope=tik.scope_gm)

    def get_gm_element(self):
        """
        get the size of input data

        Parameters
        ----------

        Returns
        -------
        gm_element_number: the size of input data
        """
        gm_element_number = int(functools.reduce(lambda i, j: i * j,
                                                 self.input_shape))

        return gm_element_number

    def get_core_num(self):
        """
        get the core number from the size of input data

        Parameters
        ----------

        Returns
        -------
        target_core_number: the core number
        """
        target_core_number = 1
        if self.gm_element_num // (MAX_CORE_NUM *
                                   self.once_burst_element_number *
                                   constant_util.DATA_SIZE_TWO) != 0:
            target_core_number = MAX_CORE_NUM

        return target_core_number

    def get_core_param(self):
        """
        calculate the data that each core should process

        Parameters
        ----------

        Returns
        -------
        first_core_number: The amount of data that the first core should process
        not_first_core_number: The amount of data that should not be processed
        by the first core
        """
        first_core_number = not_first_core_number = self.gm_element_num
        if self.core_num == MAX_CORE_NUM:
            # for every core hava the same data number
            first_core_number = not_first_core_number = int(self.gm_element_num
                                                            // MAX_CORE_NUM)
            if self.gm_element_num % MAX_CORE_NUM != 0:
                first_core_number = self.gm_element_num - (MAX_CORE_NUM - 1) \
                                    * not_first_core_number

        return int(first_core_number), int(not_first_core_number)

    def get_loop_cycle_num(self):
        """
        calculate the number of pingpang cycles per core

        Parameters
        ----------

        Returns
        -------
        loop_cycle: the number of pingpang cycles
        """
        loop_cycle = 1
        if self.first_core_num >= 64:
            loop_cycle = self.first_core_num \
                         // (self.once_burst_element_number *
                             constant_util.DATA_SIZE_TWO)

        return int(loop_cycle)

    def get_first_core_last_loop_param(self):
        """
        calculate the amount of data processed by the tail block of
        the first core

        Parameters
        ----------

        Returns
        -------
        last_loop_number: the first core's tail block processing data
        """
        last_loop_number = self.first_core_num - \
                           (self.loop_cycle
                            * self.not_last_loop_num)

        return int(last_loop_number)

    def single_data_move_mte2_function(self, loop_input, loop_element_num):
        """
        move data from gm to ub for single core

        Parameters
        ----------
        loop_input: the starting address of the gm data
        loop_element_num: the ub data size

        Returns
        -------
        ub_size: the ub data size
        input_x_ub: the ub tensor
        """
        ub_size = loop_element_num
        if loop_element_num % self.once_burst_element_number != 0:
            ub_size = (loop_element_num // self.once_burst_element_number + 1) \
                      * self.once_burst_element_number
        input_x_ub = self.tik_instance.Tensor(self.input_dtype, (ub_size,),
                                              name="input_x_ub",
                                              scope=tik.scope_ubuf)
        data_move_repeats_mte2 = int(ub_size * self.input_data_size
                                     / constant_util.BLOCK_SIZE)
        self.tik_instance.data_move(input_x_ub, self.input_x_gm[loop_input],
                                    constant_util.SID, constant_util.DEFAULT_NBURST,
                                    data_move_repeats_mte2,
                                    constant_util.STRIDE_ZERO,
                                    constant_util.STRIDE_ZERO)

        return ub_size, input_x_ub

    def multicore_data_move_mte2_function(self, loop_input, loop_element_num):
        """
        move data from ub to gm for multi core

        Parameters
        ----------
        loop_input: the starting address of the gm data
        loop_element_num: the ub data size

        Returns
        -------
        ub_size: the ub data size
        input_x_ub: the ub tensor for input data
        """
        ub_size = loop_element_num
        input_x_ub = self.tik_instance.Tensor(self.input_dtype, (ub_size,),
                                              name="input_x_ub",
                                              scope=tik.scope_ubuf)
        data_move_repeats_mte2 = int(ub_size * self.input_data_size
                                     / constant_util.BLOCK_SIZE)
        self.tik_instance.data_move(input_x_ub, self.input_x_gm[loop_input],
                                    constant_util.SID, constant_util.DEFAULT_NBURST,
                                    data_move_repeats_mte2,
                                    constant_util.STRIDE_ZERO,
                                    constant_util.STRIDE_ZERO)

        return ub_size, input_x_ub

    def single_data_move_mte3_function(self, loop_input, output_y_ub_int8,
                                       loop_element_num):
        """
        move data from gm to ub for single core

        Parameters
        ----------
        loop_input: the starting address of the gm data
        output_y_ub_int8: the ub tensor for output data with type int8
        loop_element_num: the ub data size

        Returns
        -------
        None
        """
        if loop_element_num % (self.once_burst_element_number *
                               constant_util.DATA_SIZE_TWO) != 0:
            one_burst_byte_size = self.once_burst_element_number *\
                                  constant_util.DATA_SIZE_TWO
            first_burst_element = (loop_element_num // one_burst_byte_size + 1)\
                                  * one_burst_byte_size

            data_move_repeats_mte3 = int(first_burst_element *
                                         self.output_data_size
                                         / constant_util.BLOCK_SIZE)
            self.tik_instance.data_move(self.output_y_gm[loop_input],
                                        output_y_ub_int8,
                                        constant_util.SID,
                                        constant_util.DEFAULT_NBURST,
                                        data_move_repeats_mte3,
                                        constant_util.STRIDE_ZERO,
                                        constant_util.STRIDE_ZERO)
        else:
            data_move_repeats_mte3 = int(loop_element_num*self.output_data_size
                                         / constant_util.BLOCK_SIZE)
            self.tik_instance.data_move(self.output_y_gm[loop_input],
                                        output_y_ub_int8, constant_util.SID,
                                        constant_util.DEFAULT_NBURST,
                                        data_move_repeats_mte3,
                                        constant_util.STRIDE_ZERO,
                                        constant_util.STRIDE_ZERO)

    def multicore_data_move_mte3_function(self, loop_input, output_y_ub_int8,
                                          loop_element_num):
        """
        move data from gm to ub for multi core

        Parameters
        ----------
        loop_input: the starting address of the gm data
        output_y_ub_int8: the ub tensor for output data with type int8
        loop_element_num: the ub data size

        Returns
        -------
        None
        """
        data_move_repeats_mte3 = int(loop_element_num*self.output_data_size
                                     / constant_util.BLOCK_SIZE)
        self.tik_instance.data_move(self.output_y_gm[loop_input],
                                    output_y_ub_int8, constant_util.SID,
                                    constant_util.DEFAULT_NBURST,
                                    data_move_repeats_mte3,
                                    constant_util.STRIDE_ZERO,
                                    constant_util.STRIDE_ZERO)

    def get_src_tensor(self):
        """
        produce two tensors with all zero or all one

        Parameters
        ----------

        Returns
        -------
        src0_ub: the tensor with zero
        src1_ub: the tensor with one
        """
        one_scalar = self.tik_instance.Scalar(dtype="float16",
                                              name="one_scalar", init_value=1.0)
        zero_scalar = self.tik_instance.Scalar(dtype="float16",
                                               name="zero_scalar",
                                               init_value=0.0)
        src0_ub = self.tik_instance.Tensor("float16",
                                           (self.once_repeat_element_number,),
                                           name="src0_ub", scope=tik.scope_ubuf)
        src1_ub = self.tik_instance.Tensor("float16",
                                           (self.once_repeat_element_number,),
                                           name="src1_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(constant_util.MASK128, src0_ub, one_scalar,
                                     constant_util.DEFAULT_REPEAT_TIME,
                                     constant_util.STRIDE_ONE,
                                     constant_util.REPEAT_STRIDE_EIGHT)
        self.tik_instance.vector_dup(constant_util.MASK128, src1_ub, zero_scalar,
                                     constant_util.DEFAULT_REPEAT_TIME,
                                     constant_util.STRIDE_ONE,
                                     constant_util.REPEAT_STRIDE_EIGHT)

        return src0_ub, src1_ub

    def get_cmp_mask(self):
        """
        convert input tensor to cmpmask for vsel

        Parameters
        ----------

        Returns
        -------
        cmp_mask_dst_ub: the cmpmask of the input tensor
        """
        cmp_mask_dst_ub = self.tik_instance.Tensor("float16", (
            self.once_repeat_element_number,),
                                                   name="cmp_mask_dst_ub",
                                                   scope=tik.scope_ubuf)

        return cmp_mask_dst_ub

    def population_count_compute(self, ub_size, input_x_ub, src0_ub, src1_ub):
        """
        describe the population_count calculation process

        Parameters
        ----------
        ub_size: the ub data size
        input_x_ub: the ub tensor for input data
        src0_ub: the tensor with zero
        src1_ub: the tensor with one

        Returns
        -------
        output_ub_int8: the ub tensor for output data with type int8
        """
        output_ub_float16 = self.tik_instance.Tensor("float16", (ub_size,),
                                                     name="output_ub_float16",
                                                     scope=tik.scope_ubuf)
        output_ub_int8 = self.tik_instance.Tensor(self.output_dtype,
                                                  (ub_size,),
                                                  name="output_ub_int8",
                                                  scope=tik.scope_ubuf)
        cmp_mask_tensor = input_x_ub.reinterpret_cast_to("uint64")
        cmp_mask_dst_ub = self.get_cmp_mask()
        with self.tik_instance.for_range(0, ub_size / MAX_SINGLE_REPEAT_NUM) \
                as sub_cycle:
            cmp_mask = self.tik_instance.mov_tensor_to_cmpmask(
                cmp_mask_tensor[constant_util.DATA_SIZE_TWO * sub_cycle])

            self.tik_instance.vsel(constant_util.MASK128, VSEL_MODE, cmp_mask_dst_ub,
                                   cmp_mask,
                                   src0_ub, src1_ub,
                                   constant_util.DEFAULT_REPEAT_TIME,
                                   constant_util.STRIDE_ONE, constant_util.STRIDE_ONE,
                                   constant_util.STRIDE_ONE,
                                   constant_util.REPEAT_STRIDE_EIGHT,
                                   constant_util.REPEAT_STRIDE_EIGHT,
                                   constant_util.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vcgadd(constant_util.MASK128, output_ub_float16[
                MAX_SINGLE_REPEAT_NUM * sub_cycle],
                                     cmp_mask_dst_ub,
                                     constant_util.DEFAULT_REPEAT_TIME,
                                     constant_util.STRIDE_ONE,
                                     constant_util.STRIDE_ONE,
                                     constant_util.REPEAT_STRIDE_EIGHT)
        self.tik_instance.vconv(ub_size, '',
                                output_ub_int8[0],
                                output_ub_float16[0],
                                constant_util.STRIDE_ONE,
                                constant_util.STRIDE_ONE,
                                constant_util.STRIDE_ONE,
                                constant_util.REPEAT_STRIDE_FOUR,
                                constant_util.REPEAT_STRIDE_EIGHT)

        return output_ub_int8

    def single_core_compute(self, src0_ub, src1_ub):
        """
        calculated on single core

        Parameters
        ----------
        src0_ub: the tensor with zero
        src1_ub: the tensor with one

        Returns
        -------
        None
        """
        if self.loop_cycle == 1:
            loop_input = 0
            ub_size, input_x_ub = self.single_data_move_mte2_function(
                loop_input, self.first_core_num)
            output_y_ub_int8 = self.population_count_compute(ub_size,
                                                             input_x_ub,
                                                             src0_ub,
                                                             src1_ub)
            self.single_data_move_mte3_function(loop_input,
                                                output_y_ub_int8,
                                                self.first_core_num)
        else:
            with self.tik_instance.for_range(0,
                                             self.loop_cycle,
                                             thread_num=THREAD_NUM) as cycle:
                loop_input = cycle * self.not_last_loop_num
                ub_size, input_x_ub = self.single_data_move_mte2_function(
                    loop_input, self.not_last_loop_num)
                output_y_ub_int8 = self.population_count_compute(
                    ub_size, input_x_ub, src0_ub, src1_ub)
                self.single_data_move_mte3_function(loop_input,
                                                    output_y_ub_int8,
                                                    self.not_last_loop_num)
            if self.last_loop_num != 0:
                loop_input = (self.loop_cycle - 1) * \
                             self.not_last_loop_num + \
                             self.last_loop_num

                ub_size, input_x_ub = self.single_data_move_mte2_function(
                    loop_input, self.not_last_loop_num)
                output_y_ub_int8 = self.population_count_compute(
                    ub_size,
                    input_x_ub, src0_ub, src1_ub)
                self.single_data_move_mte3_function(loop_input,
                                                    output_y_ub_int8,
                                                    self.not_last_loop_num)

    def multi_core_compute(self, block_num, src0_ub, src1_ub):
        """
        calculated on multi core

        Parameters
        ----------
        src0_ub: the tensor with zero
        src1_ub: the tensor with one

        Returns
        -------
        None
        """
        if self.loop_cycle == 1:
            loop_input = block_num * self.not_first_core_num
            ub_size, input_x_ub = self.multicore_data_move_mte2_function(
                loop_input, self.not_last_loop_num)
            output_y_ub_int8 = self.population_count_compute(
                ub_size, input_x_ub, src0_ub, src1_ub)
            self.multicore_data_move_mte3_function(loop_input,
                                                   output_y_ub_int8,
                                                   self.not_last_loop_num)
        else:
            loop_input = block_num * self.not_first_core_num
            with self.tik_instance.for_range(0,
                                             self.loop_cycle,
                                             thread_num=THREAD_NUM) as cycle:
                loop_input = cycle * self.not_last_loop_num + \
                             loop_input
                ub_size, input_x_ub = self.multicore_data_move_mte2_function(
                    loop_input,
                    self.not_last_loop_num)
                output_y_ub_int8 = self.population_count_compute(
                    ub_size, input_x_ub, src0_ub, src1_ub)
                self.multicore_data_move_mte3_function(loop_input,
                                                       output_y_ub_int8,
                                                       self.not_last_loop_num)
        if self.last_loop_num != 0:
            loop_input = block_num * self.not_first_core_num
            loop_input = (self.loop_cycle - 1) * \
                         self.not_last_loop_num + \
                         self.last_loop_num + loop_input

            ub_size, input_x_ub = self.multicore_data_move_mte2_function(
                loop_input, self.not_last_loop_num)
            output_y_ub_int8 = self.population_count_compute(
                ub_size, input_x_ub, src0_ub, src1_ub)
            self.multicore_data_move_mte3_function(loop_input,
                                                   output_y_ub_int8,
                                                   self.not_last_loop_num)

    def tik_instance_function(self, kernel_name):
        """
        the entry of population_count calculation

        Parameters
        ----------
        kernel_name: cce kernel name, default value is "population_count"

        Returns
        -------
        None
        """
        if self.core_num == 1:
            src0_ub, src1_ub = self.get_src_tensor()
            self.single_core_compute(src0_ub, src1_ub)
        else:
            with self.tik_instance.for_range(0, self.core_num,
                                             block_num=self.core_num) \
                    as block_num:
                src0_ub, src1_ub = self.get_src_tensor()
                self.multi_core_compute(block_num, src0_ub, src1_ub)

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[self.input_x_gm],
                                   outputs=[self.output_y_gm])
