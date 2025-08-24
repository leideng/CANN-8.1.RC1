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
eye
"""

import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.constant_util import SHAPE_SIZE_LIMIT


class Eye:
    """
    Function: use to create a 2-D tensor with ones on the diagobal and zeros elsewhere
    Create: 2020-07-10
    Modify: 2020-12-17
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, y, num_rows, num_columns, batch_shape, dtype="float32", kernel_name="eye"):
        """
        initialize the eye function

        Parameters
        ----------
        Returns
        -------
        """

        if num_rows <= 0:
            raise ValueError("The num_rows must be greater than 0, got {}.".format(num_rows))

        for index, value in enumerate(batch_shape):
            if value <= 0:
                raise ValueError("The batch_shape[{}] should be more than 0.".format(index))

        self.y = y
        self.num_rows = num_rows
        self.num_columns = num_rows if num_columns is None or num_columns <= 0 else num_columns
        self.batch_shape = batch_shape
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(disable_debug=False)
        self.batch_num = 1
        self.dtype = dtype
        self.max_loop_time = 65535

        block_bytes_size = 32

        # get the capacity of Unified Buffer (bytes)
        ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        # get the number of element in one blok according to 'dtype'
        dtype_bytes_size = tbe_platform.get_bit_len(self.dtype) // 8
        self.num_elem_each_block = block_bytes_size // dtype_bytes_size

        # the max number of element can be used in UB
        self.max_num_elem_each_ub = ub_size_bytes // dtype_bytes_size

        # the number of one in output
        self.num_one = min(self.num_rows, self.num_columns)

        # the offset of each one in output_y_gm
        self.offset_one = self.num_columns + 1

        # the number of element in one aicore
        self.num_elem_one_batch = self.num_rows * self.num_columns

        # the number of element in Global Memory
        self.num_elem_output_y_gm = self.num_elem_one_batch
        if batch_shape:
            for item in batch_shape:
                self.batch_num *= item
        self.num_elem_output_y_gm *= self.batch_num

        # the max value of 'mask'
        self.vec_mask_max = 8 * self.num_elem_each_block

        # the output tensor in Global Memory
        if self.num_elem_output_y_gm > SHAPE_SIZE_LIMIT:
            raise RuntimeError("The tensor is too large.")

        self.output_y_gm = self.tik_instance.Tensor(self.dtype,
                                                    (self.num_elem_output_y_gm,),
                                                    name="output_y_gm",
                                                    scope=tbe_platform.scope_gm)

        self.scalar_zero = self.tik_instance.Scalar(init_value=0, dtype=self.dtype)
        self.scalar_one = self.tik_instance.Scalar(init_value=1, dtype=self.dtype)
        self.scalar_max_num_ub = self.tik_instance.Scalar()

    def eye_compute(self):
        """
        compute the eye function

        Parameters
        ----------
        Returns
        -------
        """

        if self.offset_one < self.num_elem_each_block:
            self.eye_compute_small_col_any_row_matrix_with_any_batch_shape()
        elif self.num_elem_each_block <= self.offset_one <= self.max_num_elem_each_ub:
            self.eye_compute_mid_col_any_row_matrix_with_any_batch_shape()
        else:
            self.eye_compute_large_col_any_row_matrix_with_any_batch_shape()

        # make the shape of output to batch_shape + (num_rows, num_columns)
        shape = self.batch_shape + (self.num_rows, self.num_columns)
        self.output_y_gm = self.output_y_gm.reshape(shape)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[],
            outputs=[self.output_y_gm]
        )

        return self.tik_instance

    def eye_compute_small_col_any_row_matrix_with_any_batch_shape(self):
        """
        compute in small_col_any_row_matrix_with_any_batch_shape

        Prerequisites:
            - col < (num_elem_each_block - 1)

        Cases:
            row < num_elem_each_block
            row >= num_elem_each_block

        Parameters
        ----------
        Returns
        -------
        """

        if self.num_rows < self.num_elem_each_block * 2:
            self.eye_compute_small_col_small_row_matrix_with_any_batch_shape()
        else:
            self.eye_compute_small_col_large_row_matrix_with_any_batch_shape()

    def eye_compute_small_col_small_row_matrix_with_any_batch_shape(self):
        """
        compute in small_col_small_row_matrix_with_any_batch_shape

        Prerequisites:
            - col < (num_elem_each_block - 1)
            - row < num_elem_each_block * 2

        Cases:
            batch_num <= num_elem_each_block
            batch_num > num_elem_each_block

        Parameters
        ----------
        Returns
        -------
        """
        if self.batch_num <= self.num_elem_each_block:
            self.eye_compute_small_col_small_row_matrix_with_small_batch_shape(0, self.batch_num)
        else:
            self.eye_compute_small_col_small_row_matrix_with_large_batch_shape()

    def eye_compute_small_col_small_row_matrix_with_small_batch_shape(self, offset_batch, batch_num):
        """
        compute in small_col_small_row_matrix_with_small_batch_shape

        Prerequisites:
            - col < (num_elem_each_block - 1)
            - row < num_elem_each_block * 2
            - batch_num <= num_elem_each_block

        Cases:
            No cases.

        Parameters
        ----------
        offset_gm : int
            element offset in GM
        batch_num : int
            batch number to compute

        Returns
        -------
        """
        if batch_num > self.num_elem_each_block:
            raise ValueError("The batch_num({}) is too large.".format(batch_num))

        offset_gm = self.num_elem_one_batch * offset_batch
        ub_size = self.num_elem_one_batch * batch_num
        ub_size = self.to_block_unit(ub_size)
        with self.tik_instance.new_stmt_scope():
            output_y_ub = self.tik_instance.Tensor(self.dtype,
                                                   (ub_size,),
                                                   name="output_y_ub",
                                                   scope=tbe_platform.scope_ubuf)
            self.dup_zero_to_ub(output_y_ub, ub_size)

            # set one in batch_num batches
            for i in range(batch_num):
                offset_gm_i = i * self.num_elem_one_batch
                for j in range(self.num_one):
                    offset_gm_j = offset_gm_i + j * self.offset_one
                    output_y_ub[offset_gm_j].set_as(self.scalar_one)

            # move data from ub to gm
            self.tik_instance.data_move(self.output_y_gm[offset_gm],
                                        output_y_ub, 0, 1, ub_size // self.num_elem_each_block, 0, 0)

    def eye_compute_small_col_small_row_matrix_with_large_batch_shape(self):
        """
        compute in small_col_small_row_matrix_with_large_batch_shape

        Prerequisites:
            - col < (num_elem_each_block - 1)
            - row < num_elem_each_block * 2
            - batch_num > num_elem_each_block

        Cases:
            No cases.

        Parameters
        ----------
        Returns
        -------
        """
        loop_time = self.batch_num // self.num_elem_each_block
        rest_batch_num = self.batch_num % self.num_elem_each_block

        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time, block_num=loop_time) as index:
                offset_batch = index * self.num_elem_each_block
                self.eye_compute_small_col_small_row_matrix_with_small_batch_shape(offset_batch,
                                                                                   self.num_elem_each_block)

        if rest_batch_num > 0:
            offset_batch = loop_time * self.num_elem_each_block
            self.eye_compute_small_col_small_row_matrix_with_small_batch_shape(offset_batch,
                                                                               rest_batch_num)

    def eye_compute_small_col_large_row_matrix_with_any_batch_shape(self):
        """
        compute in small_col_small_row_matrix_with_any_batch_shape

        Prerequisites:
            - col < (num_elem_each_block - 1)
            - row >= num_elem_each_block * 2

        Cases:
            No cases.

        Parameters
        ----------
        Returns
        -------
        """
        self.eye_compute_zeros_in_tail()

        loop_time = self.batch_num

        with self.tik_instance.for_range(0, loop_time, block_num=loop_time) as index:
            offset_gm = index * self.num_elem_one_batch

            ub_size = self.num_one * self.offset_one
            ub_size = self.to_block_unit(ub_size)
            output_y_ub = self.tik_instance.Tensor(self.dtype,
                                                   (ub_size,),
                                                   name="output_y_ub",
                                                   scope=tbe_platform.scope_ubuf)
            self.dup_zero_to_ub(output_y_ub, ub_size)
            for i in range(self.num_one):
                offset_ub = i * self.offset_one
                output_y_ub[offset_ub].set_as(self.scalar_one)

            # move data from ub to gm
            self.tik_instance.data_move(self.output_y_gm[offset_gm],
                                        output_y_ub, 0, 1, ub_size // self.num_elem_each_block, 0, 0)

    def eye_compute_mid_col_any_row_matrix_with_any_batch_shape(self):
        """
        compute in mid_col_any_row_matrix_with_any_batch_shape

        Prerequisites:
            - (num_elem_each_block - 1) <= col <= (max_num_elem_each_ub - 1)

        Cases:
            No cases.

        Parameters
        ----------
        Returns
        -------
        """
        loop_time = self.num_one - 1

        if loop_time > 0:
            self.eye_compute_zeros_in_tail()
            ub_size = self.to_block_unit(self.offset_one)

            with self.tik_instance.for_range(0, loop_time, block_num=loop_time) as index:
                offset_row = index * self.offset_one
                output_y_ub = self.tik_instance.Tensor(self.dtype, (ub_size,), name="output_y_ub",
                                                       scope=tbe_platform.scope_ubuf)
                self.dup_zero_to_ub(output_y_ub, ub_size)
                output_y_ub[0].set_as(self.scalar_one)
                if self.offset_one % self.num_elem_each_block:
                    output_y_ub[self.offset_one].set_as(self.scalar_one)
                for i in range(self.batch_num):
                    offset_gm = i * self.num_elem_one_batch
                    # move data from ub to gm
                    self.tik_instance.data_move(self.output_y_gm[offset_gm + offset_row],
                                                output_y_ub, 0, 1, ub_size // self.num_elem_each_block, 0, 0)
            self.eye_compute_last_one_in_batches()
        else:
            ub_size = self.to_block_unit(self.num_elem_one_batch)
            output_y_ub = self.tik_instance.Tensor(
                self.dtype, (ub_size,), name="output_y_ub", scope=tbe_platform.scope_ubuf)
            self.dup_zero_to_ub(output_y_ub, ub_size)
            output_y_ub[0].set_as(self.scalar_one)
            for i in range(self.batch_num):
                offset_gm = i * self.num_elem_one_batch
                # move data from ub to gm
                self.tik_instance.data_move(self.output_y_gm[offset_gm],
                                            output_y_ub, 0, 1, ub_size // self.num_elem_each_block, 0, 0)

    def eye_compute_large_col_any_row_matrix_with_any_batch_shape(self):
        """
        compute in large_col_any_row_matrix_with_any_batch_shape

        Prerequisites:
            - col >= max_num_elem_each_ub

        Parameters
        ----------
        Returns
        -------
        """
        ub_size = self.max_num_elem_each_ub // 2

        loop_time = self.num_one - 1

        if 0 < loop_time <= self.max_loop_time:
            self.eye_compute_zeros_in_tail()
            offset_list = self.get_offset_list(self.offset_one, ub_size)

            with self.tik_instance.for_range(0, loop_time, block_num=loop_time) as index:
                output_one_ub = self.tik_instance.Tensor(self.dtype, (ub_size,), name="output_one_ub",
                                                         scope=tbe_platform.scope_ubuf)
                output_zeros_ub = self.tik_instance.Tensor(self.dtype, (ub_size,), name="output_zeros_ub",
                                                           scope=tbe_platform.scope_ubuf)
                self.dup_zero_to_ub(output_one_ub, ub_size)
                self.dup_zero_to_ub(output_zeros_ub, ub_size)
                output_one_ub[0].set_as(self.scalar_one)

                offset_row = index * self.offset_one

                for i in range(self.batch_num):
                    offset_gm = i * self.num_elem_one_batch

                    self.tik_instance.data_move(self.output_y_gm[offset_gm + offset_row],
                                                output_one_ub, 0, 1, ub_size // self.num_elem_each_block, 0, 0)

                    for item in offset_list:
                        self.tik_instance.data_move(self.output_y_gm[offset_gm + offset_row + item],
                                                    output_zeros_ub, 0, 1, ub_size // self.num_elem_each_block, 0, 0)
            self.eye_compute_last_one_in_batches()
        elif loop_time == 0:

            offset_list = self.get_offset_list(self.num_elem_one_batch, ub_size)

            output_one_ub = self.tik_instance.Tensor(self.dtype,
                                                     (ub_size,),
                                                     name="output_one_ub",
                                                     scope=tbe_platform.scope_ubuf)
            output_zeros_ub = self.tik_instance.Tensor(self.dtype,
                                                       (ub_size,),
                                                       name="output_zeros_ub",
                                                       scope=tbe_platform.scope_ubuf)
            self.dup_zero_to_ub(output_one_ub, ub_size)
            self.dup_zero_to_ub(output_zeros_ub, ub_size)
            output_one_ub[0].set_as(self.scalar_one)

            for i in range(self.batch_num):
                offset_gm = i * self.num_elem_one_batch
                self.tik_instance.data_move(self.output_y_gm[offset_gm],
                                            output_one_ub, 0, 1, ub_size // self.num_elem_each_block, 0, 0)
                for item in offset_list:
                    self.tik_instance.data_move(self.output_y_gm[offset_gm + item],
                                                output_zeros_ub, 0, 1, ub_size // self.num_elem_each_block, 0, 0)
        else:
            raise ValueError("The tensor({}, {}) is too large.".format(self.num_rows, self.num_columns))

    def eye_compute_zeros_in_tail(self):
        """
        compute zeros in tail.

        Prerequisites:

        Cases:
            row == col : no Zeros
            row < col : col - row Zeros
            row > col : (row - col) * col Zeros

        Parameters
        ----------
        Returns
        -------
        """
        if self.num_rows == self.num_columns:
            return
        if self.num_rows < self.num_columns:
            zeros_in_tail = self.num_columns - self.num_rows
        else:
            zeros_in_tail = (self.num_rows - self.num_columns) * self.num_columns

        zeros_to_fill = self.to_block_unit(zeros_in_tail)

        ub_list = self.get_ub_list(zeros_to_fill)

        if not ub_list:
            return
        ub_size = ub_list[0]

        # do it each batch
        with self.tik_instance.new_stmt_scope():
            zeros = self.tik_instance.Tensor(self.dtype,
                                             (ub_size,),
                                             name="zeros",
                                             scope=tbe_platform.scope_ubuf)
            self.dup_zero_to_ub(zeros, ub_size)

            for i in range(self.batch_num):
                offset_gm = (i + 1) * self.num_elem_one_batch - zeros_to_fill
                for item in ub_list:
                    self.tik_instance.data_move(self.output_y_gm[offset_gm],
                                                zeros, 0, 1, item // self.num_elem_each_block, 0, 0)
                    offset_gm += item

    def eye_compute_last_one_in_batches(self):
        """
        compute last one each batch.

        Prerequisites:
            - num_elem_one_batch <= num_elem_each_block

        Cases:
            row == col : no Zeros
            row < col : col - row Zeros
            row > col : (row - col) * col Zeros

        Parameters
        ----------

        Returns
        -------

        """
        with self.tik_instance.new_stmt_scope():
            output_y_ub = self.tik_instance.Tensor(self.dtype,
                                                   (self.num_elem_each_block,),
                                                   name="output_y_ub",
                                                   scope=tbe_platform.scope_ubuf)
            self.dup_zero_to_ub(output_y_ub, self.num_elem_each_block)

            if self.offset_one < self.num_elem_each_block:
                output_y_ub[0].set_as(self.scalar_one)
                last_one = (self.num_one - 1) * self.offset_one + 1
                for i in range(self.batch_num):
                    offset_gm = i * self.num_elem_one_batch
                    offset_gm = offset_gm + last_one
                    self.tik_instance.data_move(self.output_y_gm[offset_gm],
                                                output_y_ub, 0, 1, 1, 0, 0)
            else:
                output_y_ub[self.num_elem_each_block - 1].set_as(self.scalar_one)
                last_one = (self.num_one - 1) * self.offset_one + 1
                for i in range(self.batch_num):
                    offset_gm = i * self.num_elem_one_batch
                    offset_gm = offset_gm + last_one - self.num_elem_each_block
                    self.tik_instance.data_move(self.output_y_gm[offset_gm],
                                                output_y_ub, 0, 1, 1, 0, 0)

    def dup_zero_to_ub(self, output_y_ub, num_elem_output_y_ub):
        """
        dup zero in UB

        Parameters
        ----------

        Returns
        -------

        """
        # make output_ub to zero
        loop_time = num_elem_output_y_ub // self.vec_mask_max
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                offset = loop_index * self.vec_mask_max
                self.tik_instance.vec_dup(self.vec_mask_max, output_y_ub[offset], 0, 1, 8)
        last_num = num_elem_output_y_ub % self.vec_mask_max
        if last_num > 0:
            self.tik_instance.vec_dup(last_num, output_y_ub[loop_time * self.vec_mask_max], 0, 1, 8)

    def to_block_unit(self, num):
        """
        set num in num_elem_each_block unit

        Parameters
        ----------
        num : int

        Returns
        -------

        """
        if num <= 0:
            raise ValueError("The Arg num is too small.")

        return math.ceil(num / self.num_elem_each_block) * self.num_elem_each_block

    def get_ub_list(self, num):
        """
        get sizes and times of ub to data_move

        Parameters
        ----------
        num : int

        Returns
        -------

        """
        ub_list = []
        for _ in range(num // self.max_num_elem_each_ub):
            ub_list.append(self.max_num_elem_each_ub)
        if num % self.max_num_elem_each_ub:
            ub_list.append(num % self.max_num_elem_each_ub)
        return ub_list

    def get_offset_list(self, row_size, ub_size):
        """
        get offset of ub to data_move in one row

        Parameters
        ----------
        row_size : int

        ub_size : int

        Returns
        -------

        """
        offset_list = []
        offset = 0
        for _ in range(row_size // ub_size - 1):
            offset_list.append(offset + ub_size)
            offset += ub_size
        if self.offset_one % ub_size:
            offset_list.append(row_size - ub_size)

        return offset_list


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def eye(y, num_rows, num_columns, batch_shape, dtype=0, kernel_name="eye"):
    """
    calculating data

    Parameters
    ----------
    output_y : dict
        data of output
    num_rows : int
        the number of row
    num_columns : int, optional
        the number of columns with default being n
    batch_shape : list
        A list or tuple of Python integers or a 1-D int32Tensor. It is provided in TensorFlow, not in PyTorch
    dtype : int
        the data type of element
    kernel_name : str
        kernel name, default value is "eye"

    Returns
    -------
    Tik instance
    """
    data_type = {
        0: "float32",
        1: "float16",
        2: "int8",
        3: "int32",
        4: "uint8",
        6: "int16"
    }
    check_tuple = ("float16", "float32", "int32", "int8", "uint8", "int16")

    # check the data type
    y_dtype = y.get("dtype").lower()
    if y_dtype not in check_tuple:
        raise ValueError("There is no y_dtype that is {}.".format(dtype))

    if data_type.get(dtype) is None:
        raise ValueError("There is no dtype that is {}.".format(dtype))

    input_data_type = y_dtype
    if dtype != 0:
        input_data_type = data_type[dtype]

    eye_instance = Eye(y, num_rows, num_columns,
                       batch_shape=batch_shape,
                       dtype=input_data_type,
                       kernel_name=kernel_name)
    tik_instance = eye_instance.eye_compute()
    return tik_instance
