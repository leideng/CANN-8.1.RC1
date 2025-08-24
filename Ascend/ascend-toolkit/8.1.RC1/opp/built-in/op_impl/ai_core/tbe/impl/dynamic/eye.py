# Copyright 2023 Huawei Technologies Co., Ltd
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
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context


class Constant:
    MAX_INT32_VALUE = 2 ** 31 - 1
    BLOCK_BYTES_SIZE = 32
    EIGHT_BIT = 8
    BUFFER_SIZE = 16 * 1024
    TILING_ARG_NUM = 7
    TILING_ARG_DTYPE = "int64"
    TILING_MODE_SMALL_MATRIX = 0
    TILING_MODE_MID_MATRIX = 1
    TILING_MODE_LARGE_MATRIX = 2


class Eye:
    """
    Function: use to create a 2-D tensor with ones on the diagobal and zeros elsewhere
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, y, num_rows, num_columns, batch_shape, dtype, kernel_name="eye"):
        """
        initialize the eye function

        Parameters
        ----------
        Returns
        -------
        """
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.aicore_num = tik.Dprofile().get_aicore_num()
        self.dtype = y.get("dtype").lower()
        if self.dtype == "bool":
            self.dtype = "int8"

        self.num_rows = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        self.num_columns = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        self.batch_num = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE, init_value=1)
        self.task_per_core = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        self.task_tail = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        self.tiling_mode  = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        self.tiling_core_num = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_ARG_DTYPE, [Constant.TILING_ARG_NUM],
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.get_tiling_args()

        # get the number of element in one blok according to 'dtype'
        dtype_bytes_size = tbe_platform.get_bit_len(self.dtype) // Constant.EIGHT_BIT
        self.num_elem_per_block = Constant.BLOCK_BYTES_SIZE // dtype_bytes_size
        
        # the number of one in output
        self.num_ones_per_batch = self.tik_instance.Scalar(init_value=self.num_columns)
        with self.tik_instance.if_scope(self.num_rows < self.num_columns):
            self.num_ones_per_batch.set_as(self.num_rows)

        # the base offset of each one in output_y_gm
        self.offset_one = self.num_columns + 1

        # the number of element in one aicore
        self.num_elem_per_batch = self.num_rows * self.num_columns
        self.output_y_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32_VALUE], name="output_y_gm",
                                                    scope=tbe_platform.scope_gm, is_atomic_add=True)
        self.scalar_one = self.tik_instance.Scalar(init_value = 1, dtype=self.dtype)

    @staticmethod
    def ceil_div(val_x, val_y):
        """
        ceiling division
        """
        return (val_x + val_y - 1) // val_y

    def get_tiling_args(self):
        """get_tiling_args"""
        tiling_ub = self.tik_instance.Tensor(Constant.TILING_ARG_DTYPE, [Constant.TILING_ARG_NUM], 
                                            name='tiling_ub', scope=tik.scope_ubuf)
        burst = self.ceil_div(Constant.TILING_ARG_NUM,
                              Constant.BLOCK_BYTES_SIZE // (tbe_platform.get_bit_len("int64") // Constant.EIGHT_BIT))
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, burst, 0, 0)
        self.num_rows.set_as(tiling_ub[0])
        self.num_columns.set_as(tiling_ub[1])
        self.batch_num.set_as(tiling_ub[2])
        self.task_per_core.set_as(tiling_ub[3])
        self.task_tail.set_as(tiling_ub[4])
        self.tiling_mode.set_as(tiling_ub[5])
        self.tiling_core_num.set_as(tiling_ub[6])

    def task_schedule(self):
        """multi-core task scheduling"""
        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as i_idx:
            with self.tik_instance.for_range(0, self.task_per_core) as j_idx:
                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_SMALL_MATRIX):
                    self.compute_small_matrix()
                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_MID_MATRIX):
                    self.compute_mid_matrix((i_idx * self.task_per_core + j_idx) * self.num_elem_per_batch)
                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_LARGE_MATRIX):
                    self.compute_large_matrix((i_idx * self.task_per_core + j_idx) * self.offset_one)
            with self.tik_instance.if_scope(i_idx < self.task_tail):
                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_SMALL_MATRIX):
                    self.compute_small_matrix()
                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_MID_MATRIX):
                    self.compute_mid_matrix((self.task_per_core * self.tiling_core_num + i_idx) * 
                                             self.num_elem_per_batch)
                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_LARGE_MATRIX):
                    self.compute_large_matrix((self.task_per_core * self.tiling_core_num + i_idx) * 
                                               self.offset_one)
        
        tbe_context.get_context().add_compile_info(
            "vars", {
                "aicore_num": self.aicore_num,
                "num_elem_per_block": self.num_elem_per_block
            })

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[],
            outputs=[self.output_y_gm],
            flowtable=[self.tiling_gm]
        )
        return self.tik_instance
        
    def compute_small_matrix(self):
        """
        compute in small_matrix

        Prerequisites:
            - num_ones_per_batch * num_columns <= num_elem_per_block

        Parameters
        ----------
        Returns
        -------
        """
        output_y_ub = self.tik_instance.Tensor(self.dtype, [self.num_elem_per_block], 
                                               name="output_y_ub", scope=tbe_platform.scope_ubuf)
        # move 0's to y_ub
        self.tik_instance.data_move(output_y_ub, self.output_y_gm, 0, 1, 1, 0, 0)
        offset_ub = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE, init_value=0)
        with self.tik_instance.for_range(0, self.num_ones_per_batch) as one_idx:
            offset_ub.set_as(one_idx * (self.num_columns + 1))
            output_y_ub[offset_ub].set_as(self.scalar_one)
            
        # move output to y_gm
        with self.tik_instance.for_range(0, self.batch_num) as batch_idx:
            offset_batch = batch_idx * self.num_elem_per_batch
            self.tik_instance.data_move(self.output_y_gm[offset_batch], output_y_ub, 0, 1, 1, 0, 0)

    def compute_mid_matrix(self, offset_batch):
        """
        compute in mid_matrix

        Prerequisites:
            - num_ones_per_batch * num_columns > num_elem_per_block
            - num_columns <= num_elem_per_block

        Parameters
        ----------
        Returns
        -------
        """
        # number of ones excluding the last block
        front_num_ones = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        # number of ones within the last block
        tail_num_ones = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        offset_gm = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        offset_ub = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE)
        num_elem_ub = self.num_ones_per_batch * self.num_columns
        # 32B aligned number of elements, max size = 32B * 32B
        aligned_num_elem = (self.ceil_div(self.num_ones_per_batch * self.num_columns, self.num_elem_per_block) * 
                            self.num_elem_per_block)
        output_y_ub = self.tik_instance.Tensor(self.dtype, [aligned_num_elem], 
                                               name="output_y_ub", scope=tbe_platform.scope_ubuf)
        burst = num_elem_ub // self.num_elem_per_block
        front_num_ones = (burst * self.num_elem_per_block - 1) // self.offset_one + 1
        self.tik_instance.data_move(output_y_ub, self.output_y_gm[offset_batch + offset_gm], 0, 1, burst, 0, 0)
        
        with self.tik_instance.for_range(0, front_num_ones) as one_idx:
            offset_ub.set_as(one_idx * self.offset_one)
            output_y_ub[offset_ub].set_as(self.scalar_one)
        self.tik_instance.data_move(self.output_y_gm[offset_batch + offset_gm], output_y_ub, 0, 1, burst, 0, 0)
        
        # address backtracing for the last "one", in case of data trampling between batches(cores)
        offset_gm.set_as(num_elem_ub - self.num_elem_per_block)
        tail_num_ones.set_as(self.num_ones_per_batch - front_num_ones)
        self.tik_instance.data_move(output_y_ub, self.output_y_gm[offset_batch + offset_gm], 0, 1, 1, 0, 0)
        
        # move output to y_gm
        with self.tik_instance.for_range(0, tail_num_ones) as one_idx:
            offset_ub.set_as((front_num_ones + one_idx) * self.offset_one - offset_gm)
            output_y_ub[offset_ub].set_as(self.scalar_one)
        self.tik_instance.data_move(self.output_y_gm[offset_batch + offset_gm], output_y_ub, 0, 1, 1, 0, 0)

    def compute_large_matrix(self, offset_gm):
        """
        compute in large_matrix

        Prerequisites:
            - num_ones_per_batch * num_columns > num_elem_per_block
            - num_columns > num_elem_per_block

        Parameters
        ----------
        Returns
        -------
        """
        offset_gm = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE, init_value=offset_gm)
        offset_ub = self.tik_instance.Scalar(Constant.TILING_ARG_DTYPE, init_value=0)
        # offset of the last "one" of output
        offset_last_one = (self.num_ones_per_batch - 1) * self.offset_one
        output_y_ub = self.tik_instance.Tensor(self.dtype, [self.num_elem_per_block], 
                                               name="output_y_ub", scope=tbe_platform.scope_ubuf)
        
        # last "one" needs address backtracing, in case of data trampling between cores
        with self.tik_instance.if_scope(tik.all(self.num_rows != 1, offset_gm == offset_last_one)):
            offset_gm.set_as(offset_gm - self.num_elem_per_block + 1)
            offset_ub.set_as(self.num_elem_per_block - 1)
            
        # move output to y_gm
        with self.tik_instance.for_range(0, self.batch_num) as batch_idx:
            offset_batch = batch_idx * self.num_elem_per_batch
            self.tik_instance.data_move(output_y_ub, self.output_y_gm[offset_gm + offset_batch], 0, 1, 1, 0, 0)
            output_y_ub[offset_ub].set_as(self.scalar_one)
            self.tik_instance.data_move(self.output_y_gm[offset_gm + offset_batch], output_y_ub, 0, 1, 1, 0, 0)


#'pylint: disable=too-many-arguments
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
    eye_instance = Eye(y, num_rows, num_columns,
                       batch_shape,
                       dtype=dtype,
                       kernel_name=kernel_name)

    tik_instance = eye_instance.task_schedule()
    return tik_instance
