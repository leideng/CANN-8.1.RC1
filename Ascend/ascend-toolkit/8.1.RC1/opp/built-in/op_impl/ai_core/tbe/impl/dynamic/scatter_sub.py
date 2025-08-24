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
scatter_sub
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.util_common import is_support_out_of_bound_index

# max int64 value
MAX_INT64_VALUE = 2 ** 64 - 1
# tiling param num
TILING_ARG_NUM = 17
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# 8 bit
EIGHT_BIT = 8
# bytes of one block
BLOCK_BYTES = 32


# 'pylint: disable=too-many-arguments,too-many-instance-attributes,unused-argument
# 'pylint: disable=too-many-public-methods
# 'pylint: disable=attribute-defined-outside-init
class ScatterSub():
    """
       Function: use to store scatter_sub base parameters
       Modify : 2020-10-29
    """

    def __init__(self, var, indices, updates, var_out, use_locking, kernel_name):
        """
        Init ScatterAdd parameters

        Parameters
        ----------
        var: dict
            the dict of input tensor.
        indices: dict
            the dict of input tensor.
        updates: dict
            the dict of input tensor.
        var_out: dict
            the dict of output tensor.
        use_locking: bool
            not used in this compute, default value is "False".
        kernel_name: str
            cce kernel name, default value is "scatter_sub".

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.var_dtype = var.get("dtype").lower()
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_dtype = updates.get("dtype").lower()
        self.out_dtype = var_out.get("dtype").lower()

        self.check_input_params()

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE)
        self.var_dtype_bytes_size = get_bit_len(self.var_dtype) // EIGHT_BIT
        self.indices_dtype_bytes_size = get_bit_len(self.indices_dtype) // EIGHT_BIT
        self.var_data_each_block = BLOCK_BYTES // self.var_dtype_bytes_size
        self.indices_data_each_block = BLOCK_BYTES // self.indices_dtype_bytes_size

        self.var_ub_num = self.ub_size_bytes // 8 * 3 // self.var_dtype_bytes_size
        self.updates_ub_num = self.ub_size_bytes // 8 * 3 // self.var_dtype_bytes_size
        self.indices_ub_num = self.ub_size_bytes // 4 // self.indices_dtype_bytes_size

        if self.var_dtype in ("float32", "int32"):
            self.data_num_one_repeat = 64
        else:
            self.data_num_one_repeat = 128

        self.var_ub = None
        self.updates_ub = None
        self.indices_ub = None
        self.var_tile_ub = None
        self.updates_tile_ub = None
        self.zero_ub = None
        self.var_read_index = None
        self.core_loop_index = None
        self.var_value = None
        self.update_value = None
        self.var_burst_len = None
        self.indices_burst_len = None
        self.updates_burst_len = None
        self.each_core_max_indice = None
        self.last_var_num = None
        self.var_offset = None
        self.tiling_ub = None
        self.var_last_num_off = None
        self.repeat = None
        self.mask = None
        self.init_scalar_data()

    def init_scalar_data(self):
        self.core_num_var = self.tik_instance.Scalar(name="core_num_var", init_value=self.ai_core_num)
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
        self.indice_step = self.tik_instance.Scalar("int64", name="indice_step")
        self.core_num = self.tik_instance.Scalar("int64", name="core_num")
        self.update_data_num = self.tik_instance.Scalar("int64", name="update_data_num")
        self.indices_loop_num = self.tik_instance.Scalar("int64", name="indices_loop_num")
        self.indices_last_num = self.tik_instance.Scalar("int64", name="indices_last_num")
        self.updates_num = self.tik_instance.Scalar("int64", name="updates_num")
        self.updates_loop_num = self.tik_instance.Scalar("int64", name="updates_loop_num")
        self.updates_last_num = self.tik_instance.Scalar("int64", name="updates_last_num")
        self.var_num = self.tik_instance.Scalar("int64", name="var_num")
        self.var_loop_num = self.tik_instance.Scalar("int64", name="var_loop_num")
        self.var_last_num = self.tik_instance.Scalar("int64", name="var_last_num")
        self.var_each_core_burst_len = self.tik_instance.Scalar("int64", name="var_each_core_burst_len")
        self.var_last_core_burst_len = self.tik_instance.Scalar("int64", name="var_last_core_burst_len")
        self.max_indice = self.tik_instance.Scalar("int64", name="max_indice")
        self.var_each_core_data = self.tik_instance.Scalar("int64", name="var_each_core_data")
        self.tiling_gm = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE,), name="var_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype, (MAX_INT64_VALUE,),
                                                   name="indices_gm", scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE,),
                                                   name="updates_gm",
                                                   scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE,), name="out_gm", scope=tik.scope_gm)

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    def check_input_params(self):
        """
        Check whether the input parameters is valid or not
        """
        indices_support_dtype_list = ("int32", "int64")
        var_support_dtype_list = ("float32", "int32", "float16")
        para_check.check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        para_check.check_dtype(self.var_dtype, var_support_dtype_list, param_name="var")
        if self.var_dtype != self.updates_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "updates", "var",
                                                                  self.updates_dtype, self.var_dtype)
        if self.var_dtype != self.out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "out", "var", self.out_dtype,
                                                                  self.var_dtype)

    def tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from scatter_sub tiling

        Returns
        -------
        None
        """
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.indice_step.set_as(self.tiling_ub[1])
        self.core_num.set_as(self.tiling_ub[2])
        self.update_data_num.set_as(self.tiling_ub[3])
        self.indices_loop_num.set_as(self.tiling_ub[4])
        self.indices_last_num.set_as(self.tiling_ub[5])
        self.updates_num.set_as(self.tiling_ub[6])
        self.updates_loop_num.set_as(self.tiling_ub[7])
        self.updates_last_num.set_as(self.tiling_ub[8])
        self.var_num.set_as(self.tiling_ub[9])
        self.var_loop_num.set_as(self.tiling_ub[10])
        self.var_last_num.set_as(self.tiling_ub[11])
        self.var_each_core_burst_len.set_as(self.tiling_ub[12])
        self.var_last_core_burst_len.set_as(self.tiling_ub[13])
        self.max_indice.set_as(self.tiling_ub[14])
        self.var_each_core_data.set_as(self.tiling_ub[15])
        self.set_running_core_num(self.tiling_ub[16])

    def init_ub_tensor(self):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.var_ub = self.tik_instance.Tensor(self.var_dtype, (self.var_ub_num,), name="var_ub", scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(self.var_dtype, (self.updates_ub_num,),
                                                   name="updates_ub",
                                                   scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (self.indices_ub_num,),
                                                   name="indices_ub",
                                                   scope=tik.scope_ubuf)
        self.var_tile_ub = self.tik_instance.Tensor(self.var_dtype, (self.var_data_each_block,),
                                                    name="var_tile_ub",
                                                    scope=tik.scope_ubuf)
        self.updates_tile_ub = self.tik_instance.Tensor(self.var_dtype, (self.var_data_each_block,),
                                                        name="updates_tile_ub",
                                                        scope=tik.scope_ubuf)
        self.zero_ub = self.tik_instance.Tensor(self.var_dtype, (self.var_data_each_block,),
                                                name="zero_ub",
                                                scope=tik.scope_ubuf)

        self.var_read_index = self.tik_instance.Scalar("int32", name="var_read_index")
        self.var_read_index.set_as(0)
        self.core_loop_index = self.tik_instance.Scalar("int32", name="core_loop_index")
        self.core_loop_index.set_as(0)
        self.var_value = self.tik_instance.Scalar(self.var_dtype, name="var_value")
        self.var_value.set_as(0)
        self.update_value = self.tik_instance.Scalar(self.var_dtype, name="update_value")
        self.update_value.set_as(0)
        self.var_burst_len = self.tik_instance.Scalar("int32", name="var_burst_len")
        self.var_burst_len.set_as(0)
        self.indices_burst_len = self.tik_instance.Scalar("int32", name="indices_burst_len")
        self.indices_burst_len.set_as(0)
        self.updates_burst_len = self.tik_instance.Scalar("int32", name="updates_burst_len")
        self.updates_burst_len.set_as(0)
        self.each_core_max_indice = self.tik_instance.Scalar("int32", name="each_core_max_indice")
        self.each_core_max_indice.set_as(0)
        self.mask = self.tik_instance.Scalar("int32", name="mask")
        self.mask.set_as(0)
        self.repeat = self.tik_instance.Scalar("int32", name="repeat")
        self.repeat.set_as(0)
        self.var_last_num_off = self.tik_instance.Scalar("int32", name="var_last_num_off")
        self.var_last_num_off.set_as(0)
        self.last_var_num = self.tik_instance.Scalar("int32", name="last_var_num")
        self.last_var_num.set_as(0)
        self.var_offset = self.tik_instance.Scalar("int32", name="var_offset")
        self.var_offset.set_as(0)

    def move_indices(self, indices_in_index, indice_num):
        """
        Move indices, choose branch

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(indice_num % self.indices_data_each_block == 0):
            self.indices_burst_len.set_as(indice_num // self.indices_data_each_block)
        with self.tik_instance.else_scope():
            self.indices_burst_len.set_as((indice_num // self.indices_data_each_block) + 1)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[indices_in_index], 0, 1, self.indices_burst_len, 0,
                                    0)

        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.traversing_32b_aligned_ub_store_all_var_and_update(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.traversing_32b_aligned_ub_store_all_var(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.traversing_32b_aligned_ub_store_all_update(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 4):
            self.circulate_indices_not_atomic(indices_in_index, indice_num, 4)
        with self.tik_instance.if_scope(self.tiling_mode == 5):
            self.traversing_less_than_one_block_single_core_var_and_update(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 6):
            self.traversing_less_than_one_block_single_core_var(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 7):
            self.traversing_less_than_one_block_single_core_update(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 8):
            self.traversing_less_than_one_block_ub_not_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 9):
            self.circulate_indices_not_atomic(indices_in_index, indice_num, 9)
        with self.tik_instance.if_scope(self.tiling_mode == 10):
            self.circulate_indices_not_atomic(indices_in_index, indice_num, 10)

    def traversing_32b_aligned_ub_store_all_var_and_update(self, indices_in_index, indice_num):
        """
        32b aligned ub can store all var and update

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        updates_burst_len = self.updates_num // self.var_data_each_block
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, updates_burst_len, 0, 0)

        with self.tik_instance.if_scope(self.core_loop_index < self.core_num - 1):
            self.each_core_max_indice.set_as((self.core_loop_index + 1) * self.indice_step)
            self.var_burst_len.set_as(self.var_each_core_burst_len)
        with self.tik_instance.else_scope():
            self.each_core_max_indice.set_as(self.max_indice)
            self.var_burst_len.set_as(self.var_last_core_burst_len)
        self.updates_the_var_mode6(indices_in_index, indice_num, self.each_core_max_indice, self.var_burst_len)

    def updates_the_var_mode6(self, indices_in_index, indice_num, max_indice, burst_len):
        """
        32b aligned ub can store all var and update

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        max_indice: int32
            max index
        burst_len: int32
            move data block
        repeat: int32
            repeat times of vector instruct
        mask: int32
            the mask of vector instruct
        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.var_ub, self.var_gm[self.core_loop_index * self.var_each_core_data], 0, 1,
                                    burst_len, 0, 0)
        self.last_var_num.set_as(self.update_data_num)
        self.var_offset.set_as(0)

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope(max_indice > self.var_read_index):
                    with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat):
                        with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat * 255):
                            self.tik_instance.vec_sub(
                                self.data_num_one_repeat,
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num],
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num],
                                self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num], 255, 8,
                                8, 8)
                            self.var_offset.set_as(255 * self.data_num_one_repeat)
                            self.last_var_num.set_as(self.update_data_num - 255 * self.data_num_one_repeat)
                        self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
                        self.mask.set_as(self.last_var_num % self.data_num_one_repeat)
                        self.tik_instance.vec_sub(
                            self.data_num_one_repeat,
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num + self.var_offset],
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num + self.var_offset],
                            self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num +
                                            self.var_offset], self.repeat, 8, 8, 8)
                        with self.tik_instance.if_scope(self.mask != 0):
                            self.tik_instance.vec_sub(
                                self.mask, self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                                       self.update_data_num + self.data_num_one_repeat * self.repeat +
                                                       self.var_offset],
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num + self.data_num_one_repeat * self.repeat +
                                            self.var_offset],
                                self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num +
                                                self.data_num_one_repeat * self.repeat + self.var_offset], 1, 8, 8, 8)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_sub(
                            self.update_data_num,
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num],
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num],
                            self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num], 1, 8, 8, 8)
        self.tik_instance.data_move(self.var_gm[self.core_loop_index * self.var_each_core_data], self.var_ub, 0, 1,
                                    burst_len, 0, 0)

    def traversing_32b_aligned_ub_store_all_var(self, indices_in_index, indice_num):
        """
        32b aligned ub can store all var

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.core_loop_index < self.core_num - 1):
            self.each_core_max_indice.set_as((self.core_loop_index + 1) * self.indice_step)
            self.var_burst_len.set_as(self.var_each_core_burst_len)
        with self.tik_instance.else_scope():
            self.each_core_max_indice.set_as(self.max_indice)
            self.var_burst_len.set_as(self.var_last_core_burst_len)
        self.updates_the_var_mode7(indices_in_index, indice_num, self.each_core_max_indice, self.var_burst_len)

    def updates_the_var_mode7(self, indices_in_index, indice_num, max_indice, burst_len):
        """
        32b aligned ub can store all var

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        max_indice: int32
            max index
        burst_len: int32
            move data block
        repeat: int32
            repeat times of vector instruct
        mask: int32
            the mask of vector instruct
        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.var_ub, self.var_gm[self.core_loop_index * self.var_each_core_data], 0, 1,
                                    burst_len, 0, 0)
        self.last_var_num.set_as(self.update_data_num)
        self.var_offset.set_as(0)

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope(max_indice > self.var_read_index):
                    self.tik_instance.data_move(
                        self.updates_ub, self.updates_gm[(indices_in_index + indices_ub_index) * self.update_data_num],
                        0, 1, self.update_data_num // self.var_data_each_block, 0, 0)
                    with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat):
                        with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat * 255):
                            self.tik_instance.vec_sub(
                                self.data_num_one_repeat,
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num],
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num], self.updates_ub, 255, 8, 8, 8)
                            self.var_offset.set_as(255 * self.data_num_one_repeat)
                            self.last_var_num.set_as(self.update_data_num - 255 * self.data_num_one_repeat)
                        self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
                        self.mask.set_as(self.last_var_num % self.data_num_one_repeat)
                        self.tik_instance.vec_sub(
                            self.data_num_one_repeat,
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num + self.var_offset],
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num + self.var_offset], self.updates_ub[self.var_offset],
                            self.repeat, 8, 8, 8)
                        with self.tik_instance.if_scope(self.mask != 0):
                            self.tik_instance.vec_sub(
                                self.mask, self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                                       self.update_data_num + self.data_num_one_repeat * self.repeat +
                                                       self.var_offset],
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num + self.data_num_one_repeat * self.repeat +
                                            self.var_offset],
                                self.updates_ub[self.data_num_one_repeat * self.repeat + self.var_offset], 1, 8, 8, 8)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_sub(
                            self.update_data_num,
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num],
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num], self.updates_ub, 1, 8, 8, 8)
        self.tik_instance.data_move(self.var_gm[self.core_loop_index * self.var_each_core_data], self.var_ub, 0, 1,
                                    burst_len, 0, 0)

    def traversing_32b_aligned_ub_store_all_update(self, indices_in_index, indice_num):
        """
        32b aligned ub can store all update

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        update_burst_len = self.updates_num // self.var_data_each_block
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, update_burst_len, 0, 0)
        updates_burst_len = self.update_data_num // self.var_data_each_block
        self.last_var_num.set_as(self.update_data_num)
        self.var_offset.set_as(0)

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.tik_instance.data_move(self.var_ub, self.var_gm[self.var_read_index * self.update_data_num], 0,
                                                1, updates_burst_len, 0, 0)
                    with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat):
                        with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat * 255):
                            self.tik_instance.vec_sub(
                                self.data_num_one_repeat, self.var_ub, self.var_ub,
                                self.updates_ub[(indices_ub_index + indices_in_index) * self.update_data_num], 255, 8,
                                8, 8)
                            self.var_offset.set_as(255 * self.data_num_one_repeat)
                            self.last_var_num.set_as(self.update_data_num - 255 * self.data_num_one_repeat)
                        self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
                        self.mask.set_as(self.last_var_num % self.data_num_one_repeat)
                        self.tik_instance.vec_sub(
                            self.data_num_one_repeat, self.var_ub[self.var_offset], self.var_ub[self.var_offset],
                            self.updates_ub[(indices_ub_index + indices_in_index) * self.update_data_num +
                                            self.var_offset], self.repeat, 8, 8, 8)
                        with self.tik_instance.if_scope(self.mask != 0):
                            self.tik_instance.vec_sub(
                                self.mask, self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                self.updates_ub[(indices_ub_index + indices_in_index) * self.update_data_num +
                                                self.data_num_one_repeat * self.repeat + self.var_offset], 1, 8, 8, 8)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_sub(
                            self.update_data_num, self.var_ub, self.var_ub,
                            self.updates_ub[(indices_ub_index + indices_in_index) * self.update_data_num], 1, 8, 8, 8)
                    self.tik_instance.data_move(self.var_gm[self.var_read_index * self.update_data_num], self.var_ub, 0,
                                                1, updates_burst_len, 0, 0)

    def circulate_indices_not_atomic(self, indices_in_index, indice_num, mode):
        """
        Circulate the index in the indices

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.var_read_index.set_as(self.var_read_index * self.update_data_num)
                    update_in_index = (indices_in_index + indices_ub_index) * self.update_data_num
                    if mode == 4:
                        self.traversing_var_mode4(update_in_index)
                    if mode == 9:
                        self.traversing_var_mode9(update_in_index)
                    if mode == 10:
                        self.traversing_var_mode10(update_in_index)

    def traversing_var_mode4(self, update_in_index):
        """
        Traversing the index in the updates of branch 4

        Parameters
        ----------
        update_in_index: int32
            Updates index on GM
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.var_loop_num) as var_loop_index:
            self.move_var(self.var_read_index + var_loop_index * self.var_ub_num, self.var_ub_num,
                          update_in_index + var_loop_index * self.var_ub_num, 4)

        with self.tik_instance.if_scope(self.var_last_num > 0):
            self.move_var(self.var_read_index + self.var_loop_num * self.var_ub_num, self.var_last_num,
                          update_in_index + self.var_loop_num * self.var_ub_num, 4)

    def traversing_var_mode9(self, update_in_index):
        """
        Traversing the index in the updates of branch 9

        Parameters
        ----------
        update_in_index: int32
            Updates index on GM
        Returns
        -------
        None
        """
        var_last_num_once = self.var_last_num // self.var_data_each_block * self.var_data_each_block
        self.move_var(self.var_read_index, var_last_num_once, update_in_index, 9)
        var_last_num_twice = self.var_last_num - var_last_num_once
        self.move_var_tail(self.var_read_index + self.var_last_num - self.var_data_each_block, var_last_num_twice,
                           update_in_index + self.var_last_num - self.var_data_each_block, self.var_data_each_block, 9)
        self.tik_instance.data_move(self.var_gm[self.var_read_index], self.var_ub, 0, 1,
                                    var_last_num_once // self.var_data_each_block, 0, 0)

    def traversing_var_mode10(self, update_in_index):
        """
        Traversing the index in the updates of branch 10

        Parameters
        ----------
        update_in_index: int32
            Updates index on GM
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.var_loop_num) as var_loop_index:
            self.move_var(self.var_read_index + var_loop_index * self.var_ub_num, self.var_ub_num,
                          update_in_index + var_loop_index * self.var_ub_num, 10)

        self.var_last_num_off.set_as((self.var_last_num // self.var_data_each_block + 1) * self.var_data_each_block)
        self.move_var_tail(self.var_read_index + self.update_data_num - self.var_last_num_off, self.var_last_num,
                           update_in_index + self.update_data_num - self.var_last_num_off, self.var_last_num_off, 10)

    def move_var(self, var_in_index, var_num, update_in_index, mode):
        """
        Traversing the var

        Parameters
        ----------
        var_in_index: int32
            Var index on UB
        var_num: int32
            Var num move
        update_in_index: int32
            update index on UB
        Returns
        -------
        None
        """
        var_burst_len = var_num // self.var_data_each_block
        self.tik_instance.data_move(self.var_ub, self.var_gm[var_in_index], 0, 1, var_burst_len, 0, 0)
        self.tik_instance.data_move(self.updates_ub, self.updates_gm[update_in_index], 0, 1, var_burst_len, 0, 0)
        self.last_var_num.set_as(var_num)
        self.mask.set_as(var_num % self.data_num_one_repeat)
        self.var_offset.set_as(0)

        with self.tik_instance.if_scope(var_num >= self.data_num_one_repeat):
            with self.tik_instance.if_scope(var_num >= self.data_num_one_repeat * 255):
                self.tik_instance.vec_sub(self.data_num_one_repeat, self.var_ub, self.var_ub, self.updates_ub, 255, 8,
                                          8, 8)
                self.var_offset.set_as(255 * self.data_num_one_repeat)
                self.last_var_num.set_as(var_num - 255 * self.data_num_one_repeat)
            self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
            self.mask.set_as(self.last_var_num % self.data_num_one_repeat)
            self.tik_instance.vec_sub(self.data_num_one_repeat, self.var_ub[self.var_offset],
                                      self.var_ub[self.var_offset], self.updates_ub[self.var_offset], self.repeat, 8, 8,
                                      8)
            with self.tik_instance.if_scope(self.mask != 0):
                self.tik_instance.vec_sub(self.mask,
                                          self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                          self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                          self.updates_ub[self.data_num_one_repeat * self.repeat + self.var_offset], 1,
                                          8, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vec_sub(self.mask, self.var_ub, self.var_ub, self.updates_ub, 1, 8, 8, 8)

        if mode in (4, 10):
            self.tik_instance.data_move(self.var_gm[var_in_index], self.var_ub, 0, 1, var_burst_len, 0, 0)

    def move_var_tail(self, var_in_index, var_num, update_in_index, var_forward_offset, mode):
        """
        Traversing the var tail

        Parameters
        ----------
        var_in_index: int32
            Var index on UB
        var_num: int32
            Var num move
        update_in_index: int32
            update index on UB
        var_forward_offset: int32
            32 aligned of var_num
        Returns
        -------
        None
        """
        if mode == 9:
            self.tik_instance.data_move(self.var_tile_ub, self.var_gm[var_in_index], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.updates_tile_ub, self.updates_gm[update_in_index], 0, 1, 1, 0, 0)
            self.tik_instance.vec_sub(self.var_data_each_block, self.var_tile_ub, self.var_tile_ub,
                                      self.updates_tile_ub, 1, 8, 8, 8)
            self.tik_instance.data_move(self.var_gm[var_in_index], self.var_tile_ub, 0, 1, 1, 0, 0)

        if mode == 10:
            self.var_burst_len.set_as((var_num // self.var_data_each_block) + 1)
            self.tik_instance.data_move(self.var_ub, self.var_gm[var_in_index], 0, 1, self.var_burst_len, 0, 0)
            self.tik_instance.data_move(self.updates_ub, self.updates_gm[update_in_index], 0, 1, self.var_burst_len, 0,
                                        0)
            self.last_var_num.set_as(var_forward_offset)
            self.var_offset.set_as(0)

            with self.tik_instance.if_scope(var_forward_offset >= self.data_num_one_repeat):
                with self.tik_instance.if_scope(var_forward_offset >= self.data_num_one_repeat * 255):
                    self.tik_instance.vector_dup(var_forward_offset - var_num, self.updates_ub, 0, 1, 1, 8)
                    self.tik_instance.vec_sub(self.data_num_one_repeat, self.var_ub, self.var_ub, self.updates_ub, 255,
                                              8, 8, 8)
                    self.var_offset.set_as(255 * self.data_num_one_repeat)
                    self.last_var_num.set_as(var_forward_offset - 255 * self.data_num_one_repeat)
                self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
                self.mask.set_as(self.last_var_num - self.repeat * self.data_num_one_repeat)
                self.tik_instance.vector_dup(var_forward_offset - var_num, self.updates_ub, 0, 1, 1, 8)
                self.tik_instance.vec_sub(self.data_num_one_repeat, self.var_ub[self.var_offset],
                                          self.var_ub[self.var_offset], self.updates_ub[self.var_offset], self.repeat,
                                          8, 8, 8)
                with self.tik_instance.if_scope(self.mask != 0):
                    self.tik_instance.vec_sub(self.mask,
                                              self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                              self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                              self.updates_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                              1, 8, 8, 8)
            with self.tik_instance.else_scope():
                self.tik_instance.vector_dup(var_forward_offset - var_num, self.updates_ub, 0, 1, 1, 8)
                self.tik_instance.vec_sub(var_forward_offset, self.var_ub, self.var_ub, self.updates_ub, 1, 8, 8, 8)
            self.tik_instance.data_move(self.var_gm[var_in_index], self.var_ub, 0, 1, self.var_burst_len, 0, 0)

    def traversing_less_than_one_block_single_core_var_and_update(self, indices_in_index, indice_num):
        """
        updateDataNum is less than 1 block, ub can store all var and update

        Parameters
        ----------
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.var_num % self.var_data_each_block == 0):
            self.var_burst_len.set_as(self.var_num // self.var_data_each_block)
        with self.tik_instance.else_scope():
            self.var_burst_len.set_as((self.var_num // self.var_data_each_block) + 1)
        self.tik_instance.data_move(self.var_ub, self.var_gm, 0, 1, self.var_burst_len, 0, 0)

        with self.tik_instance.if_scope(self.updates_num % self.var_data_each_block == 0):
            self.updates_burst_len.set_as(self.updates_num // self.var_data_each_block)
        with self.tik_instance.else_scope():
            self.updates_burst_len.set_as((self.updates_num // self.var_data_each_block) + 1)
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, self.updates_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(self.var_read_index * self.update_data_num)
            with self.tik_instance.for_range(0, self.update_data_num) as ub_index:
                self.var_value.set_as(self.var_ub[self.var_read_index + ub_index])
                self.var_tile_ub[ub_index].set_as(self.var_value)
                self.update_value.set_as(self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num +
                                                         ub_index])
                self.updates_tile_ub[ub_index].set_as(self.update_value)
            self.tik_instance.vec_sub(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.updates_tile_ub, 1,
                                      8, 8, 8)
            with self.tik_instance.for_range(0, self.update_data_num) as ub_index:
                self.var_value.set_as(self.var_tile_ub[ub_index])
                self.var_ub[self.var_read_index + ub_index].set_as(self.var_value)

        self.tik_instance.data_move(self.var_gm, self.var_ub, 0, 1, self.var_burst_len, 0, 0)

    def traversing_less_than_one_block_single_core_var(self, indices_in_index, indice_num):
        """
        updateDataNum is less than 1 block, ub can store all var

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.var_num % self.var_data_each_block == 0):
            self.var_burst_len.set_as(self.var_num // self.var_data_each_block)
        with self.tik_instance.else_scope():
            self.var_burst_len.set_as((self.var_num // self.var_data_each_block) + 1)
        self.tik_instance.data_move(self.var_ub, self.var_gm, 0, 1, self.var_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(self.var_read_index * self.update_data_num)
            self.tik_instance.data_move(self.updates_tile_ub,
                                        self.updates_gm[(indices_in_index + indices_ub_index) * self.update_data_num],
                                        0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.update_data_num) as ub_index:
                self.var_value.set_as(self.var_ub[self.var_read_index + ub_index])
                self.var_tile_ub[ub_index].set_as(self.var_value)
            self.tik_instance.vec_sub(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.updates_tile_ub, 1,
                                      8, 8, 8)
            with self.tik_instance.for_range(0, self.update_data_num) as ub_index:
                self.var_value.set_as(self.var_tile_ub[ub_index])
                self.var_ub[self.var_read_index + ub_index].set_as(self.var_value)

        self.tik_instance.data_move(self.var_gm, self.var_ub, 0, 1, self.var_burst_len, 0, 0)

    def traversing_less_than_one_block_single_core_update(self, indices_in_index, indice_num):
        """
        updateDataNum is less than 1 block, ub can store all update

        Parameters
        ----------
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.updates_num % self.var_data_each_block == 0):
            self.updates_burst_len.set_as(self.updates_num // self.var_data_each_block)
        with self.tik_instance.else_scope():
            self.updates_burst_len.set_as((self.updates_num // self.var_data_each_block) + 1)
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, self.updates_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(self.var_read_index * self.update_data_num)
            self.tik_instance.data_move(self.var_tile_ub, self.var_gm[self.var_read_index], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.var_data_each_block) as ub_index:
                self.update_value.set_as(self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num +
                                                         ub_index])
                self.updates_tile_ub[ub_index].set_as(self.update_value)
            self.tik_instance.vec_sub(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.updates_tile_ub, 1,
                                      8, 8, 8)
            self.tik_instance.data_move(self.var_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)

    def traversing_less_than_one_block_ub_not_enough(self, indices_in_index, indice_num):
        """
        updateDataNum is less than 1 block, ub can't store all var and update

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(self.var_read_index * self.update_data_num)
            self.tik_instance.data_move(self.var_tile_ub, self.var_gm[self.var_read_index], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.updates_tile_ub,
                                        self.updates_gm[(indices_in_index + indices_ub_index) * self.update_data_num],
                                        0, 1, 1, 0, 0)
            self.tik_instance.vec_sub(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.updates_tile_ub, 1,
                                      8, 8, 8)
            self.tik_instance.data_move(self.var_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)

    def traversing_indices(self):
        """
        Traversing the index in the indices

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.indices_loop_num) as indices_loop_index:
            self.move_indices(indices_loop_index * self.indices_ub_num, self.indices_ub_num)

        with self.tik_instance.if_scope(self.indices_last_num > 0):
            self.move_indices(self.indices_loop_num * self.indices_ub_num, self.indices_last_num)

    def scatter_sub_compute_tiling(self):
        """
        Main process of scatter_sub

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 5, 0, 0)
        self.tiling_args()

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_index:
            with self.tik_instance.if_scope(core_index < self.core_num):
                self.init_ub_tensor()
                self.core_loop_index.set_as(core_index)
                self.traversing_indices()

    def scatter_sub_operator(self):
        """
        scatter_sub_operator

        Parameters
        ----------
        None

        Returns:
        ----------
        compile info
        """
        self.scatter_sub_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": self.ai_core_num,
                "var_size": self.var_dtype_bytes_size,
                "indices_size": self.indices_dtype_bytes_size
            })
        tbe_context.get_context().add_compile_info("is_tik", True)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var_gm, self.indices_gm, self.updates_gm),
                                   outputs=(self.out_gm),
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)


def scatter_sub_tik(var, indices, updates, var_out, kernel_name="ScatterSub"):
    obj = ScatterSub(var, indices, updates, var_out, False, kernel_name)
    return obj.scatter_sub_operator()


def scatter_sub_dsl(var, indices, updates, var_out, kernel_name="ScatterSub", impl_mode=None):
    """
    scatter_sub interface for dsl
    """
    check_list_var = ("float16", "float32", "int32")
    check_list_indices = ("int32", "int64")
    check_list_updates = ("float16", "float32", "int32")
    dtype_var = var.get("dtype").lower()
    dtype_indices = indices.get("dtype").lower()
    dtype_updates = updates.get("dtype").lower()
    para_check.check_dtype(dtype_var, check_list_var, param_name="var")
    para_check.check_dtype(dtype_indices, check_list_indices, param_name="indices")
    para_check.check_dtype(dtype_updates, check_list_updates, param_name="updates")

    op_type = "scatter"
    reduction = "sub"
    ins = classify([var, indices, updates], op_type)
    schedules, tensors = [], []
    for var_input, indices_input, updates_input in ins:
        with tbe.compute():
            var_shape, indices_shape, updates_shape = \
                shape_util.variable_shape([var_input, indices_input, updates_input], op_type)
            var_tensor = tvm.placeholder(var_shape, name="var", dtype=dtype_var)
            indices_tensor = tvm.placeholder(indices_shape, name="indices", dtype=dtype_indices)
            updates_tensor = tvm.placeholder(updates_shape, name="updates", dtype=dtype_updates)
            res = tbe.scatter(var_tensor, indices_tensor, updates_tensor, reduction,
                              support_out_of_bound_index=is_support_out_of_bound_index(impl_mode))
            tensors.append([var_tensor, indices_tensor, updates_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=unused-argument
@register_operator("ScatterSub")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scatter_sub(var, indices, updates, var_out, use_locking=False, kernel_name="scatter_sub", impl_mode=None):
    """
    scatter_sub interface

    Parameters
    ----------
    var_dict: input var shape, dtype and range
    indices_dict: input indices shape, dtype and range
    updates_dict: input updates shape, dtype and range
    var_out_dict: output shape, dtype and range
    kernel_name: kernel name of scatter_sub op
    impl_mode: implementation mode, such as high_precision, high_performance, support_out_of_bound_index.

    Returns
    -------
    compile info
    """
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        scatter_sub_tik(var, indices, updates, var_out, kernel_name)
    else:
        scatter_sub_dsl(var, indices, updates, var_out, kernel_name, impl_mode)
