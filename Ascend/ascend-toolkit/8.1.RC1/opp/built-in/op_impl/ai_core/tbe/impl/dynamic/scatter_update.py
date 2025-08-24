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
scatter_update
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len
from impl.util import util_tik_comm_func
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify

# max int64 value
MAX_INT64_VALUE = 2 ** 64 - 1
# tiling param num
TILING_ARG_NUM = 18
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# 8 bit
EIGHT_BIT = 8
# bytes of one block
BLOCK_BYTES = 32
# MAX BURST LEN
MAX_BURST_LEN = 65535


# 'pylint: disable=too-many-arguments,too-many-instance-attributes,unused-argument,invalid-name
class ScatterUpdate():
    """
       Function: use to store scatter_update base parameters
       Modify : 2020-10-29
    """

    # 'pylint: disable=too-many-statements
    def __init__(self, var, indices, updates, var_out, use_locking, kernel_name, opname="scatter_update"):
        """
        Init ScatterUpdate parameters

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
            cce kernel name, default value is "scatter_update".

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.opname = opname
        self.var_dtype = var.get("dtype").lower()
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_dtype = updates.get("dtype").lower()
        self.out_dtype = var_out.get("dtype").lower()

        self.check_input_params()

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE)
        self.var_dtype_bytes_size = get_bit_len(self.var_dtype) // EIGHT_BIT
        self.var_data_each_block = BLOCK_BYTES // self.var_dtype_bytes_size
        self.ub_tensor_size = (self.ub_size_bytes // self.var_dtype_bytes_size // 2 // self.var_data_each_block *
                               self.var_data_each_block)
        self.indices_dtype_bytes_size = get_bit_len(self.indices_dtype) // EIGHT_BIT
        self.indices_data_each_block = BLOCK_BYTES // self.indices_dtype_bytes_size

        self.updates_ub_num = self.ub_size_bytes // 2 // self.var_dtype_bytes_size
        self.indices_ub_num = self.ub_size_bytes // 2 // self.indices_dtype_bytes_size
        self.tiling_gm = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE,), name="var_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype, (MAX_INT64_VALUE,),
                                                   name="indices_gm", scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE,),
                                                   name="updates_gm",
                                                   scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE,), name="out_gm", scope=tik.scope_gm)

        self.updates_ub = None
        self.indices_ub = None
        self.var_tile_ub = None
        self.updates_tile_ub = None
        self.var_vconv_ub = None
        self.updates_vconv_ub = None
        self.tiling_ub = None
        self.var_read_index = None
        self.core_loop_index = None
        self.update_value = None
        self.indices_burst_len = None
        self.updates_burst_len = None
        self.tiling_mode = None
        self.indice_step = None
        self.core_num = None
        self.update_data_num = None
        self.indices_loop_num = None
        self.indices_last_num = None
        self.updates_num = None
        self.updates_loop_num = None
        self.updates_last_num = None
        self.each_core_compute_num = None
        self.each_core_loop_num = None
        self.each_core_loop_compute_num = None
        self.each_core_last_num = None
        self.last_core_compute_num = None
        self.last_core_loop_num = None
        self.last_core_loop_compute_num = None
        self.last_core_last_num = None
        self.var_data_ub = None
        self.core_num_var = self.tik_instance.Scalar(name="core_num_var", init_value=self.ai_core_num)

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    def check_input_params(self):
        """
        Check whether the input parameters is valid or not
        """
        indices_support_dtype_list = ("int32", "int64")
        var_support_dtype_list = ("float32", "float16", "int8", "uint8")
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
        tiling_ub: tensor, runtime params from scatter_update tiling

        Returns
        -------
        None
        """
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
        self.indice_step = self.tik_instance.Scalar("int64", name="indice_step")
        self.core_num = self.tik_instance.Scalar("int64", name="core_num")
        self.update_data_num = self.tik_instance.Scalar("int64", name="update_data_num")
        self.indices_loop_num = self.tik_instance.Scalar("int64", name="indices_loop_num")
        self.indices_last_num = self.tik_instance.Scalar("int64", name="indices_last_num")
        self.updates_num = self.tik_instance.Scalar("int64", name="updates_num")
        self.updates_loop_num = self.tik_instance.Scalar("int64", name="updates_loop_num")
        self.updates_last_num = self.tik_instance.Scalar("int64", name="updates_last_num")

        self.tiling_mode.set_as(self.tiling_ub[0])
        self.indice_step.set_as(self.tiling_ub[1])
        self.core_num.set_as(self.tiling_ub[2])
        self.update_data_num.set_as(self.tiling_ub[3])
        self.indices_loop_num.set_as(self.tiling_ub[4])
        self.indices_last_num.set_as(self.tiling_ub[5])
        self.updates_num.set_as(self.tiling_ub[6])
        self.updates_loop_num.set_as(self.tiling_ub[7])
        self.updates_last_num.set_as(self.tiling_ub[8])
        if self.opname == "inplace_update":
            self.each_core_compute_num = self.tik_instance.Scalar("int64", name="each_core_compute_num")
            self.each_core_loop_num = self.tik_instance.Scalar("int64", name="each_core_loop_num")
            self.each_core_loop_compute_num = self.tik_instance.Scalar("int64", name="each_core_loop_compute_num")
            self.each_core_last_num = self.tik_instance.Scalar("int64", name="each_core_last_num")
            self.last_core_compute_num = self.tik_instance.Scalar("int64", name="last_core_compute_num")
            self.last_core_loop_num = self.tik_instance.Scalar("int64", name="last_core_loop_num")
            self.last_core_loop_compute_num = self.tik_instance.Scalar("int64", name="last_core_loop_compute_num")
            self.last_core_last_num = self.tik_instance.Scalar("int64", name="last_core_last_num")
            self.each_core_compute_num.set_as(self.tiling_ub[9])
            self.each_core_loop_num.set_as(self.tiling_ub[10])
            self.each_core_loop_compute_num.set_as(self.tiling_ub[11])
            self.each_core_last_num.set_as(self.tiling_ub[12])
            self.last_core_compute_num.set_as(self.tiling_ub[13])
            self.last_core_loop_num.set_as(self.tiling_ub[14])
            self.last_core_loop_compute_num.set_as(self.tiling_ub[15])
            self.last_core_last_num.set_as(self.tiling_ub[16])
            self.set_running_core_num(self.tiling_ub[17])
        else:
            self.set_running_core_num(self.tiling_ub[9])

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
        self.var_vconv_ub = self.tik_instance.Tensor("float16", (32,), name="var_vconv_ub", scope=tik.scope_ubuf)
        self.updates_vconv_ub = self.tik_instance.Tensor("float16", (32,),
                                                         name="updates_vconv_ub",
                                                         scope=tik.scope_ubuf)

        self.var_read_index = self.tik_instance.Scalar("int32", name="var_read_index")
        self.var_read_index.set_as(0)
        self.core_loop_index = self.tik_instance.Scalar("int32", name="core_loop_index")
        self.core_loop_index.set_as(0)
        self.update_value = self.tik_instance.Scalar(self.var_dtype, name="update_value")
        self.update_value.set_as(0)
        self.indices_burst_len = self.tik_instance.Scalar("int32", name="indices_burst_len")
        self.indices_burst_len.set_as(0)
        self.updates_burst_len = self.tik_instance.Scalar("int32", name="updates_burst_len")
        self.updates_burst_len.set_as(0)

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
            self.traversing_updates_32b_aligned_and_ub_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.circulate_indices(indices_in_index, indice_num, 2)
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.traversing_updates_single_core_and_ub_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 4):
            self.traversing_updates_single_core_and_ub_not_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 5):
            self.circulate_indices(indices_in_index, indice_num, 5)

    def circulate_indices(self, indices_in_index, indice_num, mode):
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
                    self.traversing_updates(indices_ub_index, indices_in_index, mode)

    def traversing_updates(self, indices_ub_index, indices_in_index, mode):
        """
        Traversing the index in the updates

        Parameters
        ----------
        indices_ub_index: int32
            Indices index on UB
        indices_in_index: int32
            Indices index on GM
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.updates_loop_num > 0):
            with self.tik_instance.for_range(0, self.updates_loop_num) as updates_loop_index:
                self.update_var((indices_in_index + indices_ub_index) * self.update_data_num +
                                updates_loop_index * self.updates_ub_num, self.updates_ub_num,
                                self.var_read_index * self.update_data_num + updates_loop_index * self.updates_ub_num,
                                mode)

        with self.tik_instance.if_scope(self.updates_last_num > 0):
            self.update_var((indices_in_index + indices_ub_index) * self.update_data_num +
                            self.updates_loop_num * self.updates_ub_num, self.updates_last_num,
                            self.var_read_index * self.update_data_num + self.updates_loop_num * self.updates_ub_num,
                            mode)

    def update_var(self, updates_loop_index, update_num, var_loop_index, mode):
        """
        Update the update fragment corresponding to the index

        Parameters
        ----------
        updates_loop_index: int32
            Updates index on GM
        update_num: int32
            the number of indexes in the updates on UB
        var_loop_index: int32
            Var index on GM
        Returns
        -------
        None
        """
        if mode == 2:
            self.updates_burst_len.set_as(update_num // self.var_data_each_block)
            self.tik_instance.data_move(self.updates_ub, self.updates_gm[updates_loop_index], 0, 1,
                                        self.updates_burst_len, 0, 0)
            self.tik_instance.data_move(self.out_gm[var_loop_index], self.updates_ub, 0, 1, self.updates_burst_len, 0,
                                        0)

        if mode == 5:
            with self.tik_instance.if_scope(update_num % self.var_data_each_block == 0):
                self.updates_burst_len.set_as(update_num // self.var_data_each_block)
            with self.tik_instance.else_scope():
                self.updates_burst_len.set_as(update_num // self.var_data_each_block + 1)
            self.tik_instance.data_move(self.updates_ub, self.updates_gm[updates_loop_index], 0, 1,
                                        self.updates_burst_len, 0, 0)

            with self.tik_instance.if_scope(update_num % self.var_data_each_block == 0):
                self.tik_instance.data_move(self.out_gm[var_loop_index], self.updates_ub, 0, 1, self.updates_burst_len,
                                            0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.out_gm[var_loop_index], self.updates_ub, 0, 1,
                                            self.updates_burst_len - 1, 0, 0)
                self.tik_instance.data_move(self.updates_tile_ub,
                                            self.updates_gm[updates_loop_index + update_num - self.var_data_each_block],
                                            0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.out_gm[var_loop_index + update_num - self.var_data_each_block],
                                            self.updates_tile_ub, 0, 1, 1, 0, 0)

    def traversing_updates_32b_aligned_and_ub_enough(self, indices_in_index, indice_num):
        """
        updateDataNum is 32B aligned, ub can store all updatesNum

        Parameters
        ----------
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        update_burst_len = self.updates_num // self.var_data_each_block
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, update_burst_len, 0, 0)
        updates_burst_len = self.update_data_num // self.var_data_each_block

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.tik_instance.data_move(
                        self.out_gm[self.var_read_index * self.update_data_num],
                        self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num], 0, 1,
                        updates_burst_len, 0, 0)

    def traversing_updates_single_core_and_ub_enough(self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block, ub can store all updatesNum

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
            self.updates_burst_len.set_as(self.updates_num // self.var_data_each_block + 1)
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, self.updates_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index * self.update_data_num], 0, 1,
                                        1, 0, 0)
            with self.tik_instance.for_range(0, self.update_data_num) as updates_ub_index:
                self.update_value.set_as(self.updates_ub[(indices_ub_index + indices_in_index) * self.update_data_num +
                                                         updates_ub_index])
                self.updates_tile_ub[updates_ub_index].set_as(self.update_value)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub, self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.updates_vconv_ub, self.updates_tile_ub, 1, 8, 4)
                self.tik_instance.vec_muls(self.update_data_num, self.var_vconv_ub, self.var_vconv_ub, 0, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num, self.var_vconv_ub, self.var_vconv_ub,
                                          self.updates_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub, self.var_vconv_ub, 1, 8, 4)
            else:
                self.tik_instance.vec_muls(self.update_data_num, self.var_tile_ub, self.var_tile_ub, 0, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num, self.var_tile_ub, self.var_tile_ub,
                                          self.updates_tile_ub, 1, 8, 8, 8)
            self.tik_instance.data_move(self.out_gm[self.var_read_index * self.update_data_num], self.var_tile_ub, 0, 1,
                                        1, 0, 0)

    def traversing_updates_single_core_and_ub_not_enough(self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block, ub can't store all updatesNum

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
            self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index * self.update_data_num], 0, 1,
                                        1, 0, 0)
            self.tik_instance.data_move(self.updates_tile_ub,
                                        self.updates_gm[(indices_in_index + indices_ub_index) * self.update_data_num],
                                        0, 1, 1, 0, 0)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub, self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.updates_vconv_ub, self.updates_tile_ub, 1, 8, 4)
                self.tik_instance.vec_muls(self.update_data_num, self.var_vconv_ub, self.var_vconv_ub, 0, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num, self.var_vconv_ub, self.var_vconv_ub,
                                          self.updates_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub, self.var_vconv_ub, 1, 8, 4)
            else:
                self.tik_instance.vec_muls(self.update_data_num, self.var_tile_ub, self.var_tile_ub, 0, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num, self.var_tile_ub, self.var_tile_ub,
                                          self.updates_tile_ub, 1, 8, 8, 8)
            self.tik_instance.data_move(self.out_gm[self.var_read_index * self.update_data_num], self.var_tile_ub, 0, 1,
                                        1, 0, 0)

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
        with self.tik_instance.if_scope(self.indices_loop_num > 0):
            with self.tik_instance.for_range(0, self.indices_loop_num) as indices_loop_index:
                self.move_indices(indices_loop_index * self.indices_ub_num, self.indices_ub_num)

        with self.tik_instance.if_scope(self.indices_last_num > 0):
            self.move_indices(self.indices_loop_num * self.indices_ub_num, self.indices_last_num)

    def _copy_in_and_out(self, copy_len, copy_in_offset, copy_out_offset):
        """
            Function: move var form gm to ub, and move var form ub to output_gm
        """
        burst_len = copy_len // self.var_data_each_block
        nburst = util_tik_comm_func.ceil_div(burst_len, MAX_BURST_LEN)
        burst_len = burst_len // nburst
        self.var_data_ub = self.tik_instance.Tensor(self.var_dtype, (self.ub_tensor_size,),
                                                    name="var_data_ub", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.if_scope(copy_len >= self.var_data_each_block):
                self.tik_instance.data_move(self.var_data_ub,
                                            self.var_gm[copy_in_offset],
                                            0, nburst, burst_len, 0, 0)
                self.tik_instance.data_move(self.out_gm[copy_out_offset],
                                            self.var_data_ub,
                                            0, nburst, burst_len, 0, 0)
            moved_num = nburst * burst_len * self.var_data_each_block
            with self.tik_instance.if_scope(copy_len - moved_num != 0):
                back_offset = self.var_data_each_block - (copy_len - moved_num) % self.var_data_each_block
                copy_in_offset = copy_in_offset + moved_num - back_offset
                copy_out_offset = copy_in_offset
                burst_len = (copy_len - moved_num + back_offset) / self.var_data_each_block
                self.tik_instance.data_move(self.var_data_ub,
                                            self.var_gm[copy_in_offset],
                                            0, 1, burst_len, 0, 0)
                self.tik_instance.data_move(self.out_gm[copy_out_offset],
                                            self.var_data_ub,
                                            0, 1, burst_len, 0, 0)

    def copy_xgm_to_ub(self, core_index):
        """
            Function: compute offset
        """
        x_offset = core_index * self.each_core_compute_num
        with self.tik_instance.if_scope(core_index < self.core_num - 1):
            with self.tik_instance.for_range(0, self.each_core_loop_num) as loop_index:
                gm_in_offset = x_offset + loop_index * self.each_core_loop_compute_num
                gm_out_offset = gm_in_offset
                self._copy_in_and_out(self.each_core_loop_compute_num, gm_in_offset, gm_out_offset)
            with self.tik_instance.if_scope(self.each_core_last_num != 0):
                gm_in_offset = x_offset + self.each_core_loop_num * self.each_core_loop_compute_num
                gm_out_offset = gm_in_offset
                self._copy_in_and_out(self.each_core_last_num, gm_in_offset, gm_out_offset)

        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.last_core_loop_num) as loop_index:
                gm_in_offset = x_offset + loop_index * self.last_core_loop_compute_num
                gm_out_offset = gm_in_offset
                self._copy_in_and_out(self.last_core_loop_compute_num, gm_in_offset, gm_out_offset)
            with self.tik_instance.if_scope(self.last_core_last_num != 0):
                gm_in_offset = x_offset + self.last_core_loop_num * self.last_core_loop_compute_num
                gm_out_offset = gm_in_offset
                self._copy_in_and_out(self.last_core_last_num, gm_in_offset, gm_out_offset)

    def scatter_update_compute_tiling(self):
        """
        Main process of scatter_update

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
                if self.opname == "inplace_update":
                    self.copy_xgm_to_ub(core_index)
                self.init_ub_tensor()
                self.core_loop_index.set_as(core_index)
                self.traversing_indices()

    def scatter_update_operator(self):
        """
        scatter_update operation

        Parameters
        ----------
        None

        Returns:
        ----------
        compile info
        """
        self.scatter_update_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
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


def scatter_update_tik(var, indices, updates, var_out, kernel_name="ScatterUpdate"):
    obj = ScatterUpdate(var, indices, updates, var_out, False, kernel_name)
    return obj.scatter_update_operator()


def scatter_update_dsl(var, indices, updates, var_out, kernel_name="ScatterUpdate", impl_mode=None):
    """
    scatter_update interface for dsl
    """
    check_list_var = ("float16", "float32", "int32", "bfloat16", "int64")
    check_list_indices = ("int32", "int64")
    check_list_updates = ("float16", "float32", "int32", "bfloat16", "int64")
    dtype_var = var.get("dtype").lower()
    dtype_indices = indices.get("dtype").lower()
    dtype_updates = updates.get("dtype").lower()
    para_check.check_dtype(dtype_var, check_list_var, param_name="var")
    para_check.check_dtype(dtype_indices, check_list_indices, param_name="indices")
    para_check.check_dtype(dtype_updates, check_list_updates, param_name="updates")
    tbe_context.get_context().add_compile_info("impl_mode", impl_mode)
    if dtype_var not in ("float16", "float32", "bfloat16") and impl_mode == "high_performance":
        impl_mode = "high_precision"
        tbe_context.get_context().add_compile_info("impl_mode", impl_mode)

    op_type = "scatter"
    reduction = "update"
    ins = classify([var, indices, updates], op_type)
    schedules, tensors = [], []
    for var_input, indices_input, updates_input in ins:
        with tbe.compute():
            var_shape, indices_shape, updates_shape = \
                shape_util.variable_shape([var_input, indices_input, updates_input], op_type)
            var_tensor = tvm.placeholder(var_shape, name="var", dtype=dtype_var)
            indices_tensor = tvm.placeholder(indices_shape, name="indices", dtype=dtype_indices)
            updates_tensor = tvm.placeholder(updates_shape, name="updates", dtype=dtype_updates)
            res = tbe.scatter(var_tensor, indices_tensor, updates_tensor, reduction, support_out_of_bound_index=True)
            tensors.append([var_tensor, indices_tensor, updates_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=unused-argument
@register_operator("ScatterUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scatter_update(var, indices, updates, var_out, use_locking=False, kernel_name="scatter_update",
                   impl_mode="high_precision"):
    """
    scatter_update interface

    Parameters
    ----------
    var: input var shape, dtype and range
    indices: input indices shape, dtype and range
    updates: input updates shape, dtype and range
    var_out: output shape, dtype and range
    use_locking: bool
    kernel_name: kernel name of scatter_add op
    impl_mode: implementation mode, such as high_precision, high_performance, support_out_of_bound_index.

    Returns
    -------
    compile info
    """
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        scatter_update_tik(var, indices, updates, var_out, kernel_name)
    else:
        scatter_update_dsl(var, indices, updates, var_out, kernel_name, impl_mode)
