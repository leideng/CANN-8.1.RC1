# Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
scatter_nd_update
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

# max int64 value
MAX_INT64_VALUE = 2 ** 64 - 1
# tiling param num
TILING_ARG_NUM = 25
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# 8 bit
EIGHT_BIT = 8
# bytes of one block
BLOCK_BYTES = 32
# vnchw unit
UNIT = 16
# hp core_num
HP_CORENUM = 32
# hp_ubsize
HP_UBSIZE = 262144


# 'pylint: disable=too-many-arguments,too-many-instance-attributes
# 'pylint: disable=invalid-name,attribute-defined-outside-init,unused-argument
class ScatterNdUpdate():
    """
    Function: use to store scatter_nd_update base parameters
    Modify: 2020-10-29
    """

    def __init__(self, var, indices, updates, var_out, use_locking,
                 kernel_name):
        """
        Init ScatterNdUpdate parameters
        Parameters
        _________
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
            cce kernel name, default value is "scatter_nd_update"

        Returns
        _______
        None.
        enable_const_fold
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.var_dtype = var.get("dtype").lower()
        self.var_dtype = "int8" if self.var_dtype == "bool" else self.var_dtype
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_dtype = updates.get("dtype").lower()
        self.updates_dtype = "int8" if self.updates_dtype == "bool" else self.updates_dtype
        self.out_dtype = var_out.get("dtype").lower()
        self.out_dtype = "int8" if self.out_dtype == "bool" else self.out_dtype
        self.check_input_params()

        self.support_hp = self.check_support_hp()

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(
            tbe_platform.UB_SIZE) - RESERVED_UB_SIZE
        self.var_dtype_bytes_size = get_bit_len(
            self.var_dtype) // EIGHT_BIT
        self.indices_dtype_bytes_size = get_bit_len(
            self.indices_dtype) // EIGHT_BIT
        self.var_data_each_block = BLOCK_BYTES // self.var_dtype_bytes_size
        self.indices_data_each_block = BLOCK_BYTES // self.indices_dtype_bytes_size

        self.updates_ub_num = self.ub_size_bytes // 2 // self.var_dtype_bytes_size
        self.indices_ub_num = self.ub_size_bytes // 2 // self.indices_dtype_bytes_size
        self.tiling_gm = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype,
                                               (MAX_INT64_VALUE,),
                                               name="var_gm",
                                               scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype,
                                                   (MAX_INT64_VALUE,),
                                                   name="indices_gm",
                                                   scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.var_dtype,
                                                   (MAX_INT64_VALUE,),
                                                   name="updates_gm",
                                                   scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype,
                                               (MAX_INT64_VALUE,),
                                               name="out_gm",
                                               scope=tik.scope_gm)

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
        self.var_offset_index_tiling = None
        self.tiling_data_ub = None
        self.vnchw_num = None
        self.indices_gm_offset_left = None
        self.vnchw_left = None
        self.vnchw_num_last = None
        self.indices_front_dim = None
        self.indices_last_dim = None
        self.core_num_var = self.tik_instance.Scalar(name="core_num_var", init_value=self.ai_core_num)

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    @staticmethod
    def check_support_hp():
        """
        hp only support for 910
        reason:
        1. some calculations are customized for 910
        2. there's a better way to do it for other soc
        """
        if tbe_platform.get_soc_spec(tbe_platform.CORE_NUM) != HP_CORENUM:
            return False
        if tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) != HP_UBSIZE:
            return False
        return True

    def check_input_params(self):
        """
        Check whether the input parameters is valid or not
        """
        indices_support_dtype_list = ("int32", "int64")
        var_support_dtype_list = ("float32", "float16", "int8", "uint8")
        para_check.check_dtype(self.indices_dtype,
                               indices_support_dtype_list,
                               param_name="indices")
        para_check.check_dtype(self.var_dtype,
                               var_support_dtype_list,
                               param_name="var")
        if self.var_dtype != self.updates_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(
                self.kernel_name, "updates", "var", self.updates_dtype,
                self.var_dtype)
        if self.var_dtype != self.out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(
                self.kernel_name, "out", "var", self.out_dtype, self.var_dtype)

    def calc_indices(self, indices_ub_index):
        """
        calculate indices on gm
        """
        self.var_read_index.set_as(0)
        with self.tik_instance.for_range(0, self.indices_last_dim) as i:
            tiling_offset = self.tik_instance.Scalar("int64",
                                                     name="tiling_offset")
            updates_sum_index = self.tik_instance.Scalar(
                "int32", name="updates_sum_index")
            indices_sum_index = self.tik_instance.Scalar(
                "int64", name="indices_sum_index")
            indices_sum_index.set_as(indices_ub_index * self.indices_last_dim +
                                     i)
            updates_sum_index.set_as(self.indices_ub[indices_sum_index])
            tiling_offset.set_as(self.var_offset_index_tiling[i])
            self.var_read_index.set_as(self.var_read_index +
                                       updates_sum_index * tiling_offset)

    def tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        __________
        tiling_ub: tensor, runtime params from scatter_nd_update tiling

        Returns
        _______
        None
        """
        self.tiling_mode = self.tik_instance.Scalar("int64",
                                                    name="tiling_mode")
        self.indice_step = self.tik_instance.Scalar("int64",
                                                    name="indice_step")
        self.core_num = self.tik_instance.Scalar("int64", name="core_num")
        self.update_data_num = self.tik_instance.Scalar("int64",
                                                        name="update_data_num")
        self.indices_loop_num = self.tik_instance.Scalar(
            "int64", name="indices_loop_num")
        self.indices_last_num = self.tik_instance.Scalar(
            "int64", name="indices_last_num")
        self.updates_num = self.tik_instance.Scalar("int64",
                                                    name="updates_num")
        self.updates_loop_num = self.tik_instance.Scalar(
            "int64", name="updates_loop_num")
        self.updates_last_num = self.tik_instance.Scalar(
            "int64", name="updates_last_num")
        self.indices_last_dim = self.tik_instance.Scalar(
            "int64", name="indices_last_dim")
        self.indices_front_dim = self.tik_instance.Scalar("int64",
                                                        name="indices_front_dim")
        self.var_offset_index_tiling = self.tik_instance.ScalarArray(
            dtype='int64', length=7, name='var_offset_index_tiling')

        self.tiling_mode.set_as(self.tiling_ub[7])
        self.indice_step.set_as(self.tiling_ub[8])
        self.core_num.set_as(self.tiling_ub[9])
        self.update_data_num.set_as(self.tiling_ub[10])
        self.indices_loop_num.set_as(self.tiling_ub[11])
        self.indices_last_num.set_as(self.tiling_ub[12])
        self.updates_num.set_as(self.tiling_ub[13])
        self.updates_loop_num.set_as(self.tiling_ub[14])
        self.updates_last_num.set_as(self.tiling_ub[15])
        self.indices_last_dim.set_as(self.tiling_ub[16])
        self.indices_front_dim.set_as(self.tiling_ub[17])
        for index in range(7):
            self.var_offset_index_tiling[index].set_as(self.tiling_ub[index])

    def init_ub_tensor(self):
        """
        Compute the ub size of tensors

        Parameters
        __________
        None

        Returns
        _______
        None
        """
        self.updates_ub = self.tik_instance.Tensor(self.var_dtype,
                                                   (self.updates_ub_num,),
                                                   name="updates_ub",
                                                   scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype,
                                                   (self.indices_ub_num,),
                                                   name="indices_ub",
                                                   scope=tik.scope_ubuf)
        self.var_tile_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="var_tile_ub",
            scope=tik.scope_ubuf)
        self.updates_tile_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="updates_tile_ub",
            scope=tik.scope_ubuf)
        self.var_vconv_ub = self.tik_instance.Tensor("float16", (32,),
                                                     name="var_vconv_ub",
                                                     scope=tik.scope_ubuf)
        self.updates_vconv_ub = self.tik_instance.Tensor(
            "float16", (32,), name="updates_vconv_ub", scope=tik.scope_ubuf)

        self.var_read_index = self.tik_instance.Scalar("int64",
                                                       name="var_read_index")
        self.var_read_index.set_as(0)
        self.core_loop_index = self.tik_instance.Scalar("int64",
                                                        name="core_loop_index")
        self.core_loop_index.set_as(0)
        self.update_value = self.tik_instance.Scalar(self.var_dtype,
                                                     name="update_value")
        self.update_value.set_as(0)
        self.indices_burst_len = self.tik_instance.Scalar(
            "int64", name="indices_burst_len")
        self.indices_burst_len.set_as(0)
        self.updates_burst_len = self.tik_instance.Scalar(
            "int64", name="updates_burst_len")
        self.updates_burst_len.set_as(0)

    def move_indices(self, indices_in_index, indice_num):
        """
        Move indices, choose branch
        Parameters
        __________
        indices_in_index: int64
            indices index on GM
        indice_num: int64
            the number of indexes in the indices on UB

        Returns
        _______
        None
        """
        self.indices_burst_len.set_as((indice_num + self.indices_data_each_block - 1) // self.indices_data_each_block)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            indices_gm_int8 = self.indices_gm.reinterpret_cast_to("int8")
            indices_ub_int8 = self.indices_ub.reinterpret_cast_to("int8")
            self.tik_instance.data_move_pad(indices_ub_int8,
                                            indices_gm_int8[indices_in_index * self.indices_dtype_bytes_size], 1,
                                            indice_num * self.indices_dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(self.indices_ub, self.indices_gm[indices_in_index], 0, 1,
                                        self.indices_burst_len, 0, 0)

        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.traversing_updates_32b_aligned_and_ub_enough(
                indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.circulate_indices(indices_in_index, indice_num, 2)
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.traversing_updates_single_core_and_ub_enough(
                indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 4):
            self.traversing_updates_single_core_and_ub_not_enough(
                indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 5):
            self.circulate_indices(indices_in_index, indice_num, 5)

    def circulate_indices(self, indices_in_index, indice_num, mode):
        """
        Circulate the index in the indices

        Parameters
        __________
        indices_in_index: int64
            Indices index on GM
        indice_num: int64
            the number of indexes in the indices on UB

        Returns
        _______
        None
        """

        with self.tik_instance.for_range(
                0, indice_num // self.indices_last_dim) as indices_ub_index:
            with self.tik_instance.if_scope(
                    indices_ub_index < indice_num // self.indices_last_dim):
                self.calc_indices(indices_ub_index)
                with self.tik_instance.if_scope(
                        self.core_loop_index *
                        self.indice_step <= self.var_read_index):
                    with self.tik_instance.if_scope(
                            (self.core_loop_index + 1) *
                            self.indice_step > self.var_read_index):
                        self.traversing_updates(indices_ub_index,
                                                indices_in_index, mode)

    def traversing_updates(self, indices_ub_index, indices_in_index, mode):
        """
        Traversing the index in the updates
        Parameters
        __________
        indices_ub_index: int64
            Indices index on UB
        indice_in_index: int64
            Indices index on GM

        Returns
        _______
        None
        """
        with self.tik_instance.if_scope(self.updates_loop_num > 0):
            with self.tik_instance.for_range(
                    0, self.updates_loop_num) as updates_loop_index:
                self.update_var((indices_in_index // self.indices_last_dim +
                                 indices_ub_index) * self.update_data_num +
                                updates_loop_index * self.updates_ub_num,
                                self.updates_ub_num,
                                updates_loop_index * self.updates_ub_num +
                                self.var_read_index, mode)

        with self.tik_instance.if_scope(self.updates_last_num > 0):
            self.update_var((indices_in_index // self.indices_last_dim +
                             indices_ub_index) * self.update_data_num +
                            self.updates_loop_num * self.updates_ub_num,
                            self.updates_last_num,
                            self.updates_loop_num * self.updates_ub_num +
                            self.var_read_index, mode)

    def update_var(self, updates_loop_index, updates_num, var_loop_index, mode):
        """
        Update the update fragment corresponding to the index
        Parameters
        __________
        updates_loop_index: int64
            Update index on GM
        updates_num: int64
            the number of indexes in the updates on UB
        var_loop_index: int64
            Var index on GM

        Returns
        _______
        None
        """
        if mode == 2:
            self.updates_burst_len.set_as(updates_num // self.var_data_each_block)
            self.tik_instance.data_move(self.updates_ub, self.updates_gm[updates_loop_index], 0, 1,
                                        self.updates_burst_len, 0, 0)
            self.tik_instance.data_move(self.out_gm[var_loop_index], self.updates_ub, 0, 1, self.updates_burst_len, 0,
                                        0)
        if mode == 5:
            self.updates_burst_len.set_as((updates_num + self.var_data_each_block - 1) // self.var_data_each_block)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_instance.data_move_pad(self.updates_ub, self.updates_gm[updates_loop_index], 1,
                                                updates_num * self.var_dtype_bytes_size, 0, 0)
                self.tik_instance.data_move_pad(self.out_gm[var_loop_index], self.updates_ub, 1,
                                                updates_num * self.var_dtype_bytes_size, 0, 0)
            else:
                self.tik_instance.data_move(self.updates_ub, self.updates_gm[updates_loop_index], 0, 1,
                                            self.updates_burst_len, 0, 0)
                with self.tik_instance.if_scope(updates_num % self.var_data_each_block == 0):
                    self.tik_instance.data_move(self.out_gm[var_loop_index], self.updates_ub, 0, 1,
                                                self.updates_burst_len, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.out_gm[var_loop_index], self.updates_ub, 0, 1,
                                                self.updates_burst_len - 1, 0, 0)
                    fallback_offset = updates_num - self.var_data_each_block
                    self.tik_instance.data_move(self.updates_tile_ub,
                                                self.updates_gm[updates_loop_index + fallback_offset], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(self.out_gm[var_loop_index + fallback_offset], self.updates_tile_ub, 0,
                                                1, 1, 0, 0)

    def traversing_updates_32b_aligned_and_ub_enough(self, indices_in_index, indice_num):
        """
        updateDataNum is 32B aligned ub can store all updatesNum

        Parameters
        __________
        indice_num: int64
            the number of indexes in the indices on UB
        Returns
        _______
        None
        """
        update_burst_len = self.updates_num // self.var_data_each_block
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, update_burst_len, 0, 0)
        updates_burst_len = self.update_data_num // self.var_data_each_block
        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.tik_instance.data_move(self.out_gm[self.var_read_index],
                                                self.updates_ub[(indices_in_index + indices_ub_index) *
                                                self.update_data_num], 0, 1, updates_burst_len, 0, 0)

    def traversing_updates_single_core_and_ub_enough(self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block, ub can store all updatesNum

        Parameters
        _________
        indice_num: int64
            the number of indexes in the indices on UB
        Returns
        _______
        None
        """
        self.updates_burst_len.set_as((self.updates_num + self.var_data_each_block - 1) // self.var_data_each_block)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(self.updates_ub, self.updates_gm, 1,
                                            self.updates_num * self.var_dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, self.updates_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_instance.data_move_pad(self.var_tile_ub, self.out_gm[self.var_read_index], 1,
                                                self.update_data_num * self.var_dtype_bytes_size, 0, 0)
            else:
                self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(
                    0, self.update_data_num) as updates_ub_index:
                self.update_value.set_as(
                    self.updates_ub[(indices_ub_index + indices_in_index //
                                     self.indices_last_dim) *
                                    self.update_data_num + updates_ub_index])
                self.updates_tile_ub[updates_ub_index].set_as(
                    self.update_value)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub,
                                           self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.updates_vconv_ub,
                                           self.updates_tile_ub, 1, 8, 4)
                self.tik_instance.vec_muls(self.update_data_num,
                                           self.var_vconv_ub,
                                           self.var_vconv_ub, 0, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num,
                                          self.var_vconv_ub, self.var_vconv_ub,
                                          self.updates_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub,
                                           self.var_vconv_ub, 1, 8, 4)
            else:
                self.tik_instance.vec_muls(self.update_data_num,
                                           self.var_tile_ub, self.var_tile_ub,
                                           0, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num,
                                          self.var_tile_ub, self.var_tile_ub,
                                          self.updates_tile_ub, 1, 8, 8, 8)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_instance.data_move_pad(self.out_gm[self.var_read_index], self.var_tile_ub, 1,
                                                self.update_data_num * self.var_dtype_bytes_size, 0, 0)
            else:
                self.tik_instance.data_move(self.out_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)

    def traversing_updates_single_core_and_ub_not_enough(
            self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block , ub can't store all updatesNum

        Parameters
        __________
        indices_in_index: int64
            indices index on GM
        indice_num: int64
            the number of indexes in the indices on UB
        Returns
        _______
        None
        """
        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            updates_offset = (indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_instance.data_move_pad(self.var_tile_ub, self.out_gm[self.var_read_index], 1,
                                                self.update_data_num * self.var_dtype_bytes_size, 0, 0)
                self.tik_instance.data_move_pad(self.updates_tile_ub, self.updates_gm[updates_offset], 1,
                                                self.update_data_num * self.var_dtype_bytes_size, 0, 0)
            else:
                self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index], 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.updates_tile_ub, self.updates_gm[updates_offset], 0, 1, 1, 0, 0)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub,
                                           self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.updates_vconv_ub,
                                           self.updates_tile_ub, 1, 8, 4)
                self.tik_instance.vec_muls(self.update_data_num,
                                           self.var_vconv_ub,
                                           self.var_vconv_ub, 0, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num,
                                          self.var_vconv_ub, self.var_vconv_ub,
                                          self.updates_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub,
                                           self.var_vconv_ub, 1, 8, 4)
            else:
                self.tik_instance.vec_muls(self.update_data_num,
                                           self.var_tile_ub, self.var_tile_ub,
                                           0, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num,
                                          self.var_tile_ub, self.var_tile_ub,
                                          self.updates_tile_ub, 1, 8, 8, 8)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_instance.data_move_pad(self.out_gm[self.var_read_index], self.var_tile_ub, 1,
                                                self.update_data_num * self.var_dtype_bytes_size, 0, 0)
            else:
                self.tik_instance.data_move(self.out_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)

    def traversing_indices(self):
        """
        Traversing the index in the indices

        Parameters
        __________
        None

        Returns
        _______
        None
        """
        max_ub_indices_num = self.indices_ub_num // self.indices_last_dim * self.indices_last_dim
        with self.tik_instance.if_scope(self.indices_loop_num > 0):
            with self.tik_instance.for_range(
                    0, self.indices_loop_num) as indices_loop_index:
                self.move_indices(indices_loop_index * max_ub_indices_num,
                                  max_ub_indices_num)
        with self.tik_instance.if_scope(self.indices_last_num > 0):
            self.move_indices(self.indices_loop_num * max_ub_indices_num,
                              self.indices_last_num)

    def scatter_nd_update_compute_tiling(self):
        """
        Main process of scatter_nd_update

        Parameters
        __________
        None

        Returns
        _______
        None
        """
        self.tiling_ub = self.tik_instance.Tensor("int64",
                                                  (TILING_ARG_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 7, 0, 0)
        self.tiling_args()
        with self.tik_instance.for_range(
                0, self.core_num_var, block_num=self.core_num_var) as core_index:
            with self.tik_instance.if_scope(self.tiling_mode == 6):
                if self.var_dtype == "float32" and self.indices_dtype == "int32" and self.support_hp:
                    self.init_scalar_tensor4hp()
                    self.core_loop_index.set_as(core_index)
                    self.traversing_indices4hp()
            with self.tik_instance.elif_scope(self.tiling_mode == 0):
                self.init_ub_tensor()
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(core_index < self.core_num):
                    self.init_ub_tensor()
                    self.core_loop_index.set_as(core_index)
                    self.traversing_indices()

    def scatter_nd_update_operator(self):
        """
        scatter_nd_update operation

        Parameters
        __________
        None

        Returns:
        _______
        compile info
        """
        self.scatter_nd_update_compute_tiling()
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": self.ai_core_num,
                "var_size": self.var_dtype_bytes_size,
                "indices_size": self.indices_dtype_bytes_size,
                "support_hp": 1 if self.support_hp else 0
            })
        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }
        tbe_context.get_context().add_compile_info("is_tik", True)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var_gm, self.indices_gm,
                                           self.updates_gm),
                                   outputs=(self.out_gm),
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

    def init_scalar_tensor4hp(self):
        '''
        only support 1980,float32,indices_last_dim < 8
        init ub tensor for high perm
        '''
        self.vnchw_num = self.tik_instance.Scalar("int32", name="vnchw_num")
        self.indices_ub_loop_num = self.tik_instance.Scalar(
            "int32", name="indices_ub_loop_num"
        )
        self.vnchw_left = self.tik_instance.Scalar("int32", name="vnchw_left")
        self.indices_gm_offset_left = self.tik_instance.Scalar(
            "int32", name="indices_gm_offset_left")
        self.vnchw_num_last = self.tik_instance.Scalar("int32",
                                                       name="vnchw_num_last")
        self.core_loop_index = self.tik_instance.Scalar("int64",
                                                        name="core_loop_index")

        self.core_loop_index.set_as(0)
        self.vnchw_num.set_as(self.tiling_ub[19])
        self.indices_ub_loop_num.set_as(self.tiling_ub[20])
        self.vnchw_left.set_as(self.tiling_ub[21])
        self.indices_gm_offset_left.set_as(self.tiling_ub[22])
        self.vnchw_num_last.set_as(self.tiling_ub[23])
        self.set_running_core_num(self.tiling_ub[24])

        tiling_data_conv_ub = self.tik_instance.Tensor(
            'int64', (4,), name='tiling_data_conv_ub', scope=tik.scope_ubuf)
        tiling_data_scalar = self.tik_instance.Scalar(
            self.indices_dtype, name='tiling_data_scalar')
        self.tiling_data_ub = self.tik_instance.Tensor(self.indices_dtype,
                                                       (64 * 7,),
                                                       name='tiling_data_ub',
                                                       scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.indices_last_dim) as i:
            tiling_data_conv_ub[0].set_as(self.var_offset_index_tiling[i])
            tiling_data_conv_ub = tiling_data_conv_ub.reinterpret_cast_to(
                "int32")
            tiling_data_scalar.set_as(tiling_data_conv_ub[0][0])
            tiling_data_conv_ub = tiling_data_conv_ub.reinterpret_cast_to(
                "int64")
            self.tik_instance.vector_dup(64, self.tiling_data_ub[64 * i],
                                         tiling_data_scalar, 1, 1, 8)

    def traversing_indices4hp(self):
        indices_ub_num = self.tik_instance.Scalar("int32",
                                                  name="indices_gm_offset")
        indices_gm_offset = self.tik_instance.Scalar("int32",
                                                     name="indices_gm_offset")

        indices_ub_num.set_as(self.vnchw_num *
                              (self.indices_last_dim * UNIT * UNIT))
        with self.tik_instance.for_range(
                0, self.indices_ub_loop_num) as indice_ub_loop_index:
            indices_gm_offset.set_as(indice_ub_loop_index * indices_ub_num)
            self.cal_indices_and_update(indices_gm_offset, self.vnchw_num)

        with self.tik_instance.if_scope(self.vnchw_num_last > 0):
            indices_gm_offset.set_as(self.indices_ub_loop_num * indices_ub_num)
            self.cal_indices_and_update(indices_gm_offset, self.vnchw_num_last)

        with self.tik_instance.if_scope(self.vnchw_left > 0):
            self.cal_indices_and_update(self.indices_gm_offset_left, 1)

    def cal_indices_and_update(self, indices_gm_offset, vnchw_num):
        indices_ub_num = self.tik_instance.Scalar("int64",
                                                  name="indices_ub_num")
        indices_ub_num.set_as(vnchw_num *
                              (self.indices_last_dim * UNIT * UNIT))

        indices_ub = self.tik_instance.Tensor(self.indices_dtype,
                                              (indices_ub_num,),
                                              name='indices_ub',
                                              scope=tik.scope_ubuf)
        indices_ub_vnchw = self.tik_instance.Tensor(self.indices_dtype,
                                                    (indices_ub_num,),
                                                    name='indices_ub_vnchw',
                                                    scope=tik.scope_ubuf)

        # data move out_gm to indices_ub
        self.data_move_indices(indices_ub, indices_gm_offset, vnchw_num,
                               self.indices_last_dim,
                               self.indices_data_each_block)

        # transpose indices (16, 16, vnchw_num, last_dim, 2) -> (16, last_dim, 16, vnchw_num, 2)
        self.transpose_indices(indices_ub, indices_ub_vnchw, vnchw_num,
                               self.indices_last_dim,
                               self.indices_data_each_block,
                               self.indices_dtype, indices_ub_num)

        # accumlate indices (16, last_dim, 16, vnchw_num) -> (16, vnchw_num, last_dim, 16)
        self.muls_add_mod(indices_ub_vnchw, indices_ub, vnchw_num,
                          self.indices_last_dim, self.indices_data_each_block,
                          indices_ub_num)

        # update by indice
        self.update_by_indice(indices_ub_vnchw, indices_ub, vnchw_num,
                              self.indices_last_dim,
                              self.indices_data_each_block,
                              indices_gm_offset // self.indices_last_dim)

    def data_move_indices(self, indices_ub, indices_gm_offset, vnchw_num,
                          last_dim, ele_num_one_block):
        # data_move indices from gm to ub
        self.tik_instance.data_move(
            indices_ub, self.indices_gm[indices_gm_offset], 0, 1,
            vnchw_num * UNIT * UNIT * last_dim // ele_num_one_block, 0, 0)

    def transpose_indices(self, ub_vnchw_src, ub_vnchw_dst, vnchw_num,
                          last_dim, ele_num_one_block, indices_dtype,
                          indices_ub_num):
        '''
        UNIT: 16
        ub_vnchw_src: vnchw src
        ub_vnchw_dst: vnchw dst
        vnchw_num: data_num = vnchw_num * UNIT * UNIT * last_dim
        last_dim: keep dim num
        '''
        # if vnchwconv can not support float32, need cast to float16
        CAST_NUM = 1
        if not tbe_platform.api_check_support('tik.vnchwconv', indices_dtype):
            # float32 cast_to float16
            CAST_NUM = 2
            # flat
            ub_vnchw_src = ub_vnchw_src.reinterpret_cast_to('float16').reshape(
                (indices_ub_num * CAST_NUM,))
            ub_vnchw_dst = ub_vnchw_dst.reinterpret_cast_to('float16').reshape(
                (indices_ub_num * CAST_NUM,))
            ele_num_one_block *= CAST_NUM

        # init vnchwconv list
        # (16, 16 * vnchw_num, last_dim, 2) -> (16 * vnchw_num, last_dim, 2, 16)
        ub_src_list = [
            ub_vnchw_src[UNIT * last_dim * vnchw_num * i * CAST_NUM]
            for i in range(UNIT)
        ]
        ub_dst_list = [ub_vnchw_dst[UNIT * i] for i in range(UNIT)]

        self.tik_instance.vnchwconv(True, True, ub_dst_list, ub_src_list,
                                    vnchw_num * last_dim * CAST_NUM,
                                    UNIT * UNIT // ele_num_one_block,
                                    UNIT // ele_num_one_block)

        # (16*vnchw_num, last_dim, 2, 16) -> (last_dim, 16*vnchw_num, 2, 16)
        with self.tik_instance.for_range(0, last_dim) as i:
            self.tik_instance.data_move(
                ub_vnchw_src[UNIT * UNIT * vnchw_num * i * CAST_NUM],
                ub_vnchw_dst[UNIT * i * CAST_NUM], 0, UNIT * vnchw_num,
                                                      UNIT * CAST_NUM // ele_num_one_block,
                                                      UNIT * CAST_NUM // ele_num_one_block * (last_dim - 1), 0)

        # init vnchwconv list
        # (last_dim, 16*n, 2, 16) -> (16, last_dim, 16*vnchw_num, 2)
        ub_src_list_after_data_move = [
            ub_vnchw_src[UNIT * i] for i in range(UNIT)
        ]
        ub_dst_list_after_data_move = [
            ub_vnchw_dst[UNIT * last_dim * vnchw_num * i * CAST_NUM]
            for i in range(UNIT)
        ]

        self.tik_instance.vnchwconv(True, True, ub_dst_list_after_data_move,
                                    ub_src_list_after_data_move,
                                    vnchw_num * last_dim * CAST_NUM,
                                    UNIT // ele_num_one_block,
                                    UNIT * UNIT // ele_num_one_block)

        # (16, last_dim, 16*vnchw_num, 2) -> (16, last_dim, 16*vnchw_num)
        if CAST_NUM == 2:
            ub_vnchw_src = ub_vnchw_src.reinterpret_cast_to("int32")
            ub_vnchw_dst = ub_vnchw_dst.reinterpret_cast_to("int32")

    def muls_add_mod(self, ub_src, ub_dst, vnchw_num, last_dim,
                     ele_num_one_block, indices_ub_num):
        # used to cal index % 32, alse index && 31
        mod_ub = self.tik_instance.Tensor(
            self.indices_dtype,
            (64,),  # 8 BLOCK
            name='mod_ub',
            scope=tik.scope_ubuf)
        # used to conv int32 to float32 to support soc version
        vconv_ub = self.tik_instance.Tensor('float32',
                                            (indices_ub_num // last_dim,),
                                            name='vconv_ub',
                                            scope=tik.scope_ubuf)

        # (a % 2^n) = (a && (2^n-1))
        self.tik_instance.vector_dup(64, mod_ub, 31, 1, 1, 8)

        # data move to accumlate data
        # (16, last_dim, 16*vnchw_num) -> (last_dim, 16, 16*vnchw_num)
        with self.tik_instance.for_range(0, last_dim) as i:
            self.tik_instance.data_move(
                ub_dst[UNIT * UNIT * vnchw_num * i],
                ub_src[UNIT * vnchw_num * i], 0, UNIT,
                vnchw_num * UNIT // ele_num_one_block,
                vnchw_num * UNIT * (last_dim - 1) // ele_num_one_block, 0)

        # 910 vmuls not support int32, use vmul instead
        with self.tik_instance.for_range(0, last_dim) as i:
            self.tik_instance.vec_mul(
                64, ub_src[UNIT * UNIT * vnchw_num * i],
                ub_dst[UNIT * UNIT * vnchw_num * i],
                self.tiling_data_ub[64 * i],
                UNIT * UNIT * vnchw_num // ele_num_one_block // 8, 8, 8, 0)

        with self.tik_instance.for_range(1, last_dim) as i:
            self.tik_instance.vec_add(
                64, ub_src, ub_src, ub_src[UNIT * UNIT * vnchw_num * i],
                UNIT * UNIT * vnchw_num // ele_num_one_block // 8, 8, 8, 8)

        # index // 32 % 32 to decrease conflicts between cores
        # not support int32 vdiv, must vconv to float32
        self.tik_instance.vconv(
            64, 'none', vconv_ub, ub_src,
            UNIT * UNIT * vnchw_num // ele_num_one_block // 8, 1, 1, 8, 8)

        # index // 32
        self.tik_instance.vmuls(
            64, vconv_ub, vconv_ub, 1 / 32,
                                    UNIT * UNIT * vnchw_num // ele_num_one_block // 8, 1, 1, 8, 8)

        self.tik_instance.vconv(
            64, 'floor', ub_dst, vconv_ub,
            UNIT * UNIT * vnchw_num // ele_num_one_block // 8, 1, 1, 8, 8)

        # cast to uint16 to use vec_and
        ub_dst = ub_dst.reinterpret_cast_to('uint16')
        ub_src = ub_src.reinterpret_cast_to('uint16')
        mod_ub = mod_ub.reinterpret_cast_to('uint16')

        self.tik_instance.vec_and(
            128, ub_dst, ub_dst, mod_ub,
            UNIT * UNIT * vnchw_num // ele_num_one_block // 8, 8, 8, 0)

        # cast back
        ub_dst = ub_dst.reinterpret_cast_to('int32')
        ub_src = ub_src.reinterpret_cast_to('int32')
        mod_ub = mod_ub.reinterpret_cast_to('int32')

    def update_by_indice(self, ub_index, ub_core_num, vnchw_num, last_dim,
                         ele_num_one_block, update_gm_offset):
        updates_ub = self.tik_instance.Tensor(self.var_dtype, (8,),
                                              name="updates_ub",
                                              scope=tik.scope_ubuf)
        zero_ub = self.tik_instance.Tensor(self.var_dtype, (8,),
                                           name='zero_ub',
                                           scope=tik.scope_ubuf)
        var_ub = self.tik_instance.Tensor(self.var_dtype, (8,),
                                          name='var_ub',
                                          scope=tik.scope_ubuf)
        self.tik_instance.vec_muls(8, zero_ub, zero_ub, 0, 1, 8, 8)
        self.tik_instance.set_atomic_add('float32')
        core_num_index = self.tik_instance.Scalar('int32',
                                                  name='core_num_index')
        updates_gm_index = self.tik_instance.Scalar('int32',
                                                    name='updates_gm_index')
        out_gm_index = self.tik_instance.Scalar('int32', name='out_gm_index')

        # loop all indice, (-var+update) atomic add var to update
        with self.tik_instance.for_range(0, vnchw_num * UNIT * UNIT) as indices_index:
            core_num_index.set_as(ub_core_num[indices_index])
            with self.tik_instance.if_scope(core_num_index == self.core_loop_index):
                updates_gm_index.set_as((update_gm_offset + indices_index) * self.update_data_num)
                out_gm_index.set_as(ub_index[indices_index])
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_instance.data_move_pad(updates_ub, self.updates_gm[updates_gm_index], 1,
                                                    self.update_data_num * self.var_dtype_bytes_size, 0, 0)
                    self.tik_instance.data_move_pad(var_ub, self.out_gm[out_gm_index], 1,
                                                    self.update_data_num * self.var_dtype_bytes_size, 0, 0)
                else:
                    self.tik_instance.data_move(updates_ub, self.updates_gm[updates_gm_index], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(var_ub, self.out_gm[out_gm_index], 0, 1, 1, 0, 0)
                self.tik_instance.vec_muls(self.update_data_num, var_ub, var_ub, -1, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num, zero_ub, updates_ub, var_ub, 1, 8, 8, 8)
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_instance.data_move_pad(self.out_gm[out_gm_index], zero_ub, 1,
                                                    self.update_data_num * self.var_dtype_bytes_size, 0, 0)
                else:
                    self.tik_instance.data_move(self.out_gm[out_gm_index], zero_ub, 0, 1, 1, 0, 0)

        self.tik_instance.set_atomic_add(0)


def scatter_nd_update_tik(var, indices, updates, var_out, use_locking, kernel_name="scatter_nd_update"):
    obj = ScatterNdUpdate(var, indices, updates, var_out, False, kernel_name)
    return obj.scatter_nd_update_operator()


def scatter_nd_update_dsl(var, indices, updates, var_out, use_locking,
                          kernel_name="scatter_nd_update", impl_mode=None):
    """
    scatter_nd_sub interface for dsl
    """
    check_list_var = ("float16", "float32", "int32", "bfloat16", "int64", "int8")
    check_list_indices = ("int32", "int64")
    check_list_updates = ("float16", "float32", "int32", "bfloat16", "int64", "int8")
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

    op_type = "scatter_nd"
    reduction = "update"
    ins = classify([var, indices, updates], op_type)
    schedules, tensors = [], []
    for var_input, indices_input, updates_input in ins:
        with tbe.compute():
            var_shape, indices_shape, updates_shape = \
                shape_util.variable_shape([var_input, indices_input, updates_input], "scatter")
            var_tensor = tvm.placeholder(var_shape, name="var", dtype=dtype_var)
            indices_tensor = tvm.placeholder(indices_shape, name="indices", dtype=dtype_indices)
            updates_tensor = tvm.placeholder(updates_shape, name="updates", dtype=dtype_updates)
            res = tbe.scatter_nd(var_tensor, indices_tensor, updates_tensor, reduction,
                                 support_out_of_bound_index=True)
            tensors.append([var_tensor, indices_tensor, updates_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


@register_operator("ScatterNdUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def scatter_nd_update(var,
                      indices,
                      updates,
                      var_out,
                      use_locking=False,
                      kernel_name="scatter_nd_update",
                      impl_mode="high_precision"):
    """
    scatter_nd_update interface
    Parameters
    __________
    var: input var shape, dtype and range
    indices: input indices shape, dtype and range
    updates: input updates shape, dtype and range
    var_out: output shape, dtype and range
    use_locking: bool
    kernel_name: kernel_name of scatter_nd_update op
    impl_mode: implementation mode, such as high_precision, high_performance, support_out_of_bound_index.
    Returns
    _______
    compile info
    """
    var_dtype = var.get("dtype").lower()
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") or var_dtype == "bool" or var_dtype == "int8":
        scatter_nd_update_tik(var, indices, updates, var_out, False, kernel_name)
    else:
        scatter_nd_update_dsl(var, indices, updates, var_out, False, kernel_name, impl_mode)
