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
scatter_nd_sub
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


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max int64 value
    MAX_INT64_VALUE = 2 ** 64 - 1
    # tiling param num
    TILING_ARG_NUM = 19
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # 8 bit
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = 32


# 'pylint: disable=too-many-arguments,too-many-instance-attributes
# 'pylint: disable=invalid-name,attribute-defined-outside-init,unused-argument
class ScatterNdSub():
    """
    Function: use to store scatter_nd_sub base parameters
    Modify: 2020-10-29
    """

    def __init__(self, var, indices, subs, var_out, use_locking, kernel_name):
        """
        Init ScatterNdSub parameters
        Paramters
        _________
        var: dict
            the dict of input tensor.
        indices: dict
            the dict of input tensor.
        sub: dict
            the dict of input tensor.
        var_out: dict
            the dict of output tensor.
        use_locking: bool
            not used in this compute, default value is "False".
        kernel_name: str
            cce kernel name, default value is "scatter_nd_sub"
        Returns
        _______
        None.
        enable_const_fold
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.var_dtype = var.get("dtype").lower()
        self.indice_dtype = indices.get("dtype").lower()
        self.subs_dtype = subs.get("dtype").lower()
        self.out_dtype = var_out.get("dtype").lower()

        self.check_input_params()

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_SIZE
        self.var_dtype_bytes_size = get_bit_len(self.var_dtype) // Constant.EIGHT_BIT
        self.indices_dtype_bytes_size = get_bit_len(self.indice_dtype) // Constant.EIGHT_BIT
        self.var_data_each_block = Constant.BLOCK_BYTES // self.var_dtype_bytes_size
        self.indices_data_each_block = Constant.BLOCK_BYTES // self.indices_dtype_bytes_size

        self.subs_ub_num = self.ub_size_bytes // 96 * 32 // self.var_dtype_bytes_size
        self.indices_ub_num = self.ub_size_bytes // 96 * 32 // self.indices_dtype_bytes_size
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (Constant.MAX_INT64_VALUE,),
                                               name="var_gm",
                                               scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indice_dtype, (Constant.MAX_INT64_VALUE,),
                                                   name="indices_gm",
                                                   scope=tik.scope_gm)
        self.subs_gm = self.tik_instance.Tensor(self.var_dtype, (Constant.MAX_INT64_VALUE,),
                                                name="subs_gm",
                                                scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype, (Constant.MAX_INT64_VALUE,),
                                               name="out_gm",
                                               scope=tik.scope_gm)

        self.subs_ub = None
        self.var_ub = None
        self.indices_ub = None
        self.var_tile_ub = None
        self.subs_tile_ub = None
        self.var_vconv_ub = None
        self.subs_vconv_ub = None
        self.tiling_ub = None

        self.var_read_index = None
        self.core_loop_index = None

        self.update_value = None
        self.indices_burst_len = None
        self.subs_burst_len = None
        self.tiling_mode = None
        self.indice_step = None
        self.core_num = None
        self.update_data_num = None

        self.indices_loop_num = None
        self.indices_last_num = None

        self.subs_num = None
        self.subs_loop_num = None
        self.subs_last_num = None
        self.indices_align_num = None
        self.indices_front_dim = None
        self.indices_last_dim = None
        self.core_num_var = self.tik_instance.Scalar(name="core_num_var", init_value=self.ai_core_num)

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    def check_input_params(self):
        """
        Check whether the input parameters is valid or not
        """
        indices_support_dtype_list = ("int32", "int64")
        var_support_dtype_list = ("float32", "float16", "int32")
        para_check.check_dtype(self.indice_dtype, indices_support_dtype_list, param_name="indices")
        para_check.check_dtype(self.var_dtype, var_support_dtype_list, param_name="var")
        if self.var_dtype != self.subs_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "sub", "var", self.subs_dtype,
                                                                  self.var_dtype)
        if self.var_dtype != self.out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "out", "var", self.out_dtype,
                                                                  self.var_dtype)

    def calc_indices(self, indices_ub_index):
        """
        calculate indices on gm
        """
        self.var_read_index.set_as(0)
        with self.tik_instance.for_range(0, self.indices_last_dim) as i:
            tiling_offset = self.tik_instance.Scalar("int64", name="tmp")
            subs_sum_index = self.tik_instance.Scalar("int32", name="subs_sum_index")
            indices_sum_index = self.tik_instance.Scalar("int64", name="indices_sum_index")
            indices_sum_index.set_as(indices_ub_index * self.indices_last_dim + i)
            subs_sum_index.set_as(self.indices_ub[indices_sum_index])
            tiling_offset.set_as(self.tiling_ub[i])
            self.var_read_index.set_as(self.var_read_index + subs_sum_index * tiling_offset)

    def tiling_args(self):
        """
        get runtime params from tiling
        Parameters
        __________
        tiling_ub: tensor, runtime params from scatter_nd_sub tiling
        Returns
        _______
        None
        """
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
        self.indice_step = self.tik_instance.Scalar("int64", name="indice_step")
        self.core_num = self.tik_instance.Scalar("int64", name="core_num")
        self.update_data_num = self.tik_instance.Scalar("int64", name="update_data_num")
        self.indices_loop_num = self.tik_instance.Scalar("int64", name="indices_loop_num")
        self.indices_last_num = self.tik_instance.Scalar("int64", name="indices_last_num")
        self.subs_num = self.tik_instance.Scalar("int64", name="subs_num")
        self.subs_loop_num = self.tik_instance.Scalar("int64", name="subs_loop_num")
        self.subs_last_num = self.tik_instance.Scalar("int64", name="subs_last_num")
        self.indices_last_dim = self.tik_instance.Scalar("int64", name="indices_last_dim")
        self.indices_front_dim = self.tik_instance.Scalar("int64", name="indices_front_dim")
        self.indices_align_num = self.tik_instance.Scalar("int64", name="indices_align_num")

        self.tiling_mode.set_as(self.tiling_ub[7])
        self.indice_step.set_as(self.tiling_ub[8])
        self.core_num.set_as(self.tiling_ub[9])
        self.update_data_num.set_as(self.tiling_ub[10])
        self.indices_loop_num.set_as(self.tiling_ub[11])
        self.indices_last_num.set_as(self.tiling_ub[12])
        self.subs_num.set_as(self.tiling_ub[13])
        self.subs_loop_num.set_as(self.tiling_ub[14])
        self.subs_last_num.set_as(self.tiling_ub[15])
        self.indices_last_dim.set_as(self.tiling_ub[16])
        self.indices_front_dim.set_as(self.tiling_ub[17])
        self.indices_align_num.set_as(self.indices_ub_num // self.indices_last_dim * self.indices_last_dim)
        self.set_running_core_num(self.tiling_ub[18])

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
        self.subs_ub = self.tik_instance.Tensor(self.var_dtype, (self.subs_ub_num,),
                                                name="subs_ub",
                                                scope=tik.scope_ubuf)
        self.var_ub = self.tik_instance.Tensor(self.var_dtype, (self.subs_ub_num,), name="var_ub", scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(self.indice_dtype, (self.indices_ub_num,),
                                                   name="indices_ub",
                                                   scope=tik.scope_ubuf)
        self.var_tile_ub = self.tik_instance.Tensor(self.var_dtype, (self.var_data_each_block,),
                                                    name="var_tile_ub",
                                                    scope=tik.scope_ubuf)
        self.subs_tile_ub = self.tik_instance.Tensor(self.var_dtype, (self.var_data_each_block,),
                                                     name="subs_tile_ub",
                                                     scope=tik.scope_ubuf)
        self.var_vconv_ub = self.tik_instance.Tensor("float16", (32,), name="var_vconv_ub", scope=tik.scope_ubuf)
        self.subs_vconv_ub = self.tik_instance.Tensor("float16", (32,), name="subs_vconv_ub", scope=tik.scope_ubuf)

        self.var_read_index = self.tik_instance.Scalar("int64", name="var_read_index")
        self.var_read_index.set_as(0)
        self.core_loop_index = self.tik_instance.Scalar("int64", name="core_loop_index")
        self.core_loop_index.set_as(0)
        self.update_value = self.tik_instance.Scalar(self.var_dtype, name="update_value")
        self.update_value.set_as(0)
        self.indices_burst_len = self.tik_instance.Scalar("int64", name="indices_burst_len")
        self.indices_burst_len.set_as(0)
        self.subs_burst_len = self.tik_instance.Scalar("int64", name="subs_burst_len")
        self.subs_burst_len.set_as(0)

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
        with self.tik_instance.if_scope(indice_num % self.indices_data_each_block == 0):
            self.indices_burst_len.set_as(indice_num // self.indices_data_each_block)
        with self.tik_instance.else_scope():
            self.indices_burst_len.set_as((indice_num // self.indices_data_each_block) + 1)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[indices_in_index], 0, 1, self.indices_burst_len, 0,
                                    0)

        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.traversing_subs_32b_aligned_and_ub_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.circulate_indices(indices_in_index, indice_num, 2)
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.traversing_subs_single_core_and_ub_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 4):
            self.traversing_subs_single_core_and_ub_not_enough(indices_in_index, indice_num)
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
        with self.tik_instance.for_range(0, self.indices_front_dim) as indices_ub_index:
            with self.tik_instance.if_scope(indices_ub_index < indice_num // self.indices_last_dim):
                self.calc_indices(indices_ub_index)
                with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                    with self.tik_instance.if_scope(
                            (self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                        self.traversing_subs(indices_ub_index, indices_in_index, mode)

    def traversing_subs(self, indices_ub_index, indices_in_index, mode):
        """
        Traversing the index in the subs
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
        with self.tik_instance.if_scope(self.subs_loop_num > 0):
            with self.tik_instance.for_range(0, self.subs_loop_num) as subs_loop_index:
                self.update_var((indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num +
                                subs_loop_index * self.subs_ub_num, self.subs_ub_num,
                                subs_loop_index * self.subs_ub_num + self.var_read_index, mode)
        with self.tik_instance.if_scope(self.subs_last_num > 0):
            self.update_var((indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num +
                            self.subs_loop_num * self.subs_ub_num, self.subs_last_num,
                            self.subs_loop_num * self.subs_ub_num + self.var_read_index, mode)

    def vsub(self, update_num):
        """
        vadd
        """
        mask = 256 // self.var_dtype_bytes_size
        loop = (update_num + mask - 1) // mask // 255
        last = (update_num + mask - 1) // mask % 255
        lastindex = loop * 255 + last - 1
        lastmaskscalar = self.tik_instance.Scalar("int64", name="lastmask")
        lastmask = update_num % mask
        lastmaskscalar.set_as(lastmask)
        with self.tik_instance.if_scope(lastmaskscalar == 0):
            lastmaskscalar.set_as(mask)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as i:
                self.tik_instance.vec_sub(mask, self.subs_ub[i * 255 * mask], self.var_ub[i * 255 * mask],
                                          self.subs_ub[i * 255 * mask], 255, 8, 8, 8)
        with self.tik_instance.if_scope(last > 0):
            self.tik_instance.vec_sub(mask, self.subs_ub[loop * 255 * mask], self.var_ub[loop * 255 * mask],
                                      self.subs_ub[loop * 255 * mask], last - 1, 8, 8, 8)
            self.tik_instance.vec_sub(lastmaskscalar, self.subs_ub[lastindex * mask], self.var_ub[lastindex * mask],
                                      self.subs_ub[lastindex * mask], 1, 8, 8, 8)

    def vec_sub_with_subr(self, update_num, subs_subr, var_subr):
        """
        vec_add_with_addr
        """
        mask = 256 // self.var_dtype_bytes_size
        loop = (update_num + mask - 1) // mask // 255
        last = (update_num + mask - 1) // mask % 255
        lastindex = loop * 255 + last - 1
        lastmaskscalar = self.tik_instance.Scalar("int64", name="lastmask")
        lastmask = update_num % mask
        lastmaskscalar.set_as(lastmask)
        with self.tik_instance.if_scope(lastmaskscalar == 0):
            lastmaskscalar.set_as(mask)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as i:
                self.tik_instance.vec_sub(mask, self.subs_ub[i * 255 * mask + subs_subr],
                                          self.var_ub[i * 255 * mask + var_subr],
                                          self.subs_ub[i * 255 * mask + subs_subr], 255, 8, 8, 8)
        with self.tik_instance.if_scope(last > 0):
            self.tik_instance.vec_sub(mask, self.subs_ub[loop * 255 * mask + subs_subr],
                                      self.var_ub[loop * 255 * mask + var_subr],
                                      self.subs_ub[loop * 255 * mask + subs_subr], last - 1, 8, 8, 8)
            self.tik_instance.vec_sub(lastmaskscalar, self.subs_ub[lastindex * mask + subs_subr],
                                      self.var_ub[lastindex * mask + var_subr],
                                      self.subs_ub[lastindex * mask + subs_subr], 1, 8, 8, 8)

    def update_var(self, subs_loop_index, update_num, var_loop_index, mode):
        """
        Update the update fragment corresponding to the index
        Parameters
        __________
        subs_loop_index: int64
            Update index on GM
        update_num: int64
            the number of indexes in the subs on UB
        var_loop_index: int64
            Var index on GM

        Returns
        _______
        None
        """
        if mode == 2:
            self.subs_burst_len.set_as(update_num // self.var_data_each_block)
            self.tik_instance.data_move(self.subs_ub, self.subs_gm[subs_loop_index], 0, 1, self.subs_burst_len, 0, 0)
            self.tik_instance.data_move(self.var_ub, self.out_gm[var_loop_index], 0, 1, self.subs_burst_len, 0, 0)
            self.vsub(update_num)
            self.tik_instance.data_move(self.out_gm[var_loop_index], self.subs_ub, 0, 1, self.subs_burst_len, 0, 0)
        if mode == 5:
            with self.tik_instance.if_scope(update_num % self.var_data_each_block == 0):
                self.subs_burst_len.set_as(update_num // self.var_data_each_block)
            with self.tik_instance.else_scope():
                self.subs_burst_len.set_as(update_num // self.var_data_each_block + 1)
            self.tik_instance.data_move(self.subs_ub, self.subs_gm[subs_loop_index], 0, 1, self.subs_burst_len, 0, 0)
            self.tik_instance.data_move(self.subs_tile_ub,
                                        self.subs_gm[subs_loop_index + update_num - self.var_data_each_block], 0, 1, 1,
                                        0, 0)
            with self.tik_instance.if_scope(update_num % self.var_data_each_block == 0):
                self.tik_instance.data_move(self.var_ub, self.out_gm[var_loop_index], 0, 1, self.subs_burst_len, 0, 0)
                self.vsub(update_num)
                self.tik_instance.data_move(self.out_gm[var_loop_index], self.subs_ub, 0, 1, self.subs_burst_len, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.var_ub, self.out_gm[var_loop_index], 0, 1, self.subs_burst_len, 0, 0)
                self.tik_instance.data_move(self.var_tile_ub,
                                            self.out_gm[var_loop_index + update_num - self.var_data_each_block], 0, 1,
                                            1, 0, 0)
                self.vsub(update_num)
                self.tik_instance.vec_sub(self.var_data_each_block, self.subs_tile_ub[0], self.var_tile_ub[0],
                                          self.subs_tile_ub[0], 1, 8, 8, 8)
                self.tik_instance.data_move(self.out_gm[var_loop_index + update_num - self.var_data_each_block],
                                            self.subs_tile_ub, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.out_gm[var_loop_index], self.subs_ub, 0, 1, self.subs_burst_len - 1, 0,
                                            0)

    def traversing_subs_32b_aligned_and_ub_enough(self, indices_in_index, indice_num):
        """
        updateDataNum is 32B aligned ub can store all subsNum
        Parameters
        __________
        indice_num: int64
            the number of indexes in the indices on UB
        Returns
        _______
        None
        """
        update_burst_len = self.subs_num // self.var_data_each_block
        self.tik_instance.data_move(self.subs_ub, self.subs_gm, 0, 1, update_burst_len, 0, 0)
        subs_burst_len = self.update_data_num // self.var_data_each_block
        with self.tik_instance.for_range(0, self.indices_front_dim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.tik_instance.data_move(self.var_ub, self.out_gm[self.var_read_index], 0, 1, subs_burst_len, 0,
                                                0)
                    self.vec_sub_with_subr(self.update_data_num,
                                           (indices_in_index + indices_ub_index) * self.update_data_num, 0)
                    self.tik_instance.data_move(
                        self.out_gm[self.var_read_index],
                        self.subs_ub[(indices_in_index + indices_ub_index) * self.update_data_num], 0, 1,
                        subs_burst_len, 0, 0)

    def traversing_subs_single_core_and_ub_enough(self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block, ub can store all subsNum
        Paramters
        _________
        indice_num: int64
            the number of indexes in the indices on UB
        Returns
        _______
        None
        """
        with self.tik_instance.if_scope(self.subs_num % self.var_data_each_block == 0):
            self.subs_burst_len.set_as(self.subs_num // self.var_data_each_block)
        with self.tik_instance.else_scope():
            self.subs_burst_len.set_as(self.subs_num // self.var_data_each_block + 1)
        self.tik_instance.data_move(self.subs_ub, self.subs_gm, 0, 1, self.subs_burst_len, 0, 0)
        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.update_data_num) as subs_ub_index:
                self.update_value.set_as(
                    self.subs_ub[(indices_ub_index + indices_in_index // self.indices_last_dim) * self.update_data_num +
                                 subs_ub_index])
                self.subs_tile_ub[subs_ub_index].set_as(self.update_value)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub, self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.subs_vconv_ub, self.subs_tile_ub, 1, 8, 4)
                self.tik_instance.vec_sub(self.update_data_num, self.var_vconv_ub, self.var_vconv_ub,
                                          self.subs_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub, self.var_vconv_ub, 1, 8, 4)
            else:
                self.tik_instance.vec_sub(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.subs_tile_ub,
                                          1, 8, 8, 8)
            self.tik_instance.data_move(self.out_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)

    def traversing_subs_single_core_and_ub_not_enough(self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block , ub can't store all subsNum
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
            self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(
                self.subs_tile_ub,
                self.subs_gm[(indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num], 0,
                1, 1, 0, 0)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub, self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.subs_vconv_ub, self.subs_tile_ub, 1, 8, 4)
                self.tik_instance.vec_sub(self.update_data_num, self.var_vconv_ub, self.var_vconv_ub,
                                          self.subs_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub, self.var_vconv_ub, 1, 8, 4)
            else:
                self.tik_instance.vec_sub(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.subs_tile_ub,
                                          1, 8, 8, 8)
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
        with self.tik_instance.if_scope(self.indices_loop_num > 0):
            with self.tik_instance.for_range(0, self.indices_loop_num) as indices_loop_index:
                self.move_indices(indices_loop_index * self.indices_align_num, self.indices_align_num)
        with self.tik_instance.if_scope(self.indices_last_num > 0):
            self.move_indices(self.indices_loop_num * self.indices_align_num, self.indices_last_num)

    def scatter_nd_sub_compute_tiling(self):
        """
        Main process of scatter_nd_sub
        Parameters
        __________
        None
        Returns
        _______
        None
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 5, 0, 0)
        self.tiling_args()
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_index:
            with self.tik_instance.if_scope(core_index < self.core_num):
                self.init_ub_tensor()
                self.core_loop_index.set_as(core_index)
                self.traversing_indices()

    def scatter_nd_sub_operator(self):
        """
        scatter_nd_sub operation
        Parameters
        __________
        None
        Returns:
        _______
        compile info
        """
        self.scatter_nd_sub_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        ub_size_bytes = self.ub_size_bytes // 96 * 64
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": ub_size_bytes,
                "core_num": self.ai_core_num,
                "var_size": self.var_dtype_bytes_size,
                "indices_size": self.indices_dtype_bytes_size,
                "support_atomic": 0
            })
        tbe_context.get_context().add_compile_info("is_tik", True)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var_gm, self.indices_gm, self.subs_gm),
                                   outputs=(self.out_gm),
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)


def scatter_nd_sub_tik(var, indices, subs, var_out, use_locking, kernel_name="scatter_nd_sub"):
    obj = ScatterNdSub(var, indices, subs, var_out, False, kernel_name)
    return obj.scatter_nd_sub_operator()


def scatter_nd_sub_dsl(var, indices, subs, var_out, use_locking, kernel_name="scatter_nd_sub", impl_mode=None):
    """
    scatter_nd_sub interface for dsl
    """
    check_list_var = ("float16", "float32", "int32")
    check_list_indices = ("int32", "int64")
    check_list_subs = ("float16", "float32", "int32")
    dtype_var = var.get("dtype").lower()
    dtype_indices = indices.get("dtype").lower()
    dtype_subs = subs.get("dtype").lower()
    para_check.check_dtype(dtype_var, check_list_var, param_name="var")
    para_check.check_dtype(dtype_indices, check_list_indices, param_name="indices")
    para_check.check_dtype(dtype_subs, check_list_subs, param_name="subs")

    op_type = "scatter_nd"
    reduction = "sub"
    ins = classify([var, indices, subs], op_type)
    schedules, tensors = [], []
    for var_input, indices_input, subs_input in ins:
        with tbe.compute():
            var_shape, indices_shape, subs_shape = \
                shape_util.variable_shape([var_input, indices_input, subs_input], "scatter")
            var_tensor = tvm.placeholder(var_shape, name="var", dtype=dtype_var)
            indices_tensor = tvm.placeholder(indices_shape, name="indices", dtype=dtype_indices)
            subs_tensor = tvm.placeholder(subs_shape, name="subs", dtype=dtype_subs)
            res = tbe.scatter_nd(var_tensor, indices_tensor, subs_tensor, reduction,
                                 support_out_of_bound_index=is_support_out_of_bound_index(impl_mode))
            tensors.append([var_tensor, indices_tensor, subs_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=unused-argument
@register_operator("ScatterNdSub")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scatter_nd_sub(var, indices, subs, var_out, use_locking=False, kernel_name="scatter_nd_sub", impl_mode=None):
    """
    scatter_nd_sub interface
    Parameters
    __________
    var: input var shape, dtype and range
    indices: input indices shape, dtype and range
    subs: input subs shape, dtype and range
    var_out: output shape, dtype and range
    use_locking: bool
    kernel_name: kernel_name of scatter_sub op
    impl_mode: implementation mode, such as high_precision, high_performance, support_out_of_bound_index.
    Returns
    _______
    compile info
    """
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        scatter_nd_sub_tik(var, indices, subs, var_out, False, kernel_name)
    else:
        scatter_nd_sub_dsl(var, indices, subs, var_out, False, kernel_name, impl_mode)
