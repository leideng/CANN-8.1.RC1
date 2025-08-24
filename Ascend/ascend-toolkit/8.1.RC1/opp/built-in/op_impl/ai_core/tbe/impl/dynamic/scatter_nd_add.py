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
scatter_nd_add
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpImplMode
from tbe.common.platform import get_bit_len
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl import common_util
from impl.util.util_common import is_support_out_of_bound_index


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max int64 value
    MAX_INT64_VALUE = 2**64 - 1
    # tiling param num
    TILING_ARG_NUM = 20
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # 8 bit
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = 32
    # float16
    DTYPE_F16 = "float16"
    # size of float16
    DTYPE_F16_SIZE = 2


def ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


# 'pylint: disable=too-many-arguments,too-many-instance-attributes
# 'pylint: disable=invalid-name,attribute-defined-outside-init,unused-argument
class ScatterNdAdd():
    """
    Function: use to store scatter_nd_add base parameters
    Modify: 2020-10-29
    """

    def __init__(self, var, indices, adds, var_out, use_locking, kernel_name, impl_mode):
        """
        Init ScatterNdAdd parameters
        Paramters
        _________
        var: dict
            the dict of input tensor.
        indices: dict
            the dict of input tensor.
        add: dict
            the dict of input tensor.
        var_out: dict
            the dict of output tensor.
        use_locking: bool
            not used in this compute, default value is "False".
        kernel_name: str
            cce kernel name, default value is "scatter_nd_add"
        Returns
        _______
        None.
        enable_const_fold
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.var_dtype = var.get("dtype").lower()
        self.indice_dtype = indices.get("dtype").lower()
        self.adds_dtype = adds.get("dtype").lower()
        self.out_dtype = var_out.get("dtype").lower()

        self.action_func = self.tik_instance.vec_add
        self.check_input_params()
        if self.var_dtype in ("bool"):
            self.var_dtype = "uint8"
            self.adds_dtype = "uint8"
            self.action_func = self.tik_instance.vec_max

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_SIZE
        self.var_dtype_bytes_size = tbe_platform.get_bit_len(self.var_dtype) // Constant.EIGHT_BIT
        self.indices_dtype_bytes_size = tbe_platform.get_bit_len(self.indice_dtype) // Constant.EIGHT_BIT
        self.var_data_each_block = Constant.BLOCK_BYTES // self.var_dtype_bytes_size
        self.indices_data_each_block = Constant.BLOCK_BYTES // self.indices_dtype_bytes_size
        self.support_atomic = False
        self.support_out_of_bound_index = True if impl_mode == OpImplMode.SUPPORT_OUT_OF_BOUND_INDEX else False
        if tbe_platform.api_check_support("tik.set_atomic_add",
                                          self.adds_dtype) and self.var_dtype not in ("int8", "uint8"):
            set_atomic_add_reflect_list = {"float32": 1, "float16": 2, "int16": 3, "int32": 4, "int8": 5}
            self.set_atomic_add_value = set_atomic_add_reflect_list.get(self.out_dtype)
            self.need_cast = False
            self.updates_data_each_block = self.var_data_each_block
            self.support_atomic = True
            self.updates_ub_num = self.ub_size_bytes // 2 // self.var_dtype_bytes_size
            self.indices_ub_num = self.ub_size_bytes // 2 // self.indices_dtype_bytes_size
        elif self.var_dtype not in ("int8", "uint8"):
            self.adds_ub_num = self.ub_size_bytes // 96 * 32 // self.var_dtype_bytes_size
            self.indices_ub_num = self.ub_size_bytes // 96 * 32 // self.indices_dtype_bytes_size
            self.ub_size_bytes = self.ub_size_bytes // 96 * 64
        else:
            self.adds_ub_num = self.ub_size_bytes // 7 // self.var_dtype_bytes_size
            self.adds_ub_num = self.adds_ub_num // 128 * 128
            self.ub_size_bytes = self.adds_ub_num * 2 * self.var_dtype_bytes_size
            # need keep the same as tiling compute
            self.indices_ub_num = self.ub_size_bytes // 2 // self.indices_dtype_bytes_size

        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM, ),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (Constant.MAX_INT64_VALUE, ),
                                               name="var_gm",
                                               scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indice_dtype, (Constant.MAX_INT64_VALUE, ),
                                                   name="indices_gm",
                                                   scope=tik.scope_gm)
        self.adds_gm = self.tik_instance.Tensor(self.var_dtype, (Constant.MAX_INT64_VALUE, ),
                                                name="adds_gm",
                                                scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype, (Constant.MAX_INT64_VALUE, ),
                                               name="out_gm",
                                               scope=tik.scope_gm)

        self.adds_ub = None
        self.var_ub = None
        self.adds_ub_f16 = None
        self.var_ub_f16 = None
        self.indices_ub = None
        self.var_tile_ub = None
        self.adds_tile_ub = None
        self.var_vconv_ub = None
        self.adds_vconv_ub = None
        self.tiling_ub = None

        self.var_read_index = None
        self.core_loop_index = None

        self.update_value = None
        self.indices_burst_len = None
        self.adds_burst_len = None
        self.tiling_mode = None
        self.indice_step = None
        self.core_num = None
        self.update_data_num = None

        self.indices_loop_num = None
        self.indices_last_num = None

        self.adds_num = None
        self.adds_loop_num = None
        self.adds_last_num = None
        self.core_num_var = self.tik_instance.Scalar(name="core_num_var", init_value=self.ai_core_num)

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    def check_input_params(self):
        """
        Check whether the input parameters is valid or not
        """
        indices_support_dtype_list = ("int32", "int64")
        var_support_dtype_list = ("float32", "float16", "int32", "int8", "uint8", "bool")
        para_check.check_dtype(self.indice_dtype, indices_support_dtype_list, param_name="indices")
        para_check.check_dtype(self.var_dtype, var_support_dtype_list, param_name="var")
        if self.var_dtype != self.adds_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "add", "var", self.adds_dtype,
                                                                  self.var_dtype)
        if self.var_dtype != self.out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "out", "var", self.out_dtype,
                                                                  self.var_dtype)

    def calc_indices(self, indices_ub_index):
        """
        calculate indices on gm
        """
        self.var_read_index.set_as(0)
        for i in range(7):
            tiling_offset = self.tik_instance.Scalar("int64", name="tmp")
            adds_sum_index = self.tik_instance.Scalar("int64", name="adds_sum_index")
            indices_sum_index = self.tik_instance.Scalar("int64", name="indices_sum_index")
            indices_sum_index.set_as(indices_ub_index * self.indices_last_dim + i)
            adds_sum_index.set_as(self.indices_ub[indices_sum_index])
            tiling_offset.set_as(self.var_offset_index_tiling[i])
            self.var_read_index.set_as(self.var_read_index + adds_sum_index * tiling_offset)

    def tiling_args(self):
        """
        get runtime params from tiling
        Parameters
        __________
        tiling_ub: tensor, runtime params from scatter_nd_add tiling
        Returns
        _______
        None
        """
        self.var_offset_index_tiling = self.tik_instance.ScalarArray(dtype="int64",
                                                                     length=7,
                                                                     name="var_offset_index_tiling")
        for index in range(7):
            self.var_offset_index_tiling[index].set_as(self.tiling_ub[index])

        if self.support_atomic:
            self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
            self.core_num = self.tik_instance.Scalar("int64", name="core_num")
            self.update_data_num = self.tik_instance.Scalar("int64", name="update_data_num")
            self.indices_each_core_data = self.tik_instance.Scalar("int64", name="indices_each_core_data")
            self.indices_last_core_data = self.tik_instance.Scalar("int64", name="indices_last_core_data")
            self.each_core_indices_loop_num = self.tik_instance.Scalar("int64", name="each_core_indices_loop_num")
            self.each_core_indices_last_num = self.tik_instance.Scalar("int64", name="each_core_indices_last_num")
            self.last_core_indices_loop_num = self.tik_instance.Scalar("int64", name="last_core_indices_loop_num")
            self.last_core_indices_last_num = self.tik_instance.Scalar("int64", name="last_core_indices_last_num")
            self.indices_last_dim = self.tik_instance.Scalar("int64", name="indices_last_dim")
            self.updates_loop_num = self.tik_instance.Scalar("int64", name="updates_loop_num")
            self.updates_last_num = self.tik_instance.Scalar("int64", name="updates_last_num")

            self.tiling_mode.set_as(self.tiling_ub[7])
            self.core_num.set_as(self.tiling_ub[8])
            self.indices_last_dim.set_as(self.tiling_ub[9])
            self.update_data_num.set_as(self.tiling_ub[10])
            self.indices_each_core_data.set_as(self.tiling_ub[11])
            self.indices_last_core_data.set_as(self.tiling_ub[12])
            self.each_core_indices_loop_num.set_as(self.tiling_ub[13])
            self.each_core_indices_last_num.set_as(self.tiling_ub[14])
            self.last_core_indices_loop_num.set_as(self.tiling_ub[15])
            self.last_core_indices_last_num.set_as(self.tiling_ub[16])
            self.updates_loop_num.set_as(self.tiling_ub[17])
            self.updates_last_num.set_as(self.tiling_ub[18])
            self.set_running_core_num(self.tiling_ub[19])

        else:
            self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
            self.indice_step = self.tik_instance.Scalar("int64", name="indice_step")
            self.core_num = self.tik_instance.Scalar("int64", name="core_num")
            self.update_data_num = self.tik_instance.Scalar("int64", name="update_data_num")
            self.indices_loop_num = self.tik_instance.Scalar("int64", name="indices_loop_num")
            self.indices_last_num = self.tik_instance.Scalar("int64", name="indices_last_num")
            self.adds_num = self.tik_instance.Scalar("int64", name="adds_num")
            self.adds_loop_num = self.tik_instance.Scalar("int64", name="adds_loop_num")
            self.adds_last_num = self.tik_instance.Scalar("int64", name="adds_last_num")
            self.indices_last_dim = self.tik_instance.Scalar("int64", name="indices_last_dim")
            self.indicesFrontDim = self.tik_instance.Scalar("int64", name="indicesFrontDim")
            self.indicesAlignNum = self.tik_instance.Scalar("int64", name="indicesAlignNum")

            self.tiling_mode.set_as(self.tiling_ub[7])
            self.indice_step.set_as(self.tiling_ub[8])
            self.core_num.set_as(self.tiling_ub[9])
            self.update_data_num.set_as(self.tiling_ub[10])
            self.indices_loop_num.set_as(self.tiling_ub[11])
            self.indices_last_num.set_as(self.tiling_ub[12])
            self.adds_num.set_as(self.tiling_ub[13])
            self.adds_loop_num.set_as(self.tiling_ub[14])
            self.adds_last_num.set_as(self.tiling_ub[15])
            self.indices_last_dim.set_as(self.tiling_ub[16])
            self.indicesFrontDim.set_as(self.tiling_ub[17])
            self.indicesAlignNum.set_as(self.indices_ub_num // self.indices_last_dim * self.indices_last_dim)
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
        if self.support_atomic:
            self.updates_tile_ub = self.tik_instance.Tensor(self.adds_dtype, (self.updates_data_each_block, ),
                                                            name="updates_tile_ub",
                                                            scope=tik.scope_ubuf)
            self.updates_ub = self.tik_instance.Tensor(self.adds_dtype, (self.updates_ub_num, ),
                                                       name="updates_ub",
                                                       scope=tik.scope_ubuf)

            self.indices_ub = self.tik_instance.Tensor(self.indice_dtype, (self.indices_ub_num, ),
                                                       name="indices_ub",
                                                       scope=tik.scope_ubuf)
            self.var_read_index = self.tik_instance.Scalar("int32", name="var_read_index")
            self.var_read_index.set_as(0)
            self.core_loop_index = self.tik_instance.Scalar("int32", name="core_loop_index")
            self.core_loop_index.set_as(0)
            self.update_data_block_align_num = self.tik_instance.Scalar("int32", name="update_data_block_align_num")
            self.update_data_block_align_num.set_as((self.update_data_num + self.updates_data_each_block - 1) //
                                                    self.updates_data_each_block * self.updates_data_each_block)
            self.zero_ub = self.tik_instance.Tensor(self.adds_dtype, (self.update_data_block_align_num, ),
                                                    name="zero_ub",
                                                    scope=tik.scope_ubuf)

        else:
            self.adds_ub = self.tik_instance.Tensor(self.var_dtype, (self.adds_ub_num, ),
                                                    name="adds_ub",
                                                    scope=tik.scope_ubuf)
            self.var_ub = self.tik_instance.Tensor(self.var_dtype, (self.adds_ub_num, ),
                                                   name="var_ub",
                                                   scope=tik.scope_ubuf)
            self.indices_ub = self.tik_instance.Tensor(self.indice_dtype, (self.indices_ub_num, ),
                                                       name="indices_ub",
                                                       scope=tik.scope_ubuf)
            if self.var_dtype in ("int8", "uint8"):
                self.adds_ub_f16 = self.tik_instance.Tensor(Constant.DTYPE_F16, (self.adds_ub_num, ),
                                                            name="adds_ub_f16",
                                                            scope=tik.scope_ubuf)
                self.var_ub_f16 = self.tik_instance.Tensor(Constant.DTYPE_F16, (self.adds_ub_num, ),
                                                           name="var_ub_f16",
                                                           scope=tik.scope_ubuf)
            self.var_tile_ub = self.tik_instance.Tensor(self.var_dtype, (128, ),
                                                        name="var_tile_ub",
                                                        scope=tik.scope_ubuf)
            self.adds_tile_ub = self.tik_instance.Tensor(self.var_dtype, (128, ),
                                                         name="adds_tile_ub",
                                                         scope=tik.scope_ubuf)
            self.var_vconv_ub = self.tik_instance.Tensor("float16", (128, ), 
                                                         name="var_vconv_ub", scope=tik.scope_ubuf)
            self.adds_vconv_ub = self.tik_instance.Tensor("float16", (128, ),
                                                         name="adds_vconv_ub", scope=tik.scope_ubuf)

            self.var_read_index = self.tik_instance.Scalar("int64", name="var_read_index")
            self.var_read_index.set_as(0)
            self.core_loop_index = self.tik_instance.Scalar("int64", name="core_loop_index")
            self.core_loop_index.set_as(0)
            self.update_value = self.tik_instance.Scalar(self.var_dtype, name="update_value")
            self.update_value.set_as(0)
            self.indices_burst_len = self.tik_instance.Scalar("int64", name="indices_burst_len")
            self.indices_burst_len.set_as(0)
            self.adds_burst_len = self.tik_instance.Scalar("int64", name="adds_burst_len")
            self.adds_burst_len.set_as(0)

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
        if tbe_platform.api_check_support("tik.data_move_pad"):
            indices_gm_int8 = self.indices_gm.reinterpret_cast_to("int8")
            indices_ub_int8 = self.indices_ub.reinterpret_cast_to("int8")
            self.tik_instance.data_move_pad(indices_ub_int8,
                                            indices_gm_int8[indices_in_index * self.indices_dtype_bytes_size], 1,
                                            indice_num * self.indices_dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(self.indices_ub, self.indices_gm[indices_in_index], 0, 1,
                                        ceil_div(indice_num, self.indices_data_each_block), 0, 0)

        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.traversing_adds_32b_aligned_and_ub_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.circulate_indices(indices_in_index, indice_num, 2)
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.traversing_adds_single_core_and_ub_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 4):
            self.traversing_adds_single_core_and_ub_not_enough(indices_in_index, indice_num)
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
        with self.tik_instance.for_range(0, self.indicesFrontDim) as indices_ub_index:
            with self.tik_instance.if_scope(indices_ub_index < indice_num // self.indices_last_dim):
                self.calc_indices(indices_ub_index)
                with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                    with self.tik_instance.if_scope(
                        (self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                        self.traversing_adds(indices_ub_index, indices_in_index, mode)

    def traversing_adds(self, indices_ub_index, indices_in_index, mode):
        """
        Traversing the index in the adds
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
        with self.tik_instance.if_scope(self.adds_loop_num > 0):
            with self.tik_instance.for_range(0, self.adds_loop_num) as adds_loop_index:
                self.update_var((indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num +
                                adds_loop_index * self.adds_ub_num, self.adds_ub_num,
                                adds_loop_index * self.adds_ub_num + self.var_read_index, mode)
        with self.tik_instance.if_scope(self.adds_last_num > 0):
            self.update_var((indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num +
                            self.adds_loop_num * self.adds_ub_num, self.adds_last_num,
                            self.adds_loop_num * self.adds_ub_num + self.var_read_index, mode)

    def vadd(self, update_num):
        """
        vadd
        """
        adds_ub_tmp = self.adds_ub
        var_ub_tmp = self.var_ub
        mask = 256 // self.var_dtype_bytes_size
        if self.var_dtype in ("int8", "uint8"):
            mask = 256 // Constant.DTYPE_F16_SIZE
            adds_ub_tmp = self.adds_ub_f16
            var_ub_tmp = self.var_ub_f16
        loop = (update_num + mask - 1) // mask // 255
        last = (update_num + mask - 1) // mask % 255
        lastindex = loop * 255 + last - 1
        lastmask_scalar = self.tik_instance.Scalar("int64", name="lastmask")
        lastmask = update_num % mask
        lastmask_scalar.set_as(lastmask)
        with self.tik_instance.if_scope(lastmask_scalar == 0):
            lastmask_scalar.set_as(mask)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as i:
                self.action_func(mask, adds_ub_tmp[i * 255 * mask], adds_ub_tmp[i * 255 * mask],
                                 var_ub_tmp[i * 255 * mask], 255, 8, 8, 8)
        with self.tik_instance.if_scope(last > 0):
            self.action_func(mask, adds_ub_tmp[loop * 255 * mask], adds_ub_tmp[loop * 255 * mask],
                             var_ub_tmp[loop * 255 * mask], last - 1, 8, 8, 8)
            self.action_func(lastmask_scalar, adds_ub_tmp[lastindex * mask], adds_ub_tmp[lastindex * mask],
                             var_ub_tmp[lastindex * mask], 1, 8, 8, 8)

    def vec_add_with_addr(self, update_num, adds_addr, var_addr):
        """
        vec_add_with_addr
        """
        adds_ub_tmp = self.adds_ub
        var_ub_tmp = self.var_ub
        mask = 256 // self.var_dtype_bytes_size
        if self.var_dtype in ("int8", "uint8"):
            mask = 256 // Constant.DTYPE_F16_SIZE
            adds_ub_tmp = self.adds_ub_f16
            var_ub_tmp = self.var_ub_f16
        loop = (update_num + mask - 1) // mask // 255
        last = (update_num + mask - 1) // mask % 255
        lastindex = loop * 255 + last - 1
        lastmask_scalar = self.tik_instance.Scalar("int64", name="lastmask")
        lastmask = update_num % mask
        lastmask_scalar.set_as(lastmask)
        with self.tik_instance.if_scope(lastmask_scalar == 0):
            lastmask_scalar.set_as(mask)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as i:
                self.action_func(mask, adds_ub_tmp[i * 255 * mask + adds_addr], adds_ub_tmp[i * 255 * mask + adds_addr],
                                 var_ub_tmp[i * 255 * mask + var_addr], 255, 8, 8, 8)
        with self.tik_instance.if_scope(last > 0):
            self.action_func(mask, adds_ub_tmp[loop * 255 * mask + adds_addr],
                             adds_ub_tmp[loop * 255 * mask + adds_addr], var_ub_tmp[loop * 255 * mask + var_addr],
                             last - 1, 8, 8, 8)
            self.action_func(lastmask_scalar, adds_ub_tmp[lastindex * mask + adds_addr],
                             adds_ub_tmp[lastindex * mask + adds_addr], var_ub_tmp[lastindex * mask + var_addr], 1, 8,
                             8, 8)

    def update_var(self, adds_loop_index, update_num, var_loop_index, mode):
        """
        Update the update fragment corresponding to the index
        Parameters
        __________
        adds_loop_index: int64
            Update index on GM
        update_num: int64
            the number of indexes in the adds on UB
        var_loop_index: int64
            Var index on GM

        Returns
        _______
        None
        """
        if mode == 2:
            self.adds_burst_len.set_as(update_num // self.var_data_each_block)
            self.move_data_to_ub(self.adds_ub, self.adds_ub_f16, self.adds_gm, adds_loop_index, 0, update_num)
            self.move_data_to_ub(self.var_ub, self.var_ub_f16, self.out_gm, var_loop_index, 0, update_num)
            self.vadd(update_num)
            self.move_data_to_gm(self.adds_ub, self.adds_ub_f16, var_loop_index, 0, update_num)
        if mode == 5:
            with self.tik_instance.if_scope(update_num % self.var_data_each_block == 0):
                self.adds_burst_len.set_as(update_num // self.var_data_each_block)
            with self.tik_instance.else_scope():
                self.adds_burst_len.set_as(update_num // self.var_data_each_block + 1)
            self.move_data_to_ub(self.adds_ub, self.adds_ub_f16, self.adds_gm, adds_loop_index, 0, update_num)
            self.move_data_to_ub(self.adds_tile_ub, self.adds_vconv_ub, self.adds_gm,
                                 adds_loop_index + update_num - self.var_data_each_block, 0, self.var_data_each_block)
            with self.tik_instance.if_scope(update_num % self.var_data_each_block == 0):
                self.move_data_to_ub(self.var_ub, self.var_ub_f16, self.out_gm, var_loop_index, 0, update_num)
                self.vadd(update_num)
                self.move_data_to_gm(self.adds_ub, self.adds_ub_f16, var_loop_index, 0, update_num)
            with self.tik_instance.else_scope():
                self.move_data_to_ub(self.var_ub, self.var_ub_f16, self.out_gm, var_loop_index, 0, update_num)
                self.move_data_to_ub(self.var_tile_ub, self.var_vconv_ub, self.out_gm,
                                     var_loop_index + update_num - self.var_data_each_block, 0,
                                     self.var_data_each_block)
                self.vadd(update_num)
                if self.var_dtype not in ("int8", "uint8"):
                    self.action_func(self.var_data_each_block, self.adds_tile_ub[0], self.adds_tile_ub[0],
                                              self.var_tile_ub[0], 1, 8, 8, 8)
                else:
                    self.action_func(self.var_data_each_block, self.adds_vconv_ub[0], self.adds_vconv_ub[0],
                                              self.var_vconv_ub[0], 1, 8, 8, 8)
                self.move_data_to_gm(self.adds_ub, self.adds_ub_f16, var_loop_index, 0,
                                     update_num - self.var_data_each_block)
                self.move_data_to_gm(self.adds_tile_ub, self.adds_vconv_ub,
                                     var_loop_index + update_num - self.var_data_each_block, 0,
                                     self.var_data_each_block)

    # 'pylint: disable=huawei-too-many-arguments
    def move_data_to_ub(self, adds_ub, adds_ub_f16, gm_addr, gm_offset, ub_offset, adds_num):
        """
        data move for adds, if adds type is 1B, needs covert to f16
        __________
        offset: data offset gm
        adds_num: the number of adds
        Returns
        _______
        None
        """
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(adds_ub[ub_offset], gm_addr[gm_offset], 1,
                                            adds_num * self.var_dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(adds_ub[ub_offset], gm_addr[gm_offset], 0, 1,
                                        ceil_div(adds_num, self.var_data_each_block), 0, 0)
        if self.var_dtype in ("int8", "uint8"):
            common_util.conv_i1_to_s2(self.tik_instance, adds_ub_f16[ub_offset], adds_ub[ub_offset], adds_num)

    def move_data_to_gm(self, adds_ub, adds_ub_f16, gm_offset, ub_offset, adds_num):
        """
        data move for adds, if adds type is 1B, needs covert to f16
        __________
        offset: data offset gm
        adds_num: the number of adds
        Returns
        _______
        None
        """
        if self.var_dtype in ("int8", "uint8"):
            common_util.conv_s2_to_i1(self.tik_instance, adds_ub[ub_offset], adds_ub_f16[ub_offset], adds_num)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(self.out_gm[gm_offset], adds_ub[ub_offset], 1,
                                            adds_num * self.var_dtype_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(self.out_gm[gm_offset], adds_ub[ub_offset], 0, 1,
                                        ceil_div(adds_num, self.var_data_each_block), 0, 0)

    def traversing_adds_32b_aligned_and_ub_enough(self, indices_in_index, indice_num):
        """
        updateDataNum is 32B aligned ub can store all addsNum
        Parameters
        __________
        indice_num: int64
            the number of indexes in the indices on UB
        Returns
        _______
        None
        """
        self.move_data_to_ub(self.adds_ub, self.adds_ub_f16, self.adds_gm, 0, 0, self.adds_num)
        with self.tik_instance.for_range(0, self.indicesFrontDim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.move_data_to_ub(self.var_ub, self.var_ub_f16, self.out_gm, self.var_read_index, 0,
                                         self.update_data_num)
                    self.vec_add_with_addr(self.update_data_num,
                                           (indices_in_index + indices_ub_index) * self.update_data_num, 0)
                    self.move_data_to_gm(self.adds_ub, self.adds_ub_f16, self.var_read_index,
                                         (indices_in_index + indices_ub_index) * self.update_data_num,
                                         self.update_data_num)

    def traversing_adds_single_core_and_ub_enough(self, indices_in_index,
                                                  indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block, ub can store all addsNum
        Paramters
        _________
        indice_num: int64
            the number of indexes in the indices on UB
        Returns
        _______
        None
        """
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(self.adds_ub, self.adds_gm, 1, self.adds_num * self.var_dtype_bytes_size, 0,
                                            0)
        else:
            self.tik_instance.data_move(self.adds_ub, self.adds_gm, 0, 1,
                                        ceil_div(self.adds_num, self.var_data_each_block), 0, 0)
        with self.tik_instance.for_range(
                0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_instance.data_move_pad(self.var_tile_ub, self.out_gm[self.var_read_index], 1,
                                                self.update_data_num * self.var_dtype_bytes_size, 0, 0)
            else:
                self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(
                    0, self.update_data_num) as adds_ub_index:
                self.update_value.set_as(
                    self.adds_ub[(indices_ub_index +
                                  indices_in_index // self.indices_last_dim) *
                                 self.update_data_num + adds_ub_index])
                self.adds_tile_ub[adds_ub_index].set_as(self.update_value)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub,
                                           self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.adds_vconv_ub,
                                           self.adds_tile_ub, 1, 8, 4)
                self.action_func(self.update_data_num,
                                          self.var_vconv_ub, self.var_vconv_ub,
                                          self.adds_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub,
                                           self.var_vconv_ub, 1, 8, 4)
            else:
                self.action_func(self.update_data_num,
                                          self.var_tile_ub, self.var_tile_ub,
                                          self.adds_tile_ub, 1, 8, 8, 8)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_instance.data_move_pad(self.out_gm[self.var_read_index], self.var_tile_ub, 1,
                                                self.update_data_num * self.var_dtype_bytes_size, 0, 0)
            else:                            
                self.tik_instance.data_move(self.out_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)

    def traversing_adds_single_core_and_ub_not_enough(self, indices_in_index,
                                                      indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block , ub can't store all addsNum
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
        with self.tik_instance.for_range(
                0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            adds_gm_offset = (indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_instance.data_move_pad(self.var_tile_ub, self.out_gm[self.var_read_index], 1,
                                                self.update_data_num * self.var_dtype_bytes_size, 0, 0)
                self.tik_instance.data_move_pad(self.adds_tile_ub, self.adds_gm[adds_gm_offset], 1,
                                                self.update_data_num * self.var_dtype_bytes_size, 0, 0)
            else:
                self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index], 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.adds_tile_ub, self.adds_gm[adds_gm_offset], 0, 1, 1, 0, 0)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub,
                                           self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.adds_vconv_ub,
                                           self.adds_tile_ub, 1, 8, 4)
                self.action_func(self.update_data_num,
                                          self.var_vconv_ub, self.var_vconv_ub,
                                          self.adds_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub,
                                           self.var_vconv_ub, 1, 8, 4)
            else:
                self.action_func(self.update_data_num,
                                          self.var_tile_ub, self.var_tile_ub,
                                          self.adds_tile_ub, 1, 8, 8, 8)
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
        with self.tik_instance.if_scope(self.indices_loop_num > 0):
            with self.tik_instance.for_range(
                    0, self.indices_loop_num) as indices_loop_index:
                self.move_indices(indices_loop_index * self.indicesAlignNum,
                                  self.indicesAlignNum)
        with self.tik_instance.if_scope(self.indices_last_num > 0):
            self.move_indices(self.indices_loop_num * self.indicesAlignNum,
                              self.indices_last_num)

    def traversing_indices_deepfm(self):
        """
        Traversing the index in the indices for deepfm

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.tiling_mode == 6):
            self.indices_ub_num = self.indices_ub_num // self.indices_last_dim * self.indices_last_dim
        with self.tik_instance.if_scope(self.tiling_mode == 7):
            self.indices_ub_num = (self.indices_ub_num // self.update_data_block_align_num //
                                   self.indices_last_dim * self.indices_last_dim)

        with self.tik_instance.if_scope(
                self.core_loop_index < self.core_num - 1):
            with self.tik_instance.for_range(
                    0, self.each_core_indices_loop_num) as indices_loop_index:
                self.deepfm_perf(
                    self.core_loop_index * self.indices_each_core_data +
                    indices_loop_index * self.indices_ub_num,
                    self.indices_ub_num)
            with self.tik_instance.if_scope(
                    self.each_core_indices_last_num > 0):
                self.deepfm_perf(
                    self.core_loop_index * self.indices_each_core_data +
                    self.each_core_indices_loop_num * self.indices_ub_num,
                    self.each_core_indices_last_num)
        with self.tik_instance.if_scope(self.core_loop_index == self.core_num -
                                        1):
            with self.tik_instance.for_range(
                    0, self.last_core_indices_loop_num) as indices_loop_index:
                self.deepfm_perf(
                    self.core_loop_index * self.indices_each_core_data +
                    indices_loop_index * self.indices_ub_num,
                    self.indices_ub_num)
            with self.tik_instance.if_scope(
                    self.last_core_indices_last_num > 0):
                self.deepfm_perf(
                    self.core_loop_index * self.indices_each_core_data +
                    self.last_core_indices_loop_num * self.indices_ub_num,
                    self.last_core_indices_last_num)

    def deepfm_perf(self, indices_in_index, indice_num):
        """
        Move indices, deepfm branch

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
        indices_burst_len = ceil_div(indice_num, self.indices_data_each_block)
        self.tik_instance.data_move(self.indices_ub,
                                    self.indices_gm[indices_in_index], 0, 1,
                                    indices_burst_len, 0, 0)

        with self.tik_instance.if_scope(self.tiling_mode == 6):
            self.update_data_block_align_num.set_as(self.update_data_num //
                                                    self.updates_data_each_block *
                                                    self.updates_data_each_block)
            with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index_outer:
                self.calc_indices(indices_ub_index_outer)
                var_read_index = self.var_read_index
                updates_read_index = (indices_in_index // self.indices_last_dim +
                                      indices_ub_index_outer) * self.update_data_num
                self.atomic_datamove_large(var_read_index, updates_read_index)

        with self.tik_instance.if_scope(self.tiling_mode == 7):
            self.atomic_datamove_small(indices_in_index, indice_num)

    def atomic_datamove_small(self, indices_in_index, indice_num):
        is_align = self.update_data_num % self.updates_data_each_block
        with self.tik_instance.if_scope(is_align == 0):
            update_read_index = (indices_in_index // self.indices_last_dim) * self.update_data_num
            update_burst_len = (indice_num // self.indices_last_dim *
                                self.update_data_num) // self.updates_data_each_block
            self.tik_instance.data_move(
                self.updates_ub, self.adds_gm[update_read_index], 0, 1, update_burst_len, 0, 0)
            self.tik_instance.set_atomic_add(self.set_atomic_add_value)
            with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as idx:
                self.calc_indices(idx)
                update_ub_idx = idx * self.update_data_num
                if self.support_out_of_bound_index:
                    with self.tik_instance.if_scope(self.var_read_index > -1):
                        with self.tik_instance.if_scope(self.var_read_index > -1):
                            self.tik_instance.data_move(
                                self.out_gm[self.var_read_index], self.updates_ub[update_ub_idx], 0, 1,
                                self.update_data_num // self.updates_data_each_block, 0, 0)
                else:
                    self.tik_instance.data_move(
                        self.out_gm[self.var_read_index], self.updates_ub[update_ub_idx], 0, 1,
                        self.update_data_num // self.updates_data_each_block, 0, 0)
            self.tik_instance.set_atomic_add(0)

        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as idx:
                update_read_index = (indices_in_index // self.indices_last_dim + idx) * self.update_data_num
                update_ub_idx = idx * self.update_data_block_align_num
                update_burst_len = ceil_div(self.update_data_num, self.updates_data_each_block)
                self.tik_instance.data_move(
                    self.updates_ub[update_ub_idx], self.adds_gm[update_read_index], 0, 1,
                    update_burst_len, 0, 0)

            self.tik_instance.set_atomic_add(self.set_atomic_add_value)
            with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as idx:
                self.calc_indices(idx)
                self._do_vec_dup()
                update_ub_idx = idx * self.update_data_block_align_num
                update_burst_len = ceil_div(self.update_data_num, self.updates_data_each_block)
                self._do_vec_add(update_ub_idx)
                self.tik_instance.data_move(
                    self.out_gm[self.var_read_index], self.zero_ub, 0, 1,
                    update_burst_len, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def atomic_datamove_large(self, var_read_index, update_read_index):
        # data move all update
        with self.tik_instance.for_range(
                0, self.updates_loop_num) as updates_loop_index:
            var_read_index_1 = var_read_index + updates_loop_index * self.updates_ub_num
            update_read_index_1 = update_read_index + updates_loop_index * self.updates_ub_num
            self.atomic_datamove_updateubnum(var_read_index_1,
                                             update_read_index_1,
                                             self.updates_ub_num)
        with self.tik_instance.if_scope(self.updates_last_num > 0):
            var_read_index_1 = var_read_index + self.updates_loop_num * self.updates_ub_num
            update_read_index_1 = update_read_index + self.updates_loop_num * self.updates_ub_num
            self.atomic_datamove_updateubnum(var_read_index_1,
                                             update_read_index_1,
                                             self.updates_last_num)

    def _do_vec_dup(self):
        max_dump_mask = self.updates_data_each_block * Constant.EIGHT_BIT
        repeat_num = self.update_data_block_align_num // max_dump_mask
        remain_mask = self.update_data_block_align_num % max_dump_mask
        with self.tik_instance.if_scope(repeat_num > 0):
            self.tik_instance.vec_dup(max_dump_mask,
                                      self.zero_ub,
                                      0, repeat_num, Constant.EIGHT_BIT)
        dump_offset = repeat_num * max_dump_mask
        with self.tik_instance.if_scope(remain_mask > 0):
            self.tik_instance.vec_dup(remain_mask,
                                      self.zero_ub[dump_offset],
                                      0, 1, Constant.EIGHT_BIT)

    def _do_vec_add(self, update_ub_idx):
        max_mask = self.updates_data_each_block * Constant.EIGHT_BIT
        repeat_num = self.update_data_num // max_mask
        remain_mask = self.update_data_num % max_mask
        with self.tik_instance.if_scope(repeat_num > 0):
            self.tik_instance.vec_add(max_mask, self.zero_ub,
                                      self.zero_ub, self.updates_ub[update_ub_idx],
                                      repeat_num, Constant.EIGHT_BIT, Constant.EIGHT_BIT, Constant.EIGHT_BIT)
        remain_offset = repeat_num * max_mask
        with self.tik_instance.if_scope(remain_mask > 0):
            self.tik_instance.vec_add(remain_mask, self.zero_ub[remain_offset],
                                      self.zero_ub[remain_offset], self.updates_ub[update_ub_idx + remain_offset],
                                      1, Constant.EIGHT_BIT, Constant.EIGHT_BIT, Constant.EIGHT_BIT)

    def atomic_datamove_updateubnum(self, var_read_index, update_read_index,
                                    update_data_num):
        # data move one time fill update ub
        update_data_block_align_num = update_data_num // self.updates_data_each_block * self.updates_data_each_block
        with self.tik_instance.if_scope(update_data_block_align_num > 0):
            self.atomic_datamove_32B(var_read_index, update_read_index,
                                     update_data_block_align_num)
        with self.tik_instance.if_scope(
                update_data_num != update_data_block_align_num):
            self.atomic_datamove_less_oneblock(
                var_read_index + update_data_block_align_num,
                update_read_index + update_data_block_align_num,
                update_data_num - update_data_block_align_num)

    def atomic_datamove_32B(self, var_read_index, update_read_index,
                            update_data_num):
        # data move 32B align
        if self.need_cast:
            max_repeat_loop_num = update_data_num // Constant.MASK_CAST // Constant.MAX_REPEAT
            max_repeat_loop_last = update_data_num // Constant.MASK_CAST % Constant.MAX_REPEAT
            repeat_loop_left = update_data_num % Constant.MASK_CAST
            self.tik_instance.data_move(
                self.updates_ub, self.adds_gm[update_read_index], 0, 1,
                update_data_num // self.updates_data_each_block, 0, 0)
            with self.tik_instance.for_range(
                    0, max_repeat_loop_num) as max_repeat_loop_index:
                updates_fp32_ub_index = Constant.MAX_REPEAT * max_repeat_loop_index * Constant.MASK_CAST
                updates_ub_index = Constant.MAX_REPEAT * max_repeat_loop_index * Constant.MASK_CAST
                self.tik_instance.vconv(
                    Constant.MASK_CAST, "none",
                    self.updates_fp32_ub[updates_fp32_ub_index],
                    self.updates_ub[updates_ub_index], Constant.MAX_REPEAT, 1,
                    1, 8, 4)
            with self.tik_instance.if_scope(max_repeat_loop_last > 0):
                updates_fp32_ub_index = Constant.MAX_REPEAT * max_repeat_loop_num * Constant.MASK_CAST
                updates_ub_index = Constant.MAX_REPEAT * max_repeat_loop_num * Constant.MASK_CAST
                self.tik_instance.vconv(
                    Constant.MASK_CAST, "none",
                    self.updates_fp32_ub[updates_fp32_ub_index],
                    self.updates_ub[updates_ub_index], max_repeat_loop_last, 1,
                    1, 8, 4)
            with self.tik_instance.if_scope(repeat_loop_left > 0):
                max_repeat = Constant.MAX_REPEAT * max_repeat_loop_num * Constant.MASK_CAST
                updates_fp32_ub_index = max_repeat + max_repeat_loop_last * Constant.MASK_CAST
                updates_ub_index = max_repeat + max_repeat_loop_last * Constant.MASK_CAST
                self.tik_instance.vconv(
                    repeat_loop_left, "none",
                    self.updates_fp32_ub[updates_fp32_ub_index],
                    self.updates_ub[updates_ub_index], 1, 1, 1, 8, 4)
            self.tik_instance.set_atomic_add(self.set_atomic_add_value)
            self.tik_instance.data_move(
                self.out_gm[var_read_index], self.updates_fp32_ub, 0, 1,
                update_data_num // self.updates_data_each_block * 2, 0, 0)
            self.tik_instance.set_atomic_add(0)
        else:
            self.tik_instance.data_move(
                self.updates_ub, self.adds_gm[update_read_index], 0, 1,
                update_data_num // self.updates_data_each_block, 0, 0)
            self.tik_instance.set_atomic_add(self.set_atomic_add_value)
            self.tik_instance.data_move(
                self.out_gm[var_read_index], self.updates_ub, 0, 1,
                update_data_num // self.updates_data_each_block, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def atomic_datamove_less_oneblock(self, var_read_index, update_read_index,
                                      update_data_num):
        # data move less than one block
        if self.need_cast:
            self.tik_instance.data_move(self.updates_tile_ub,
                                        self.adds_gm[update_read_index], 0, 1,
                                        1, 0, 0)
            self.tik_instance.vec_muls(self.updates_data_each_block,
                                       self.zero_ub, self.updates_tile_ub, 0,
                                       1, 8, 8)
            self.tik_instance.vec_add(update_data_num, self.zero_ub,
                                      self.zero_ub, self.updates_tile_ub, 1, 8,
                                      8, 8)
            self.tik_instance.vconv(self.updates_data_each_block, "none",
                                    self.zero_fp32_ub, self.zero_ub, 1, 1, 1,
                                    8, 4)
            self.tik_instance.set_atomic_add(self.set_atomic_add_value)
            self.tik_instance.data_move(self.out_gm[var_read_index],
                                        self.zero_fp32_ub, 0, 1, 2, 0, 0)
            self.tik_instance.set_atomic_add(0)
        else:
            self.tik_instance.data_move(self.updates_tile_ub,
                                        self.adds_gm[update_read_index], 0, 1,
                                        1, 0, 0)
            self.tik_instance.vec_muls(self.updates_data_each_block,
                                       self.zero_ub, self.updates_tile_ub, 0,
                                       1, 8, 8)
            self.tik_instance.vec_add(update_data_num, self.zero_ub,
                                      self.zero_ub, self.updates_tile_ub, 1, 8,
                                      8, 8)
            self.tik_instance.set_atomic_add(self.set_atomic_add_value)
            self.tik_instance.data_move(self.out_gm[var_read_index],
                                        self.zero_ub, 0, 1, 1, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def scatter_nd_add_compute_tiling(self):
        """
        Main process of scatter_nd_add
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
        with self.tik_instance.for_range(
                0, self.core_num_var, block_num=self.core_num_var) as core_index:
            if self.support_atomic:
                self.init_ub_tensor()
                self.core_loop_index.set_as(core_index)
                self.traversing_indices_deepfm()

            else:
                with self.tik_instance.if_scope(core_index < self.core_num):
                    self.init_ub_tensor()
                    self.core_loop_index.set_as(core_index)
                    self.traversing_indices()

    def scatter_nd_add_operator(self):
        """
        scatter_nd_add operation
        Parameters
        __________
        None
        Returns:
        _______
        compile info
        """
        self.scatter_nd_add_compute_tiling()
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": self.ai_core_num,
                "var_size": self.var_dtype_bytes_size,
                "indices_size": self.indices_dtype_bytes_size,
                "support_atomic": 1 if self.support_atomic else 0
            })
        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }
        tbe_context.get_context().add_compile_info("is_tik", True)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var_gm, self.indices_gm,
                                           self.adds_gm),
                                   outputs=(self.out_gm),
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)


def scatter_nd_add_tik(var, indices, adds, var_out, use_locking, kernel_name, impl_mode):
    obj = ScatterNdAdd(var, indices, adds, var_out, False, kernel_name, impl_mode)
    return obj.scatter_nd_add_operator()


def scatter_nd_add_dsl(var, indices, adds, var_out, use_locking, kernel_name, impl_mode):
    """
    scatter_nd_add interface for dsl
    """
    check_list_var = ("float16", "float32", "int32")
    check_list_indices = ("int32", "int64")
    check_list_adds = ("float16", "float32", "int32")
    dtype_var = var.get("dtype").lower()
    dtype_indices = indices.get("dtype").lower()
    dtype_adds = adds.get("dtype").lower()
    para_check.check_dtype(dtype_var, check_list_var, param_name="var")
    para_check.check_dtype(dtype_indices, check_list_indices, param_name="indices")
    para_check.check_dtype(dtype_adds, check_list_adds, param_name="adds")

    op_type = "scatter_nd"
    reduction = "add"
    ins = classify([var, indices, adds], op_type)
    schedules, tensors = [], []
    for var_input, indices_input, adds_input in ins:
        with tbe.compute():
            var_shape, indices_shape, adds_shape = \
                shape_util.variable_shape([var_input, indices_input, adds_input], "scatter")
            var_tensor = tvm.placeholder(var_shape, name="var", dtype=dtype_var)
            indices_tensor = tvm.placeholder(indices_shape, name="indices", dtype=dtype_indices)
            adds_tensor = tvm.placeholder(adds_shape, name="adds", dtype=dtype_adds)
            res = tbe.scatter_nd(var_tensor, indices_tensor, adds_tensor, reduction,
                                 support_out_of_bound_index=is_support_out_of_bound_index(impl_mode))
            tensors.append([var_tensor, indices_tensor, adds_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=unused-argument
@register_operator("ScatterNdAdd")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def scatter_nd_add(var,
                   indices,
                   adds,
                   var_out,
                   use_locking=False,
                   kernel_name="scatter_nd_add",
                   impl_mode=None):
    """
    scatter_nd_add interface
    Parameters
    __________
    var: input var shape, dtype and range
    indices: input indices shape, dtype and range
    adds: input adds shape, dtype and range
    var_out: output shape, dtype and range
    use_locking: bool
    kernel_name: kernel_name of scatter_add op
    impl_mode: implementation mode, such as high_precision, high_performance, support_out_of_bound_index.
    Returns
    _______
    compile info
    """
    dtype_var = var.get("dtype").lower()

    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") or dtype_var in ("bool", "int8", "uint8"):
        scatter_nd_add_tik(var, indices, adds, var_out, False, kernel_name, impl_mode)
    else:
        scatter_nd_add_dsl(var, indices, adds, var_out, False, kernel_name, impl_mode)
