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
dynamic sparse_apply_ftrl
"""
from distutils.command.config import config
from functools import reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.utils import op_tiling
from tbe.common.utils import decode


# 'pylint: disable=unrecognized-inline-option,too-many-instance-attributes
# 'pylint: disable=unused-variable,attribute-defined-outside-init,unused-argument
def ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    ceil value
    """
    return (value + factor - 1) // factor


def total_num(shape):
    """
    total_num
    """
    shape_total_num = reduce(lambda a, b: a * b, shape)
    return shape_total_num


# 'pylint: disable=invalid-name, too-many-locals, too-many-arguments, too-many-statements
# 'pylint: disable=unused-argument, too-many-instance-attributes
class SparseApplyFtrl():
    """
    Function: class that execute sparse_apply_ftrl
    """
    # one block size takes up 32b
    BLOCK_SIZE = 32
    # digit 256
    DIGIT_256 = 256
    CO_EXSIT_PART = 6
    # The 4KB space of UB is used to store indices data
    UB_INDICES_SIZE = 4 * 1024
    UB_2K_SIZE = 2 * 1024

    # paramsRow is not small than 32B and one core calc multi rows
    TILING_MODE_1 = 1
    # paramsRow is smaller than 32B
    TILING_MODE_2 = 2
    # paramsRow is not small than 32B and multi cores calc one row
    TILING_MODE_3 = 3

    TILING_ARG_NUM = 20

    # the max size of SHAPE is 2^63 - 1
    MAX_SHAPE_SIZE = 2**63 - 1

    TYPE_LEN_DICT = {"float32": 4, "int32": 4, "int64": 8}

    def __init__(self, input_dicts, output_dicts, kernel_name):
        """
        constructor of SparseApplyFtrl

        Parameters
        ----------
        input_dicts: contains var_dict, accum_dict, linear_dict, grad_dict, indices_dict
        output_dicts: contains var_out_dict, accum_out_dict, linear_out_dict
        kernel_name: kernel name, default value is "sparse_apply_ftrl"

        Returns
        -------
        None
        """
        self.kernel_name = kernel_name
        self.is_const = tbe_context.get_context().get_op_mode() == "static"
        self.input_dicts = input_dicts
        self.input_params_check(input_dicts, output_dicts)

        self.var_dtype = input_dicts[0].get("dtype").lower()
        self.indices_dtype = input_dicts[4].get("dtype").lower()
        self.tiling_dtype = "int64"

        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile, disable_debug=False)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.var_dsize = self.TYPE_LEN_DICT.get(self.var_dtype)
        self.block_elem = self.BLOCK_SIZE // self.var_dsize
        self.vector_elem = self.DIGIT_256 // self.var_dsize
        self.indices_dsize = self.TYPE_LEN_DICT.get(self.indices_dtype)
        self.block_indices = self.BLOCK_SIZE // self.indices_dsize
        self.indices_nums_once = self.UB_INDICES_SIZE // self.indices_dsize
        self.remain_size = self.ub_size - self.UB_2K_SIZE - self.UB_INDICES_SIZE
        # The remaining UB space is divided into six parts
        self.one_part_size = self.remain_size // self.CO_EXSIT_PART
        self.one_part_elem = self.one_part_size // self.var_dsize
        self.one_part_elem = self.one_part_elem // self.vector_elem * self.vector_elem

        self.var_shape = (total_num(input_dicts[0].get("shape")) if self.is_const else self.MAX_SHAPE_SIZE,)
        self.grad_shape = (total_num(input_dicts[3].get("shape")) if self.is_const else self.MAX_SHAPE_SIZE,)
        self.indices_shape = (total_num(input_dicts[4].get("shape")) if self.is_const else self.MAX_SHAPE_SIZE,)
        self.tiling_shape = (self.TILING_ARG_NUM,)
        self.block_shape = (self.block_elem,)

        self.var = self.tik_instance.Tensor(self.var_dtype, self.var_shape, name="var", scope=tik.scope_gm)
        self.accum = self.tik_instance.Tensor(self.var_dtype, self.var_shape, name="accum", scope=tik.scope_gm)
        self.linear = self.tik_instance.Tensor(self.var_dtype, self.var_shape, name="linear", scope=tik.scope_gm)
        self.grad = self.tik_instance.Tensor(self.var_dtype, self.grad_shape, name="grad", scope=tik.scope_gm)
        self.indices = self.tik_instance.Tensor(self.indices_dtype, self.indices_shape,
                                                name="indices", scope=tik.scope_gm)
        self.lr_gm = self.tik_instance.Tensor(self.var_dtype, (1,), name="lr_gm", scope=tik.scope_gm)
        self.l1_gm = self.tik_instance.Tensor(self.var_dtype, (1,), name="l1_gm", scope=tik.scope_gm)
        self.l2_gm = self.tik_instance.Tensor(self.var_dtype, (1,), name="l2_gm", scope=tik.scope_gm)
        self.lr_power_gm = self.tik_instance.Tensor(self.var_dtype, (1,), name="lr_power_gm", scope=tik.scope_gm)

        self.var_out = self.tik_instance.Tensor(self.var_dtype, shape=self.var_shape,
                                                name="var_out", scope=tik.scope_gm)
        self.accum_out = self.tik_instance.Tensor(self.var_dtype, shape=self.var_shape,
                                                  name="accum_out", scope=tik.scope_gm)
        self.linear_out = self.tik_instance.Tensor(self.var_dtype, shape=self.var_shape,
                                                   name="linear_out", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, self.tiling_shape,
                                                  name="ddr_arg", scope=tik.scope_gm)

        self.tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, self.tiling_shape,
                                             name="tiling_ub", scope=tik.scope_ubuf)
        self.lr_ub = self.tik_instance.Tensor(self.var_dtype, self.block_shape, name="lr_ub", scope=tik.scope_ubuf)
        self.l1_ub = self.tik_instance.Tensor(self.var_dtype, self.block_shape, name="l1_ub", scope=tik.scope_ubuf)
        self.l2_ub = self.tik_instance.Tensor(self.var_dtype, self.block_shape, name="l2_ub", scope=tik.scope_ubuf)
        self.lr_power_ub = self.tik_instance.Tensor(
            self.var_dtype, self.block_shape, name="lr_power_ub", scope=tik.scope_ubuf)

        self.lr = self.tik_instance.Scalar(dtype=self.var_dtype, name="lr")
        self.l1 = self.tik_instance.Scalar(dtype=self.var_dtype, name="l1")
        self.l2 = self.tik_instance.Scalar(dtype=self.var_dtype, name="l2")
        self.lr_power = self.tik_instance.Scalar(dtype=self.var_dtype, name="lr_power")
        self.lr_rec = 1.0 / self.lr
        # tiling data
        self.tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.tail_process_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tail_process_core")
        self.indices_num_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_each_core")
        self.indices_num_remaining = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_remaining")
        self.indices_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_loop_num")
        self.indices_nums_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_nums_last")
        self.var_row_elem = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="var_row_elem")
        self.var_rows = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="var_rows")
        self.indices_step = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_step")
        self.num_multi_rows = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="num_multi_rows")
        self.core_num_per_indice = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="core_num_per_indice")
        self.elems_per_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_per_core")
        self.elems_last_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_last_core")
        self.elems_per_loop = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_per_loop")
        self.elems_core_loop = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_core_loop")
        self.elems_core_remain = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_core_remain")
        self.elems_last_core_loop = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_last_core_loop")
        self.elems_last_core_remain = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="elems_last_core_remain")
        self.tail_block_offset = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tail_block_offset")

        self.var_cur_row = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="var_cur_row")
        self.core_rows_start_index = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="core_rows_start_index")
        self.core_rows_end_index = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="core_rows_end_index")
        self.cached_rows_start_index = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="cached_rows_start_index")

        self.grad_cur_row = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="grad_cur_row")
        self.var_offset = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="var_offset")
        self.grad_offset = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="grad_offset")
        self.dma_in_burst_len = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dma_in_burst_len")
        self.repeat = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="repeat")
        self.dma_out_burst_len = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dma_out_burst_len")

    def input_params_check(self, input_dicts, output_dicts):
        """
        check if the inputs are valid

        Parameters
        ----------
        input_dicts: contains var_dict, accum_dict, linear_dict, grad_dict, indices_dict
        output_dicts: contains var_out_dict, accum_out_dict, linear_out_dict

        Returns
        -------
        None
        """
        var_dtype = input_dicts[0].get("dtype").lower()
        accum_dtype = input_dicts[1].get("dtype").lower()
        linear_dtype = input_dicts[2].get("dtype").lower()
        grad_dtype = input_dicts[3].get("dtype").lower()
        indices_dtype = input_dicts[4].get("dtype").lower()
        var_out_dtype = output_dicts[0].get("dtype").lower()
        var_support_dtype_list = ("float32",)
        indices_support_dtype_list = ("int32", "int64")
        para_check.check_dtype(var_dtype, var_support_dtype_list, param_name="var")
        para_check.check_dtype(accum_dtype, var_support_dtype_list, param_name="accum")
        para_check.check_dtype(linear_dtype, var_support_dtype_list, param_name="linear")
        para_check.check_dtype(grad_dtype, var_support_dtype_list, param_name="grad")
        para_check.check_dtype(var_out_dtype, var_support_dtype_list, param_name="var_out")
        para_check.check_dtype(indices_dtype, indices_support_dtype_list, param_name="indices")

        var_shape = input_dicts[0].get("shape")
        accum_shape = input_dicts[1].get("shape")
        linear_shape = input_dicts[2].get("shape")
        grad_shape = input_dicts[3].get("shape")
        indices_shape = input_dicts[4].get("shape")
        var_out_shape = output_dicts[0].get("shape")

        # check shape
        if len(var_shape) != len(accum_shape):
            error_detail = "the shape of var and accum must be equal"
            error_manager_vector.raise_err_two_input_shape_invalid(self.kernel_name, "var", "accum", error_detail)
        if len(var_shape) != len(linear_shape):
            error_detail = "the shape of var and accum must be equal"
            error_manager_vector.raise_err_two_input_shape_invalid(self.kernel_name, "var", "linear", error_detail)
        if len(var_shape) != len(grad_shape):
            error_detail = "the shape of var and linear must be equal"
            error_manager_vector.raise_err_two_input_shape_invalid(self.kernel_name, "var", "grad", error_detail)
        if len(var_shape) != len(var_out_shape):
            error_detail = "the shape of var and var_out must be equal"
            error_manager_vector.raise_err_two_input_shape_invalid(self.kernel_name, "var", "var_out", error_detail)
        if len(indices_shape) != 1:
            error_detail = "the shape of indices must be 1"
            error_manager_vector.raise_err_input_shape_invalid(self.kernel_name, "indices", error_detail)

    def get_tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # get run tiling data
        self.tik_instance.data_move(
            self.tiling_ub, self.tiling_gm, 0, 1,
            ceil_value(self.TILING_ARG_NUM * self.TYPE_LEN_DICT.get(self.tiling_dtype), self.BLOCK_SIZE), 0, 0)
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.need_core_num.set_as(self.tiling_ub[1])
        self.tail_process_core.set_as(self.tiling_ub[2])
        self.indices_num_each_core.set_as(self.tiling_ub[3])
        self.indices_num_remaining.set_as(self.tiling_ub[4])
        self.indices_loop_num.set_as(self.tiling_ub[5])
        self.indices_nums_last.set_as(self.tiling_ub[6])
        self.var_row_elem.set_as(self.tiling_ub[7])
        self.var_rows.set_as(self.tiling_ub[8])
        self.indices_step.set_as(self.tiling_ub[9])
        self.num_multi_rows.set_as(self.tiling_ub[10])
        self.core_num_per_indice.set_as(self.tiling_ub[11])
        self.elems_per_core.set_as(self.tiling_ub[12])
        self.elems_last_core.set_as(self.tiling_ub[13])
        self.elems_per_loop.set_as(self.tiling_ub[14])
        self.elems_core_loop.set_as(self.tiling_ub[15])
        self.elems_core_remain.set_as(self.tiling_ub[16])
        self.elems_last_core_loop.set_as(self.tiling_ub[17])
        self.elems_last_core_remain.set_as(self.tiling_ub[18])
        self.tail_block_offset.set_as(self.tiling_ub[19])

    def get_const_tiling_args(self):
        compile_info = {"vars": {
            "core_num": self.core_num,
            "ub_size": self.ub_size,
            "indices_dsize": self.indices_dsize
        }}
        run_info = op_tiling.do_op_tiling("SparseApplyFtrl", compile_info, self.input_dicts, [])
        tiling_format = {
            "tiling_mode": "int64",
            "need_core_num": "int64",
            "tail_process_core": "int64",
            "indices_num_each_core": "int64",
            "indices_num_remaining": "int64",
            "indices_loop_num": "int64",
            "indices_nums_last": "int64",
            "var_row_elem": "int64",
            "var_rows": "int64",
            "indices_step": "int64",
            "num_multi_rows": "int64",
            "core_num_per_indice": "int64",
            "elems_per_core": "int64",
            "elems_last_core": "int64",
            "elems_per_loop": "int64",
            "elems_core_loop": "int64",
            "elems_core_remain": "int64",
            "elems_last_core_loop": "int64",
            "elems_last_core_remain": "int64",
            "tail_block_offset": "int64",
        }
        tiling_data = decode(run_info.get("tiling_data"), tiling_format)
        self.tiling_mode.set_as(tiling_data.get("tiling_mode"))
        self.need_core_num.set_as(tiling_data.get("need_core_num"))
        self.tail_process_core.set_as(tiling_data.get("tail_process_core"))
        self.indices_num_each_core.set_as(tiling_data.get("indices_num_each_core"))
        self.indices_num_remaining.set_as(tiling_data.get("indices_num_remaining"))
        self.indices_loop_num.set_as(tiling_data.get("indices_loop_num"))
        self.indices_nums_last.set_as(tiling_data.get("indices_nums_last"))
        self.var_row_elem.set_as(tiling_data.get("var_row_elem"))
        self.var_rows.set_as(tiling_data.get("var_rows"))
        self.indices_step.set_as(tiling_data.get("indices_step"))
        self.num_multi_rows.set_as(tiling_data.get("num_multi_rows"))
        self.core_num_per_indice.set_as(tiling_data.get("core_num_per_indice"))
        self.elems_per_core.set_as(tiling_data.get("elems_per_core"))
        self.elems_last_core.set_as(tiling_data.get("elems_last_core"))
        self.elems_per_loop.set_as(tiling_data.get("elems_per_loop"))
        self.elems_core_loop.set_as(tiling_data.get("elems_core_loop"))
        self.elems_core_remain.set_as(tiling_data.get("elems_core_remain"))
        self.elems_last_core_loop.set_as(tiling_data.get("elems_last_core_loop"))
        self.elems_last_core_remain.set_as(tiling_data.get("elems_last_core_remain"))
        self.tail_block_offset.set_as(tiling_data.get("tail_block_offset"))

    def get_input_scalar(self):
        self.tik_instance.data_move(self.lr_ub, self.lr_gm, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.l1_ub, self.l1_gm, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.l2_ub, self.l2_gm, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.lr_power_ub, self.lr_power_gm, 0, 1, 1, 0, 0)
        self.lr.set_as(self.lr_ub[0])
        self.l1.set_as(self.l1_ub[0])
        self.l2.set_as(self.l2_ub[0])
        self.lr_power.set_as(self.lr_power_ub[0])

    def compute_mode_2(self, block_id):
        """
        compute for tiling mode 2: smaller than 32B of var row elements

        Parameters
        ----------
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (self.indices_nums_once,),
                                         name="indices_ub",
                                         scope=tik.scope_ubuf)
        var_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="var_ub", scope=tik.scope_ubuf)
        accum_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="accum_ub", scope=tik.scope_ubuf)
        linear_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="linear_ub", scope=tik.scope_ubuf)
        grad_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="grad_ub", scope=tik.scope_ubuf)
        tmp_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="tmp_ub", scope=tik.scope_ubuf)
        tmp2_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="tmp2_ub", scope=tik.scope_ubuf)
        ub_tuples = (var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub)

        var_ub_block = tik_instance.Tensor(self.var_dtype, self.block_shape, name="var_ub_block", scope=tik.scope_ubuf)
        accum_ub_block = tik_instance.Tensor(self.var_dtype,
                                             self.block_shape,
                                             name="accum_ub_block",
                                             scope=tik.scope_ubuf)
        linear_ub_block = tik_instance.Tensor(self.var_dtype,
                                              self.block_shape,
                                              name="linear_ub_block",
                                              scope=tik.scope_ubuf)
        grad_ub_block = tik_instance.Tensor(self.var_dtype,
                                            self.block_shape,
                                            name="grad_ub_block",
                                            scope=tik.scope_ubuf)
        ub_block_tuples = (var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block)

        self.cached_rows_start_index.set_as(self.var_rows)

        self.core_rows_start_index.set_as(self.indices_step * block_id)
        with self.tik_instance.if_scope(block_id < self.need_core_num - 1):
            self.core_rows_end_index.set_as(self.indices_step * (block_id + 1))
        with self.tik_instance.else_scope():
            self.core_rows_end_index.set_as(self.var_rows)

        # process indices_num_each_core: indices_nums_once * indices_loop_num + indices_nums_last
        burst_len_indices = ceil_value(self.indices_nums_once, self.block_indices)
        burst_len_grad = ceil_value(self.indices_nums_once * self.var_row_elem, self.block_elem)
        burst_len_multi_row = ceil_value(self.num_multi_rows * self.var_row_elem, self.block_elem)
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = indices_loop_i * self.indices_nums_once
            # move indices and grad data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, burst_len_indices, 0, 0)
            tik_instance.data_move(grad_ub, self.grad[indices_num_offset * self.var_row_elem], 0, 1, burst_len_grad, 0,
                                   0)

            self.calc_multi_indices(indices_ub, self.indices_nums_once, burst_len_multi_row, ub_tuples, ub_block_tuples)

        with tik_instance.if_scope(self.indices_nums_last > 0):
            indices_num_offset = self.indices_loop_num * self.indices_nums_once
            burst_len_indices = ceil_value(self.indices_nums_last, self.block_indices)
            burst_len_grad = ceil_value(self.indices_nums_last * self.var_row_elem, self.block_elem)
            # move indices and grad data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, burst_len_indices, 0, 0)
            tik_instance.data_move(grad_ub, self.grad[indices_num_offset * self.var_row_elem], 0, 1, burst_len_grad, 0,
                                   0)

            self.calc_multi_indices(indices_ub, self.indices_nums_last, burst_len_multi_row, ub_tuples, ub_block_tuples)

    # 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,huawei-too-many-arguments
    def calc_multi_indices(self, indices_ub, indices_num, burst_len_multi_row, ub_tuples, ub_block_tuples):
        """
        calculate multi rows, multi rows will read at one to avoid loading
        little data from gm to ubuf at a high frequency

        Parameters
        ----------
        indices_ub: indices_ub
        indices_num: how many indices to calculate
        burst_len_multi_row: burst length of multi row
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        ub_block_tuples: contains var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        with tik_instance.for_range(0, indices_num) as indices_i:
            self.var_cur_row.set_as(indices_ub[indices_i])

            # check whether current indices is within the processing range of the core
            with tik_instance.if_scope(
                    tik.all(self.var_cur_row >= self.core_rows_start_index,
                            self.var_cur_row < self.core_rows_end_index)):
                # check whether the var, accum, linear corresponding to current indices is cached in the UB
                with tik_instance.if_scope(
                        tik.all(self.var_cur_row >= self.cached_rows_start_index,
                                self.var_cur_row < self.cached_rows_start_index + self.num_multi_rows)):
                    self.calc_a_small_row(indices_i, ub_tuples, ub_block_tuples)
                with tik_instance.else_scope():
                    with tik_instance.if_scope(self.cached_rows_start_index < self.var_rows):
                        self.save_multi_rows(ub_tuples, burst_len_multi_row)
                    self.load_multi_rows(ub_tuples, burst_len_multi_row)
                    self.calc_a_small_row(indices_i, ub_tuples, ub_block_tuples)
        with tik_instance.if_scope(self.cached_rows_start_index < self.var_rows):
            self.save_multi_rows(ub_tuples, burst_len_multi_row)

    def calc_a_small_row(self, grad_idx, ub_tuples, ub_block_tuples):
        """
        calc a small whole row

        Parameters
        ----------
        grad_idx: row index of grad_ub
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        ub_block_tuples: contains var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block

        Returns
        -------
        None
        """
        var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub = ub_tuples
        var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block = ub_block_tuples
        offset = self.var_cur_row - self.cached_rows_start_index

        with self.tik_instance.for_range(0, self.var_row_elem) as i:
            var_ub_block[i].set_as(var_ub[offset * self.var_row_elem + i])
            accum_ub_block[i].set_as(accum_ub[offset * self.var_row_elem + i])
            linear_ub_block[i].set_as(linear_ub[offset * self.var_row_elem + i])
            grad_ub_block[i].set_as(grad_ub[grad_idx * self.var_row_elem + i])

        calc_tuples = (var_ub_block, accum_ub_block, linear_ub_block, grad_ub_block, tmp_ub, tmp2_ub)
        self.sparse_calc(calc_tuples, 0, self.var_row_elem, 1)

        with self.tik_instance.for_range(0, self.var_row_elem) as i:
            var_ub[offset * self.var_row_elem + i].set_as(var_ub_block[i])
            accum_ub[offset * self.var_row_elem + i].set_as(accum_ub_block[i])
            linear_ub[offset * self.var_row_elem + i].set_as(linear_ub_block[i])

    def load_multi_rows(self, ub_tuples, burst_len_multi_row):
        """
        load multi rows of var, accum and linear from gm to ub

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        burst_len_multi_row: burst length of multi row

        Returns
        -------
        None
        """
        var_ub, accum_ub, linear_ub = ub_tuples[:3]
        with self.tik_instance.if_scope(self.var_cur_row + self.num_multi_rows <= self.core_rows_end_index):
            self.cached_rows_start_index.set_as(self.var_cur_row)
        with self.tik_instance.else_scope():
            self.cached_rows_start_index.set_as(self.core_rows_end_index - self.num_multi_rows)

        self.tik_instance.data_move(var_ub, self.var[self.cached_rows_start_index * self.var_row_elem], 0, 1,
                                    burst_len_multi_row, 0, 0)
        self.tik_instance.data_move(accum_ub, self.accum[self.cached_rows_start_index * self.var_row_elem], 0, 1,
                                    burst_len_multi_row, 0, 0)
        self.tik_instance.data_move(linear_ub, self.linear[self.cached_rows_start_index * self.var_row_elem], 0, 1,
                                    burst_len_multi_row, 0, 0)

    def save_multi_rows(self, ub_tuples, burst_len_multi_row):
        """
        save multi rows var, accum and linear to gm

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        burst_len_multi_row: burst length of multi row

        Returns
        -------
        None
        """
        var_ub, accum_ub, linear_ub = ub_tuples[:3]
        self.tik_instance.data_move(self.var[self.cached_rows_start_index * self.var_row_elem], var_ub, 0, 1,
                                    burst_len_multi_row, 0, 0)
        self.tik_instance.data_move(self.accum[self.cached_rows_start_index * self.var_row_elem], accum_ub, 0, 1,
                                    burst_len_multi_row, 0, 0)
        self.tik_instance.data_move(self.linear[self.cached_rows_start_index * self.var_row_elem], linear_ub, 0, 1,
                                    burst_len_multi_row, 0, 0)
        self.cached_rows_start_index.set_as(self.var_rows)

    def compute_mode_3(self, block_id):
        """
        var_row_elem is not smaller than 32B and multi cores calc one row

        Parameters
        ----------
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (self.indices_nums_once,),
                                         name="indices_ub",
                                         scope=tik.scope_ubuf)
        var_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="var_ub", scope=tik.scope_ubuf)
        accum_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="accum_ub", scope=tik.scope_ubuf)
        linear_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="linear_ub", scope=tik.scope_ubuf)
        grad_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="grad_ub", scope=tik.scope_ubuf)
        tmp_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="tmp_ub", scope=tik.scope_ubuf)
        tmp2_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="tmp2_ub", scope=tik.scope_ubuf)
        ub_tuples = (var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub)

        burst_len_indices = ceil_value(self.indices_nums_once, self.block_indices)
        # move indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices, 0, 1, burst_len_indices, 0, 0)

        self.grad_cur_row.set_as(block_id / self.core_num_per_indice)
        self.var_cur_row.set_as(indices_ub[self.grad_cur_row])
        self.calc_core_partial(self.var_cur_row, self.grad_cur_row, block_id, ub_tuples)

    def calc_core_partial(self, var_id, grad_id, block_id, ub_tuples):
        """
        calculate partial of a row by cores

        Parameters
        ----------
        var_id: row index
        grad_id: grad index
        block_id: core index
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        core_start_offset = (block_id - grad_id * self.core_num_per_indice) * self.elems_per_core
        self.var_offset.set_as(self.var_cur_row * self.var_row_elem + core_start_offset)
        self.grad_offset.set_as(self.grad_cur_row * self.var_row_elem + core_start_offset)

        with tik_instance.if_scope(block_id == (grad_id + 1) * self.core_num_per_indice - 1):
            self.calc_last_core(ub_tuples)

        with tik_instance.else_scope():
            self.dma_in_burst_len.set_as(self.elems_per_loop // self.block_elem)
            self.dma_out_burst_len.set_as(self.elems_per_loop // self.block_elem)
            self.repeat.set_as(self.elems_per_loop // self.vector_elem)
            with tik_instance.for_range(0, self.elems_core_loop) as elem_loop_i:
                self.calc_one_loop(ub_tuples, self.dma_in_burst_len, self.repeat, self.dma_out_burst_len)
                self.var_offset.set_as(self.var_offset + self.elems_per_loop)
                self.grad_offset.set_as(self.grad_offset + self.elems_per_loop)

    def compute_mode_1(self, block_id):
        """
        var_row_elem is not small than 32B and one core calc multi rows

        Parameters
        ----------
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (self.indices_nums_once,),
                                         name="indices_ub",
                                         scope=tik.scope_ubuf)
        var_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="var_ub", scope=tik.scope_ubuf)
        accum_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="accum_ub", scope=tik.scope_ubuf)
        linear_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="linear_ub", scope=tik.scope_ubuf)
        grad_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="grad_ub", scope=tik.scope_ubuf)
        tmp_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="tmp_ub", scope=tik.scope_ubuf)
        tmp2_ub = tik_instance.Tensor(self.var_dtype, (self.one_part_elem,), name="tmp2_ub", scope=tik.scope_ubuf)
        ub_tuples = (var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub)

        # process indices_num_each_core: indices_nums_once * indices_loop_num + indices_nums_last
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_nums_once
            self.process_num_indices(ub_tuples, indices_ub, self.indices_nums_once, indices_num_offset)

        with tik_instance.if_scope(self.indices_nums_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                 self.indices_loop_num * self.indices_nums_once
            self.process_num_indices(ub_tuples, indices_ub, self.indices_nums_last, indices_num_offset)

        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.indices_num_remaining)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id
            self.process_num_indices(ub_tuples, indices_ub, 1, indices_num_offset)

    def process_num_indices(self, ub_tuples, indices_ub, indices_num, indices_num_offset):
        """
        process num indices

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        indices_ub: indices ub
        indices_num: the number of indices
        indices_num_offset: the offset of indices in gm

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # move indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                               ceil_value(indices_num, self.block_indices), 0, 0)

        with tik_instance.for_range(0, indices_num) as indices_i:
            self.var_cur_row.set_as(indices_ub[indices_i])

            self.var_offset.set_as(self.var_cur_row * self.var_row_elem)
            self.grad_offset.set_as((indices_num_offset + indices_i) * self.var_row_elem)

            self.calc_last_core(ub_tuples)

    def calc_last_core(self, ub_tuples):
        var_ub, accum_ub, linear_ub, grad_ub = ub_tuples[0:4]

        self.dma_in_burst_len.set_as(self.elems_per_loop // self.block_elem)
        self.dma_out_burst_len.set_as(self.elems_per_loop // self.block_elem)
        self.repeat.set_as(self.elems_per_loop // self.vector_elem)
        with self.tik_instance.for_range(0, self.elems_last_core_loop) as elem_loop_i:
            self.calc_one_loop(ub_tuples, self.dma_in_burst_len, self.repeat, self.dma_out_burst_len)
            self.var_offset.set_as(self.var_offset + self.elems_per_loop)
            self.grad_offset.set_as(self.grad_offset + self.elems_per_loop)

        with self.tik_instance.if_scope(self.elems_last_core_remain > 0):
            self.dma_in_burst_len.set_as((self.elems_last_core_remain + self.block_elem - 1) // self.block_elem)
            self.dma_out_burst_len.set_as((self.elems_last_core_remain + self.block_elem - 1) // self.block_elem)
            self.repeat.set_as((self.elems_last_core_remain + self.vector_elem - 1) // self.vector_elem)
            with self.tik_instance.if_scope(self.tail_block_offset > 0):
                self.dma_out_burst_len.set_as(self.dma_out_burst_len - 1)
            self.calc_one_loop(ub_tuples, self.dma_in_burst_len, self.repeat, self.dma_out_burst_len)

        with self.tik_instance.if_scope(self.tail_block_offset > 0):
            for i in range(self.block_elem):
                var_ub[i].set_as(var_ub[self.tail_block_offset + i])
                accum_ub[i].set_as(accum_ub[self.tail_block_offset + i])
                linear_ub[i].set_as(linear_ub[self.tail_block_offset + i])
            self.tik_instance.data_move(self.var_out[self.var_offset + self.tail_block_offset], var_ub, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.accum_out[self.var_offset + self.tail_block_offset],
                                        accum_ub, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.linear_out[self.var_offset + self.tail_block_offset],
                                        linear_ub, 0, 1, 1, 0, 0)

    def calc_one_loop(self, ub_tuples, dma_in_burst_len, repeat, dma_out_burst_len):
        var_ub, accum_ub, linear_ub, grad_ub = ub_tuples[0:4]

        self.tik_instance.data_move(var_ub, self.var[self.var_offset], 0, 1, dma_in_burst_len, 0, 0)
        self.tik_instance.data_move(accum_ub, self.accum[self.var_offset], 0, 1, dma_in_burst_len, 0, 0)
        self.tik_instance.data_move(linear_ub, self.linear[self.var_offset], 0, 1, dma_in_burst_len, 0, 0)
        self.tik_instance.data_move(grad_ub, self.grad[self.grad_offset], 0, 1, dma_in_burst_len, 0, 0)

        self.sparse_calc(ub_tuples, 0, self.vector_elem, repeat)

        self.tik_instance.data_move(self.var_out[self.var_offset], var_ub, 0, 1, dma_out_burst_len, 0, 0)
        self.tik_instance.data_move(self.accum_out[self.var_offset], accum_ub, 0, 1, dma_out_burst_len, 0, 0)
        self.tik_instance.data_move(self.linear_out[self.var_offset], linear_ub, 0, 1, dma_out_burst_len, 0, 0)

    def sparse_calc(self, ub_tuples, offset, mask, repeat):
        """
        calculate data according to the Ftrl-proximal scheme

        Parameters
        ----------
        ub_tuples: contains var_ub, accum_ub, linear_ub, grad_ub, tmp_ub, tmp2_ub
        offset: offset of var_row_elem
        mask: effective operation on element
        repeat: repeated iterations times

        Returns
        -------
        None
        """
        var_ub = ub_tuples[0][offset]
        accum_ub = ub_tuples[1][offset]
        linear_ub = ub_tuples[2][offset]
        grad_ub = ub_tuples[3][offset]
        tmp_ub = ub_tuples[4][offset]
        tmp2_ub = ub_tuples[5][offset]

        self.tik_instance.vmul(mask, tmp_ub, grad_ub, grad_ub, repeat, 1, 1, 1, 8, 8, 8)
        # linear += grad, grad will not used after this operation
        self.tik_instance.vadd(mask, linear_ub, grad_ub, linear_ub, repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vln(mask, grad_ub, accum_ub, repeat, 1, 1, 8, 8)

        self.tik_instance.vmuls(mask, grad_ub, grad_ub, -self.lr_power, repeat, 1, 1, 8, 8)

        self.tik_instance.vexp(mask, grad_ub, grad_ub, repeat, 1, 1, 8, 8)

        self.tik_instance.vadd(mask, accum_ub, accum_ub, tmp_ub, repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vln(mask, tmp_ub, accum_ub, repeat, 1, 1, 8, 8)

        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, -self.lr_power, repeat, 1, 1, 8, 8)

        self.tik_instance.vexp(mask, tmp_ub, tmp_ub, repeat, 1, 1, 8, 8)

        self.tik_instance.vmuls(mask, tmp2_ub, tmp_ub, self.lr_rec, repeat, 1, 1, 8, 8)

        self.tik_instance.vsub(mask, tmp_ub, grad_ub, tmp_ub, repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, self.lr_rec, repeat, 1, 1, 8, 8)

        self.tik_instance.vmul(mask, tmp_ub, tmp_ub, var_ub, repeat, 1, 1, 1, 8, 8, 8)

        # linear out
        self.tik_instance.vadd(mask, linear_ub, tmp_ub, linear_ub, repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vector_dup(mask, tmp_ub, self.l1, repeat, 1, 8)
        self.tik_instance.vmin(mask, grad_ub, linear_ub, tmp_ub, repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vector_dup(mask, tmp_ub, -self.l1, repeat, 1, 8)
        self.tik_instance.vmax(mask, tmp_ub, grad_ub, tmp_ub, repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, tmp_ub, tmp_ub, linear_ub, repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadds(mask, tmp2_ub, tmp2_ub, 2 * self.l2, repeat, 1, 1, 8, 8)

        self.tik_instance.vdiv(mask, var_ub, tmp_ub, tmp2_ub, repeat, 1, 1, 1, 8, 8, 8)

    def sparse_apply_ftrl_compute_tiling(self):
        """
        Main process of sparse_apply_ftrl

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        with tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            if self.is_const:
                self.get_const_tiling_args()
            else:
                self.get_tiling_args()
            self.get_input_scalar()

            with tik_instance.if_scope(block_id < self.need_core_num):
                # self.TILING_MODE_1: var_row_elem is not small than 32B and one core calc multi rows
                with tik_instance.if_scope(self.tiling_mode == self.TILING_MODE_1):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_1(block_id)
                with tik_instance.else_scope():
                    # self.TILING_MODE_2: var_row_elem is smaller than 32B
                    with tik_instance.if_scope(self.tiling_mode == self.TILING_MODE_2):
                        with tik_instance.new_stmt_scope():
                            self.compute_mode_2(block_id)
                    # self.TILING_MODE_3: var_row_elem is not smaller than 32B and multi cores calc one row
                    with tik_instance.else_scope():
                        with tik_instance.new_stmt_scope():
                            self.compute_mode_3(block_id)


    def sparse_apply_ftrl_compute(self):
        if self.is_const:
            self.sparse_apply_ftrl_compute_const()
        else:
            self.sparse_apply_ftrl_compute_dynamic()


    def sparse_apply_ftrl_compute_const(self):
        self.sparse_apply_ftrl_compute_tiling()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var, self.accum, self.linear, self.grad, self.indices,
                                           self.lr_gm, self.l1_gm, self.l2_gm, self.lr_power_gm),
                                   outputs=(self.var_out, self.accum_out, self.linear_out),
                                   config={"enable_const_fold": True})

    def sparse_apply_ftrl_compute_dynamic(self):
        """
        compute of sparse_apply_ftrl

        Parameters
        ----------
        None

        Returns
        -------
        compile info
        """
        self.sparse_apply_ftrl_compute_tiling()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var, self.accum, self.linear, self.grad, self.indices,
                                           self.lr_gm, self.l1_gm, self.l2_gm, self.lr_power_gm),
                                   outputs=(self.var_out, self.accum_out, self.linear_out),
                                   flowtable=(self.tiling_gm,),
                                   enable_l2=True)

        # add compile info
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.core_num,
            "ub_size": self.ub_size,
            "indices_dsize": self.indices_dsize
        })


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,huawei-too-many-arguments
@register_operator("SparseApplyFtrl")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sparse_apply_ftrl(var_dict,
                      accum_dict,
                      linear_dict,
                      grad_dict,
                      indices_dict,
                      lr,
                      l1,
                      l2,
                      lr_power,
                      var_out_dict,
                      accum_out_dict,
                      linear_out_dict,
                      use_locking=False,
                      kernel_name="sparse_apply_ftrl"):
    """
    sparse_apply_ftrl interface, update the variable referenced by resource.

    Parameters
    ----------
    var_dict: data of input var, only support float32
    accum_dict: data of input accum, only support float32
    linear_dict: data of input linear, only support float32
    grad_dict: data of input grad, only support float32
    indices_dict: data of input indices, only support int32
    lr: data of input lr, only supports support float32
    l1: data of input l1, only supports support float32
    l2: data of input l2, only supports support float32
    lr_power: data of input lr_power, only support float32
    var_out_dict: data of input var, only support float32
    accum_out_dict: data of input accum, only support float32
    linear_out_dict: data of input linear, only support float32
    use_locking: bool, not used
    kernel_name: str, kernel name, default value is sparse_apply_ftrl

    Returns
    -------
    None
    """
    input_dicts = (var_dict, accum_dict, linear_dict, grad_dict, indices_dict)
    output_dicts = (var_out_dict, accum_out_dict, linear_out_dict)
    obj = SparseApplyFtrl(input_dicts, output_dicts, kernel_name)

    obj.sparse_apply_ftrl_compute()
