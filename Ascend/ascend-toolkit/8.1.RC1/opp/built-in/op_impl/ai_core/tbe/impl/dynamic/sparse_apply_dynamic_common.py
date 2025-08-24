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
sparse_apply_dynamic_common
"""
import functools
from tbe import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util import util_common
from tbe.common.platform import get_bit_len


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    MAX_INT32 = 2 ** 31 - 1
    GM_ALLOC_SHAPE = (MAX_INT32,)
    TILING_SHAPE = (16,)
    UB_2K_SIZE = 2 * 1024


def _get_dtype_byte(dtype):
    return get_bit_len(dtype) // 8


def _exec_front_last_diff(tik_instance, num, part, fun, fun_tail):
    front = num // part
    last = num - front * part
    with tik_instance.for_range(0, front, name="front") as i:
        fun(i * part, part)
    with tik_instance.if_scope(last > 0):
        fun_tail(front * part, last)


def _exec_front_last_diff_scalar(tik_instance, num, part, fun, fun_tail):
    front = num // part
    last = num - front * part
    part_scalar = tik_instance.Scalar(dtype="int64", name="part")
    part_scalar.set_as(part)
    with tik_instance.for_range(0, front, name="front") as i:
        fun(i * part, part_scalar)
    with tik_instance.if_scope(last > 0):
        fun_tail(front * part, last)


def _exec_front_last(tik_instance, num, part, fun):
    _exec_front_last_diff(tik_instance, num, part, fun, fun)


def _isinstance_if(tik_instance, condition, fun):
    if isinstance(condition, int):
        if condition:
            fun()
    else:
        with tik_instance.if_scope(condition):
            fun()


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-instance-attributes,unnecessary-lambda,consider-using-enumerate
class SparseApplyDynamic:
    """
    Base Class for sparse apply op
    For specific sparse apply op, such as sparse_apply_ftrl, need to inherit
    this class and implement calc function on their own.
    """

    # 'pylint: disable=too-many-statements
    def __init__(self, var, grad, indices, kernel_name):
        """
        Init sparse_apply  base parameters

        Parameters
        ----------
        grad: dict
            data of grad
            datatype supports float32
        indices: dict
            data of indices
            datatype supports int32 and int64
        kernel_name: str
            the name of the operator

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()

        self.grad_shape = grad.get("shape")
        self.grad_dtype = grad.get("dtype").lower()

        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()

        self.indices_shape = indices.get("shape")
        self.indices_dtype = indices.get("dtype").lower()

        self.scalar_shape = (1,)
        self.kernel_name = kernel_name

        self.indices_dtype_bytes_size = get_bit_len(self.indices_dtype) // 8
        self.grad_dtype_bytes_size = get_bit_len(self.grad_dtype) // 8

        self.grad_each_block = 32 // self.grad_dtype_bytes_size
        self.indices_data_each_block = 32 // self.indices_dtype_bytes_size

        # Reserved 1024 Bytes for inputs and outputs 32B alignment
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 1024)

        one_block_bytes_size = tbe_platform.VECTOR_INST_BLOCK_WIDTH // tbe_platform.VECTOR_INST_BLOCK_NUM
        self.block_len = one_block_bytes_size // self.grad_dtype_bytes_size
        self.ub_take_parts = 1

        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype,
                                                   Constant.GM_ALLOC_SHAPE,
                                                   name="indices_gm",
                                                   scope=tik.scope_gm)
        self.grad_gm = self.tik_instance.Tensor(self.grad_dtype,
                                                Constant.GM_ALLOC_SHAPE,
                                                name="grad_gm",
                                                scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int32", Constant.TILING_SHAPE, name="tiling_gm", scope=tik.scope_gm)

        self.rows = self.indices_shape[0]
        if len(self.grad_shape) > 1:
            self.each_row_data_num = functools.reduce(lambda x, y: x * y, self.grad_shape[1:])
        else:
            self.each_row_data_num = 1

        self.input_tensor = []
        self.input_scalar_gm = []
        self.output = []
        self.tail_ub = []
        self.ub = []
        self.ub_reserved = []
        self.scalar_ub_reserved = []
        self.align_ub = []
        self.align_ub_info = []
        self.tensor_map = {}
        self.scalar_gm_map = {}
        self.cur_index = self.tik_instance.Scalar(self.indices_dtype)
        self.num_one_repeat = tbe_platform.VECTOR_INST_BLOCK_NUM * self.block_len

        self.reg_row_start = self.tik_instance.Scalar(self.indices_dtype)
        self.reg_cur_row = self.tik_instance.Scalar(self.indices_dtype)
        self.reg_core_last_rows = self.tik_instance.Scalar(self.indices_dtype)
        self.reg_row_start_core = self.tik_instance.Scalar(self.indices_dtype)
        self.var_rows = self.tik_instance.Scalar("int32")
        self.var_ub_shape = None
        self.indices_ub_shape = None
        self.cols_per_part = None
        self.num_indices_per_batch = None
        self.cache_threshold_col = None
        self.num_multi_rows = self.tik_instance.Scalar("int32")
        self.partial_factor = None
        self.cols_per_core = self.tik_instance.Scalar("int32")
        self.cols_last_core = self.tik_instance.Scalar("int32")
        self.indices_step = None
        self.need_core_num = self.tik_instance.Scalar("int32")
        self.indices_ub_number = None
        self.grad_ub = None
        self.indices_ub = None
        self.grad_align_ub = None
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.num_indices = self.tik_instance.Scalar(dtype="int32", name="num_indices")
        self.each_row_data_num = self.tik_instance.Scalar(dtype="int32", name="each_row_data_num")
        self.cache_threshold_col = self.block_len - 1
        self.remain_size = None
        self.one_part_size = None
        self.tiling_ub = None

    def _get_tiling_const(self):
        ub_indices_size = 4 * 1024
        self.remain_size = self.ub_size_bytes - Constant.UB_2K_SIZE - ub_indices_size
        self.one_part_size = self.remain_size // self.ub_take_parts
        self.cols_per_part = self.one_part_size // self.indices_dtype_bytes_size
        vector_eles = tbe_platform.VECTOR_INST_BLOCK_WIDTH // _get_dtype_byte("float32")
        self.cols_per_part = self.cols_per_part // vector_eles * vector_eles

        self.var_ub_shape = (self.cols_per_part,)
        self.indices_ub_shape = (ub_indices_size // self.indices_dtype_bytes_size,)
        self.indices_ub_number = ub_indices_size // self.indices_dtype_bytes_size

    def _get_tiling_args(self):
        self.tiling_ub = self.tik_instance.Tensor("int32",
                                                  Constant.TILING_SHAPE,
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)

        self._get_tiling_const()
        self.var_rows.set_as(self.tiling_ub[0])
        self.num_indices.set_as(self.tiling_ub[1])
        self.each_row_data_num.set_as(self.tiling_ub[2])
        self.need_core_num.set_as(self.tiling_ub[3])
        self.num_multi_rows.set_as(self.tiling_ub[4])
        self.cols_per_core.set_as(self.tiling_ub[5])
        self.cols_last_core.set_as(self.tiling_ub[6])

        self.reg_row_start.set_as(self.var_rows + 1)
        self.indices_step = self.var_rows // self.need_core_num
        self.partial_factor = self.core_num // self.num_indices

    def add_input(self, name, dtype):
        """
        called by external, describe the info of inputs excepts indices and grad every input will alloc a tik gm tensor,
        and passed to BuildCCE func with grad and indices as inputs

        Parameters
        ----------
        name: string type, name of the input
        dtype: type of the input

        Returns
        -------
        None
        """
        tensor = self.tik_instance.Tensor(dtype, Constant.GM_ALLOC_SHAPE, name=name, scope=tik.scope_gm)
        self.input_tensor.append(tensor)
        self.tensor_map[name] = tensor

    def allocate_scalar_gm(self, name, dtype):
        """
        allocate memory in gm for scalar

        Parameters
        ----------
        name: string type, name of the input_scalar
        dtype: type of the input_scalar

        Returns
        -------
        None
        """
        scalar_gm = self.tik_instance.Tensor(dtype, self.scalar_shape, name=name, scope=tik.scope_gm)
        self.input_scalar_gm.append(scalar_gm)
        self.scalar_gm_map[name] = scalar_gm

    def add_output(self, name, dtype):
        """
        called by external, describe the info of outputs
        every output will alloc a tik gm tensor, and passed to BuildCCE func as outputs

        Parameters
        ----------
        name: string type, name of the input
        dtype: type of the input

        Returns
        -------
        None
        """
        tensor = self.tik_instance.Tensor(dtype, Constant.GM_ALLOC_SHAPE, name=name, scope=tik.scope_gm)
        self.output.append(tensor)
        self.tensor_map[name] = tensor

        tail_tensor = self.tik_instance.Tensor(dtype, (self.block_len,), name=name + "_tail_ub", scope=tik.scope_ubuf)
        self.tail_ub.append(tail_tensor)

    def reserve_ub(self, name, dtype, align_name=None, is_scalar=False):
        """
        called by external, to reserve a ubuf space
        every reservation has equal ubuf sapce

        Parameters
        ----------
        name: string type, name of the input
        dtype: type of the input
        align_name: bool type, if True, will alloc a extra small ubuf space

        Returns
        -------
        None
        """
        if is_scalar:
            self.scalar_ub_reserved.append((name, dtype))
        else:
            self.ub_reserved.append((name, dtype))
        if align_name:
            self.align_ub_info.append((align_name, dtype))
        self.ub_take_parts += 1

    def _get_ub(self, name):
        """
        called by calc fun, get ubuf addr

        Parameters
        ----------
        name: string type, name of the ubuf

        Returns
        -------
        Tensor
        """
        return self.tensor_map.get(name)

    def _get_scalar_gm(self, name):
        """
        called by calc fun, get scalar ubuf addr

        Parameters
        ----------
        name: string type, name of the ubuf

        Returns
        -------
        Tensor
        """
        return self.scalar_gm_map.get(name)

    def _alloc_ub(self):
        """
        alloc ub tensor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.grad_ub = self.tik_instance.Tensor(self.grad_dtype,
                                                self.var_ub_shape,
                                                name="grad_ub",
                                                scope=tik.scope_ubuf)

        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype,
                                                   self.indices_ub_shape,
                                                   name="indices_ub",
                                                   scope=tik.scope_ubuf)

        self.grad_align_ub = self.tik_instance.Tensor(self.grad_dtype, (self.block_len,),
                                                      name="grad_align_ub",
                                                      scope=tik.scope_ubuf)

        for name, dtype in self.ub_reserved:
            tensor = self.tik_instance.Tensor(dtype, self.var_ub_shape, name=name, scope=tik.scope_ubuf)
            self.ub.append(tensor)
            self.tensor_map[name] = tensor

        if self.scalar_ub_reserved:
            for name, dtype in self.scalar_ub_reserved:
                tensor = self.tik_instance.Tensor(dtype, self.scalar_shape, name=name, scope=tik.scope_ubuf)
                self.tensor_map[name] = tensor

        for name, dtype in self.align_ub_info:
            tensor = self.tik_instance.Tensor(dtype, (self.block_len,), name=name, scope=tik.scope_ubuf)
            self.align_ub.append(tensor)
            self.tensor_map[name] = tensor

    def _load_row_part(self, var_idx, grad_idx, offset, cnt):
        """
        load a row or part of a row if row is too long

        Parameters
        ----------
        var_idx: row index of input on global
        grad_idx: row index of grad on global
        offset: offset of this part
        cnt: num elements of this part

        Returns
        -------
        None
        """
        burst_len = util_common.ceil(cnt, self.grad_each_block)
        for i in range(len(self.input_tensor)):
            self.tik_instance.data_move(self.ub[i], self.input_tensor[i][var_idx * self.each_row_data_num + offset], 0,
                                        1, burst_len, 0, 0)

        self.tik_instance.data_move(self.grad_ub, self.grad_gm[grad_idx * self.each_row_data_num + offset], 0, 1,
                                    burst_len, 0, 0)

    def _save_row_part(self, var_idx, offset, cnt):
        """
        save a row or part of a row if row is too long

        Parameters
        ----------
        var_idx: row index of input on global
        offset: offset of this part
        cnt: num elements of a this part

        Returns
        -------
        None
        """
        burst_len = util_common.ceil(cnt, self.grad_each_block)
        for i in range(len(self.output)):
            self.tik_instance.data_move(self.output[i][var_idx * self.each_row_data_num + offset], self.ub[i], 0, 1,
                                        burst_len, 0, 0)

    def _save_row_part_safely(self, var_idx, offset, cnt):
        """
        save a row safely, if is last part, need to be safely written, because last part by not be 32B aligned and will
        over write the next row

        Parameters
        ----------
        var_idx: row index of input on global
        offset: offset of this part
        cnt: num elements of a this part

        Returns
        -------
        None
        """
        burst_len = cnt // self.grad_each_block
        with self.tik_instance.if_scope(burst_len > 0):
            for i in range(len(self.output)):
                self.tik_instance.data_move(self.output[i][var_idx * self.each_row_data_num + offset], self.ub[i], 0, 1,
                                            burst_len, 0, 0)
        with self.tik_instance.if_scope(cnt % self.grad_each_block != 0):
            for i in range(len(self.output)):
                with self.tik_instance.for_range(0, self.block_len) as j:
                    self.tail_ub[i][j] = self.ub[i][cnt - self.block_len + j]
                self.tik_instance.data_move(
                    self.output[i][var_idx * self.each_row_data_num + offset + cnt - self.block_len], self.tail_ub[i],
                    0, 1, 1, 0, 0)

    def _load_indices(self, start, cnt):
        """
        load indices form gm to ubuf

        Parameters
        ----------
        start: offset on gm
        cnt: how many indices to load

        Returns
        -------
        None
        """
        burst_len = util_common.ceil(cnt, self.indices_data_each_block)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[start], 0, 1, burst_len, 0, 0)

    @staticmethod
    def _calculate(repeat_times, mask, offset):
        """
        remain for sub calss to implement their own calculation logic

        Parameters
        ----------
        repeat_times: repeat count pass to tik instruction call
        mask: mask pass to tik instruction call
        offset: offset of ubuf

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def _calc_part(self, part_size):
        """
        calc a row or a part of a row if row is too long

        Parameters
        ----------
        part_size: num elements of a part

        Returns
        -------
        None
        """
        num_one_repeat = self.num_one_repeat
        repeat_255 = part_size // (255 * num_one_repeat)
        repeat = (part_size - repeat_255 * (255 * num_one_repeat)) // num_one_repeat
        remain = part_size % num_one_repeat

        with self.tik_instance.for_range(0, repeat_255) as i:
            self._calculate(255, num_one_repeat, i * 255 * num_one_repeat)

        _isinstance_if(self.tik_instance, repeat > 0,
                       lambda: self._calculate(repeat, num_one_repeat, repeat_255 * 255 * num_one_repeat))

        _isinstance_if(self.tik_instance, remain > 0,
                       lambda: self._calculate(1, remain, repeat_255 * 255 * num_one_repeat + repeat * num_one_repeat))

    def _calc_a_indices(self, var_idx, grad_idx):
        """
        calc a whole row, if a row a too long, will divide multi part to load, calculate and save.

        Parameters
        ----------
        var_idx: row index of input on global
        grad_idx: row index of grad on global

        Returns
        -------
        None
        """

        def _do_calc(offset, part_len):
            self._load_row_part(var_idx, grad_idx, offset, part_len)
            self._calc_part(part_len)
            self._save_row_part(var_idx, offset, part_len)

        def _do_calc_tail(offset, part_len):
            self._load_row_part(var_idx, grad_idx, offset, part_len)
            self._calc_part(part_len)
            self._save_row_part_safely(var_idx, offset, part_len)

        _exec_front_last_diff_scalar(self.tik_instance, self.each_row_data_num, self.cols_per_part,
                                     lambda offset, part_len: _do_calc(offset, part_len),
                                     lambda offset, part_len: _do_calc_tail(offset, part_len))

    def _travel_indices_batch(self, start, cnt):
        """
        travel indices per batch

        Parameters
        ----------
        start: offset of indices
        cnt: how many indices of a batch

        Returns
        -------
        None
        """

        self._load_indices(start, cnt)

        with self.tik_instance.for_range(0, cnt) as j:
            self.cur_index.set_as(self.indices_ub[j])
            self._calc_a_indices(self.cur_index, start + j)

    def _travel_indices(self, block_idx):
        """
        travel indices

        Parameters
        ----------
        block_idx: core idx

        Returns
        -------
        None
        """
        batch_cnt = self.tik_instance.Scalar(dtype="int32", name="batch_cnt")
        with self.tik_instance.if_scope(self.num_indices < self.indices_ub_number):
            batch_cnt.set_as(self.num_indices)
        with self.tik_instance.else_scope():
            batch_cnt.set_as(self.indices_ub_number)
       
        num_indices_per_core = self.num_indices // self.need_core_num
        turning = self.num_indices % self.need_core_num

        with self.tik_instance.if_scope(turning > 0):
            with self.tik_instance.if_scope(block_idx < turning):
                _exec_front_last(
                    self.tik_instance, num_indices_per_core + 1, batch_cnt,
                    lambda offset, part_len: self._travel_indices_batch(block_idx *
                                                                        (num_indices_per_core + 1) + offset, part_len))
            with self.tik_instance.else_scope():
                _exec_front_last(
                    self.tik_instance, num_indices_per_core, batch_cnt,
                    lambda offset, part_len: self._travel_indices_batch(
                        turning * (num_indices_per_core + 1) +
                        (block_idx - turning) * num_indices_per_core + offset, part_len))

        with self.tik_instance.else_scope():
            _exec_front_last(
                self.tik_instance, num_indices_per_core, batch_cnt, lambda offset, part_len: self._travel_indices_batch(
                    block_idx * num_indices_per_core + offset, part_len))

    def _calc_a_small_row(self, grad_idx):
        """
        calc a small whole row

        Parameters
        ----------
        grad_idx: row index of grad on global

        Returns
        -------
        None
        """
        offset = self.tik_instance.Scalar("int32")
        offset.set_as(self.reg_cur_row - self.reg_row_start)
        for i in range(len(self.input_tensor)):
            with self.tik_instance.for_range(0, self.each_row_data_num) as j:
                self.align_ub[i][j].set_as(self.ub[i][offset * self.each_row_data_num + j])
        with self.tik_instance.for_range(0, self.each_row_data_num) as i:
            self.grad_align_ub[i].set_as(self.grad_ub[grad_idx * self.each_row_data_num + i])

        self._calculate(1, self.each_row_data_num, 0)

        for i in range(len(self.input_tensor)):
            with self.tik_instance.for_range(0, self.each_row_data_num) as j:
                self.ub[i][offset * self.each_row_data_num + j].set_as(self.align_ub[i][j])

    def _row_in_core_exp(self, block_idx):
        """
        expression of whether current row is processed in the core

        Parameters
        ----------
        block_idx: core index

        Returns
        -------
        expression
        """
        return tik.all(self.reg_cur_row >= block_idx * self.indices_step, self.reg_cur_row < self.reg_core_last_rows)

    def _row_in_ub_exp(self):
        """
        expression of whether current row is already loaded on ubuf

        Parameters
        ----------
        None

        Returns
        -------
        expression
        """
        return tik.all(self.reg_cur_row >= self.reg_row_start,
                       self.reg_cur_row < self.reg_row_start + self.num_multi_rows)

    def _loaded_exp(self):
        """
        expression of whether first batch rows loaded from gm

        Parameters
        ----------
        None

        Returns
        -------
        expression
        """
        return self.reg_row_start < self.var_rows

    def _load_multi_rows(self):
        """
        load multi input rows from gm, except indices and grad

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        burst_len = self.tik_instance.Scalar("int32")
        burst_len.set_as(util_common.ceil(self.num_multi_rows * self.each_row_data_num, self.grad_each_block))
        with self.tik_instance.if_scope(self.reg_cur_row + self.num_multi_rows <= self.reg_core_last_rows):
            self.reg_row_start.set_as(self.reg_cur_row)
        with self.tik_instance.else_scope():
            self.reg_row_start.set_as(self.reg_core_last_rows - self.num_multi_rows)

        for i in range(len(self.input_tensor)):
            self.tik_instance.data_move(self.ub[i], self.input_tensor[i][self.reg_row_start * self.each_row_data_num],
                                        0, 1, burst_len, 0, 0)

    def _save_multi_rows(self):
        """
        save multy rows to gm

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        burst_len = self.tik_instance.Scalar("int32")
        burst_len.set_as(util_common.ceil(self.num_multi_rows * self.each_row_data_num, self.grad_each_block))
        for i in range(len(self.output)):
            self.tik_instance.data_move(self.output[i][self.reg_row_start * self.each_row_data_num], self.ub[i], 0, 1,
                                        burst_len, 0, 0)

    def _calc_multi_indices(self, block_idx, num_indices):
        """
        calculate multi rows, multi rows will read at one to avoid loading little data from gm to ubuf at a high
        frequency

        Parameters
        ----------
        num_indices: how many indices to calculate

        Returns
        -------
        None
        """

        with self.tik_instance.for_range(0, num_indices) as i:
            self.reg_cur_row.set_as(self.indices_ub[i])
            with self.tik_instance.if_scope(self._row_in_core_exp(block_idx)):
                with self.tik_instance.if_scope(self._row_in_ub_exp()):
                    self._calc_a_small_row(i)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(self._loaded_exp()):
                        self._save_multi_rows()
                    self._load_multi_rows()
                    self._calc_a_small_row(i)
        with self.tik_instance.if_scope(self._loaded_exp()):
            self._save_multi_rows()

    def _travel_multi_indices(self, block_idx):
        """
        _travel_multi_indices

        Parameters
        ----------
        block_idx: core idx

        Returns
        -------
        None
        """
        loop_cnt = self.tik_instance.Scalar("int32")
        loop_cnt.set_as(self.num_indices // self.indices_ub_number)
        indices_last_num = self.num_indices - loop_cnt * self.indices_ub_number

        burst_len = self.tik_instance.Scalar("int32")
        burst_len.set_as(util_common.ceil(self.indices_ub_number, self.indices_data_each_block))
        burst_len_grad = self.tik_instance.Scalar("int32")
        burst_len_grad.set_as(util_common.ceil(self.indices_ub_number * self.each_row_data_num, self.grad_each_block))

        with self.tik_instance.if_scope(block_idx < self.need_core_num - 1):
            self.reg_core_last_rows.set_as(self.indices_step * (block_idx + 1))
        with self.tik_instance.else_scope():
            self.reg_core_last_rows.set_as(self.var_rows)

        with self.tik_instance.for_range(0, loop_cnt) as i:
            self.tik_instance.data_move(self.indices_ub, self.indices_gm[i * self.indices_ub_number], 0, 1, burst_len,
                                        0, 0)
            self.tik_instance.data_move(self.grad_ub, self.grad_gm[i * self.indices_ub_number * self.each_row_data_num],
                                        0, 1, burst_len_grad, 0, 0)
            self._calc_multi_indices(block_idx, self.indices_ub_number)

        with self.tik_instance.if_scope(indices_last_num > 0):
            burst_len.set_as(util_common.ceil(indices_last_num, self.indices_data_each_block))
            burst_len_grad.set_as(util_common.ceil(indices_last_num * self.each_row_data_num, self.grad_each_block))
            self.tik_instance.data_move(self.indices_ub, self.indices_gm[loop_cnt * self.indices_ub_number], 0, 1,
                                        burst_len, 0, 0)
            self.tik_instance.data_move(self.grad_ub,
                                        self.grad_gm[loop_cnt * self.indices_ub_number * self.each_row_data_num], 0, 1,
                                        burst_len_grad, 0, 0)
            self._calc_multi_indices(block_idx, indices_last_num)

    def _calc_core_partial(self, var_idx, grad_idx, block_idx):
        """
        calc partial of a row by cores

        Parameters
        ----------
        var_idx: row_idx
        grad_idx: grad_idx
        block_idx: core idx

        Returns
        -------
        None
        """
        core_start_offset = (block_idx - self.reg_row_start_core * self.partial_factor) * self.cols_per_core

        def _do_calc(offset, part_len):
            self._load_row_part(var_idx, grad_idx, offset + core_start_offset, part_len)
            self._calc_part(part_len)
            self._save_row_part(var_idx, offset + core_start_offset, part_len)

        def _do_calc_tail(offset, part_len):
            self._load_row_part(var_idx, grad_idx, offset + core_start_offset, part_len)
            self._calc_part(part_len)
            self._save_row_part_safely(var_idx, offset + core_start_offset, part_len)

        with self.tik_instance.if_scope(block_idx == (self.reg_row_start_core + 1) * self.partial_factor - 1):
            _exec_front_last_diff_scalar(self.tik_instance, self.cols_last_core, self.cols_per_part,
                                         lambda offset, part_len: _do_calc(
                                             offset, part_len),
                                         lambda offset, part_len: _do_calc_tail(offset, part_len))

        with self.tik_instance.else_scope():
            _exec_front_last_diff_scalar(self.tik_instance, self.cols_per_core, self.cols_per_part,
                                         lambda offset, part_len: _do_calc(
                                             offset, part_len),
                                         lambda offset, part_len: _do_calc_tail(offset, part_len))

    def _travel_partial_indices(self, block_idx):
        """
        _travel_partial_indices

        Parameters
        ----------
        block_idx: core idx

        Returns:
        ----------
        None
        """
        burst_len = util_common.ceil(self.num_indices, self.indices_data_each_block)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm, 0, 1, burst_len, 0, 0)
        self.reg_row_start_core.set_as(block_idx / self.partial_factor)
        self.reg_cur_row.set_as(self.indices_ub[self.reg_row_start_core])
        self._calc_core_partial(self.reg_cur_row, self.reg_row_start_core, block_idx)

    def _travel_the_indices(self):
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_idx:
            with self.tik_instance.if_scope(block_idx < self.need_core_num):
                with self.tik_instance.if_scope(self.each_row_data_num <= self.cache_threshold_col):
                    self._travel_multi_indices(block_idx)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(self.need_core_num > self.num_indices):
                        self._travel_partial_indices(block_idx)
                    with self.tik_instance.else_scope():
                        self._travel_indices(block_idx)

    def sparse_apply_operator(self):
        """
        SparseAdagrad operation

        Parameters
        ----------
        None

        Returns:
        ----------
        tik_instance: tik instance
        """
        self._get_tiling_args()
        self._alloc_ub()
        self._travel_the_indices()
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.core_num,
                "ub_size": self.ub_size_bytes,
                "indices_dsize": self.indices_dtype_bytes_size,
                "ub_take_parts": 1,
                "ub_block_num": len(self.ub_reserved) + 1,
                "cache_threshold_col": self.cache_threshold_col
            })

        inputs_gm = self.input_tensor + self.input_scalar_gm + [self.grad_gm, self.indices_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=inputs_gm,
                                   outputs=self.output,
                                   flowtable=(self.tiling_gm,),
                                   enable_l2=False)

        return self.tik_instance
