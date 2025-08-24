#!/usr/bin/python
# -*- coding: utf-8 -*-
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
sparse_apply_common
"""
import functools
import math

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-many-arguments,too-many-locals,invalid-name
# 'pylint: disable=too-many-instance-attributes
class SparseApply():
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
        self.device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.grad_shape = grad.get("shape")
        self.grad_dtype = grad.get("dtype").lower()

        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()

        self.indices_shape = indices.get("shape")
        self.indices_dtype = indices.get("dtype").lower()

        self.scalar_shape = (1,)
        self.indices_num = self.indices_shape[0]
        self.kernel_name = kernel_name

        self.indices_dtype_bytes_size = tbe_platform.get_bit_len(self.indices_dtype) // 8
        self.grad_dtype_bytes_size = tbe_platform.get_bit_len(self.grad_dtype) // 8

        self.grad_each_block = 32 // self.grad_dtype_bytes_size
        self.indices_data_each_block = 32 // self.indices_dtype_bytes_size

        # Reserved 1024 Bytes for inputs and outputs 32B alignment
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 1024)

        one_block_bytes_size = tbe_platform.VECTOR_INST_BLOCK_WIDTH // tbe_platform.VECTOR_INST_BLOCK_NUM
        self.block_len = one_block_bytes_size // self.grad_dtype_bytes_size

        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype,
                                                   self.indices_shape,
                                                   name="indices_gm",
                                                   scope=tbe_platform.scope_gm)
        self.grad_gm = self.tik_instance.Tensor(self.grad_dtype, self.grad_shape, name="grad_gm",
                                                scope=tbe_platform.scope_gm)

        self.rows = self.indices_shape[0]
        if len(self.grad_shape) > 1:
            self.each_row_data_num = functools.reduce(lambda x, y: x * y, self.grad_shape[1:])
        else:
            self.each_row_data_num = 1

        self._check_param_common()

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
        self.num_indices = self.indices_shape[0]
        self.num_one_repeat = tbe_platform.VECTOR_INST_BLOCK_NUM * self.block_len

        self.reg_row_start = self.tik_instance.Scalar(self.indices_dtype)
        self.reg_cur_row = self.tik_instance.Scalar(self.indices_dtype)
        self.reg_core_last_rows = self.tik_instance.Scalar(self.indices_dtype)
        self.reg_row_start_core = self.tik_instance.Scalar(self.indices_dtype)
        self.var_rows = None
        self.var_cols = None
        self.var_ub_shape = None
        self.indices_ub_shape = None
        self.cols_per_part = None
        self.num_indices_per_batch = None
        self.cache_threshold_col = None
        self.num_multi_rows = None
        self.block_num = None
        self.partial_factor = None
        self.cols_per_core = None
        self.cols_last_core = None
        self.indices_step = None
        self.indices_ub_number = None
        self.grad_ub = None
        self.indices_ub = None
        self.grad_align_ub = None

    def _check_param_common(self):
        """
        Check parameter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        para_check.check_shape(self.indices_shape)
        para_check.check_shape(self.grad_shape)

        para_check.check_dtype(self.indices_dtype, ("int32", "int64"))
        para_check.check_dtype(self.grad_dtype, ("float32"))

        if self.grad_shape[1:] != self.var_shape[1:]:
            error_manager_vector.raise_err_check_params_rules(
                self.kernel_name,
                "grad's shape must be the same as var's shape except first dimension, var's shape is %s" %
                self.var_shape, "grad", self.grad_shape)

        if len(self.indices_shape) != 1:
            error_manager_vector.raise_err_input_param_range_invalid(self.kernel_name, 'indices', 1, 1,
                                                                     len(self.indices_shape))

        if self.grad_shape[0] != self.indices_shape[0]:
            error_manager_vector.raise_err_check_params_rules(
                self.kernel_name,
                "grad must be the same shape as indices in first dimension, grad's shape is %s" % self.grad_shape,
                'indices', self.indices_shape)

    def set_var_shape(self, shape):
        """
        set var rows, called by external

        Parameters
        ----------
        num: number of the first dimension of var

        Returns
        -------
        None
        """
        self.var_rows = shape[0]
        self.reg_row_start.set_as(shape[0] + 1)
        self.var_cols = 1
        for i in range(1, len(shape)):
            self.var_cols *= shape[i]

    def add_input(self, name, dtype, shape):
        """
        called by external, describe the info of inputs excepts indices and grad every input will alloc a tik gm tensor,
        and passed to BuildCCE func with grad and indices as inputs

        Parameters
        ----------
        name: string type, name of the input
        dtype: type of the input
        shape: shape of the input

        Returns
        -------
        None
        """
        tensor = self.tik_instance.Tensor(dtype, shape, name=name, scope=tbe_platform.scope_gm)
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
        scalar_gm = self.tik_instance.Tensor(dtype, self.scalar_shape, name=name, scope=tbe_platform.scope_gm)
        self.input_scalar_gm.append(scalar_gm)
        self.scalar_gm_map[name] = scalar_gm

    def add_output(self, name, dtype, shape):
        """
        called by external, describe the info of outputs
        every output will alloc a tik gm tensor, and passed to BuildCCE func as outputs

        Parameters
        ----------
        name: string type, name of the input
        dtype: type of the input
        shape: shape of the input

        Returns
        -------
        None
        """
        tensor = self.tik_instance.Tensor(dtype, shape, name=name, scope=tbe_platform.scope_gm)
        self.output.append(tensor)
        self.tensor_map[name] = tensor

        tail_tensor = self.tik_instance.Tensor(dtype, (self.block_len,), name=name + "_tail_ub",
                                               scope=tbe_platform.scope_ubuf)
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

    def _calc_tilling_param(self):
        """
        calculate tiling parameters

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # +1 for grad
        ub_block_num = len(self.ub_reserved) + 1

        indices_size_bytes = self.ub_size_bytes // 100
        indices_cnt = indices_size_bytes // self.indices_dtype_bytes_size

        if self.indices_num <= indices_cnt:
            indices_cnt = self.indices_num

        remain_ub_mem = self.ub_size_bytes - indices_cnt * self.indices_dtype_bytes_size
        update_num_per_ub = remain_ub_mem // self.grad_dtype_bytes_size // ub_block_num
        update_num_per_ub = update_num_per_ub // self.grad_each_block * self.grad_each_block

        if self.var_cols > update_num_per_ub and self.var_cols % update_num_per_ub < 8:
            # last part of a rows must process at leat one block
            update_num_per_ub -= self.grad_each_block

        self.var_ub_shape = (update_num_per_ub,)
        self.indices_ub_shape = (indices_cnt,)
        self.cols_per_part = update_num_per_ub
        self.num_indices_per_batch = indices_cnt
        self.cache_threshold_col = self.block_len - 1

        self.num_multi_rows = 32  # must be 32 factor for align
        if self.var_rows < self.num_multi_rows:
            self.num_multi_rows = self.var_rows

        core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.block_num = core_num
        if self.num_indices < core_num and self.each_row_data_num > self.cache_threshold_col:
            self.block_num = self.num_indices
            if self.block_num * 2 <= core_num and self.each_row_data_num >= core_num * self.grad_each_block:
                self.partial_factor = core_num // self.block_num
                self.block_num = self.block_num * self.partial_factor
                self.cols_per_core = self.each_row_data_num // self.partial_factor
                self.cols_per_core = self.cols_per_core // self.grad_each_block * self.grad_each_block
                self.cols_last_core = self.each_row_data_num - self.cols_per_core * (self.partial_factor - 1)
                if 0 < self.cols_last_core % self.cols_per_part < self.grad_each_block:
                    self.cols_per_part -= self.grad_each_block

        if self.each_row_data_num <= self.cache_threshold_col:
            self.block_num = self.var_rows // self.num_multi_rows
            if self.block_num > core_num:
                self.block_num = core_num
            if self.block_num <= 0:
                self.block_num = 1

        self.indices_step = self.var_rows // self.block_num
        self.indices_ub_number = indices_cnt

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
                                                scope=tbe_platform.scope_ubuf)

        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype,
                                                   self.indices_ub_shape,
                                                   name="indices_ub",
                                                   scope=tbe_platform.scope_ubuf)

        self.grad_align_ub = self.tik_instance.Tensor(self.grad_dtype, (self.block_len,),
                                                      name="grad_align_ub",
                                                      scope=tbe_platform.scope_ubuf)

        for name, dtype in self.ub_reserved:
            tensor = self.tik_instance.Tensor(dtype, self.var_ub_shape, name=name, scope=tbe_platform.scope_ubuf)
            self.ub.append(tensor)
            self.tensor_map[name] = tensor

        if self.scalar_ub_reserved:
            for name, dtype in self.scalar_ub_reserved:
                tensor = self.tik_instance.Tensor(dtype, self.scalar_shape, name=name, scope=tbe_platform.scope_ubuf)
                self.tensor_map[name] = tensor

        for name, dtype in self.align_ub_info:
            tensor = self.tik_instance.Tensor(dtype, (self.block_len,), name=name, scope=tbe_platform.scope_ubuf)
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
        burst_len = math.ceil(cnt / self.grad_each_block)
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
        burst_len = math.ceil(cnt / self.grad_each_block)
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
        if burst_len > 0:
            for i in range(len(self.output)):
                self.tik_instance.data_move(self.output[i][var_idx * self.each_row_data_num + offset], self.ub[i], 0, 1,
                                            burst_len, 0, 0)
        if cnt % self.grad_each_block != 0:
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
        burst_len = math.ceil(cnt / self.indices_data_each_block)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[start], 0, 1, burst_len, 0, 0)

    def _calculate(self, repeat_times, mask, offset):
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

        if repeat_255 > 0:
            with self.tik_instance.for_range(0, repeat_255) as i:
                self._calculate(255, num_one_repeat, i * 255 * num_one_repeat)
        if repeat > 0:
            self._calculate(repeat, num_one_repeat, repeat_255 * 255 * num_one_repeat)

        if remain > 0:
            self._calculate(1, remain, repeat_255 * 255 * num_one_repeat + repeat * num_one_repeat)

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

        num_part = self.each_row_data_num // self.cols_per_part
        cnt = self.cols_per_part
        remain = self.each_row_data_num % self.cols_per_part

        if num_part > 0:
            with self.tik_instance.for_range(0, num_part) as i:
                self._load_row_part(var_idx, grad_idx, i * cnt, cnt)
                self._calc_part(cnt)
                self._save_row_part(var_idx, i * cnt, cnt)

        if remain > 0:
            self._load_row_part(var_idx, grad_idx, num_part * cnt, remain)
            self._calc_part(remain)
            self._save_row_part_safely(var_idx, num_part * cnt, remain)

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
        batch_cnt = self.num_indices_per_batch
        num_indices_per_core = self.num_indices // self.block_num
        turning = self.num_indices % self.block_num

        if turning > 0:
            with self.tik_instance.if_scope(block_idx < turning):
                num_batch1 = (num_indices_per_core + 1) // batch_cnt
                remain1 = (num_indices_per_core + 1) % batch_cnt
                core_offset1 = block_idx * (num_indices_per_core + 1)
                if num_batch1 > 0:
                    with self.tik_instance.for_range(0, num_batch1) as i:
                        self._travel_indices_batch(core_offset1 + i * batch_cnt, batch_cnt)
                if remain1 > 0:
                    self._travel_indices_batch(core_offset1 + num_batch1 * batch_cnt, remain1)
            with self.tik_instance.else_scope():
                num_batch2 = (num_indices_per_core) // batch_cnt
                remain2 = (num_indices_per_core) % batch_cnt
                core_offset2 = turning * (num_indices_per_core + 1) + (block_idx - turning) * num_indices_per_core
                if num_batch2 > 0:
                    with self.tik_instance.for_range(0, num_batch2) as i:
                        self._travel_indices_batch(core_offset2 + i * batch_cnt, batch_cnt)
                if remain2 > 0:
                    self._travel_indices_batch(core_offset2 + num_batch2 * batch_cnt, remain2)

        else:
            num_batch = num_indices_per_core // batch_cnt
            remain = num_indices_per_core % batch_cnt
            core_offset = block_idx * num_indices_per_core
            if num_batch > 0:
                with self.tik_instance.for_range(0, num_batch) as i:
                    self._travel_indices_batch(core_offset + i * batch_cnt, batch_cnt)
            if remain > 0:
                self._travel_indices_batch(core_offset + num_batch * batch_cnt, remain)

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
        offset = self.reg_cur_row - self.reg_row_start
        for i in range(len(self.input_tensor)):
            with self.tik_instance.for_range(0, self.each_row_data_num) as j:
                self.align_ub[i][j].set_as(self.ub[i][offset * self.each_row_data_num + j])

        for i in range(self.each_row_data_num):
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
        burst_len = math.ceil(self.num_multi_rows * self.each_row_data_num / self.grad_each_block)
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
        burst_len = math.ceil(self.num_multi_rows * self.each_row_data_num / self.grad_each_block)
        for i in range(len(self.output)):
            self.tik_instance.data_move(self.output[i][self.reg_row_start * self.each_row_data_num], self.ub[i], 0, 1,
                                        burst_len, 0, 0)

    def _calc_multi_indices(self, block_idx, indices_num):
        """
        calculate multi rows, multi rows will read at one to avoid loading little data from gm to ubuf at a high
        frequency

        Parameters
        ----------
        indices_num: how many indices to calculate

        Returns
        -------
        None
        """

        with self.tik_instance.for_range(0, indices_num) as i:
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
        loop_cnt = self.num_indices // self.indices_ub_number

        burst_len = math.ceil(self.indices_ub_number / self.indices_data_each_block)
        burst_len_grad = math.ceil(self.indices_ub_number * self.each_row_data_num / self.grad_each_block)

        with self.tik_instance.if_scope(block_idx < self.block_num - 1):
            self.reg_core_last_rows.set_as(self.indices_step * (block_idx + 1))
        with self.tik_instance.else_scope():
            self.reg_core_last_rows.set_as(self.var_rows)

        with self.tik_instance.for_range(0, loop_cnt) as i:
            self.tik_instance.data_move(self.indices_ub, self.indices_gm[i * self.indices_ub_number], 0, 1, burst_len,
                                        0, 0)
            self.tik_instance.data_move(self.grad_ub, self.grad_gm[i * self.indices_ub_number * self.each_row_data_num],
                                        0, 1, burst_len_grad, 0, 0)
            self._calc_multi_indices(block_idx, self.indices_ub_number)

        indices_last_num = self.num_indices % self.indices_ub_number
        if indices_last_num > 0:
            burst_len = math.ceil(indices_last_num / self.indices_data_each_block)
            burst_len_grad = math.ceil(indices_last_num * self.each_row_data_num / self.grad_each_block)
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

        with self.tik_instance.if_scope(block_idx == (self.reg_row_start_core + 1) * self.partial_factor - 1):
            num_part = self.cols_last_core // self.cols_per_part
            cnt = self.cols_per_part
            remain = self.cols_last_core % self.cols_per_part
            if num_part > 0:
                with self.tik_instance.for_range(0, num_part) as i:
                    self._load_row_part(var_idx, grad_idx, i * cnt + core_start_offset, cnt)
                    self._calc_part(cnt)
                    self._save_row_part(var_idx, i * cnt + core_start_offset, cnt)

            if remain > 0:
                self._load_row_part(var_idx, grad_idx, num_part * cnt + core_start_offset, remain)
                self._calc_part(remain)
                self._save_row_part_safely(var_idx, num_part * cnt + core_start_offset, remain)
        with self.tik_instance.else_scope():
            num_part = self.cols_per_core // self.cols_per_part
            cnt = self.cols_per_part
            remain = self.cols_per_core % self.cols_per_part
            if num_part > 0:
                with self.tik_instance.for_range(0, num_part) as i:
                    self._load_row_part(var_idx, grad_idx, i * cnt + core_start_offset, cnt)
                    self._calc_part(cnt)
                    self._save_row_part(var_idx, i * cnt + core_start_offset, cnt)

            if remain > 0:
                self._load_row_part(var_idx, grad_idx, num_part * cnt + core_start_offset, remain)
                self._calc_part(remain)
                self._save_row_part_safely(var_idx, num_part * cnt + core_start_offset, remain)

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
        burst_len = math.ceil(self.num_indices / self.indices_data_each_block)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm, 0, 1, burst_len, 0, 0)
        self.reg_row_start_core.set_as(block_idx / self.partial_factor)
        self.reg_cur_row.set_as(self.indices_ub[self.reg_row_start_core])
        self._calc_core_partial(self.reg_cur_row, self.reg_row_start_core, block_idx)

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

        self._calc_tilling_param()
        self._alloc_ub()

        with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as block_idx:
            if self.each_row_data_num <= self.cache_threshold_col:
                self._travel_multi_indices(block_idx)
            elif self.block_num > self.num_indices:
                self._travel_partial_indices(block_idx)
            else:
                self._travel_indices(block_idx)

        inputs_gm = self.input_tensor + self.input_scalar_gm + [self.grad_gm, self.indices_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=inputs_gm, outputs=self.output, enable_l2=False)

        return self.tik_instance
