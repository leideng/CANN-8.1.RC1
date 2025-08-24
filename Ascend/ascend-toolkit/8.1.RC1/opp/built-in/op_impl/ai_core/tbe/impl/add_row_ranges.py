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
add_row_ranges
"""
import math

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    # get available ub size
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)


# 'pylint: disable=invalid-name
def _check_param(x, src, indices, kernel_name):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    x: dict
        shape and datatype
    src: dict
        shape and datatype
    indices: dict
        shape and datatype
    kernel_name: str
        kernel_name, default value is 'add_row_ranges'

    Returns
    -------
    None
    """
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    src_shape = src.get("shape")
    src_dtype = src.get("dtype").lower()
    indices_shape = indices.get("shape")
    indices_dtype = indices.get("dtype").lower()

    para_check.check_shape(x_shape, param_name="x")
    para_check.check_shape(src_shape, param_name="src")
    para_check.check_dtype(x_dtype, ("float32",), param_name="x")
    para_check.check_dtype(src_dtype, ("float32",), param_name="src")
    para_check.check_dtype(indices_dtype, ("int32",), param_name="indices")

    if len(x_shape) != 2 or len(src_shape) != 2 or len(indices_shape) != 2:
        expected_value = "equal to 2"
        real_value = "not equal to 2"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "length of input_shape",
                                                           expected_value, real_value)

    if indices_shape[0] != x_shape[0]:
        error_manager_vector.raise_err_inputs_shape_not_equal(kernel_name, 'indices_shape[0]', 'x_shape[0]',
                                                              indices_shape[0], x_shape[0], "equal")

    if src_shape[1] != x_shape[1]:
        error_manager_vector.raise_err_inputs_shape_not_equal(kernel_name, 'src_shape[1]', 'x_shape[1]',
                                                              src_shape[1], x_shape[1], "equal")


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class AddRowRanges():
    """AddRowRanges main functions
    """
    def __init__(self, x, src, indices, kernel_name="add_row_ranges"):
        """init AddRowRanges base parameters
        """
        self.tik_instance = tik.Tik()
        self.x_shape = x.get("shape")
        self.src_shape = src.get("shape")
        self.indices_shape = indices.get("shape")
        self.x_dtype = x.get("dtype")
        self.src_dtype = src.get("dtype")
        self.indices_dtype = indices.get("dtype")
        self.kernel_name = kernel_name
        self.x_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape, name="x_gm", scope=tik.scope_gm)
        self.src_gm = self.tik_instance.Tensor(self.src_dtype, self.src_shape, name="src_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype, self.indices_shape, name="indices_gm",
                                                   scope=tik.scope_gm)
        self.x_out_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape, name="x_out_gm", scope=tik.scope_gm)
        self.core_counts = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    def add_row_ranges_compute(self):
        """
        add_row_ranges main part
        """
        branch_flag = self._calc_branch_flag()
        # not need cut and n <= 2040
        if branch_flag == 1:
            self._fun_no_cut()
        # src need cut n only
        elif branch_flag == 2:
            self._fun_cut_n()
        # src need cut m and n
        elif branch_flag == 3:
            self._fun_cut_n_and_m()
        else:
            self._fun_small_n()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.x_gm, self.src_gm, self.indices_gm],
                                   outputs=(self.x_out_gm))

        return self.tik_instance

    def _calc_branch_flag(self):
        """
        funtion check if need cut src or x
        """
        src_n_len = math.ceil(self.src_shape[1] / 8)
        data_not_cut = self.src_shape[0] * src_n_len * 32 + 8 * (src_n_len * 8 + 2) * 4 + 64
        n = int((Constant.UB_SIZE / 4 - 16) / (self.src_shape[0] + 3))

        # not need cut and n <= 2040
        if data_not_cut <= Constant.UB_SIZE and self.src_shape[1] <= 2040 and self.x_shape[1] % 8 == 0:
            branch_flag = 1
        # src need cut m only
        elif n >= 8 and self.src_shape[1] >= 8 and self.x_shape[1] % 8 == 0:
            branch_flag = 2
        # src need cut m and n
        elif self.src_shape[1] >= 8:
            branch_flag = 3
        else:
            branch_flag = 4

        return branch_flag

    # 'pylint: disable=too-many-locals
    def _fun_no_cut(self):
        """
        funtion with no need cut src and n <= 2040
        """
        start_index = self.tik_instance.Scalar(dtype="int32")
        end_index = self.tik_instance.Scalar(dtype="int32")
        mask_scaler = self.tik_instance.Scalar(dtype="int32")
        block_stride = self.tik_instance.Scalar(dtype="int32")
        # while src is align
        m_size, indices_burst_len = 8, 2
        block_dim = math.ceil(self.x_shape[0] / m_size)
        x_burst_len = m_size * self.x_shape[1] // 8
        x_burst_len_tail = (self.x_shape[0] - (block_dim - 1) * m_size) * self.x_shape[1] // 8

        indices_burst_len_tail = math.ceil((self.x_shape[0] - (block_dim - 1) * m_size) / 4)
        block_stride.set_as(self.x_shape[1] / 8)
        src_burst_len = self.src_shape[0] * self.src_shape[1] // 8
        src_ub = self.tik_instance.Tensor(self.src_dtype, self.src_shape, name="src_ub",
                                          scope=tik.scope_ubuf)
        self.tik_instance.data_move(src_ub, self.src_gm, 0, 1, src_burst_len, 0, 0)
        x_ub = self.tik_instance.Tensor(self.x_dtype, (m_size * self.x_shape[1], ),
                                        name="x_ub", scope=tik.scope_ubuf)
        indices_ub = self.tik_instance.Tensor(self.indices_dtype, (math.ceil(m_size / 4) * 8, ),
                                              name="indices_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, block_dim, block_num=block_dim) as block_index:
            with self.tik_instance.if_scope(block_index != block_dim - 1):
                self._calc_no_cut_align(x_ub, block_index, m_size, x_burst_len, indices_ub, indices_burst_len,
                                        start_index, end_index, src_ub, mask_scaler, block_stride, 0)
            with self.tik_instance.else_scope():
                self._calc_no_cut_align(x_ub, block_index, m_size, x_burst_len_tail,
                                        indices_ub, indices_burst_len_tail, start_index, end_index, src_ub,
                                        mask_scaler, block_stride, self.x_shape[0] - block_index * m_size)

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _calc_no_cut_align(self, x_ub, block_index, m_size, x_burst_len, indices_ub, indices_burst_len,
                           start_index, end_index, src_ub, mask_scaler, block_stride, m_size_tail):
        """
        funtion with no need cut
        """
        add_loop = math.ceil(self.x_shape[1] / 64)
        self.tik_instance.data_move(x_ub, self.x_gm[block_index * m_size * self.x_shape[1]], 0, 1,
                                    x_burst_len, 0, 0)
        self.tik_instance.data_move(indices_ub, self.indices_gm[block_index * m_size * 2], 0, 1,
                                    indices_burst_len, 0, 0)
        m_loop_num = self.tik_instance.Scalar(dtype="int32", init_value=m_size)
        with self.tik_instance.if_scope(m_size_tail > 0):
            m_loop_num.set_as(m_size_tail)
        with self.tik_instance.for_range(0, m_loop_num) as m_index:
            start_index.set_as(indices_ub[m_index * 2])
            end_index.set_as(indices_ub[m_index * 2 + 1])
            with self.tik_instance.if_scope(end_index > start_index):
                with self.tik_instance.for_range(start_index, end_index) as index:
                    with self.tik_instance.for_range(0, add_loop) as add_index:
                        with self.tik_instance.if_scope(add_index != add_loop - 1):
                            self.tik_instance.vadd(64, x_ub[m_index * self.x_shape[1] + add_index * 64],
                                                   src_ub[index * block_stride * 8 + add_index * 64],
                                                   x_ub[m_index * self.x_shape[1] + add_index * 64],
                                                   1, 1, 1, 1, 0, 0, 0)
                        with self.tik_instance.else_scope():
                            mask_scaler.set_as(self.x_shape[1] - add_index * 64)
                            self.tik_instance.vadd(mask_scaler,
                                                   x_ub[m_index * self.x_shape[1] + add_index * 64],
                                                   src_ub[index * block_stride * 8 + add_index * 64],
                                                   x_ub[m_index * self.x_shape[1] + add_index * 64],
                                                   1, 1, 1, 1, 0, 0, 0)
        self.tik_instance.data_move(self.x_out_gm[block_index * m_size * self.x_shape[1]], x_ub, 0, 1,
                                    x_burst_len, 0, 0)

    # 'pylint: disable=too-many-locals,too-many-statements
    def _fun_cut_n(self):
        """
        src is align and need cut n only
        """
        n_loop_num, n_size, n_tail = self._calc_cut_size()
        m_loop_num = math.ceil(self.x_shape[0] / 8)
        m_tail = self.x_shape[0] - (m_loop_num - 1) * 8
        m_tail_loop_num = math.ceil(m_tail / 4)
        m_inner_loop_num = math.ceil(m_tail / 2)
        src_ub = self.tik_instance.Tensor(self.src_dtype, (self.src_shape[0] * n_size, ), name="src_ub",
                                          scope=tik.scope_ubuf)
        x_ub_ping = self.tik_instance.Tensor(self.x_dtype, (n_size, ), name="x_ub_ping", scope=tik.scope_ubuf)
        x_ub_pong = self.tik_instance.Tensor(self.x_dtype, (n_size, ), name="x_ub_pong", scope=tik.scope_ubuf)
        indices_ub = self.tik_instance.Tensor(self.indices_dtype, (16, ), name="indices_ub", scope=tik.scope_ubuf)
        start_index = self.tik_instance.Scalar(dtype="int32")
        end_index = self.tik_instance.Scalar(dtype="int32")
        mask_scaler = self.tik_instance.Scalar(dtype="int32")
        with self.tik_instance.for_range(0, n_loop_num, block_num=n_loop_num) as block_index:
            with self.tik_instance.if_scope(block_index != n_loop_num - 1):
                with self.tik_instance.for_range(0, self.src_shape[0]) as src_index:
                    self.tik_instance.data_move(src_ub[src_index * n_size],
                                                self.src_gm[src_index * self.src_shape[1] + block_index * n_size],
                                                0, 1, n_size // 8, 0, 0)
                with self.tik_instance.for_range(0, m_loop_num) as m_index:
                    with self.tik_instance.if_scope(m_index != m_loop_num - 1):
                        self.tik_instance.data_move(indices_ub, self.indices_gm[m_index * 16], 0, 1, 2, 0, 0)
                        with self.tik_instance.for_range(0, 4) as m_index_inner:
                            # ping
                            self._calc_cut_n(x_ub_ping, m_index * 8 + m_index_inner * 2, block_index, n_size,
                                             indices_ub, m_index_inner * 4, src_ub, start_index, end_index,
                                             mask_scaler, 0)
                            # pong
                            self._calc_cut_n(x_ub_pong, m_index * 8 + m_index_inner * 2 + 1, block_index, n_size,
                                             indices_ub, m_index_inner * 4 + 2, src_ub, start_index, end_index,
                                             mask_scaler, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(indices_ub, self.indices_gm[m_index * 16], 0, 1,
                                                    m_tail_loop_num, 0, 0)
                        with self.tik_instance.for_range(0, m_inner_loop_num) as m_index_inner:
                            with self.tik_instance.if_scope(m_index_inner != m_inner_loop_num - 1):
                                # ping
                                self._calc_cut_n(x_ub_ping, m_index * 8 + m_index_inner * 2, block_index, n_size,
                                                 indices_ub, m_index_inner * 4, src_ub, start_index, end_index,
                                                 mask_scaler, 0)
                                # pong
                                self._calc_cut_n(x_ub_pong, m_index * 8 + m_index_inner * 2 + 1, block_index,
                                                 n_size, indices_ub, m_index_inner * 4 + 2, src_ub, start_index,
                                                 end_index, mask_scaler, 0)
                            with self.tik_instance.else_scope():
                                with self.tik_instance.if_scope(m_tail % 2 == 0):
                                    # ping
                                    self._calc_cut_n(x_ub_ping, m_index * 8 + m_index_inner * 2, block_index,
                                                     n_size, indices_ub, m_index_inner * 4, src_ub, start_index,
                                                     end_index, mask_scaler, 0)
                                    # pong
                                    self._calc_cut_n(x_ub_pong, m_index * 8 + m_index_inner * 2 + 1, block_index,
                                                     n_size, indices_ub, m_index_inner * 4 + 2, src_ub, start_index,
                                                     end_index, mask_scaler, 0)
                                with self.tik_instance.else_scope():
                                    # ping
                                    self._calc_cut_n(x_ub_ping, m_index * 8 + m_index_inner * 2, block_index, n_size,
                                                     indices_ub, m_index_inner * 4, src_ub, start_index, end_index,
                                                     mask_scaler, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.src_shape[0]) as src_index:
                    self.tik_instance.data_move(src_ub[src_index * n_tail],
                                                self.src_gm[src_index * self.src_shape[1] + block_index * n_size],
                                                0, 1, n_tail // 8, 0, 0)
                with self.tik_instance.for_range(0, m_loop_num) as m_index:
                    with self.tik_instance.if_scope(m_index != m_loop_num - 1):
                        self.tik_instance.data_move(indices_ub, self.indices_gm[m_index * 16], 0, 1, 2, 0, 0)
                        with self.tik_instance.for_range(0, 4) as m_index_inner:
                            # ping
                            self._calc_cut_n(x_ub_ping, m_index * 8 + m_index_inner * 2, block_index, n_size,
                                             indices_ub, m_index_inner * 4, src_ub, start_index, end_index,
                                             mask_scaler, n_tail)
                            # pong
                            self._calc_cut_n(x_ub_pong, m_index * 8 + m_index_inner * 2 + 1, block_index, n_size,
                                             indices_ub, m_index_inner * 4 + 2, src_ub, start_index, end_index,
                                             mask_scaler, n_tail)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(indices_ub, self.indices_gm[m_index * 16], 0, 1,
                                                    m_tail_loop_num, 0, 0)
                        with self.tik_instance.for_range(0, m_inner_loop_num) as m_index_inner:
                            with self.tik_instance.if_scope(m_index_inner != m_inner_loop_num - 1):
                                # ping
                                self._calc_cut_n(x_ub_ping, m_index * 8 + m_index_inner * 2, block_index, n_size,
                                                 indices_ub, m_index_inner * 4, src_ub, start_index, end_index,
                                                 mask_scaler, n_tail)
                                # pong
                                self._calc_cut_n(x_ub_pong, m_index * 8 + m_index_inner * 2 + 1, block_index, n_size,
                                                 indices_ub, m_index_inner * 4 + 2, src_ub, start_index, end_index,
                                                 mask_scaler, n_tail)
                            with self.tik_instance.else_scope():
                                with self.tik_instance.if_scope(m_tail % 2 == 0):
                                    # ping
                                    self._calc_cut_n(x_ub_ping, m_index * 8 + m_index_inner * 2, block_index, n_size,
                                                     indices_ub, m_index_inner * 4, src_ub, start_index, end_index,
                                                     mask_scaler, n_tail)
                                    # pong
                                    self._calc_cut_n(x_ub_pong, m_index * 8 + m_index_inner * 2 + 1, block_index,
                                                     n_size, indices_ub, m_index_inner * 4 + 2, src_ub, start_index,
                                                     end_index, mask_scaler, n_tail)
                                with self.tik_instance.else_scope():
                                    # ping
                                    self._calc_cut_n(x_ub_ping, m_index * 8 + m_index_inner * 2, block_index,
                                                     n_size, indices_ub, m_index_inner * 4, src_ub, start_index,
                                                     end_index, mask_scaler, n_tail)

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _calc_cut_n(self, x_ub, x_index, block_index, n_size, indices_ub, indices_index, src_ub, start_index,
                    end_index, mask_scaler, n_tail):
        """
        funtion with cut n only
        """
        n_real_size = n_size if n_tail == 0 else n_tail
        add_loop = math.ceil(n_real_size / 64)
        start_index.set_as(indices_ub[indices_index])
        end_index.set_as(indices_ub[indices_index + 1])
        self.tik_instance.data_move(x_ub, self.x_gm[x_index * self.x_shape[1] + block_index * n_size],
                                    0, 1, n_real_size // 8, 0, 0)
        with self.tik_instance.if_scope(end_index > start_index):
            with self.tik_instance.for_range(start_index, end_index) as index:
                with self.tik_instance.for_range(0, add_loop) as add_index:
                    with self.tik_instance.if_scope(add_index != add_loop - 1):
                        self.tik_instance.vadd(64, x_ub[add_index * 64],
                                               src_ub[index * n_real_size + add_index * 64],
                                               x_ub[add_index * 64],
                                               1, 1, 1, 1, 0, n_real_size // 8, 0)
                    with self.tik_instance.else_scope():
                        mask_scaler.set_as(n_real_size - add_index * 64)
                        self.tik_instance.vadd(mask_scaler, x_ub[add_index * 64],
                                               src_ub[index * n_real_size + add_index * 64],
                                               x_ub[add_index * 64],
                                               1, 1, 1, 1, 0, n_real_size // 8, 0)
        self.tik_instance.data_move(self.x_out_gm[x_index * self.x_shape[1] + block_index * n_size],
                                    x_ub, 0, 1, n_real_size // 8, 0, 0)

    def _calc_cut_size(self):
        """
        funtion to calc cut size
        """
        n_len = int((Constant.UB_SIZE / 4 - 16) / (self.src_shape[0] + 3))
        if n_len > 128:
            n_size = min(128, self.src_shape[1])
        elif n_len > 64:
            n_size = min(64, self.src_shape[1])
        else:
            n_size = min(n_len // 8 * 8, self.src_shape[1])

        n_loop_num = math.ceil(self.src_shape[1] / n_size)
        n_tail = self.src_shape[1] - (n_loop_num - 1) * n_size

        return n_loop_num, n_size, n_tail

    # 'pylint: disable=too-many-locals,too-many-statements
    def _fun_cut_n_and_m(self):
        """
        funtion with src need cut m and n
        """
        if self.src_shape[1] >= 128:
            n_size = 128
        elif self.src_shape[1] >= 64:
            n_size = 64
        else:
            n_size = 8

        n_tail_offset = 0
        n_loop_num = math.ceil(self.src_shape[1] / n_size)
        if self.src_shape[1] % 8 == 0:
            n_tail = self.src_shape[1] - (n_loop_num - 1) * n_size
        else:
            n_real_tail = self.src_shape[1] - (n_loop_num - 1) * n_size
            n_tail = math.ceil(n_real_tail / 8) * 8
            n_tail_offset = n_tail - n_real_tail

        start_index = self.tik_instance.Scalar(dtype="int32")
        end_index = self.tik_instance.Scalar(dtype="int32")
        m_size = 8
        block_dim = math.ceil(self.x_shape[0] / m_size)
        m_tail = self.x_shape[0] - (block_dim - 1) * 8
        m_tail_loop_num = math.ceil(m_tail / 4)
        m_inner_loop_num = math.ceil(m_tail / 2)
        src_ub = self.tik_instance.Tensor(self.src_dtype, (n_size, ), name="src_ub", scope=tik.scope_ubuf)
        x_ub_ping = self.tik_instance.Tensor(self.x_dtype, (n_size, ), name="x_ub_ping", scope=tik.scope_ubuf)
        x_ub_pong = self.tik_instance.Tensor(self.x_dtype, (n_size, ), name="x_ub_pong", scope=tik.scope_ubuf)
        x_ub_bak = self.tik_instance.Tensor(self.x_dtype, (n_size, ), name="x_ub_bak", scope=tik.scope_ubuf)
        indices_ub = self.tik_instance.Tensor(self.indices_dtype, (16, ), name="indices_ub",
                                              scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, block_dim, block_num=block_dim) as block_index:
            with self.tik_instance.if_scope(block_index != block_dim - 1):
                self.tik_instance.data_move(indices_ub, self.indices_gm[block_index * 16], 0, 1, 2, 0, 0)
                with self.tik_instance.for_range(0, 4) as m_index_inner:
                    start_index.set_as(indices_ub[m_index_inner * 4])
                    end_index.set_as(indices_ub[m_index_inner * 4 + 1])
                    # ping
                    self._calc_cut_n_and_m(x_ub_ping, x_ub_bak, m_index_inner * 2, block_index, n_loop_num, src_ub,
                                           start_index, end_index, n_size, n_tail, n_tail_offset)
                    # pong
                    start_index.set_as(indices_ub[m_index_inner * 4 + 2])
                    end_index.set_as(indices_ub[m_index_inner * 4 + 3])
                    self._calc_cut_n_and_m(x_ub_pong, x_ub_bak, m_index_inner * 2 + 1, block_index, n_loop_num, src_ub,
                                           start_index, end_index, n_size, n_tail, n_tail_offset)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(indices_ub, self.indices_gm[block_index * 16], 0, 1, m_tail_loop_num, 0, 0)
                with self.tik_instance.for_range(0, m_inner_loop_num) as m_index_inner:
                    start_index.set_as(indices_ub[m_index_inner * 4])
                    end_index.set_as(indices_ub[m_index_inner * 4 + 1])
                    with self.tik_instance.if_scope(m_index_inner != m_inner_loop_num - 1):
                        # ping
                        self._calc_cut_n_and_m(x_ub_ping, x_ub_bak, m_index_inner * 2, block_index, n_loop_num, src_ub,
                                               start_index, end_index, n_size, n_tail, n_tail_offset)
                        # pong
                        start_index.set_as(indices_ub[m_index_inner * 4 + 2])
                        end_index.set_as(indices_ub[m_index_inner * 4 + 3])
                        self._calc_cut_n_and_m(x_ub_pong, x_ub_bak, m_index_inner * 2 + 1, block_index, n_loop_num,
                                               src_ub, start_index, end_index, n_size, n_tail, n_tail_offset)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(m_tail % 2 == 0):
                            # ping
                            self._calc_cut_n_and_m(x_ub_ping, x_ub_bak, m_index_inner * 2, block_index, n_loop_num,
                                                   src_ub, start_index, end_index, n_size, n_tail, n_tail_offset)
                            # pong
                            start_index.set_as(indices_ub[m_index_inner * 4 + 2])
                            end_index.set_as(indices_ub[m_index_inner * 4 + 3])
                            self._calc_cut_n_and_m(x_ub_pong, x_ub_bak, m_index_inner * 2 + 1, block_index, n_loop_num,
                                                   src_ub, start_index, end_index, n_size, n_tail, n_tail_offset)
                        with self.tik_instance.else_scope():
                            # ping
                            self._calc_cut_n_and_m(x_ub_ping, x_ub_bak, m_index_inner * 2, block_index, n_loop_num,
                                                   src_ub, start_index, end_index, n_size, n_tail, n_tail_offset)

    # 'pylint: disable=too-many-statements
    def _calc_cut_n_and_m(self, x_ub, x_ub_bak, x_index, block_index, n_loop_num, src_ub, start_index, end_index,
                          n_size, n_tail, n_tail_offset):
        """
        funtion with cut n only
        """
        with self.tik_instance.for_range(0, n_loop_num) as n_index_inner:
            with self.tik_instance.if_scope(n_index_inner != n_loop_num - 1):
                self.tik_instance.data_move(x_ub,
                                            self.x_gm[(block_index * 8 + x_index) * self.x_shape[1] + n_index_inner *
                                                      n_size],
                                            0, 1, n_size // 8, 0, 0)
                with self.tik_instance.if_scope(end_index > start_index):
                    with self.tik_instance.for_range(start_index, end_index) as add_index:
                        self.tik_instance.data_move(src_ub,
                                                    self.src_gm[add_index * self.src_shape[1] + n_index_inner * n_size],
                                                    0, 1, n_size // 8, 0, 0)
                        if n_size >= 64:
                            self.tik_instance.vadd(64, x_ub, src_ub, x_ub, n_size // 64, 1, 1, 1, 8, 8, 8)
                        else:
                            self.tik_instance.vadd(8, x_ub, src_ub, x_ub, n_size // 8, 1, 1, 1, 1, 1, 1)
                with self.tik_instance.if_scope(n_index_inner == n_loop_num - 2):
                    self.tik_instance.data_move(x_ub_bak, x_ub, 0, 1, n_size // 8, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.x_out_gm[(block_index * 8 + x_index) * self.x_shape[1] + n_index_inner * n_size],
                        x_ub, 0, 1, n_size // 8, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(x_ub,
                                            self.x_gm[(block_index * 8 + x_index) * self.x_shape[1] + n_index_inner *
                                                      n_size - n_tail_offset],
                                            0, 1, n_tail // 8, 0, 0)
                with self.tik_instance.if_scope(end_index > start_index):
                    with self.tik_instance.for_range(start_index, end_index) as add_index:
                        self.tik_instance.data_move(src_ub,
                                                    self.src_gm[add_index * self.src_shape[1] + n_index_inner *
                                                                n_size - n_tail_offset],
                                                    0, 1, n_tail // 8, 0, 0)
                        if n_size >= 64:
                            self.tik_instance.vadd(64, x_ub, src_ub, x_ub, n_size // 64, 1, 1, 1, 8, 8, 8)
                        else:
                            self.tik_instance.vadd(8, x_ub, src_ub, x_ub, n_tail // 8, 1, 1, 1, 1, 1, 1)
                self.tik_instance.data_move(self.x_out_gm[(block_index * 8 + x_index) * self.x_shape[1] +
                                                          n_index_inner * n_size - n_tail_offset],
                                            x_ub, 0, 1, n_tail // 8, 0, 0)
        if n_loop_num > 1:
            self.tik_instance.data_move(self.x_out_gm[(block_index * 8 + x_index) * self.x_shape[1] +
                                                      (n_loop_num - 2) * n_size],
                                        x_ub_bak, 0, 1, n_size // 8, 0, 0)

    def _fun_small_n(self):
        """
        funtion with n < 8
        """
        src_ub = self.tik_instance.Tensor(self.src_dtype, (8, ), name="src_ub", scope=tik.scope_ubuf)
        x_ub = self.tik_instance.Tensor(self.x_dtype, (8, 8), name="x_ub_ping", scope=tik.scope_ubuf)
        x_ub_bak = self.tik_instance.Tensor(self.x_dtype, (8, 8), name="x_ub_pong", scope=tik.scope_ubuf)
        indices_ub = self.tik_instance.Tensor(self.indices_dtype, (8, 2), name="indices_ub", scope=tik.scope_ubuf)
        start_index = self.tik_instance.Scalar(dtype="int32")
        end_index = self.tik_instance.Scalar(dtype="int32")
        m_loop_num = math.ceil(self.x_shape[0] / 8)
        m_tail = self.x_shape[0] - (m_loop_num - 1) * 8
        m_tail_block = math.ceil(m_tail / 4)
        with self.tik_instance.for_range(0, m_loop_num) as m_loop_index:
            with self.tik_instance.if_scope(m_loop_index == 0):
                with self.tik_instance.if_scope(m_loop_num == 1):
                    with self.tik_instance.for_range(0, m_tail) as index:
                        self.tik_instance.data_move(x_ub[index * 8], self.x_gm[index * self.x_shape[1]],
                                                    0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, 8) as index:
                        self.tik_instance.data_move(x_ub[index * 8], self.x_gm[index * self.x_shape[1]],
                                                    0, 1, 1, 0, 0)
            with self.tik_instance.if_scope(m_loop_index <= m_loop_num - 2):
                with self.tik_instance.if_scope(m_loop_index == m_loop_num - 2):
                    with self.tik_instance.for_range(0, m_tail) as index:
                        self.tik_instance.data_move(x_ub_bak[index * 8],
                                                    self.x_gm[((m_loop_index + 1) * 8 + index) * self.x_shape[1]],
                                                    0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, 8) as m_index:
                        self.tik_instance.data_move(x_ub_bak[m_index * 8],
                                                    self.x_gm[((m_loop_index + 1) * 8 + m_index) * self.x_shape[1]],
                                                    0, 1, 1, 0, 0)
            with self.tik_instance.if_scope(m_loop_index != m_loop_num - 1):
                self.tik_instance.data_move(indices_ub, self.indices_gm[m_loop_index * 16], 0, 1, 2, 0, 0)
                with self.tik_instance.for_range(0, 8) as m_index:
                    start_index.set_as(indices_ub[m_index * 2])
                    end_index.set_as(indices_ub[m_index * 2 + 1])
                    with self.tik_instance.if_scope(end_index > start_index):
                        with self.tik_instance.for_range(start_index, end_index) as add_index:
                            self.tik_instance.data_move(src_ub, self.src_gm[add_index * self.src_shape[1]],
                                                        0, 1, 1, 0, 0)
                            self.tik_instance.vadd(8, x_ub[m_index * 8], src_ub, x_ub[m_index * 8], 1, 1, 1, 1,
                                                   8, 8, 8)
                    self.tik_instance.data_move(self.x_out_gm[(m_loop_index * 8 + m_index) * self.x_shape[1]],
                                                x_ub[m_index * 8], 0, 1, 1, 0, 0)
                self.tik_instance.data_move(x_ub, x_ub_bak, 0, 1, 8, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(indices_ub, self.indices_gm[m_loop_index * 16], 0, 1, m_tail_block, 0, 0)
                with self.tik_instance.for_range(0, m_tail) as m_index:
                    start_index.set_as(indices_ub[m_index * 2])
                    end_index.set_as(indices_ub[m_index * 2 + 1])
                    with self.tik_instance.if_scope(end_index > start_index):
                        with self.tik_instance.for_range(start_index, end_index) as add_index:
                            self.tik_instance.data_move(src_ub, self.src_gm[add_index * self.src_shape[1]],
                                                        0, 1, 1, 0, 0)
                            self.tik_instance.vadd(8, x_ub[m_index * 8], src_ub, x_ub[m_index * 8],
                                                   1, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.data_move(self.x_out_gm[(m_loop_index * 8 + m_index) * self.x_shape[1]],
                                                x_ub[m_index * 8], 0, 1, 1, 0, 0)


# 'pylint: disable=invalid-name,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def add_row_ranges(x, src, indices, x_out, kernel_name="add_row_ranges"):
    """
    Algorithm: x(r, c) += src(j, c), j ranges from indices[r, 0] through indices[r, 1].

    Parameters
    ----------
    x: dict
        shape and datatype
    src: dict
        shape and datatype
    indices: dict
        shape and datatype
    x_out: dict
        The max pooled output tensor.
    kernel_name: str
        kernel_name, default value is 'add_row_ranges'

    Returns
    -------
    add_row_ranges_reslut: reslut of add_row_ranges
    """
    _check_param(x, src, indices, kernel_name)
    add_row_ranges_instance = AddRowRanges(x, src, indices, kernel_name)
    return add_row_ranges_instance.add_row_ranges_compute()
