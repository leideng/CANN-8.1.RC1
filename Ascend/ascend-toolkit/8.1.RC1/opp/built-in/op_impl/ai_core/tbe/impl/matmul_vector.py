#!/usr/bin/python
# -*- coding: utf-8 -*-
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
matmul_vector
"""
from __future__ import absolute_import
from __future__ import division

from tbe import tvm
from tbe import tik
from tbe.common import platform as cce
from te.platform import insn_cmd
from impl import constant_util as constant
from impl.util.platform_adapter import build_config
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpImplMode
from impl.util import util_common
from impl.util.util_common import write_code
from impl.util.platform_adapter import tbe_platform

from .transpose_d import _do_storage_align
from .transpose_d import _tilling_axis_not_last


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class MatmulVector(object):
    """
    MatmulVector class
    """
    DTYPE_SIZE = {
        'uint8': 1,
        'float16': 2,
        'float32': 4
    }
    # int32's max value
    MAX_SHAPE_SIZE = constant.SHAPE_SIZE_LIMIT
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # 8 bit
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = constant.BLOCK_SIZE
    # 256 bytes
    VECTOR_BYTES = constant.VECTOR_BYTE_SIZE
    MAX_REPEAT_TIMES = constant.MAX_REPEAT_TIMES
    BLOCK_PER_REPEAT = 8
    FP32_DTYPE = "float32"
    NUM_FP32_PER_BLOCK = BLOCK_BYTES // DTYPE_SIZE.get(FP32_DTYPE)
    FP32_REPEAT_SIZE = BLOCK_PER_REPEAT * NUM_FP32_PER_BLOCK

    def __init__(self, shape_a, shape_b, shape_bias):
        """
        Init MatmulVector parameters
        """
        self.tik_inst = tik.Tik()
        self.dtype = self.FP32_DTYPE
        self.m = shape_a[0]
        self.n = shape_b[1]
        self.k = shape_a[1]
        self.bias_flag = True if len(shape_bias) > 0 else False

        self.dsize = self.DTYPE_SIZE.get(self.dtype)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - self.RESERVED_UB_SIZE
        self.block_elems = self.BLOCK_BYTES // self.dsize
        self.vector_elems = self.VECTOR_BYTES // self.dsize

        # tiling args
        self.m0 = None
        self.last_m0 = None
        self.n0 = None
        self.k0 = None
        self.pre_core_m = None
        self.pre_core_n = None
        self.pre_core_k = None
        self.m_pre_core_loops = None
        self.n_pre_core_loops = None
        self.is_atomic_add = False

        self.tensor_bias_ub = None
        self.tensor_b_l1 = None

        # get tiling data
        self._get_tiling_args()
        self._init_gm_tensor()

    def matmul_vector_compute(self):
        """
        main process of matmul_vector
        """
        with self.tik_inst.for_range(0, self.m_core * self.n_core, block_num=self.m_core * self.n_core) as core_i:
            n_core_idx = core_i % self.n_core
            with self.tik_inst.new_stmt_scope():
                # move bias data to ub
                pre_n_align = util_common.align(self.pre_core_n, self.block_elems)
                n_gm_offset = n_core_idx * self.pre_core_n
                if self.bias_flag:
                    self.tensor_bias_ub = self.tik_inst.Tensor(self.dtype, (pre_n_align,), name="tensor_bias_ub",
                                                               scope=tik.scope_ubuf)
                    self.tik_inst.data_move(self.tensor_bias_ub, self.tensor_bias_gm[n_gm_offset], 0, 1,
                                            pre_n_align // self.NUM_FP32_PER_BLOCK, 0, 0)

                # move tensor_b data to l1
                l1_n_expand = self._move_b_2_l1(n_gm_offset)

                with self.tik_inst.for_range(0, self.m_pre_core_loops) as m_idx:
                    with self.tik_inst.for_range(0, self.n_pre_core_loops) as n_idx:
                        pre_core_compute_params = [core_i, m_idx, n_idx, self.m0, self.n0, self.k0, l1_n_expand]
                        self._pre_core_compute(pre_core_compute_params)
                    if self.last_n0 > 0:
                        pre_core_compute_params = [core_i, m_idx, self.n_pre_core_loops, self.m0, self.last_n0, self.k0,
                                                   l1_n_expand]
                        self._pre_core_compute(pre_core_compute_params)
                if self.last_m0 > 0:
                    with self.tik_inst.for_range(0, self.n_pre_core_loops) as n_idx:
                        pre_core_compute_params = [core_i, self.m_pre_core_loops, n_idx, self.last_m0, self.n0, self.k0,
                                                   l1_n_expand]
                        self._pre_core_compute(pre_core_compute_params)
                    if self.last_n0 > 0:
                        pre_core_compute_params = [core_i, self.m_pre_core_loops, self.n_pre_core_loops, self.last_m0,
                                                   self.last_n0, self.k0, l1_n_expand]
                        self._pre_core_compute(pre_core_compute_params)

    def _init_gm_tensor(self):
        """
        init gm tensor
        """
        self.tensor_a_gm = self.tik_inst.Tensor(self.dtype, (self.m, self.k,), name="tensor_a_gm", scope=tik.scope_gm)
        self.tensor_b_gm = self.tik_inst.Tensor(self.dtype, (self.k, self.n,), name="tensor_b_gm", scope=tik.scope_gm)
        if not self.is_atomic_add:
            self.tensor_output_gm = self.tik_inst.Tensor(self.dtype, (self.m, self.n,), name="tensor_output_gm",
                                                         scope=tik.scope_gm)
        else:
            self.tensor_output_gm = self.tik_inst.Tensor(self.dtype, (self.m, self.n,), name="tensor_output_gm",
                                                         scope=tik.scope_gm, is_atomic_add=True)
        if self.bias_flag:
            self.tensor_bias_gm = self.tik_inst.Tensor(self.dtype, (self.n,), name="tensor_bias_gm", scope=tik.scope_gm)
            self.inputs = [self.tensor_a_gm, self.tensor_b_gm, self.tensor_bias_gm]
        else:
            self.inputs = [self.tensor_a_gm, self.tensor_b_gm]

    def _get_core(self, size, core_num, check_flag):
        """
        get core info
        """
        core = 1
        for idx in range(core_num, 0, -1):
            flag = size // idx >= self.block_elems if check_flag else True
            if size % idx == 0 and flag:
                core = idx
                break
        return core

    def _get_m_with_max_n(self, k, n):
        """
        get max m
        """
        k_align = util_common.align(k, self.block_elems)
        max_ub_size = self.ub_size // self.dsize
        tensor_b_ub_size = k_align * n
        tensor_trans_b_ub_size = n * k_align
        m = (max_ub_size - tensor_b_ub_size - tensor_trans_b_ub_size) // (n + k_align)
        return m

    def _get_tiling_args(self):
        """
        get tiling info
        """
        self.k0 = self.k
        m0 = self._get_m_with_max_n(self.k0, self.n)
        if m0 > 0:
            self.n_core = 1
            self.n0 = self.n
            self.m_core = self._get_core(self.m, self.core_num, False)
            m_max = self.m // self.m_core
            self.m0 = m0 if m0 < m_max else m_max
        else:
            self.n_core = self._get_core(self.n, self.core_num, True)
            self.m_core = self._get_core(self.m, self.core_num // self.n_core, False)
            self.n0 = self.block_elems
            self.m0 = self.block_elems
            pre_core_n = self.n // self.n_core
            n_max = pre_core_n // self.block_elems
            m_max = self.m // self.m_core
            for idx in range(1, n_max):
                n0_align = util_common.align(pre_core_n // idx, self.block_elems)
                m0 = self._get_m_with_max_n(self.k0, n0_align)
                if pre_core_n % idx == 0 and m0 > 0:
                    self.n0 = pre_core_n // idx
                    self.m0 = m0 if m0 < m_max else m_max
                    break

        self.pre_core_m = self.m // self.m_core
        self.pre_core_n = self.n // self.n_core
        self.pre_core_k = self.k

        self.m_pre_core_loops = self.pre_core_m // self.m0
        self.n_pre_core_loops = self.pre_core_n // self.n0
        self.last_m0 = self.pre_core_m % self.m0
        self.last_n0 = self.pre_core_n % self.n0

        if self.n0 % self.block_elems != 0 or self.last_n0 % self.block_elems != 0:
            self.is_atomic_add = True

    def _move_b_2_l1(self, tensor_b_gm_offset):
        """
        move tensor b from gm to l1
        """
        l1_n_expand = False
        pre_core_n_align = util_common.align(self.pre_core_n, self.block_elems)
        size_align = util_common.align(self.k0 * self.pre_core_n, self.block_elems)
        self.tensor_b_l1 = self.tik_inst.Tensor(self.dtype, (size_align,), name="tensor_b_l1",
                                                scope=tik.scope_cbuf)
        if self.pre_core_n == self.n:
            size_align = util_common.align(self.k0 * self.pre_core_n, self.block_elems)
            self.tik_inst.data_move(self.tensor_b_l1, self.tensor_b_gm[tensor_b_gm_offset], 0,
                                    1, size_align // self.block_elems, 0, 0)
        elif self.n % self.block_elems == 0 and self.pre_core_n % self.block_elems == 0:
            self.tik_inst.data_move(self.tensor_b_l1,
                                    self.tensor_b_gm[tensor_b_gm_offset], 0,
                                    self.k0, self.pre_core_n // self.block_elems,
                                    (self.n - self.pre_core_n) // self.block_elems, 0)
        else:
            l1_n_expand = True
            self.tensor_b_l1 = self.tik_inst.Tensor(self.dtype, (self.k0 * pre_core_n_align,), name="tensor_b_l1",
                                                    scope=tik.scope_cbuf)
            with self.tik_inst.for_range(0, self.k0) as idx:
                self.tik_inst.data_move(self.tensor_b_l1[idx * pre_core_n_align],
                                        self.tensor_b_gm[tensor_b_gm_offset + idx * self.n], 0,
                                        1, pre_core_n_align // self.block_elems, 0, 0)
        return l1_n_expand

    def _move_b_2_ub(self, n0, tensor_b_ub, tensor_b_l1_offset, l1_n_expand):
        """
        move tensor b from l1 to ub
        """
        n0_align = util_common.align(n0, self.block_elems)
        pre_core_n_align = util_common.align(self.pre_core_n, self.block_elems)
        pre_core_n = self.pre_core_n if not l1_n_expand else pre_core_n_align
        if n0 == self.pre_core_n:
            size_align = util_common.align(self.k0 * n0, self.block_elems) if not l1_n_expand else self.k0 * n0_align
            cur_n0 = n0 if not l1_n_expand else n0_align
            self.tik_inst.data_move(tensor_b_ub, self.tensor_b_l1[tensor_b_l1_offset], 0,
                                    1, size_align // self.block_elems, 0, 0)
        elif n0 % self.block_elems == 0 and pre_core_n % self.block_elems == 0:
            cur_n0 = n0
            self.tik_inst.data_move(tensor_b_ub, self.tensor_b_l1[tensor_b_l1_offset], 0,
                                    self.k0, n0 // self.block_elems,
                                    (pre_core_n - n0) // self.block_elems, 0)
        else:
            size = self.pre_core_n if not l1_n_expand else pre_core_n_align
            cur_n0 = n0_align
            with self.tik_inst.for_range(0, self.k0) as idx:
                self.tik_inst.data_move(tensor_b_ub[idx * n0_align],
                                        self.tensor_b_l1[tensor_b_l1_offset + idx * size],
                                        0, 1, n0_align // self.block_elems, 0, 0)
        return cur_n0

    def _move_a_2_ub(self, m0, tensor_a_ub, tensor_a_gm_offset):
        """
        move tensor a from gm to ub
        """
        if self.k0 % self.block_elems == 0:
            self.tik_inst.data_move(tensor_a_ub, self.tensor_a_gm[tensor_a_gm_offset], 0,
                                    1, m0 * self.k0 // self.block_elems, 0, 0)
        else:
            k0_align = util_common.align(self.k0, self.block_elems)
            for i in range(m0):
                self.tik_inst.data_move(tensor_a_ub[i * k0_align],
                                        self.tensor_a_gm[tensor_a_gm_offset + i * self.k0], 0,
                                        1, k0_align // self.block_elems, 0, 0)

    def _reduce_sum_k_le_64(self, shape_info, tensor_list, offset_list):
        """
        reduce sum k axis, when k in [1, 64]
        """
        k0, n0, rep_stride = shape_info
        src_ub, dst_ub = tensor_list
        src_offset, dst_offset = offset_list
        self.tik_inst.vcadd(k0,
                            dst_ub[dst_offset],
                            src_ub[src_offset],
                            n0,
                            1, 1, rep_stride)

    def _reduce_sum_k_gt_64(self, shape_info, tensor_list, offset_list):
        """
        reduce sum k axis, when k in [64, 4096]
        """
        k0, n0, rep_stride = shape_info
        src_ub, _ = tensor_list
        num = k0 // self.vector_elems
        tail = k0 % self.vector_elems
        if num > 0 and tail > 0:
            self.tik_inst.vadd(tail,
                               src_ub,
                               src_ub,
                               src_ub[num * self.vector_elems],
                               n0,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
        if num > 1:
            for l_i in range(1, num):
                self.tik_inst.vadd(self.vector_elems,
                                   src_ub,
                                   src_ub,
                                   src_ub[l_i * self.vector_elems],
                                   n0,
                                   1, 1, 1, rep_stride, rep_stride, rep_stride)
        self._reduce_sum_k_le_64([self.vector_elems, n0, rep_stride], tensor_list, offset_list)

    def _move_data_out(self, m0, n0, output_offset, tensor_out_ub):
        """
        move data to gm
        """
        n0_align = util_common.align(n0, self.block_elems)
        if n0 % self.block_elems == 0:
            if n0 == self.n:
                self.tik_inst.data_move(self.tensor_output_gm[output_offset], tensor_out_ub, 0, 1,
                                        m0 * n0 // self.NUM_FP32_PER_BLOCK, 0, 0)
            else:
                self.tik_inst.data_move(self.tensor_output_gm[output_offset], tensor_out_ub, 0,
                                        m0, n0 // self.NUM_FP32_PER_BLOCK,
                                        0, (self.n - n0) // self.NUM_FP32_PER_BLOCK)
        else:
            self.tik_inst.set_atomic_add(1)
            for i in range(m0):
                self.tik_inst.data_move(self.tensor_output_gm[output_offset + i * self.n],
                                        tensor_out_ub[i * n0_align], 0,
                                        1, n0_align // self.NUM_FP32_PER_BLOCK,
                                        0, 0)
            self.tik_inst.set_atomic_add(0)

    def _out_tensor_set_zero(self, m0, n0, src_ub):
        """
        set out tensor to zero
        """
        n0_align = util_common.align(n0, self.block_elems)
        repeat = m0 * n0_align // self.vector_elems
        tail = m0 * n0_align % self.vector_elems
        if repeat > 0:
            self.tik_inst.vector_dup(self.vector_elems, src_ub, 0, repeat, 1, 8)
        if tail > 0:
            offset = repeat * self.vector_elems
            self.tik_inst.vector_dup(tail, src_ub[offset], 0, 1, 1, 8)

    # 'pylint: disable=too-many-locals
    def _pre_core_compute(self, pre_core_compute_params):
        """
        compute for one pre core
        """
        core_i, m_idx, n_idx, m0, n0, k0, l1_n_expand = pre_core_compute_params
        m_core_idx = core_i // self.n_core
        n_core_idx = core_i % self.n_core
        k0_align = util_common.align(k0, self.block_elems)
        n0_align = util_common.align(n0, self.block_elems)

        tensor_b_ub = self.tik_inst.Tensor(self.dtype, (k0_align * n0_align,), name="tensor_b_ub", scope=tik.scope_ubuf)
        tensor_trans_b = self.tik_inst.Tensor(self.dtype, (k0_align * n0_align,), name="tensor_trans_b",
                                              scope=tik.scope_ubuf)
        tensor_a_ub = self.tik_inst.Tensor(self.dtype, (m0 * k0_align,), name="tensor_a_ub", scope=tik.scope_ubuf)
        tensor_out_ub = self.tik_inst.Tensor(self.dtype, (m0 * n0_align,), name="tensor_out_ub", scope=tik.scope_ubuf)

        # move tensor_a data to ub
        tensor_a_gm_offset = (m_core_idx * self.pre_core_m + m_idx * self.m0) * self.k
        self._move_a_2_ub(m0, tensor_a_ub, tensor_a_gm_offset)

        # move tensor_b data to ub
        tensor_b_l1_offset = n_idx * self.n0
        cur_n0 = self._move_b_2_ub(n0, tensor_b_ub, tensor_b_l1_offset, l1_n_expand)

        # transpose tensor_b shape (k0, n0) to (n0, k0)
        chw2hwc = False
        channels = cur_n0
        m_len = k0_align
        self.tik_inst.v4dtrans(chw2hwc, tensor_trans_b, tensor_b_ub, m_len, channels)

        if n0 % self.block_elems != 0:
            self._out_tensor_set_zero(m0, n0, tensor_out_ub)

        with self.tik_inst.for_range(0, m0) as i:
            # calculate tensor_a (1, k0) * tensor_b (1, k0)
            vmul_repeat_times = k0 // self.vector_elems
            vmul_last_elems = k0 % self.vector_elems
            vmul_offset = vmul_repeat_times * self.vector_elems
            if vmul_repeat_times > 0:
                for j in range(n0):
                    self.tik_inst.vmul(self.vector_elems,
                                       tensor_b_ub[j * k0_align],
                                       tensor_a_ub[i * k0_align],
                                       tensor_trans_b[j * k0_align],
                                       vmul_repeat_times, 1, 1, 1, 8, 8, 8)
            if vmul_last_elems > 0:
                for j in range(n0):
                    self.tik_inst.vmul(vmul_last_elems,
                                       tensor_b_ub[j * k0_align + vmul_offset],
                                       tensor_a_ub[i * k0_align + vmul_offset],
                                       tensor_trans_b[j * k0_align + vmul_offset],
                                       1, 1, 1, 1, 8, 8, 8)

            # reduce_add to result of multiplying tensor_a and tensor_b
            rep_stride = k0_align // self.block_elems
            if k0 <= self.vector_elems:
                self._reduce_sum_k_le_64([k0, n0, rep_stride], [tensor_b_ub, tensor_out_ub], [0, i * n0_align])
            else:
                self._reduce_sum_k_gt_64([k0, n0, rep_stride], [tensor_b_ub, tensor_out_ub], [0, i * n0_align])

            # calculate add bias, n0 in [1, 64]
            if self.bias_flag:
                repeat_times = n0 // self.vector_elems
                last_elems = n0 % self.vector_elems
                offset = repeat_times * self.vector_elems
                if repeat_times > 0:
                    self.tik_inst.vadd(self.vector_elems, tensor_out_ub[i * n0_align], tensor_out_ub[i * n0_align],
                                       self.tensor_bias_ub[n_idx * self.n0], repeat_times, 1, 1, 1, 8, 8, 8)
                if last_elems > 0:
                    self.tik_inst.vadd(last_elems, tensor_out_ub[i * n0_align + offset],
                                       tensor_out_ub[i * n0_align + offset],
                                       self.tensor_bias_ub[n_idx * self.n0 + offset], 1, 1, 1, 1, 8, 8, 8)
        # move tensor_out data to gm
        output_offset = (m_core_idx * self.pre_core_m + m_idx * self.m0) * self.n + \
                        n_core_idx * self.pre_core_n + n_idx * self.n0
        self._move_data_out(m0, n0, output_offset, tensor_out_ub)


def _check_high_performance_case(shape_a, shape_b, shape_bias, other_info):
    impl_mode, src_type = other_info
    if src_type != "float32":
        return False
    if impl_mode != OpImplMode.HIGH_PERFORMANCE:
        return False

    support_v4dtrans = tbe_platform.api_check_support("tik.v4dtrans", "float32")
    if not support_v4dtrans:
        return False

    high_performance_case_list = (
        ((200, 1680), (1680, 256), (256,)),
        ((200, 256), (256, 128), (128,)),
        ((200, 42), (42, 10), (16,)),
        ((200, 464), (464, 256), (256,)),
        ((200, 128), (128, 64), (64,)),
        ((200, 10), (10, 42), (48,)),
        ((200, 64), (64, 32), (32,)),
    )
    shape_case = (tuple(shape_a), tuple(shape_b), tuple(shape_bias))

    if shape_case in high_performance_case_list:
        return True

    return False


# 'pylint: disable=locally-disabled,unnecessary-lambda
# 'pylint: disable=too-many-locals,too-many-statements,too-many-lines,too-many-branches
def _schedule_large_km_kn(shape, list_computes, src_type):
    """
    Matrix multiplication matmul_vector for KN x KN, schedule for the km_kn when the shape is large
    ----------
    """
    result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([result.op])
    axis_outer = 0
    axis_inner = 1

    schedule[the_result_ub].reorder(the_result_ub.op.reduce_axis[0],
                                    the_result_ub.op.axis[axis_outer],
                                    the_result_ub.op.axis[axis_inner])
    schedule[tensor_a_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_outer])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_outer])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_outer])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_outer])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub], the_result_ub.op.reduce_axis[0])

    if tensor_bais_ub is not None:
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[axis_outer])
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[axis_outer])
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[axis_outer])

    n_axis_inner = _get_tiling_km_kn(shape)

    axis_one = schedule[result].split(result.op.axis[axis_outer], factor=n_axis_inner[0])
    axis_two = schedule[result].split(result.op.axis[axis_inner], factor=n_axis_inner[1])

    m_axis = shape[0]
    core_num = _get_core_num(m_axis)
    if core_num != -1:
        core_facotr = int(m_axis / core_num)
    else:
        core_facotr = int(m_axis)

    core_axis = schedule[result].split(axis_one[0],
                                       factor=core_facotr)

    schedule[result].reorder(core_axis[0], core_axis[1], axis_two[0],
                             axis_one[1], axis_two[1])

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].compute_at(schedule[result], axis_two[0])
    else:
        schedule[the_result_ub].compute_at(schedule[result], axis_two[0])

    if core_num != -1 and (n_axis_inner[1]*core_facotr) % 16 == 0:
        schedule[result].bind(core_axis[0], tvm.thread_axis('blockIdx.x'))

    if src_type == "int32":
        schedule[tensor_result_ub_cast].compute_at(schedule[result], axis_two[0])

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_inner],
                                                  insn_cmd.CAST_ROUND)

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)

    if tensor_bais_ub is not None:
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_inner],
                                                 insn_cmd.CAST)
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_inner + 1],
                                          insn_cmd.MULVS)
    schedule[the_result_ub].emit_insn(the_result_ub.op.axis[axis_inner], insn_cmd.ADD)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_inner],
                                               insn_cmd.ADD)

    schedule[result].emit_insn(axis_two[1], insn_cmd.DMA_COPY)

    return schedule


def _get_tiling_km_kn(shape):
    """
    Matrix multiplication matmul_vector for KN x KN, get the tiling num for M, N, K
    ----------
    """
    # the float32 num take up the four bytes, there float32_size equal four
    float32_size = 4
    ub_size = cce.get_soc_spec(cce.UB_SIZE) / float32_size / 2
    shape_n = shape[1]
    n_axis_inner = shape_n
    n_outter = 1
    min_m_axis = 1

    if _get_restriction_km_kn(min_m_axis, n_axis_inner) < ub_size:
        return min_m_axis, n_axis_inner

    while True:
        if _get_restriction_km_kn(min_m_axis, n_axis_inner) < ub_size:
            break
        n_outter = n_outter + 1

        if shape_n % n_outter != 0:
            n_axis_inner = shape_n // n_outter + 1
        else:
            n_axis_inner = shape_n // n_outter

    return min_m_axis, n_axis_inner


def _get_restriction_km_kn(m_axis_inner, n_axis_inner):
    """
    Matrix multiplication matmul_vector for KN x KN, get the space in ub
    ----------
    """
    # the ub block size is eight*float32_size, there is eight
    block_size = 8

    if n_axis_inner % block_size != 0:
        n_axis_inner = block_size*(n_axis_inner // block_size + 1)

    the_result = m_axis_inner + n_axis_inner + 2*m_axis_inner*n_axis_inner

    return the_result


# 'pylint: disable=locally-disabled,too-many-arguments
def _compute_for_km_kn(tensor_a_ub, tensor_b_ub, shape_a, shape_b, tensor_bais_ub, src_type):
    """
    Matrix multiplication matmul_vector for KN x KN, The compute for MK x NK
    ----------
    """
    # set output shape format is M x N.
    output_shape = (shape_a[1], shape_b[1])
    output_shape_mul = (shape_a[0], shape_a[1], shape_b[1])
    tensor_temp_a = tensor_a_ub
    tensor_temp_b = tensor_b_ub
    tensor_temp_bias = tensor_bais_ub
    if src_type == "int32":
        tensor_temp_a = tvm.compute(shape_a, lambda *i: shape_util.cast(tensor_a_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        tensor_temp_b = tvm.compute(shape_b, lambda *i: shape_util.cast(tensor_b_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        if tensor_bais_ub is not None:
            tensor_temp_bias = tvm.compute(output_shape,
                                           lambda *i: shape_util.cast(tensor_bais_ub(*i), "float32"),
                                           name='tensor_bais_ub_cast')

    the_result_mul_ub = tvm.compute(output_shape_mul,
                                    lambda k, m, n: tensor_temp_b(k, n) * tensor_temp_a(k, m),
                                    name="the_result_mul_ub")

    reduce_k_axis = tvm.reduce_axis((0, shape_a[0]), name="reduce_k_axis")
    the_result_ub = tvm.compute(output_shape,
                                lambda m, n: tvm.sum(the_result_mul_ub[reduce_k_axis, m, n],
                                                     axis=reduce_k_axis), name="the_result_ub")
    the_result_bais_ub = None
    the_result_temp = the_result_ub

    if tensor_bais_ub is not None:
        the_result_bais_ub = tvm.compute(output_shape,
                                         lambda m, n: the_result_ub(m, n) +
                                         tensor_temp_bias(m, n), name="the_result_ub")
        the_result_temp = the_result_bais_ub

    if src_type == "int32":
        tensor_result_ub_cast = tvm.compute(output_shape,
                                            lambda *i: shape_util.cast(the_result_temp(*i), "int32"),
                                            name='tensor_result_ub_cast')
    else:
        tensor_result_ub_cast = the_result_temp

    the_result = tvm.compute(output_shape, lambda *i: tensor_result_ub_cast(*i), name='the_result')

    compute_res = [tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias,
                   the_result_mul_ub, the_result_ub, the_result_bais_ub, the_result]

    return compute_res


def _matmul_new_km_kn_cce(tensor_a, tensor_b, tensor_bais, src_type):
    """
    algorithm: Matrix multiplication matmul_vector for KM x KN situation
    ----------
    """
    shape_a = (tensor_a.shape[0].value, tensor_a.shape[1].value)
    shape_b = (tensor_b.shape[0].value, tensor_b.shape[1].value)
    output_bais = (tensor_a.shape[1].value, tensor_b.shape[1].value, tensor_a.shape[0].value)

    tensor_a_ub = tvm.compute(shape_a, lambda *i: tensor_a(*i), name='tensor_a_ub')
    tensor_b_ub = tvm.compute(shape_b, lambda *i: tensor_b(*i), name='tensor_b_ub')

    if tensor_bais is not None:
        tensor_bais_ub = tvm.compute(output_bais,
                                     lambda m, n: tensor_bais(n), name="tensor_bais_ub")
    else:
        tensor_bais_ub = None

    tensor_temp_a, tensor_temp_b, tensor_result_ub_cast,\
    tensor_temp_bias, the_result_mul_ub,\
    the_result_ub, the_result_bais_ub, the_result = _compute_for_km_kn(tensor_a_ub, tensor_b_ub,
                                                                       shape_a, shape_b,
                                                                       tensor_bais_ub, src_type)

    shape_schedule = (shape_a[1], shape_b[1], shape_a[0])
    schedule = _schedule_large_km_kn(shape_schedule, (the_result, tensor_a_ub, tensor_b_ub,
                                                      the_result_mul_ub,
                                                      the_result_ub, tensor_bais_ub,
                                                      the_result_bais_ub,
                                                      tensor_temp_a,
                                                      tensor_temp_b,
                                                      tensor_result_ub_cast,
                                                      tensor_temp_bias), src_type)
    schedule[tensor_a_ub].double_buffer()
    schedule[tensor_b_ub].double_buffer()

    return schedule, the_result


def _schedule_mini_mk_kn(list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x KN, schedule for mini shape
    ----------
    """
    result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([result.op])
    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    axis_outer = 0
    axis_inner = 1

    schedule[the_result_ub].reorder(the_result_ub.op.axis[axis_outer],
                                    the_result_ub.op.reduce_axis[0],
                                    the_result_ub.op.axis[axis_inner])

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_inner],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_inner],
                                                 insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_inner + 1], \
                              insn_cmd.MULVS)
    schedule[the_result_ub].emit_insn(the_result_ub.op.axis[axis_inner],
                                      insn_cmd.ADD)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_inner],
                                               insn_cmd.ADD)

    schedule[result].emit_insn(result.op.axis[axis_inner], insn_cmd.DMA_COPY)

    return schedule


def _schedule_mid_mk_kn(list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x KN, schedule for mid shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])
    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)


    axis_outer = 0
    axis_inner = 1
    schedule[the_result_ub].reorder(the_result_ub.op.axis[axis_outer],
                                    the_result_ub.op.reduce_axis[0],
                                    the_result_ub.op.axis[axis_inner])
    schedule[tensor_a_ub].compute_at(schedule[the_result_ub],
                                     the_result_ub.op.axis[axis_outer])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_inner])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_ub],
                                           the_result_ub.op.axis[axis_outer])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_inner])

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[axis_outer])
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[axis_outer])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub],
                                           the_result_ub.op.reduce_axis[0])

    if src_type == "int32":
        schedule[the_result_ub].compute_at(schedule[tensor_result_ub_cast],
                                           tensor_result_ub_cast.op.axis[axis_outer])
        if the_result_bais_ub is not None:
            schedule[the_result_bais_ub].compute_at(schedule[tensor_result_ub_cast],
                                                    tensor_result_ub_cast.op.axis[axis_outer])
    elif tensor_bais_ub is not None:
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[axis_outer])

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_inner], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_inner],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_inner], insn_cmd.DMA_COPY)
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_inner],
                                               insn_cmd.ADD)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_inner],
                                                 insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_inner + 1], \
                              insn_cmd.MULVS)
    schedule[the_result_ub].emit_insn(the_result_ub.op.axis[axis_inner],
                                      insn_cmd.ADD)

    schedule[the_result].emit_insn(the_result.op.axis[axis_inner], insn_cmd.DMA_COPY)

    return schedule


def _schedule_large_mk_kn(shape, list_computes, src_type, trans_flag):
    """
    Matrix multiplication matmul_vector for MK x KN, schedule for large shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])
    tiling_number = _get_tiling_mk_kn(shape)

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    schedule[the_result_ub].reorder(the_result_ub.op.axis[0],
                                    the_result_ub.op.reduce_axis[0], the_result_ub.op.axis[1])

    axis_one = schedule[tensor_a_ub].split(tensor_a_ub.op.axis[1], factor=tiling_number[2])

    if src_type == "int32":
        axis_one_cast = schedule[tensor_temp_a].split(tensor_temp_a.op.axis[1],
                                                      factor=tiling_number[2])

    axis_two = schedule[the_result_ub].split(the_result_ub.op.reduce_axis[0],
                                             factor=tiling_number[2])

    schedule[tensor_a_ub].compute_at(schedule[the_result_ub], axis_two[0])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[1])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_ub], axis_two[0])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[1])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub], axis_two[1])

    axis_three = schedule[the_result].split(the_result.op.axis[0],
                                            factor=tiling_number[0])
    axis_four = schedule[the_result].split(the_result.op.axis[1],
                                           factor=tiling_number[1])

    m_axis = shape[0]
    core_num = _get_core_num(m_axis)
    if core_num != -1:
        core_facotr = int(m_axis / core_num)
    else:
        core_facotr = int(m_axis)

    core_axis = schedule[the_result].split(axis_three[0],
                                           factor=core_facotr)

    schedule[the_result].reorder(core_axis[0], core_axis[1], axis_four[0],
                                 axis_three[1], axis_four[1])

    if the_result_bais_ub is not None:
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[0])
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[0])
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[0])
        schedule[the_result_bais_ub].compute_at(schedule[the_result], axis_four[0])
    else:
        schedule[the_result_ub].compute_at(schedule[the_result], axis_four[0])

    if core_num != -1 and tiling_number[1] % 8 == 0 and trans_flag:
        schedule[the_result].bind(core_axis[0], tvm.thread_axis('blockIdx.x'))

    if src_type == "int32":
        schedule[tensor_result_ub_cast].compute_at(schedule[the_result], axis_four[0])
        schedule[tensor_temp_a].emit_insn(axis_one_cast[1], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[1], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[1],
                                                  insn_cmd.CAST_ROUND)

    schedule[tensor_a_ub].emit_insn(axis_one[1], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[1], insn_cmd.DMA_COPY)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[1], insn_cmd.DMA_COPY)
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[1],
                                               insn_cmd.ADD)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[1],
                                                 insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[2], \
                              insn_cmd.MULVS)
    schedule[the_result_ub].emit_insn(the_result_ub.op.axis[1],
                                      insn_cmd.ADD)

    schedule[the_result].emit_insn(axis_four[1], insn_cmd.DMA_COPY)

    return schedule


def _get_tiling_mk_kn(shape):
    """
    Matrix multiplication matmul_vector for MK x KN, get the tiling num for M, N, K
    ----------
    """
    # the float32 num take up the four bytes, there float32_size equal four
    float32_size = 4
    ub_size = cce.get_soc_spec(cce.UB_SIZE) / float32_size / 2
    shape_n = shape[1]
    shape_k = shape[2]
    n_axis_outer = 1
    k_axis_outer = 1
    n_axis_inner = shape_n
    k_axis_inner = shape_k
    min_m_axis = 1
    min_k_axis = 2

    if _get_restriction_mk_kn(min_m_axis, n_axis_inner,
                              min_k_axis, shape_n, shape_k) < ub_size:
        while True:
            if _get_restriction_mk_kn(min_m_axis, n_axis_inner,
                                      k_axis_inner, shape_n, shape_k) < ub_size:
                break
            k_axis_outer = k_axis_outer + 1
            if shape_k % k_axis_outer != 0:
                k_axis_inner = shape_k // k_axis_outer + 1
            else:
                k_axis_inner = shape_k // k_axis_outer
    else:
        while True:
            if _get_restriction_mk_kn(min_m_axis, n_axis_inner,
                                      min_k_axis, shape_n, shape_k) < ub_size:
                k_axis_inner = 2
                break
            n_axis_outer = n_axis_outer + 1
            if shape_n % n_axis_outer != 0:
                n_axis_inner = shape_n // n_axis_outer + 1
            else:
                n_axis_inner = shape_n // n_axis_outer

    return min_m_axis, n_axis_inner, k_axis_inner


def _get_restriction_mk_kn(m_axis_inner, n_axis_inner, k_axis_inner, shape_n, shape_k):
    """
    Matrix multiplication matmul_vector for MK x KN, get the compute space in ub,
    the space is little than us_size
    ----------
    """
    # the ub block size is eight*float32_size, there is eight
    block_size = 8
    n_axis_be_divided = False
    k_axis_be_divided = False

    if shape_n % n_axis_inner != 0:
        n_axis_be_divided = True
        n_axis_remainder = shape_n % n_axis_inner

    if shape_k % k_axis_inner != 0:
        k_axis_be_divided = True
        k_axis_remainder = shape_k % k_axis_inner

    if k_axis_inner % block_size != 0:
        cur_k_axis_inner = block_size*(k_axis_inner // block_size + 1)
    else:
        cur_k_axis_inner = k_axis_inner

    if n_axis_inner % block_size != 0:
        cur_n_axis_inner = block_size*(n_axis_inner // block_size + 1)
    else:
        cur_n_axis_inner = n_axis_inner
    the_result = m_axis_inner*cur_n_axis_inner + cur_k_axis_inner + 2*cur_n_axis_inner

    if n_axis_be_divided:
        the_result = the_result + max(3*n_axis_remainder + k_axis_inner, cur_n_axis_inner)

    if k_axis_be_divided:
        the_result = the_result + k_axis_remainder + cur_n_axis_inner

    return the_result


def _get_schedule_mk_kn(shape, list_compute, src_type, trans_flag):
    """
    Matrix multiplication matmul_vector for MK x KN, choose the schedule for different shape
    ----------
    """

    schedule = _schedule_large_mk_kn(shape, list_compute, src_type, trans_flag)

    return schedule


def _compute_for_mk_kn(tensor_a_ub, tensor_b_ub, shape_a, shape_b, tensor_bais_ub, src_type):
    """
    Matrix multiplication matmul_vector for MK x KN, The compute for MK x KN
    ----------
    """
    output_shape = (shape_a[0], shape_b[1])
    output_shape_mul = (shape_a[0], shape_a[1], shape_b[1])

    tensor_temp_bias = tensor_bais_ub
    tensor_temp_a = tensor_a_ub
    tensor_temp_b = tensor_b_ub

    if src_type == "int32":
        tensor_temp_a = tvm.compute(shape_a, lambda *i: shape_util.cast(tensor_a_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        tensor_temp_b = tvm.compute(shape_b, lambda *i: shape_util.cast(tensor_b_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        if tensor_bais_ub is not None:
            tensor_temp_bias = tvm.compute(output_shape,
                                           lambda *i: shape_util.cast(tensor_bais_ub(*i), "float32"),
                                           name='tensor_bais_ub_cast')

    the_result_mul_ub = tvm.compute(output_shape_mul, \
                             lambda m, k, n: tensor_temp_b(k, n) * tensor_temp_a(m, k), \
                             name="the_result_mul_ub")

    reduce_k_axis = tvm.reduce_axis((0, shape_a[1]), name="reduce_k_axis")
    the_result_ub = tvm.compute(output_shape, \
                         lambda m, n: tvm.sum(the_result_mul_ub[m, reduce_k_axis, n],
                                              axis=reduce_k_axis), name="the_result_ub")
    the_result_bais_ub = None
    the_result_temp = the_result_ub

    if tensor_bais_ub is not None:
        the_result_bais_ub = tvm.compute(output_shape,
                                         lambda m, n: the_result_ub(m, n) +
                                         tensor_temp_bias(m, n),
                                         name="the_result_ub")
        the_result_temp = the_result_bais_ub

    if src_type == "int32":
        tensor_result_ub_cast = tvm.compute(output_shape,
                                            lambda *i: shape_util.cast(the_result_temp(*i), "int32"),
                                            name='tensor_result_ub_cast')
    else:
        tensor_result_ub_cast = the_result_temp

    the_result = tvm.compute(output_shape, lambda *i: tensor_result_ub_cast(*i), name='the_result')

    compute_res = [tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias,
                   the_result_mul_ub, the_result_ub, the_result_bais_ub, the_result]

    return compute_res


def _matmul_new_mk_kn_cce(tensor_a, tensor_b, tensor_bais, src_type):
    """
    algorithm: Matrix multiplication matmul_vector for MK x KN situation
    ----------
    """
    shape_a = (tensor_a.shape[0].value, tensor_a.shape[1].value)
    shape_b = (tensor_b.shape[0].value, tensor_b.shape[1].value)
    output_bais = (tensor_a.shape[0].value, tensor_b.shape[1].value, tensor_a.shape[1].value)

    tensor_a_ub = tvm.compute(shape_a, lambda *i: tensor_a(*i), name='tensor_a_ub')
    tensor_b_ub = tvm.compute(shape_b, lambda *i: tensor_b(*i), name='tensor_b_ub')

    if tensor_bais is not None:
        tensor_bais_ub = tvm.compute(output_bais,
                                     lambda m, n: tensor_bais(n), name="tensor_bais_ub")
    else:
        tensor_bais_ub = None

    tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias, the_result_mul_ub, \
    the_result_ub, the_result_bais_ub, the_result = _compute_for_mk_kn(tensor_a_ub, tensor_b_ub,
                                                                       shape_a, shape_b,
                                                                       tensor_bais_ub, src_type)

    schedule = _get_schedule_mk_kn((shape_a[0], shape_b[1], shape_a[1]), \
                                   (the_result, tensor_a_ub, tensor_b_ub,
                                    the_result_mul_ub, the_result_ub, \
                                    tensor_bais_ub, the_result_bais_ub,
                                    tensor_temp_a, tensor_temp_b,
                                    tensor_result_ub_cast,
                                    tensor_temp_bias,), src_type, True)

    schedule[tensor_a_ub].double_buffer()
    schedule[tensor_b_ub].double_buffer()

    return schedule, the_result


def _schedule_mini_mk_nk(list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK, schedule for mini shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])

    axis_two = 1

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub], the_result_ub.op.axis[axis_two])

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_two], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_two], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_two], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_two], insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_two],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_two], insn_cmd.DMA_COPY)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_two],
                                                 insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_two + 1],
                                          insn_cmd.MUL)
    schedule[the_result_ub].emit_insn(the_result_ub.op.reduce_axis[0],
                                      insn_cmd.REDUCE_SUM)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_two],
                                               insn_cmd.ADD)
    schedule[the_result].emit_insn(the_result.op.axis[axis_two], insn_cmd.DMA_COPY)

    return schedule


def _schedule_mid_mk_nk(list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK schedule for mid shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])

    axis_one = 0
    axis_two = 1

    schedule[tensor_a_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_two])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub],
                                     the_result_mul_ub.op.axis[axis_two])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_two])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           the_result_mul_ub.op.axis[axis_two])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub],
                                           the_result_ub.op.axis[axis_two])

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[axis_one])
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[axis_one])
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[axis_one])
            schedule[the_result_bais_ub].compute_at(schedule[tensor_result_ub_cast],
                                                    tensor_result_ub_cast.op.axis[axis_one])
    elif src_type == "int32":
        schedule[the_result_ub].compute_at(schedule[tensor_result_ub_cast],
                                           tensor_result_ub_cast.op.axis[axis_one])

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    schedule[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[axis_two], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[axis_two], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(tensor_temp_a.op.axis[axis_two],
                                          insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(tensor_temp_b.op.axis[axis_two],
                                          insn_cmd.CAST)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[axis_two],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[axis_two],
                                           insn_cmd.DMA_COPY)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[axis_two],
                                                 insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(the_result_mul_ub.op.axis[axis_two+1],
                                          insn_cmd.MUL)
    schedule[the_result_ub].emit_insn(the_result_ub.op.reduce_axis[0],
                                      insn_cmd.REDUCE_SUM)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[axis_two],
                                               insn_cmd.ADD)
    schedule[the_result].emit_insn(the_result.op.axis[axis_two], insn_cmd.DMA_COPY)

    return schedule


def _single_tiling(shape, schedule, list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK handle some axis
    ----------
    """
    tiling_number = _get_tiling_mk_nk(shape)
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]

    axis_one = schedule[tensor_a_ub].split(tensor_a_ub.op.axis[1],
                                           factor=tiling_number[2])
    axis_two = schedule[tensor_b_ub].split(tensor_b_ub.op.axis[1],
                                           factor=tiling_number[2])

    if src_type == "int32":
        axis_one_cast = schedule[tensor_temp_a].split(tensor_temp_a.op.axis[1],
                                                      factor=tiling_number[2])
        axis_two_cast = schedule[tensor_temp_b].split(tensor_temp_b.op.axis[1],
                                                      factor=tiling_number[2])

    axis_three = schedule[the_result_mul_ub].split(the_result_mul_ub.op.axis[2],
                                                   factor=tiling_number[2])
    axis_four = schedule[the_result_ub].split(the_result_ub.op.reduce_axis[0],
                                              factor=tiling_number[2])

    schedule[tensor_a_ub].compute_at(schedule[the_result_mul_ub], axis_three[0])
    schedule[tensor_b_ub].compute_at(schedule[the_result_mul_ub], axis_three[0])

    if src_type == "int32":
        schedule[tensor_temp_a].compute_at(schedule[the_result_mul_ub],
                                           axis_three[0])
        schedule[tensor_temp_b].compute_at(schedule[the_result_mul_ub],
                                           axis_three[0])

    schedule[the_result_mul_ub].compute_at(schedule[the_result_ub], axis_four[0])
    schedule[tensor_a_ub].emit_insn(axis_one[1], insn_cmd.DMA_COPY)
    schedule[tensor_b_ub].emit_insn(axis_two[1], insn_cmd.DMA_COPY)

    if src_type == "int32":
        schedule[tensor_temp_a].emit_insn(axis_one_cast[1], insn_cmd.CAST)
        schedule[tensor_temp_b].emit_insn(axis_two_cast[1], insn_cmd.CAST)

    schedule[the_result_mul_ub].emit_insn(axis_three[1], insn_cmd.MUL)
    schedule[the_result_ub].emit_insn(axis_four[1], insn_cmd.REDUCE_SUM)


def _schedule_large_mk_nk(shape, list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK schedule for large shape
    ----------
    """
    the_result = list_computes[0]
    tensor_a_ub = list_computes[1]
    tensor_b_ub = list_computes[2]
    the_result_mul_ub = list_computes[3]
    the_result_ub = list_computes[4]
    tensor_bais_ub = list_computes[5]
    the_result_bais_ub = list_computes[6]
    tensor_temp_a = list_computes[7]
    tensor_temp_b = list_computes[8]
    tensor_result_ub_cast = list_computes[9]
    tensor_temp_bias = list_computes[10]

    schedule = tvm.create_schedule([the_result.op])

    tiling_number = _get_tiling_mk_nk(shape)
    _single_tiling(shape, schedule, list_computes, src_type)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].compute_at(schedule[the_result_bais_ub],
                                            the_result_bais_ub.op.axis[0])
        schedule[the_result_ub].compute_at(schedule[the_result_bais_ub],
                                           the_result_bais_ub.op.axis[0])
        if src_type == "int32":
            schedule[tensor_temp_bias].compute_at(schedule[the_result_bais_ub],
                                                  the_result_bais_ub.op.axis[0])

    axis_one = schedule[the_result].split(the_result.op.axis[0], factor=tiling_number[0])
    axis_two = schedule[the_result].split(the_result.op.axis[1], factor=tiling_number[1])

    m_axis = shape[0]
    core_num = _get_core_num(m_axis)

    if core_num != -1:
        core_factor = int(m_axis / core_num)
    else:
        core_factor = int(m_axis)

    core_axis = schedule[the_result].split(axis_one[0], factor=core_factor)

    schedule[the_result].reorder(core_axis[0], core_axis[1], axis_two[0], axis_one[1], axis_two[1])

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].compute_at(schedule[the_result], axis_two[0])
    else:
        schedule[the_result_ub].compute_at(schedule[the_result], axis_two[0])

    align_size = tiling_number[1] + 8
    schedule[the_result_ub].storage_align(the_result_ub.op.axis[0], align_size, 0)

    if src_type == "int32":
        schedule[tensor_result_ub_cast].compute_at(schedule[the_result], axis_two[0])

    if core_num != -1 and (core_factor*tiling_number[1]) % 8 == 0 and tiling_number[1] % 8 == 0:
        schedule[the_result].bind(core_axis[0], tvm.thread_axis('blockIdx.x'))

    schedule[tensor_a_ub].set_scope(cce.scope_ubuf)
    schedule[tensor_b_ub].set_scope(cce.scope_ubuf)

    if src_type == "int32":
        schedule[tensor_temp_a].set_scope(cce.scope_ubuf)
        schedule[tensor_temp_b].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].set_scope(cce.scope_ubuf)
        schedule[tensor_result_ub_cast].emit_insn(tensor_result_ub_cast.op.axis[1],
                                                  insn_cmd.CAST_ROUND)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].set_scope(cce.scope_ubuf)
        if src_type == "int32":
            schedule[tensor_temp_bias].set_scope(cce.scope_ubuf)

    schedule[the_result_mul_ub].set_scope(cce.scope_ubuf)
    schedule[the_result_ub].set_scope(cce.scope_ubuf)


    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].set_scope(cce.scope_ubuf)

    if tensor_bais_ub is not None:
        schedule[tensor_bais_ub].emit_insn(tensor_bais_ub.op.axis[1], insn_cmd.DMA_COPY)
        if src_type == "int32":
            schedule[tensor_temp_bias].emit_insn(tensor_temp_bias.op.axis[1],
                                                 insn_cmd.CAST)

    if the_result_bais_ub is not None:
        schedule[the_result_bais_ub].emit_insn(the_result_bais_ub.op.axis[1],
                                               insn_cmd.ADD)

    schedule[the_result].emit_insn(axis_two[1], insn_cmd.DMA_COPY)

    return schedule


def _get_core_num(m_axis):
    """

    :param m_axis:
    :return:
    """
    res = -1
    if m_axis > 32 or m_axis == 32:
        for i in range(32, -1, -1):
            if (m_axis % i) == 0:
                res = i
                break

    return res


def _get_tiling_mk_nk(shape):
    """
    Matrix multiplication matmul_vector for MK x NK get the tiling num for M, N, K
    ----------
    """
    # the float32 num take up the four bytes, there float32_size equal four
    float32_size = 4
    ub_size = cce.get_soc_spec(cce.UB_SIZE) / float32_size / 2
    shape_n = shape[1]
    shape_k = shape[2]
    n_axis_outer = 1
    k_axis_outer = 1
    n_axis_inner = shape_n
    k_axis_inner = shape_k

    min_m_axis = 1
    if shape_n % 8 == 0:
        min_n_axis = 8
    else:
        min_n_axis = 2

    if _get_restraint_mk_nk(min_m_axis, n_axis_inner, k_axis_inner) < ub_size:
        return min_m_axis, n_axis_inner, k_axis_inner

    if _get_restraint_mk_nk(min_m_axis, min_n_axis, k_axis_inner) < ub_size:
        while True:
            if _get_restraint_mk_nk(min_m_axis, n_axis_inner, k_axis_inner) < ub_size:
                m_axis_inner = 1
                break
            n_axis_outer = n_axis_outer + 1
            if shape_n % n_axis_outer != 0:
                n_axis_inner = shape_n // n_axis_outer + 1
            else:
                n_axis_inner = shape_n // n_axis_outer
    else:
        while True:
            if _get_restraint_mk_nk(min_m_axis, min_n_axis, k_axis_inner) < ub_size:
                m_axis_inner = 1
                n_axis_inner = min_n_axis
                break
            k_axis_outer = k_axis_outer + 1
            if shape_k % k_axis_outer != 0:
                k_axis_inner = shape_k // k_axis_outer + 1
            else:
                k_axis_inner = shape_k // k_axis_outer

    if shape_n == 21128:
        n_axis_inner = 7040

    return m_axis_inner, n_axis_inner, k_axis_inner


def _get_restraint_mk_nk(m_axis_inner, n_axis_inner, k_axis_inner):
    """
    Matrix multiplication matmul_vector for MK x NK get the space in ub
    ----------
    """
    # the ub block size is eight*float32_size, there is eight
    block_size = 8

    if k_axis_inner % block_size != 0:
        k_axis_inner = block_size*(k_axis_inner // block_size + 1)

    if n_axis_inner % block_size != 0:
        n_axis_inner = block_size*(n_axis_inner // block_size + 1)

    the_result = m_axis_inner*n_axis_inner + 3*k_axis_inner + 3*n_axis_inner

    return the_result


def _get_schedule_mk_nk(shape, list_computes, src_type):
    """
    Matrix multiplication matmul_vector for MK x NK choose the schedule for different shape
    ----------
    """

    schedule = _schedule_large_mk_nk(shape, list_computes, src_type)

    return schedule


def _compute_for_mk_nk(tensor_a_ub, tensor_b_ub, shape_a, shape_b, tensor_bais_ub, src_type):
    """
    The compute for Matrix multiplication MK x NK Situation
    ----------
    """
    output_shape = (shape_a[0], shape_b[0])
    output_shape_mul = (shape_a[0], shape_b[0], shape_a[1])
    tensor_temp_a = tensor_a_ub
    tensor_temp_b = tensor_b_ub
    tensor_temp_bias = tensor_bais_ub

    if src_type == "int32":
        tensor_temp_a = tvm.compute(shape_a, lambda *i: shape_util.cast(tensor_a_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        tensor_temp_b = tvm.compute(shape_b, lambda *i: shape_util.cast(tensor_b_ub(*i), "float32"),
                                    name='tensor_a_ub_cast')
        if tensor_bais_ub is not None:
            tensor_temp_bias = tvm.compute(output_shape,
                                           lambda *i: shape_util.cast(tensor_bais_ub(*i), "float32"),
                                           name='tensor_bais_ub_cast')

    the_result_mul_ub = tvm.compute(output_shape_mul, lambda m, n, k:
                                    tensor_temp_a(m, k)*tensor_temp_b(n, k),
                                    name="the_result_mul_ub")
    reduce_k_axis = tvm.reduce_axis((0, shape_a[1]), name="reduce_k_axis")
    the_result_ub = tvm.compute(output_shape, \
                                lambda m, n: tvm.sum(the_result_mul_ub[m, n, reduce_k_axis],
                                                     axis=reduce_k_axis), name="the_result_ub")
    the_result_bais_ub = None
    the_result_temp = the_result_ub

    if tensor_bais_ub is not None:
        the_result_bais_ub = tvm.compute(output_shape,
                                         lambda m, n: the_result_ub(m, n) +
                                         tensor_temp_bias(m, n), name="the_result_ub")
        the_result_temp = the_result_bais_ub

    if src_type == "int32":
        tensor_result_ub_cast = tvm.compute(output_shape, lambda *i: shape_util.cast(the_result_temp(*i),
                                                                               "int32"),
                                            name='tensor_result_ub_cast')
    else:
        tensor_result_ub_cast = the_result_temp

    the_result = tvm.compute(output_shape, lambda *i: tensor_result_ub_cast(*i), name='the_result')

    compute_res = [tensor_temp_a, tensor_temp_b, tensor_result_ub_cast,
                   tensor_temp_bias, the_result_mul_ub, the_result_ub, the_result_bais_ub, the_result]

    return compute_res


def _matmul_new_mk_nk_cce(tensor_a, tensor_b, tensor_bais, src_type):
    """
    algorithm: Matrix multiplication matmul_vector for MK x NK Situation
    ----------
    """
    shape_a = (tensor_a.shape[0].value, tensor_a.shape[1].value)
    shape_b = (tensor_b.shape[0].value, tensor_b.shape[1].value)
    output_bais = (shape_a[0], shape_b[0])

    tensor_a_ub = tvm.compute(shape_a, lambda *i: tensor_a(*i), name='tensor_a_ub')
    tensor_b_ub = tvm.compute(shape_b, lambda *i: tensor_b(*i), name='tensor_b_ub')

    if tensor_bais is not None:
        tensor_bais_ub = tvm.compute(output_bais, \
                                     lambda m, n: tensor_bais(n), name="tensor_bais_ub")
    else:
        tensor_bais_ub = None

    tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias, the_result_mul_ub, \
    the_result_ub, the_result_bais_ub, the_result = _compute_for_mk_nk(tensor_a_ub, tensor_b_ub,
                                                                       tensor_a.shape,
                                                                       tensor_b.shape,
                                                                       tensor_bais_ub, src_type)

    schedule_shape = (tensor_a.shape[0].value, tensor_b.shape[0].value, tensor_a.shape[1].value)
    schedule = _get_schedule_mk_nk(schedule_shape, \
                              (the_result, tensor_a_ub, tensor_b_ub, the_result_mul_ub,
                               the_result_ub, \
                               tensor_bais_ub, the_result_bais_ub,
                               tensor_temp_a, tensor_temp_b,
                               tensor_result_ub_cast,
                               tensor_temp_bias), src_type)

    schedule[tensor_a_ub].double_buffer()
    schedule[tensor_b_ub].double_buffer()

    return schedule, the_result


def _tranpose_schedule(schedule, data, data_ub, res, shape_res, dtype):
    """
    Schedule permutes the dimensions and the last axis is not transposed

    Parameters
    ----------
    """
    sch = schedule
    sch[data_ub].set_scope(cce.scope_ubuf)

    split_axis, split_factor = _tilling_axis_not_last(shape_res, dtype)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis], factor=split_factor)
    sch[data_ub].compute_at(sch[res], axis_outer)
    sch[data_ub].emit_insn(data_ub.op.axis[split_axis], insn_cmd.DMA_COPY)
    sch = _do_storage_align(sch, data_ub, shape_res, dtype)
    sch[res].emit_insn(axis_inner, insn_cmd.DMA_COPY)
    tensor_list = [data, res]

    return sch, tensor_list


def _tranpose_notchange_last(data, shape_res, perm, dtype):
    """
    permutes the dimensions and the last axis is not transposed

    Parameters
    ----------
    """

    def _perm_to_flag(perm):
        """
        get the flag for permutation according to perm

        """
        flag = perm[:]# 'pylint: disable=unnecessary-comprehension
        for i, item in enumerate(perm):
            flag[item] = i

        return flag

    def _permute(*index):
        """
        function of permute the dimensions of data

        """
        for i, item in enumerate(_perm_to_flag(perm)):
            if i == 0:
                res_axis = (index[item],)
            else:
                res_axis = res_axis + (index[item],)

        return res_axis
    ub_name = ["data_ub_1", "data_ub_2"]
    res_name = ["res_1", "res_2"]
    if dtype == "1":
        data_ub = tvm.compute(shape_res, lambda *index: data(*_permute(*index)), name=ub_name[0])
        res = tvm.compute(shape_res, lambda *index: data_ub(*index), name=res_name[0])
    else:
        data_ub = tvm.compute(shape_res, lambda *index: data(*_permute(*index)), name=ub_name[1])
        res = tvm.compute(shape_res, lambda *index: data_ub(*index), name=res_name[1])
    return res, data_ub


def _matmul_new_km_nk_cce(tensor_a_pre, tensor_b_pre, tensor_bais, src_type):
    """
    algorithm: Matrix multiplication matmul_vector for KM x NK Situation
    ----------
    """
    # transpose A
    tensor_a, data_ub_a = _tranpose_notchange_last(tensor_a_pre, (tensor_a_pre.shape[1].value,
                                                                  tensor_a_pre.shape[0].value, 1),
                                                   [1, 0, 2], "1")
    # transpose B
    tensor_b, data_ub_b = _tranpose_notchange_last(tensor_b_pre, (tensor_b_pre.shape[1].value,
                                                                  tensor_b_pre.shape[0].value, 1),
                                                   [1, 0, 2], src_type)
    # create schedule
    shape_a = [tensor_a.shape[0].value, tensor_a.shape[1].value]
    shape_b = [tensor_b.shape[0].value, tensor_b.shape[1].value]

    output_bais = (tensor_a.shape[0].value, tensor_b.shape[1].value, tensor_a.shape[1].value)

    tensor_a_ub = tvm.compute(shape_a, lambda m, j: tensor_a(m, j, 0), name='tensor_a_ub')
    tensor_b_ub = tvm.compute(shape_b, lambda m, j: tensor_b(m, j, 0), name='tensor_b_ub')

    if tensor_bais is not None:
        tensor_bais_ub = tvm.compute(output_bais,
                                     lambda m, n: tensor_bais(n), name="tensor_bais_ub")
    else:
        tensor_bais_ub = None

    tensor_temp_a, tensor_temp_b, tensor_result_ub_cast, tensor_temp_bias, the_result_mul_ub, \
    the_result_ub, the_result_bais_ub, the_result = _compute_for_mk_kn(tensor_a_ub, tensor_b_ub,
                                                                       shape_a, shape_b,
                                                                       tensor_bais_ub, src_type)
    schedule = _get_schedule_mk_kn((shape_a[0], shape_b[1], shape_a[1]), \
                                   (the_result, tensor_a_ub, tensor_b_ub,
                                    the_result_mul_ub, the_result_ub, \
                                    tensor_bais_ub, the_result_bais_ub,
                                    tensor_temp_a, tensor_temp_b,
                                    tensor_result_ub_cast,
                                    tensor_temp_bias,), src_type, False)

    shape_a.append(1)
    shape_b.append(1)

    _tranpose_schedule(schedule, tensor_a_pre, data_ub_a, tensor_a, shape_a, src_type)
    _tranpose_schedule(schedule, tensor_b_pre, data_ub_b, tensor_b, shape_b, src_type)

    sch_res = [schedule, the_result, tensor_a, tensor_b]

    return sch_res


# 'pylint: disable=locally-disabled,too-many-arguments
def matmul_vector_cce(shape_a, shape_b, para_dict, src_type, shape_bias):
    """
    algorithm: matmul_vector
    calculating  matrix multiplication with bias, use vector mode ,C = A*B + bias

    Parameters
    ----------
    shape_a : list or tuple
        shape of tensor_a
    shape_b : list or tuple
        shape of tensor_b
    para_dict : dict
        para dict of matmul
    src_type : str
        the data type, assume src_dtype equals dst_dtype,
        only support float32
    shape_bias : list or tuple
        the shape of tensor_bias

    Returns
    -------
    None
    """
    tensor_bias = None
    trans_flag = para_dict.get("trans_a") and para_dict.get("trans_b")
    impl_mode = para_dict.get("impl_mode")

    if trans_flag:
        shape_a = list(shape_a)
        shape_b = list(shape_b)

        shape_a.append(1)
        shape_b.append(1)

        tensor_a = tvm.placeholder(shape_a, name='tensor_a', dtype=src_type)
        tensor_b = tvm.placeholder(shape_b, name='tensor_b', dtype=src_type)
    else:
        tensor_a = tvm.placeholder(shape_a, name='tensor_a', dtype=src_type)
        tensor_b = tvm.placeholder(shape_b, name='tensor_b', dtype=src_type)

    if len(shape_bias) > 0:
        tensor_bias = tvm.placeholder(shape_bias, name='tensor_bias', dtype=src_type)

    tensor_a_gm = None
    tensor_b_gm = None

    if para_dict.get("trans_a"):
        if para_dict.get("trans_b"):
            trans_flag = True
            schedule, the_result, tensor_a_gm, \
            tensor_b_gm = _matmul_new_km_nk_cce(tensor_a, tensor_b,
                                                tensor_bias, src_type)
        else:
            schedule, the_result = _matmul_new_km_kn_cce(tensor_a, tensor_b, tensor_bias, src_type)
    elif para_dict.get("trans_b"):
        schedule, the_result = _matmul_new_mk_nk_cce(tensor_a, tensor_b, tensor_bias, src_type)
    else:
        if _check_high_performance_case(shape_a, shape_b, shape_bias, [impl_mode, src_type]):
            obj = MatmulVector(shape_a, shape_b, shape_bias)
            obj.matmul_vector_compute()
            tik_inst = obj.tik_inst
            tik_inst.BuildCCE(kernel_name=para_dict.get("kernel_name"),
                              inputs=obj.inputs,
                              outputs=(obj.tensor_output_gm,))
            return
        schedule, the_result = _matmul_new_mk_kn_cce(tensor_a, tensor_b, tensor_bias, src_type)

    if tensor_bias is not None:
        schedule[tensor_bias].double_buffer()

    if tensor_bias is not None:
        build_list = [tensor_a, tensor_b, tensor_bias, the_result]
    else:
        build_list = [tensor_a, tensor_b, the_result]

    if trans_flag:
        build_list = [tensor_a, tensor_b, the_result, tensor_a_gm, tensor_b_gm]

    with build_config():
        tvm.build(schedule, build_list, "cce", name=para_dict.get("kernel_name"))
    if trans_flag:
        wk_size_a = shape_a[0] * shape_a[1] * 4
        wk_size_b = shape_b[0] * shape_b[1] * 4
        workspace_dict = {"workspace": {"num": 2, "size": [wk_size_a, wk_size_b]}}
        write_code(workspace_dict, para_dict.get("kernel_name"))
