#!/usr/bin/python
# -*- coding: utf-8 -*-
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
non_zero_with_value
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    common constants
    """
    UB_MINIMUM_SIZE = 32
    SHAPE_DTYPE = "uint32"
    UB_REPEAT_SIZE = 64
    REPEAT_STRIDE = 8
    BLOCK_STRIDE = 1
    dtype_dict = {"float16": 2, "float32": 4, "int32": 4, "uint32": 4, "int64": 8}


def _ceil(x_1, x_2):
    return (x_1 + x_2 - 1) // x_2


class NonZero:
    """Function: use to store nonzero paramters
    """

    def __init__(self, x_shape, x_dtype, y_dtype, kernel_name):
        """Init NonZero base parameters
        """
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.ub_minimum_num = Constant.UB_MINIMUM_SIZE // Constant.dtype_dict.get(x_dtype)
        self.size = x_shape[0] * x_shape[1]
        self.x_shape_one_dim = (self.size,)
        self.tiling = 8192
        if tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // 1024 < 256:
            self.tiling = 4096
        self.multi_core_partition()
        self.init_tensor()

    def init_tensor(self):
        """
        init tensors for ub space and gm space.
        """
        # Number of non_zero elements
        self.num = self.tik_instance.Scalar(Constant.SHAPE_DTYPE, "num", init_value=0)
        # Number of non_zero elements in a single core
        self.num_blk = self.tik_instance.Scalar(Constant.SHAPE_DTYPE, "num_blk")

        self.zero_scalar_uint32 = self.tik_instance.Scalar(init_value=0, dtype=Constant.SHAPE_DTYPE)
        self.zero_scalar_int32 = self.tik_instance.Scalar(init_value=0, dtype=self.y_dtype)
        self.zero_scalar_fp32 = self.tik_instance.Scalar(init_value=0, dtype=self.x_dtype)
        self.scalar_2 = self.tik_instance.Scalar("uint32", "scalar_2", init_value=2)
        self.scalar_1 = self.tik_instance.Scalar("uint32", "scalar_1", init_value=1)

        self.x_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape, name="x", scope=tik.scope_gm)
        # Temporary storage of output data in workspace
        out_last_dim = (self.one_core_num + self.ub_minimum_num - 1) // \
                       self.ub_minimum_num * self.ub_minimum_num
        self.data_out = self.tik_instance.Tensor(self.y_dtype, (self.core_loop, 2, out_last_dim),
                                                 name="data_out",
                                                 scope=tik.scope_gm,
                                                 is_workspace=True)
        self.data_value_out = self.tik_instance.Tensor(self.x_dtype, (self.core_loop, out_last_dim),
                                                       name="data_value_out",
                                                       scope=tik.scope_gm,
                                                       is_workspace=True)
        # Temporary storage of output data in workspace
        self.shape_out = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (self.core_loop, self.ub_minimum_num),
                                                  name="shape_out",
                                                  scope=tik.scope_gm,
                                                  is_workspace=True)

        # Final output data
        self.res_gm = self.tik_instance.Tensor(self.y_dtype, (2, self.num), name="res_gm", scope=tik.scope_gm)
        self.res_value_gm = self.tik_instance.Tensor(self.y_dtype, (self.num,), name="res_value_gm", scope=tik.scope_gm)
        # Final output shape
        self.count_gm = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (8,), name="count_gm", scope=tik.scope_gm)

        self.shape_value_out_gm = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (9,),
                                                           name="shape_value_out_gm",
                                                           scope=tik.scope_gm)

        # The offset of the current core output
        self.offset_gm = self.tik_instance.Scalar(Constant.SHAPE_DTYPE, "offset_gm", init_value=0)
        # Multi-core synchronization Tensor
        self.sync_workspace = self.tik_instance.Tensor("int64", (self.core_loop * 32 // 8,),
                                                       name="barrier_workspace",
                                                       scope=tik.scope_gm,
                                                       is_workspace=True,
                                                       is_atomic_add=True)

        # The number of bytes required for the workspace
        workspace_data_out = Constant.dtype_dict.get(self.y_dtype) * self.core_loop * 2 * self.one_core_num
        workspace_data_value_out = Constant.dtype_dict.get(self.x_dtype) * self.core_loop * 2 * self.one_core_num
        workspace_shape_out = Constant.dtype_dict.get(Constant.SHAPE_DTYPE) * self.core_loop * self.ub_minimum_num
        workspace_sync_barrier = Constant.dtype_dict.get("int64") * (self.core_loop * Constant.UB_MINIMUM_SIZE //
                                                                 Constant.REPEAT_STRIDE)
        self.workspace = [workspace_data_out, workspace_data_value_out, workspace_shape_out, workspace_sync_barrier]

    def multi_core_partition(self):
        """
        Calculate the number of sub-cores, the amount of data calculated per core and
        the amount of data calculated by the last core
        """
        self.core_loop = 8
        self.one_core_num = _ceil(self.size, self.core_loop)
        self.core_loop = _ceil(self.size, self.one_core_num)
        self.last_core_num = self.size - (self.core_loop - 1) * self.one_core_num

    def non_zero_compute(self):
        """
        non_zero_compute
        """
        with self.tik_instance.for_range(0, self.core_loop, block_num=self.core_loop) as blk_idx:
            with self.tik_instance.if_scope(blk_idx < self.core_loop - 1):
                cur_core_num = self.one_core_num
            with self.tik_instance.else_scope():
                cur_core_num = self.last_core_num
            self.compute_one_core(blk_idx, cur_core_num)

            # block_barrier needs to bind more than 1 core
            if self.core_loop > 1:
                self.tik_instance.block_barrier(self.sync_workspace)

            shape_out_ub = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (self.core_loop, self.ub_minimum_num),
                                                    name="shape_out_ub",
                                                    scope=tik.scope_ubuf)
            shape_out_ub_2 = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (8,),
                                                      name="shape_out_ub_2",
                                                      scope=tik.scope_ubuf)

            self._data_move(shape_out_ub, self.shape_out, self.core_loop)

            # Data handling after block_barrier
            self.multi_core_sync(blk_idx, shape_out_ub)

            # The shape_out_ub_2 is (2,2,n), The first number represents the dim number of the output shape
            shape_out_ub_2[0].set_as(self.num)
            self._data_move(self.count_gm, shape_out_ub_2, 1)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm],
                                   outputs=[self.res_value_gm, self.res_gm, self.count_gm])

        return self.tik_instance

    # 'pylint: disable=too-many-locals,too-many-statements
    def compute_one_core(self, blk_idx, cur_core_num):
        """
        compute fuction for each core.
        """
        tiling_loop = _ceil(cur_core_num, self.tiling)
        tiling_tail = cur_core_num - (tiling_loop - 1) * self.tiling
        # The number of non-zero elements in the current core
        # 'pylint: disable=attribute-defined-outside-init,class-attribute-defined-outside-init
        self.res_blk_num_tensor = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (self.ub_minimum_num,),
                                                           name="res_blk_num_tensor",
                                                           scope=tik.scope_ubuf)
        vec_mask = 8
        self.tik_instance.vector_dup(vec_mask, self.res_blk_num_tensor, self.zero_scalar_uint32, Constant.BLOCK_STRIDE,
                                     Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE)

        with self.tik_instance.for_range(0, tiling_loop) as t_idx:
            with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                self.compute_one_loop(blk_idx, t_idx, self.tiling)
            with self.tik_instance.else_scope():
                self.compute_one_loop(blk_idx, t_idx, tiling_tail)

        self._data_move(self.shape_out[blk_idx, 0], self.res_blk_num_tensor, 1)

    def compute_one_loop(self, blk_idx, t_idx, cur_loop_num):
        """
        compute function for each loop.
        """
        # 'pylint: disable=unused-variable
        row, col = self.x_shape
        blk_size = cur_loop_num
        align_num = 64
        all_tail = blk_size % align_num
        # Due to the limitation of the vcmpvs_ne instruction
        # the input elements processed by ub at one time need to be 64 aligned
        blk_align_size = _ceil(blk_size, align_num) * align_num
        x_shape_one_loop = (blk_align_size,)

        x_ub = self.tik_instance.Tensor(self.x_dtype, x_shape_one_loop, name="x_ub", scope=tik.scope_ubuf)

        row_auxiliary_matrix = self.tik_instance.Tensor(self.y_dtype,
                                                        x_shape_one_loop,
                                                        name="row_auxiliary_matrix",
                                                        scope=tik.scope_ubuf)
        col_auxiliary_matrix = self.tik_instance.Tensor(self.y_dtype,
                                                        x_shape_one_loop,
                                                        name="col_auxiliary_matrix",
                                                        scope=tik.scope_ubuf)

        res_blk_num = self.tik_instance.Scalar(dtype=Constant.SHAPE_DTYPE, name="res_blk_num")
        res_blk_num_cur_core = self.tik_instance.Scalar(dtype=Constant.SHAPE_DTYPE, name="res_blk_num_cur_core")

        vreduce_mask = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (blk_align_size // 32,),
                                                name="vreduce_mask",
                                                scope=tik.scope_ubuf)

        # Initialize the auxiliary matrix of rows and columns
        row_auxiliary_matrix = self._build_row_index_mtr(blk_idx, t_idx, col, row_auxiliary_matrix, blk_size)
        col_auxiliary_matrix = self._build_col_index_mtr(blk_align_size, blk_size, col, col_auxiliary_matrix, blk_idx,
                                                         t_idx)

        if all_tail == 0:
            offset = blk_idx * self.one_core_num + t_idx * self.tiling
            self._data_move(x_ub, self.x_gm[offset // col, offset % col], blk_align_size // self.ub_minimum_num)

        else:
            offset = blk_idx * self.one_core_num + t_idx * self.tiling
            dma_burst = blk_size // self.ub_minimum_num
            dma_tail = blk_size % self.ub_minimum_num
            self.v_dup(x_ub, self.zero_scalar_fp32, blk_align_size, self.x_dtype)
            if dma_burst > 0:
                self._data_move(x_ub, self.x_gm[offset // col, offset % col], dma_burst)
            # move input elements that are less than ub_minimun
            if dma_tail > 0:
                gm_offset = dma_burst * self.ub_minimum_num + offset
                ub_offset = dma_burst * self.ub_minimum_num
                unit_tensor = self.tik_instance.Tensor(self.x_dtype, (self.ub_minimum_num,),
                                                       name="unit_tensor",
                                                       scope=tik.scope_ubuf)
                self._data_move(unit_tensor, self.x_gm[gm_offset // col, gm_offset % col], 1)
                with self.tik_instance.for_range(0, dma_tail) as _idx:
                    x_ub[ub_offset + _idx].set_as(unit_tensor[_idx])

        self.gen_mask(vreduce_mask, x_ub, self.zero_scalar_fp32, blk_align_size, self.x_dtype)

        dst_ub_row = self.tik_instance.Tensor(self.y_dtype, (blk_align_size,), name="dst_ub_row", scope=tik.scope_ubuf)
        dst_ub_col = self.tik_instance.Tensor(self.y_dtype, (blk_align_size,), name="dst_ub_col", scope=tik.scope_ubuf)

        dst_ub_value = self.tik_instance.Tensor(self.x_dtype, (blk_align_size,),
                                                name="dst_ub_value",
                                                scope=tik.scope_ubuf)

        # Calculate the row index of non-zero elements
        self.tik_instance.vreduce(blk_align_size, dst_ub_row, row_auxiliary_matrix, vreduce_mask, 1, 1,
                                  Constant.REPEAT_STRIDE, 1, 0, res_blk_num, "counter")
        # Calculate the col index of non-zero elements
        self.tik_instance.vreduce(blk_align_size, dst_ub_col, col_auxiliary_matrix, vreduce_mask, 1, 1,
                                  Constant.REPEAT_STRIDE, 1, 0, None, "counter")

        self.tik_instance.vreduce(blk_align_size, dst_ub_value, x_ub, vreduce_mask, 1, 1, Constant.REPEAT_STRIDE, 1, 0,
                                  None, "counter")

        tail_n = res_blk_num % self.ub_minimum_num
        burst_ub = res_blk_num // self.ub_minimum_num
        res_blk_num_cur_core.set_as(self.res_blk_num_tensor[0])
        data_out_offset = res_blk_num_cur_core
        # move out to workspace
        with self.tik_instance.if_scope(burst_ub > 0):
            self._data_move(self.data_out[blk_idx, 0, data_out_offset], dst_ub_row, burst_ub)
            self._data_move(self.data_out[blk_idx, 1, data_out_offset], dst_ub_col, burst_ub)
            self._data_move(self.data_value_out[blk_idx, data_out_offset], dst_ub_value, burst_ub)

        with self.tik_instance.if_scope(tail_n > 0):
            row_align_tensor = self.tik_instance.Tensor(self.y_dtype, (self.ub_minimum_num,),
                                                        name="row_align_tensor",
                                                        scope=tik.scope_ubuf)
            col_align_tensor = self.tik_instance.Tensor(self.y_dtype, (self.ub_minimum_num,),
                                                        name="col_align_tensor",
                                                        scope=tik.scope_ubuf)
            value_align_tensor = self.tik_instance.Tensor(self.x_dtype, (self.ub_minimum_num,),
                                                          name="value_align_tensor",
                                                          scope=tik.scope_ubuf)

            tail_offset_gm_scalar = self.tik_instance.Scalar(init_value=0, dtype="int64")
            tail_offset_ub_scalar = self.tik_instance.Scalar(init_value=0, dtype="int64")
            tail_offset_gm = burst_ub * self.ub_minimum_num + res_blk_num_cur_core
            tail_offset_ub = burst_ub * self.ub_minimum_num
            tail_offset_gm_scalar.set_as(tail_offset_gm)
            tail_offset_ub_scalar.set_as(tail_offset_ub)
            with self.tik_instance.if_scope(burst_ub == 0):
                tail_offset_gm_scalar.set_as(res_blk_num_cur_core)
                tail_offset_ub_scalar.set_as(0)

            with self.tik_instance.for_range(0, tail_n) as _idx:
                row_align_tensor[_idx].set_as(dst_ub_row[tail_offset_ub + _idx])
                col_align_tensor[_idx].set_as(dst_ub_col[tail_offset_ub + _idx])
                value_align_tensor[_idx].set_as(dst_ub_value[tail_offset_ub + _idx])

            self._data_move(self.data_out[blk_idx, 0, tail_offset_gm], row_align_tensor, 1)
            self._data_move(self.data_out[blk_idx, 1, tail_offset_gm], col_align_tensor, 1)
            self._data_move(self.data_value_out[blk_idx, tail_offset_gm], value_align_tensor, 1)

        # Update the non-zero elements of the current core
        res_blk_num_cur_core.set_as(res_blk_num_cur_core + res_blk_num)
        self.res_blk_num_tensor[0].set_as(res_blk_num_cur_core)

    def v_dup(self, dst, scalar, size, x_dtype):
        """
        vector dump data for each loop.
        """
        unit = 256 // (Constant.dtype_dict.get(x_dtype))
        repeat = size // unit
        left = size % unit
        repeat_loop = repeat // 255
        repeat_left = repeat % 255
        repeat_max_value = 255
        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                self.tik_instance.vector_dup(unit, dst[rpt_idx * repeat_max_value * unit], scalar, repeat_max_value, 1,
                                             Constant.REPEAT_STRIDE)
        if repeat_left > 0:
            self.tik_instance.vector_dup(unit, dst[repeat_loop * repeat_max_value * unit], scalar, repeat_left, 1,
                                         Constant.REPEAT_STRIDE)
        if left > 0:
            self.tik_instance.vector_dup(left, dst[repeat * unit], scalar, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                                         Constant.REPEAT_STRIDE)

    def gen_mask(self, dst, src, scalar, size, x_dtype):
        """
        get the mask for values non_zero elements.
        """
        unit = 256 // (Constant.dtype_dict.get(x_dtype))
        repeat = size // unit
        left = size % unit
        repeat_loop = repeat // 255
        repeat_left = repeat % 255
        repeat_max_value = 255

        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * repeat_max_value * unit
                self.tik_instance.vcmpvs_ne(dst[offset // 32], src[offset], scalar, repeat_max_value, 1,
                                            Constant.REPEAT_STRIDE)
        if repeat_left > 0:
            offset = repeat_loop * repeat_max_value * unit
            self.tik_instance.vcmpvs_ne(dst[offset // 32], src[offset], scalar, repeat_left, 1, Constant.REPEAT_STRIDE)
        if left > 0:
            offset = (repeat - 1) * 64 + left
            self.tik_instance.vcmpvs_ne(dst[offset // 32], src[offset], scalar, 1, 1, Constant.REPEAT_STRIDE)

    # 'pylint: disable=too-many-locals,too-many-statements
    def multi_core_sync(self, blk_idx, shape_out_ub):
        """
        sync data for cores in sync workspace.
        """
        tmp_ub = self.tik_instance.Tensor(self.y_dtype, (2, self.tiling), name="tmp_ub", scope=tik.scope_ubuf)
        value_tmp_ub = self.tik_instance.Tensor(self.x_dtype, (self.tiling,), name="value_tmp_ub", scope=tik.scope_ubuf)
        # Calculate the offset of the current core output
        with self.tik_instance.if_scope(blk_idx > 0):
            with self.tik_instance.for_range(0, blk_idx) as o_idx:
                self.num_blk.set_as(shape_out_ub[o_idx, 0])
                self.offset_gm.set_as(self.offset_gm + self.num_blk)
        # Calculate the number of non-zeor elements
        with self.tik_instance.for_range(0, self.core_loop) as _idx:
            self.num_blk.set_as(shape_out_ub[_idx, 0])
            self.num.set_as(self.num + self.num_blk)
        # The number of non-zeor elements in the current core
        self.num_blk.set_as(shape_out_ub[blk_idx, 0])

        mv_out_loop = self.num_blk // self.tiling
        mv_out_tail = self.num_blk % self.tiling

        with self.tik_instance.if_scope(mv_out_loop > 0):
            with self.tik_instance.for_range(0, mv_out_loop) as mvo_idx:
                mvo_offset = mvo_idx * self.tiling
                # workspace to UB
                self._data_move(tmp_ub[0, 0], self.data_out[blk_idx, 0, mvo_offset], self.tiling // self.ub_minimum_num)
                self._data_move(tmp_ub[1, 0], self.data_out[blk_idx, 1, mvo_offset], self.tiling // self.ub_minimum_num)
                self._data_move(value_tmp_ub, self.data_value_out[blk_idx, mvo_offset],
                                self.tiling // self.ub_minimum_num)

                # UB to GM
                self._data_move(self.res_gm[0, self.offset_gm + mvo_offset], tmp_ub[0, 0],
                                self.tiling // self.ub_minimum_num)
                self._data_move(self.res_gm[1, self.offset_gm + mvo_offset], tmp_ub[1, 0],
                                self.tiling // self.ub_minimum_num)
                self._data_move(self.res_value_gm[self.offset_gm + mvo_offset], value_tmp_ub,
                                self.tiling // self.ub_minimum_num)

        tail_n = mv_out_tail % self.ub_minimum_num
        burst_ub = mv_out_tail // self.ub_minimum_num

        with self.tik_instance.if_scope(mv_out_tail > 0):
            mvo_offset = mv_out_loop * self.tiling
            with self.tik_instance.if_scope(burst_ub > 0):
                # workspace to UB
                self._data_move(tmp_ub[0, 0], self.data_out[blk_idx, 0, mvo_offset], burst_ub)
                self._data_move(tmp_ub[1, 0], self.data_out[blk_idx, 1, mvo_offset], burst_ub)
                self._data_move(value_tmp_ub, self.data_value_out[blk_idx, mvo_offset], burst_ub)
                # UB to GM
                self._data_move(self.res_gm[0, self.offset_gm + mvo_offset], tmp_ub[0, 0], burst_ub)
                self._data_move(self.res_gm[1, self.offset_gm + mvo_offset], tmp_ub[1, 0], burst_ub)
                self._data_move(self.res_value_gm[self.offset_gm + mvo_offset], value_tmp_ub, burst_ub)

            with self.tik_instance.if_scope(tail_n > 0):
                # Case 1, borrow data from the back to prevent tramplling between multiple cores
                with self.tik_instance.if_scope(self.num_blk < 8):
                    row_align_tensor = self.tik_instance.Tensor(self.y_dtype, (self.ub_minimum_num,),
                                                                name="row_align_tensor",
                                                                scope=tik.scope_ubuf)
                    col_align_tensor = self.tik_instance.Tensor(self.y_dtype, (self.ub_minimum_num,),
                                                                name="col_align_tensor",
                                                                scope=tik.scope_ubuf)
                    value_align_tensor = self.tik_instance.Tensor(self.x_dtype, (self.ub_minimum_num,),
                                                                  name="value_align_tensor",
                                                                  scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, tail_n) as _idx:
                        row_align_tensor[_idx].set_as(self.data_out[blk_idx, 0, _idx])
                        col_align_tensor[_idx].set_as(self.data_out[blk_idx, 1, _idx])
                        value_align_tensor[_idx].set_as(self.data_value_out[blk_idx, _idx])

                    next_num_blk = self.tik_instance.Scalar(Constant.SHAPE_DTYPE, "next_num_blk")
                    remain = self.tik_instance.Scalar(Constant.SHAPE_DTYPE, "remain")
                    loop_size = self.tik_instance.Scalar(Constant.SHAPE_DTYPE, "loop_size")
                    remain.set_as(self.ub_minimum_num - tail_n)

                    with self.tik_instance.for_range(blk_idx + 1, self.core_loop) as b_idx:
                        next_num_blk.set_as(self.shape_out[b_idx, 0])

                        with self.tik_instance.if_scope(next_num_blk < remain):
                            loop_size.set_as(next_num_blk)
                        with self.tik_instance.else_scope():
                            loop_size.set_as(remain)
                        with self.tik_instance.for_range(0, loop_size) as n_idx:
                            with self.tik_instance.if_scope(remain > 0):
                                row_align_tensor[8 - remain].set_as(self.data_out[b_idx, 0, n_idx])
                                col_align_tensor[8 - remain].set_as(self.data_out[b_idx, 1, n_idx])
                                value_align_tensor[8 - remain].set_as(self.data_value_out[b_idx, n_idx])
                                remain.set_as(remain - 1)

                    with self.tik_instance.if_scope(remain > 0):
                        with self.tik_instance.for_range(0, self.core_loop) as b_idx:
                            next_num_blk.set_as(self.shape_out[b_idx, 0])

                            with self.tik_instance.if_scope(next_num_blk < remain):
                                loop_size.set_as(next_num_blk)
                            with self.tik_instance.else_scope():
                                loop_size.set_as(remain)
                            with self.tik_instance.for_range(0, loop_size) as n_idx:
                                with self.tik_instance.if_scope(remain > 0):
                                    row_align_tensor[8 - remain].set_as(self.data_out[b_idx, 1, n_idx])
                                    remain.set_as(remain - 1)

                    out_gm_offset = self.offset_gm
                    self._data_move(self.res_gm[0, out_gm_offset], row_align_tensor, 1)
                    self._data_move(self.res_gm[1, out_gm_offset], col_align_tensor, 1)
                    self._data_move(self.res_value_gm[out_gm_offset], value_align_tensor, 1)

                # Case 2, use gm_address_back to prevent tramplling between multiple cores
                with self.tik_instance.else_scope():
                    ub_offset = burst_ub * self.ub_minimum_num
                    gm_offset = (burst_ub - 1) * self.ub_minimum_num + tail_n + mvo_offset

                    self._data_move(tmp_ub[0, ub_offset], self.data_out[blk_idx, 0, gm_offset], 1)
                    self._data_move(tmp_ub[1, ub_offset], self.data_out[blk_idx, 1, gm_offset], 1)
                    self._data_move(value_tmp_ub[ub_offset], self.data_value_out[blk_idx, gm_offset], 1)

                    out_gm_offset = self.offset_gm + gm_offset
                    self._data_move(self.res_gm[0, out_gm_offset], tmp_ub[0, ub_offset], 1)
                    self._data_move(self.res_gm[1, out_gm_offset], tmp_ub[1, ub_offset], 1)
                    self._data_move(self.res_value_gm[out_gm_offset], value_tmp_ub[ub_offset], 1)

    def _data_move(self, dst, src, burst_value):
        """
        data_move
        """
        self.tik_instance.data_move(dst, src, sid=0, nburst=1, burst=burst_value, src_stride=0, dst_stride=0)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _build_row_index_mtr(self, blk_idx, t_idx, col, row_auxiliary_matrix, blk_size):
        """
        build row index matrix for input
        :return: row_auxiliary_matrix
        """
        row_add_tensor = self.tik_instance.Tensor(self.y_dtype, (Constant.UB_REPEAT_SIZE,),
                                                  name="add_tensor",
                                                  scope=tik.scope_ubuf)
        row_offset = self.tik_instance.Scalar("int64",
                                              "row_offset",
                                              init_value=blk_idx * self.one_core_num + t_idx * self.tiling)
        self.tik_instance.vector_dup(Constant.UB_REPEAT_SIZE, row_add_tensor, row_offset // col, Constant.BLOCK_STRIDE,
                                     Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE)
        init_start = self.tik_instance.Scalar("int64", "init_start", init_value=0)
        mask = self.tik_instance.Scalar("int64", "mask", init_value=0)
        repeat_times = self.tik_instance.Scalar("int64", "repeat_times", init_value=0)
        front_nums = self.tik_instance.Scalar("int64", "front_nums", init_value=0)
        cur_tail = self.tik_instance.Scalar("int64", "cur_tail", init_value=0)
        with self.tik_instance.if_scope(row_offset % col != 0):  # for front_nums
            front_nums.set_as(col - (row_offset - row_offset // col * col))
            with self.tik_instance.if_scope(front_nums > blk_size):
                front_nums.set_as(blk_size)
            mask.set_as(Constant.UB_REPEAT_SIZE)
            dst_star = 0
            repeat_times.set_as(front_nums // mask)
            self.tik_instance.vadds(mask, row_auxiliary_matrix[dst_star], row_add_tensor, 0, repeat_times,
                                    Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE, 0)
            cur_tail.set_as(front_nums % Constant.UB_REPEAT_SIZE)
            with self.tik_instance.if_scope(cur_tail != 0):
                self.tik_instance.vadds(cur_tail, row_auxiliary_matrix[repeat_times * Constant.UB_REPEAT_SIZE],
                                        row_add_tensor, 0, 1, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                                        Constant.REPEAT_STRIDE, 0)
            self.tik_instance.vadds(Constant.UB_REPEAT_SIZE, row_add_tensor, row_add_tensor, 1, 1,
                                    Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE, 0)
            init_start.set_as(front_nums)
        no_front_blk_size = self.tik_instance.Scalar("int64", "no_front_blk_size", init_value=blk_size - front_nums)
        build_index_row_loop = self.tik_instance.Scalar("int64",
                                                        "build_index_row_loop",
                                                        init_value=no_front_blk_size // col)
        with self.tik_instance.for_range(0, build_index_row_loop) as row_idx:  # for full-col rows
            cur_loop_nums = col
            mask = min(cur_loop_nums, Constant.UB_REPEAT_SIZE)
            dst_star = row_idx * col + init_start
            repeat_times.set_as(cur_loop_nums // mask)
            self.tik_instance.vadds(mask, row_auxiliary_matrix[dst_star], row_add_tensor, 0, repeat_times,
                                    Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE, 0)
            cur_tail.set_as(cur_loop_nums % Constant.UB_REPEAT_SIZE)
            with self.tik_instance.if_scope(cur_tail != 0):
                self.tik_instance.vadds(cur_tail, row_auxiliary_matrix[dst_star + repeat_times * mask], row_add_tensor,
                                        0, 1, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE, 0)
            self.tik_instance.vadds(Constant.UB_REPEAT_SIZE, row_add_tensor, row_add_tensor, 1, 1,
                                    Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE, 0)

        row_index_tail_nums = self.tik_instance.Scalar("int64",
                                                       "row_index_tail_nums",
                                                       init_value=no_front_blk_size - build_index_row_loop * col)
        with self.tik_instance.if_scope(row_index_tail_nums > 0):  # for last tail
            front_nums.set_as(row_index_tail_nums)
            mask = self.tik_instance.Scalar("int64", "mask", init_value=Constant.UB_REPEAT_SIZE)
            dst_star = self.tik_instance.Scalar("int64", "dst_star", init_value=build_index_row_loop * col + init_start)
            repeat_times.set_as(front_nums // mask)
            self.tik_instance.vadds(mask, row_auxiliary_matrix[dst_star], row_add_tensor, 0, repeat_times,
                                    Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE, 0)
            cur_tail.set_as(front_nums % Constant.UB_REPEAT_SIZE)
            with self.tik_instance.if_scope(cur_tail != 0):
                self.tik_instance.vadds(cur_tail,
                                        row_auxiliary_matrix[dst_star + repeat_times * Constant.UB_REPEAT_SIZE],
                                        row_add_tensor, 0, 1, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                                        Constant.REPEAT_STRIDE, 0)
        return row_auxiliary_matrix

    # 'pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    def _build_col_index_mtr(self, blk_align_size, blk_size, col, col_auxiliary_matrix, blk_idx, t_idx):
        """
        build col index matrix for input
        :return: col_auxiliary_matrix
        """
        col_add_tensor = self.tik_instance.Tensor(self.y_dtype,
                                                  (max(min(blk_align_size, col), Constant.UB_REPEAT_SIZE),),
                                                  name="add_tensor",
                                                  scope=tik.scope_ubuf)
        col_offset = self.tik_instance.Scalar("int64",
                                              "col_offset",
                                              init_value=blk_idx * self.one_core_num + t_idx * self.tiling)
        front_size = self.tik_instance.Scalar("int32", "front_size", init_value=0)
        build_index_col_loop = self.tik_instance.Scalar("int32", "build_index_col_loop", init_value=0)
        with self.tik_instance.if_scope(col_offset % col != 0):
            front_size.set_as(col - (col_offset % col))
            with self.tik_instance.if_scope(front_size > blk_size):
                front_size.set_as(blk_size)
            build_index_col_loop.set_as(build_index_col_loop + 1)
        blk_size_no_front = self.tik_instance.Scalar("int32", "blk_size_no_front", init_value=blk_size - front_size)
        build_index_col_loop.set_as(build_index_col_loop + (blk_size_no_front // col))
        col_tail_nums = self.tik_instance.Scalar("int32", "col_tail_nums", init_value=blk_size_no_front % col)

        with self.tik_instance.if_scope(col_tail_nums != 0):
            build_index_col_loop.set_as(build_index_col_loop + 1)

        col_loop_nums_ald = self.tik_instance.Scalar("int32", "col_loop_nums_ald", init_value=0)
        with self.tik_instance.for_range(0, build_index_col_loop) as col_idx:
            value_start = self.tik_instance.Scalar("int32",
                                                   "value_start",
                                                   init_value=(col_offset + col_loop_nums_ald) % col)
            init_nums = min(min(blk_align_size, col), Constant.UB_REPEAT_SIZE)
            for init_idx in range(0, init_nums):
                col_add_tensor[init_idx].set_as(value_start + init_idx)

            cur_loop_nums = self.tik_instance.Scalar("int32", "cur_loop_nums", init_value=col)
            with self.tik_instance.if_scope(col_idx == 0):
                with self.tik_instance.if_scope(front_size != 0):
                    cur_loop_nums.set_as(front_size)
            with self.tik_instance.if_scope(col_idx == (build_index_col_loop - 1)):
                with self.tik_instance.if_scope(col_tail_nums != 0):
                    cur_loop_nums.set_as(col_tail_nums)

            mulps_align_num = self.tik_instance.Scalar("int32",
                                                       "mulps_align_num",
                                                       init_value=cur_loop_nums // Constant.UB_REPEAT_SIZE)
            mulps_tail = self.tik_instance.Scalar("int32",
                                                  "mulps_tail",
                                                  init_value=mulps_align_num * Constant.UB_REPEAT_SIZE)
            this_col_nums_alr = self.tik_instance.Scalar("int32", "this_col_nums_alr", init_value=0)
            this_add_tensor_nums = self.tik_instance.Scalar("int32", "this_add_tensor_nums", init_value=0)

            with self.tik_instance.if_scope(mulps_align_num > 0):
                self.tik_instance.vadds(Constant.UB_REPEAT_SIZE, col_auxiliary_matrix[col_loop_nums_ald],
                                        col_add_tensor, 0, 1, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                                        Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
                this_col_nums_alr.set_as(Constant.UB_REPEAT_SIZE)
                this_add_tensor_nums.set_as(Constant.UB_REPEAT_SIZE)

            ln_max_loop = self.tik_instance.Scalar("int32",
                                                   "ln_max_loop",
                                                   init_value=mulps_align_num * Constant.UB_REPEAT_SIZE -
                                                   Constant.UB_REPEAT_SIZE)
            start_init_nums = self.tik_instance.Scalar("int32", "start_init_nums", init_value=Constant.UB_REPEAT_SIZE)
            # 'pylint: disable=unused-variable
            with self.tik_instance.for_range(0, ln_max_loop) as emit_idx:
                cur_emit_nums = self.tik_instance.Scalar("int32", "cur_emit_nums", init_value=start_init_nums)
                mask = self.tik_instance.Scalar("int32", "mask", init_value=Constant.UB_REPEAT_SIZE)
                repeat_times = self.tik_instance.Scalar("int32",
                                                        "repeat_times",
                                                        init_value=cur_emit_nums // Constant.UB_REPEAT_SIZE)
                dst_star = self.tik_instance.Scalar("int32",
                                                    "dst_star",
                                                    init_value=col_loop_nums_ald + this_col_nums_alr)
                src_star = self.tik_instance.Scalar("int32", "src_star", init_value=this_add_tensor_nums)
                self.tik_instance.vadds(mask, col_auxiliary_matrix[dst_star], col_add_tensor, cur_emit_nums,
                                        repeat_times, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                                        Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
                self.tik_instance.vadds(mask, col_add_tensor[src_star], col_auxiliary_matrix[dst_star], 0, repeat_times,
                                        Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE,
                                        Constant.REPEAT_STRIDE)
                this_col_nums_alr.set_as(this_col_nums_alr + cur_emit_nums)
                this_add_tensor_nums.set_as(this_add_tensor_nums + cur_emit_nums)
                start_init_nums.set_as(start_init_nums * 2)

                with self.tik_instance.if_scope(start_init_nums >= mulps_tail):
                    ln_max_loop.set_as(0)
                with self.tik_instance.if_scope(2 * start_init_nums > mulps_tail):
                    ln_max_loop.set_as(0)
            mulps_tail.set_as(mulps_align_num * Constant.UB_REPEAT_SIZE - start_init_nums)
            with self.tik_instance.if_scope(mulps_tail > 0):
                mask = self.tik_instance.Scalar("int32", "mask", init_value=Constant.UB_REPEAT_SIZE)
                repeat_times = self.tik_instance.Scalar("int32", "repeat_times", init_value=mulps_tail // mask)
                dst_star = self.tik_instance.Scalar("int32",
                                                    "dst_star",
                                                    init_value=col_loop_nums_ald + this_col_nums_alr)
                self.tik_instance.vadds(mask, col_auxiliary_matrix[dst_star], col_add_tensor, this_col_nums_alr,
                                        repeat_times, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                                        Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
                this_col_nums_alr.set_as(this_col_nums_alr + mulps_tail)
            tail_nums = self.tik_instance.Scalar("int32",
                                                 "tail_nums",
                                                 init_value=cur_loop_nums % Constant.UB_REPEAT_SIZE)
            with self.tik_instance.if_scope(tail_nums > 0):
                self.tik_instance.vadds(tail_nums, col_auxiliary_matrix[col_loop_nums_ald + this_col_nums_alr],
                                        col_add_tensor, this_col_nums_alr, 1, Constant.BLOCK_STRIDE,
                                        Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE, 0)
            col_loop_nums_ald.set_as(col_loop_nums_ald + cur_loop_nums)

        return col_auxiliary_matrix


# 'pylint: disable=too-many-arguments, unused-argument,
def check_supported(x, value, index, count, transpose, kernel_name="non_zero_with_value"):
    """
    check the attr transpose
    Go to AICPU when transpose is False
    """
    if transpose:
        return True, ""

    reason = "the attr transpose if false is not supported!"

    return False, reason


# 'pylint: disable=duplicate-string, too-many-arguments, huawei-too-many-arguments
def _check_input_params(x, value, index, count, transpose, kernel_name="non_zero_with_value"):
    """
    check input parameters
    """
    para_check.check_dtype(x.get("dtype").lower(),
                           ("float32",),
                           param_name="x")
    para_check.check_shape(x.get("shape"), min_rank=2, max_rank=2, param_name="x")

    para_check.check_dtype(value.get("dtype").lower(),
                           ("float32",),
                           param_name="value")
    para_check.check_shape(value.get("shape"), min_rank=1, max_rank=1, param_name="value")

    para_check.check_dtype(index.get("dtype").lower(),
                           ("int32",),
                           param_name="index")
    para_check.check_shape(index.get("shape"), min_rank=1, max_rank=1, param_name="index")

    para_check.check_dtype(count.get("dtype").lower(),
                           ("int32",),
                           param_name="count")
    count_shape = count.get("shape")
    para_check.check_shape(count_shape, min_rank=1, max_rank=1, param_name="count")
    if (count_shape[0] != 1):
        error_manager_vector.raise_err_specific_reson(kernel_name, "count shape should be (1,)")


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,too-many-arguments
@register_operator("NonZeroWithValue")
def non_zero_with_value(x, value, index, count, transpose, kernel_name="non_zero_with_value"):
    """
    return a 2-D tensor where each row is the index for a nonzero value

    Paramters
    ---------
    x: dict
        data of input, support "float32"
    y: dict
        index of output
    kernel_name: str
        kernel_name, default value is "non_zero"

    Returns
    ---------
    tik_instance
    """
    _check_input_params(x, value, index, count, transpose, kernel_name)

    x_shape = x.get("shape")
    if len(x_shape) < 2:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "dim num of x should not be less than 2")

    x_dtype = x.get("dtype").lower()
    y_dtype = "int32"
    obj = NonZero(x_shape, x_dtype, y_dtype, kernel_name)
    tik_instance = obj.non_zero_compute()
    return tik_instance
