"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

concat_v2_d: Concatenates tensors along one dimension.
The number of dimensions of input tensors must match,
and all dimensions except 'axis' must be equal.
tf ConcactV2 op

"""
# 'pylint: disable=too-many-lines
from __future__ import absolute_import
import math

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_tik_comm_func import gm2ub
from impl.util.util_tik_comm_func import ub2gm
from impl.util.util_tik_comm_func import ceil_div
from impl import common_util
from impl import constant_util as constant
from impl.util.platform_adapter import tbe_context
from impl.util import util_common


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    MAX_SIZE = 2 ** 31 - 1
    VNCHW_BLOCK_SIZE = 512
    VNCHW_ELEMENT_FP16 = 256
    DATA_MOVE_MAX_REPEAT = 4095
    MAX_REPEAT_TIMES = 255


# 'pylint: disable=cell-var-from-loop
def ceil_32bytes_align_count(count, dtype):
    """
    ceil_32bytes_align_count
    """
    type_size = common_util.get_data_size(dtype)
    block_elements = constant.BLOCK_SIZE // type_size
    block_count = math.ceil(count / block_elements)
    return block_count * block_elements


# 'pylint: disable=too-many-arguments,too-many-locals
def _get_mask2concat_ub(instance: tik.Tik, count, src_index, dtype):
    """
    get 128bit mask for concat ub
    :param instance: tik instance
    :param count: count of element to concat from src ub to dst ub
    :param src_index: the index to concat
    :param dtype: dtype
    :return: [h_64_mask, l_64_mask]
    """
    dtype_size = common_util.get_data_size(dtype)
    if not tbe_platform.api_check_support("tik.vadd", dtype):
        ori_dtype_size = common_util.get_data_size(dtype)
        covert_dtype_map = {
            1: "float16",
            2: "float16",
            4: "float32",
            8: "float32",
        }

        dtype_size = common_util.get_data_size(covert_dtype_map[ori_dtype_size])
        count = count * ori_dtype_size // dtype_size
        src_index = ceil_div(src_index * ori_dtype_size, dtype_size)

    # `{dtype_size: (max_hight_mask, max_low_mask)}`
    dtype_vadd_mask_map = {
        1: (2 ** 64 - 1, 2 ** 64 - 1),
        2: (2 ** 64 - 1, 2 ** 64 - 1),
        4: (0, 2 ** 64 - 1)
    }

    block_element = constant.BLOCK_SIZE // dtype_size
    dst_reserve = src_index % block_element
    repeat_times = count // (block_element * 8)
    if isinstance(count, int) and isinstance(src_index, int):
        h_64_mask = dtype_vadd_mask_map[dtype_size][0]
        l_64_mask = dtype_vadd_mask_map[dtype_size][1]
        l_64_mask = l_64_mask & (l_64_mask << dst_reserve)
        if repeat_times == 0 and count != block_element * 8:
            if count > 64:
                h_64_mask = h_64_mask & ((1 << (count - 64)) - 1)
            else:
                h_64_mask = 0
                l_64_mask = l_64_mask & ((1 << count) - 1)
        return [h_64_mask, l_64_mask]

    h_64_mask = instance.Scalar(dtype="int64", name="h_64_mask", init_value=dtype_vadd_mask_map[dtype_size][0])
    l_64_mask = instance.Scalar(dtype="int64", name="l_64_mask", init_value=dtype_vadd_mask_map[dtype_size][1])
    scalar_one = instance.Scalar(dtype="int64", name="scalar_one", init_value=1)
    l_64_mask.set_as(l_64_mask & (l_64_mask << dst_reserve))
    with instance.if_scope(repeat_times == 0):
        with instance.if_scope(count != block_element * 8):
            with instance.if_scope(count > 64):
                h_64_mask.set_as(h_64_mask & ((scalar_one << (count - 64)) - 1))
            with instance.else_scope():
                h_64_mask.set_as(0)
                with instance.if_scope(count != 64):
                    l_64_mask.set_as(l_64_mask & ((scalar_one << count) - 1))
                with instance.else_scope():
                    l_64_mask.set_as(l_64_mask & -1)
    return [h_64_mask, l_64_mask]


# 'pylint: disable=invalid-name
def _vadd(instance: tik.Tik, mask, dst: tik.Tensor, src0: tik.Tensor, src1: tik.Tensor, repeat_times,
          dst_blk_stride, src0_blk_stride, src1_blk_stride,
          dst_rep_stride, src0_rep_stride, src1_rep_stride):
    """
    _vadd
    """
    max_repeat_stride = 255
    dtype_size = common_util.get_data_size(dst.dtype)
    block_element = constant.BLOCK_SIZE // dtype_size
    with instance.if_scope(dst_rep_stride <= max_repeat_stride):
        loop_times = repeat_times // Constant.MAX_REPEAT_TIMES
        tail_times = repeat_times % Constant.MAX_REPEAT_TIMES
        with instance.for_range(0, loop_times) as loop_index:
            instance.vadd(mask,
                          dst[dst_rep_stride * block_element * Constant.MAX_REPEAT_TIMES * loop_index],
                          src0[src0_rep_stride * block_element * Constant.MAX_REPEAT_TIMES * loop_index],
                          src1[src1_rep_stride * block_element * Constant.MAX_REPEAT_TIMES * loop_index],
                          Constant.MAX_REPEAT_TIMES,
                          dst_blk_stride, src0_blk_stride, src1_blk_stride,
                          dst_rep_stride, src0_rep_stride, src1_rep_stride)
        with instance.if_scope(tail_times > 0):
            instance.vadd(mask,
                          dst[dst_rep_stride * block_element * Constant.MAX_REPEAT_TIMES * loop_times],
                          src0[src0_rep_stride * block_element * Constant.MAX_REPEAT_TIMES * loop_times],
                          src1[src1_rep_stride * block_element * Constant.MAX_REPEAT_TIMES * loop_times],
                          tail_times,
                          dst_blk_stride, src0_blk_stride, src1_blk_stride,
                          dst_rep_stride, src0_rep_stride, src1_rep_stride)
    with instance.else_scope():
        with instance.for_range(0, repeat_times) as row_idx:
            instance.vadd(mask,
                          dst[dst_rep_stride * block_element * row_idx],
                          src0[src0_rep_stride * block_element * row_idx],
                          src1[src1_rep_stride * block_element * row_idx],
                          1,
                          dst_blk_stride, src0_blk_stride, src1_blk_stride,
                          0, 0, 0)


def _concat_ub_vadd(instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, dst_index, src_index, count, row_count,
                    dst_row_stride, src_row_stride, mask=None, repeat_times=None, tail_count=None):
    """
    _concat_ub_vadd
    """
    if dst.scope != tik.scope_ubuf or src.scope != tik.scope_ubuf:
        error_detail = "dst and src must be UB, but dst is {} and src is {}.".format(dst.scope, src.scope)
        error_manager_vector.raise_err_specific_reson("concat_v2_d", error_detail)

    if dst.dtype != src.dtype:
        error_detail = "dst.dtype[{}] != src.dtype[{}].".format(dst.dtype, src.dtype)
        error_manager_vector.raise_err_specific_reson("concat_v2_d", error_detail)

    if not tbe_platform.api_check_support("tik.vadd", dst.dtype):
        error_detail = "{} is not supported by vadd.".format(dst.dtype)
        error_manager_vector.raise_err_specific_reson("concat_v2_d", error_detail)

    dtype_size = common_util.get_data_size(dst.dtype)
    block_element = constant.BLOCK_SIZE // dtype_size

    dst_reserve = dst_index % block_element
    new_dst_index = dst_index - dst_reserve
    new_src_index = src_index - dst_reserve
    count = count + dst_reserve
    if not mask:
        mask = _get_mask2concat_ub(instance, count, src_index, dst.dtype)
    if not repeat_times:
        repeat_times = count // (block_element * 8)
    if not tail_count:
        tail_count = count - repeat_times * block_element * 8

    with instance.new_stmt_scope():
        zero_ub = instance.Tensor(dst.dtype, (block_element,), scope=tik.scope_ubuf, name="zero_ub")
        instance.vector_dup(block_element, zero_ub, 0, 1, 0, 0)
        _vadd(instance, mask, dst[new_dst_index], src[new_src_index], zero_ub, row_count, 1, 1, 0,
              dst_row_stride, src_row_stride, 0)
        with instance.if_scope(repeat_times > 0):
            with instance.for_range(1, repeat_times) as repeat_idx:
                _vadd(instance, block_element * 8, dst[new_dst_index + block_element * 8 * repeat_idx],
                      src[new_src_index + block_element * 8 * repeat_idx], zero_ub,
                      row_count, 1, 1, 0, dst_row_stride, src_row_stride, 0)

            new_dst_index = new_dst_index + block_element * 8 * repeat_times
            new_src_index = new_src_index + block_element * 8 * repeat_times
            with instance.if_scope(tail_count > 0):
                _vadd(instance, tail_count, dst[new_dst_index], src[new_src_index], zero_ub, row_count, 1, 1, 0,
                      dst_row_stride, src_row_stride, 0)


# 'pylint: disable=invalid-name
def _data_move_all_align(tik_instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, nburst, burst, dst_stride):
    """
    _data_move_all_align
    """
    data_move_max_stride = 65535
    inst = tik_instance
    type_size = common_util.get_data_size(dst.dtype)
    block_element = constant.BLOCK_SIZE // type_size
    with inst.if_scope(dst_stride <= data_move_max_stride):
        data_move_loops = ceil_div(nburst, Constant.DATA_MOVE_MAX_REPEAT)
        with inst.for_range(0, data_move_loops) as idx:
            out_addr = Constant.DATA_MOVE_MAX_REPEAT * (burst + dst_stride) * idx * block_element
            in_addr = Constant.DATA_MOVE_MAX_REPEAT * burst * idx * block_element
            with inst.if_scope(idx * Constant.DATA_MOVE_MAX_REPEAT + Constant.DATA_MOVE_MAX_REPEAT <= nburst):
                repeat_times = Constant.DATA_MOVE_MAX_REPEAT
                inst.data_move(dst[out_addr], src[in_addr], 0, repeat_times, burst, 0, dst_stride)
            with inst.else_scope():
                repeat_times = nburst - Constant.DATA_MOVE_MAX_REPEAT * idx
                inst.data_move(dst[out_addr], src[in_addr], 0, repeat_times, burst, 0, dst_stride)
    with inst.else_scope():
        with inst.for_range(0, nburst) as row_idx:
            out_addr = (burst + dst_stride) * row_idx * block_element
            ub_addr = burst * row_idx * block_element
            ub2gm(inst, dst[out_addr:], src[ub_addr:], burst * block_element, burst)


# 'pylint:disable=too-many-instance-attributes,too-few-public-methods
class ConcatV2:
    """
    ConcatV2
    """

    class TilingParam:
        """
        TilingParam
        """

        def __init__(self, input_values, inst: tik.Tik):
            self.tik_instance = inst
            dtype = "int64"

            # data in tiling_gm likes:
            # 0---- 1----    2----          3----
            # axis, out_dim, max_inner_dim, min_inner_dim,
            # 4----                5----
            # output_inner_length, input_count
            # 6----    7----
            # reserve, reserve
            # 8----             9----
            # first_inner_dims, first_output_idx,
            # second_inner_dims, second_output_idx
            # ...
            self.dtype = dtype
            self.input_values = input_values
            self.axis = inst.Scalar(dtype, name="axis")
            self.out_dim = inst.Scalar(dtype, name="out_dim")
            self.max_inner_dim = inst.Scalar(dtype, name="max_inner_dim")
            self.min_inner_dim = inst.Scalar(dtype, name="min_inner_dim")
            self.output_inner_length = inst.Scalar(dtype,
                                                   name="output_inner_length")

            tiling_ub_size = len(input_values) * 2 + 8
            tiling_ub_size = ceil_32bytes_align_count(tiling_ub_size, dtype)
            max_input_count = 64
            tiling_gm_size = max_input_count * 2 + 8
            self.tiling_ub_size = tiling_ub_size
            self.tiling_gm = inst.Tensor(dtype, (tiling_gm_size,),
                                         name="tiling_gm",
                                         scope=tik.scope_gm)

            self._need_ub_size = (self.tiling_ub_size *
                                  common_util.get_data_size(dtype))
            self.data_dtype = self.input_values[0].get("dtype")
            self.block_element = constant.BLOCK_SIZE // common_util.get_data_size(self.data_dtype)
            self._tiling_ub = None
            self._dims = []
            self.all_align = inst.Scalar(dtype, name="all_align", init_value=0)
            self.only_last_input_not_align = inst.Scalar(dtype, name="only_last_input_not_align", init_value=0)
            self.mask_cycle_lines = self.block_element
            self.mask_cycle_output_inner_burst = self.output_inner_length
            self._mask_cycle_inner_burst = []

        def init(self):
            """
            :return:
            """
            inst = self.tik_instance
            dtype = self.dtype
            head_count = 8
            for i, _ in enumerate(self.input_values):
                self._dims.append(inst.Scalar(dtype=dtype, name="inner_dim" + str(i)))
                self._dims.append(inst.Scalar(dtype=dtype, name="output_index" + str(i)))
            with inst.new_stmt_scope():
                self._tiling_ub = inst.Tensor(dtype, (self.tiling_ub_size,),
                                              name="tiling_ub",
                                              scope=tik.scope_ubuf)
                gm2ub(inst, self._tiling_ub, self.tiling_gm, self.tiling_ub_size)
                self.axis.set_as(self._tiling_ub[0])
                self.out_dim.set_as(self._tiling_ub[1])
                self.max_inner_dim.set_as(self._tiling_ub[2])
                self.min_inner_dim.set_as(self._tiling_ub[3])
                self.output_inner_length.set_as(self._tiling_ub[4])

                self.all_align.set_as(1)
                for i, _ in enumerate(self.input_values):
                    index = head_count + i * 2
                    self._dims[i * 2].set_as(self._tiling_ub[index])
                    self._dims[i * 2 + 1].set_as(self._tiling_ub[index + 1])
                    if i == len(self.input_values) - 1:
                        self.only_last_input_not_align.set_as(self.all_align)
                        with inst.if_scope(self._dims[i * 2] % self.block_element == 0):
                            self.only_last_input_not_align.set_as(0)
                    self.all_align.set_as(self.all_align + self._dims[i * 2] % self.block_element)

        def init_mask_cycle_info(self):
            """
            init_mask_cycle_info
            """
            inst = self.tik_instance
            tmp_all_align = inst.Scalar(dtype="int8", name="tmp_all_align", init_value=0)
            block_element = self.block_element
            self.mask_cycle_lines = inst.Scalar(dtype=self.dtype, name="mask_cycle_lines")
            self.mask_cycle_output_inner_burst = inst.Scalar(dtype=self.dtype, name="mask_cycle_output_inner_burst")
            for index, _ in enumerate(self.input_values):
                inner_dim, _ = self.get_dims(index)
                burst = inst.Scalar(dtype=self.dtype, name="mask_cycle_inner_burst" + str(index))
                self._mask_cycle_inner_burst.append(burst)

            for _, line in enumerate(range(1, block_element + 1)):
                with inst.if_scope(tmp_all_align != len(self.input_values)):
                    tmp_all_align.set_as(0)
                    for index, _ in enumerate(self.input_values):
                        inner_dim, _ = self.get_dims(index)
                        with inst.if_scope(inner_dim * line % self.block_element == 0):
                            tmp_all_align.set_as(tmp_all_align + 1)
                    with inst.if_scope(tmp_all_align == len(self.input_values)):
                        self.mask_cycle_lines.set_as(line)
                        with inst.if_scope(self.mask_cycle_lines % 2 != 0):
                            self.mask_cycle_lines.set_as(self.mask_cycle_lines * 2)
                        lines = self.mask_cycle_lines
                        self.mask_cycle_output_inner_burst.set_as(self.output_inner_length * lines // block_element)

                        # max repeat stride is 255
                        with inst.if_scope(self.mask_cycle_output_inner_burst > 255):
                            tmp_all_align.set_as(0)
                        for index, _ in enumerate(self.input_values):
                            inner_dim, _ = self.get_dims(index)
                            self._mask_cycle_inner_burst[index].set_as(inner_dim * lines // block_element)

        def get_inst_dims(self):
            """
            for public call
            """
            return self._dims

        def get_dims(self, input_index):
            """
            :param input_index: index of input tensors
            :return: inner dims, output_index of each row
            """
            index = input_index * 2
            return self._dims[index], self._dims[index + 1]

        def get_mask_cycle_burst(self, input_index):
            """
            get mask cycle burst of input
            """
            return self._mask_cycle_inner_burst[input_index]

        def set_tiling_ub(self, tiling_ub):
            """
            for public call
            """
            self._tiling_ub = tiling_ub

        def update_tiling(self, src_dtype, dst_dtype):
            """
            update inner dims information multiply by multi_times
            :param src_dtype: src dtype
            :param dst_dtype: dst dtype
            :return: None
            """
            src_type_size = common_util.get_data_size(src_dtype)
            dst_type_size = common_util.get_data_size(dst_dtype)
            self.max_inner_dim.set_as(self.max_inner_dim * src_type_size // dst_type_size)
            self.min_inner_dim.set_as(self.min_inner_dim * src_type_size // dst_type_size)
            self.output_inner_length.set_as(self.output_inner_length * src_type_size // dst_type_size)
            for _, dim_i in enumerate(self._dims):
                dim_i.set_as(dim_i * src_type_size // dst_type_size)

        # 'pylint: disable=no-self-use
        @staticmethod
        def need_ub_size():
            """
            :return: ub size needed by tiling
            """
            return 0

    def __init__(self, input_values, axis, kernel_name):
        self.tik_instance = tik.Tik()
        self.tik_profiling = tik.Dprofile()
        self.tiling_param = self.TilingParam(input_values, self.tik_instance)
        self.aicore_num = self.tik_profiling.get_aicore_num()
        self.kernel_name = kernel_name
        self.axis = axis

        self.dtype = input_values[0].get("dtype").lower()
        self.output_shape = (Constant.MAX_SIZE,)
        self.input_shape = (Constant.MAX_SIZE,)

        self.input_tensors, self.output_tensor = self._init_gm_tensor(self.input_shape, self.output_shape,
                                                                      len(input_values),
                                                                      self.dtype)

        dtype_bytes_size = common_util.get_data_size(self.dtype)
        self.type_size = dtype_bytes_size
        self.ele_each_block = constant.BLOCK_SIZE // dtype_bytes_size
        valid_ub_size = self.tik_profiling.get_unified_buffer_size()
        valid_ub_size -= self.tiling_param.need_ub_size()
        self.ub_buffer_length = valid_ub_size

        # reserve two block size for not 32 bytes align
        self.ub_buffer_length -= constant.BLOCK_SIZE * 2

        # make ub_buffer_length 32 bytes align
        self.ub_buffer_length //= constant.BLOCK_SIZE
        self.ub_buffer_length *= constant.BLOCK_SIZE

        self.ub_buffer_length //= dtype_bytes_size

    def _init_gm_tensor(self, input_shape, output_shape, input_count, dtype):
        """
        init gm tensor

        Parameters
        ----------
        input_shape: list
            shape of input tensor
        output_shape: list
            shape of output tensor
        dtype: str
            data type

        Returns
        -------
        input_tensors: tik tensor
            input gm tensor
        output_tensor: tik tensor
            output gm tensor
        """
        input_tensors = []
        for _, index in enumerate(range(input_count)):
            tensor_name = "_".join(["gm_input", str(index)])
            gm_tensor = self.tik_instance.Tensor(dtype, input_shape, name=tensor_name, scope=tik.scope_gm)
            input_tensors.append(gm_tensor)

        output_tensor = self.tik_instance.Tensor(dtype, output_shape, name="gm_output", scope=tik.scope_gm)

        return input_tensors, output_tensor

    def concat_compute(self):
        """
        build concat op

        Returns
        -------
        None
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        with inst.for_range(0, aicore_num, name="core_idx", block_num=aicore_num) as i:
            self.tiling_param.init()
            with inst.if_scope(self.tiling_param.out_dim == 1):
                self._concat_first_dim(i)
            with inst.else_scope():
                with inst.if_scope(self._is_all_align_do_multi_output_lines()):
                    self._concat_all_align_with_multi_output_lines(i)
                with inst.else_scope():
                    output_inner_length = self.tiling_param.output_inner_length
                    with inst.if_scope(tik.any(output_inner_length * self.ele_each_block < self.ub_buffer_length // 4,
                                               self.tiling_param.min_inner_dim < self.ele_each_block)):
                        self._concat_small_inner(i)
                    with inst.else_scope():
                        self._concat_large_inner(i)

    def build(self):
        """
        build

        Returns
        -------
        tik_instance
        """
        tbe_context.get_context().add_compile_info("is_tik", True)
        tbe_context.get_context().add_compile_info("vars", {
            "input_size": len(self.input_tensors),
            "concat_dim": self.axis,
            "block_dim": self.aicore_num
        })

        inst = self.tik_instance
        opt_config = {"out_of_bound_sync_check": True,
                      "enable_const_fold": True}
        inst.BuildCCE(kernel_name=self.kernel_name, inputs=self.input_tensors, outputs=(self.output_tensor,),
                      flowtable=[self.tiling_param.tiling_gm],
                      config=opt_config,
                      enable_l2=False)

        return inst

    def _get_ceil_32bytes_count(self, count: tik.Scalar):
        """
        get ceil of 32 bytes
        """
        ceil_num = ceil_div(count, self.ele_each_block)
        return ceil_num * self.ele_each_block

    # 'pylint: disable=invalid-name,unused-variable,too-many-statements
    def _concat_inner_dim_each_split(self, out_dim_idx, inner_dim_split_idx):
        """
        concat inner dim each split
        """
        for index, _ in enumerate(self.input_tensors):
            self._concat_compute_tensor_inner_dim(out_dim_idx, inner_dim_split_idx, index)

    def _copy_one_row(self, row_idx, tensor_index):
        """
        copy_one_row
        """
        inst = self.tik_instance
        factor = self.ub_buffer_length // 2
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        input_gm = self.input_tensors[tensor_index]
        output_gm = self.output_tensor
        with inst.new_stmt_scope():
            ub_length = factor
            ub = inst.Tensor(input_gm.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            loops = ceil_div(inner_dims, ub_length)
            with inst.for_range(0, loops) as inner_dim_split_idx:
                in_start_index = inner_dim_split_idx * factor + inner_dims * row_idx
                output_dim = self.tiling_param.output_inner_length
                out_start_index = output_idx + inner_dim_split_idx * factor + output_dim * row_idx
                with inst.if_scope(in_start_index < inner_dims * (1 + row_idx)):
                    count = inst.Scalar("int64", name="count")
                    count.set_as(inner_dims * (1 + row_idx) - in_start_index)
                    with inst.if_scope(count > ub_length):
                        count.set_as(ub_length)

                    gm2ub(inst, ub, input_gm[in_start_index:], count)
                    ub2gm(inst, output_gm[out_start_index:], ub, count)

    def _copy_one_block(self, row_idx, tensor_index):
        """
        copy_one_block
        """
        inst = self.tik_instance
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        input_gm = self.input_tensors[tensor_index]
        output_gm = self.output_tensor
        with inst.new_stmt_scope():
            ub_length = self.ele_each_block
            ub = inst.Tensor(input_gm.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            in_start_index = inner_dims * row_idx
            output_dim = self.tiling_param.output_inner_length
            out_start_index = output_idx + output_dim * row_idx
            with inst.if_scope(in_start_index < inner_dims * (1 + row_idx)):
                gm2ub(inst, ub, input_gm[in_start_index:], self.ele_each_block)
                ub2gm(inst, output_gm[out_start_index:], ub, self.ele_each_block)

    def _concat_compute_tensor_inner_dim(self, out_dim_idx, inner_dim_split_idx, tensor_index):
        """
        concat_compute_tensor_inner_dim
        """
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        with self.tik_instance.if_scope(inner_dims > 0):
            with self.tik_instance.if_scope(inner_dims % self.ele_each_block == 0):
                self._concat_tensor_align_inner_dim(out_dim_idx, inner_dim_split_idx, tensor_index)
            with self.tik_instance.else_scope():
                self._concat_tensor_not_align_inner_dim(out_dim_idx, inner_dim_split_idx, tensor_index)

    def _concat_tensor_align_inner_dim(self, out_dim_idx, inner_dim_split_idx, tensor_index):
        """
        _concat_tensor_align_inner_dim
        """
        inst = self.tik_instance
        factor = self.ub_buffer_length
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        input_gm = self.input_tensors[tensor_index]
        output_gm = self.output_tensor
        with inst.new_stmt_scope():
            ub_length = self.ub_buffer_length
            ub = inst.Tensor(input_gm.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            in_start_index = inner_dim_split_idx * factor + inner_dims * out_dim_idx

            output_dim = self.tiling_param.output_inner_length
            out_start_index = output_idx + inner_dim_split_idx * factor + output_dim * out_dim_idx
            with inst.if_scope(in_start_index < inner_dims * (1 + out_dim_idx)):
                count = inst.Scalar("int64", name="count")
                count.set_as(inner_dims * (1 + out_dim_idx) - in_start_index)
                with inst.if_scope(count > ub_length):
                    count.set_as(ub_length)

                gm2ub(inst, ub, input_gm[in_start_index:], count)
                ub2gm(inst, output_gm[out_start_index:], ub, count)

    def _concat_tensor_not_align_inner_dim(self, out_dim_idx, inner_dim_split_idx, tensor_index):
        """
        _concat_tensor_not_align_inner_dim
        """
        inst = self.tik_instance
        factor = self.ub_buffer_length
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        input_gm = self.input_tensors[tensor_index]
        output_gm = self.output_tensor

        with inst.new_stmt_scope():
            ub_length = self.ub_buffer_length
            ub = inst.Tensor(input_gm.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            in_start_index = inner_dim_split_idx * factor + inner_dims * out_dim_idx

            output_dim = self.tiling_param.output_inner_length
            out_start_index = output_idx + inner_dim_split_idx * factor + output_dim * out_dim_idx
            with inst.if_scope(in_start_index < inner_dims * (1 + out_dim_idx)):
                count = inner_dims * (1 + out_dim_idx) - in_start_index
                with inst.if_scope(count > ub_length):
                    gm2ub(inst, ub, input_gm[in_start_index:], ub_length)
                    ub2gm(inst, output_gm[out_start_index:], ub, ub_length)
                with inst.else_scope():
                    with inst.if_scope(inner_dim_split_idx > 0):
                        align_count = self._get_ceil_32bytes_count(count)
                        redundant_count = align_count - count
                        new_in_start_index = in_start_index - redundant_count
                        new_out_start_index = out_start_index - redundant_count
                        gm2ub(inst, ub, input_gm[new_in_start_index:], align_count)
                        ub2gm(inst, output_gm[new_out_start_index:], ub, align_count)
                    with inst.else_scope():
                        gm2ub(inst, ub, input_gm[in_start_index:], self.ele_each_block)
                        ub2gm(inst, output_gm[out_start_index:], ub, self.ele_each_block)

                        in_start_index += self.ele_each_block
                        out_start_index += self.ele_each_block
                        align_count = self._get_ceil_32bytes_count(count - self.ele_each_block)
                        redundant_count = align_count - count + self.ele_each_block
                        new_in_start_index = in_start_index - redundant_count
                        new_out_start_index = out_start_index - redundant_count
                        with inst.if_scope(align_count > 0):
                            gm2ub(inst, ub, input_gm[new_in_start_index:], align_count)
                            ub2gm(inst, output_gm[new_out_start_index:], ub, align_count)

    def _is_all_align_do_multi_output_lines(self):
        """
        _is_all_align_do_multi_output_lines
        """
        ub_len = self.ub_buffer_length // 4
        output_inner_dims = self.tiling_param.output_inner_length
        return tik.all(self.tiling_param.all_align == 1,
                       ub_len // output_inner_dims > 1,
                       self.tiling_param.min_inner_dim > self.ele_each_block)

    def _concat_large_inner(self, core_idx):
        """
        tiling with out_dims and split of inner_dims
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        out_dims = self.tiling_param.out_dim
        max_inner_dim = self.tiling_param.max_inner_dim
        inner_dims_loops = ceil_div(max_inner_dim, self.ub_buffer_length)
        max_loops = out_dims * inner_dims_loops
        out_loops = ceil_div(max_loops, aicore_num)
        with inst.for_range(0, out_loops, name="out_loops_idx") as i:
            loop_idx = i + out_loops * core_idx
            with inst.if_scope(loop_idx < max_loops):
                out_dim_idx = loop_idx / inner_dims_loops
                inner_dim_split_idx = loop_idx % inner_dims_loops
                self._concat_inner_dim_each_split(out_dim_idx, inner_dim_split_idx)

    def _concat_only_last_input_not_align(self, core_idx):
        """
        _concat_only_last_input_not_align
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        out_dims = self.tiling_param.out_dim
        row_each_core = ceil_div(out_dims, aicore_num)
        do_lines = inst.Scalar(dtype="int64", name="do_lines", init_value=row_each_core)
        with inst.if_scope(row_each_core * core_idx < out_dims):
            with inst.if_scope(row_each_core * core_idx + do_lines > out_dims):
                do_lines.set_as(out_dims - row_each_core * core_idx)
            for index, _ in enumerate(self.input_tensors):
                tensor_index = len(self.input_tensors) - 1 - index
                inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
                with inst.if_scope(inner_dims > 0):
                    with inst.for_range(0, do_lines // 2) as idx:
                        out_dim_idx = row_each_core * core_idx + idx * 2
                        self._copy_one_row(out_dim_idx, tensor_index)
                        out_dim_idx = row_each_core * core_idx + idx * 2 + 1
                        self._copy_one_row(out_dim_idx, tensor_index)
                    with inst.if_scope(do_lines % 2 != 0):
                        out_dim_idx = row_each_core * core_idx + do_lines - 1
                        self._copy_one_row(out_dim_idx, tensor_index)
            with inst.if_scope(row_each_core * core_idx + row_each_core < out_dims):
                corrected = inst.Scalar(dtype="int8", name="corrected")
                corrected.set_as(0)
                for index, _ in enumerate(self.input_tensors):
                    inner_dims, output_idx = self.tiling_param.get_dims(index)
                    with inst.if_scope(tik.all(inner_dims > 0, corrected == 0)):
                        self._copy_one_block(row_each_core * core_idx + row_each_core, index)
                        corrected.set_as(1)

    def _concat_all_align_with_multi_output_lines(self, core_idx):
        """
        _concat_all_align_with_multi_output_lines
        """
        inst = self.tik_instance
        ub_len = self.ub_buffer_length // 6 // self.ele_each_block * self.ele_each_block
        out_dims = self.tiling_param.out_dim
        output_inner_dims = self.tiling_param.output_inner_length
        ub_can_copy_lines = ub_len // output_inner_dims
        row_each_copy = inst.Scalar(dtype="int64", name="row_each_copy",
                                    init_value=ceil_div(out_dims, self.aicore_num))
        with inst.if_scope(row_each_copy > ub_can_copy_lines):
            row_each_copy.set_as(ub_can_copy_lines)
        loops_each_core = ceil_div(ceil_div(ceil_div(out_dims, self.aicore_num), row_each_copy), 2) * 2
        row_each_copy.set_as(ceil_div(ceil_div(out_dims, self.aicore_num), loops_each_core))
        row_each_core = loops_each_core * row_each_copy
        output_tensor = self.output_tensor
        input_tensors = self.input_tensors
        output_start_addr = row_each_core * core_idx * output_inner_dims
        with inst.new_stmt_scope():
            do_lines = inst.Scalar(dtype="int64", name="do_lines", init_value=row_each_copy)
            with inst.for_range(0, loops_each_core // 2) as loop_idx:
                in_ub1 = inst.Tensor(dtype=self.dtype, shape=(ub_len,), scope=tik.scope_ubuf, name="in_ub1")
                in_ub2 = inst.Tensor(dtype=self.dtype, shape=(ub_len,), scope=tik.scope_ubuf, name="in_ub2")
                in_ub3 = inst.Tensor(dtype=self.dtype, shape=(ub_len,), scope=tik.scope_ubuf, name="in_ub3")
                in_ub4 = inst.Tensor(dtype=self.dtype, shape=(ub_len,), scope=tik.scope_ubuf, name="in_ub4")
                out_ub1 = inst.Tensor(dtype=self.dtype, shape=(ub_len,), scope=tik.scope_ubuf, name="out_ub1")
                out_ub2 = inst.Tensor(dtype=self.dtype, shape=(ub_len,), scope=tik.scope_ubuf, name="out_ub2")
                in_ub_list1 = [in_ub1, in_ub2]
                in_ub_list2 = [in_ub3, in_ub4]

                def ping_pong_func(loop_idx, in_ub_list, out_ub):
                    with inst.if_scope(row_each_core * core_idx + loop_idx * row_each_copy < out_dims):
                        output_addr = output_start_addr + (loop_idx * row_each_copy * output_inner_dims)
                        with inst.if_scope(row_each_core * core_idx +
                                           loop_idx * row_each_copy + row_each_copy > out_dims):
                            do_lines.set_as(out_dims - row_each_core * core_idx - loop_idx * row_each_copy)
                        for index, input_tensor in enumerate(input_tensors):
                            in_ub = in_ub_list[index % len(in_ub_list)]
                            inner_dims, output_idx = self.tiling_param.get_dims(index)
                            with inst.if_scope(inner_dims > 0):
                                input_start_addr = row_each_core * core_idx * inner_dims
                                input_addr = input_start_addr + loop_idx * row_each_copy * inner_dims
                                gm2ub(inst, in_ub, input_tensor[input_addr:], do_lines * inner_dims)
                                inst.data_move(out_ub[output_idx:], in_ub, 0, do_lines,
                                               inner_dims // self.ele_each_block,
                                               0, (output_inner_dims - inner_dims) // self.ele_each_block)
                        ub2gm(inst, output_tensor[output_addr:], out_ub, do_lines * output_inner_dims)

                ping_pong_func(loop_idx * 2, in_ub_list1, out_ub1)
                ping_pong_func(loop_idx * 2 + 1, in_ub_list2, out_ub2)

    def _concat_small_inner(self, core_idx):
        """
        tiling with out_dims
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        out_dims = self.tiling_param.out_dim
        ub_len = self.ub_buffer_length // 4
        ub_len = ub_len // self.ele_each_block * self.ele_each_block
        output_inner_dim = self.tiling_param.output_inner_length
        ub_can_storage_lines_vnchwconv = ub_len // self.ele_each_block
        ub_can_copy_lines = inst.Scalar(
            dtype="int64", name="ub_can_copy_lines",
            init_value=self.ub_buffer_length // (output_inner_dim * 4) // self.ele_each_block * self.ele_each_block)
        if self.type_size % 2 == 0:
            min_inner_dim = self.tiling_param.min_inner_dim
            max_inner_dim = self.tiling_param.max_inner_dim
            with inst.if_scope(tik.all(self.type_size % 2 == 0,
                                       max_inner_dim == min_inner_dim,
                                       ub_can_storage_lines_vnchwconv >= output_inner_dim,
                                       16 % min_inner_dim == 0,
                                       out_dims * min_inner_dim * self.type_size >= Constant.VNCHW_BLOCK_SIZE)):
                self._concat_with_vnchwconv(core_idx, out_dims, ub_len)
            with inst.else_scope():
                with inst.if_scope(tik.all(output_inner_dim >= self.ele_each_block,
                                           ub_can_copy_lines >= self.ele_each_block)):
                    lines_each_core = ceil_div(ceil_div(out_dims, self.aicore_num),
                                               self.ele_each_block) * self.ele_each_block
                    with inst.if_scope(lines_each_core < ub_can_copy_lines):
                        ub_can_copy_lines.set_as(lines_each_core)
                    self._concat_small_inner_each_core_multi_line(core_idx, out_dims, ub_can_copy_lines)
                with inst.else_scope():
                    with inst.if_scope(self.tiling_param.only_last_input_not_align == 1):
                        self._concat_only_last_input_not_align(core_idx)
                    with inst.else_scope():
                        count_each_core = ceil_div(out_dims, aicore_num)
                        self._concat_small_inner_each_core_one_line(core_idx, out_dims, count_each_core)
        else:
            with inst.if_scope(tik.all(output_inner_dim >= self.ele_each_block,
                                       ub_can_copy_lines >= self.ele_each_block)):
                lines_each_core = ceil_div(ceil_div(out_dims, self.aicore_num),
                                           self.ele_each_block) * self.ele_each_block
                with inst.if_scope(lines_each_core < ub_can_copy_lines):
                    ub_can_copy_lines.set_as(lines_each_core)
                self._concat_small_inner_each_core_multi_line(core_idx, out_dims, ub_can_copy_lines)
            with inst.else_scope():
                with inst.if_scope(self.tiling_param.only_last_input_not_align == 1):
                    self._concat_only_last_input_not_align(core_idx)
                with inst.else_scope():
                    count_each_core = ceil_div(out_dims, aicore_num)
                    self._concat_small_inner_each_core_one_line(core_idx, out_dims, count_each_core)

    def _concat_small_inner_each_core_one_line(self, core_idx, out_dims, count_each_core):
        """
        _concat_small_inner_each_core_one_line
        """
        inst = self.tik_instance
        with inst.for_range(0, count_each_core, name="inner_loop") as j:
            row_idx = j + count_each_core * core_idx
            with inst.if_scope(row_idx < out_dims):
                with inst.if_scope(j != count_each_core - 1):
                    self._concat_small_inner_each_core_not_last_row(row_idx)
                with inst.else_scope():
                    self._concat_small_inner_each_core_last_row(row_idx)

    def _concat_small_inner_each_core_not_last_row(self, row_idx):
        self._concat_small_inner_each_core_without_treat_overlap(row_idx, self.input_tensors)

    def _concat_small_inner_each_core_last_row(self, row_idx):
        self._concat_small_inner_each_core_without_treat_overlap(row_idx,
                                                                 self.input_tensors[0:len(self.input_tensors) - 1])
        self._concat_small_inner_each_core_last_row_last_tensor(row_idx)

    def _concat_small_inner_each_core_without_treat_overlap(self, row_idx, tensors):
        """
        _concat_small_inner_each_core_without_treat_overlap
        """
        inst = self.tik_instance
        output_tensor = self.output_tensor
        output_inner_len = self.tiling_param.output_inner_length
        ub_length = self.ub_buffer_length
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            ub_data_count = inst.Scalar("int32", name="ub_data_count")
            ub_data_count.set_as(0)
            tmp_ub = inst.Tensor(self.dtype, (self.ele_each_block,), scope=tik.scope_ubuf, name="tmp_ub")

            out_row_start_idx = output_inner_len * row_idx
            out_start_idx = inst.Scalar("int64", name="ub_data_count")
            out_start_idx.set_as(out_row_start_idx)
            for index, input_tensor in enumerate(tensors):
                inner_dim, output_idx = self.tiling_param.get_dims(index)
                with inst.if_scope(inner_dim > 0):
                    in_start_idx = inner_dim * row_idx
                    with inst.if_scope(ub_data_count >= self.ele_each_block):
                        ub2gm(inst, output_tensor[out_start_idx:], out_ub, ub_data_count)
                        ub_data_count.set_as(0)

                    with inst.if_scope(ub_data_count == 0):
                        out_start_idx.set_as(out_row_start_idx + output_idx)

                    with inst.if_scope(inner_dim < self.ele_each_block):
                        gm2ub(inst, tmp_ub, input_tensor[in_start_idx:], inner_dim)
                        with inst.for_range(0, inner_dim) as scalar_idx:
                            out_ub[ub_data_count] = tmp_ub[scalar_idx]
                            ub_data_count.set_as(ub_data_count + 1)

                    with inst.else_scope():
                        with inst.if_scope(ub_data_count > 0):
                            ub2gm(inst, output_tensor[out_start_idx:], out_ub, ub_data_count)
                            ub_data_count.set_as(0)
                            out_start_idx.set_as(out_row_start_idx + output_idx)

                        loops = ceil_div(inner_dim, ub_length)
                        with inst.for_range(0, loops, name="inner_loop") as idx:
                            in_start_idx = ub_length * idx + inner_dim * row_idx
                            out_start_idx.set_as(ub_length * idx + out_row_start_idx + output_idx)
                            count = inst.Scalar("int64", name="count")
                            count.set_as(inner_dim * (1 + row_idx) - in_start_idx)
                            with inst.if_scope(count > ub_length):
                                count.set_as(ub_length)

                            gm2ub(inst, out_ub, input_tensor[in_start_idx:], count)
                            ub2gm(inst, output_tensor[out_start_idx:], out_ub, count)

            with inst.if_scope(ub_data_count > 0):
                ub2gm(inst, output_tensor[out_start_idx:], out_ub, ub_data_count)

    def _concat_small_inner_each_core_last_row_last_tensor(self, row_idx):
        """
        _concat_small_inner_each_core_last_row_last_tensor
        """
        inst = self.tik_instance
        ub_length = self.ub_buffer_length
        output_inner_len = self.tiling_param.output_inner_length
        out_dims = self.tiling_param.out_dim
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            output_tensor = self.output_tensor
            last_idx = len(self.input_tensors) - 1
            input_tensor = self.input_tensors[last_idx]
            inner_dim, output_idx = self.tiling_param.get_dims(last_idx)
            with inst.if_scope(inner_dim > 0):
                out_start_idx = inst.Scalar("int64", name="ub_data_count")
                ub_data_count = inst.Scalar("int32", name="ub_data_count")
                tmp_ub = inst.Tensor(self.dtype, (self.ele_each_block,), scope=tik.scope_ubuf, name="tmp_ub")
                out_start_idx.set_as(row_idx * output_inner_len + output_idx)
                with inst.if_scope(inner_dim < self.ele_each_block):
                    gm2ub(inst, out_ub, input_tensor[inner_dim * row_idx], inner_dim)
                    ub_data_count.set_as(inner_dim)
                    pad_count = inst.Scalar("int32", name="pad_count")
                    pad_count.set_as(self.ele_each_block - inner_dim)
                    loops = ceil_div(pad_count, output_inner_len)
                    with inst.for_range(0, loops) as loop:
                        new_out_dim_idx = row_idx + loop
                        with inst.if_scope(new_out_dim_idx < out_dims):
                            for idx, tmp_tensor in enumerate(self.input_tensors):
                                temp_inner_dims, _ = self.tiling_param.get_dims(idx)
                                with inst.if_scope(temp_inner_dims > 0):
                                    with inst.if_scope(ub_data_count < self.ele_each_block):
                                        gm2ub(inst, tmp_ub, tmp_tensor[(row_idx + loop + 1) * temp_inner_dims],
                                              self.ele_each_block)
                                        with inst.for_range(0, temp_inner_dims) as scalar_idx:
                                            with inst.if_scope(ub_data_count < self.ele_each_block):
                                                out_ub[ub_data_count] = tmp_ub[scalar_idx]
                                                ub_data_count.set_as(ub_data_count + 1)

                    ub2gm(inst, output_tensor[out_start_idx:], out_ub, inner_dim)
                with inst.else_scope():
                    loops = ceil_div(inner_dim, ub_length)
                    with inst.for_range(0, loops, name="inner_loop") as idx:
                        in_start_idx = (ub_length * idx + inner_dim * row_idx)
                        out_start_idx.set_as(ub_length * idx + output_inner_len * row_idx + output_idx)
                        count = inner_dim * (row_idx + 1) - in_start_idx
                        with inst.if_scope(count > ub_length):
                            gm2ub(inst, out_ub, input_tensor[in_start_idx:], ub_length)
                            ub2gm(inst, output_tensor[out_start_idx:], out_ub, ub_length)
                        with inst.else_scope():
                            with inst.if_scope(idx > 0):
                                align_count = self._get_ceil_32bytes_count(count)
                                redundant_cnt = (align_count - count)
                                new_in_start_index = in_start_idx - redundant_cnt
                                new_out_start_index = out_start_idx - redundant_cnt
                                gm2ub(inst, out_ub, input_tensor[new_in_start_index:], count)
                                ub2gm(inst, output_tensor[new_out_start_index:], out_ub, count)
                            with inst.else_scope():
                                gm2ub(inst, out_ub, input_tensor[in_start_idx:], self.ele_each_block)
                                ub2gm(inst, output_tensor[out_start_idx:], out_ub, self.ele_each_block)
                                in_start_idx += self.ele_each_block
                                out_start_idx += self.ele_each_block
                                align_count = self._get_ceil_32bytes_count(count - self.ele_each_block)
                                redundant_cnt = align_count - count + self.ele_each_block
                                new_in_start_index = in_start_idx - redundant_cnt
                                new_out_start_index = out_start_idx - redundant_cnt
                                with inst.if_scope(align_count > 0):
                                    gm2ub(inst, out_ub, input_tensor[new_in_start_index:], align_count)
                                    ub2gm(inst, output_tensor[new_out_start_index:], out_ub, align_count)

    def _concat_small_inner_each_core_multi_line(self, core_idx, out_dims, ub_can_copy_lines):
        """
        _concat_small_inner_each_core_multi_line
        """
        inst = self.tik_instance
        if tbe_platform.api_check_support("tik.vadd", self.dtype):
            with inst.if_scope(self.tiling_param.max_inner_dim >= self.ele_each_block):
                self._concat_small_inner_each_core_multi_line_by_vadd(core_idx, out_dims, ub_can_copy_lines)
            with inst.else_scope():
                self._concat_small_inner_each_core_multi_line_by_scalar(core_idx, out_dims)
        else:
            self._concat_small_inner_each_core_multi_line_by_scalar(core_idx, out_dims)

    def _concat_small_inner_each_core_multi_line_by_vadd(self, core_idx, out_dims, ub_can_copy_lines):
        """
        _concat_small_inner_each_core_multi_line_by_vadd
        """
        inst = self.tik_instance
        ub_copy_times = ceil_div(out_dims, ub_can_copy_lines)
        ub_copy_times_each_core = ceil_div(ub_copy_times, self.aicore_num)
        to_do_count = inst.Scalar(dtype="int64", name="to_do_count")
        self.tiling_param.init_mask_cycle_info()
        with inst.for_range(0, ub_copy_times_each_core, name="inner_loop", thread_num=1) as j:
            row_idx = j * ub_can_copy_lines + ub_can_copy_lines * ub_copy_times_each_core * core_idx
            with inst.if_scope(row_idx < out_dims):
                with inst.if_scope(row_idx + ub_can_copy_lines <= out_dims):
                    to_do_count.set_as(ub_can_copy_lines)
                with inst.else_scope():
                    to_do_count.set_as(out_dims - row_idx)
                self._concat_small_inner_each_core_multi_line_by_vadd_each_loop(row_idx, to_do_count)

    def _concat_small_inner_each_core_multi_line_by_scalar(self, core_idx, out_dims):
        """
        _concat_small_inner_each_core_multi_line_by_scalar
        """
        inst = self.tik_instance
        ub_len = self.ub_buffer_length // 2
        block_element = self.ele_each_block
        output_inner_dim = self.tiling_param.output_inner_length
        aicore_num = self.aicore_num
        lines_ub_copy = inst.Scalar(dtype="int64", name="lines_ub_copy")
        lines_ub_copy.set_as(ub_len // output_inner_dim // block_element * block_element)
        lines_each_core = ceil_div(ceil_div(out_dims, aicore_num), block_element) * block_element
        with inst.if_scope(lines_ub_copy > lines_each_core):
            lines_ub_copy.set_as(lines_each_core)
        ub_copy_times = ceil_div(out_dims, lines_ub_copy)
        ub_copy_times_each_core = ceil_div(ub_copy_times, self.aicore_num)
        to_do_count = inst.Scalar(dtype="int64", name="to_do_count")
        with inst.for_range(0, ub_copy_times_each_core, name="inner_loop", thread_num=1) as j:
            row_idx = j * lines_ub_copy + lines_ub_copy * ub_copy_times_each_core * core_idx
            with inst.if_scope(row_idx < out_dims):
                with inst.if_scope(row_idx + lines_ub_copy <= out_dims):
                    to_do_count.set_as(lines_ub_copy)
                with inst.else_scope():
                    to_do_count.set_as(out_dims - row_idx)
                self._concat_small_inner_each_core_multi_line_by_scalar_each_loop(row_idx, to_do_count, ub_len)

    def _concat_small_inner_each_core_multi_line_by_vadd_each_loop(self, row_idx, lines):
        """
        _concat_small_inner_each_core_multi_line_by_vadd_each_loop
        """
        inst = self.tik_instance
        tensors = self.input_tensors
        output_tensor = self.output_tensor
        output_inner_len = self.tiling_param.output_inner_length
        ub_length = self.ub_buffer_length // 3 // self.ele_each_block * self.ele_each_block
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            tmp_ub0 = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="tmp_ub0")
            tmp_ub1 = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="tmp_ub1")
            row_idx = inst.Scalar("int64", name="row_idx", init_value=row_idx)
            lines = inst.Scalar("int64", name="lines", init_value=lines)

            with inst.if_scope(tik.all(row_idx == 0, lines >= self.ele_each_block)):
                # if core 0 first handle self.ele_each_block rows
                self._concat_small_inner_each_core_multi_lines_first_block_element_rows(out_ub, tmp_ub0, lines)
                row_idx.set_as(row_idx + self.ele_each_block)
                lines.set_as(lines - self.ele_each_block)
            with inst.if_scope(lines > 0):
                out_row_start_idx = output_inner_len * row_idx
                out_start_idx = inst.Scalar("int64", name="ub_data_count")
                out_start_idx.set_as(out_row_start_idx)
                cycle_lines = self.tiling_param.mask_cycle_lines
                output_inner_burst = self.tiling_param.mask_cycle_output_inner_burst
                repeat_times = lines // cycle_lines
                for index, input_tensor in enumerate(tensors):
                    inner_dim, output_idx = self.tiling_param.get_dims(index)
                    inner_dim_burst = self.tiling_param.get_mask_cycle_burst(index)
                    with inst.if_scope(inner_dim > 0):
                        with inst.if_scope(repeat_times > 0):
                            with inst.for_range(0, cycle_lines // 2) as line_idx:
                                def ping_pong_func(out_ub, tmp_ub, line_idx):
                                    out_ub_idx = output_inner_len * (line_idx - row_idx) + output_idx
                                    pre_redundant_cnt = out_ub_idx % self.ele_each_block
                                    gm2ub(inst, tmp_ub, input_tensor[inner_dim * line_idx - pre_redundant_cnt:],
                                          inner_dim * repeat_times * cycle_lines + pre_redundant_cnt)
                                    _concat_ub_vadd(inst, out_ub, tmp_ub, out_ub_idx, pre_redundant_cnt, inner_dim,
                                                    repeat_times, output_inner_burst, inner_dim_burst)

                                ping_pong_func(out_ub, tmp_ub0, row_idx + line_idx * 2)
                                ping_pong_func(out_ub, tmp_ub1, row_idx + line_idx * 2 + 1)
                        with inst.for_range(row_idx + repeat_times * cycle_lines, row_idx + lines) as line_idx:
                            out_ub_idx = output_inner_len * (line_idx - row_idx) + output_idx
                            pre_redundant_cnt = out_ub_idx % self.ele_each_block
                            gm2ub(inst, tmp_ub0, input_tensor[inner_dim * line_idx - pre_redundant_cnt:],
                                  inner_dim + pre_redundant_cnt)
                            _concat_ub_vadd(inst, out_ub, tmp_ub0, out_ub_idx, pre_redundant_cnt, inner_dim, 1, 0, 0)
                ub2gm(inst, output_tensor[out_start_idx:], out_ub, output_inner_len * lines)

    def _concat_small_inner_each_core_multi_line_by_scalar_each_loop(self, row_idx, lines, ub_length):
        """
        concat small inner by scalar each loop
        """
        inst = self.tik_instance
        tensors = self.input_tensors
        output_tensor = self.output_tensor
        output_inner_len = self.tiling_param.output_inner_length
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            tmp_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="tmp_ub")
            out_start_idx = output_inner_len * row_idx
            for index, input_tensor in enumerate(tensors):
                inner_dim, output_idx = self.tiling_param.get_dims(index)
                with inst.if_scope(inner_dim > 0):
                    gm2ub(inst, tmp_ub, input_tensor[inner_dim * row_idx:], inner_dim * lines)
                    with inst.for_range(0, lines) as line_idx:
                        with inst.for_range(0, inner_dim) as element_idx:
                            out_ub[output_inner_len * line_idx + output_idx + element_idx] = tmp_ub[inner_dim *
                                                                                                    line_idx +
                                                                                                    element_idx]
            ub2gm(inst, output_tensor[out_start_idx:], out_ub, output_inner_len * lines)

    def _concat_small_inner_each_core_multi_lines_first_block_element_rows(self, out_ub, tmp_ub, lines):
        """
        concat small inner dim when each_core_multi_lines_first_block_element_rows
        """
        inst = self.tik_instance
        tensors = self.input_tensors
        output_tensor = self.output_tensor
        output_inner_len = self.tiling_param.output_inner_length
        ub_list = [out_ub, tmp_ub]
        with inst.for_range(0, self.ele_each_block - 1) as ii:
            for index, input_tensor in enumerate(tensors):
                inner_dim, output_idx = self.tiling_param.get_dims(index)
                with inst.if_scope(inner_dim > 0):
                    cur_out_ub = ub_list[index % len(ub_list)]
                    gm2ub(inst, cur_out_ub, input_tensor[inner_dim * ii], inner_dim)
                    ub2gm(inst, output_tensor[output_inner_len * ii + output_idx], cur_out_ub, inner_dim)

        ii = self.ele_each_block - 1
        for index, input_tensor in enumerate(tensors):
            inner_dim, output_idx = self.tiling_param.get_dims(index)
            with inst.if_scope(inner_dim > 0):
                align_size = ceil_div(inner_dim, self.ele_each_block) * self.ele_each_block
                with inst.if_scope(tik.any(lines > self.ele_each_block, output_idx + align_size <= output_inner_len)):
                    burst = ceil_div(inner_dim, self.ele_each_block)
                    cur_out_ub = ub_list[index % len(ub_list)]
                    gm2ub(inst, cur_out_ub, input_tensor[inner_dim * ii], inner_dim, burst)
                    ub2gm(inst, output_tensor[output_inner_len * ii + output_idx], cur_out_ub, inner_dim, burst)
                with inst.else_scope():
                    reserve_count = align_size - inner_dim
                    burst = ceil_div(align_size, self.ele_each_block)
                    gm2ub(inst, tmp_ub, input_tensor[inner_dim * ii - reserve_count], align_size, burst)
                    gm2ub(inst, out_ub, output_tensor[output_inner_len * ii + output_idx - reserve_count],
                          align_size, burst)
                    _concat_ub_vadd(inst, out_ub, tmp_ub, reserve_count, reserve_count, inner_dim, 1, 0, 0)
                    ub2gm(inst, output_tensor[output_inner_len * ii + output_idx - reserve_count], out_ub,
                          align_size, burst)

    def _concat_with_vnchwconv(self, core_idx, out_dims, ub_len):
        """
        concat with vnchwconv
        """
        need_recover = False
        ori_dtype = self.dtype
        if self._check_need_convert2float16():
            ub_len = ub_len * common_util.get_data_size(self.dtype) // common_util.get_data_size("float16")
            self._convert_dtype("float16")
            need_recover = True
        inst = self.tik_instance
        output_inner_dim = self.tiling_param.output_inner_length
        inner_dim = self.tiling_param.min_inner_dim
        vnchwconv_input_lines = Constant.VNCHW_ELEMENT_FP16 // inner_dim
        output_lines_vnchwconv = inst.Scalar(dtype="int64", name="output_lines_vnchwconv")
        output_lines_vnchwconv.set_as(ub_len // len(self.input_tensors) // inner_dim //
                                      vnchwconv_input_lines * vnchwconv_input_lines)
        lines_each_core = ceil_div(ceil_div(out_dims, self.aicore_num), vnchwconv_input_lines) * vnchwconv_input_lines
        with inst.if_scope(lines_each_core < output_lines_vnchwconv):
            output_lines_vnchwconv.set_as(lines_each_core)
        output_tensor = self.output_tensor
        vnchwconv_times = ceil_div(lines_each_core, output_lines_vnchwconv)
        do_lines = inst.Scalar(dtype="int64", name="do_lines", init_value=output_lines_vnchwconv)
        all_in_ub_index = []
        input_tensors = self.input_tensors
        for index, _ in enumerate(input_tensors):
            _, output_idx = self.tiling_param.get_dims(index)
            all_in_ub_index.append(inst.Scalar(dtype="int64", init_value=output_idx * output_lines_vnchwconv))
        repeat_times = ceil_div(inner_dim * output_lines_vnchwconv, Constant.VNCHW_ELEMENT_FP16)
        tensor_count = len(input_tensors)
        with inst.new_stmt_scope():
            # when input count is 64, double buffer will make ccec compile stack overflow
            with inst.for_range(0, vnchwconv_times, thread_num=1) as idx:
                in_ub = inst.Tensor(self.dtype, (ub_len,), scope=tik.scope_ubuf, name="in_ub")
                out_ub = inst.Tensor(self.dtype, (ub_len,), scope=tik.scope_ubuf, name="out_ub")
                row_idx = output_lines_vnchwconv * idx + lines_each_core * core_idx
                with inst.if_scope(row_idx < out_dims):
                    with inst.if_scope(row_idx + output_lines_vnchwconv > out_dims):
                        do_lines.set_as(out_dims - row_idx)
                    out_row_start_idx = output_inner_dim * row_idx
                    for index, input_tensor in enumerate(self.input_tensors):
                        inner_dim, output_idx = self.tiling_param.get_dims(index)
                        in_ub_index = all_in_ub_index[index]
                        in_start_idx = inner_dim * row_idx
                        gm2ub(inst, in_ub[in_ub_index], input_tensor[in_start_idx], inner_dim * do_lines)
                        dst_list = []
                        for i, _ in enumerate(range(16)):
                            tensor_start = index * self.ele_each_block * inner_dim
                            row_start = (self.ele_each_block * (i % inner_dim) +
                                         self.ele_each_block * (i // inner_dim) * output_inner_dim)
                            dst_list.append(out_ub[tensor_start + row_start])
                        src_list = [in_ub[in_ub_index + self.ele_each_block * i] for i in range(16)]
                        with inst.if_scope(repeat_times == 1):
                            inst.vnchwconv(False, False, dst_list, src_list, repeat_times, 0, 0)
                        with inst.else_scope():
                            inst.vnchwconv(False, False, dst_list, src_list, repeat_times, 16 * tensor_count, 16)
                    for index, _ in enumerate(input_tensors):
                        dst_list = [in_ub[index * self.ele_each_block + self.ele_each_block * tensor_count * i]
                                    for i in range(16)]
                        src_list = [out_ub[index * Constant.VNCHW_ELEMENT_FP16 + self.ele_each_block * i] \
                        for i in range(16)]
                        with inst.if_scope(repeat_times == 1):
                            inst.vnchwconv(False, False, dst_list, src_list, repeat_times, 0, 0)
                        with inst.else_scope():
                            inst.vnchwconv(False, False, dst_list, src_list, repeat_times,
                                           16 * tensor_count, 16 * tensor_count)
                    ub2gm(inst, output_tensor[out_row_start_idx], in_ub, output_inner_dim * do_lines)
        if need_recover:
            self._convert_dtype(ori_dtype)

    def _check_need_convert2float16(self):
        """
        _check_need_convert2float16
        """
        if not tbe_platform.api_check_support("tik.vnchwconv", self.dtype) or self.type_size != 2:
            return True
        return False

    def _convert_dtype(self, dtype):
        """
        convert dtype
        """
        for index, _ in enumerate(self.input_tensors):
            self.input_tensors[index] = self.input_tensors[index].reinterpret_cast_to(dtype)
        self.output_tensor = self.output_tensor.reinterpret_cast_to(dtype)
        src_dtype_size = common_util.get_data_size(self.dtype)
        dst_dtype_size = common_util.get_data_size(dtype)
        if self.type_size != dst_dtype_size:
            self.tiling_param.update_tiling(self.dtype, dtype)
        self.type_size = dst_dtype_size
        self.ele_each_block = self.ele_each_block * src_dtype_size // dst_dtype_size
        self.dtype = dtype

    def _concat_first_dim(self, core_idx):
        """
        concat first dim
        """
        aicore_num = self.aicore_num
        inst = self.tik_instance
        ub_len = self.ub_buffer_length
        output_tensor = self.output_tensor
        with inst.new_stmt_scope():
            in_out_ub = inst.Tensor(dtype=self.dtype, shape=(ub_len,), scope=tik.scope_ubuf, name="in_out_ub")
            for index, input_tensor in enumerate(self.input_tensors):
                inner_dim, output_idx = self.tiling_param.get_dims(index)
                data_count_each_core = ceil_div(ceil_div(inner_dim, aicore_num),
                                                self.ele_each_block) * self.ele_each_block
                copy_data_count = inst.Scalar(dtype="int64", name="copy_data_count")
                core_data_count = inst.Scalar(dtype="int64", name="core_data_count")
                with inst.if_scope(data_count_each_core * core_idx < inner_dim):
                    core_data_count.set_as(inner_dim - data_count_each_core * core_idx)
                    with inst.if_scope(core_data_count > data_count_each_core):
                        core_data_count.set_as(data_count_each_core)
                    repeat_times = ceil_div(core_data_count, ub_len)
                    with inst.for_range(0, repeat_times) as loop_idx:
                        copy_data_count.set_as(core_data_count - loop_idx * ub_len)
                        with inst.if_scope(copy_data_count > ub_len):
                            copy_data_count.set_as(ub_len)
                        input_addr = ub_len * loop_idx + data_count_each_core * core_idx
                        output_addr = ub_len * loop_idx + output_idx + data_count_each_core * core_idx
                        with inst.if_scope(tik.any(copy_data_count % self.ele_each_block == 0,
                                                   core_idx == 0)):
                            gm2ub(inst, in_out_ub, input_tensor[input_addr], copy_data_count)
                            ub2gm(inst, output_tensor[output_addr], in_out_ub, copy_data_count)
                        with inst.else_scope():
                            align_count = ceil_div(copy_data_count, self.ele_each_block) * self.ele_each_block
                            rollback_count = align_count - copy_data_count
                            gm2ub(inst, in_out_ub, input_tensor[input_addr - rollback_count], align_count)
                            ub2gm(inst, output_tensor[output_addr - rollback_count], in_out_ub, align_count)


def _check_shape(input_values, shape_name):
    """
    check the length of input shape must be equal
    """
    dim_num = len(input_values[0].get(shape_name))
    for _, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get(shape_name)
        if len(shape_input) != dim_num:
            error_manager_vector.raise_err_inputs_shape_not_equal("concat_v2_d",
                                                                  "shape_input",
                                                                  "dim_num",
                                                                  len(shape_input),
                                                                  dim_num,
                                                                  dim_num)


# 'pylint: disable=unused-argument
def _check_params(input_values, axis):
    """
    check params
    """
    dtype_lists = []
    for input_value in input_values:
        dtype_lists.append(input_value.get("dtype"))

    dtype = dtype_lists[0]
    for index, dtype_ in enumerate(dtype_lists):
        if dtype != dtype_:
            error_manager_vector.raise_err_inputs_dtype_not_equal("concat_v2_d",
                                                           "input_values[0]",
                                                           "input_values[%s]" % index,
                                                           dtype,
                                                           dtype_)


# 'pylint: disable=unused-argument
def concat_v2_d_tik(input_values, output_data, axis, kernel_name="concat_v2_d"):
    """
    algorithm: concat_v2_d

    Parameters
    ----------
    input_values : A list of dict objects.
                 dict with keys(shape and dtype) of tensor
                 dtype only support float32, int8, int16, int32, int64, uint8,
                 uint16, uint32, uint64, float16
    output_data : A dict resulting from concatenation of the input tensors
    axis : scalar,in the range [-rank(values), rank(values))
    kernel_name : cce kernel name, default value is "concat_v2_d"

    Returns
    -------
    tik instance
    """
    if not util_common.is_unknown_rank_input(input_values):
        _check_params(input_values, axis)

    # update axis base on input format
    for _, _input_dict in enumerate(input_values):
        ori_shape = _input_dict.get("ori_shape")
        if -2 not in ori_shape:
            # can not use unkownrank shape to update the axis
            input_format = _input_dict.get("format")
            ori_format = _input_dict.get("ori_format")
            axis = util_common.update_axis_for_other_format(ori_shape, axis, input_format, ori_format)
            break

    concat_instance = ConcatV2(input_values, axis, kernel_name)
    concat_instance.concat_compute()
    return concat_instance.build()
