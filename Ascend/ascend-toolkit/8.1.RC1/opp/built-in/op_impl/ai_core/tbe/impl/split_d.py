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
split_d
"""
import functools

import numpy as np
from impl import copy_only
from impl import split_last_dim
from impl.split_equal import SplitEqual
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tvm
from impl.util.util_binary import get_bit_len
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from tbe.dsl.compute.array import split_compute_com
from tbe.dsl.static_schedule.split_schedule import split_schedule_com

# vtranspose can deal 16*16
TRANSPOSE_SIZE = 256


# 'pylint: disable = unused-argument,too-many-locals
def get_op_support_info(input_value, output_data, split_dim, num_split, kernel_name="split_d"):
    """
    get_op_support_info
    """
    shape_value_len = len(input_value.get("shape"))
    format_value = input_value.get("format").upper()
    ori_format = input_value.get("ori_format").upper()
    if format_value == "NC1HWC0":
        split_dim = shape_util.axis_transform_5d(split_dim, ori_format)
    if split_dim < 0:
        split_dim += shape_value_len
    if format_value in ("ND", "NC1HWC0", "FRACTAL_NZ"):
        axis_split_matrix = []
        for i in range(0, shape_value_len - 1):
            if i != split_dim:
                output_list = []
                for j in range(0, num_split):
                    output_0 = [j, [i]]
                    output_list.append(output_0)
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput(*output_list)]
                axis_split_matrix.append(split_0)
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def check_supported(input_value, output_data, split_dim, num_split, kernel_name="split_d"):
    """
    check_supported
    """
    return True


# 'pylint: disable=locally-disabled,too-many-locals,too-many-statements,too-many-instance-attributes,too-many-arguments
class SplitMov:
    """Function: use to finish SplitMov main functions
    """

    def __init__(self, shape, dtype, split_dim, num_split, size_splits=None, kernel_name="split_d"):
        """init base parameters
        """
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.kernel_name = kernel_name
        self.dtype = dtype
        self.dtype_size = get_bit_len(self.dtype) // 8
        self.one_block_ele = 32 // self.dtype_size
        self.half_ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // self.dtype_size // 2 -
                            self.one_block_ele)

        if not size_splits:
            size_splits = [shape[split_dim] // num_split] * num_split
        self.size_splits = []
        self.output_shapes = []
        if split_dim == 0:
            self.input_shape = [int(np.prod(shape))]
            for size in size_splits:
                self.size_splits.append(self.input_shape[0] // shape[split_dim] * size)
                self.output_shapes.append([self.input_shape[0] // shape[split_dim] * size])
            self.split_dim = 0
        else:
            out_dim = int(np.prod(shape[:split_dim]))
            if out_dim == 1:
                self.input_shape = [int(np.prod(shape))]
                for size in size_splits:
                    self.size_splits.append(self.input_shape[0] // shape[split_dim] * size)
                    self.output_shapes.append([self.input_shape[0] // shape[split_dim] * size])
                self.split_dim = 0
            else:
                self.input_shape = [int(np.prod(shape[:split_dim])), int(np.prod(shape[split_dim:len(shape)]))]
                for size in size_splits:
                    self.size_splits.append(self.input_shape[1] // shape[split_dim] * size)
                    self.output_shapes.append([self.input_shape[0], self.input_shape[1] // shape[split_dim] * size])
                self.split_dim = 1

        self.input_tensor, self.output_tensors = self.init_gm_tensor()

    def init_gm_tensor(self):
        """init gm tensor
        """
        input_tensor = self.tik_instance.Tensor(self.dtype, self.input_shape, name="gm_input", scope=tik.scope_gm)

        output_tensors = []
        for index, tensor_shape in enumerate(self.output_shapes):
            tensor_name = "gm_output_{}".format(str(index))
            gm_tensor = self.tik_instance.Tensor(self.dtype, tensor_shape, name=tensor_name, scope=tik.scope_gm)
            output_tensors.append(gm_tensor)

        return input_tensor, output_tensors

    def get_one_core_ele(self, output_shape):
        """get_one_core_ele
        """
        total_ele = output_shape[self.split_dim]
        last_ele = total_ele % self.aicore_num

        if last_ele == 0:
            one_core_ele = total_ele // self.aicore_num
        else:
            one_core_ele = (total_ele // self.aicore_num // self.one_block_ele * self.one_block_ele)
            last_ele = total_ele - (self.aicore_num - 1) * one_core_ele

        return one_core_ele, last_ele

    def split_compute_for_tensor(self, move_in_index, output_tensor, one_core_ele):
        """split_compute_for_tensor
        """
        loop_burst_len = 0
        if one_core_ele < self.half_ub_ele:
            loop_num = 0
            one_loop_ele = 0
            last_ele = one_core_ele
            ub_size = self.half_ub_ele
        else:
            if one_core_ele % self.half_ub_ele < self.one_block_ele:
                ub_size = self.half_ub_ele - self.one_block_ele
            else:
                ub_size = self.half_ub_ele
            loop_num = one_core_ele // ub_size
            one_loop_ele = ub_size
            last_ele = one_core_ele % ub_size
            loop_burst_len = ub_size // self.one_block_ele
        if loop_num > 0:
            if loop_num > 1:
                multi_thread = 2
            else:
                multi_thread = 1
            with self.tik_instance.for_range(0, loop_num, thread_num=multi_thread) as inner_loop:
                ub_tensor = self.tik_instance.Tensor(self.dtype, (ub_size,), name="ub_tmp", scope=tik.scope_ubuf)
                offset = inner_loop * one_loop_ele
                self.tik_instance.data_move(ub_tensor, self.input_tensor[move_in_index:][offset], 0, 1, loop_burst_len,
                                            0, 0)
                self.tik_instance.data_move(output_tensor[offset], ub_tensor, 0, 1, loop_burst_len, 0, 0)
        if last_ele > 0:
            with self.tik_instance.for_range(0, 1) as _:
                ub_tensor = self.tik_instance.Tensor(self.dtype, (ub_size,), name="ub_tmp", scope=tik.scope_ubuf)
                offset = loop_num * one_loop_ele
                if last_ele // self.one_block_ele != 0:
                    last_burst_len = last_ele // self.one_block_ele
                    self.tik_instance.data_move(ub_tensor, self.input_tensor[move_in_index:][offset], 0, 1,
                                                last_burst_len, 0, 0)
                    self.tik_instance.data_move(output_tensor[offset], ub_tensor, 0, 1, last_burst_len, 0, 0)

                if last_ele % self.one_block_ele != 0:
                    ub_last = self.tik_instance.Tensor(self.dtype, (self.one_block_ele,),
                                                       name="ub_last",
                                                       scope=tik.scope_ubuf)
                    offset = one_core_ele - self.one_block_ele
                    self.tik_instance.data_move(ub_last, self.input_tensor[move_in_index:][offset], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(output_tensor[offset], ub_last, 0, 1, 1, 0, 0)

    def split_compute_first_dim_for_core(self, core_index):
        """split_compute_first_dim_for_core
        """
        out_offset = 0
        for tensor_index, output_shape in enumerate(self.output_shapes):
            one_core_ele, last_ele = self.get_one_core_ele(output_shape)
            if last_ele == 0:
                move_in_index = (out_offset + one_core_ele * core_index)
                move_out_index = (one_core_ele * core_index)
                self.split_compute_for_tensor(move_in_index, self.output_tensors[tensor_index][move_out_index:],
                                              one_core_ele)
            else:
                with self.tik_instance.if_scope(core_index < self.aicore_num - 1):
                    move_in_index = (out_offset + one_core_ele * core_index)
                    move_out_index = (one_core_ele * core_index)
                    self.split_compute_for_tensor(move_in_index, self.output_tensors[tensor_index][move_out_index:],
                                                  one_core_ele)
                with self.tik_instance.else_scope():
                    move_in_index = (out_offset + one_core_ele * core_index)
                    move_out_index = (one_core_ele * core_index)
                    self.split_compute_for_tensor(move_in_index, self.output_tensors[tensor_index][move_out_index:],
                                                  last_ele)
            out_offset += output_shape[self.split_dim]

    def split_compute_last_dim_for_core(self, core_index):
        """split_compute_last_dim_for_core
        """
        is_div = self.input_shape[0] % self.aicore_num
        if is_div != 0:
            out_offset = 0
            out_loop = self.input_shape[0]
            for tensor_index, output_shape in enumerate(self.output_shapes):
                one_core_ele, last_ele = self.get_one_core_ele(output_shape)
                if last_ele == 0:
                    with self.tik_instance.for_range(0, out_loop, thread_num=2) as loop_index:
                        move_in_index = (out_offset + one_core_ele * core_index +
                                         loop_index * self.input_shape[self.split_dim])
                        move_out_index = (one_core_ele * core_index + loop_index * output_shape[self.split_dim])
                        self.split_compute_for_tensor(move_in_index, self.output_tensors[tensor_index][move_out_index:],
                                                      one_core_ele)
                else:
                    with self.tik_instance.if_scope(core_index < self.aicore_num - 1):
                        with self.tik_instance.for_range(0, out_loop, thread_num=2) as loop_index:
                            move_in_index = (out_offset + one_core_ele * core_index +
                                             loop_index * self.input_shape[self.split_dim])
                            move_out_index = (one_core_ele * core_index + loop_index * output_shape[self.split_dim])
                            self.split_compute_for_tensor(move_in_index,
                                                          self.output_tensors[tensor_index][move_out_index:],
                                                          one_core_ele)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(0, out_loop, thread_num=2) as loop_index:
                            move_in_index = (out_offset + one_core_ele * core_index +
                                             loop_index * self.input_shape[self.split_dim])
                            move_out_index = (one_core_ele * core_index + loop_index * output_shape[self.split_dim])
                            self.split_compute_for_tensor(move_in_index,
                                                          self.output_tensors[tensor_index][move_out_index:], last_ele)
                out_offset += output_shape[self.split_dim]
        else:
            out_offset = 0
            out_loop = self.input_shape[0] // self.aicore_num
            thread_num = 1
            if out_loop != 1:
                thread_num = 2
            for tensor_index, output_shape in enumerate(self.output_shapes):
                one_core_ele = output_shape[self.split_dim]
                with self.tik_instance.for_range(0, out_loop, thread_num=thread_num) as loop_index:
                    move_in_index = (out_offset + core_index * out_loop * self.input_shape[self.split_dim] +
                                     loop_index * self.input_shape[self.split_dim])
                    move_out_index = (core_index * out_loop * output_shape[self.split_dim] +
                                      loop_index * output_shape[self.split_dim])
                    self.split_compute_for_tensor(move_in_index, self.output_tensors[tensor_index][move_out_index:],
                                                  one_core_ele)
                out_offset += output_shape[self.split_dim]

    def split_mov_compute(self):
        """split_mov_compute
        """
        if self.split_dim == 0:
            with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as index:
                self.split_compute_first_dim_for_core(index)
        else:
            with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as index:
                self.split_compute_last_dim_for_core(index)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_tensor],
                                   outputs=self.output_tensors,
                                   enable_l2=False)

    def check_whether_use_split_mov(self):
        """check if split_d schedule support this shape
        """
        is_supported = True
        for _, output_shape in enumerate(self.output_shapes):
            split_dim_len = output_shape[self.split_dim]
            if self.split_dim == 0 and \
                    split_dim_len // self.aicore_num < 2 * self.one_block_ele:
                is_supported = False
                return is_supported
            if self.split_dim == 1 and \
                    self.input_shape[0] % self.aicore_num == 0 and \
                    split_dim_len < 2 * self.one_block_ele:
                is_supported = False
                return is_supported
            if self.split_dim == 1 and \
                    self.input_shape[0] % self.aicore_num != 0 and \
                    split_dim_len // self.aicore_num < 2 * self.one_block_ele:
                is_supported = False
                return is_supported

        return is_supported


class SplitLastDimVnv:
    """Function: use to finish SplitLastDimVnv main functions
    """

    def __init__(self, shape, dtype, output_shapes, split_dim, num_split, kernel_name):
        """init base parameters
        """
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype = dtype
        self.dtype_size = get_bit_len(self.dtype) // 8
        self.block_ele = 32 // self.dtype_size
        self.half_ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // self.dtype_size // 2 - self.block_ele)
        # shape length must be 2
        self.shape = shape
        self.output_shapes = output_shapes
        # split_dim must be 1
        self.split_dim = split_dim
        self.num_split = num_split
        self.kernel_name = kernel_name
        self.input_tensor, self.output_tensors = self.init_gm_tensor()

    def init_gm_tensor(self):
        """init gm tensor
        """
        input_tensor = self.tik_instance.Tensor(self.dtype, self.shape, name="gm_input", scope=tik.scope_gm)

        output_tensors = []
        for index, tensor_shape in enumerate(self.output_shapes):
            tensor_name = "gm_output_{}".format(str(index))
            gm_tensor = self.tik_instance.Tensor(self.dtype, tensor_shape, name=tensor_name, scope=tik.scope_gm)
            output_tensors.append(gm_tensor)

        return input_tensor, output_tensors

    def split_last_dim_vnc_compute_for_core(self, src_offset_core, dst_offset_core, core_seg, tail_ele):
        """split_last_dim_vnc_compute_for_core
        """
        max_seg = self.half_ub_ele // (TRANSPOSE_SIZE * (2 * self.num_split + 2))
        if tail_ele != 0:
            core_seg = core_seg - 1
        core_seg = max(core_seg, 0)
        loop_num = core_seg // max_seg
        last_seg = core_seg % max_seg

        def _inner(src_offset, dst_offset, seg):
            """inner function
            """
            ub_x = self.tik_instance.Tensor(self.dtype, (max_seg * TRANSPOSE_SIZE * self.num_split,), tik.scope_ubuf,
                                            "ub_x")
            ub_y = self.tik_instance.Tensor(self.dtype, (max_seg * TRANSPOSE_SIZE * self.num_split,), tik.scope_ubuf,
                                            "ub_y")
            ub_m = self.tik_instance.Tensor(self.dtype, (max_seg * TRANSPOSE_SIZE,), tik.scope_ubuf, "ub_m")
            ub_n = self.tik_instance.Tensor(self.dtype, (max_seg * TRANSPOSE_SIZE,), tik.scope_ubuf, "ub_n")

            # copy gm to ub
            self.tik_instance.data_move(ub_x, self.input_tensor[src_offset], 0, 1,
                                        seg * TRANSPOSE_SIZE * self.num_split // self.block_ele, 0, 0)

            # vadds & vtranspose
            for num_idx in range(self.num_split):
                src_offset_ub = num_idx * self.block_ele
                dst_offset_ub = num_idx * TRANSPOSE_SIZE

                self.tik_instance.vadds(128, ub_m, ub_x[src_offset_ub], 0, 2 * seg, 1,
                                        self.num_split, 8, self.num_split * 8)
                for trans_idx in range(seg):
                    src_offset_trans = trans_idx * TRANSPOSE_SIZE
                    dst_offset_trans = dst_offset_ub + self.num_split * trans_idx * TRANSPOSE_SIZE
                    self.tik_instance.vtranspose(ub_y[dst_offset_trans], ub_m[src_offset_trans])

            for num_idx in range(self.num_split):
                src_offset_ub = num_idx * self.block_ele
                self.tik_instance.vadds(128, ub_m, ub_y[src_offset_ub], 0, 2 * seg,
                                        1, self.num_split, 8, self.num_split * 8)
                for trans_idx in range(seg):
                    src_offset_trans = trans_idx * TRANSPOSE_SIZE
                    dst_offset_trans = trans_idx * TRANSPOSE_SIZE
                    self.tik_instance.vtranspose(ub_n[dst_offset_trans], ub_m[src_offset_trans])
                # copy ub to gm
                self.tik_instance.data_move(self.output_tensors[num_idx][dst_offset], ub_n, 0, 1,
                                            seg * TRANSPOSE_SIZE // self.block_ele, 0, 0)

        thread = 2 if loop_num > 1 else 1
        with self.tik_instance.for_range(0, loop_num, thread_num=thread) as loop_idx:
            src_offset = src_offset_core + loop_idx * max_seg * TRANSPOSE_SIZE * self.shape[1]
            dst_offset = dst_offset_core + loop_idx * max_seg * TRANSPOSE_SIZE * self.output_shapes[0][1]
            _inner(src_offset, dst_offset, max_seg)
        if last_seg != 0:
            with self.tik_instance.for_range(0, 1):
                src_offset = src_offset_core + loop_num * max_seg * TRANSPOSE_SIZE * self.shape[1]
                dst_offset = dst_offset_core + loop_num * max_seg * TRANSPOSE_SIZE * \
                    self.output_shapes[0][1]
                _inner(src_offset, dst_offset, last_seg)
        if tail_ele != 0:
            with self.tik_instance.for_range(0, 1):
                src_offset = (self.shape[0] - TRANSPOSE_SIZE) * self.shape[1]
                dst_offset = (self.shape[0] - TRANSPOSE_SIZE) * self.output_shapes[0][1]
                _inner(src_offset, dst_offset, 1)

    def split_last_dim_vnc_compute(self):
        """split_last_dim_vnc_compute
        """
        align_seg = (self.shape[0] + TRANSPOSE_SIZE - 1) // TRANSPOSE_SIZE
        tail_ele = align_seg * TRANSPOSE_SIZE - self.shape[0]
        one_core_seg = (align_seg + self.core_num - 1) // self.core_num
        act_core_num = align_seg // one_core_seg
        if align_seg % one_core_seg != 0:
            act_core_num = act_core_num + 1
        last_core_seg = align_seg - (act_core_num - 1) * one_core_seg

        with self.tik_instance.for_range(0, act_core_num, block_num=act_core_num) as core_idx:
            src_offset_core = core_idx * one_core_seg * TRANSPOSE_SIZE * self.shape[1]
            dst_offset_core = core_idx * one_core_seg * TRANSPOSE_SIZE * self.output_shapes[0][1]
            if tail_ele == 0 and last_core_seg == one_core_seg:
                self.split_last_dim_vnc_compute_for_core(src_offset_core,
                                                         dst_offset_core, one_core_seg, 0)
            else:
                with self.tik_instance.if_scope(core_idx < act_core_num - 1):
                    self.split_last_dim_vnc_compute_for_core(src_offset_core,
                                                             dst_offset_core, one_core_seg, 0)
                with self.tik_instance.else_scope():
                    self.split_last_dim_vnc_compute_for_core(src_offset_core, dst_offset_core,
                                                             last_core_seg, tail_ele)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_tensor],
                                   outputs=self.output_tensors,
                                   enable_l2=False)
        return self.tik_instance


def split_d_compute(input_value, output_data, split_dim, num_split, kernel_name="split_d"):
    """Split a tensor into `num_split` tensors along one dimension.

    Parameters
    ----------
    input_value: TVM tensor
        input tensor.
    output_data: list or tuple
        the list of output tensor.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        an integer indicating the number of split_d along `split_dim`.
    kernel_name: str
        cce kernel name, default value is "split_d".

    Returns
    -------
    output_shape_list: list
        the list of output shapes.
    output_tensor_list: list
        the list of output tensors, output tensor type is TVM tensor.
    """
    shape = shape_util.shape_to_list(input_value.shape)
    size = shape[split_dim] // num_split

    size_splits = [size] * num_split

    output_shape_list, output_tensor_list = split_compute_com(input_value, split_dim, size_splits)

    return output_shape_list, output_tensor_list


def op_select_format(input_value, output_data, split_dim, num_split, kernel_name="split_d"):
    """
    1.when input x's ori_shape in ["NCHW", "NHWC"] and split_d by
    dim N, H, W and dim C of x's ori_shape can be divisible by 16(32
    when dtype is int8). the Op SplitD can support ND and NC1HWC0.

        for example:
        x : Tensor of (shape=(2, 16, 32), "ND")
        the Op Select can process with NC1HWC0:
        x : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")

    2.when input x's ori_shape dimension is greater then 2 and
    do not split with last 2 dim. the Op SplitD can support ND and FRACTAL_NZ.

        for example:
        x : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        the Op Select can process with NC1HWC0:
        x : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    """
    dtype = input_value.get("dtype").lower()
    if dtype == "int8":
        c0_len = 32
    else:
        c0_len = 16
    output_org_shape_list = []
    output_org_format_list = []
    is_support_hd = True
    support_ori_format = \
        util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) \
        + util_common.get_fused_format_str(["N", "H", "W", "C"])
    input_ori_shape = input_value.get("ori_shape")
    input_ori_format = input_value.get("ori_format")
    split_dim = split_dim % len(input_ori_shape)

    for _, output_dict in enumerate(output_data):
        ori_format = output_dict.get("ori_format").upper()
        ori_shape = output_dict.get("ori_shape")
        output_org_shape_list.append(ori_shape)
        output_org_format_list.append(ori_format)

        if ori_format not in support_ori_format or len(input_ori_shape) != len(input_ori_format) \
                or len(ori_format) != len(ori_shape):
            is_support_hd = False
            break

        # when split_d by N,H,W, support NC1HWC0
        if ori_format[split_dim] != "C":
            break

        # when split_d by C, but output size not C0 align donot support NC1HWC0
        if ori_shape[split_dim] % c0_len != 0:
            is_support_hd = False
            break

    is_support_nz = False
    if len(input_ori_shape) > 2:
        # if do not split with last two dim, will support nz
        if split_dim < len(input_ori_shape) - 2:
            is_support_nz = True

    split_with_5hd_not_align = \
        split_last_dim.SplitWith5HD(input_value, output_data,
                                    split_dim, num_split, kernel_name)
    is_support_other_5hd = split_with_5hd_not_align.check_op_select()

    dtype_base = ["float16", "float", "int32", "int8", "int16", "int64", "uint8",
                  "uint16", "uint32", "uint64", "bool"]
    dtype_5hd = ["float16", "float", "int32", "int8", "int16", "uint16", "uint32"]
    dtype_base_out = dtype_base.copy()
    format_base_out = ["ND"] * len(dtype_base)

    if is_support_hd and not util_common.is_dynamic_input([input_value]):
        other_format = "NC1HWC0" if len(input_ori_shape) == 4 else "NDC1HWC0"
        dtype_base_out = dtype_base_out + dtype_5hd
        format_base_out = format_base_out + [other_format] * len(dtype_5hd)

    if is_support_nz and not util_common.is_dynamic_input([input_value]):
        dtype_base_out = dtype_base_out + dtype_base
        format_base_out = format_base_out + ["FRACTAL_NZ"] * len(dtype_base)

    if is_support_other_5hd and not util_common.is_dynamic_input([input_value]):
        dtype_base_out = dtype_base_out + ["float16", "int16", "uint16"]
        format_base_out = format_base_out + ["NC1HWC0"] * 3

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    input0 = util_select_op_base.gen_param(classify="input0", name="x", datatype=dtype_str,
                                           format=format_str)
    output0 = util_select_op_base.gen_param(classify="output0", name="y", datatype=dtype_str,
                                            format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.DYNAMIC_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def split_d(input_value, output_data, split_dim, num_split, kernel_name="split_d"):
    """Split a tensor into `num_split` tensors along one dimension.

    Parameters
    ----------
    input_value: dict
        the dict of input tensor.
    output_data: list or tuple
        the list of output tensor.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        an integer indicating the number of split_d along `split_dim`.
    kernel_name: str
        cce kernel name, default value is "split_d".

    Returns
    -------
    None.
    """
    input_value = util_common.update_shape_base_other_format(input_value)
    input_format = input_value.get("format")
    ori_format = input_value.get("ori_format")
    ori_shape = input_value.get("ori_shape")
    # update axis base on input format
    split_dim = util_common.update_axis_for_other_format(ori_shape, split_dim,
                                                         input_format, ori_format)

    shape = input_value.get("shape")
    dtype = input_value.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                  "float16", "float32")

    para_check.check_shape(shape, param_name="input_value")
    para_check.check_dtype(dtype_lower, check_list, param_name="input_value")

    shape_len = len(shape)
    split_dim = shape_util.axis_check(shape_len, split_dim)

    if num_split < 1:
        expected_value = "must be greater or equal to 1"
        real_value = "less to 1"
        error_manager_vector.raise_err_input_value_invalid("split", "The num_split", expected_value,
                                                           real_value)

    split_with_5hd_not_align = split_last_dim.SplitWith5HD(input_value, output_data,
                                                           split_dim, num_split, kernel_name)
    if split_with_5hd_not_align.check_5hd_vnchw():
        split_with_5hd_not_align.do_5hd_split_cut_by_batch()
        return

    if shape[split_dim] % num_split != 0:
        error_detail = "The num_split (%d) must be divisible by the length of split_dim (%d)" \
                       % (num_split, shape[split_dim])
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "input_value", error_detail)

    if num_split == 1:
        copy_only.copy_only(input_value, input_value, kernel_name)
        return

    split_mov = SplitMov(shape, dtype_lower, split_dim, num_split, None, kernel_name)
    new_shape = split_mov.input_shape
    new_split_dim = split_mov.split_dim
    new_size_splits = split_mov.size_splits
    new_output_shapes = split_mov.output_shapes
    input_size = functools.reduce(lambda x, y: x * y, new_shape)

    if dtype_lower == "float16" and new_split_dim == len(new_shape) - 1 and \
            new_size_splits[0] == 1 and num_split <= 16 \
            and input_size >= TRANSPOSE_SIZE * num_split:
        split_vnc = SplitLastDimVnv(new_shape, dtype_lower, new_output_shapes,
                                    new_split_dim, num_split, kernel_name)
        split_vnc.split_last_dim_vnc_compute()
        return

    if split_last_dim.check_use_last_dim_branch(new_shape, dtype_lower, new_split_dim, num_split,
                                                new_size_splits):
        split_last_dim.split_last_dim(new_shape, dtype_lower, new_split_dim, num_split,
                                      new_size_splits, kernel_name)
        return

    if split_mov.check_whether_use_split_mov():
        split_mov.split_mov_compute()
        return

    is_split_last_dim = new_split_dim == 1
    if is_split_last_dim:
        re_args = SplitEqual.reinter_split_equal(new_shape, dtype_lower, new_size_splits)
        re_shape, re_dtype, re_size_splits, resize = re_args
        if resize > 0:
            split_equal = SplitEqual(re_shape, re_dtype, new_split_dim, re_size_splits, resize, kernel_name)
            if split_equal.check_support():
                split_equal.run()
                return

    data = tvm.placeholder(shape, name="data", dtype=dtype_lower)
    output_shape_list, output_tensor_list = split_d_compute(data, output_data,
                                                            split_dim, num_split, kernel_name)

    sch, build_list = split_schedule_com(data, split_dim, output_shape_list, output_tensor_list)

    with tbe_build.build_config():
        tvm.build(sch, build_list, "cce", name=kernel_name)
