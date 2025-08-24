# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
split_v
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from tbe.common.platform import get_bit_len


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


def align_value(value, factor):
    """
    Alignment based on factor.

    Parameters
    ----------
    value: input number
    factor: alignment base

    Returns
    -------
    res:
    """
    return (value + factor - 1) // factor * factor


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments
# 'pylint: disable=too-many-instance-attributes,too-many-public-methods,too-many-lines
class SplitV():
    """
    Function: class that execute split_v
    """

    MAX_SHAPE_SIZE = 2**32 - 1
    # tiling param num
    TILING_ARG_NUM = 24
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # 8 bit
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = 32
    # vtranspose can deal 16*16
    TRANSPOSE_SIZE = 256

    # tiling mode
    # num_split is 1
    MODE_1 = 1
    # split axis 0, or shape_before is 1
    MODE_2 = 2
    # split axis 0, or shape_before is 1, and shape_dim is greater than core num
    MODE_8 = 8
    # split axis is 1, tiling with shape_before
    MODE_3 = 3
    # only support fp16, num_split <= 16, and size_splits[i] is 1
    MODE_4 = 4
    # size_splits[i] is smaller than 32B, e.g [187264,33], 33->[5,6,7,4,3,2,6]
    MODE_5 = 5
    # only split_v, e.g int16,[48000,256], 256->[80,80,80,1,1,1,13]
    MODE_6 = 6
    # only split_v, only support fp16, e.g [2028,85], 85->[2,2,1,80]
    MODE_7 = 7
    # sub mode of mode 3
    SUB_MODE_1 = 1
    SUB_MODE_2 = 2
    SUB_MODE_3 = 3
    SUB_MODE_4 = 4
    SUB_MODE_5 = 5

    def __init__(self, x, size_splits, split_dim, y, num_split, kernel_name):
        """
        Init split_v parameters

        Parameters
        ----------
        x: dict
            the dict of input tensor.
        size_splits: dict
            the dict of input size_splits tensor.
            Specifies a list containing the sizes of each output tensor along the split dimension.
        split_dim: dict
            the dict of input split_dim tensor.
            An int, specifies the dimension along which to split.
        y: list or tuple
            the list of output tensor.
        num_split: int
            an integer indicating the number of outputs.
        kernel_name: str
            cce kernel name, default value is "split_v".

        Returns
        -------
        None
        """
        self.x = x
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.num_split = num_split
        self.kernel_name = kernel_name
        self.input_dtype = x.get("dtype").lower()
        if self.input_dtype == "bfloat16":
            self.input_dtype == "float16"

        self.is_split_v = False
        self.size_splits_dtype = "int32"
        if len(size_splits) > 0:
            self.is_split_v = True
            self.size_splits_dtype = size_splits.get("dtype").lower()

        self.split_dim_dtype = split_dim.get("dtype").lower()
        self.output_dtype = y[0].get("dtype").lower()
        self.input_dsize = get_bit_len(self.input_dtype) // self.EIGHT_BIT
        self.size_splits_dsize = get_bit_len(self.size_splits_dtype) // self.EIGHT_BIT
        self.block_elems = self.BLOCK_BYTES // self.input_dsize
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - self.RESERVED_UB_SIZE
        self.ub_elems = self.ub_size // self.input_dsize
        self.ub_elems = (self.ub_elems // self.block_elems) * self.block_elems
        self.tiling_dtype = "int64"
        self.tiling_align = align_value(self.TILING_ARG_NUM, 4)
        tiling_args = self.init_gm_tensor()
        self.tiling_gm, self.x_gm, self.size_splits_gm, self.split_dim_gm, self.outputs_gm = tiling_args

        self.block_ub, self.tiling_ub = [None, None]

        # mode 8
        self.outer_loop, self.outer_tail, self.block_num, self.num1, self.num2 = [None, None, None, None, None]

        # mode 3
        self.row_elems_offset, self.split_i_mode, self.row_num1, self.row_num2 = [None, None, None, None]

        # tiling params
        self.tiling_mode = None
        self.need_core_num = None
        self.input_elems = None
        self.shape_dim = None
        self.data_each_core = None
        self.data_last_core = None
        self.loop_num = None
        self.last_num = None
        self.one_loop_elems = None
        self.loop_num_last_core = None
        self.last_num_last_core = None
        self.one_loop_elems_last_core = None

        self.shape_after_dim, self.shape_before, self.shape_after, self.multi_move = [None, None, None, None]

        self.tail_ele = None
        self.one_core_seg = None
        self.seg_loop_num = None
        self.last_seg = None
        self.last_core_seg = None
        self.seg_loop_num_last_core = None
        self.last_seg_last_core = None

        self.size_value_split = None

    def init_gm_tensor(self):
        """
        init gm tensor

        Parameters
        ----------
        None

        Returns
        -------
        gm tensors
        """
        tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (self.TILING_ARG_NUM,),
                                             name="tiling_gm",
                                             scope=tik.scope_gm)
        x_gm = self.tik_instance.Tensor(self.input_dtype, (self.MAX_SHAPE_SIZE,), name="x", scope=tik.scope_gm)
        split_dim_gm = self.tik_instance.Tensor(self.split_dim_dtype, (1,), name="split_dim", scope=tik.scope_gm)
        size_splits_gm = None
        if self.is_split_v:
            size_splits_gm = self.tik_instance.Tensor(self.size_splits_dtype, (64,),
                                                      name="size_splits",
                                                      scope=tik.scope_gm)

        outputs_gm = []
        for i in range(self.num_split):
            tensor_name = "gm_output_{}".format(i)
            gm_tensor = self.tik_instance.Tensor(self.input_dtype, (self.MAX_SHAPE_SIZE,),
                                                 name=tensor_name,
                                                 scope=tik.scope_gm)
            outputs_gm.append(gm_tensor)

        tiling_args = [tiling_gm, x_gm, size_splits_gm, split_dim_gm, outputs_gm]
        return tiling_args

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
        self.tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.input_elems = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="input_elems")
        self.shape_dim = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="shape_dim")
        self.data_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="data_each_core")
        self.data_last_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="data_last_core")
        self.loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="loop_num")
        self.last_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="last_num")
        self.one_loop_elems = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="one_loop_elems")
        self.loop_num_last_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="loop_num_last_core")
        self.last_num_last_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="last_num_last_core")
        self.one_loop_elems_last_core = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                                 name="one_loop_elems_last_core")
        self.shape_after_dim = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="shape_after_dim")
        self.shape_before = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="shape_before")
        self.shape_after = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="shape_after")
        self.multi_move = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="multi_move")

        self.tail_ele = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tail_ele")
        self.one_core_seg = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="one_core_seg")
        self.seg_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="seg_loop_num")
        self.last_seg = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="last_seg")
        self.last_core_seg = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="last_core_seg")
        self.seg_loop_num_last_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="seg_loop_num_last_core")
        self.last_seg_last_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="last_seg_last_core")

        self.size_value_split = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="size_value_split")

        self.tiling_mode.set_as(self.tiling_ub[0])
        self.need_core_num.set_as(self.tiling_ub[1])
        self.input_elems.set_as(self.tiling_ub[2])
        self.shape_dim.set_as(self.tiling_ub[3])
        self.data_each_core.set_as(self.tiling_ub[4])
        self.data_last_core.set_as(self.tiling_ub[5])
        self.loop_num.set_as(self.tiling_ub[6])
        self.last_num.set_as(self.tiling_ub[7])
        self.one_loop_elems.set_as(self.tiling_ub[8])
        self.loop_num_last_core.set_as(self.tiling_ub[9])
        self.last_num_last_core.set_as(self.tiling_ub[10])
        self.one_loop_elems_last_core.set_as(self.tiling_ub[11])

        self.shape_after_dim.set_as(self.tiling_ub[12])
        self.shape_before.set_as(self.tiling_ub[13])
        self.shape_after.set_as(self.tiling_ub[14])
        self.multi_move.set_as(self.tiling_ub[15])

        self.tail_ele.set_as(self.tiling_ub[16])
        self.one_core_seg.set_as(self.tiling_ub[17])
        self.seg_loop_num.set_as(self.tiling_ub[18])
        self.last_seg.set_as(self.tiling_ub[19])
        self.last_core_seg.set_as(self.tiling_ub[20])
        self.seg_loop_num_last_core.set_as(self.tiling_ub[21])
        self.last_seg_last_core.set_as(self.tiling_ub[22])

        self.size_value_split.set_as(self.tiling_ub[23])

    def compute_move_copy(self, core_id, loop_num, one_loop_elems, last_num):
        """
        move copy

        Parameters
        ----------
        core_id: core index
        loop_num: loop num
        one_loop_elems: element num of one loop
        last_num: last num

        Returns
        -------
        None
        """
        data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems,), name="data_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, loop_num) as loop_index:
            self.move_copy_process(core_id, data_ub, loop_index, one_loop_elems, one_loop_elems)
        with self.tik_instance.if_scope(last_num > 0):
            self.move_copy_process(core_id, data_ub, loop_num, last_num, one_loop_elems)

    def move_copy_process(self, core_id, data_ub, loop_index, elem_num, one_loop_elems):
        """
        move process

        Parameters
        ----------
        core_id: core index
        data_ub: data_ub
        loop_index: loop index
        elem_num: element number
        one_loop_elems: element num of one loop

        Returns
        -------
        None
        """
        burst_len = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="burst_len")
        burst_len.set_as(ceil_value(elem_num, self.block_elems))
        move_offset = core_id * self.data_each_core + loop_index * one_loop_elems
        with self.tik_instance.if_scope(tik.all(elem_num % self.block_elems != 0, burst_len > 1)):
            # move last 1 block
            align_offset = elem_num - self.block_elems
            self.tik_instance.data_move(self.block_ub, self.x_gm[move_offset + align_offset], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.outputs_gm[0][move_offset + align_offset], self.block_ub, 0, 1, 1, 0, 0)
            burst_len.set_as(elem_num // self.block_elems)
        self.tik_instance.data_move(data_ub, self.x_gm[move_offset], 0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.outputs_gm[0][move_offset], data_ub, 0, 1, burst_len, 0, 0)

    def compute_mode_2(self, core_id):
        """
        mode 2 compute: (shape_dim, shape_after_dim), split shape_dim
        """
        tik_instance = self.tik_instance
        size_splits_ub = self.get_size_splits_ub()

        # process part of output for all outputs
        split_size_i = tik_instance.Scalar(dtype=self.size_splits_dtype, name="split_size_i")
        i_need_core_num = tik_instance.Scalar(dtype=self.tiling_dtype, name="i_need_core_num")
        each_core_elems = tik_instance.Scalar(dtype=self.tiling_dtype, name="each_core_elems")
        last_core_elems = tik_instance.Scalar(dtype=self.tiling_dtype, name="last_core_elems")
        x_elems_offset_base = tik_instance.Scalar(dtype=self.tiling_dtype, name="x_elems_offset_base", init_value=0)

        for out_i in range(0, self.num_split):
            split_size_i.set_as(size_splits_ub[out_i])
            out_i_elems = split_size_i * self.shape_after_dim

            # tiling
            with tik_instance.if_scope(out_i_elems <= self.block_elems * 8):
                i_need_core_num.set_as(1)
                each_core_elems.set_as(0)
                last_core_elems.set_as(out_i_elems)
            with tik_instance.else_scope():
                each_core_elems_temp = ceil_value(out_i_elems, self.need_core_num)
                each_core_elems.set_as(align_value(each_core_elems_temp, self.block_elems))

                need_core_temp = ceil_value(out_i_elems, each_core_elems)
                last_core_elems_temp = out_i_elems - (need_core_temp - 1) * each_core_elems
                with tik_instance.if_scope(
                        tik.all(last_core_elems_temp > 0, last_core_elems_temp < self.block_elems, need_core_temp > 1)):
                    i_need_core_num.set_as(need_core_temp - 1)
                    last_core_elems.set_as(each_core_elems + last_core_elems_temp)
                with tik_instance.else_scope():
                    i_need_core_num.set_as(need_core_temp)
                    last_core_elems.set_as(last_core_elems_temp)

            with tik_instance.if_scope(core_id < i_need_core_num - 1):
                self.compute_mode_2_one_core(out_i, core_id, each_core_elems, x_elems_offset_base)
            with tik_instance.if_scope(core_id == i_need_core_num - 1):
                self.compute_mode_2_last_core(out_i, core_id, each_core_elems, last_core_elems, x_elems_offset_base)

            # update x_elems_offset_base
            x_elems_offset_base.set_as(x_elems_offset_base + out_i_elems)

    def compute_mode_2_last_core(self, out_i, core_id, each_core_elems, last_core_elems, x_elems_offset_base):
        """
        compute_mode_2_last_core
        """
        with self.tik_instance.new_stmt_scope():
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems,), name="data_ub", scope=tik.scope_ubuf)
            loops = last_core_elems // self.ub_elems
            tail = last_core_elems % self.ub_elems

            burst_len = self.ub_elems // self.block_elems
            with self.tik_instance.for_range(0, loops) as loop_i:
                # process self.ub_elems
                dst_offset = each_core_elems * core_id + self.ub_elems * loop_i
                x_offset = x_elems_offset_base + dst_offset
                self.tik_instance.data_move(data_ub, self.x_gm[x_offset], 0, 1, burst_len, 0, 0)
                self.tik_instance.data_move(self.outputs_gm[out_i][dst_offset], data_ub, 0, 1, burst_len, 0, 0)

            with self.tik_instance.if_scope(tail > 0):
                # process tail of last core
                dst_offset = each_core_elems * core_id + self.ub_elems * loops
                x_offset = x_elems_offset_base + dst_offset
                with self.tik_instance.if_scope(tik.all(tail < self.block_elems, loops > 0)):
                    align_elems = self.block_elems - tail
                    self.tik_instance.data_move(data_ub, self.x_gm[x_offset - align_elems], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(self.outputs_gm[out_i][dst_offset - align_elems], data_ub, 0, 1, 1, 0,
                                                0)
                with self.tik_instance.else_scope():
                    burst_len = ceil_value(tail, self.block_elems)
                    self.tik_instance.data_move(data_ub, self.x_gm[x_offset], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.outputs_gm[out_i][dst_offset], data_ub, 0, 1, burst_len, 0, 0)

    def compute_mode_2_one_core(self, out_i, core_id, each_core_elems, x_elems_offset_base):
        """
        compute_mode_2_one_core
        """
        with self.tik_instance.new_stmt_scope():
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems,), name="data_ub", scope=tik.scope_ubuf)
            loops = each_core_elems // self.ub_elems
            tail = each_core_elems % self.ub_elems

            burst_len = self.ub_elems // self.block_elems
            with self.tik_instance.for_range(0, loops) as loop_i:
                # process self.ub_elems
                dst_offset = each_core_elems * core_id + self.ub_elems * loop_i
                x_offset = x_elems_offset_base + dst_offset
                self.tik_instance.data_move(data_ub, self.x_gm[x_offset], 0, 1, burst_len, 0, 0)
                self.tik_instance.data_move(self.outputs_gm[out_i][dst_offset], data_ub, 0, 1, burst_len, 0, 0)

            with self.tik_instance.if_scope(tail > 0):
                # process tail
                burst_len_tail = tail // self.block_elems
                dst_offset = each_core_elems * core_id + self.ub_elems * loops
                x_offset = x_elems_offset_base + dst_offset
                self.tik_instance.data_move(data_ub, self.x_gm[x_offset], 0, 1, burst_len_tail, 0, 0)
                self.tik_instance.data_move(self.outputs_gm[out_i][dst_offset], data_ub, 0, 1, burst_len_tail, 0, 0)

    def update_size_splits(self, size_splits_ub):
        """
        update data if -1 is in size_splits
        """
        size_splits_sum = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="size_splits_sum", init_value=0)
        size_temp = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="size_temp")
        index = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="index", init_value=-1)
        with self.tik_instance.for_range(0, self.num_split) as i:
            size_temp.set_as(size_splits_ub[i])
            with self.tik_instance.if_scope(size_temp != -1):
                size_splits_sum.set_as(size_splits_sum + size_temp)
            with self.tik_instance.else_scope():
                index.set_as(i)

        if self.x.get("format") == "FRACTAL_NZ":
            factor = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="factor", init_value=16)
            temp_value = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="temp_value", init_value=0)
            with self.tik_instance.for_range(0, self.num_split) as i:
                temp_value.set_as(size_splits_ub[i])
                size_splits_ub[i].set_as((temp_value + 15) / factor)

        with self.tik_instance.if_scope(index != -1):
            size_splits_ub[index].set_as(self.shape_dim - size_splits_sum)

    def sub_mode2_internal(self, out_i, data_ub, rows, row_elems, row_count, rows_offset):
        """
        sub mode 2 compute internal
        """
        tik_instance = self.tik_instance
        inner_loop = rows // row_count
        inner_tail = rows % row_count
        src_stride = (self.shape_after - row_elems) // self.block_elems
        with tik_instance.for_range(0, inner_loop) as inner_i:
            x_gm_offset = rows_offset * self.shape_after + inner_i * row_count * self.shape_after + \
                          self.row_elems_offset
            tik_instance.data_move(data_ub, self.x_gm[x_gm_offset], 0, row_count, row_elems // self.block_elems,
                                   src_stride, 0)
            # move result to output gm
            tik_instance.data_move(self.outputs_gm[out_i][rows_offset * row_elems + row_count * row_elems * inner_i],
                                   data_ub, 0, 1, row_count * row_elems // self.block_elems, 0, 0)

        with tik_instance.if_scope(inner_tail > 0):
            x_gm_offset = rows_offset * self.shape_after + inner_loop * row_count * self.shape_after + \
                          self.row_elems_offset
            tik_instance.data_move(data_ub, self.x_gm[x_gm_offset], 0, inner_tail, row_elems // self.block_elems,
                                   src_stride, 0)
            # move result to output gm
            tik_instance.data_move(self.outputs_gm[out_i][rows_offset * row_elems + row_count * row_elems * inner_loop],
                                   data_ub, 0, 1, inner_tail * row_elems // self.block_elems, 0, 0)

    def compute_sub_mode2(self, out_i, core_id, row_elems, outer, tail):
        """
        sub mode 2 compute of mode 3
        """
        with self.tik_instance.new_stmt_scope():
            row_count = self.tik_instance.Scalar(dtype="int32", name="row_count")
            row_count.set_as(self.ub_elems // row_elems)
            # 4095 is the max nburst
            with self.tik_instance.if_scope(row_count > 4095):
                row_count.set_as(4095)
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems,), name="data_ub", scope=tik.scope_ubuf)

            with self.tik_instance.if_scope(core_id < tail):
                rows_offset = core_id * self.row_num1
                self.sub_mode2_internal(out_i, data_ub, self.row_num1, row_elems, row_count, rows_offset)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(outer > 0):
                    rows_offset = tail * self.row_num1 + (core_id - tail) * self.row_num2
                    self.sub_mode2_internal(out_i, data_ub, self.row_num2, row_elems, row_count, rows_offset)

    def sub_mode3_internal(self, out_i, data_ub, rows, row_elems, row_count, rows_offset):
        """
        sub mode 3 compute internal
        """
        tik_instance = self.tik_instance
        inner_loop = rows // row_count
        inner_tail = rows % row_count
        with tik_instance.for_range(0, inner_loop) as inner_i:
            x_gm_offset = rows_offset * self.shape_after + inner_i * row_count * self.shape_after + \
                          self.row_elems_offset
            with tik_instance.for_range(0, row_count) as count_i:
                tik_instance.data_move(data_ub[row_elems * count_i],
                                       self.x_gm[x_gm_offset + self.shape_after * count_i], 0, 1,
                                       row_elems // self.block_elems, 0, 0)
            # move result to output gm
            tik_instance.data_move(self.outputs_gm[out_i][rows_offset * row_elems + row_count * row_elems * inner_i],
                                   data_ub, 0, 1, row_count * row_elems // self.block_elems, 0, 0)

        with tik_instance.if_scope(inner_tail > 0):
            x_gm_offset = rows_offset * self.shape_after + inner_loop * row_count * self.shape_after + \
                          self.row_elems_offset
            with tik_instance.for_range(0, inner_tail) as count_i:
                tik_instance.data_move(data_ub[row_elems * count_i],
                                       self.x_gm[x_gm_offset + self.shape_after * count_i], 0, 1,
                                       row_elems // self.block_elems, 0, 0)
            # move result to output gm
            tik_instance.data_move(self.outputs_gm[out_i][rows_offset * row_elems + row_count * row_elems * inner_loop],
                                   data_ub, 0, 1, inner_tail * row_elems // self.block_elems, 0, 0)

    def compute_sub_mode3(self, out_i, core_id, row_elems, outer, tail):
        """
        sub mode 3 compute of mode 3
        """
        with self.tik_instance.new_stmt_scope():
            row_count = self.ub_elems // row_elems
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems,), name="data_ub", scope=tik.scope_ubuf)

            with self.tik_instance.if_scope(core_id < tail):
                rows_offset = core_id * self.row_num1
                self.sub_mode3_internal(out_i, data_ub, self.row_num1, row_elems, row_count, rows_offset)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(outer > 0):
                    rows_offset = tail * self.row_num1 + (core_id - tail) * self.row_num2
                    self.sub_mode3_internal(out_i, data_ub, self.row_num2, row_elems, row_count, rows_offset)

    def sub_mode4_internal(self, out_i, data_ub, rows, row_elems, rows_offset):
        """
        sub mode 4 compute internal
        """
        tik_instance = self.tik_instance
        burst_len = ceil_value(row_elems, self.block_elems)
        align_offset = row_elems - self.block_elems

        with tik_instance.for_range(0, rows - 1) as row_i:
            rows_i_offset = rows_offset + row_i
            x_gm_offset = rows_i_offset * self.shape_after + self.row_elems_offset
            tik_instance.data_move(data_ub, self.x_gm[x_gm_offset], 0, 1, burst_len, 0, 0)
            # move row_elems to output gm
            tik_instance.data_move(self.outputs_gm[out_i][rows_i_offset * row_elems], data_ub, 0, 1, burst_len, 0, 0)

        # process last row
        rows_i_offset = rows_offset + (rows - 1)
        x_gm_offset = rows_i_offset * self.shape_after + self.row_elems_offset
        tik_instance.data_move(data_ub, self.x_gm[x_gm_offset], 0, 1, burst_len - 1, 0, 0)
        tik_instance.data_move(self.block_ub, self.x_gm[x_gm_offset + align_offset], 0, 1, 1, 0, 0)
        # move result to output gm
        tik_instance.data_move(self.outputs_gm[out_i][rows_i_offset * row_elems], data_ub, 0, 1, burst_len - 1, 0, 0)
        tik_instance.data_move(self.outputs_gm[out_i][rows_i_offset * row_elems + align_offset], self.block_ub, 0, 1, 1,
                               0, 0)

    def compute_sub_mode4(self, out_i, core_id, row_elems, outer, tail):
        """
        sub mode 4 compute of mode 3
        """
        with self.tik_instance.new_stmt_scope():
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems,), name="data_ub", scope=tik.scope_ubuf)

            with self.tik_instance.if_scope(core_id < tail):
                rows_offset = core_id * self.row_num1
                self.sub_mode4_internal(out_i, data_ub, self.row_num1, row_elems, rows_offset)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(outer > 0):
                    rows_offset = tail * self.row_num1 + (core_id - tail) * self.row_num2
                    self.sub_mode4_internal(out_i, data_ub, self.row_num2, row_elems, rows_offset)

    def sub_mode5_internal(self, out_i, data_ub, rows, row_elems, row_count, rows_offset):
        """
        sub mode 5 compute internal
        """
        tik_instance = self.tik_instance
        inner_loop = rows // row_count
        inner_tail = rows % row_count
        row_count_tail = tik_instance.Scalar(dtype=self.tiling_dtype, name="row_count_tail")
        inner_loop_new = tik_instance.Scalar(dtype=self.tiling_dtype, name="inner_loop_new")
        row_count_tail.set_as(inner_tail)
        inner_loop_new.set_as(inner_loop)
        with tik_instance.if_scope(tik.all(inner_tail > 0, inner_loop > 0, inner_tail * row_elems < self.block_elems)):
            row_count_tail.set_as(row_count + inner_tail)
            inner_loop_new.set_as(inner_loop - 1)

        with tik_instance.for_range(0, inner_loop_new) as inner_i:
            x_gm_offset = (rows_offset + inner_i * row_count) * self.shape_after + self.row_elems_offset
            with tik_instance.for_range(0, row_count) as row_i:
                tik_instance.data_move(self.block_ub, self.x_gm[x_gm_offset + self.shape_after * row_i], 0, 1, 1, 0, 0)
                with tik_instance.for_range(0, row_elems) as elem_i:
                    data_ub[row_i * row_elems + elem_i].set_as(self.block_ub[elem_i])
            # move result to output gm
            tik_instance.data_move(self.outputs_gm[out_i][(rows_offset + inner_i * row_count) * row_elems], data_ub, 0,
                                   1, row_count * row_elems // self.block_elems, 0, 0)

        with tik_instance.if_scope(row_count_tail > 0):
            x_gm_offset = (rows_offset + inner_loop_new * row_count) * self.shape_after + self.row_elems_offset
            with tik_instance.for_range(0, row_count_tail) as row_i:
                tik_instance.data_move(self.block_ub, self.x_gm[x_gm_offset + self.shape_after * row_i], 0, 1, 1, 0, 0)
                with tik_instance.for_range(0, row_elems) as elem_i:
                    data_ub[row_i * row_elems + elem_i].set_as(self.block_ub[elem_i])

            # move result to output gm
            burst_len = ceil_value(row_count_tail * row_elems, self.block_elems)
            tail_elems = row_count_tail * row_elems % self.block_elems
            with tik_instance.if_scope(tail_elems > 0):
                with tik_instance.if_scope(burst_len > 1):
                    align_offset = row_count_tail * row_elems - self.block_elems
                    with tik_instance.for_range(0, self.block_elems) as index_i:
                        self.block_ub[index_i].set_as(data_ub[align_offset + index_i])
                    output_offset = (rows_offset + row_count * inner_loop_new) * row_elems
                    tik_instance.data_move(self.outputs_gm[out_i][output_offset], data_ub, 0, 1, burst_len - 1, 0, 0)
                    tik_instance.data_move(self.outputs_gm[out_i][output_offset + align_offset], self.block_ub, 0, 1, 1,
                                           0, 0)
                with tik_instance.else_scope():
                    # only one core required, and all data of output_i is smaller than 32B
                    tik_instance.data_move(self.outputs_gm[out_i][0], data_ub, 0, 1, 1, 0, 0)
            with tik_instance.else_scope():
                tik_instance.data_move(self.outputs_gm[out_i][(rows_offset + row_count * inner_loop_new) * row_elems],
                                       data_ub, 0, 1, burst_len, 0, 0)

    def compute_sub_mode5(self, out_i, core_id, row_elems, outer, tail):
        """
        sub mode 5 compute of mode 3
        """
        with self.tik_instance.new_stmt_scope():
            row_count = (self.ub_elems // row_elems) // self.block_elems * self.block_elems
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems + 2 * self.block_elems,),
                                               name="data_ub",
                                               scope=tik.scope_ubuf)

            with self.tik_instance.if_scope(core_id < tail):
                rows_offset = core_id * self.row_num1
                self.sub_mode5_internal(out_i, data_ub, self.row_num1, row_elems, row_count, rows_offset)

            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(outer > 0):
                    rows_offset = tail * self.row_num1 + (core_id - tail) * self.row_num2
                    self.sub_mode5_internal(out_i, data_ub, self.row_num2, row_elems, row_count, rows_offset)

    def sub_mode1_internal(self, out_i, rows, row_elems, data_ub, one_row_loop, one_row_tail, rows_offset):
        """
        sub mode 1 compute internal
        """
        tik_instance = self.tik_instance
        burst_len = self.ub_elems // self.block_elems

        burst_len_tail = ceil_value(one_row_tail, self.block_elems)
        with tik_instance.for_range(0, rows - 1) as row_i:
            rows_i_offset = rows_offset + row_i
            x_gm_offset = rows_i_offset * self.shape_after + self.row_elems_offset
            with tik_instance.for_range(0, one_row_loop) as loop_i:
                tik_instance.data_move(data_ub, self.x_gm[x_gm_offset + loop_i * self.ub_elems], 0, 1, burst_len, 0, 0)
                # move row_elems to output gm
                tik_instance.data_move(self.outputs_gm[out_i][rows_i_offset * row_elems + loop_i * self.ub_elems],
                                       data_ub, 0, 1, burst_len, 0, 0)
            # process one_row_tail
            tik_instance.data_move(data_ub, self.x_gm[x_gm_offset + one_row_loop * self.ub_elems], 0, 1, burst_len_tail,
                                   0, 0)
            tik_instance.data_move(self.outputs_gm[out_i][rows_i_offset * row_elems + one_row_loop * self.ub_elems],
                                   data_ub, 0, 1, burst_len_tail, 0, 0)

        # process last row
        rows_i_offset = rows_offset + (rows - 1)
        x_gm_offset = rows_i_offset * self.shape_after + self.row_elems_offset
        with tik_instance.for_range(0, one_row_loop) as loop_i:
            tik_instance.data_move(data_ub, self.x_gm[x_gm_offset + loop_i * self.ub_elems], 0, 1, burst_len, 0, 0)
            tik_instance.data_move(self.outputs_gm[out_i][rows_i_offset * row_elems + loop_i * self.ub_elems], data_ub,
                                   0, 1, burst_len, 0, 0)
        # process one_row_tail of last row
        with tik_instance.if_scope(one_row_tail > 0):
            burst_len_last_tail = tik_instance.Scalar(dtype=self.tiling_dtype, name="burst_len_last_tail")
            burst_len_last_tail.set_as(ceil_value(one_row_tail, self.block_elems))
            with tik_instance.if_scope(one_row_tail % self.block_elems > 0):
                align_offset = one_row_tail - self.block_elems
                last_tail_offset = one_row_loop * self.ub_elems + align_offset
                tik_instance.data_move(self.block_ub, self.x_gm[x_gm_offset + last_tail_offset], 0, 1, 1, 0, 0)
                tik_instance.data_move(self.outputs_gm[out_i][rows_i_offset * row_elems + last_tail_offset],
                                       self.block_ub, 0, 1, 1, 0, 0)
                burst_len_last_tail.set_as(one_row_tail // self.block_elems)
            tik_instance.data_move(data_ub, self.x_gm[x_gm_offset + one_row_loop * self.ub_elems], 0, 1,
                                   burst_len_last_tail, 0, 0)
            tik_instance.data_move(self.outputs_gm[out_i][rows_i_offset * row_elems + one_row_loop * self.ub_elems],
                                   data_ub, 0, 1, burst_len_last_tail, 0, 0)

    def compute_sub_mode1(self, out_i, core_id, row_elems, outer, tail):
        """
        sub mode 1 compute of mode 3
        """
        with self.tik_instance.new_stmt_scope():
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems + self.block_elems,),
                                               name="data_ub",
                                               scope=tik.scope_ubuf)

            one_row_loop = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="one_row_loop")
            one_row_loop.set_as(row_elems // self.ub_elems)
            one_row_tail = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="one_row_tail")
            one_row_tail.set_as(row_elems % self.ub_elems)
            with self.tik_instance.if_scope(tik.all(one_row_tail > 0, one_row_tail < self.block_elems)):
                one_row_loop.set_as(row_elems // self.ub_elems - 1)
                one_row_tail.set_as(self.ub_elems + row_elems % self.ub_elems)

            with self.tik_instance.if_scope(core_id < tail):
                rows_offset = core_id * self.row_num1
                self.sub_mode1_internal(out_i, self.row_num1, row_elems, data_ub, one_row_loop, one_row_tail,
                                        rows_offset)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(outer > 0):
                    rows_offset = tail * self.row_num1 + (core_id - tail) * self.row_num2
                    self.sub_mode1_internal(out_i, self.row_num2, row_elems, data_ub, one_row_loop, one_row_tail,
                                            rows_offset)

    def determine_sub_mode(self, row_elems):
        """
        determine the sub_mode
        """
        with self.tik_instance.if_scope(row_elems > self.ub_elems):
            self.split_i_mode.set_as(self.SUB_MODE_1)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(row_elems % self.block_elems == 0):
                with self.tik_instance.if_scope(self.multi_move == 1):
                    self.split_i_mode.set_as(self.SUB_MODE_2)
                with self.tik_instance.else_scope():
                    self.split_i_mode.set_as(self.SUB_MODE_3)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(row_elems >= self.block_elems):
                    self.split_i_mode.set_as(self.SUB_MODE_4)
                with self.tik_instance.else_scope():
                    self.split_i_mode.set_as(self.SUB_MODE_5)

    def compute_mode_3(self, core_id):
        """
        mode 3 compute
        """
        tik_instance = self.tik_instance
        size_splits_ub = self.get_size_splits_ub()

        # process part of output for all outputs
        split_size_i = tik_instance.Scalar(dtype=self.size_splits_dtype, name="split_size_i")
        core_num_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="core_num_i")
        self.split_i_mode = tik_instance.Scalar(dtype=self.tiling_dtype, name="split_i_mode")
        self.row_num1 = tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num1")
        self.row_num2 = tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num2")
        self.row_elems_offset = tik_instance.Scalar(dtype=self.tiling_dtype, name="row_elems_offset", init_value=0)

        # must use python for loop
        for out_i in range(0, self.num_split):
            split_size_i.set_as(size_splits_ub[out_i])
            row_elems = split_size_i * self.shape_after_dim

            # calculate the number of cores required for output_i
            outer = self.shape_before // self.need_core_num
            tail = self.shape_before % self.need_core_num
            core_num_i.set_as(tail)
            self.row_num1.set_as(1)
            self.row_num2.set_as(1)
            with tik_instance.if_scope(outer > 0):
                core_num_i.set_as(self.need_core_num)
                with tik_instance.if_scope(tail > 0):
                    self.row_num1.set_as(outer + 1)
                    self.row_num2.set_as(outer)
                with tik_instance.else_scope():
                    self.row_num1.set_as(outer)
                    self.row_num2.set_as(outer)
            with tik_instance.if_scope(self.row_num2 * row_elems < self.block_elems):
                # output i only need one core
                core_num_i.set_as(1)
                self.row_num1.set_as(self.shape_before)
                self.row_num2.set_as(self.shape_before)

            # determine the sub-mode
            self.determine_sub_mode(row_elems)

            self.compute_sub_mode(out_i, core_id, core_num_i, row_elems, outer, tail)

            # update x offset for next output
            self.row_elems_offset.set_as(self.row_elems_offset + row_elems)

    def compute_sub_mode(self, out_i, core_id, core_num_i, row_elems, outer, tail):
        """
        compute sub mode of mode 3
        """
        tik_instance = self.tik_instance
        with tik_instance.if_scope(core_id < core_num_i):
            with tik_instance.if_scope(self.split_i_mode == self.SUB_MODE_2):
                self.compute_sub_mode2(out_i, core_id, row_elems, outer, tail)
            with tik_instance.if_scope(self.split_i_mode == self.SUB_MODE_3):
                self.compute_sub_mode3(out_i, core_id, row_elems, outer, tail)
            with tik_instance.if_scope(self.split_i_mode == self.SUB_MODE_4):
                self.compute_sub_mode4(out_i, core_id, row_elems, outer, tail)
            with tik_instance.if_scope(self.split_i_mode == self.SUB_MODE_5):
                self.compute_sub_mode5(out_i, core_id, row_elems, outer, tail)
            with tik_instance.if_scope(self.split_i_mode == self.SUB_MODE_1):
                self.compute_sub_mode1(out_i, core_id, row_elems, outer, tail)

    def split_last_dim_vnc_compute_internal(self, src_offset, dst_offset, seg, max_seg):
        """
        split_last_dim_vnc_compute internal
        """
        with self.tik_instance.new_stmt_scope():
            ub_x = self.tik_instance.Tensor(self.input_dtype, (max_seg * self.TRANSPOSE_SIZE * self.num_split,),
                                            scope=tik.scope_ubuf,
                                            name="ub_x")
            ub_y = self.tik_instance.Tensor(self.input_dtype, (max_seg * self.TRANSPOSE_SIZE * self.num_split,),
                                            scope=tik.scope_ubuf,
                                            name="ub_y")
            ub_m = self.tik_instance.Tensor(self.input_dtype, (max_seg * self.TRANSPOSE_SIZE,),
                                            scope=tik.scope_ubuf,
                                            name="ub_m")
            ub_n = self.tik_instance.Tensor(self.input_dtype, (max_seg * self.TRANSPOSE_SIZE,),
                                            scope=tik.scope_ubuf,
                                            name="ub_n")

            # copy gm to ub
            self.tik_instance.data_move(ub_x, self.x_gm[src_offset], 0, 1,
                                        seg * self.TRANSPOSE_SIZE * self.num_split // self.block_elems, 0, 0)

            # vadds and vtranspose
            with self.tik_instance.for_range(0, self.num_split) as num_idx:
                src_offset_ub = num_idx * self.block_elems
                dst_offset_ub = num_idx * self.TRANSPOSE_SIZE

                self.tik_instance.vadds(128, ub_m, ub_x[src_offset_ub], 0, 2 * seg, 1, self.num_split, 8,
                                        self.num_split * 8)
                with self.tik_instance.for_range(0, seg) as trans_idx:
                    src_offset_trans = trans_idx * self.TRANSPOSE_SIZE
                    dst_offset_trans = dst_offset_ub + self.num_split * trans_idx * self.TRANSPOSE_SIZE
                    self.tik_instance.vtranspose(ub_y[dst_offset_trans], ub_m[src_offset_trans])

            for out_i in range(0, self.num_split):
                src_offset_ub = out_i * self.block_elems
                self.tik_instance.vadds(128, ub_m, ub_y[src_offset_ub], 0, 2 * seg, 1, self.num_split, 8,
                                        self.num_split * 8)
                with self.tik_instance.for_range(0, seg) as trans_idx:
                    src_offset_trans = trans_idx * self.TRANSPOSE_SIZE
                    dst_offset_trans = trans_idx * self.TRANSPOSE_SIZE
                    self.tik_instance.vtranspose(ub_n[dst_offset_trans], ub_m[src_offset_trans])
                # copy ub to gm
                self.tik_instance.data_move(self.outputs_gm[out_i][dst_offset], ub_n, 0, 1,
                                            seg * self.TRANSPOSE_SIZE // self.block_elems, 0, 0)

    def split_last_dim_vnc_compute_for_core(self, max_seg, src_offset_core, dst_offset_core):
        """
        split last dim with vtranspose
        """
        with self.tik_instance.if_scope(self.seg_loop_num > 1):
            # double buffer
            with self.tik_instance.for_range(0, self.seg_loop_num, thread_num=2) as loop_idx:
                src_offset = src_offset_core + loop_idx * max_seg * self.TRANSPOSE_SIZE * self.shape_after
                dst_offset = dst_offset_core + loop_idx * max_seg * self.TRANSPOSE_SIZE * 1
                self.split_last_dim_vnc_compute_internal(src_offset, dst_offset, max_seg, max_seg)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.seg_loop_num) as loop_idx:
                src_offset = src_offset_core + loop_idx * max_seg * self.TRANSPOSE_SIZE * self.shape_after
                dst_offset = dst_offset_core + loop_idx * max_seg * self.TRANSPOSE_SIZE * 1
                self.split_last_dim_vnc_compute_internal(src_offset, dst_offset, max_seg, max_seg)

        with self.tik_instance.if_scope(self.last_seg > 0):
            src_offset = src_offset_core + self.seg_loop_num * max_seg * self.TRANSPOSE_SIZE * self.shape_after
            dst_offset = dst_offset_core + self.seg_loop_num * max_seg * self.TRANSPOSE_SIZE * 1
            self.split_last_dim_vnc_compute_internal(src_offset, dst_offset, self.last_seg, max_seg)

    def split_last_dim_vnc_compute_last_core(self, max_seg, src_offset_core, dst_offset_core):
        """
        split last dim with vtranspose for last core
        """
        with self.tik_instance.if_scope(self.seg_loop_num_last_core > 1):
            # double buffer
            with self.tik_instance.for_range(0, self.seg_loop_num_last_core, thread_num=2) as loop_idx:
                src_offset = src_offset_core + loop_idx * max_seg * self.TRANSPOSE_SIZE * self.shape_after
                dst_offset = dst_offset_core + loop_idx * max_seg * self.TRANSPOSE_SIZE * 1
                self.split_last_dim_vnc_compute_internal(src_offset, dst_offset, max_seg, max_seg)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.seg_loop_num_last_core) as loop_idx:
                src_offset = src_offset_core + loop_idx * max_seg * self.TRANSPOSE_SIZE * self.shape_after
                dst_offset = dst_offset_core + loop_idx * max_seg * self.TRANSPOSE_SIZE * 1
                self.split_last_dim_vnc_compute_internal(src_offset, dst_offset, max_seg, max_seg)

        with self.tik_instance.if_scope(self.last_seg_last_core > 0):
            src_offset = src_offset_core + \
                         self.seg_loop_num_last_core * max_seg * self.TRANSPOSE_SIZE * self.shape_after
            dst_offset = dst_offset_core + self.seg_loop_num_last_core * max_seg * self.TRANSPOSE_SIZE * 1
            self.split_last_dim_vnc_compute_internal(src_offset, dst_offset, self.last_seg_last_core, max_seg)
        with self.tik_instance.if_scope(self.tail_ele != 0):
            src_offset = (self.shape_before - self.TRANSPOSE_SIZE) * self.shape_after
            dst_offset = (self.shape_before - self.TRANSPOSE_SIZE) * 1
            self.split_last_dim_vnc_compute_internal(src_offset, dst_offset, 1, max_seg)

    def compute_mode_4(self, core_id):
        """
        mode 4 compute
        """
        with self.tik_instance.if_scope(core_id < self.need_core_num):
            src_offset_core = core_id * self.one_core_seg * self.TRANSPOSE_SIZE * self.shape_after
            # size_splits is [1,1,1...]
            dst_offset_core = core_id * self.one_core_seg * self.TRANSPOSE_SIZE * 1

            max_seg = (self.ub_elems // 2) // (self.TRANSPOSE_SIZE * (2 * self.num_split + 2))

            with self.tik_instance.if_scope(core_id < self.need_core_num - 1):
                self.split_last_dim_vnc_compute_for_core(max_seg, src_offset_core, dst_offset_core)
            with self.tik_instance.else_scope():
                self.split_last_dim_vnc_compute_last_core(max_seg, src_offset_core, dst_offset_core)

    def compute_mode_5_internal(self, rows, gm_rows_offset, size_splits_ub):
        """
        process rows: use scalar
        """
        with self.tik_instance.new_stmt_scope():
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems // 2,),
                                               name="data_ub",
                                               scope=tik.scope_ubuf)
            result_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems // 2,),
                                                 name="result_ub",
                                                 scope=tik.scope_ubuf)

            burst_len = (rows * self.shape_after) // self.block_elems
            x_offset = gm_rows_offset * self.shape_after
            self.tik_instance.data_move(data_ub, self.x_gm[x_offset], 0, 1, burst_len, 0, 0)

            split_size_i = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="split_size_i")
            row_elems_offset = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_elem_offset", init_value=0)

            for out_i in range(0, self.num_split):
                split_size_i.set_as(size_splits_ub[out_i])
                row_elems = split_size_i * self.shape_after_dim
                gm_offset = gm_rows_offset * row_elems
                burst_len_out = (rows * row_elems) // self.block_elems

                with self.tik_instance.for_range(0, rows) as row_i:
                    rows_offset = row_i * self.shape_after
                    result_offset = row_i * row_elems
                    with self.tik_instance.for_range(0, row_elems) as elem_i:
                        result_ub[result_offset + elem_i].set_as(data_ub[rows_offset + row_elems_offset + elem_i])

                # move result to gm
                self.tik_instance.data_move(self.outputs_gm[out_i][gm_offset], result_ub, 0, 1, burst_len_out, 0, 0)

                # update x offset for next output
                row_elems_offset.set_as(row_elems_offset + row_elems)

    def compute_mode_5_last_internal(self, rows, gm_rows_offset, size_splits_ub):
        """
        process rows for last core: use scalar
        """
        with self.tik_instance.new_stmt_scope():
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems // 2,),
                                               name="data_ub",
                                               scope=tik.scope_ubuf)
            result_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems // 2,),
                                                 name="result_ub",
                                                 scope=tik.scope_ubuf)

            burst_len = ceil_value(rows * self.shape_after, self.block_elems)
            x_offset = gm_rows_offset * self.shape_after
            self.tik_instance.data_move(data_ub, self.x_gm[x_offset], 0, 1, burst_len, 0, 0)

            split_size_i = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="split_size_i")
            row_elems_offset = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_elem_offset", init_value=0)

            for out_i in range(0, self.num_split):
                split_size_i.set_as(size_splits_ub[out_i])
                row_elems = split_size_i * self.shape_after_dim
                gm_offset = gm_rows_offset * row_elems
                burst_len_out = ceil_value(rows * row_elems, self.block_elems)
                tail = rows * row_elems % self.block_elems

                with self.tik_instance.for_range(0, rows) as row_i:
                    rows_offset = row_i * self.shape_after
                    result_offset = row_i * row_elems
                    with self.tik_instance.for_range(0, row_elems) as elem_i:
                        result_ub[result_offset + elem_i].set_as(data_ub[rows_offset + row_elems_offset + elem_i])

                # move result to gm
                with self.tik_instance.if_scope(tail > 0):
                    with self.tik_instance.if_scope(burst_len_out > 1):
                        align_offset = rows * row_elems - self.block_elems
                        with self.tik_instance.for_range(0, self.block_elems) as index_i:
                            self.block_ub[index_i].set_as(result_ub[align_offset + index_i])
                        self.tik_instance.data_move(self.outputs_gm[out_i][gm_offset], result_ub, 0, 1,
                                                    burst_len_out - 1, 0, 0)
                        self.tik_instance.data_move(self.outputs_gm[out_i][gm_offset + align_offset], self.block_ub, 0,
                                                    1, 1, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(self.outputs_gm[out_i][gm_offset], result_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.outputs_gm[out_i][gm_offset], result_ub, 0, 1, burst_len_out, 0, 0)

                # update x offset for next output
                row_elems_offset.set_as(row_elems_offset + row_elems)

    def compute_mode_5(self, core_id):
        """
        mode 5 compute
        """
        size_splits_ub = self.get_size_splits_ub()

        with self.tik_instance.if_scope(core_id < self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.loop_num) as loop_index:
                # process self.one_loop_elems rows for every output
                gm_rows_offset = self.one_loop_elems * loop_index + core_id * self.data_each_core
                self.compute_mode_5_internal(self.one_loop_elems, gm_rows_offset, size_splits_ub)

            with self.tik_instance.if_scope(self.last_num > 0):
                # process self.last_num rows for every output
                gm_rows_offset = self.one_loop_elems * self.loop_num + core_id * self.data_each_core
                self.compute_mode_5_internal(self.last_num, gm_rows_offset, size_splits_ub)

        with self.tik_instance.if_scope(core_id == self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.loop_num_last_core) as loop_index:
                # process self.one_loop_elems_last_core rows for every output
                gm_rows_offset = self.one_loop_elems_last_core * loop_index + core_id * self.data_each_core
                self.compute_mode_5_internal(self.one_loop_elems_last_core, gm_rows_offset, size_splits_ub)

            with self.tik_instance.if_scope(self.last_num_last_core > 0):
                # process self.last_num_last_core rows
                gm_rows_offset = self.one_loop_elems_last_core * self.loop_num_last_core + \
                                 core_id * self.data_each_core
                self.compute_mode_5_last_internal(self.last_num_last_core, gm_rows_offset, size_splits_ub)

    def compute_mode_6(self, core_id):
        """
        mode 6 compute
        """
        size_splits_ub = self.get_size_splits_ub()

        max_rows = 8 * self.block_elems
        with self.tik_instance.if_scope(core_id < self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.loop_num) as loop_index:
                # process max_rows rows for every output
                gm_rows_offset = core_id * self.data_each_core + max_rows * loop_index
                self.compute_mode_6_internal(max_rows, gm_rows_offset, size_splits_ub)

            with self.tik_instance.if_scope(self.last_num > 0):
                # process self.last_num rows for every output
                gm_rows_offset = core_id * self.data_each_core + max_rows * self.loop_num
                self.compute_mode_6_internal(self.last_num, gm_rows_offset, size_splits_ub)

        with self.tik_instance.if_scope(core_id == self.need_core_num - 1):
            with self.tik_instance.for_range(0, self.loop_num_last_core) as loop_index:
                # process max_rows rows for every output in last core
                gm_rows_offset = core_id * self.data_each_core + max_rows * loop_index
                self.compute_mode_6_internal(max_rows, gm_rows_offset, size_splits_ub)

            with self.tik_instance.if_scope(self.last_num_last_core > 0):
                # process self.last_num_last_core rows
                gm_rows_offset = core_id * self.data_each_core + max_rows * self.loop_num_last_core
                self.compute_mode_6_internal(self.last_num_last_core, gm_rows_offset, size_splits_ub)

    def compute_mode_6_internal(self, rows, gm_rows_offset, size_splits_ub):
        """
        mode 6 compute internal
        """
        with self.tik_instance.new_stmt_scope():
            data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems // 2,),
                                               name="data_ub",
                                               scope=tik.scope_ubuf)
            out_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems // 2,),
                                              name="out_ub",
                                              scope=tik.scope_ubuf)

            burst_len = (rows * self.shape_after) // self.block_elems
            x_offset = gm_rows_offset * self.shape_after
            self.tik_instance.data_move(data_ub, self.x_gm[x_offset], 0, 1, burst_len, 0, 0)

            split_size_i = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="split_size_i")
            row_elems_offset = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_elem_offset", init_value=0)

            for out_i in range(0, self.num_split):
                split_size_i.set_as(size_splits_ub[out_i])
                row_elems = split_size_i * self.shape_after_dim
                gm_offset = gm_rows_offset * row_elems
                with self.tik_instance.if_scope(
                        tik.any(row_elems_offset == 0,
                                tik.all(row_elems_offset % self.block_elems == 0, row_elems % self.block_elems == 0))):
                    src_stride = (self.shape_after - row_elems) // self.block_elems
                    self.tik_instance.data_move(self.outputs_gm[out_i][gm_offset], data_ub[row_elems_offset], 0, rows,
                                                row_elems // self.block_elems, src_stride, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, rows) as row_i:
                        with self.tik_instance.for_range(0, row_elems) as elem_i:
                            out_ub[row_i * row_elems + elem_i].set_as(data_ub[row_i * self.shape_after +
                                                                              row_elems_offset + elem_i])
                    self.tik_instance.data_move(self.outputs_gm[out_i][gm_offset], out_ub, 0, 1,
                                                ceil_value(rows * row_elems, self.block_elems), 0, 0)

                # update x offset for next output
                row_elems_offset.set_as(row_elems_offset + row_elems)

    def compute_mode_7(self, core_id):
        """
        mode 7 compute
        """
        size_splits_ub = self.get_size_splits_ub()

        max_rows = 16 * self.block_elems
        with self.tik_instance.if_scope(core_id < self.need_core_num - 1):
            # process max_rows rows for every output
            with self.tik_instance.if_scope(self.loop_num > 1):
                with self.tik_instance.for_range(0, self.loop_num, thread_num=2) as loop_index:
                    gm_rows_offset = core_id * self.data_each_core + max_rows * loop_index
                    self.compute_mode_7_internal(max_rows, gm_rows_offset, size_splits_ub)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.loop_num) as loop_index:
                    gm_rows_offset = core_id * self.data_each_core + max_rows * loop_index
                    self.compute_mode_7_internal(max_rows, gm_rows_offset, size_splits_ub)

        with self.tik_instance.if_scope(core_id == self.need_core_num - 1):
            # process max_rows rows for every output in last core
            with self.tik_instance.if_scope(self.loop_num_last_core > 1):
                with self.tik_instance.for_range(0, self.loop_num_last_core, thread_num=2) as loop_index:
                    gm_rows_offset = core_id * self.data_each_core + max_rows * loop_index
                    self.compute_mode_7_internal(max_rows, gm_rows_offset, size_splits_ub)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.loop_num_last_core) as loop_index:
                    gm_rows_offset = core_id * self.data_each_core + max_rows * loop_index
                    self.compute_mode_7_internal(max_rows, gm_rows_offset, size_splits_ub)

            with self.tik_instance.if_scope(self.last_num_last_core > 0):
                # process self.last_num_last_core rows: move forward to 256 rows
                gm_rows_offset = self.shape_before - max_rows
                self.compute_mode_7_internal(max_rows, gm_rows_offset, size_splits_ub)

    def compute_mode_7_internal(self, rows, gm_rows_offset, size_splits_ub):
        """
        mode 7 compute internal
        """
        with self.tik_instance.new_stmt_scope():
            shape_after_max = 128
            data_ub = self.tik_instance.Tensor(self.input_dtype, (rows * shape_after_max,),
                                               name="data_ub",
                                               scope=tik.scope_ubuf)
            out_ub = self.tik_instance.Tensor(self.input_dtype, (rows * self.block_elems,),
                                              name="out_ub",
                                              scope=tik.scope_ubuf)

            split_size_i = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="split_size_i")
            row_elems_offset = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_elem_offset", init_value=0)

            # process size_splits[0] ~ size_splits[-2]
            x_offset_base = gm_rows_offset * self.shape_after
            with self.tik_instance.for_range(0, 16) as idx:
                x_offset = x_offset_base + idx * self.shape_after
                self.tik_instance.data_move(data_ub[idx * self.block_elems], self.x_gm[x_offset], 0, rows // 16, 1,
                                            self.shape_after - 1, 15)

            for out_i in range(0, self.num_split - 1):
                split_size_i.set_as(size_splits_ub[out_i])
                row_elems = split_size_i * self.shape_after_dim
                gm_offset = gm_rows_offset * row_elems
                with self.tik_instance.for_range(0, rows) as row_i:
                    with self.tik_instance.for_range(0, row_elems) as elem_i:
                        out_ub[row_i * row_elems + elem_i].set_as(data_ub[row_i * self.block_elems + row_elems_offset +
                                                                          elem_i])
                self.tik_instance.data_move(self.outputs_gm[out_i][gm_offset], out_ub, 0, 1,
                                            rows * row_elems // self.block_elems, 0, 0)

                # update x offset for next output
                row_elems_offset.set_as(row_elems_offset + row_elems)

            # process size_splits[-1]
            x_offset_base = gm_rows_offset * self.shape_after + row_elems_offset
            split_size_i.set_as(size_splits_ub[self.num_split - 1])
            row_elems = split_size_i * self.shape_after_dim
            with self.tik_instance.for_range(0, 16) as idx:
                x_offset = x_offset_base + idx * self.shape_after
                self.tik_instance.data_move(data_ub[idx * row_elems], self.x_gm[x_offset], 0, rows // 16,
                                            row_elems // self.block_elems,
                                            self.shape_after - row_elems // self.block_elems,
                                            row_elems * 15 // self.block_elems)
            gm_offset = gm_rows_offset * row_elems
            self.tik_instance.data_move(self.outputs_gm[self.num_split - 1][gm_offset], data_ub, 0, 1,
                                        rows * row_elems // self.block_elems, 0, 0)

    def mode_8_tiling(self):
        """
        mode 8 tiling, cores are divided based on num_split
        """
        self.outer_loop = self.tik_instance.Scalar(dtype="int32", name="mode_8_outer_loop",
                                                   init_value=self.num_split // self.need_core_num)
        self.outer_tail = self.tik_instance.Scalar(dtype="int32", name="mode_8_outer_tail",
                                                   init_value=self.num_split % self.need_core_num)
        self.block_num = self.tik_instance.Scalar(dtype="int32", name="mode_8_block_num",
                                                  init_value=self.outer_tail)
        # number of outputs processed by each core
        self.num1 = self.tik_instance.Scalar(dtype="int32", name="mode_8_num1",
                                             init_value=1)
        self.num2 = self.tik_instance.Scalar(dtype="int32", name="mode_8_num2",
                                             init_value=1)
        with self.tik_instance.if_scope(self.outer_loop > 0):
            self.block_num.set_as(self.need_core_num)
            with self.tik_instance.if_scope(self.outer_tail > 0):
                self.num1.set_as(self.outer_loop + 1)
                self.num2.set_as(self.outer_loop)
            with self.tik_instance.else_scope():
                self.num1.set_as(self.outer_loop)
                self.num2.set_as(self.outer_loop)

    def process_one_output(self, data_ub, output_index, split_size, split_size_offset):
        """
        process one output
        """
        # the x_gm offset of the output[i]
        gm_offset_i = split_size_offset * self.shape_after_dim

        output_elems = split_size * self.shape_after_dim
        loop = output_elems // self.ub_elems
        tail_elems = output_elems % self.ub_elems

        burst_len = self.ub_elems // self.block_elems
        with self.tik_instance.for_range(0, loop) as loop_i:
            tmp_offset = self.ub_elems * loop_i
            self.tik_instance.data_move(data_ub, self.x_gm[gm_offset_i + tmp_offset], 0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(self.outputs_gm[output_index][tmp_offset], data_ub, 0, 1, burst_len, 0, 0)

        with self.tik_instance.if_scope(tail_elems > 0):
            tmp_offset = self.ub_elems * loop
            burst_len_tail = ceil_value(tail_elems, self.block_elems)
            self.tik_instance.data_move(data_ub, self.x_gm[gm_offset_i + tmp_offset], 0, 1, burst_len_tail, 0, 0)
            self.tik_instance.data_move(self.outputs_gm[output_index][tmp_offset], data_ub, 0, 1, burst_len_tail, 0, 0)

    def compute_mode_8_internal(self, num, size_splits_ub, output_index_offset):
        """
        mode 8 compute internal
        """
        data_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_elems,), name="data_ub", scope=tik.scope_ubuf)
        current_split_size = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="current_split_size")
        split_size_temp = self.tik_instance.Scalar(dtype=self.size_splits_dtype, name="split_size_temp")
        # calculate split_size_offset
        split_size_offset = self.tik_instance.Scalar(dtype=self.size_splits_dtype,
                                                     name="split_size_offset",
                                                     init_value=0)
        with self.tik_instance.for_range(0, output_index_offset) as i:
            split_size_temp.set_as(size_splits_ub[i])
            split_size_offset.set_as(split_size_offset + split_size_temp)

        # process num outputs
        for output_index in range(0, self.num_split):
            with self.tik_instance.if_scope(
                    tik.all(output_index >= output_index_offset, output_index < output_index_offset + num)):
                current_split_size.set_as(size_splits_ub[output_index])
                self.process_one_output(data_ub, output_index, current_split_size, split_size_offset)

                # update split_size_offset
                split_size_offset.set_as(split_size_offset + current_split_size)

    def compute_mode_8(self, core_id):
        """
        mode 8 compute
        """
        size_splits_ub = self.get_size_splits_ub()

        with self.tik_instance.if_scope(core_id < self.outer_tail):
            output_index_offset = core_id * self.num1
            self.compute_mode_8_internal(self.num1, size_splits_ub, output_index_offset)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.outer_loop > 0):
                output_index_offset = self.outer_tail * self.num1 + (core_id - self.outer_tail) * self.num2
                self.compute_mode_8_internal(self.num2, size_splits_ub, output_index_offset)

    def get_size_splits_ub(self):
        """
        get size_splits ub
        """
        data_block = self.BLOCK_BYTES // self.size_splits_dsize
        num_split_align = align_value(self.num_split, data_block)
        size_splits_ub = self.tik_instance.Tensor(self.size_splits_dtype, (num_split_align,),
                                                  name="size_splits_ub",
                                                  scope=tik.scope_ubuf)
        if self.is_split_v:
            # move size_splits data to ub
            self.tik_instance.data_move(size_splits_ub, self.size_splits_gm, 0, 1, num_split_align // data_block, 0, 0)
            self.update_size_splits(size_splits_ub)
        else:
            # size_splits dtype is int32, max of num_split is 62 in single split op
            self.tik_instance.vector_dup(num_split_align, size_splits_ub, self.size_value_split, 1, 1, 8)

        return size_splits_ub

    def split_v_compute_tiling(self):
        """
        main process of split_v

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        tik_instance = self.tik_instance

        # get run tiling data
        self.tiling_ub = tik_instance.Tensor(self.tiling_dtype, (self.tiling_align,),
                                                name="tiling_ub",
                                                scope=tik.scope_ubuf)
        tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // 4, 0, 0)
        self.get_tiling_args()

        with tik_instance.for_range(0, self.need_core_num, block_num=self.need_core_num) as block_id:
            self.block_ub = tik_instance.Tensor(self.input_dtype, (self.block_elems,),
                                                name="block_ub",
                                                scope=tik.scope_ubuf)
            # self.MODE_1
            if self.num_split == 1:
                with tik_instance.if_scope(self.tiling_mode == self.MODE_1):
                    with tik_instance.if_scope(block_id < self.need_core_num - 1):
                        with tik_instance.new_stmt_scope():
                            self.compute_move_copy(block_id, self.loop_num, self.one_loop_elems, self.last_num)
                    with tik_instance.if_scope(block_id == self.need_core_num - 1):
                        with tik_instance.new_stmt_scope():
                            self.compute_move_copy(block_id, self.loop_num_last_core, self.one_loop_elems_last_core,
                                                   self.last_num_last_core)
            else:
                with tik_instance.if_scope(self.tiling_mode == self.MODE_8):
                    self.mode_8_tiling()
                    with tik_instance.if_scope(block_id < self.block_num):
                        with tik_instance.new_stmt_scope():
                            self.compute_mode_8(block_id)

                with tik_instance.if_scope(self.tiling_mode == self.MODE_2):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_2(block_id)

                if self.input_dtype == "float16" and self.num_split <= 16:
                    with tik_instance.if_scope(self.tiling_mode == self.MODE_4):
                        with tik_instance.new_stmt_scope():
                            self.compute_mode_4(block_id)

                with tik_instance.if_scope(self.tiling_mode == self.MODE_5):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_5(block_id)

                if self.is_split_v:
                    with tik_instance.if_scope(self.tiling_mode == self.MODE_6):
                        with tik_instance.new_stmt_scope():
                            self.compute_mode_6(block_id)

                    if self.input_dtype == "float16" and self.num_split <= 16:
                        with tik_instance.if_scope(self.tiling_mode == self.MODE_7):
                            with tik_instance.new_stmt_scope():
                                self.compute_mode_7(block_id)

                with tik_instance.if_scope(self.tiling_mode == self.MODE_3):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_3(block_id)


def check_input_params(x, size_splits, split_dim, y, num_split):
    """
    check input parameters
    """
    # split_v has 3 input tensors, so 61 is the maximum of output tensors
    if num_split > 61 or num_split < 1:
        error_manager_vector.raise_err_input_value_invalid("split_v", "num_split", "61 is the maximum of num_split",
                                                           num_split)

    x_dtype = x.get("dtype").lower()
    size_splits_dtype = size_splits.get("dtype").lower()
    split_dim_dtype = split_dim.get("dtype").lower()
    output_dtype = y[0].get("dtype").lower()

    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                  "float16", "float32", "bfloat16", "bool")
    para_check.check_dtype(x_dtype, check_list, param_name="x")
    check_list = ("int32", "int64")
    para_check.check_dtype(size_splits_dtype, check_list, param_name="size_splits")
    para_check.check_dtype(split_dim_dtype, check_list, param_name="split_dim")

    if x_dtype != output_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("split_v", "x_dtype", "output_dtype", x_dtype,
                                                              output_dtype)


# 'pylint: disable=too-many-arguments
def split_v_tik(x, size_splits, split_dim, y, num_split, kernel_name):
    '''
    split_v interface for tik
    '''

    obj = SplitV(x, size_splits, split_dim, y, num_split, kernel_name)
    obj.split_v_compute_tiling()

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {
        "core_num": obj.core_num,
        "ub_elems": obj.ub_elems,
        "num_split": obj.num_split
    })
    # It is used to distinguish between Tik implementation and DSL implementation in the tilling phase
    tbe_context.get_context().add_compile_info("is_tik", True)

    tik_inst = obj.tik_instance
    tik_inst.BuildCCE(kernel_name=obj.kernel_name,
                      inputs=(obj.x_gm, obj.size_splits_gm, obj.split_dim_gm),
                      outputs=obj.outputs_gm,
                      flowtable=(obj.tiling_gm,),
                      enable_l2=True)
    return tik_inst


# 'pylint: disable=too-many-arguments
def split_v_compute(x, size_splits, axis, y, num_split, kernel_name):
    """
    Split_v compute

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    size_splits: dict
        the dict of input size_splits tensor.
        Specifies a list containing the sizes of each output tensor along the split dimension.
    split_dim: dict
        the dict of input split_dim tensor.
        An int, specifies the dimension along which to split.
    y: list or tuple
        the list of output tensor.
    num_split: int
        an integer indicating the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v".

    Returns
    -------
    res: TVM tensor
        the result of Split_v
    """
    res = tbe.split(x, axis, size_splits)

    return res


# 'pylint: disable=too-many-arguments
def split_v_dsl(x, size_splits, split_dim, y, num_split, kernel_name):
    '''
    split_v interface for dsl
    '''
    dtype_x = x.get("dtype")
    size_type = size_splits.get("dtype")
    input1 = tvm.placeholder((1,), dtype=size_type, name="input1")
    input2 = tvm.placeholder((1,), dtype=split_dim.get("dtype"), name="input2")
    tbe_context.get_context().add_compile_info("split_axis_idx", 2)
    tbe_context.get_context().add_compile_info("size_splits_idx", 1)

    extra_params = {"avg_split": False, "num_split":num_split}
    ins = classify([x, split_dim, size_splits], "split", extra_params)
    schedules, tensors = [], []
    for input_x_, axis_, size_splits_ in ins:
        with tbe.compute():
            shape_x, size_splits = shape_util.variable_shape([input_x_, size_splits_], "split")
            input_tensors = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")

            res = split_v_compute(input_tensors, size_splits, axis_, y, num_split, kernel_name)

            tensors.append([input_tensors, input1, input2, *res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name":kernel_name, "tensor_list":tensors}
    tbe.build(schedules, config)


@register_operator("SplitV")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.DYNAMIC_OUTPUT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def split_v(x, size_splits, split_dim, y, num_split, kernel_name="split_v"):
    """
    Split a tensor into num_split tensors along one dimension.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    size_splits: dict
        the dict of input size_splits tensor.
        Specifies a list containing the sizes of each output tensor along the split dimension.
        Can contain one -1 indicating that dimension is to be inferred.
    split_dim: dict
        the dict of input split_dim tensor.
        An int, specifies the dimension along which to split.
    y: list or tuple
        the list of output tensor.
    num_split: int
        an integer indicating the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v".

    Returns
    -------
    compile info
    """
    if num_split is None:
        num_split = len(y)

    check_input_params(x, size_splits, split_dim, y, num_split)
    input_format = x.get("format")
    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and input_format != "FRACTAL_NZ":
        split_v_dsl(x, size_splits, split_dim, y, num_split, kernel_name)
    else:
        split_v_tik(x, size_splits, split_dim, y, num_split, kernel_name)
