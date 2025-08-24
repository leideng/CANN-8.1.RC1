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
non_zero
"""
from copy import deepcopy
from functools import reduce as function_reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util import util_common


# 'pylint: disable=too-few-public-methods, too-many-arguments
class Constant:
    """
    The class for constant
    """
    TRANS_FP32_MODE = 2
    FP16_ALIGH_NUM = 16
    UB_MINIMUM_SIZE = 32
    MAX_REPEAT_TIME = 255
    VECTOR_BLOCK_SIZE = 256
    SIZE = 128
    TRANS_SIZE = 512
    Y_BASE_DTYPE = "int32"
    SHAPE_DTYPE = "uint32"
    UB_REPEAT_SIZE = 64
    REPEAT_STRIDE = 8
    BLOCK_STRIDE = 1
    TILING_ARG_NUM = 15
    BLOCK_INT64 = 4
    MAX_INT32 = 2 ** 31 - 1
    TILING_MODE_0 = 0
    TILING_MODE_1 = 1
    MAX_DIM = 8
    FP16_MASK = 128
    INT64_BYTE = 8
    INT32_BYTE = 4


class TilingArgs():
    """NonZero tiling args"""
    def __init__(self, core_num, one_core_num, last_core_num, front_core, data_out_dim) -> None:
        """"args"""
        self.core_num = core_num
        self.one_core_num = one_core_num
        self.last_core_num = last_core_num
        self.front_core = front_core
        self.data_out_dim = data_out_dim


def _ceil(x_1, x_2):
    return (x_1 + x_2 - 1) // x_2


def _trans_ceil(x_1, x_2):
    res = (x_1 // x_2 + 1) if x_1 != 8 else 4
    return res


# 'pylint: disable=too-many-instance-attributes
class NonZero():
    """Function: use to store nonzero paramters
    """
    # 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments
    def __init__(self, x_shape, x_dtype, y_dtype, transpose, kernel_name):
        """Init NonZero base parameters
        """
        self.dtype_dict = {"int64": 8, "float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "uint32": 4,
                           "int8": 1, "bool": 1}
        self.res_blk_num_tensor = None
        self.col_add_tensor = None
        self.x_shape = list(x_shape)
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.x_dtype_ub = "float32"
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.ub_minimum_num = Constant.UB_MINIMUM_SIZE // self.dtype_dict.get(Constant.Y_BASE_DTYPE)
        self.one_block_num = Constant.UB_MINIMUM_SIZE // self.dtype_dict.get(x_dtype)
        self.dim = len(self.x_shape)
        self.transpose = transpose
        self.core_loop_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.is_v200 = tbe_platform.api_check_support("tik.data_move_pad")
        self.tiling = 2040
        self.init_tensor()

        # tilingdata
        self.tiling_dtype = "int64"
        self.tiling_align = util_common.align(Constant.TILING_ARG_NUM, Constant.BLOCK_INT64)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.core_num_var = self.tik_instance.Scalar(self.tiling_dtype, "core_num_var",
                                                     init_value=1)
        self.one_core_num_var = self.tik_instance.Scalar(self.tiling_dtype, "one_core_num_var",
                                                         init_value=1)
        self.last_core_num_var = self.tik_instance.Scalar(self.tiling_dtype, "last_core_num_var",
                                                          init_value=1)
        self.rank0 = self.tik_instance.Scalar(self.tiling_dtype, "rank0",
                                            init_value=1)
        self.rank1 = self.tik_instance.Scalar(self.tiling_dtype, "rank1",
                                            init_value=1)
        self.rank2 = self.tik_instance.Scalar(self.tiling_dtype, "rank2",
                                            init_value=1)
        self.rank3 = self.tik_instance.Scalar(self.tiling_dtype, "rank3",
                                            init_value=1)
        self.rank4 = self.tik_instance.Scalar(self.tiling_dtype, "rank4",
                                            init_value=1)
        self.rank5 = self.tik_instance.Scalar(self.tiling_dtype, "rank5",
                                              init_value=1)
        self.row = self.tik_instance.Scalar(self.tiling_dtype, "row",
                                            init_value=1)
        self.col = self.tik_instance.Scalar(self.tiling_dtype, "col",
                                            init_value=1)
        self.front_core = self.tik_instance.Scalar(self.tiling_dtype, "front_core",
                                                   init_value=1)
        self.out_last_dim = self.tik_instance.Scalar(self.tiling_dtype, "out_last_dim",
                                                     init_value=1)
        self.tiling_mode = self.tik_instance.Scalar(self.tiling_dtype, "tiling_mode",
                                                    init_value=0)
        self.seg_dim = self.tik_instance.Scalar(self.tiling_dtype, "seg_dim",
                                                    init_value=1)

    def init_tensor(self):
        """
        init_tensor
        """
        # Number of non_zero elements
        self.num = self.tik_instance.Scalar(Constant.SHAPE_DTYPE, "num", init_value=0)
        # Number of non_zero elements in a single core
        self.num_blk = self.tik_instance.Scalar(Constant.SHAPE_DTYPE, "num_blk")

        self.zero_scalar_uint32 = self.tik_instance.Scalar(init_value=0, dtype=Constant.SHAPE_DTYPE)
        self.zero_scalar_int32 = self.tik_instance.Scalar(init_value=0, dtype=Constant.Y_BASE_DTYPE)
        self.zero_scalar_fp32 = self.tik_instance.Scalar(init_value=0, dtype=self.x_dtype_ub)
        self.zero_scalar_fp16 = self.tik_instance.Scalar(init_value=0, dtype="float16")
        self.scalar_2 = self.tik_instance.Scalar("uint32", "scalar_2", init_value=2)

        self.x_gm = self.tik_instance.Tensor(self.x_dtype, (Constant.MAX_INT32, ), name="x", scope=tik.scope_gm)
        # Temporary storage of output data in workspace
        self.data_out = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (Constant.MAX_INT32, ),
                                                 name="data_out",
                                                 scope=tik.scope_gm,
                                                 is_workspace=True)
        # Temporary storage of output data in workspace
        self.shape_out = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (Constant.MAX_INT32, ),
                                                  name="shape_out",
                                                  scope=tik.scope_gm,
                                                  is_workspace=True)

        # Final output data
        if self.transpose:
            self.res_gm = self.tik_instance.Tensor(self.y_dtype, (self.dim, self.num), name="res_gm",\
                                                   scope=tik.scope_gm)
        else:
            self.res_gm = self.tik_instance.Tensor(self.y_dtype, (self.num, self.dim), name="res_gm",\
                                                   scope=tik.scope_gm)
        # Final output shape
        self.shape_out_gm = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (9,),
                                                     name="shape_out_gm",
                                                     scope=tik.scope_gm)

        # The offset of the current core output
        self.offset_gm = self.tik_instance.Scalar(Constant.SHAPE_DTYPE, "offset_gm", init_value=0)
        # Multi-core synchronization Tensor
        self.sync_workspace = self.tik_instance.Tensor("int64", (Constant.MAX_INT32, ),
                                                       name="barrier_workspace",
                                                       scope=tik.scope_gm,
                                                       is_workspace=True,
                                                       is_atomic_add=True)

    def update_static_tiling_args(self):
        """update staic tiling args"""
        shape_pad = [0 for _ in range(Constant.MAX_DIM - self.dim)]
        shape_all = shape_pad + self.x_shape
        rank = [self.rank0, self.rank1, self.rank2, self.rank3, self.rank4, self.rank5, self.row, self.col]
        for i, dim in enumerate(shape_all):
            rank[i].set_as(dim)
        if self.dim == 1:
            self.row.set_as(1)
            one_core_num = _ceil(self.x_shape[0], self.core_loop_num)
            core_num = _ceil(self.x_shape[0], one_core_num)
            last_core_num = self.x_shape[0] - one_core_num * (core_num - 1)
            front_core = core_num - 1 if self.x_shape[0] % core_num != 0 else 0
            data_out_dim = _ceil(one_core_num, self.ub_minimum_num) * self.ub_minimum_num
            tiling_args = TilingArgs(core_num, one_core_num, last_core_num, front_core, data_out_dim)
            tiling_mode = 0
            seg_dim = self.x_shape[0]
        else:
            tiling_mode = 1 if self.x_shape[0] == 1 and self.dim > 2 else 0
            if tiling_mode:
                seg_dim = self.x_shape[1]
                x_shape = deepcopy(self.x_shape)
                x_shape.pop(0)
                tiling_args = self.cal_running_info(x_shape)
            else:
                seg_dim = self.x_shape[0]
                x_shape = deepcopy(self.x_shape)
                tiling_args = self.cal_running_info(x_shape)
        self.core_num_var.set_as(tiling_args.core_num)
        # multipartion
        self.one_core_num_var.set_as(tiling_args.one_core_num)
        self.last_core_num_var.set_as(tiling_args.last_core_num)
        self.front_core.set_as(tiling_args.front_core)
        self.out_last_dim.set_as(tiling_args.data_out_dim)
        self.tiling_mode.set_as(tiling_mode)
        self.seg_dim.set_as(seg_dim)

    def cal_running_info(self, shape):
        """running tiling"""
        tmp_data = function_reduce(lambda x, y: x * y, shape[1:-1], 1)
        col_align = _ceil(shape[-1], self.ub_minimum_num) * self.ub_minimum_num
        if shape[0] < self.core_loop_num:
            core_num = shape[0]
            one_core_num = function_reduce(lambda x, y: x * y, shape[1:])
            last_core_num = one_core_num
            front_core = 0
            if shape[-1] % self.ub_minimum_num != 0:
                data_out_dim = col_align * tmp_data
            else:
                data_out_dim = _ceil(one_core_num, self.ub_minimum_num) * self.ub_minimum_num
        else:
            core_num = self.core_loop_num
            avg_row = shape[0] // core_num
            tail_row = shape[0] % core_num
            if tail_row > 0:
                one_core_num = (avg_row + 1) * function_reduce(lambda x, y: x * y, shape[1:])
                last_core_num = avg_row *  function_reduce(lambda x, y: x * y, shape[1:])
                front_core = tail_row
                if shape[-1] % self.ub_minimum_num != 0:
                    data_out_dim = col_align * tmp_data * (avg_row + 1)
                else:
                    data_out_dim = _ceil(one_core_num, self.ub_minimum_num) * self.ub_minimum_num
            else:
                one_core_num = avg_row * function_reduce(lambda x, y: x * y, shape[1:])
                last_core_num = one_core_num
                front_core = 0
                if shape[-1] % self.ub_minimum_num != 0:
                    data_out_dim = col_align * tmp_data * avg_row
                else:
                    data_out_dim = _ceil(one_core_num, self.ub_minimum_num) * self.ub_minimum_num
        tiling_args = TilingArgs(core_num, one_core_num, last_core_num, front_core, data_out_dim)
        return tiling_args

    def non_zero_compute(self):
        """
        non_zero_compute
        """
        # tilingdata
        is_unkonun = [i > 0 for i in self.x_shape]
        if all(is_unkonun):
            self.update_static_tiling_args()
        else:
            tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                                scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // Constant.BLOCK_INT64,
                                        0, 0)
            self.rank0.set_as(tiling_ub[0])
            self.rank1.set_as(tiling_ub[1])
            self.rank2.set_as(tiling_ub[2])
            self.rank3.set_as(tiling_ub[3])
            self.rank4.set_as(tiling_ub[4])
            self.rank5.set_as(tiling_ub[5])
            self.row.set_as(tiling_ub[6])
            self.col.set_as(tiling_ub[7])
            self.core_num_var.set_as(tiling_ub[8])
            # multipartion
            self.one_core_num_var.set_as(tiling_ub[9])
            self.last_core_num_var.set_as(tiling_ub[10])
            self.front_core.set_as(tiling_ub[11])
            self.out_last_dim.set_as(tiling_ub[12])
            self.tiling_mode.set_as(tiling_ub[13])
            self.seg_dim.set_as(tiling_ub[14])
        self.col_add_tensor = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (Constant.UB_REPEAT_SIZE, ),
                                                       name="col_add_tensor", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, Constant.UB_REPEAT_SIZE) as init_idx:
            self.col_add_tensor[init_idx].set_as(init_idx)

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as blk_idx:
            with self.tik_instance.if_scope(blk_idx < self.front_core):
                cur_core_num = self.one_core_num_var
                row_num = self.seg_dim // self.core_num_var + 1
                begin_idx = blk_idx * row_num
                end_idx = (blk_idx + 1) * row_num
                row_offset = cur_core_num / row_num
                offset = blk_idx * self.one_core_num_var
                self.compute_one_core(blk_idx, cur_core_num, begin_idx, end_idx, row_offset, offset)
            with self.tik_instance.else_scope():
                cur_core_num = self.last_core_num_var
                row_num = self.seg_dim // self.core_num_var
                begin_idx = self.front_core + blk_idx * row_num
                end_idx = self.front_core + (blk_idx + 1) * row_num
                row_offset = cur_core_num / row_num
                offset = self.front_core * self.one_core_num_var + (blk_idx - self.front_core) * cur_core_num
                self.compute_one_core(blk_idx, cur_core_num, begin_idx, end_idx, row_offset, offset)

            # block_barrier needs to bind more than 1 core
            with self.tik_instance.if_scope(self.core_num_var > 1):
                self.tik_instance.block_barrier(self.sync_workspace)

            shape_out_ub = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (self.core_num_var, self.ub_minimum_num),
                                                    name="shape_out_ub",
                                                    scope=tik.scope_ubuf)
            shape_out_ub_2 = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (9,),
                                                      name="shape_out_ub_2",
                                                      scope=tik.scope_ubuf)

            self.tik_instance.data_move(shape_out_ub,
                                        self.shape_out,
                                        sid=0,
                                        nburst=1,
                                        burst=self.core_num_var,
                                        src_stride=0,
                                        dst_stride=0)

            # The shape_out_ub_2 is (2,2,n), The first number represents the dim number of the output shape
            shape_out_ub_2[0].set_as(self.scalar_2)
            if self.transpose:
                # Data handling after block_barrier
                self.multi_core_sync(blk_idx, shape_out_ub)
                shape_out_ub_2[1].set_as(self.dim)
                shape_out_ub_2[2].set_as(self.num)
            else:
                if self.dim == 1:
                    self.multi_core_sync(blk_idx, shape_out_ub)
                else:
                    self.multi_core_sync_trans(blk_idx, shape_out_ub)
                shape_out_ub_2[1].set_as(self.num)
                shape_out_ub_2[2].set_as(self.dim)
            if self.is_v200:
                self.tik_instance.data_move_pad(self.shape_out_gm,
                                                shape_out_ub_2, 1, 3 * Constant.INT32_BYTE, 0, 0)
            else:
                self.tik_instance.data_move(self.shape_out_gm,
                                            shape_out_ub_2,
                                            sid=0,
                                            nburst=1,
                                            burst=1,
                                            src_stride=0,
                                            dst_stride=0)

        # init_value
        opt_config = {"enable_const_fold": True}
        tbe_context.get_context().add_compile_info("block_dim", self.core_loop_num)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm],
                                   outputs=[self.res_gm, self.shape_out_gm],
                                   flowtable=(self.tiling_gm,),
                                   config=opt_config)

        return self.tik_instance

    # 'pylint: disable=huawei-too-many-arguments
    def compute_one_core(self, blk_idx, cur_core_num, begin_idx, end_idx, row_offset, offset):
        """
        compute_one_core
        """
        tiling_loop = _ceil(self.col, self.tiling)
        tiling_tail = self.col - (tiling_loop - 1) * self.tiling
        # The number of non-zero elements in the current core
        self.res_blk_num_tensor = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (self.ub_minimum_num,),
                                                           name="res_blk_num_tensor",
                                                           scope=tik.scope_ubuf)
        vec_mask = self.ub_minimum_num
        self.tik_instance.vector_dup(vec_mask, self.res_blk_num_tensor, self.zero_scalar_uint32, Constant.BLOCK_STRIDE,
                                     Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE)

        if self.dim == 1:
            offset = blk_idx * self.one_core_num_var
            tiling_loop = _ceil(cur_core_num, self.tiling)
            tiling_tail = cur_core_num - (tiling_loop - 1) * self.tiling
            with self.tik_instance.for_range(0, tiling_loop) as t_idx:
                offset = offset + t_idx * self.tiling
                with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                    self.compute_one_loop(blk_idx, 0, 0, 0, 0, 0, 0, 0, offset, self.tiling)
                with self.tik_instance.else_scope():
                    self.compute_one_loop(blk_idx, 0, 0, 0, 0, 0, 0, 0, offset, tiling_tail)
        elif self.dim == 2:
            self.compute_two_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)
        elif self.dim == 3:
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
                self.compute_two_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset,
                                     offset)
            with self.tik_instance.else_scope():
                if self.is_v200:
                    with self.tik_instance.if_scope(tik.all(self.row > 1000, self.col <= 10)):
                        self.compute_high_performance(blk_idx, begin_idx, end_idx, row_offset,
                                                    offset)
                    with self.tik_instance.else_scope():
                        self.compute_three_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset,
                                            offset)
                else:
                    self.compute_three_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset,
                                            offset)
        elif self.dim == 4:
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
                self.compute_three_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset,\
                                       offset)
            with self.tik_instance.else_scope():
                self.compute_four_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)
        elif self.dim == 5:
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
                self.compute_four_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)
            with self.tik_instance.else_scope():
                self.compute_five_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)
        elif self.dim == 6:
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
                self.compute_five_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)
            with self.tik_instance.else_scope():
                self.compute_six_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)
        elif self.dim == 7:
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
                self.compute_six_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)
            with self.tik_instance.else_scope():
                self.compute_seven_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)
        elif self.dim == 8:
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
                self.compute_seven_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)
            with self.tik_instance.else_scope():
                self.compute_eight_dim(blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset)

        self.tik_instance.data_move(self.shape_out[blk_idx*self.ub_minimum_num],
                                    self.res_blk_num_tensor,
                                    sid=0,
                                    nburst=1,
                                    burst=1,
                                    src_stride=0,
                                    dst_stride=0)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def compute_two_dim(self, blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset):
        """
        compute_two_dim
        """
        with self.tik_instance.for_range(begin_idx, end_idx) as row_idx:
            offset = offset + (row_idx - begin_idx) * row_offset
            with self.tik_instance.for_range(0, tiling_loop) as t_idx:
                offset = offset + t_idx * self.tiling
                with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                    self.compute_one_loop(blk_idx, 0, 0, 0, 0, 0, 0, row_idx, offset, self.tiling, t_idx)
                with self.tik_instance.else_scope():
                    self.compute_one_loop(blk_idx, 0, 0, 0, 0, 0, 0, row_idx, offset, tiling_tail, t_idx)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def compute_three_dim(self, blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset):
        """
        compute_three_dim
        """
        with self.tik_instance.for_range(begin_idx, end_idx) as rank5_idx:
            offset = offset + (rank5_idx - begin_idx) * row_offset
            with self.tik_instance.for_range(0, self.row) as row_idx:
                offset = offset + row_idx * self.col
                with self.tik_instance.for_range(0, tiling_loop) as t_idx:
                    offset = offset + t_idx * self.tiling
                    with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                        self.compute_one_loop(blk_idx, 0, 0, 0, 0, 0, rank5_idx, row_idx, offset, self.tiling,\
                                t_idx)
                    with self.tik_instance.else_scope():
                        self.compute_one_loop(blk_idx, 0, 0, 0, 0, 0, rank5_idx, row_idx, offset, tiling_tail,\
                                t_idx)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def compute_four_dim(self, blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset):
        """
        compute_four_dim
        """
        with self.tik_instance.for_range(begin_idx, end_idx) as rank4_idx:
            offset = offset + (rank4_idx - begin_idx) * row_offset
            with self.tik_instance.for_range(0, self.rank5) as rank5_idx:
                offset = offset + rank5_idx * self.row * self.col
                with self.tik_instance.for_range(0, self.row) as row_idx:
                    offset = offset + row_idx * self.col
                    with self.tik_instance.for_range(0, tiling_loop) as t_idx:
                        offset = offset + t_idx * self.tiling
                        with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                            self.compute_one_loop(blk_idx, 0, 0, 0, 0, rank4_idx, rank5_idx, row_idx, offset,\
                                    self.tiling, t_idx)
                        with self.tik_instance.else_scope():
                            self.compute_one_loop(blk_idx, 0, 0, 0, 0, rank4_idx, rank5_idx, row_idx, offset,\
                                    tiling_tail, t_idx)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def compute_five_dim(self, blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset):
        """
        compute_five_dim
        """
        with self.tik_instance.for_range(begin_idx, end_idx) as rank3_idx:
            offset = offset + (rank3_idx - begin_idx) * row_offset
            with self.tik_instance.for_range(0, self.rank4) as rank4_idx:
                offset = offset + rank4_idx * self.rank5 * self.row * self.col
                with self.tik_instance.for_range(0, self.rank5) as rank5_idx:
                    offset = offset + rank5_idx * self.row * self.col
                    with self.tik_instance.for_range(0, self.row) as row_idx:
                        offset = offset + row_idx * self.col
                        with self.tik_instance.for_range(0, tiling_loop) as t_idx:
                            offset = offset + t_idx * self.tiling
                            with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                                self.compute_one_loop(blk_idx, 0, 0, 0, rank3_idx, rank4_idx, rank5_idx,\
                                        row_idx, offset, self.tiling, t_idx)
                            with self.tik_instance.else_scope():
                                self.compute_one_loop(blk_idx, 0, 0, 0, rank3_idx, rank4_idx, rank5_idx,\
                                        row_idx, offset, tiling_tail, t_idx)

    # 'pylint: disable=too-many-locals,huawei-too-many-arguments
    def compute_six_dim(self, blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset):
        """
        compute_six_dim
        """
        with self.tik_instance.for_range(begin_idx, end_idx) as rank2_idx:
            offset = offset + (rank2_idx - begin_idx) * row_offset
            with self.tik_instance.for_range(0, self.rank3) as rank3_idx:
                offset = offset + rank3_idx * self.rank4 * self.rank5 * self.row * self.col
                with self.tik_instance.for_range(0, self.rank4) as rank4_idx:
                    offset = offset + rank4_idx * self.rank5 * self.row * self.col
                    with self.tik_instance.for_range(0, self.rank5) as rank5_idx:
                        offset = offset + rank5_idx * self.row * self.col
                        with self.tik_instance.for_range(0, self.row) as row_idx:
                            offset = offset + row_idx * self.col
                            with self.tik_instance.for_range(0, tiling_loop) as t_idx:
                                offset = offset + t_idx * self.tiling
                                with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                                    self.compute_one_loop(blk_idx, 0, 0, rank2_idx, rank3_idx, rank4_idx,\
                                            rank5_idx, row_idx, offset, self.tiling, t_idx)
                                with self.tik_instance.else_scope():
                                    self.compute_one_loop(blk_idx, 0, 0, rank2_idx, rank3_idx, rank4_idx,\
                                            rank5_idx, row_idx, offset, tiling_tail, t_idx)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def compute_seven_dim(self, blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset):
        """
        compute_seven_dim
        """
        with self.tik_instance.for_range(begin_idx, end_idx) as rank1_idx:
            offset = offset + (rank1_idx - begin_idx) * row_offset
            with self.tik_instance.for_range(0, self.rank2) as rank2_idx:
                offset = offset + rank2_idx * self.rank3 * self.rank4 * self.rank5 * self.row * self.col
                with self.tik_instance.for_range(0, self.rank3) as rank3_idx:
                    offset = offset + rank3_idx * self.rank4 * self.rank5 * self.row * self.col
                    with self.tik_instance.for_range(0, self.rank4) as rank4_idx:
                        offset = offset + rank4_idx * self.rank5 * self.row * self.col
                        with self.tik_instance.for_range(0, self.rank5) as rank5_idx:
                            offset = offset + rank5_idx * self.row * self.col
                            with self.tik_instance.for_range(0, self.row) as row_idx:
                                offset = offset + row_idx * self.col
                                with self.tik_instance.for_range(0, tiling_loop) as t_idx:
                                    offset = offset + t_idx * self.tiling
                                    with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                                        self.compute_one_loop(blk_idx, 0, rank1_idx, rank2_idx, rank3_idx,\
                                                rank4_idx, rank5_idx, row_idx, offset, self.tiling, t_idx)
                                    with self.tik_instance.else_scope():
                                        self.compute_one_loop(blk_idx, 0, rank1_idx, rank2_idx, rank3_idx,\
                                                rank4_idx, rank5_idx, row_idx, offset, tiling_tail, t_idx)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def compute_eight_dim(self, blk_idx, tiling_loop, tiling_tail, begin_idx, end_idx, row_offset, offset):
        """
        compute_eight_dim
        """
        with self.tik_instance.for_range(begin_idx, end_idx) as rank0_idx:
            offset = offset + (rank0_idx - begin_idx) * row_offset
            with self.tik_instance.for_range(0, self.rank1) as rank1_idx:
                offset = offset + rank1_idx * self. rank2 * self.rank3 * self.rank4 * self.rank5 * self.row\
                        * self.col
                with self.tik_instance.for_range(0, self.rank2) as rank2_idx:
                    offset = offset + rank2_idx * self.rank3 * self.rank4 * self.rank5 * self.row * self.col
                    with self.tik_instance.for_range(0, self.rank3) as rank3_idx:
                        offset = offset + rank3_idx * self.rank4 * self.rank5 * self.row * self.col
                        with self.tik_instance.for_range(0, self.rank4) as rank4_idx:
                            offset = offset + rank4_idx * self.rank5 * self.row * self.col
                            with self.tik_instance.for_range(0, self.rank5) as rank5_idx:
                                offset = offset + rank5_idx * self.row * self.col
                                with self.tik_instance.for_range(0, self.row) as row_idx:
                                    offset = offset + row_idx * self.col
                                    with self.tik_instance.for_range(0, tiling_loop) as t_idx:
                                        offset = offset + t_idx * self.tiling
                                        with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                                            self.compute_one_loop(blk_idx, rank0_idx, rank1_idx, rank2_idx,\
                                            rank3_idx, rank4_idx, rank5_idx, row_idx, offset, self.tiling, t_idx)
                                        with self.tik_instance.else_scope():
                                            self.compute_one_loop(blk_idx, rank0_idx, rank1_idx, rank2_idx,\
                                            rank3_idx, rank4_idx, rank5_idx, row_idx, offset, tiling_tail, t_idx)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def compute_high_performance(self, blk_idx, begin_idx, end_idx, row_offset, offset):
        """
        high performance branch for col is too small
        """
        with self.tik_instance.for_range(begin_idx, end_idx) as rank5_idx:
            offset = offset + (rank5_idx - begin_idx) * row_offset
            cur_tiling_num = row_offset
            tiling_loop = _ceil(cur_tiling_num, self.tiling)
            tiling_tail = cur_tiling_num - (tiling_loop - 1) * self.tiling
            with self.tik_instance.for_range(0, tiling_loop) as t_idx:
                offset = offset + t_idx * self.tiling
                with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                    self.compute_hp_loop(blk_idx, rank5_idx, offset, self.tiling, t_idx)
                with self.tik_instance.else_scope():
                    self.compute_hp_loop(blk_idx, rank5_idx, offset, tiling_tail, t_idx)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def compute_hp_loop(self, blk_idx, rank5_idx, offset, cur_loop_num, t_idx):
        """
        high performance branch
        """
        blk_size = cur_loop_num
        align_num = Constant.UB_REPEAT_SIZE
        all_tail = blk_size % align_num
        # Due to the limitation of the vcmpvs_ne instruction
        # the input elements processed by ub at one time need to be 64 aligned
        blk_align_size = _ceil(blk_size, align_num) * align_num
        x_shape_one_loop = (blk_align_size,)
        x_shape_one_loop_tmp = (blk_align_size * 2, )
        if self.x_dtype == "int8":
            x_ub_zero = self.tik_instance.Tensor("float16", x_shape_one_loop_tmp, name="x_ub_zero",
                                                 scope=tik.scope_ubuf)
            x_ub_tmp = self.tik_instance.Tensor("float16", x_shape_one_loop, name="x_ub_tmp", scope=tik.scope_ubuf)
            x_ub_last = self.tik_instance.Tensor(self.x_dtype_ub, x_shape_one_loop, name="x_ub_last",
                                                 scope=tik.scope_ubuf)
            self.v_dup(x_ub_zero, self.zero_scalar_fp16, blk_align_size * 2, [], "float16")
            x_ub = x_ub_zero.reinterpret_cast_to("int8")
        elif self.x_dtype == "float32":
            x_ub = self.tik_instance.Tensor(self.x_dtype, x_shape_one_loop, name="x_ub", scope=tik.scope_ubuf)
            self.v_dup(x_ub, self.zero_scalar_fp32, blk_align_size, [], self.x_dtype)
        elif self.x_dtype == "float16":
            x_ub = self.tik_instance.Tensor(self.x_dtype, x_shape_one_loop, name="x_ub", scope=tik.scope_ubuf)
            x_ub_last = self.tik_instance.Tensor(self.x_dtype_ub, x_shape_one_loop, name="x_ub_last",
                                                 scope=tik.scope_ubuf)
            self.v_dup(x_ub, self.zero_scalar_fp16, blk_align_size, [], self.x_dtype)
        elif self.x_dtype == "bfloat16":
            x_ub_zero = self.tik_instance.Tensor("float32", x_shape_one_loop, name="x_ub_zero",
                                                 scope=tik.scope_ubuf)
            x_ub_last = self.tik_instance.Tensor(self.x_dtype_ub, x_shape_one_loop, name="x_ub_last",
                                                 scope=tik.scope_ubuf)
            self.v_dup(x_ub_zero, self.zero_scalar_fp32, blk_align_size, [], self.x_dtype_ub)
            x_ub = x_ub_zero.reinterpret_cast_to("bfloat16")
        self.gm_to_ub(x_ub, blk_size, blk_align_size, all_tail, offset)
        res_blk_num = self.tik_instance.Scalar(dtype=Constant.SHAPE_DTYPE, name="res_blk_num")
        res_blk_num_cur_core = self.tik_instance.Scalar(dtype=Constant.SHAPE_DTYPE, name="res_blk_num_cur_core")
        col_auxiliary_matrix = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE,
                                                        x_shape_one_loop,
                                                        name="col_auxiliary_matrix",
                                                        scope=tik.scope_ubuf)
        row_auxiliary_matrix_int32 = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE,
                                                              x_shape_one_loop,
                                                              name="row_auxiliary_matrix_int32",
                                                              scope=tik.scope_ubuf)
        row_auxiliary_matrix = self.tik_instance.Tensor("float32",
                                                        x_shape_one_loop,
                                                        name="row_auxiliary_matrix_fp32",
                                                        scope=tik.scope_ubuf)
        auxiliary_matrix_temp = self.tik_instance.Tensor("float32",
                                                         x_shape_one_loop,
                                                         name="auxiliary_matrix_temp_fp32",
                                                         scope=tik.scope_ubuf)
        auxiliary_matrix_temp_int32 = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE,
                                                               x_shape_one_loop,
                                                               name="auxiliary_matrix_temp_int32",
                                                               scope=tik.scope_ubuf)
        sub_auxiliary_matrix = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE,
                                                        x_shape_one_loop,
                                                        name="sub_auxiliary_matrix",
                                                        scope=tik.scope_ubuf)
        vreduce_mask = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (blk_align_size // Constant.UB_MINIMUM_SIZE,),
                                                name="vreduce_mask",
                                                scope=tik.scope_ubuf)
        dst_ub_row = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_row", scope=tik.scope_ubuf)
        dst_ub_col = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_col", scope=tik.scope_ubuf)
        if self.x_dtype == "int8":
            self.gen_cast(x_ub_tmp, x_ub, blk_align_size, "float16", "int8")
            self.gen_cast(x_ub_last, x_ub_tmp, blk_align_size, "float32", "float16")
            self.gen_mask(vreduce_mask, x_ub_last, self.zero_scalar_fp32, blk_align_size, "float32")
        elif self.x_dtype == "float32":
            self.gen_mask(vreduce_mask, x_ub, self.zero_scalar_fp32, blk_align_size, self.x_dtype)
        elif self.x_dtype == "float16":
            self.gen_cast(x_ub_last, x_ub, blk_align_size, "float32", "float16")
            self.gen_mask(vreduce_mask, x_ub_last, self.zero_scalar_fp32, blk_align_size, "float32")
        elif self.x_dtype == "bfloat16":
            self.gen_cast(x_ub_last, x_ub, blk_align_size, "float32", "bfloat16")
            self.gen_mask(vreduce_mask, x_ub_last, self.zero_scalar_fp32, blk_align_size, "float32")

        row_auxiliary_matrix_int32, col_auxiliary_matrix = self._build_row_hp_index_mtr(row_auxiliary_matrix,
            auxiliary_matrix_temp, auxiliary_matrix_temp_int32, row_auxiliary_matrix_int32, sub_auxiliary_matrix,
            col_auxiliary_matrix, blk_size, t_idx)
        # Calculate the col index of non-zero elements
        self.tik_instance.vreduce(blk_align_size, dst_ub_row, row_auxiliary_matrix_int32, vreduce_mask, 1, 1,
                                Constant.REPEAT_STRIDE, 1, 0, res_blk_num, "counter")
        self.tik_instance.vreduce(blk_align_size, dst_ub_col, col_auxiliary_matrix, vreduce_mask, 1, 1,
                                Constant.REPEAT_STRIDE, 1, 0, None, "counter")
        res_blk_num_cur_core.set_as(self.res_blk_num_tensor[0])
        data_out_offset = res_blk_num_cur_core        
        dst_ub_rk5 = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_rk5", scope=tik.scope_ubuf)
        self.v_dup(dst_ub_rk5, rank5_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
        ans = [dst_ub_col, dst_ub_row, dst_ub_rk5]
        for _idx in range(self.dim):
            col_offset = data_out_offset + blk_idx * self.dim * self.out_last_dim + _idx * self.out_last_dim
            self.ub_to_workspace(ans[self.dim - 1 - _idx], res_blk_num, col_offset)
        # Update the non-zero elements of the current core
        res_blk_num_cur_core.set_as(res_blk_num_cur_core + res_blk_num)
        self.res_blk_num_tensor[0].set_as(res_blk_num_cur_core)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def compute_one_loop(self, blk_idx, rank0_idx, rank1_idx, rank2_idx, rank3_idx, rank4_idx, \
        rank5_idx, row_idx, offset, cur_loop_num=0, t_idx=0):
        """
        compute_one_loop
        """
        blk_size = cur_loop_num
        align_num = Constant.UB_REPEAT_SIZE
        all_tail = blk_size % align_num
        # Due to the limitation of the vcmpvs_ne instruction
        # the input elements processed by ub at one time need to be 64 aligned
        blk_align_size = _ceil(blk_size, align_num) * align_num
        x_shape_one_loop = (blk_align_size,)
        x_shape_one_loop_tmp = (blk_align_size * 2, )

        if self.x_dtype == "int8":
            x_ub_zero = self.tik_instance.Tensor("float16", x_shape_one_loop_tmp, name="x_ub_zero",
                                                 scope=tik.scope_ubuf)
            x_ub_tmp = self.tik_instance.Tensor("float16", x_shape_one_loop, name="x_ub_tmp", scope=tik.scope_ubuf)
            x_ub_last = self.tik_instance.Tensor(self.x_dtype_ub, x_shape_one_loop, name="x_ub_last",
                                                 scope=tik.scope_ubuf)
            self.v_dup(x_ub_zero, self.zero_scalar_fp16, blk_align_size * 2, [], "float16")
            x_ub = x_ub_zero.reinterpret_cast_to("int8")
        elif self.x_dtype == "float32":
            x_ub = self.tik_instance.Tensor(self.x_dtype, x_shape_one_loop, name="x_ub", scope=tik.scope_ubuf)
            self.v_dup(x_ub, self.zero_scalar_fp32, blk_align_size, [], self.x_dtype)
        elif self.x_dtype == "float16":
            x_ub = self.tik_instance.Tensor(self.x_dtype, x_shape_one_loop, name="x_ub", scope=tik.scope_ubuf)
            x_ub_last = self.tik_instance.Tensor(self.x_dtype_ub, x_shape_one_loop, name="x_ub_last",
                                                 scope=tik.scope_ubuf)
            self.v_dup(x_ub, self.zero_scalar_fp16, blk_align_size, [], self.x_dtype)
        elif self.x_dtype == "bfloat16":
            x_ub_zero = self.tik_instance.Tensor("float32", x_shape_one_loop, name="x_ub_zero",
                                                 scope=tik.scope_ubuf)
            x_ub_last = self.tik_instance.Tensor(self.x_dtype_ub, x_shape_one_loop, name="x_ub_last",
                                                 scope=tik.scope_ubuf)
            self.v_dup(x_ub_zero, self.zero_scalar_fp32, blk_align_size, [], self.x_dtype_ub)
            x_ub = x_ub_zero.reinterpret_cast_to("bfloat16")
        self.gm_to_ub(x_ub, blk_size, blk_align_size, all_tail, offset)
        res_blk_num = self.tik_instance.Scalar(dtype=Constant.SHAPE_DTYPE, name="res_blk_num")
        res_blk_num_cur_core = self.tik_instance.Scalar(dtype=Constant.SHAPE_DTYPE, name="res_blk_num_cur_core")
        dst_ub_col = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_col", scope=tik.scope_ubuf)
        if not tbe_platform.api_check_support("tik.vreduce"):
            dst_star = self.tik_instance.Scalar(dtype=Constant.SHAPE_DTYPE, name="dst_star", init_value=0)
            if self.x_dtype == "float16":
                self.gen_cast(x_ub_last, x_ub, blk_align_size, "float32", "float16")
                if self.dim == 1:
                    dst_ub_col, dst_star = self.row_index_matrix(dst_ub_col, dst_star, x_ub_last, blk_size, offset)
                else:
                    dst_ub_col, dst_star = self.row_index_matrix(dst_ub_col, dst_star, x_ub_last,
                                                                 blk_size, t_idx * self.tiling)
            else:
                if self.dim == 1:
                    dst_ub_col, dst_star = self.row_index_matrix(dst_ub_col, dst_star, x_ub, blk_size, offset)
                else:
                    dst_ub_col, dst_star = self.row_index_matrix(dst_ub_col, dst_star, x_ub,
                                                                 blk_size, t_idx * self.tiling)

            res_blk_num.set_as(dst_star)
        else:
            col_auxiliary_matrix = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE,
                                                            x_shape_one_loop,
                                                            name="col_auxiliary_matrix",
                                                            scope=tik.scope_ubuf)
            vreduce_mask = self.tik_instance.Tensor(Constant.SHAPE_DTYPE, (blk_align_size // Constant.UB_MINIMUM_SIZE,),
                                                    name="vreduce_mask",
                                                    scope=tik.scope_ubuf)
            # Initialize the auxiliary matrix of rows and columns
            if self.dim == 1:
                col_auxiliary_matrix = self._build_col_index_mtr(col_auxiliary_matrix, blk_size, offset)
            else:
                col_auxiliary_matrix = self._build_col_index_mtr(col_auxiliary_matrix, blk_size, t_idx * self.tiling)
            if self.x_dtype == "int8":
                self.gen_cast(x_ub_tmp, x_ub, blk_align_size, "float16", "int8")
                self.gen_cast(x_ub_last, x_ub_tmp, blk_align_size, "float32", "float16")
                self.gen_mask(vreduce_mask, x_ub_last, self.zero_scalar_fp32, blk_align_size, "float32")
            elif self.x_dtype == "float32":
                self.gen_mask(vreduce_mask, x_ub, self.zero_scalar_fp32, blk_align_size, self.x_dtype)
            elif self.x_dtype == "float16":
                self.gen_cast(x_ub_last, x_ub, blk_align_size, "float32", "float16")
                self.gen_mask(vreduce_mask, x_ub_last, self.zero_scalar_fp32, blk_align_size, "float32")
            elif self.x_dtype == "bfloat16":
                self.gen_cast(x_ub_last, x_ub, blk_align_size, "float32", "bfloat16")
                self.gen_mask(vreduce_mask, x_ub_last, self.zero_scalar_fp32, blk_align_size, "float32")
            # Calculate the col index of non-zero elements
            self.tik_instance.vreduce(blk_align_size, dst_ub_col, col_auxiliary_matrix, vreduce_mask, 1, 1,
                                    Constant.REPEAT_STRIDE, 1, 0, res_blk_num, "counter")
        res_blk_num_cur_core.set_as(self.res_blk_num_tensor[0])
        data_out_offset = res_blk_num_cur_core
        dst_ub_row = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_row", scope=tik.scope_ubuf)
        dst_ub_rk5 = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_rk5", scope=tik.scope_ubuf)
        dst_ub_rk4 = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_rk4", scope=tik.scope_ubuf)
        dst_ub_rk3 = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_rk3", scope=tik.scope_ubuf)
        dst_ub_rk2 = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_rk2", scope=tik.scope_ubuf)
        dst_ub_rk1 = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_rk1", scope=tik.scope_ubuf)
        dst_ub_rk0 = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (blk_align_size,),
                                              name="dst_ub_rk0", scope=tik.scope_ubuf)
        ans = []
        if self.dim == 1:
            ans = [dst_ub_col]
        elif self.dim == 2:
            self.v_dup(dst_ub_row, row_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            ans = [dst_ub_col, dst_ub_row]
        elif self.dim == 3:
            self.v_dup(dst_ub_row, row_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk5, rank5_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            ans = [dst_ub_col, dst_ub_row, dst_ub_rk5]
        elif self.dim == 4:
            self.v_dup(dst_ub_row, row_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk5, rank5_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk4, rank4_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            ans = [dst_ub_col, dst_ub_row, dst_ub_rk5, dst_ub_rk4]
        elif self.dim == 5:
            self.v_dup(dst_ub_row, row_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk5, rank5_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk4, rank4_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk3, rank3_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            ans = [dst_ub_col, dst_ub_row, dst_ub_rk5, dst_ub_rk4, dst_ub_rk3]
        elif self.dim == 6:
            self.v_dup(dst_ub_row, row_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk5, rank5_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk4, rank4_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk3, rank3_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk2, rank2_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            ans = [dst_ub_col, dst_ub_row, dst_ub_rk5, dst_ub_rk4, dst_ub_rk3, dst_ub_rk2]
        elif self.dim == 7:
            self.v_dup(dst_ub_row, row_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk5, rank5_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk4, rank4_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk3, rank3_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk2, rank2_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk1, rank1_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            ans = [dst_ub_col, dst_ub_row, dst_ub_rk5, dst_ub_rk4, dst_ub_rk3, dst_ub_rk2, dst_ub_rk1]
        else:
            self.v_dup(dst_ub_row, row_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk5, rank5_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk4, rank4_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk3, rank3_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk2, rank2_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk1, rank1_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            self.v_dup(dst_ub_rk0, rank0_idx, res_blk_num, [], Constant.Y_BASE_DTYPE)
            ans = [dst_ub_col, dst_ub_row, dst_ub_rk5, dst_ub_rk4, dst_ub_rk3, dst_ub_rk2, dst_ub_rk1, dst_ub_rk0]
        for _idx in range(self.dim):
            col_offset = data_out_offset + blk_idx * self.dim * self.out_last_dim + _idx * self.out_last_dim
            self.ub_to_workspace(ans[self.dim - 1 - _idx], res_blk_num, col_offset)
        # Update the non-zero elements of the current core
        res_blk_num_cur_core.set_as(res_blk_num_cur_core + res_blk_num)
        self.res_blk_num_tensor[0].set_as(res_blk_num_cur_core)

    # 'pylint: disable=too-many-locals,too-many-statements,huawei-too-many-arguments
    def gm_to_ub(self, x_ub, blk_size, blk_align_size, all_tail, offset):
        """"move gm to ub"""
        # When there is no tail block, save and withdraw as a whole, and there will be no out of bounds behavior
        with self.tik_instance.if_scope(all_tail == 0):
            self.tik_instance.data_move(x_ub,
                                        self.x_gm[offset],
                                        sid=0,
                                        nburst=1,
                                        burst=blk_align_size // self.one_block_num,
                                        src_stride=0,
                                        dst_stride=0)
        with self.tik_instance.else_scope():
            dma_burst = blk_size // self.one_block_num
            dma_tail = blk_size % self.one_block_num
            with self.tik_instance.if_scope(dma_burst > 0):
                self.tik_instance.data_move(x_ub,
                                            self.x_gm[offset],
                                            sid=0,
                                            nburst=1,
                                            burst=dma_burst,
                                            src_stride=0,
                                            dst_stride=0)
            # move input elements that are less than ub_minimun
            with self.tik_instance.if_scope(dma_tail > 0):
                gm_offset = dma_burst * self.one_block_num + offset
                ub_offset = dma_burst * self.one_block_num
                unit_tensor = self.tik_instance.Tensor(self.x_dtype, (self.one_block_num,),
                                                       name="unit_tensor",
                                                       scope=tik.scope_ubuf)
                # Tail block processing may result in read out of bounds
                if self.is_v200:
                    self.tik_instance.data_move_pad(unit_tensor, self.x_gm[gm_offset], 1,
                                                    dma_tail * self.dtype_dict.get(self.x_dtype), 0, 0)
                else:
                    self.tik_instance.data_move(unit_tensor,
                                                self.x_gm[gm_offset],
                                                sid=0,
                                                nburst=1,
                                                burst=1,
                                                src_stride=0,
                                                dst_stride=0)
                with self.tik_instance.for_range(0, dma_tail) as _idx:
                    x_ub[ub_offset + _idx].set_as(unit_tensor[_idx])

    def ub_to_workspace(self, dst_ub_col, res_blk_num, col_gm_offset):
        """
        dataout fill num
        """
        burst_ub = _ceil(res_blk_num, self.ub_minimum_num)
        # move out to workspace
        with self.tik_instance.if_scope(burst_ub > 0):
            # data_out shape is [self.core_num_var, self.dim, self.out_last_dim]
            # Tail block processing may result in read out of bounds
            if self.is_v200:
                self.tik_instance.data_move_pad(self.data_out[col_gm_offset], dst_ub_col, 1,
                                                res_blk_num * Constant.INT32_BYTE, 0, 0)
            else:
                self.tik_instance.data_move(self.data_out[col_gm_offset],
                                            dst_ub_col,
                                            sid=0,
                                            nburst=1,
                                            burst=burst_ub,
                                            src_stride=0,
                                            dst_stride=0)

    # 'pylint: disable=too-many-arguments,unused-argument
    def v_dup(self, dst, scalar, size, dst_offset, x_dtype):
        """
        v_dup
        """
        unit = Constant.VECTOR_BLOCK_SIZE // (self.dtype_dict.get(x_dtype))
        repeat = size // unit
        left = size % unit
        repeat_max_value = Constant.MAX_REPEAT_TIME
        repeat_loop = repeat // repeat_max_value
        repeat_left = repeat % repeat_max_value
        with self.tik_instance.if_scope(repeat_loop > 0):
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                self.tik_instance.vector_dup(unit, dst[rpt_idx * repeat_max_value * unit], scalar, repeat_max_value, 1,
                                             Constant.REPEAT_STRIDE)
        with self.tik_instance.if_scope(repeat_left > 0):
            self.tik_instance.vector_dup(unit, dst[repeat_loop * repeat_max_value * unit], scalar, repeat_left, 1,
                                         Constant.REPEAT_STRIDE)
        with self.tik_instance.if_scope(left > 0):
            self.tik_instance.vector_dup(left, dst[repeat * unit], scalar, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                                         Constant.REPEAT_STRIDE)

    # 'pylint: disable=too-many-locals,huawei-too-many-arguments
    def gen_cast(self, dst, src, size, dst_type, src_type, round_mode="none"):
        "gen_cast"
        unit = Constant.VECTOR_BLOCK_SIZE // (self.dtype_dict.get(dst_type))
        repeat = size // unit
        left = size % unit
        repeat_max_value = Constant.MAX_REPEAT_TIME
        repeat_loop = repeat // repeat_max_value
        repeat_left = repeat % repeat_max_value
        dst_rep_stride = unit // (Constant.UB_MINIMUM_SIZE // self.dtype_dict.get(dst_type))
        src_rep_stride = unit // (Constant.UB_MINIMUM_SIZE // self.dtype_dict.get(src_type))

        offset = 0
        with self.tik_instance.if_scope(repeat_loop > 0):
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * repeat_max_value * unit
                self.tik_instance.vec_conv(unit, round_mode, dst[offset], src[offset], repeat_max_value, dst_rep_stride,
                                           src_rep_stride)
        with self.tik_instance.if_scope(repeat_left > 0):
            offset = repeat_loop * repeat_max_value * unit
            self.tik_instance.vec_conv(unit, round_mode, dst[offset], src[offset], repeat_left, dst_rep_stride,
                                           src_rep_stride)
        with self.tik_instance.if_scope(left > 0):
            offset += repeat * unit
            self.tik_instance.vec_conv(left, round_mode, dst[offset], src[offset], 1, dst_rep_stride,
                                       src_rep_stride)

    # 'pylint: disable=too-many-locals,huawei-too-many-arguments
    def gen_div(self, dst, src, src1, size, dst_type):
        """make row matrix"""
        unit = Constant.VECTOR_BLOCK_SIZE // (self.dtype_dict.get(dst_type))
        repeat = size // unit
        left = size % unit
        repeat_max_value = Constant.MAX_REPEAT_TIME
        repeat_loop = repeat // repeat_max_value
        repeat_left = repeat % repeat_max_value

        offset = 0
        with self.tik_instance.if_scope(repeat_loop > 0):
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * repeat_max_value * unit
                self.tik_instance.vmuls(unit, dst[offset], src[offset], src1, repeat_max_value, 1, 1,
                                        Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
        with self.tik_instance.if_scope(repeat_left > 0):
            offset = repeat_loop * repeat_max_value * unit
            self.tik_instance.vmuls(unit, dst[offset], src[offset], src1, repeat_left, 1, 1,
                                    Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
        with self.tik_instance.if_scope(left > 0):
            offset += repeat * unit
            self.tik_instance.vmuls(left, dst[offset], src[offset], src1, 1, 1, 1,
                                    Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)

    # 'pylint: disable=too-many-locals,huawei-too-many-arguments
    def gen_sub(self, dst, src, src1, src2, src_scalar, size, dst_type):
        """make col matrix"""
        unit = Constant.VECTOR_BLOCK_SIZE // (self.dtype_dict.get(dst_type))
        repeat = size // unit
        left = size % unit
        repeat_max_value = Constant.MAX_REPEAT_TIME
        repeat_loop = repeat // repeat_max_value
        repeat_left = repeat % repeat_max_value

        offset = 0
        with self.tik_instance.if_scope(repeat_loop > 0):
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * repeat_max_value * unit
                self.tik_instance.vmuls(unit, src1[offset], src2[offset], src_scalar, repeat_max_value, 1, 1,
                                        Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
                self.tik_instance.vec_sub(unit, dst[offset], src[offset], src1[offset], repeat_max_value,
                                          Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
        with self.tik_instance.if_scope(repeat_left > 0):
            offset = repeat_loop * repeat_max_value * unit
            self.tik_instance.vmuls(unit, src1[offset], src2[offset], src_scalar, repeat_left, 1, 1,
                                    Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
            self.tik_instance.vec_sub(unit, dst[offset], src[offset], src1[offset], repeat_left,
                                      Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
        with self.tik_instance.if_scope(left > 0):
            offset += repeat * unit
            self.tik_instance.vmuls(left, src1[offset], src2[offset], src_scalar, 1, 1, 1,
                                    Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
            self.tik_instance.vec_sub(left, dst[offset], src[offset], src1[offset], 1,
                                      Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)

    def gen_mask(self, dst, src, scalar, size, x_dtype):
        """
        gen_mask
        in order to support nan, mask gen by follow algirthm
        y=not(x==0)
        """
        def gen_ne_mask(mask):
            # vnot only support uint16
            mask = mask.reinterpret_cast_to("uint16")
            unit = Constant.FP16_MASK
            mask_shape = mask.shape
            size = function_reduce(lambda x, y : x * y, mask_shape, 1)
            repeat = size // unit
            left = size % unit
            repeat_max_value = Constant.MAX_REPEAT_TIME
            repeat_loop = repeat // repeat_max_value
            repeat_left = repeat % repeat_max_value
            with self.tik_instance.if_scope(repeat_loop > 0):
                with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                    offset = rpt_idx * repeat_max_value * unit
                    self.tik_instance.vnot(unit, mask[offset], mask[offset], repeat_max_value, 1, 1,
                                           Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
            with self.tik_instance.if_scope(repeat_left > 0):
                offset = repeat_loop * repeat_max_value * unit
                self.tik_instance.vnot(unit, mask[offset], mask[offset], repeat_left, 1, 1,
                                       Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
            with self.tik_instance.if_scope(left > 0):
                offset += repeat * unit
                self.tik_instance.vnot(left, mask[offset], mask[offset], 1, 1, 1,
                                       Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)

        unit = Constant.VECTOR_BLOCK_SIZE // (self.dtype_dict.get(x_dtype))
        repeat = size // unit
        left = size % unit
        repeat_max_value = Constant.MAX_REPEAT_TIME
        repeat_loop = repeat // repeat_max_value
        repeat_left = repeat % repeat_max_value

        with self.tik_instance.if_scope(repeat_loop > 0):
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * repeat_max_value * unit
                self.tik_instance.vcmpvs_eq(dst[offset // Constant.UB_MINIMUM_SIZE], src[offset], scalar,
                                            repeat_max_value, 1, Constant.REPEAT_STRIDE)
        with self.tik_instance.if_scope(repeat_left > 0):
            offset = repeat_loop * repeat_max_value * unit
            self.tik_instance.vcmpvs_eq(dst[offset // Constant.UB_MINIMUM_SIZE], src[offset], scalar, repeat_left, 1,
                                        Constant.REPEAT_STRIDE)
        with self.tik_instance.if_scope(left > 0):
            offset = (repeat - 1) * Constant.UB_REPEAT_SIZE + left
            self.tik_instance.vcmpvs_eq(dst[offset // Constant.UB_MINIMUM_SIZE], src[offset], scalar, 1, 1,
                                        Constant.REPEAT_STRIDE)
        gen_ne_mask(dst)

    def trans(self, src_ub, dst_ub, length):
        """
        transpose for ub_data
        """
        repeat = _ceil(length, self.ub_minimum_num)
        repeat_loop = repeat // Constant.MAX_REPEAT_TIME
        repeat_left = repeat % Constant.MAX_REPEAT_TIME
        dst_rep_stride = 16
        src_rep_stride = 1
        with self.tik_instance.if_scope(repeat_loop > 0):
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * Constant.MAX_REPEAT_TIME * self.ub_minimum_num
                src_list = [src_ub[self.tiling * ((i + offset) % self.dim)] for i in range(Constant.FP16_ALIGH_NUM)]
                dst_list = [dst_ub[self.ub_minimum_num * (i + offset)] for i in range(Constant.FP16_ALIGH_NUM)]
                self.tik_instance.vnchwconv(False, False, dst_list, src_list, Constant.MAX_REPEAT_TIME, dst_rep_stride,
                                            src_rep_stride)
        with self.tik_instance.if_scope(repeat_left > 0):
            offset = repeat_loop * Constant.MAX_REPEAT_TIME * self.ub_minimum_num
            src_list = [src_ub[self.tiling * ((i + offset) % self.dim)] for i in range(Constant.FP16_ALIGH_NUM)]
            dst_list = [dst_ub[self.ub_minimum_num * (i + offset)] for i in range(Constant.FP16_ALIGH_NUM)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list, repeat_left, dst_rep_stride, src_rep_stride)

    # 'pylint: disable=too-many-locals,huawei-too-many-arguments
    def trans_v200(self, src_ub, dst_ub, tmp_ub_trans_s, tmp_ub_trans_ss, length):
        """
        transpose for ub_data on 910
        """
        repeat = _ceil(length, self.ub_minimum_num)
        dst_rep_stride = 16
        src_rep_stride = 1
        tmp_ub_fp16 = src_ub.reinterpret_cast_to("float16")
        tmp_ub_trans_fp16 = dst_ub.reinterpret_cast_to("float16")
        src_offset = Constant.TRANS_FP32_MODE * repeat * self.ub_minimum_num
        dst_offset = repeat * Constant.FP16_ALIGH_NUM * Constant.FP16_ALIGH_NUM
        # trans all data in one col
        with self.tik_instance.for_range(0, self.dim) as _idx:
            src_list = [tmp_ub_fp16[Constant.FP16_ALIGH_NUM * i + _idx * src_offset]\
                for i in range(Constant.FP16_ALIGH_NUM)]
            dst_list = [tmp_ub_trans_fp16[Constant.FP16_ALIGH_NUM * i + _idx * dst_offset]\
                 for i in range(Constant.FP16_ALIGH_NUM)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list, repeat, dst_rep_stride, src_rep_stride)
        # ub to ub reformat
        tmp_ub_trans_s_fp16 = tmp_ub_trans_s.reinterpret_cast_to("float16")
        with self.tik_instance.for_range(0, self.dim) as _idx:
            dst_rep = (self.dim - 1) * Constant.TRANS_FP32_MODE
            self.tik_instance.data_move(tmp_ub_trans_s_fp16[_idx * Constant.TRANS_FP32_MODE * Constant.FP16_ALIGH_NUM],\
                                        tmp_ub_trans_fp16[_idx * dst_offset], 0, length, 2, 0, dst_rep)
        # trans ub
        tmp_ub_trans_ss_fp16 = tmp_ub_trans_ss.reinterpret_cast_to("float16")
        with self.tik_instance.for_range(0, self.dim) as _idx:
            src_list = [tmp_ub_trans_s_fp16[Constant.FP16_ALIGH_NUM * i + _idx * dst_offset]\
                for i in range(Constant.FP16_ALIGH_NUM)]
            dst_list = [tmp_ub_trans_ss_fp16[Constant.FP16_ALIGH_NUM * i + _idx * src_offset]\
                 for i in range(Constant.FP16_ALIGH_NUM)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list, repeat, src_rep_stride, dst_rep_stride)

    def workspace_to_gm(self, _index, tmp_ub, shape_out_ub):
        """
        move data to GM
        """
        with self.tik_instance.for_range(0, self.core_num_var) as _idx:
            with self.tik_instance.if_scope(_idx == 0):
                self.offset_gm.set_as(0)
            # Calculate the offset of the current core output
            with self.tik_instance.else_scope():
                self.num_blk.set_as(shape_out_ub[_idx - 1, 0])
                self.offset_gm.set_as(self.offset_gm + self.num_blk)

            # The number of non-zeor elements in the current core
            self.num_blk.set_as(shape_out_ub[_idx, 0])
            mv_out_loop = self.num_blk // self.tiling
            mv_out_tail = self.num_blk % self.tiling
            with self.tik_instance.if_scope(mv_out_loop > 0):
                with self.tik_instance.for_range(0, mv_out_loop) as mvo_idx:
                    mvo_offset = mvo_idx * self.tiling
                    col_gm_offset = _idx * self.dim * self.out_last_dim + _index * self.out_last_dim + mvo_offset
                    # workspace to UB
                    # Align and move out according to tiling, no out of bounds behavior will occur
                    self.tik_instance.data_move(tmp_ub,
                                                self.data_out[col_gm_offset],
                                                sid=0,
                                                nburst=1,
                                                burst=self.tiling // self.ub_minimum_num,
                                                src_stride=0,
                                                dst_stride=0)
                    if self.y_dtype == "int64":
                        tmp_ub_cast = self.tik_instance.Tensor(self.y_dtype, (self.tiling,),
                                                               name="tmp_ub_cast", scope=tik.scope_ubuf)
                        self.gen_cast(tmp_ub_cast, tmp_ub, self.tiling, "int64", "int32")
                        # UB to GM
                        self.tik_instance.data_move(self.res_gm[_index * self.num + self.offset_gm + mvo_offset],
                                                    tmp_ub_cast,
                                                    sid=0,
                                                    nburst=1,
                                                    burst=self.tiling * 2 // self.ub_minimum_num,
                                                    src_stride=0,
                                                    dst_stride=0)                        
                    else:
                        # UB to GM
                        self.tik_instance.data_move(self.res_gm[_index * self.num + self.offset_gm + mvo_offset],
                                                    tmp_ub,
                                                    sid=0,
                                                    nburst=1,
                                                    burst=self.tiling // self.ub_minimum_num,
                                                    src_stride=0,
                                                    dst_stride=0)
            burst_ub = _ceil(mv_out_tail, self.ub_minimum_num)
            with self.tik_instance.if_scope(mv_out_tail > 0):
                mvo_offset = mv_out_loop * self.tiling
                col_gm_offset = _idx * self.dim * self.out_last_dim + _index * self.out_last_dim + mvo_offset
                # Tail block processing may result in read out of bounds
                if self.is_v200:
                    self.tik_instance.data_move_pad(tmp_ub, self.data_out[col_gm_offset], 1,
                                                    mv_out_tail * Constant.INT32_BYTE, 0, 0)
                else:
                    self.tik_instance.data_move(tmp_ub, self.data_out[col_gm_offset], 0, 1, burst_ub, 0, 0)
                out_gm_offset = _index * self.num + self.offset_gm + mvo_offset
                if self.y_dtype == "int64":
                    tmp_ub_cast = self.tik_instance.Tensor(self.y_dtype, (self.tiling,),
                                                            name="tmp_ub_cast", scope=tik.scope_ubuf)
                    self.gen_cast(tmp_ub_cast, tmp_ub, mv_out_tail, "int64", "int32")
                    # UB to GM
                    burst_ub_int64 = _ceil(mv_out_tail * 2, self.ub_minimum_num)
                    # Tail block processing may result in read out of bounds
                    if self.is_v200:
                        self.res_gm = self.res_gm[out_gm_offset].reinterpret_cast_to("int8")
                        tmp_ub_cast = tmp_ub_cast.reinterpret_cast_to("int8")
                        self.tik_instance.data_move_pad(self.res_gm,
                                                        tmp_ub_cast, 1, mv_out_tail * Constant.INT64_BYTE, 0, 0)
                    else:
                        self.tik_instance.data_move(self.res_gm[out_gm_offset],
                                                    tmp_ub_cast,
                                                    sid=0,
                                                    nburst=1,
                                                    burst=burst_ub_int64,
                                                    src_stride=0,
                                                    dst_stride=0)
                else:
                     # UB to GM
                    if self.is_v200:
                        self.tik_instance.data_move_pad(self.res_gm[out_gm_offset],
                                                        tmp_ub, 1, mv_out_tail * Constant.INT32_BYTE, 0, 0)
                    else:
                        self.tik_instance.data_move(self.res_gm[out_gm_offset],
                                                    tmp_ub, 0, 1, burst_ub, 0, 0)

    def trans_workspace_to_gm_with_pad(self, blk_idx, tmp_ub, tmp_ub_trans, shape_out_ub):
        """
        transpose on 910B
        """
        # Calculate the offset of the current core output
        with self.tik_instance.if_scope(blk_idx > 0):
            with self.tik_instance.for_range(0, blk_idx) as o_idx:
                self.num_blk.set_as(shape_out_ub[o_idx, 0])
                self.offset_gm.set_as(self.offset_gm + self.num_blk)
        # The number of non-zeor elements in the current core
        self.num_blk.set_as(shape_out_ub[blk_idx, 0])
        mv_out_loop = self.num_blk // self.tiling
        mv_out_tail = self.num_blk % self.tiling
        with self.tik_instance.if_scope(mv_out_loop > 0):
            with self.tik_instance.for_range(0, mv_out_loop) as mvo_idx:
                mvo_offset = mvo_idx * self.tiling
                with self.tik_instance.for_range(0, self.dim) as dim_idx:
                    col_gm_offset = blk_idx * self.dim * self.out_last_dim + dim_idx * self.out_last_dim + mvo_offset
                    # workspace to UB
                    self.tik_instance.data_move(tmp_ub[dim_idx, 0],
                                                self.data_out[col_gm_offset],
                                                sid=0,
                                                nburst=1,
                                                burst=self.tiling // self.ub_minimum_num,
                                                src_stride=0,
                                                dst_stride=0)
                self.trans(tmp_ub, tmp_ub_trans, self.tiling)
                self.tik_instance.data_move(tmp_ub_trans, tmp_ub_trans, 0, self.tiling, 1, 1, 0)
                # UB to GM
                out_gm_offset = self.dim * self.offset_gm + self.dim * mvo_offset
                self.tik_instance.data_move_pad(self.res_gm[out_gm_offset], tmp_ub_trans, nburst=self.tiling,
                                                burst=self.dim * 4, dst_gap=0, src_gap=0, left_padding=0,
                                                right_padding=0, padding_value=None)
        burst_ub = _ceil(mv_out_tail, self.ub_minimum_num)
        with self.tik_instance.if_scope(mv_out_tail > 0):
            mvo_offset = mv_out_loop * self.tiling
            with self.tik_instance.for_range(0, self.dim) as dim_idx:
                col_gm_offset = blk_idx * self.dim * self.out_last_dim + dim_idx * self.out_last_dim + mvo_offset
                # workspace to UB
                self.tik_instance.data_move(tmp_ub[dim_idx, 0],
                                            self.data_out[col_gm_offset],
                                            sid=0,
                                            nburst=1,
                                            burst=burst_ub,
                                            src_stride=0,
                                            dst_stride=0)
            self.trans(tmp_ub, tmp_ub_trans, mv_out_tail)
            # get valid data
            self.tik_instance.data_move(tmp_ub_trans, tmp_ub_trans, 0, mv_out_tail, 1, 1, 0)
            # ub to gm
            out_gm_offset = self.dim * self.offset_gm + self.dim * mvo_offset
            self.tik_instance.data_move_pad(self.res_gm[out_gm_offset], tmp_ub_trans, nburst=mv_out_tail,
                                            burst=self.dim * 4, dst_gap=0, src_gap=0, left_padding=0,
                                            right_padding=0, padding_value=None)

    def trans_workspace_to_gm(self, shape_out_ub):
        """"
        transpose on 910
        """
        with self.tik_instance.for_range(0, self.core_num_var) as _idx:
            # Calculate the offset of the current core output
            with self.tik_instance.if_scope(_idx == 0):
                self.offset_gm.set_as(0)
            with self.tik_instance.else_scope():
                self.num_blk.set_as(shape_out_ub[_idx - 1, 0])
                self.offset_gm.set_as(self.offset_gm + self.num_blk)
            # The number of non-zeor elements in the current core
            self.num_blk.set_as(shape_out_ub[_idx, 0])
            # when self.dim increasing ub_tiling reducing
            if self.y_dtype == Constant.Y_BASE_DTYPE:
                trans_num = _trans_ceil(self.dim, Constant.BLOCK_INT64)
                ub_tiling = int(Constant.TRANS_SIZE / trans_num)
            else:
                trans_num = _ceil(self.dim, Constant.BLOCK_INT64)
                ub_tiling = int(Constant.SIZE / trans_num)
            mv_out_loop = self.num_blk // ub_tiling
            mv_out_tail = self.num_blk % ub_tiling
            with self.tik_instance.if_scope(mv_out_loop > 0):
                tmp_ub = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (self.dim, ub_tiling), name="tmp_ub",\
                        scope=tik.scope_ubuf)
                tmp_ub_trans = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, \
                    (self.dim * ub_tiling, Constant.FP16_ALIGH_NUM), name="tmp_ub_trans", scope=tik.scope_ubuf)
                tmp_ub_trans_s = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE,\
                    (self.dim * ub_tiling, Constant.FP16_ALIGH_NUM), name="tmp_ub_trans_s", scope=tik.scope_ubuf)
                tmp_ub_trans_ss = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, \
                    (Constant.FP16_ALIGH_NUM, self.dim * ub_tiling), name="tmp_ub_trans_ss", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, mv_out_loop) as mvo_idx:
                    mvo_offset = mvo_idx * ub_tiling
                    with self.tik_instance.for_range(0, self.dim) as dim_idx:
                        col_gm_offset = _idx * self.dim * self.out_last_dim + dim_idx * self.out_last_dim + mvo_offset
                        # workspace to UB
                        # Align and move out according to tiling, no out of bounds behavior will occur
                        self.tik_instance.data_move(tmp_ub[dim_idx, 0],
                                                    self.data_out[col_gm_offset],
                                                    sid=0,
                                                    nburst=1,
                                                    burst=ub_tiling // self.ub_minimum_num,
                                                    src_stride=0,
                                                    dst_stride=0)
                    self.trans_v200(tmp_ub, tmp_ub_trans, tmp_ub_trans_s, tmp_ub_trans_ss, ub_tiling)
                    # UB to GM
                    out_gm_offset = self.dim * self.offset_gm + self.dim * mvo_offset
                    burst_num = (ub_tiling // self.ub_minimum_num) * self.dim
                    if self.y_dtype == "int64":
                        tmp_ub_cast = self.tik_instance.Tensor(self.y_dtype,
                                                               (Constant.FP16_ALIGH_NUM, self.dim * ub_tiling),
                                                               name="tmp_ub_cast", scope=tik.scope_ubuf)
                        self.gen_cast(tmp_ub_cast, tmp_ub_trans_ss, Constant.FP16_ALIGH_NUM * self.dim * ub_tiling,
                                      "int64", "int32")
                        burst_ub_int64 = (ub_tiling * 2 // self.ub_minimum_num) * self.dim
                        self.tik_instance.data_move(self.res_gm[out_gm_offset], tmp_ub_cast, 0, 1, burst_ub_int64, 0, 0)
                    else:
                        self.tik_instance.data_move(self.res_gm[out_gm_offset], tmp_ub_trans_ss, 0, 1, burst_num, 0, 0)
            burst_ub = _ceil(mv_out_tail, self.ub_minimum_num)
            align_ub = burst_ub * self.ub_minimum_num
            with self.tik_instance.if_scope(mv_out_tail > 0):
                tmp_ub = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (self.dim, align_ub), name="tmp_ub",\
                        scope=tik.scope_ubuf)
                tmp_ub_trans = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, \
                    (self.dim * align_ub, Constant.FP16_ALIGH_NUM), name="tmp_ub_trans", scope=tik.scope_ubuf)
                tmp_ub_trans_s = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE,\
                    (self.dim * align_ub, Constant.FP16_ALIGH_NUM), name="tmp_ub_trans_s", scope=tik.scope_ubuf)
                tmp_ub_trans_ss = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, \
                    (Constant.FP16_ALIGH_NUM, self.dim * align_ub), name="tmp_ub_trans_ss", scope=tik.scope_ubuf)
                mvo_offset = mv_out_loop * ub_tiling
                with self.tik_instance.for_range(0, self.dim) as dim_idx:
                    col_gm_offset = _idx * self.dim * self.out_last_dim + dim_idx * self.out_last_dim + mvo_offset
                    # workspace to UB
                    # Tail block processing may result in read out of bounds
                    if self.is_v200:
                        self.tik_instance.data_move_pad(tmp_ub[dim_idx, 0], self.data_out[col_gm_offset], 1,
                                                        mv_out_tail * Constant.INT32_BYTE, 0, 0)
                    else:
                        self.tik_instance.data_move(tmp_ub[dim_idx, 0],
                                                    self.data_out[col_gm_offset],
                                                    sid=0,
                                                    nburst=1,
                                                    burst=burst_ub,
                                                    src_stride=0,
                                                    dst_stride=0)
                self.trans_v200(tmp_ub, tmp_ub_trans, tmp_ub_trans_s, tmp_ub_trans_ss, mv_out_tail)
                # UB to GM
                out_gm_offset = self.dim * self.offset_gm + self.dim * mvo_offset
                if self.y_dtype == "int64":
                    tmp_ub_cast = self.tik_instance.Tensor(self.y_dtype,
                                                           (Constant.FP16_ALIGH_NUM, self.dim * align_ub),
                                                           name="tmp_ub_cast", scope=tik.scope_ubuf)
                    self.gen_cast(tmp_ub_cast, tmp_ub_trans_ss, Constant.FP16_ALIGH_NUM * self.dim * align_ub,
                                  "int64", "int32")
                    burst_ub_int64 = _ceil(mv_out_tail * 2, self.ub_minimum_num)
                    if self.is_v200:
                        self.res_gm = self.res_gm[out_gm_offset].reinterpret_cast_to("int8")
                        tmp_ub_cast = tmp_ub_cast.reinterpret_cast_to("int8")
                        self.tik_instance.data_move_pad(self.res_gm, tmp_ub_cast,
                                                        1, mv_out_tail * self.dim * Constant.INT64_BYTE, 0, 0)
                    else:
                        self.tik_instance.data_move(self.res_gm[out_gm_offset], tmp_ub_cast, 0, 1,
                                                    burst_ub_int64 * self.dim, 0, 0)
                else:
                    if self.is_v200:
                        self.tik_instance.data_move_pad(self.res_gm[out_gm_offset], tmp_ub_trans_ss, 1,
                                                        mv_out_tail * self.dim * Constant.INT32_BYTE, 0, 0)
                    else:
                        self.tik_instance.data_move(self.res_gm[out_gm_offset], tmp_ub_trans_ss, 0, 1,
                                                    burst_ub * self.dim, 0, 0)

    # 'pylint: disable=too-many-locals,too-many-statements
    def multi_core_sync(self, blk_idx, shape_out_ub):
        """
        multi_core_sync
        """
        # Calculate the number of non-zeor elements
        with self.tik_instance.for_range(0, self.core_num_var) as _idx:
            self.num_blk.set_as(shape_out_ub[_idx, 0])
            self.num.set_as(self.num + self.num_blk)
        with self.tik_instance.if_scope(blk_idx == 0):
            with self.tik_instance.for_range(0, self.dim) as _index:
                tmp_ub = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (self.tiling, ),
                                                  name="tmp_ub", scope=tik.scope_ubuf)
                self.workspace_to_gm(_index, tmp_ub, shape_out_ub)

    # 'pylint: disable=too-many-locals,too-many-statements
    def multi_core_sync_trans(self, blk_idx, shape_out_ub):
        """
        multi_core_sync
        """
        # Calculate the number of non-zeor elements
        with self.tik_instance.for_range(0, self.core_num_var) as _idx:
            self.num_blk.set_as(shape_out_ub[_idx, 0])
            self.num.set_as(self.num + self.num_blk)
        if self.dim == Constant.MAX_DIM:
            self.tiling = Constant.TRANS_SIZE
        tmp_ub = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (self.dim, self.tiling),
                                          name="tmp_ub", scope=tik.scope_ubuf)
        tmp_ub_trans = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (self.tiling, Constant.FP16_ALIGH_NUM), \
                                                name="tmp_ub_trans", scope=tik.scope_ubuf)
        if self.is_v200 and self.y_dtype == Constant.Y_BASE_DTYPE:
            self.trans_workspace_to_gm_with_pad(blk_idx, tmp_ub, tmp_ub_trans, shape_out_ub)
        else:
            with self.tik_instance.if_scope(blk_idx == 0):
                self.trans_workspace_to_gm(shape_out_ub)

    # 'pylint: disable=too-many-locals,huawei-too-many-arguments
    def row_index_matrix(self, dst_ub_col, dst_star, x_ub, blk_size, offset=0):
        """
        cal nonzero
        """
        col_add_scalar = self.tik_instance.Scalar(Constant.Y_BASE_DTYPE, name="col_add_scalar", init_value=offset)
        with self.tik_instance.for_range(0, blk_size) as _idx:
            with self.tik_instance.if_scope(x_ub[_idx] != 0):
                dst_ub_col[dst_star].set_as(col_add_scalar)
                dst_star.set_as(dst_star + 1)
            with self.tik_instance.if_scope(col_add_scalar == self.col - 1):
                col_add_scalar.set_as(0)
            with self.tik_instance.else_scope():
                col_add_scalar.set_as(col_add_scalar + 1)
        return dst_ub_col, dst_star

    # 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments
    def _build_col_index_mtr(self, col_auxiliary_matrix, blk_size, offset=0):
        """
        build col index matrix for input
        :return: col_auxiliary_matrix
        """
        col_add_tensor_tmp = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (Constant.UB_REPEAT_SIZE, ),
                                                      name="col_add_tensor_tmp", scope=tik.scope_ubuf)
        value_start = self.tik_instance.Scalar(Constant.Y_BASE_DTYPE, "value_start", init_value=offset)
        self.tik_instance.vadds(Constant.UB_REPEAT_SIZE, col_add_tensor_tmp, self.col_add_tensor, value_start, 1,
                                Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE,
                                Constant.REPEAT_STRIDE)
        col_loop = _ceil(blk_size, Constant.UB_REPEAT_SIZE)
        with self.tik_instance.for_range(0, col_loop) as _idx:
            add_scalar = _idx * Constant.UB_REPEAT_SIZE
            self.tik_instance.vadds(Constant.UB_REPEAT_SIZE, col_auxiliary_matrix[add_scalar], col_add_tensor_tmp,\
                add_scalar, 1, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE, \
                    Constant.REPEAT_STRIDE)
        return col_auxiliary_matrix

    # 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments
    def _build_row_hp_index_mtr(self, row_auxiliary_matrix, auxiliary_matrix_temp, auxiliary_matrix_temp_int32,
            row_auxiliary_matrix_int32, sub_auxiliary_matrix, col_auxiliary_matrix, blk_size, t_idx):
        """
        build col index matrix for input
        :return: col_auxiliary_matrix
        """
        col_add_tensor_tmp = self.tik_instance.Tensor(Constant.Y_BASE_DTYPE, (Constant.UB_REPEAT_SIZE, ),
                                                      name="col_add_tensor_tmp", scope=tik.scope_ubuf)
        value_start = self.tik_instance.Scalar(Constant.Y_BASE_DTYPE, "value_start", init_value=t_idx * self.tiling)
        self.tik_instance.vadds(Constant.UB_REPEAT_SIZE, col_add_tensor_tmp, self.col_add_tensor, value_start, 1,
                                Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE,
                                Constant.REPEAT_STRIDE)
        col_loop = _ceil(blk_size, Constant.UB_REPEAT_SIZE)
        with self.tik_instance.for_range(0, col_loop) as _idx:
            add_scalar = _idx * Constant.UB_REPEAT_SIZE
            self.tik_instance.vadds(Constant.UB_REPEAT_SIZE, auxiliary_matrix_temp_int32[add_scalar], \
                col_add_tensor_tmp, add_scalar, 1, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, \
                    Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
        div_scalar_1 = self.tik_instance.Scalar("float32", "div_scalar_1", init_value=1)
        div_scalar = self.tik_instance.Scalar("float32", "div_scalar", init_value=div_scalar_1/self.col)
        self.gen_cast(auxiliary_matrix_temp, auxiliary_matrix_temp_int32, blk_size, "float32", "int32", "round")
        self.gen_div(row_auxiliary_matrix, auxiliary_matrix_temp, div_scalar, blk_size, "float32")
        self.gen_cast(row_auxiliary_matrix_int32, row_auxiliary_matrix, blk_size, "int32", "float32", "floor")
        mul_scalar = self.tik_instance.Scalar(Constant.Y_BASE_DTYPE, "mul_scalar", init_value=self.col)
        self.gen_sub(col_auxiliary_matrix, auxiliary_matrix_temp_int32, sub_auxiliary_matrix,
                     row_auxiliary_matrix_int32, mul_scalar, blk_size, "float32")
        return row_auxiliary_matrix_int32, col_auxiliary_matrix


@tbe_register.register_param_generalization("NonZero")
def non_zero_generalization(x, y, transpose, kernel_name="non_zero", generalize_config=None):
    if generalize_config is None:
        generalize_config = None
    x["ori_shape"] = [-1] * len(x["ori_shape"])
    x["shape"] = x["ori_shape"]
    x["ori_range"] = [[1, -1] * len(x["ori_shape"])]
    x["range"] = x["ori_range"]

    y["ori_shape"] = [-1, -1]
    y["shape"] = y["ori_shape"]
    y["ori_range"] = [[1, -1], [0, -1]]
    y["range"] = y["ori_range"]

    return [[x, y, transpose]]


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
@register_operator("NonZero")
def non_zero(x, y, transpose, kernel_name="non_zero"):
    """
    return a 2-D tensor where each row is the index for a nonzero value

    Paramters
    ---------
    x: dict
        data of input, support float32, bfloat16, float16, bool 
    y: dict
        index of output
    kernel_name: str
        kernel_name, default value is "non_zero"

    Returns
    ---------
    tik_instance
    """
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    supported_dtypes = ("float32", "float16", "bool", "uint8", "int8", "bfloat16")
    para_check.check_dtype(x_dtype, supported_dtypes, param_name="x")
    if x_dtype in ("bool", "uint8"):
        x_dtype = "int8"
    y_dtype = y.get("dtype").lower()
    obj = NonZero(x_shape, x_dtype, y_dtype, transpose, kernel_name)
    tik_instance = obj.non_zero_compute()
    return tik_instance
