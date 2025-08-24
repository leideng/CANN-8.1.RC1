"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tabulate_fusion_grad
"""

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl import constant_util as constant


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-few-public-methods
class TabulateFusionGrad:
    """Function: use to calc tabulate fusion grad for all loc
    """

    NUM_2 = 2
    NUM_3 = 3
    NUM_4 = 4
    NUM_5 = 5
    NUM_6 = 6
    NUM_8 = 8
    NUM_16 = 16
    NUM_32 = 32
    NUM_64 = 64
    NUM_128 = 128
    MIN_FLOAT = -3.0e+38
    MIN_UB_SIZE = 240 * 1024
    TILING_ARG_NUM = 12

    def __init__(self, table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem,
                 split_count, split_index, kernel_name):
        """
        init TabulateFusionGrad.

        Parameters
        ----------
        table : dict. shape and dtype of input data table
        table_info : dict. shape and dtype of input data table_info
        em_x : dict. shape and dtype of input data em_x
        em_ : dict. shape and dtype of input data em
        dy_ : dict. shape and dtype of input data dy
        descriptor : dict. shape and dtype of input data descriptor
        dy_dem_x : dict. shape and dtype of output data dy_dem_x
        dy_dem : dict. shape and dtype of output data dy_dem
        kernel_name : str. cce kernel name

        Returns
        -------
        None
        """
        self.kernel_name = kernel_name
        self.tik_inst = tik.Tik()

        self.op_dtype = "float32"
        self.dtype_fp16 = "float16"

        self.split_count = split_count
        self.split_index = split_index

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        self.size_tile = self.NUM_64
        self.nei_tile = self.NUM_64

        if self.ub_size < self.MIN_UB_SIZE:
            self.nei_tile = self.NUM_32

        self.tile_size = self.size_tile * self.nei_tile

        self.loc = self.tik_inst.Scalar(dtype="int64", name="loc")
        self.nnei = self.tik_inst.Scalar(dtype="int64", name="nnei")
        self.size = self.tik_inst.Scalar(dtype="int64", name="last_layer_size")
        self.loc_offset = self.tik_inst.Scalar(dtype="int64", name="loc_offset")
        self.loc_split = self.tik_inst.Scalar(dtype="int64", name="loc_split")
        self.high_core_num = self.tik_inst.Scalar("int64", name="high_core_num")
        self.low_core_num = self.tik_inst.Scalar("int64", name="low_core_num")
        self.loc_per_high_core = self.tik_inst.Scalar("int64", name="loc_per_high_core")
        self.loc_per_low_core = self.tik_inst.Scalar("int64", name="loc_per_low_core")

        self.size_align64 = self.tik_inst.Scalar(dtype="int64", name="last_layer_size_align64")
        self.em_row_size = self.tik_inst.Scalar("int64", name="em_row_size")
        self.dy_row_size = self.tik_inst.Scalar("int64", name="dy_row_size")
        self.table_row_size = self.tik_inst.Scalar("int64", name="table_row_size")

        self.lower = self.tik_inst.Scalar(self.op_dtype, name="lower")
        self.upper = self.tik_inst.Scalar(self.op_dtype, name="upper")
        self._max = self.tik_inst.Scalar(self.op_dtype, name="_max")
        self.stride0 = self.tik_inst.Scalar(self.op_dtype, name="stride0")
        self.rec_stride0 = self.tik_inst.Scalar(self.op_dtype, name="rec_stride0")
        self.stride1 = self.tik_inst.Scalar(self.op_dtype, name="stride1")
        self.rec_stride1 = self.tik_inst.Scalar(self.op_dtype, name="rec_stride1")

        self.first_stride = self.tik_inst.Scalar(self.op_dtype, name="first_stride")
        self.tmp_scalar_int32 = self.tik_inst.Scalar("int32", name="tmp_scalar_int32")
        self.max_tbl_idx = self.tik_inst.Scalar(self.op_dtype, name="max_tbl_idx")

        self.size_batch_num = self.tik_inst.Scalar("int64", name="nei_batch_num")

        self.tiling_gm = None
        self.table_gm = None
        self.table_info_gm = None
        self.em_x_gm = None
        self.em_gm = None
        self.dy_gm = None
        self.descriptor_gm = None
        self.dy_dem_x_gm = None
        self.dy_dem_gm = None

        self.lower_ub = None
        self.upper_ub = None
        self.max_ub = None

        self.em_x = None
        self.offset = None
        self.em_x_tile = None
        self.dy_dem_x = None
        self.dy_dem = None

    def compute(self):
        """
        compute
        """
        self._init_tensor()

        loc_start = self.tik_inst.Scalar(init_value=0, dtype="int32")
        loc_end = self.tik_inst.Scalar(init_value=0, dtype="int32")

        with self.tik_inst.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_i:
            self._init_scalar_var()

            with self.tik_inst.if_scope(core_i < self.high_core_num):
                loc_start.set_as(self.loc_offset + core_i * self.loc_per_high_core)
                loc_end.set_as(loc_start + self.loc_per_high_core)
            with self.tik_inst.elif_scope(core_i < self.loc):
                loc_start.set_as(self.loc_offset + self.high_core_num + core_i * self.loc_per_low_core)
                loc_end.set_as(loc_start + self.loc_per_low_core)
            with self.tik_inst.else_scope():
                loc_start.set_as(0)
                loc_end.set_as(0)

            with self.tik_inst.for_range(loc_start, loc_end, name="loc") as loc_i:
                self._compute_loc_grad(loc_i)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.ai_core_num,
                                                            "split_count": self.split_count,
                                                            "split_index": self.split_index})

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.table_gm, self.table_info_gm, self.em_x_gm, self.em_gm,
                                       self.dy_gm, self.descriptor_gm],
                               outputs=[self.dy_dem_x_gm, self.dy_dem_gm],
                               flowtable=(self.tiling_gm,),
                               config=opt_config)

        return self.tik_inst

    def _init_tensor(self):
        """
        init gm/ub tensor
        """
        self.tiling_gm = self.tik_inst.Tensor("int64", (self.TILING_ARG_NUM,), name="tiling_gm",
                                              scope=tik.scope_gm)
        self.table_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                             name="table_gm", scope=tik.scope_gm)
        self.table_info_gm = self.tik_inst.Tensor(self.op_dtype, (self.NUM_8,),
                                                  name="table_info_gm", scope=tik.scope_gm)
        self.em_x_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                            name="em_x_gm", scope=tik.scope_gm)
        self.em_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                          name="em_gm", scope=tik.scope_gm)
        self.dy_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                          name="dy_gm", scope=tik.scope_gm)
        self.descriptor_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                                  name="descriptor_gm", scope=tik.scope_gm)
        self.dy_dem_x_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                                name="dy_dem_x_gm", scope=tik.scope_gm)
        self.dy_dem_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                              name="dy_dem_gm", scope=tik.scope_gm)

        self.em_x = self.tik_inst.Tensor(self.op_dtype, (self.NUM_64,), name="em_x", scope=tik.scope_ubuf)
        self.offset = self.tik_inst.Tensor("int32", (self.nei_tile,), name="offset", scope=tik.scope_ubuf)
        self.em_x_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="em_x_tile", scope=tik.scope_ubuf)
        self.dy_dem_x = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile,), name="dy_dem_x", scope=tik.scope_ubuf)
        self.dy_dem = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile * self.NUM_4,),
                                           name="dy_dem", scope=tik.scope_ubuf)

    def _init_scalar_var(self):
        tiling_ub = self.tik_inst.Tensor("int64", (self.TILING_ARG_NUM,), name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                self.TILING_ARG_NUM * constant.DATA_SIZE_EIGHT // constant.BLOCK_SIZE, 0, 0)

        self.loc.set_as(tiling_ub[0])
        self.nnei.set_as(tiling_ub[1])
        self.size.set_as(tiling_ub[2])
        self.loc_offset.set_as(tiling_ub[3])
        self.loc_split.set_as(tiling_ub[4])
        self.high_core_num.set_as(tiling_ub[5])
        self.low_core_num.set_as(tiling_ub[6])
        self.loc_per_high_core.set_as(tiling_ub[7])
        self.loc_per_low_core.set_as(tiling_ub[8])

        self.size_align64.set_as((self.size + self.size_tile - 1) // self.size_tile * self.size_tile)
        self.em_row_size.set_as(self.nnei * self.NUM_4)
        self.dy_row_size.set_as(self.size * self.NUM_4)
        self.table_row_size.set_as(self.size_align64 * self.NUM_6)

        table_info_ub = self.tik_inst.Tensor(self.op_dtype, (self.NUM_8,), name="table_info_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(table_info_ub, self.table_info_gm, 0, 1, 1, 0, 0)

        self.lower.set_as(table_info_ub[0])
        self.upper.set_as(table_info_ub[1])
        self._max.set_as(table_info_ub[2])
        self.stride0.set_as(table_info_ub[3])
        self.rec_stride0.set_as(1 / self.stride0)
        self.stride1.set_as(table_info_ub[4])
        self.rec_stride1.set_as(1 / self.stride1)

        self.first_stride.set_as((self.upper - self.lower) * self.rec_stride0)
        self.tik_inst.scalar_conv('floor', self.tmp_scalar_int32, self.first_stride)
        self.tik_inst.scalar_conv('none', self.first_stride, self.tmp_scalar_int32)

        self.max_tbl_idx.set_as((self._max - self.upper) * self.rec_stride1)
        self.tik_inst.scalar_conv('floor', self.tmp_scalar_int32, self.max_tbl_idx)
        self.tik_inst.scalar_conv('none', self.max_tbl_idx, self.tmp_scalar_int32)
        self.max_tbl_idx.set_as(self.first_stride + self.max_tbl_idx - 1)

        self.size_batch_num.set_as(self.size_align64 // self.size_tile)

        self.lower_ub = self.tik_inst.Tensor(self.op_dtype, (self.NUM_64,), name="lower_ub", scope=tik.scope_ubuf)
        self.upper_ub = self.tik_inst.Tensor(self.op_dtype, (self.NUM_64,), name="upper_ub", scope=tik.scope_ubuf)
        self.max_ub = self.tik_inst.Tensor(self.op_dtype, (self.NUM_64,), name="max_ub", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(self.NUM_64, self.lower_ub, self.lower, 1, 1, 8)
        self.tik_inst.vector_dup(self.NUM_64, self.upper_ub, self.upper, 1, 1, 8)
        self.tik_inst.vector_dup(self.NUM_64, self.max_ub, self._max, 1, 1, 8)

    def _locate_em_x(self, nei_mask, in_ub, out_ub):
        em_x = in_ub[0]
        em_x_new, offset = out_ub[0], out_ub[1]

        em_x_tmp = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile,), name="em_x_tmp", scope=tik.scope_ubuf)
        offset_fp32 = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile,), name="offset_fp32", scope=tik.scope_ubuf)
        table_idx_ub = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile,), name="table_idx_ub", scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(nei_mask, table_idx_ub, 0, 1, nei_mask // constant.REPEAT_STRIDE_EIGHT)

        # 'condition 1: x >= lower
        # 'table_offset = (x - lower) // s0
        self.tik_inst.vadds(nei_mask, em_x_tmp, em_x, -1 * self.lower, 1, 1, 1, 8, 8)
        self.tik_inst.vmuls(nei_mask, em_x_tmp, em_x_tmp, self.rec_stride0, 1, 1, 1, 8, 8)
        self.tik_inst.vconv(nei_mask, "floor", offset, em_x_tmp, 1, 1, 1, 8, 8)
        self.tik_inst.vconv(nei_mask, "none", offset_fp32, offset, 1, 1, 1, 8, 8)
        # 'x -= (table_offset * s0) + lower
        self.tik_inst.vmuls(nei_mask, em_x_tmp, offset_fp32, self.stride0, 1, 1, 1, 8, 8)
        self.tik_inst.vadds(nei_mask, em_x_tmp, em_x_tmp, self.lower, 1, 1, 1, 8, 8)
        self.tik_inst.vsub(nei_mask, em_x_tmp, em_x, em_x_tmp, 1, 1, 1, 1, 8, 8, 8)
        # 'mask and selection: x >= lower
        cmp_mask = self.tik_inst.vcmp_ge(self.NUM_64, em_x, self.lower_ub, 1, 1)
        self.tik_inst.vsel(nei_mask, 0, table_idx_ub, cmp_mask, offset_fp32, table_idx_ub, 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsel(nei_mask, 0, em_x_new, cmp_mask, em_x_tmp, em_x_new, 1, 1, 1, 1, 8, 8, 8)

        # 'condition 2: x >= upper
        # 'table_offset = (x - upper) // s1
        self.tik_inst.vadds(nei_mask, em_x_tmp, em_x, -1 * self.upper, 1, 1, 1, 8, 8)
        self.tik_inst.vmuls(nei_mask, em_x_tmp, em_x_tmp, self.rec_stride1, 1, 1, 1, 8, 8)
        self.tik_inst.vconv(nei_mask, "floor", offset, em_x_tmp, 1, 1, 1, 8, 8)
        self.tik_inst.vconv(nei_mask, "none", offset_fp32, offset, 1, 1, 1, 8, 8)
        # 'x -= (table_offset * s1) + upper
        self.tik_inst.vmuls(nei_mask, em_x_tmp, offset_fp32, self.stride1, 1, 1, 1, 8, 8)
        self.tik_inst.vadds(nei_mask, em_x_tmp, em_x_tmp, self.upper, 1, 1, 1, 8, 8)
        self.tik_inst.vsub(nei_mask, em_x_tmp, em_x, em_x_tmp, 1, 1, 1, 1, 8, 8, 8)
        # 'table_offset = table_offset + first_stride
        self.tik_inst.vadds(nei_mask, offset_fp32, offset_fp32, self.first_stride, 1, 1, 1, 8, 8)
        # 'mask and selection: x >= upper
        cmp_mask = self.tik_inst.vcmp_ge(self.NUM_64, em_x, self.upper_ub, 1, 1)
        self.tik_inst.vsel(nei_mask, 0, table_idx_ub, cmp_mask, offset_fp32, table_idx_ub, 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsel(nei_mask, 0, em_x_new, cmp_mask, em_x_tmp, em_x_new, 1, 1, 1, 1, 8, 8, 8)

        # 'condition 3: x >= max
        # 'table_offset = max_tbl_idx
        self.tik_inst.vec_dup(nei_mask, offset_fp32, self.max_tbl_idx, 1, 8)
        # 'x = 0
        self.tik_inst.vec_dup(nei_mask, em_x_tmp, 0, 1, 8)
        # 'mask and selection: x >= max
        cmp_mask = self.tik_inst.vcmp_ge(self.NUM_64, em_x, self.max_ub, 1, 1)
        self.tik_inst.vsel(nei_mask, 0, table_idx_ub, cmp_mask, offset_fp32, table_idx_ub, 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsel(nei_mask, 0, em_x_new, cmp_mask, em_x_tmp, em_x_new, 1, 1, 1, 1, 8, 8, 8)

        self.tik_inst.vconv(nei_mask, "floor", offset, table_idx_ub, 1, 1, 1, 8, 8)

    def _em_dy_dot_tile(self, loc, nei_mask, size_offset, in_ub, out_ub):
        em_bc = in_ub[0]
        em_dy_dot_tile, dy = out_ub[0], out_ub[1]
        loc_offset = self.tik_inst.Scalar("int64", name="loc_offset", init_value=loc * self.dy_row_size)

        self.tik_inst.data_move(dy, self.dy_gm[loc_offset + size_offset], 0, 1, 
                                self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)
        self.tik_inst.data_move(dy[self.size_tile], self.dy_gm[loc_offset + self.size + size_offset], 0, 1,
                                self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)
        self.tik_inst.data_move(dy[self.size_tile * self.NUM_2],
                                self.dy_gm[loc_offset + self.size * self.NUM_2 + size_offset], 0, 1,
                                self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)
        self.tik_inst.data_move(dy[self.size_tile * self.NUM_3],
                                self.dy_gm[loc_offset + self.size * self.NUM_3 + size_offset], 0, 1,
                                self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)

        self.tik_inst.vmul(self.size_tile, em_dy_dot_tile, em_bc, dy, nei_mask, 1, 1, 1,
                           self.size_tile // constant.REPEAT_STRIDE_EIGHT, 
                           self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
        self.tik_inst.vmla(self.size_tile, em_dy_dot_tile, em_bc[self.tile_size], dy[self.size_tile],
                           nei_mask, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT, 
                           self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
        self.tik_inst.vmla(self.size_tile, em_dy_dot_tile,
                           em_bc[self.tile_size * self.NUM_2], dy[self.size_tile * self.NUM_2],
                           nei_mask, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                           self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
        self.tik_inst.vmla(self.size_tile, em_dy_dot_tile,
                           em_bc[self.tile_size * self.NUM_3], dy[self.size_tile * self.NUM_3],
                           nei_mask, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                           self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)

    def _va_tile(self, nei_mask, size_offset, in_ub, out_ub):
        offset = in_ub[0]
        va_tile = out_ub[0]

        with self.tik_inst.for_range(0, nei_mask) as nei_idx:
            table_offset = self.tik_inst.Scalar(dtype="int32", init_value=offset[nei_idx])
            table_offset.set_as(table_offset * self.table_row_size)
            self.tik_inst.data_move(va_tile[nei_idx * self.size_tile],
                                    self.table_gm[table_offset + size_offset],
                                    0, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self.tik_inst.data_move(va_tile[self.tile_size + nei_idx * self.size_tile],
                                    self.table_gm[table_offset + self.size_align64 + size_offset],
                                    0, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self.tik_inst.data_move(va_tile[self.tile_size * self.NUM_2 + nei_idx * self.size_tile],
                                    self.table_gm[table_offset + self.size_align64 * self.NUM_2 + size_offset],
                                    0, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self.tik_inst.data_move(va_tile[self.tile_size * self.NUM_3 + nei_idx * self.size_tile],
                                    self.table_gm[table_offset + self.size_align64 * self.NUM_3 + size_offset],
                                    0, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self.tik_inst.data_move(va_tile[self.tile_size * self.NUM_4 + nei_idx * self.size_tile],
                                    self.table_gm[table_offset + self.size_align64 * self.NUM_4 + size_offset],
                                    0, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self.tik_inst.data_move(va_tile[self.tile_size * self.NUM_5 + nei_idx * self.size_tile],
                                    self.table_gm[table_offset + self.size_align64 * self.NUM_5 + size_offset],
                                    0, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)

    def _compute_tile(self, nei_mask, size_offset, in_ub, out_ub):
        em_x_tile, em_dy_dot_tile, va_tile, dy = in_ub[0], in_ub[1], in_ub[2], in_ub[3]
        dy_dem_x, dy_dem = out_ub[0], out_ub[1]

        dy_dem_tmp = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile * self.NUM_4,),
                                          name="dy_dem_tmp", scope=tik.scope_ubuf)
        dy_dem_x_tmp = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile,), name="dy_dem_x_tmp", scope=tik.scope_ubuf)

        # 'res = a5 * x5 + a4 * x4 + a3 * x3 + a2 * x2 + a1 * x + a0
        # 'grad = 5 * a5 * x4 + 4 * a4 * x3 + 3 * a3 * x2 + 2 * a2 * x + a1
        # 'vpx = x
        vpx = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vpx", scope=tik.scope_ubuf)
        self.tik_inst.vadds(self.size_tile, vpx, em_x_tile, 0, nei_mask, 1, 1, 8, 8)
        # 'res = res + a1 * x
        self.tik_inst.vmla(self.size_tile, va_tile, va_tile[self.tile_size], vpx, nei_mask, 1, 1, 1, 8, 8, 8)
        # 'vxa = 2 * a2
        vxa = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vxa", scope=tik.scope_ubuf)
        self.tik_inst.vmuls(self.size_tile, vxa, va_tile[self.tile_size * self.NUM_2], self.NUM_2, nei_mask, 1, 1, 8, 8)
        # 'grad = grad + 2 * a2 * x
        self.tik_inst.vmla(self.size_tile, va_tile[self.tile_size], vxa, vpx, nei_mask, 1, 1, 1, 8, 8, 8)
        # 'vpx = x2
        self.tik_inst.vmul(self.size_tile, vpx, vpx, em_x_tile, nei_mask, 1, 1, 1, 8, 8, 8)
        # 'res = res + a2 * x2
        self.tik_inst.vmla(self.size_tile, va_tile, va_tile[self.tile_size * self.NUM_2], vpx, nei_mask,
                           1, 1, 1, 8, 8, 8)
        # 'vxa = 3 * a3
        self.tik_inst.vmuls(self.size_tile, vxa, va_tile[self.tile_size * self.NUM_3], self.NUM_3, nei_mask, 1, 1, 8, 8)
        # 'grad = grad + 3 * a3 * x2
        self.tik_inst.vmla(self.size_tile, va_tile[self.tile_size], vxa, vpx, nei_mask, 1, 1, 1, 8, 8, 8)
        # 'vpx = x3
        self.tik_inst.vmul(self.size_tile, vpx, vpx, em_x_tile, nei_mask, 1, 1, 1, 8, 8, 8)
        # 'res = res + a3 * x3
        self.tik_inst.vmla(self.size_tile, va_tile, va_tile[self.tile_size * self.NUM_3], vpx, nei_mask,
                           1, 1, 1, 8, 8, 8)
        # 'vxa = self.NUM_4 * a4
        self.tik_inst.vmuls(self.size_tile, vxa, va_tile[self.tile_size * self.NUM_4], self.NUM_4, nei_mask, 1, 1, 8, 8)
        # 'grad = grad + 4 * a4 * x3
        self.tik_inst.vmla(self.size_tile, va_tile[self.tile_size], vxa, vpx, nei_mask, 1, 1, 1, 8, 8, 8)
        # 'vpx = x4
        self.tik_inst.vmul(self.size_tile, vpx, vpx, em_x_tile, nei_mask, 1, 1, 1, 8, 8, 8)
        # 'res = res + a4 * x4
        self.tik_inst.vmla(self.size_tile, va_tile, va_tile[self.tile_size * self.NUM_4], vpx, nei_mask,
                           1, 1, 1, 8, 8, 8)
        # 'vxa = 5 * a5
        self.tik_inst.vmuls(self.size_tile, vxa, va_tile[self.tile_size * self.NUM_5], self.NUM_5, nei_mask, 1, 1, 8, 8)
        # 'grad = grad + 5 * a5 * x4
        self.tik_inst.vmla(self.size_tile, va_tile[self.tile_size], vxa, vpx, nei_mask, 1, 1, 1, 8, 8, 8)
        # 'vpx = x5
        self.tik_inst.vmul(self.size_tile, vpx, vpx, em_x_tile, nei_mask, 1, 1, 1, 8, 8, 8)
        # 'res = res + a5 * x5
        self.tik_inst.vmla(self.size_tile, va_tile, va_tile[self.tile_size * self.NUM_5], vpx, nei_mask,
                           1, 1, 1, 8, 8, 8)
        # 'dy_dem_0 = res * rr0
        self.tik_inst.vmul(self.size_tile, va_tile[self.tile_size * self.NUM_2], va_tile, dy, nei_mask,
                           1, 1, 1, 8, 8, 0)
        # 'dy_dem_1 = res * rr1
        self.tik_inst.vmul(self.size_tile, va_tile[self.tile_size * self.NUM_3], va_tile, dy[self.size_tile], nei_mask,
                           1, 1, 1, 8, 8, 0)
        # 'dy_dem_2 = res * rr2
        self.tik_inst.vmul(self.size_tile, va_tile[self.tile_size * self.NUM_4], va_tile,
                           dy[self.size_tile * self.NUM_2], nei_mask, 1, 1, 1, 8, 8, 0)
        # 'dy_dem_3 = res * rr3
        self.tik_inst.vmul(self.size_tile, va_tile[self.tile_size * self.NUM_5], va_tile,
                           dy[self.size_tile * self.NUM_3], nei_mask, 1, 1, 1, 8, 8, 0)
        # 'dy_dem_4(grad) = grad * dy_dot
        self.tik_inst.vmul(self.size_tile, em_dy_dot_tile, va_tile[self.tile_size], em_dy_dot_tile,
                           nei_mask, 1, 1, 1, 8, 8, 8)

        with self.tik_inst.if_scope((self.size - size_offset) >= self.size_tile):
            self.tik_inst.vcadd(self.size_tile, dy_dem_tmp, va_tile[self.tile_size * self.NUM_2],
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)
            self.tik_inst.vcadd(self.size_tile, dy_dem_tmp[nei_mask], va_tile[self.tile_size * self.NUM_3],
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)
            self.tik_inst.vcadd(self.size_tile, dy_dem_tmp[nei_mask * self.NUM_2], va_tile[self.tile_size * self.NUM_4],
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)
            self.tik_inst.vcadd(self.size_tile, dy_dem_tmp[nei_mask * self.NUM_3], va_tile[self.tile_size * self.NUM_5],
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)
            self.tik_inst.vcadd(self.size_tile, dy_dem_x_tmp, em_dy_dot_tile,
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)
        with self.tik_inst.else_scope():
            self.tik_inst.vcadd(self.size - size_offset, dy_dem_tmp, va_tile[self.tile_size * self.NUM_2],
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)
            self.tik_inst.vcadd(self.size - size_offset, dy_dem_tmp[nei_mask], va_tile[self.tile_size * self.NUM_3],
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)
            self.tik_inst.vcadd(self.size - size_offset, dy_dem_tmp[nei_mask * self.NUM_2],
                                va_tile[self.tile_size * self.NUM_4],
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)
            self.tik_inst.vcadd(self.size - size_offset, dy_dem_tmp[nei_mask * self.NUM_3],
                                va_tile[self.tile_size * self.NUM_5],
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)
            self.tik_inst.vcadd(self.size - size_offset, dy_dem_x_tmp, em_dy_dot_tile,
                                nei_mask, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT)

        self.tik_inst.vadd(nei_mask, dy_dem, dy_dem, dy_dem_tmp, self.NUM_4, 1, 1, 1,
                           nei_mask // constant.REPEAT_STRIDE_EIGHT,
                           nei_mask // constant.REPEAT_STRIDE_EIGHT,
                           nei_mask // constant.REPEAT_STRIDE_EIGHT)
        self.tik_inst.vadd(nei_mask, dy_dem_x, dy_dem_x, dy_dem_x_tmp, 1, 1, 1, 1,
                           nei_mask // constant.REPEAT_STRIDE_EIGHT,
                           nei_mask // constant.REPEAT_STRIDE_EIGHT,
                           nei_mask // constant.REPEAT_STRIDE_EIGHT)

    def _vnchwconv_fp32_8_4_second(self, dst_ub_fp16, src_list):
        """
        the second step of transpose fp32 data from (8, 4) to (4, 8) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            dst_list = [dst_ub_fp16[16 * 4 * i] for i in range(16)]
            self.tik_inst.vnchwconv(False, False, dst_list, src_list, 4, 16 // 16, 16 * 2 // 16)

    def _vnchwconv_fp32_16_4_second(self, dst_ub_fp16, src_list):
        """
        the second step of transpose fp32 data from (16, 4) to (4, 16) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            dst_list = [dst_ub_fp16[0], dst_ub_fp16[16]]
            for i in range(2, 9):
                dst_list = dst_list + [dst_ub_fp16[64 * i], dst_ub_fp16[64 * i + 16]]
            self.tik_inst.vnchwconv(False, False, dst_list, src_list, 4, 16 * 2 // 16, 16 * 2 // 16)

    def _vnchwconv_fp32_32_4_second(self, dst_ub_fp16, src_list):
        """
        the second step of transpose fp32 data from (32, 4) to (4, 32) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            dst_list = [dst_ub_fp16[16 * i] for i in range(4)]
            dst_list = dst_list + [dst_ub_fp16[64 * 4 + 16 * i] for i in range(12)]
            self.tik_inst.vnchwconv(False, False, dst_list, src_list, 4, 16 * 4 // 16, 16 * 2 // 16)

    def _vnchwconv_fp32_64_4_second(self, dst_ub_fp16, src_list):
        """
        the second step of transpose fp32 data from (64, 4) to (4, 64) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            dst_list = [dst_ub_fp16[16 * i] for i in range(8)]
            dst_list = dst_list + [dst_ub_fp16[64 * 8 + 16 * i] for i in range(8)]
            self.tik_inst.vnchwconv(False, False, dst_list, src_list, 4, 16 * 8 // 16, 16 * 2 // 16)

    def _vnchwconv_fp32_8x_4_second(self, dst_ub_fp16, src_ub_fp16, nei_mask):
        """
        the second step of transpose fp32 data from (64/32/16/8, 4) to (4, 64/32/16/8) by using vnchwconv
        """
        src_list = []
        for i in range(8):
            src_list = src_list + [src_ub_fp16[128 * i], src_ub_fp16[128 * i + 16]]

        with self.tik_inst.if_scope(nei_mask == self.NUM_8):
            self._vnchwconv_fp32_8_4_second(dst_ub_fp16, src_list)

        with self.tik_inst.if_scope(nei_mask == self.NUM_16):
            self._vnchwconv_fp32_16_4_second(dst_ub_fp16, src_list)

        with self.tik_inst.if_scope(nei_mask == self.NUM_32):
            self._vnchwconv_fp32_32_4_second(dst_ub_fp16, src_list)

        with self.tik_inst.if_scope(nei_mask == self.NUM_64):
            self._vnchwconv_fp32_64_4_second(dst_ub_fp16, src_list)

    def _vnchwconv_fp32_8x_4(self, em_t, em, nei_mask):
        """
        transpose fp32 data from (64/32/16/8, 4) to (4, 64/32/16/8) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            trans_dst_ub = self.tik_inst.Tensor(self.dtype_fp16, (self.NUM_16 * self.NUM_64,),
                                                name="trans_dst_ub", scope=tik.scope_ubuf)
            trans_src_ub = self.tik_inst.Tensor(self.op_dtype, (self.NUM_16 * self.NUM_32,),
                                                name="trans_src_ub", scope=tik.scope_ubuf)
            self.tik_inst.vadds(64, trans_src_ub, em, 0, (nei_mask * 4 + 64 - 1) // 64, 1, 1, 8, 8)
            trans_ub_fp16 = trans_src_ub.reinterpret_cast_to(self.dtype_fp16)

            src_list0 = [trans_ub_fp16[16 * 4 * i] for i in range(16)]
            dst_list0 = [trans_dst_ub[16 * i] for i in range(16)]
            self.tik_inst.vnchwconv(False, False, dst_list0, src_list0, 4, 16 * 16 // 16, 16 // 16)

            self._vnchwconv_fp32_8x_4_second(trans_ub_fp16, trans_dst_ub, nei_mask)

            self.tik_inst.vadds(64, em_t, trans_src_ub, 0, (nei_mask * 4 + 64 - 1) // 64, 1, 1, 8, 8)

    def _trans_fp32_8x_4(self, em_t, em, nei_mask):
        """
        transpose fp32 data from (64/32/16/8, 4) to (4, 64/32/16/8)
        """
        if tbe_platform.api_check_support("tik.v4dtrans", "float32"):
            self.tik_inst.v4dtrans(False, em_t, em, nei_mask, self.NUM_4)
        else:
            self._vnchwconv_fp32_8x_4(em_t, em, nei_mask)

    def _vnchwconv_fp32_8_or_16_second(self, dst_ub_fp16, offset, trans_ub):
        """
        the second step of transpose fp32 data from (64, 8/16) to (8/16, 64) by using vnchwconv
        """
        src_list = []
        for i in range(8):
            src_list = src_list + [trans_ub[256 * i], trans_ub[256 * i + 16]]
        dst_list = [dst_ub_fp16[offset + 16 * i] for i in range(8)]
        dst_list = dst_list + [dst_ub_fp16[offset + 8 * 128 + 16 * i] for i in range(8)]
        self.tik_inst.vnchwconv(False, False, dst_list, src_list, 8, 8 * 16 // 16, 16 * 2 // 16)

    def _vnchwconv_fp32_64_8(self, dst_ub_fp16, offset, src_ub_fp16, trans_ub):
        """
        transpose fp32 data from (64, 8) to (8, 64) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            src_list0 = [src_ub_fp16[128 * i] for i in range(16)]
            dst_list0 = [trans_ub[16 * i] for i in range(16)]
            self.tik_inst.vnchwconv(False, False, dst_list0, src_list0, 8, 16 * 16 // 16, 16 // 16)

            self._vnchwconv_fp32_8_or_16_second(dst_ub_fp16, offset, trans_ub)

    def _vnchwconv_fp32_64_16(self, dst_ub_fp16, offset, src_ub_fp16, trans_ub):
        """
        transpose fp32 data from (64, 16) to (16, 64) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            src_list0 = [src_ub_fp16[256 * i] for i in range(8)]
            src_list0 = src_list0 + [src_ub_fp16[256 * i + 16] for i in range(8)]
            dst_list0 = [trans_ub[16 * i] for i in range(16)]
            self.tik_inst.vnchwconv(False, False, dst_list0, src_list0, 8, 16 * 16 // 16, 16 * 2 // 16)

            self._vnchwconv_fp32_8_or_16_second(dst_ub_fp16, offset, trans_ub)

    def _vnchwconv_fp32_32_or_64_second(self, dst_ub_fp16, offset, trans_ub, loop_idx):
        """
        the second step of transpose fp32 data from (64, 32/64) to (32/64, 64) by using vnchwconv
        """
        src_list = []
        for i in range(8):
            src_list = src_list + [trans_ub[256 * i], trans_ub[256 * i + 16]]
        dst_list = [dst_ub_fp16[offset + 16 * i + 16 * 64 * 2 * loop_idx] for i in range(8)]
        dst_list = dst_list + [dst_ub_fp16[offset + 8 * 128 + 16 * i + 16 * 64 * 2 * loop_idx] for i in range(8)]
        self.tik_inst.vnchwconv(False, False, dst_list, src_list, 8, 8 * 16 // 16, 16 * 2 // 16)

    def _vnchwconv_fp32_64_32(self, dst_ub_fp16, offset, src_ub_fp16, trans_ub):
        """
        transpose fp32 data from (64, 32) to (32, 64) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            with self.tik_inst.for_range(0, 2) as loop_idx:
                src_list0 = [src_ub_fp16[512 * i + 16 * 2 * loop_idx] for i in range(8)]
                src_list0 = src_list0 + [src_ub_fp16[512 * i + 16 + 16 * 2 * loop_idx] for i in range(8)]
                dst_list0 = [trans_ub[16 * i] for i in range(16)]
                self.tik_inst.vnchwconv(False, False, dst_list0, src_list0, 8, 16 * 16 // 16, 32 * 2 // 16)

                self._vnchwconv_fp32_32_or_64_second(dst_ub_fp16, offset, trans_ub, loop_idx)

    def _vnchwconv_fp32_64_64(self, dst_ub_fp16, offset, src_ub_fp16, trans_ub):
        """
        transpose fp32 data from (64, 64) to (64, 64) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            with self.tik_inst.for_range(0, 4) as loop_idx:
                src_list0 = [src_ub_fp16[1024 * i + 16 * 2 * loop_idx] for i in range(8)]
                src_list0 = src_list0 + [src_ub_fp16[1024 * i + 16 + 16 * 2 * loop_idx] for i in range(8)]
                dst_list0 = [trans_ub[16 * i] for i in range(16)]
                self.tik_inst.vnchwconv(False, False, dst_list0, src_list0, 8, 16 * 16 // 16, 64 * 2 // 16)

                self._vnchwconv_fp32_32_or_64_second(dst_ub_fp16, offset, trans_ub, loop_idx)

    def _vnchwconv_fp32_64_8x(self, dst_ub, offset, src_ub, nei_mask):
        """
        transpose fp32 data from (64, 64/32/16/8) to (64/32/16/8, 64) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            offset_fp16 = offset * 2
            src_ub_fp16 = src_ub.reinterpret_cast_to(self.dtype_fp16)
            dst_ub_fp16 = dst_ub.reinterpret_cast_to(self.dtype_fp16)
            trans_ub = self.tik_inst.Tensor(self.dtype_fp16, (self.NUM_16 * self.NUM_128,),
                                            name="trans_ub", scope=tik.scope_ubuf)

            with self.tik_inst.if_scope(nei_mask == self.NUM_8):
                self._vnchwconv_fp32_64_8(dst_ub_fp16, offset_fp16, src_ub_fp16, trans_ub)

            with self.tik_inst.if_scope(nei_mask == self.NUM_16):
                self._vnchwconv_fp32_64_16(dst_ub_fp16, offset_fp16, src_ub_fp16, trans_ub)

            with self.tik_inst.if_scope(nei_mask == self.NUM_32):
                self._vnchwconv_fp32_64_32(dst_ub_fp16, offset_fp16, src_ub_fp16, trans_ub)

            with self.tik_inst.if_scope(nei_mask == self.NUM_64):
                self._vnchwconv_fp32_64_64(dst_ub_fp16, offset_fp16, src_ub_fp16, trans_ub)

    def _trans_fp32_64_8x(self, em_bc, offset, em_bc_t, m_len, nei_mask):
        """
        transpose fp32 data from (64, 64/32/16/8) to (64/32/16/8, 64)
        """
        if tbe_platform.api_check_support("tik.v4dtrans", "float32"):
            self.tik_inst.v4dtrans(False, em_bc[offset], em_bc_t, m_len, nei_mask)
        else:
            self._vnchwconv_fp32_64_8x(em_bc, offset, em_bc_t, nei_mask)

    def _em_load(self, loc, nei_offset, nei_mask, in_ub, out_ub):
        em_bc = out_ub[0]

        with self.tik_inst.new_stmt_scope(disable_sync=False):
            em = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile * self.NUM_4,), name="em", scope=tik.scope_ubuf)
            em_tmp = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile * self.NUM_4,),
                                          name="em_tmp", scope=tik.scope_ubuf)
            em_bc_tmp = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="em_bc_tmp", scope=tik.scope_ubuf)

            self.tik_inst.data_move(em, self.em_gm[loc * self.em_row_size + nei_offset * self.NUM_4],
                                    0, 1, (nei_mask * self.NUM_4) // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self._trans_fp32_8x_4(em_tmp, em, nei_mask)

            self.tik_inst.vadds(nei_mask, em_bc_tmp, em_tmp, 0, self.size_tile, 1, 1,
                                nei_mask // constant.REPEAT_STRIDE_EIGHT, 0)
            self._trans_fp32_64_8x(em_bc, 0, em_bc_tmp, self.size_tile, nei_mask)

            self.tik_inst.vadds(nei_mask, em_bc_tmp, em_tmp[nei_mask], 0, self.size_tile, 1, 1,
                                nei_mask // constant.REPEAT_STRIDE_EIGHT, 0)
            self._trans_fp32_64_8x(em_bc, self.tile_size, em_bc_tmp, self.size_tile, nei_mask)

            self.tik_inst.vadds(nei_mask, em_bc_tmp, em_tmp[nei_mask * self.NUM_2], 0, self.size_tile, 1, 1,
                                nei_mask // constant.REPEAT_STRIDE_EIGHT, 0)
            self._trans_fp32_64_8x(em_bc, self.tile_size * self.NUM_2, em_bc_tmp, self.size_tile, nei_mask)

            self.tik_inst.vadds(nei_mask, em_bc_tmp, em_tmp[nei_mask * self.NUM_3], 0, self.size_tile, 1, 1,
                                nei_mask // constant.REPEAT_STRIDE_EIGHT, 0)
            self._trans_fp32_64_8x(em_bc, self.tile_size * self.NUM_3, em_bc_tmp, self.size_tile, nei_mask)

    def _vnchwconv_fp32_4_8x(self, dy_dem_out, dem_update, nei_mask):
        """
        transpose fp32 data from (4, 64/32/16/8) to (64/32/16/8, 4) by using vnchwconv
        """
        with self.tik_inst.new_stmt_scope():
            trans_dst_ub = self.tik_inst.Tensor(self.dtype_fp16, (self.NUM_128 * self.NUM_2 * self.NUM_4,),
                                                name="trans_dst_ub", scope=tik.scope_ubuf)
            trans_src_ub = self.tik_inst.Tensor(self.op_dtype, (self.NUM_128 * self.NUM_4,),
                                                name="trans_src_ub", scope=tik.scope_ubuf)
            self.tik_inst.vadds(nei_mask, trans_src_ub, dem_update, 0, 4, 1, 1, 128 // 8, nei_mask // 8)
            trans_ub_fp16 = trans_src_ub.reinterpret_cast_to(self.dtype_fp16)

            src_list0 = [trans_ub_fp16[16 * i] for i in range(16)]
            dst_list0 = [trans_dst_ub[16 * i] for i in range(16)]
            self.tik_inst.vnchwconv(False, False, dst_list0, src_list0, 4, 16 * 16 // 16, 128 * 2 // 16)

            src_list1 = []
            for i in range(4):
                src_list1 = src_list1 + [trans_dst_ub[256 * i], trans_dst_ub[256 * i + 16]]
            for i in range(4):
                src_list1 = src_list1 + [trans_dst_ub[256 * i + 32], trans_dst_ub[256 * i + 48]]
            dst_list1 = [trans_ub_fp16[16 * 4 * i] for i in range(16)]
            self.tik_inst.vnchwconv(False, False, dst_list1, src_list1, 4, 16 // 16, 16 * 4 // 16)

            self.tik_inst.vadds(32, dy_dem_out, trans_src_ub, 0, nei_mask // 8, 1, 1, 32 // 8, 32 // 8)

    def _trans_fp32_4_8x(self, dy_dem_out, dem_update, nei_mask):
        """
        transpose fp32 data from (4, 64/32/16/8) to (64/32/16/8, 4)
        """
        if tbe_platform.api_check_support("tik.v4dtrans", "float32"):
            self.tik_inst.v4dtrans(True, dy_dem_out, dem_update, nei_mask, self.NUM_4)
        else:
            self._vnchwconv_fp32_4_8x(dy_dem_out, dem_update, nei_mask)

    def _process_last_nei(self, loc, nei_offset, nei_mask, loop_break):
        last_nei = self.tik_inst.Scalar(self.op_dtype, name="last_nei")
        em_x_block_ub = self.tik_inst.Tensor(self.op_dtype, (self.NUM_8,), name="em_x_block_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(em_x_block_ub, self.em_x_gm[(loc + 1) * self.nnei - 1], 0, 1, 1, 0, 0)
        last_nei.set_as(em_x_block_ub[0])

        dy_dem_out = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile * self.NUM_4,),
                                          name="dy_dem_out", scope=tik.scope_ubuf)
        last_nei_ub = self.tik_inst.Tensor(self.op_dtype, (self.NUM_64,), name="last_nei_ub", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(self.NUM_64, last_nei_ub, last_nei, 1, 1, 8)

        with self.tik_inst.if_scope(self.em_x[nei_mask - 1] == last_nei):
            cmp_mask = self.tik_inst.Tensor("uint64", (1,), name="cmp_mask", scope=tik.scope_ubuf)
            s_num_bit1 = self.tik_inst.Scalar("uint64", name="s_num_bit1")
            self.tik_inst.vcmpv_eq(cmp_mask, self.em_x, last_nei_ub, 1, 1, 1, 8, 0)
            s_cmp_mask = self.tik_inst.Scalar("uint64", name="s_cmp_mask", init_value=cmp_mask[0])
            self.tik_inst.scalar_countbit1(s_num_bit1, s_cmp_mask)
            s_num_bit1.set_as(nei_mask - s_num_bit1)
            times = self.tik_inst.Scalar("int64", name="times",
                                         init_value=self.nnei - (nei_offset + s_num_bit1))
            last_nei_update = self.tik_inst.Scalar(self.op_dtype, name="last_nei_update",
                                                   init_value=self.dy_dem_x[s_num_bit1])
            self.dy_dem_x[s_num_bit1].set_as(last_nei_update * times)
            dem_x_update = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile,), name="dem_x_update",
                                                scope=tik.scope_ubuf)
            self.tik_inst.vec_dup(nei_mask, dem_x_update, 0, 1, nei_mask // constant.REPEAT_STRIDE_EIGHT)
            self.tik_inst.vadd(s_num_bit1 + 1, dem_x_update, dem_x_update, self.dy_dem_x, 1, 1, 1, 1, 8, 8, 8)

            dem_update = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile * self.NUM_4,), name="dem_update",
                                              scope=tik.scope_ubuf)
            self.tik_inst.vec_dup(nei_mask, dem_update, 0, self.NUM_4, nei_mask // constant.REPEAT_STRIDE_EIGHT)

            last_nei_update.set_as(self.dy_dem[s_num_bit1])
            self.dy_dem[s_num_bit1].set_as(last_nei_update * times)
            last_nei_update.set_as(self.dy_dem[s_num_bit1 + nei_mask])
            self.dy_dem[s_num_bit1 + nei_mask].set_as(last_nei_update * times)
            last_nei_update.set_as(self.dy_dem[s_num_bit1 + nei_mask * self.NUM_2])
            self.dy_dem[s_num_bit1 + nei_mask * self.NUM_2].set_as(last_nei_update * times)
            last_nei_update.set_as(self.dy_dem[s_num_bit1 + nei_mask * self.NUM_3])
            self.dy_dem[s_num_bit1 + nei_mask * self.NUM_3].set_as(last_nei_update * times)

            self.tik_inst.vadd(s_num_bit1 + 1, dem_update, dem_update,
                               self.dy_dem, 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(s_num_bit1 + 1, dem_update[nei_mask], dem_update[nei_mask],
                               self.dy_dem[nei_mask], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(s_num_bit1 + 1, dem_update[nei_mask * self.NUM_2], dem_update[nei_mask * self.NUM_2],
                               self.dy_dem[nei_mask * self.NUM_2], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(s_num_bit1 + 1, dem_update[nei_mask * self.NUM_3], dem_update[nei_mask * self.NUM_3],
                               self.dy_dem[nei_mask * self.NUM_3], 1, 1, 1, 1, 8, 8, 8)

            self._trans_fp32_4_8x(dy_dem_out, dem_update, nei_mask)
            self.tik_inst.data_move(self.dy_dem_gm[(loc - self.loc_offset) * self.em_row_size +
                                                   nei_offset * self.NUM_4],
                                    dy_dem_out, 0, 1, (nei_mask * self.NUM_4) // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self.tik_inst.data_move(self.dy_dem_x_gm[(loc - self.loc_offset) * self.nnei + nei_offset],
                                    dem_x_update, 0, 1, nei_mask // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            loop_break.set_as(1)
        with self.tik_inst.else_scope():
            self._trans_fp32_4_8x(dy_dem_out, self.dy_dem, nei_mask)
            self.tik_inst.data_move(self.dy_dem_gm[(loc - self.loc_offset) * self.em_row_size
                                                   + nei_offset * self.NUM_4],
                                    dy_dem_out, 0, 1, (nei_mask * self.NUM_4) // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self.tik_inst.data_move(self.dy_dem_x_gm[(loc - self.loc_offset) * self.nnei + nei_offset],
                                    self.dy_dem_x, 0, 1, nei_mask // constant.REPEAT_STRIDE_EIGHT, 0, 0)

    def _process_last_nei_tail(self, loc):
        last_nei = self.tik_inst.Scalar(self.op_dtype, name="last_nei", init_value=self.em_x[7])
        dy_dem_out = self.tik_inst.Tensor(self.op_dtype, (self.NUM_32,), name="dy_dem_out", scope=tik.scope_ubuf)
        last_nei_ub = self.tik_inst.Tensor(self.op_dtype, (self.NUM_64,), name="last_nei_ub", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(self.NUM_64, last_nei_ub, last_nei, 1, 1, 8)

        cmp_mask = self.tik_inst.Tensor("uint64", (1,), name="cmp_mask", scope=tik.scope_ubuf)
        s_num_bit1 = self.tik_inst.Scalar("uint64", name="s_num_bit1")
        self.tik_inst.vcmpv_eq(cmp_mask, self.em_x, last_nei_ub, 1, 1, 1, 8, 0)
        s_cmp_mask = self.tik_inst.Scalar("uint64", name="s_cmp_mask", init_value=cmp_mask[0])
        self.tik_inst.scalar_countbit1(s_num_bit1, s_cmp_mask)
        times = self.tik_inst.Scalar("int64", name="times", init_value=s_num_bit1)
        s_num_bit1.set_as(self.NUM_8 - s_num_bit1)

        last_nei_update = self.tik_inst.Scalar(self.op_dtype, name="last_nei_update",
                                               init_value=self.dy_dem_x[s_num_bit1])
        self.dy_dem_x[s_num_bit1].set_as(last_nei_update * times)
        dem_x_update = self.tik_inst.Tensor(self.op_dtype, (self.NUM_8,), name="dem_x_update", scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(self.NUM_8, dem_x_update, 0, 1, 1)
        self.tik_inst.vadd(s_num_bit1 + 1, dem_x_update, dem_x_update, self.dy_dem_x, 1, 1, 1, 1, 8, 8, 8)

        dem_update = self.tik_inst.Tensor(self.op_dtype, (self.NUM_8 * self.NUM_4,),
                                          name="dem_update", scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(self.NUM_8, dem_update, 0, self.NUM_4, 1)

        last_nei_update.set_as(self.dy_dem[s_num_bit1])
        self.dy_dem[s_num_bit1].set_as(last_nei_update * times)
        last_nei_update.set_as(self.dy_dem[s_num_bit1 + self.NUM_8])
        self.dy_dem[s_num_bit1 + self.NUM_8].set_as(last_nei_update * times)
        last_nei_update.set_as(self.dy_dem[s_num_bit1 + self.NUM_8 * self.NUM_2])
        self.dy_dem[s_num_bit1 + self.NUM_8 * self.NUM_2].set_as(last_nei_update * times)
        last_nei_update.set_as(self.dy_dem[s_num_bit1 + self.NUM_8 * self.NUM_3])
        self.dy_dem[s_num_bit1 + self.NUM_8 * self.NUM_3].set_as(last_nei_update * times)

        self.tik_inst.vadd(s_num_bit1 + 1, dem_update, dem_update,
                           self.dy_dem, 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(s_num_bit1 + 1, dem_update[self.NUM_8], dem_update[self.NUM_8],
                           self.dy_dem[self.NUM_8], 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(s_num_bit1 + 1, dem_update[self.NUM_8 * self.NUM_2], dem_update[self.NUM_8 * self.NUM_2],
                           self.dy_dem[self.NUM_8 * self.NUM_2], 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(s_num_bit1 + 1, dem_update[self.NUM_8 * self.NUM_3], dem_update[self.NUM_8 * self.NUM_3],
                           self.dy_dem[self.NUM_8 * self.NUM_3], 1, 1, 1, 1, 8, 8, 8)

        self._trans_fp32_4_8x(dy_dem_out, dem_update, self.NUM_8)

        self.tik_inst.data_move(self.dy_dem_gm[(loc - self.loc_offset) * self.em_row_size
                                               + (self.nnei - self.NUM_8) * self.NUM_4],
                                dy_dem_out, 0, 1, self.NUM_4, 0, 0)
        self.tik_inst.data_move(self.dy_dem_x_gm[(loc - self.loc_offset) * self.nnei + self.nnei - self.NUM_8],
                                dem_x_update, 0, 1, 1, 0, 0)

    def _compute_kernel(self, loc, nei_offset, nei_mask):
        self.tik_inst.vec_dup(self.nei_tile, self.em_x, self.MIN_FLOAT, 1,
                              self.nei_tile // constant.REPEAT_STRIDE_EIGHT)
        self.tik_inst.data_move(self.em_x, self.em_x_gm[loc * self.nnei + nei_offset], 0, 1,
                                nei_mask // constant.REPEAT_STRIDE_EIGHT, 0, 0)
        with self.tik_inst.new_stmt_scope(disable_sync=False):
            em_x_new = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile,), name="em_x_new", scope=tik.scope_ubuf)
            self._locate_em_x(nei_mask, in_ub=[self.em_x], out_ub=[em_x_new, self.offset])

            em_x_bc = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="em_x_bc",
                                           scope=tik.scope_ubuf)
            self.tik_inst.vadds(nei_mask, em_x_bc, em_x_new, 0, self.size_tile, 1, 1,
                                nei_mask // constant.REPEAT_STRIDE_EIGHT, 0)
            self._trans_fp32_64_8x(self.em_x_tile, 0, em_x_bc, self.size_tile, nei_mask)

        self.tik_inst.vec_dup(nei_mask, self.dy_dem_x, 0, 1, nei_mask // constant.REPEAT_STRIDE_EIGHT)
        self.tik_inst.vec_dup(nei_mask, self.dy_dem, 0, self.NUM_4, nei_mask // constant.REPEAT_STRIDE_EIGHT)

        size_offset = self.tik_inst.Scalar("int64", name="size_offset")
        with self.tik_inst.new_stmt_scope(disable_sync=False):
            em_bc = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * self.NUM_4,),
                                         name="em_bc", scope=tik.scope_ubuf)
            self._em_load(loc, nei_offset, nei_mask, in_ub=None, out_ub=[em_bc])

            # last_layer_size tile process
            with self.tik_inst.for_range(0, self.size_batch_num) as size_batch:
                size_offset.set_as(size_batch * self.size_tile)

                em_dy_dot_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="em_dy_dot_tile",
                                                      scope=tik.scope_ubuf)
                dy = self.tik_inst.Tensor(self.op_dtype, (self.size_tile * self.NUM_4,),
                                          name="dy", scope=tik.scope_ubuf)
                self._em_dy_dot_tile(loc, nei_mask, size_offset, in_ub=[em_bc], out_ub=[em_dy_dot_tile, dy])

                va_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * self.NUM_6,), name="va_tile",
                                               scope=tik.scope_ubuf)
                self._va_tile(nei_mask, size_offset, in_ub=[self.offset], out_ub=[va_tile])
                self._compute_tile(nei_mask, size_offset, in_ub=[self.em_x_tile, em_dy_dot_tile, va_tile, dy],
                                   out_ub=[self.dy_dem_x, self.dy_dem])

    def _compute_loc_grad(self, loc):
        """
        compute grad loc by loc
        """
        nei_batch_max = self.tik_inst.Scalar("int32", name="nei_batch_max",
                                             init_value=(self.nnei + self.NUM_8 - 1) // self.NUM_8)
        loop_break = self.tik_inst.Scalar("int32", name="loop_break", init_value=0)
        nei_offset = self.tik_inst.Scalar("int32", name="nei_offset", init_value=0)
        nei_mask = self.tik_inst.Scalar("int32", name="nei_mask", init_value=0)

        with self.tik_inst.for_range(0, nei_batch_max):
            with self.tik_inst.if_scope(loop_break == 0):
                if self.nei_tile == self.NUM_64:
                    with self.tik_inst.if_scope(self.nnei >= nei_offset + self.NUM_64):
                        nei_mask.set_as(self.NUM_64)
                    with self.tik_inst.elif_scope(self.nnei >= nei_offset + self.NUM_32):
                        nei_mask.set_as(self.NUM_32)
                    with self.tik_inst.elif_scope(self.nnei >= nei_offset + self.NUM_16):
                        nei_mask.set_as(self.NUM_16)
                    with self.tik_inst.elif_scope(self.nnei >= nei_offset + self.NUM_8):
                        nei_mask.set_as(self.NUM_8)
                    with self.tik_inst.elif_scope(self.nnei > nei_offset):
                        nei_mask.set_as(self.NUM_8)
                        nei_offset.set_as(self.nnei - self.NUM_8)
                        loop_break.set_as(1)
                    with self.tik_inst.else_scope():
                        loop_break.set_as(1)
                else:
                    with self.tik_inst.if_scope(self.nnei >= nei_offset + self.NUM_32):
                        nei_mask.set_as(self.NUM_32)
                    with self.tik_inst.elif_scope(self.nnei >= nei_offset + self.NUM_16):
                        nei_mask.set_as(self.NUM_16)
                    with self.tik_inst.elif_scope(self.nnei >= nei_offset + self.NUM_8):
                        nei_mask.set_as(self.NUM_8)
                    with self.tik_inst.elif_scope(self.nnei > nei_offset):
                        nei_mask.set_as(self.NUM_8)
                        nei_offset.set_as(self.nnei - self.NUM_8)
                        loop_break.set_as(1)
                    with self.tik_inst.else_scope():
                        loop_break.set_as(1)

                with self.tik_inst.if_scope(self.nnei > nei_offset):
                    self._compute_kernel(loc, nei_offset, nei_mask)
                    with self.tik_inst.if_scope(self.nnei >= nei_offset + self.NUM_8):
                        self._process_last_nei(loc, nei_offset, nei_mask, loop_break)
                    with self.tik_inst.else_scope():
                        self._process_last_nei_tail(loc)
                    nei_offset.set_as(nei_offset + nei_mask)


def _check_params(table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem, split_count,
                  split_index, kernel_name):
    # input dtype check
    para_check.check_dtype(table.get("dtype").lower(), ("float32",), param_name="table")
    para_check.check_dtype(table_info.get("dtype").lower(), ("float32",), param_name="table_info")
    para_check.check_dtype(em_x.get("dtype").lower(), ("float32",), param_name="em_x")
    para_check.check_dtype(em_.get("dtype").lower(), ("float32",), param_name="em")
    para_check.check_dtype(dy_.get("dtype").lower(), ("float32",), param_name="dy")
    para_check.check_dtype(descriptor.get("dtype").lower(), ("float32",), param_name="descriptor")

    # output dtype check
    para_check.check_dtype(dy_dem_x.get("dtype").lower(), ("float32",), param_name="dy_dem_x")
    para_check.check_dtype(dy_dem.get("dtype").lower(), ("float32",), param_name="dy_dem")

    # input shape check
    para_check.check_shape(table.get("shape"), min_rank=2, max_rank=2, param_name="table")
    para_check.check_shape(table_info.get("shape"), min_rank=1, max_rank=1, min_size=6, max_size=6,
                           param_name="table_info")
    para_check.check_shape(em_x.get("shape"), min_rank=2, max_rank=2, param_name="em_x")
    para_check.check_shape(em_.get("shape"), min_rank=3, max_rank=3, param_name="em")
    para_check.check_shape(dy_.get("shape"), min_rank=3, max_rank=3, param_name="dy")
    para_check.check_shape(descriptor.get("shape"), min_rank=3, max_rank=3, param_name="descriptor")

    # output shape check
    para_check.check_shape(dy_dem_x.get("shape"), min_rank=2, max_rank=2, param_name="dy_dem_x")
    para_check.check_shape(dy_dem.get("shape"), min_rank=3, max_rank=3, param_name="dy_dem")

    if any((split_count < 1, split_index < 0, split_count <= split_index)):
        rule = "Failed to check split info"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)


@register_operator("TabulateFusionGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def tabulate_fusion_grad(table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem, split_count=1, split_index=0,
                         kernel_name="tabulate_fusion_grad"):
    """
    Compute TabulateFusionGrad.

    Parameters
    ----------
    table : dict. shape and dtype of input data table
    table_info : dict. shape and dtype of input data table_info
    em_x : dict. shape and dtype of input data em_x
    em_ : dict. shape and dtype of input data em
    dy_ : dict. shape and dtype of input data dy
    descriptor : dict. shape and dtype of input data descriptor
    dy_dem_x : dict. shape and dtype of output data dy_dem_x
    dy_dem : dict. shape and dtype of output data dy_dem
    split_count : int. enable/disable vector core. 1-disable, 2-enable
    split_index : int. index of AI Core/Vector Core. 0-AI Core index, 1-Vector Core Index
    kernel_name : str. cce kernel name, default value is "tabulate_fusion_grad"

    Returns
    -------
    None
    """
    _check_params(table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem,
                  split_count, split_index, kernel_name)

    obj = TabulateFusionGrad(table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem,
                             split_count, split_index, kernel_name)
    obj.compute()
