#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

sparse_reshape
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl import constant_util


class SparseReshapeInfo:
    """
    The class for constant.
    """
    MAX_INT64 = 2 ** 63 - 1
    TILING_NUM = 23
    RESERVED_SPACE = 16
    KB_BTYPE = 1024
    FIXED_NUM = 167
    FIXED_ARRAYS_NUM = 6
    INT32_SIZE_BTYPE = 4
    VEC_MAX_COMPUTE = 64
    QUARTER_SPLIT = 4
    MAX_ORI_DIM = 8
    MAX_NEW_DIM = 8


class SparseReshape:
    """
    Class for Dynamic shape for operator SparseReshape
    """

    def __init__(self, indices, shape, new_shape, y_indices, y_shape, kernel_name):
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.unified_buffer_size = tbe_platform.get_soc_spec("UB_SIZE")
        self.kernel_name = kernel_name

        self.shape = self.tik_instance.Tensor("int32", (8,), name="shape", scope=tik.scope_gm)
        self.new_shape = self.tik_instance.Tensor("int32", (8,), name="new_shape", scope=tik.scope_gm)
        self.new_shape_ub = self.tik_instance.Tensor("int32", (8,), name="new_shape_ub", scope=tik.scope_ubuf)
        self.y_shape = self.tik_instance.Tensor("int32", (8,), name="y_shape", scope=tik.scope_gm)

        self.tiling_gm = self.tik_instance.Tensor("int32", (SparseReshapeInfo.TILING_NUM,), name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.tiling_ub = self.tik_instance.Tensor("int32", (SparseReshapeInfo.TILING_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 3, 0, 0)

        self.ori_indice_num = self.tik_instance.Scalar("int32", name="ori_indice_num", init_value=self.tiling_ub[0])
        self.actual_core_num = self.tik_instance.Scalar("int32", name="actual_core_num", init_value=self.tiling_ub[1])
        self.each_core_bout = self.tik_instance.Scalar("int32", name="each_core_bout", init_value=self.tiling_ub[2])
        self.residue_bout = self.tik_instance.Scalar("int32", name="residue_bout", init_value=self.tiling_ub[3])
        self.residue_indice = self.tik_instance.Scalar("int32", name="residue_indice", init_value=self.tiling_ub[4])
        self.ori_indice_dims = self.tik_instance.Scalar("int32", name="ori_indice_dims", init_value=self.tiling_ub[5])
        self.new_indice_dims = self.tik_instance.Scalar("int32", name="new_indice_dims", init_value=self.tiling_ub[6])
        self.ub_max_bout = (self.unified_buffer_size - SparseReshapeInfo.RESERVED_SPACE * SparseReshapeInfo.KB_BTYPE -
                            SparseReshapeInfo.INT32_SIZE_BTYPE * SparseReshapeInfo.FIXED_NUM -
                            SparseReshapeInfo.INT32_SIZE_BTYPE * SparseReshapeInfo.VEC_MAX_COMPUTE *
                            (SparseReshapeInfo.MAX_ORI_DIM + SparseReshapeInfo.MAX_NEW_DIM)) // \
                           (SparseReshapeInfo.INT32_SIZE_BTYPE * SparseReshapeInfo.VEC_MAX_COMPUTE *
                            (SparseReshapeInfo.MAX_ORI_DIM + SparseReshapeInfo.FIXED_ARRAYS_NUM)) // \
                           SparseReshapeInfo.QUARTER_SPLIT

        self.ori_total_size = self.tik_instance.ScalarArray(dtype="int32", length=8)
        self.ori_total_size[0].set_as(self.tiling_ub[7])
        self.ori_total_size[1].set_as(self.tiling_ub[8])
        self.ori_total_size[2].set_as(self.tiling_ub[9])
        self.ori_total_size[3].set_as(self.tiling_ub[10])
        self.ori_total_size[4].set_as(self.tiling_ub[11])
        self.ori_total_size[5].set_as(self.tiling_ub[12])
        self.ori_total_size[6].set_as(self.tiling_ub[13])
        self.ori_total_size[7].set_as(self.tiling_ub[14])

        tem = self.tik_instance.Scalar("int32", name="tem")
        tem.set_as(self.tiling_ub[15])
        self.new_shape_ub[0].set_as(tem)
        tem.set_as(self.tiling_ub[16])
        self.new_shape_ub[1].set_as(tem)
        tem.set_as(self.tiling_ub[17])
        self.new_shape_ub[2].set_as(tem)
        tem.set_as(self.tiling_ub[18])
        self.new_shape_ub[3].set_as(tem)
        tem.set_as(self.tiling_ub[19])
        self.new_shape_ub[4].set_as(tem)
        tem.set_as(self.tiling_ub[20])
        self.new_shape_ub[5].set_as(tem)
        tem.set_as(self.tiling_ub[21])
        self.new_shape_ub[6].set_as(tem)
        tem.set_as(self.tiling_ub[22])
        self.new_shape_ub[7].set_as(tem)
        self.tik_instance.h_data_move(self.y_shape, self.new_shape_ub)

        self.indices = self.tik_instance.Tensor("int32", (self.ori_indice_dims, self.ori_indice_num), name="indices",
                                                scope=tik.scope_gm)
        self.y_indices = self.tik_instance.Tensor("int32", (self.new_indice_dims, self.ori_indice_num),
                                                  name="y_indices", scope=tik.scope_gm)

    def compute(self):
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as core_num:
            with self.tik_instance.if_scope(core_num < self.actual_core_num):
                indice_start_num = self.tik_instance.Scalar("int32", name="indice_start_num",
                                                            init_value=core_num * self.each_core_bout * 64)
                with self.tik_instance.if_scope(core_num < self.residue_bout):
                    with self.tik_instance.if_scope(self.ub_max_bout < self.each_core_bout + 1):
                        bout_loops = (self.each_core_bout + 1) // self.ub_max_bout
                        with self.tik_instance.for_range(0, bout_loops) as loop:
                            indice_start = indice_start_num + (core_num + loop * self.ub_max_bout) * 64
                            self.count_ori_indice_size(indice_start, self.ub_max_bout)

                        with self.tik_instance.if_scope((self.each_core_bout + 1) % self.ub_max_bout > 0):
                            indice_start = indice_start_num + (core_num + bout_loops * self.ub_max_bout) * 64
                            self.count_ori_indice_size(indice_start, (self.each_core_bout + 1) % self.ub_max_bout)

                    with self.tik_instance.else_scope():
                        indice_start = indice_start_num + core_num * 64
                        self.count_ori_indice_size(indice_start, self.each_core_bout + 1)

                with self.tik_instance.elif_scope(tik.all(core_num >= self.residue_bout, self.each_core_bout > 0)):
                    with self.tik_instance.if_scope(self.ub_max_bout < self.each_core_bout):
                        bout_loops = self.each_core_bout // self.ub_max_bout
                        with self.tik_instance.for_range(0, bout_loops) as loop:
                            indice_start = indice_start_num + (self.residue_bout + loop * self.ub_max_bout) * 64
                            self.count_ori_indice_size(indice_start, self.ub_max_bout)

                        with self.tik_instance.if_scope(self.each_core_bout % self.ub_max_bout > 0):
                            indice_start = indice_start_num + (self.residue_bout + bout_loops * self.ub_max_bout) * 64
                            self.count_ori_indice_size(indice_start, self.each_core_bout % self.ub_max_bout)

                    with self.tik_instance.else_scope():
                        indice_start = (self.residue_bout + core_num * self.each_core_bout) * 64
                        self.count_ori_indice_size(indice_start, self.each_core_bout)

                with self.tik_instance.if_scope(tik.all(core_num == self.actual_core_num - 1, self.residue_indice > 0)):
                    with self.tik_instance.if_scope(self.each_core_bout > 0):
                        indice_start = (self.actual_core_num * self.each_core_bout + self.residue_bout) * 64 \
                                       - 64 + self.residue_indice
                        self.count_ori_indice_size(indice_start, 1)

                    with self.tik_instance.else_scope():
                        self.indice_num_not_full(self.residue_indice)

    def count_ori_indice_size(self, indice_start_num, bout):
        ori_data = self.tik_instance.Tensor("int32", (self.ori_indice_dims, 64 * bout), name="ori_data",
                                            scope=tik.scope_ubuf)
        self.tik_instance.h_duplicate(ori_data, 0)
        indice_size = self.tik_instance.Tensor("int32", (64 * bout,), name="indice_size", scope=tik.scope_ubuf)
        temp = self.tik_instance.Tensor("int32", (64 * bout,), name="temp", scope=tik.scope_ubuf)
        scalar = self.tik_instance.Scalar("int32", name="scalar", init_value=0)
        self.tik_instance.h_duplicate(indice_size, 0)

        self.tik_instance.h_data_move(ori_data, self.indices[:, indice_start_num:indice_start_num + 64 * bout])
        with self.tik_instance.for_range(0, self.ori_indice_dims) as current_dim:
            scalar.set_as(self.ori_total_size[8 - current_dim - 1])
            self.tik_instance.h_duplicate(temp, scalar)
            self.tik_instance.vec_mul(64, temp, ori_data[current_dim, :], temp, bout, 8, 8, 8)
            self.tik_instance.h_add(indice_size, indice_size, temp)
        self.get_new_indice(indice_size, indice_start_num, bout)

    def get_new_indice(self, ori_indice_size, indice_start_num, bout):
        current_dim = self.tik_instance.Scalar("int32", name="current_dim", init_value=0)
        new_shape_dim = self.tik_instance.Scalar("float32", name="new_shape_dim", init_value=1.0)
        x = self.tik_instance.Scalar("int32", name="x", init_value=1)
        x_tensor = self.tik_instance.Tensor("int32", (64 * bout,), name="x_tensor", scope=tik.scope_ubuf)
        temp = self.tik_instance.Tensor("float32", (64 * bout,), name="temp", scope=tik.scope_ubuf)
        temp_2 = self.tik_instance.Tensor("float32", (64 * bout,), name="temp_2", scope=tik.scope_ubuf)
        ori_indice_size_float = self.tik_instance.Tensor("float32", (64 * bout,), name="ori_indice_size_float",
                                                         scope=tik.scope_ubuf)
        new_shape_ub_float = self.tik_instance.Tensor("float32", (8,), name="new_shape_ub_float", scope=tik.scope_ubuf)
        self.tik_instance.h_cast(ori_indice_size_float, ori_indice_size, "none")
        self.tik_instance.h_cast(new_shape_ub_float, self.new_shape_ub, "none")

        dst_ub = self.tik_instance.Tensor("int32", (64 * bout,), name="dst_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.new_indice_dims) as dims_loop:
            current_dim.set_as(self.new_indice_dims - dims_loop - 1)
            self.tik_instance.h_duplicate(x_tensor, x)
            self.tik_instance.h_cast(temp, x_tensor, "none")

            self.tik_instance.h_div(temp_2, ori_indice_size_float, temp)
            self.tik_instance.h_cast(x_tensor, temp_2, "floor")
            self.tik_instance.h_cast(temp_2, x_tensor, "none")
            new_shape_dim.set_as(new_shape_ub_float[current_dim])

            self.tik_instance.h_div(temp, temp_2, new_shape_dim)
            self.tik_instance.h_cast(x_tensor, temp, "floor")
            self.tik_instance.h_cast(temp, x_tensor, "none")
            self.tik_instance.h_mul(temp, temp, new_shape_dim)
            self.tik_instance.h_sub(temp, temp_2, temp)

            self.tik_instance.h_cast(dst_ub, temp, "round")
            self.tik_instance.data_move(self.y_indices[current_dim * self.ori_indice_num + indice_start_num],
                                        dst_ub, 0, 1, bout * 8, 8, 8)
            x.set_as(x * new_shape_dim)

    def indice_num_not_full(self, indice_num):
        ori_data = self.tik_instance.Tensor("int32", (self.ori_indice_dims, 64),
                                            name="ori_data", scope=tik.scope_ubuf)
        out_data = self.tik_instance.Tensor("int32", (self.new_indice_dims, self.residue_indice),
                                            name="out_data", scope=tik.scope_ubuf)
        self.tik_instance.h_duplicate(ori_data, 0)
        with self.tik_instance.for_range(0, self.ori_indice_dims) as dim:
            self.tik_instance.data_move(ori_data[dim * 64], self.indices[dim * indice_num], 0, 1, 8, 8, 8)

        indice_size = self.tik_instance.Tensor("int32", (64,), name="indice_size", scope=tik.scope_ubuf)
        temp = self.tik_instance.Tensor("int32", (64,), name="temp", scope=tik.scope_ubuf)
        scalar = self.tik_instance.Scalar("int32", name="scalar", init_value=0)
        self.tik_instance.h_duplicate(indice_size, 0)
        total = self.tik_instance.Scalar("int32", name="total", init_value=1)
        dim_value = self.tik_instance.Scalar("int32", name="dim_value", init_value=1)
        x = self.tik_instance.Scalar("int32", name="x", init_value=1)

        with self.tik_instance.for_range(0, self.ori_indice_dims) as current_dim:
            scalar.set_as(self.ori_total_size[8 - current_dim - 1])
            self.tik_instance.h_duplicate(temp, scalar)
            self.tik_instance.vec_mul(64, temp, ori_data[current_dim, :], temp, 1, 8, 8, 8)
            self.tik_instance.h_add(indice_size, indice_size, temp)

        with self.tik_instance.for_range(0, indice_num) as indice:
            x.set_as(1)
            total.set_as(indice_size[indice])

            with self.tik_instance.for_range(0, self.new_indice_dims) as new_dim:
                current_dim = self.new_indice_dims - new_dim - 1
                dim_value.set_as(self.new_shape_ub[current_dim])
                out_data[current_dim * self.residue_indice + indice].set_as(total / x % dim_value)
                x.set_as(x * dim_value)

        self.tik_instance.data_move(self.y_indices, out_data, 0, 1, 64, 8, 8)


@register_operator("SparseReshape")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sparse_reshape(indices,
                   shape,
                   new_shape,
                   y_indices,
                   y_shape,
                   kernel_name="sparse_reshape"):
    """
    Generate arg_min operator use arg_min

    Parameters
    ----------
    indices: dict
        data of indices, support "int32".
    shape: dict
        original shape, support "int32".
    new_shape: dict
        new shape, support "int32".
    y_indices: dict
        data of new indices, support "int32".
    y_shape: dict
        new shape, support "int32".
    kernel_name: str
        kernel name, default value is "sparse_reshape"

    Returns
    -------
    tik_instance
    """
    indices_dtype = indices.get("dtype")
    shape_dtype = shape.get("dtype")
    new_shape_dtype = new_shape.get("dtype")
    y_indices_dtype = y_indices.get("dtype")
    y_shape_dtype = y_shape.get("dtype")
    obj = SparseReshape(indices_dtype,
                        shape_dtype,
                        new_shape_dtype,
                        y_indices_dtype,
                        y_shape_dtype,
                        kernel_name=kernel_name)
    obj.compute()

    opt_config = {"out_of_bound_sync_check": True}
    tbe_context.get_context().add_compile_info("vars",
                                               {"ub_size": obj.unified_buffer_size, "core_num": obj.aicore_num})
    obj.tik_instance.BuildCCE(kernel_name=obj.kernel_name,
                              inputs=(obj.indices, obj.shape, obj.new_shape),
                              outputs=(obj.y_indices, obj.y_shape),
                              flowtable=obj.tiling_gm,
                              config=opt_config)

    return obj.tik_instance
