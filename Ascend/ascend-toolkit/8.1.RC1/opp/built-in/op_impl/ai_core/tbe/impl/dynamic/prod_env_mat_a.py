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

prod_env_mat_a
"""

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util import util_common
from impl.util.util_tik_comm_func import ceil_div


# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=too-few-public-methods
# 'pylint: disable=too-many-statements, too-many-arguments
class Constant:
    """
    The class for constant
    """
    MININUM_NUM_FLOAT = -(3.4028235 ** 38)
    DTYPE_BYTES = {"float32": 4, "float16": 2}
    NLOC_NUM = 1026
    NEI_NUM = 1024
    TILING_ARG_NUM = 4
    BLOCK_INT64 = 4


# 'pylint: disable=too-many-public-methods
class ProdEnvMatA:
    """
    ProdEnvMatA class
    """
    # 'pylint: disable=unused-argument
    def __init__(self, coord, types, natoms, box, mesh, davg, dstd, sel_a, rcut, rcut_smth, nlist,
                 split_count, split_index, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.tik = tik
        self.coord_shape = coord["shape"]
        self.type_shape = types["shape"]
        self.natoms_shape = natoms["shape"]
        self.box_shape = box["shape"]
        self.mesh_shape = mesh["shape"]
        self.davg_shape = davg["shape"]
        self.dstd_shape = dstd["shape"]
        self.coord_dtype = coord.get("dtype").lower()
        self.type_dtype = types.get("dtype").lower()

        self.split_count = split_count
        self.split_index = split_index

        self.nsample = 1
        if self.coord_shape[0] != -1:
            self.nsample = self.coord_shape[0]

        self.nall = self.type_shape[1]
        self.max_nall = 31040
        self.nnei = sum(sel_a)

        self.max_gm_size = 2 ** 31 - 1

        self.max_nbor_size = 256
        self.type_num = len(sel_a)
        if self.type_num != 2:
            self.max_nbor_size = 1024

        self.sel_a = self.tik_instance.ScalarArray(self.type_dtype, name="sel_a",
                                                   length=self.type_num,
                                                   init_value=sel_a)

        self.sel_a_list = sel_a

        self.sel_a_back_list = [0]
        for i in range(1, self.type_num):
            self.sel_a_back_list.append((sum(sel_a[:i]) + 7) // 8 * 8)

        self.sel_a_back = self.tik_instance.ScalarArray(self.type_dtype, name="sel_a_back",
                                                        length=self.type_num,
                                                        init_value=self.sel_a_back_list)

        self.rcut = rcut_smth
        self.rcut_smth = rcut

        self.coord_dim_num = 3
        self.mode = 3
        self.nloc = -1
        cur_nsample_mesh_size = self.mesh_shape[0] // self.nsample
        if (cur_nsample_mesh_size - 1) % Constant.NLOC_NUM != 0:
            self.mode = 1
            self.nloc = (cur_nsample_mesh_size - 1 - self.type_shape[1]) // Constant.NLOC_NUM

        self.block_num = 32 // Constant.DTYPE_BYTES.get(self.coord_dtype)

        self.nnei_align = (self.nnei + self.block_num - 1) // self.block_num * self.block_num

        self.descrpt_size = self.nnei * 4

        self.kernel_name = kernel_name

        self.nall_data_size = self.nall * self.coord_dim_num // self.block_num
        self.max_nbor_data_size = self.max_nbor_size // self.block_num

        self.repeat_once_size = self.block_num * self.block_num
        self.block_stride = 1
        self.repeat_stride = 8
        self.max_nbor_repeat_times = self.max_nbor_size // self.repeat_once_size

        self.cur_type_neighbour_nums = self.tik_instance.Scalar("uint32",
                                                                "cur_type_neighbour_nums",
                                                                init_value=0)
        self.cur_loc_nei_num = self.tik_instance.Scalar("int32",
                                                        "cur_loc_nei_num",
                                                        init_value=self.max_nbor_size)
        self.int32_type = "int32"
        self.nloc_d = self.tik_instance.Scalar(self.int32_type, "nloc_d", init_value=0)
        self.nall_d = self.tik_instance.Scalar(self.int32_type, "nall_d", init_value=0)
        self.total_nloc = self.tik_instance.Scalar(self.int32_type, "total_nloc", init_value=0)
        self.nloc_offset = self.tik_instance.Scalar(self.int32_type, "nloc_offset", init_value=0)
        self.nnei_once_repeat_nums = self.block_num * self.block_num
        self.nnei_repeat_times = self.nnei // self.nnei_once_repeat_nums
        self.max_value = self.tik_instance.Scalar("float32", "max_value",
                                                  init_value=Constant.MININUM_NUM_FLOAT)

        self.cur_op_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.cur_op_core_num_scalar = self.tik_instance.Scalar(self.int32_type, "cur_op_core_num_scalar",
                                                               init_value=self.cur_op_core_num)
        # tilingdata
        self.tiling_dtype = "int64"
        self.tiling_align = util_common.align(Constant.TILING_ARG_NUM, Constant.BLOCK_INT64)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.core_num_var = self.tik_instance.Scalar(self.tiling_dtype, "core_num_var",
                                                     init_value=self.cur_op_core_num)

        self.init_buffer()

    def init_buffer(self):
        """
        init the gm input buffer and the output buffer
        """
        self.coord_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size],
                                                 name="coord_gm", scope=self.tik.scope_gm)
        self.type_gm = self.tik_instance.Tensor(self.type_dtype, [self.max_gm_size],
                                                 name="type_gm", scope=self.tik.scope_gm)
        self.natoms_gm = self.tik_instance.Tensor(self.type_dtype, self.natoms_shape,
                                                 name="natoms_gm", scope=self.tik.scope_gm)
        self.box_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size],
                                                 name="box_gm", scope=self.tik.scope_gm)
        self.mesh_gm = self.tik_instance.Tensor(self.type_dtype, [self.max_gm_size],
                                                 name="mesh_gm", scope=self.tik.scope_gm)
        self.davg_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size],
                                                 name="davg_gm", scope=self.tik.scope_gm)
        self.dstd_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size],
                                                 name="dstd_gm", scope=self.tik.scope_gm)

        self.rij_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size],
                                               name="rij_gm", scope=self.tik.scope_gm)
        self.descrpt_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size],
                                                   name="descrpt_gm", scope=self.tik.scope_gm)
        self.descrpt_deriv_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size],
                                                         name="descrpt_deriv_gm", scope=self.tik.scope_gm)
        self.nlist_gm = self.tik_instance.Tensor(self.type_dtype, [self.max_gm_size],
                                                 name="nlist_gm", scope=self.tik.scope_gm)

        self.coord_ub_shape = [self.nall * 3]
        self.type_ub_shape = [self.nall]
        self.natoms_ub_shape = [3]
        self.mesh_ub_shape = [self.max_nbor_size]

    def data_move_to_ub_phase1(self, cur_nsample_index, type_ub):
        """
        copy type data in ub from gm.
        """
        src_offset = self.tik_instance.Scalar(self.int32_type, "src_offset",
                                              init_value=cur_nsample_index * self.nall_d)
        self.tik_instance.data_move(type_ub[0], self.type_gm[src_offset],
                                    sid=0, nburst=1, burst=self.nall_d // self.block_num,
                                    src_stride=0, dst_stride=0)

    def data_move_to_ub_phase0(self, cur_nloc_index, mesh_ub, cur_sample):
        """
        copy mesh data in ub from gm.
        """
        mesh_offset = self.tik_instance.Scalar(self.int32_type, "mesh_offset",
                                               init_value=1 + 2 * self.total_nloc +
                                                          cur_nloc_index * Constant.NEI_NUM)

        if self.mode != 3:
            mesh_offset.set_as(cur_sample * (1 + Constant.NLOC_NUM * self.nloc + self.type_shape[1]) + 1 +
                               2 * self.total_nloc + cur_nloc_index * Constant.NEI_NUM)
        self.tik_instance.data_move(mesh_ub[0], self.mesh_gm[mesh_offset],
                                    sid=0, nburst=1, burst=self.max_nbor_data_size,
                                    src_stride=0, dst_stride=0)

    def data_move_avg_to_ub(self, cur_type_idx, davg_ub, dstd_ub):
        """
        copy davg data in ub from gm.
        """
        avg_offset = self.tik_instance.Scalar(self.int32_type, "avg_offset",
                                               init_value=cur_type_idx * self.descrpt_size)
        self.tik_instance.data_move(davg_ub[0], self.davg_gm[avg_offset],
                                    sid=0, nburst=1, burst=ceil_div(self.descrpt_size, self.block_num),
                                    src_stride=0, dst_stride=0)
        self.tik_instance.data_move(dstd_ub[0], self.dstd_gm[avg_offset],
                                    sid=0, nburst=1, burst=ceil_div(self.descrpt_size, self.block_num),
                                    src_stride=0, dst_stride=0)

    def set_last_block_value(self, tensor, last_block_tensor, offset):
        """
        set the last block nums when copy ub to out.
        """
        for i in range(0, self.block_num):
            index = self.nnei * offset - self.block_num + i
            last_block_tensor[i].set_as(tensor[index])

    def data_move_res_to_gm(self, src_tensor, dst_gm, last_block_tensor,
                            block_nums, gm_offset, stride_offset):
        """
        copy res data to gm from ub buffer.
        """
        self.tik_instance.data_move(dst_gm[gm_offset], src_tensor,
                                    sid=0, nburst=1, burst=block_nums,
                                    src_stride=0, dst_stride=0)

        self.set_last_block_value(src_tensor, last_block_tensor, stride_offset)

        self.tik_instance.data_move(dst_gm[gm_offset + self.nnei * stride_offset - self.block_num],
                                    last_block_tensor,
                                    sid=0, nburst=1, burst=1,
                                    src_stride=0, dst_stride=0)

    def data_move_to_gm_last_loc(self, res_descrpt_a_tensor,
                                 res_descrpt_a_deriv_tensor, nlist,
                                 cur_nsample_index, cur_nloc_index):
        """
        copy nlis, descrpt, descrpt_deriv to gm.
        """
        last_block_tensor = self.tik_instance.Tensor(self.coord_dtype, [self.block_num],
                                                     name="last_block_tensor", scope=self.tik.scope_ubuf)
        descrpt_offset = self.tik_instance.Scalar(self.int32_type, "rij_dst_offset",
                                                  init_value=cur_nsample_index * self.nloc_d * self.nnei * 4 +
                                                             cur_nloc_index * self.nnei * 4)

        descrpt_move_blocks = (self.nnei * 4) // self.block_num

        self.data_move_res_to_gm(res_descrpt_a_tensor, self.descrpt_gm, last_block_tensor, descrpt_move_blocks,
                                 descrpt_offset, 4)

        descrpt_deriv_offset = self.tik_instance.Scalar(self.int32_type, "descrpt_deriv_offset",
                                                        init_value=cur_nsample_index * self.nloc_d * self.nnei * 12 +
                                                             cur_nloc_index * self.nnei * 12)

        descrpt_deriv_move_blocks = (self.nnei * 12) // self.block_num

        self.data_move_res_to_gm(res_descrpt_a_deriv_tensor, self.descrpt_deriv_gm, last_block_tensor,
                                 descrpt_deriv_move_blocks, descrpt_deriv_offset, 12)

        nlist_move_blocks = self.nnei // self.block_num
        nlist_dst_offset = self.tik_instance.Scalar(self.int32_type, "nlist_dst_offset",
                                                    init_value=cur_nsample_index * self.nloc_d * self.nnei +
                                                                   cur_nloc_index * self.nnei)

        self.set_last_block_value(nlist, last_block_tensor, 1)

        self.tik_instance.data_move(self.nlist_gm[nlist_dst_offset], nlist,
                                    sid=0, nburst=1, burst=nlist_move_blocks, src_stride=0, dst_stride=0)
        self.tik_instance.data_move(self.nlist_gm[nlist_dst_offset + self.nnei - self.block_num],
                                    last_block_tensor,
                                    sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)

    def data_move_rij_to_gm(self, res_rij_tensor,
                            cur_nsample_index, cur_nloc_index, loc_nei_idx, one_cor_nloc_num):
        """
        copy rij data to gm from ub buffer.
        """
        def move_rij_align():
            """
            abstract copy rij data to gm func.
            """
            rij_move_blocks = (self.nnei * 3 + self.block_num - 1) // self.block_num
            rij_dst_offset = self.tik_instance.Scalar(self.int32_type, "rij_dst_offset",
                                                      init_value=cur_nsample_index * self.nloc_d * self.nnei * 3 +
                                                                   cur_nloc_index * self.nnei * 3)
            self.tik_instance.data_move(self.rij_gm[rij_dst_offset], res_rij_tensor,
                                        sid=0, nburst=1, burst=rij_move_blocks, src_stride=0, dst_stride=0)

        if self.nnei % self.block_num == 0:
            move_rij_align()
        else:
            with self.tik_instance.if_scope(loc_nei_idx != (one_cor_nloc_num - 1)):
                move_rij_align()
            with self.tik_instance.if_scope(loc_nei_idx == (one_cor_nloc_num - 1)):
                last_block_tensor = self.tik_instance.Tensor(self.coord_dtype, [self.block_num],
                                                             name="last_block_tensor", scope=self.tik.scope_ubuf)
                rij_move_blocks = (self.nnei * 3) // self.block_num
                rij_dst_offset = self.tik_instance.Scalar(self.int32_type, "rij_dst_offset",
                                                          init_value=cur_nsample_index * self.nloc_d * self.nnei * 3 +
                                                                     cur_nloc_index * self.nnei * 3)
                self.data_move_res_to_gm(res_rij_tensor, self.rij_gm, last_block_tensor,
                                         rij_move_blocks, rij_dst_offset, 3)

    def data_move_to_gm(self, res_descrpt_a_tensor,
                        res_descrpt_a_deriv_tensor, nlist,
                        cur_nsample_index, cur_nloc_index):
        """
        copy nlist descrpt descrpt_derive data to gm from ub buffer when block align.
        """
        descrpt_offset = self.tik_instance.Scalar(self.int32_type, "descrpt_offset",
                                                  init_value=cur_nsample_index * self.nloc_d * self.nnei * 4 +
                                                             cur_nloc_index * self.nnei * 4)

        descrpt_move_blocks = self.tik_instance.Scalar(self.int32_type, "descrpt_move_blocks",
                                                       init_value=(self.nnei * 4 + self.block_num - 1)
                                                                   // self.block_num)
        self.tik_instance.data_move(self.descrpt_gm[descrpt_offset], res_descrpt_a_tensor,
                                    sid=0, nburst=1, burst=descrpt_move_blocks, src_stride=0, dst_stride=0)


        descrpt_deriv_offset = self.tik_instance.Scalar(self.int32_type, "descrpt_deriv_offset",
                                                        init_value=cur_nsample_index * self.nloc_d * self.nnei * 12 +
                                                                   cur_nloc_index * self.nnei * 12)

        descrpt_deriv_move_blocks = self.tik_instance.Scalar(self.int32_type, "descrpt_deriv_move_blocks",
                                                             init_value=(self.nnei * 12 + self.block_num - 1)
                                                                  // self.block_num)
        self.tik_instance.data_move(self.descrpt_deriv_gm[descrpt_deriv_offset], res_descrpt_a_deriv_tensor,
                                    sid=0, nburst=1, burst=descrpt_deriv_move_blocks, src_stride=0, dst_stride=0)

        nlist_move_blocks = (self.nnei + self.block_num - 1) // self.block_num
        nlist_dst_offset = self.tik_instance.Scalar(self.int32_type, "nlist_dst_offset",
                                                    init_value=cur_nsample_index * self.nloc_d * self.nnei +
                                                               cur_nloc_index * self.nnei)

        self.tik_instance.data_move(self.nlist_gm[nlist_dst_offset], nlist,
                                    sid=0, nburst=1, burst=nlist_move_blocks, src_stride=0, dst_stride=0)

    def do_avg_process(self, descrpt_tensor, descrpt_deriv_tensor, cur_type_idx):
        """
        do avg std process for descrpt and descrpt_deriv.
        """
        nnei_repeat_align = (self.nnei + self.nnei_once_repeat_nums - 1) // self.nnei_once_repeat_nums \
                            * self.nnei_once_repeat_nums
        dstd_ub = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_align * 12],
                                           name="dstd_ub", scope=self.tik.scope_ubuf)
        davg_ub = dstd_ub[nnei_repeat_align * 4]

        self.data_move_avg_to_ub(cur_type_idx, davg_ub, dstd_ub)

        descrpt_align = self.descrpt_size // self.nnei_once_repeat_nums
        descrpt_tail = self.descrpt_size % self.nnei_once_repeat_nums

        self.tik_instance.vsub(self.nnei_once_repeat_nums, descrpt_tensor, descrpt_tensor,
                               davg_ub,
                               descrpt_align,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        if descrpt_tail != 0:
            self.tik_instance.vsub(descrpt_tail, descrpt_tensor[descrpt_align * self.nnei_once_repeat_nums],
                                   descrpt_tensor[descrpt_align * self.nnei_once_repeat_nums],
                                   dstd_ub[nnei_repeat_align * 4 + descrpt_align * self.nnei_once_repeat_nums],
                                   1,
                                   self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_tensor, descrpt_tensor,
                               dstd_ub,
                               descrpt_align,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        if descrpt_tail != 0:
            self.tik_instance.vdiv(descrpt_tail, descrpt_tensor[descrpt_align * self.nnei_once_repeat_nums],
                                   descrpt_tensor[descrpt_align * self.nnei_once_repeat_nums],
                                   dstd_ub[descrpt_align * self.nnei_once_repeat_nums],
                                   1,
                                   self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)

        trans_dstd_ub = dstd_ub[nnei_repeat_align * 12 - self.nnei_align * 3 - nnei_repeat_align]
        self.tik_instance.v4dtrans(False, trans_dstd_ub, dstd_ub, self.nnei_align, 4)

        for des in range(3):
            for nn in range(3):
                self.tik_instance.vadds(self.nnei, dstd_ub[des * 3 * nnei_repeat_align + nn * nnei_repeat_align],
                                        dstd_ub[nnei_repeat_align * 12 - self.nnei_align * 3 -
                                                nnei_repeat_align + des * self.nnei_align],
                                        0,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride,
                                        0, "counter")

        self.tik_instance.vadds(self.nnei, dstd_ub[9 * nnei_repeat_align],
                                dstd_ub[nnei_repeat_align * 11],
                                0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")
        self.tik_instance.vadds(self.nnei, dstd_ub[10 * nnei_repeat_align],
                                dstd_ub[nnei_repeat_align * 11],
                                0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        descrpt_deriv_align = (nnei_repeat_align * 12) // self.nnei_once_repeat_nums
        descrpt_deriv_align_tail = 0

        max_repeat_times = 255
        if descrpt_deriv_align > max_repeat_times:
            descrpt_deriv_align_tail = descrpt_deriv_align % max_repeat_times
            descrpt_deriv_align = max_repeat_times

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_deriv_tensor, descrpt_deriv_tensor,
                               dstd_ub,
                               descrpt_deriv_align,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        if descrpt_deriv_align_tail != 0:
            self.tik_instance.vdiv(self.nnei_once_repeat_nums,
                                   descrpt_deriv_tensor[descrpt_deriv_align * self.nnei_once_repeat_nums],
                                   descrpt_deriv_tensor[descrpt_deriv_align * self.nnei_once_repeat_nums],
                                   dstd_ub[descrpt_deriv_align * self.nnei_once_repeat_nums],
                                   descrpt_deriv_align_tail,
                                   self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            descrpt_a_deriv_tensor = dstd_ub

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor,
                                    descrpt_deriv_tensor,
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align],
                                    descrpt_deriv_tensor[nnei_repeat_align],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 2],
                                    descrpt_deriv_tensor[nnei_repeat_align * 2],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 3],
                                    descrpt_deriv_tensor[nnei_repeat_align * 3],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 4],
                                    descrpt_deriv_tensor[nnei_repeat_align * 4],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 5],
                                    descrpt_deriv_tensor[nnei_repeat_align * 5],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 6],
                                    descrpt_deriv_tensor[nnei_repeat_align * 6],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 7],
                                    descrpt_deriv_tensor[nnei_repeat_align * 7],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 8],
                                    descrpt_deriv_tensor[nnei_repeat_align * 8],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 9],
                                    descrpt_deriv_tensor[nnei_repeat_align * 9],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 10],
                                    descrpt_deriv_tensor[nnei_repeat_align * 10],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(self.nnei_align, descrpt_a_deriv_tensor[self.nnei_align * 11],
                                    descrpt_deriv_tensor[nnei_repeat_align * 11],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.v4dtrans(True, descrpt_deriv_tensor, descrpt_a_deriv_tensor, self.nnei_align, 12)

    def concat_rij_result(self, rij_tensors, res_rij_tensor, free_buffer):
        """
        concat rij data in res ub buffer.
        """
        nnei_align = (self.nnei + self.block_num - 1) // self.block_num * self.block_num
        rij_tensor = free_buffer

        self.tik_instance.vadds(nnei_align, rij_tensor,
                                rij_tensors[0],
                                0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        self.tik_instance.vadds(nnei_align, rij_tensor[nnei_align],
                                rij_tensors[1],
                                0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        self.tik_instance.vadds(nnei_align, rij_tensor[nnei_align * 2],
                                rij_tensors[2],
                                0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        self.tik_instance.v4dtrans(True, res_rij_tensor, rij_tensor, nnei_align, 3)

    def concat_result(self, descrpt_a_tensors,
                      res_descrpt_a_tensor,
                      free_buffer):
        """
        concat rij descrpt descrpt_deriv data in res ub buffer.
        """
        nnei_align = (self.nnei + self.block_num - 1) // self.block_num * self.block_num

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            descrpt_a_tensor = free_buffer

            self.tik_instance.vadds(nnei_align, descrpt_a_tensor[0],
                                    descrpt_a_tensors[0],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(nnei_align, descrpt_a_tensor[nnei_align],
                                    descrpt_a_tensors[1],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(nnei_align, descrpt_a_tensor[nnei_align * 2],
                                    descrpt_a_tensors[2],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            self.tik_instance.vadds(nnei_align, descrpt_a_tensor[nnei_align * 3],
                                    descrpt_a_tensors[3],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")
            self.tik_instance.v4dtrans(True, res_descrpt_a_tensor, descrpt_a_tensor, self.nnei_align, 4)

    def compute_descrpt_a_deriv_three(self, left_value_save, rr0_factor_tensor,
                                      rr0_tensor, rr1_tensor, rr2_tensor,
                                      sw_tensor, res_vec_tensor_2, res_vec_tensor_1,
                                      descrpt_a_1, dsw_tensor, dis_rev_tensor,
                                      not_one, descrpt_a_deriv_3, descrpt_a_deriv_4,
                                      descrpt_a_deriv_5):
        """
        compute the group res for deriv.
        """
        compute_nums = self.nnei
        nnei_repeat_times = (compute_nums + self.nnei_once_repeat_nums - 1) // self.nnei_once_repeat_nums

        self.tik_instance.vmuls(self.nnei_once_repeat_nums, left_value_save,
                                rr0_factor_tensor,
                                2,
                                nnei_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, left_value_save, left_value_save,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, left_value_save, left_value_save,
                               res_vec_tensor_2,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_3, descrpt_a_1,
                               dsw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_3, descrpt_a_deriv_3,
                               dis_rev_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmuls(self.nnei_once_repeat_nums, descrpt_a_deriv_3,
                                descrpt_a_deriv_3,
                                not_one,
                                nnei_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        self.tik_instance.vadd(self.nnei_once_repeat_nums, left_value_save, left_value_save,
                               descrpt_a_deriv_3,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_3, left_value_save,
                               rr0_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_4, left_value_save,
                               rr1_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_5, left_value_save,
                               rr2_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, left_value_save, sw_tensor,
                               res_vec_tensor_1,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vsub(self.nnei_once_repeat_nums, descrpt_a_deriv_3, descrpt_a_deriv_3,
                               left_value_save,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

    def vector_compute_last_process(self, dis_rev_tensor, neighbour_coords,
                                    dis_dot_tensor,
                                    sw_tensor, dsw_tensor,
                                    res_vec_tensor, res_vec_tensor_1, res_vec_tensor_2, res_vec_tensor_3,
                                    res_descrpt_a_tensor,
                                    res_descrpt_a_deriv_tensor, free_buffer):
        """
        compute descrpt and descrpt_deriv last process.
        """
        compute_nums = self.nnei
        nnei_repeat_times = (compute_nums + 63) // self.nnei_once_repeat_nums

        descrpt_a_0 = res_descrpt_a_tensor[0]
        descrpt_a_1 = res_descrpt_a_tensor[nnei_repeat_times * self.nnei_once_repeat_nums]
        descrpt_a_2 = res_descrpt_a_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 2]
        descrpt_a_3 = res_descrpt_a_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 3]

        self.tik_instance.vrec(self.nnei_once_repeat_nums, descrpt_a_0, res_vec_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_a_1, neighbour_coords[0],
                               dis_dot_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_a_2, neighbour_coords[1],
                               dis_dot_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_a_3, neighbour_coords[2],
                               dis_dot_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        left_value_save = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_times * self.nnei_once_repeat_nums],
                                                   name="left_value_save", scope=self.tik.scope_ubuf)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, left_value_save, res_vec_tensor_3,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        descrpt_a_deriv_0 = res_descrpt_a_deriv_tensor[0]
        descrpt_a_deriv_1 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums]
        descrpt_a_deriv_2 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 2]
        descrpt_a_deriv_3 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 3]
        descrpt_a_deriv_4 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 4]
        descrpt_a_deriv_5 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 5]
        descrpt_a_deriv_6 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 6]
        descrpt_a_deriv_7 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 7]
        descrpt_a_deriv_8 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 8]
        descrpt_a_deriv_9 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 9]
        descrpt_a_deriv_10 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 10]
        descrpt_a_deriv_11 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 11]

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_0, descrpt_a_0,
                               dsw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_0, descrpt_a_deriv_0,
                               dis_rev_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        not_one = self.tik_instance.Scalar("float32", "not_one", init_value=-1)

        self.tik_instance.vmuls(self.nnei_once_repeat_nums, descrpt_a_deriv_0,
                                descrpt_a_deriv_0,
                                not_one,
                                nnei_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        self.tik_instance.vadd(self.nnei_once_repeat_nums, left_value_save, left_value_save,
                               descrpt_a_deriv_0,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_0, left_value_save,
                               neighbour_coords[0],
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_1, left_value_save,
                               neighbour_coords[1],
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_2, left_value_save,
                               neighbour_coords[2],
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        # compute res 456
        self.compute_descrpt_a_deriv_three(left_value_save, neighbour_coords[0],
                                           neighbour_coords[0], neighbour_coords[1], neighbour_coords[2],
                                           sw_tensor, res_vec_tensor_2, res_vec_tensor_1,
                                           descrpt_a_1, dsw_tensor, dis_rev_tensor,
                                           not_one, descrpt_a_deriv_3, descrpt_a_deriv_4,
                                           descrpt_a_deriv_5)

        # compute res 789
        self.compute_descrpt_a_deriv_three(left_value_save, neighbour_coords[1],
                                           neighbour_coords[1], neighbour_coords[0], neighbour_coords[2],
                                           sw_tensor, res_vec_tensor_2, res_vec_tensor_1,
                                           descrpt_a_2, dsw_tensor, dis_rev_tensor,
                                           not_one, descrpt_a_deriv_7, descrpt_a_deriv_6,
                                           descrpt_a_deriv_8)

        # compute res 10 11 12
        self.compute_descrpt_a_deriv_three(left_value_save, neighbour_coords[2],
                                           neighbour_coords[2], neighbour_coords[0], neighbour_coords[1],
                                           sw_tensor, res_vec_tensor_2, res_vec_tensor_1,
                                           descrpt_a_3, dsw_tensor, dis_rev_tensor,
                                           not_one, descrpt_a_deriv_11, descrpt_a_deriv_9,
                                           descrpt_a_deriv_10)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_0, descrpt_a_0,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_1, descrpt_a_1,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_2, descrpt_a_2,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_3, descrpt_a_3,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        descrpt_a_tensors = [descrpt_a_0, descrpt_a_1, descrpt_a_2, descrpt_a_3]

        self.concat_result(descrpt_a_tensors,
                           res_descrpt_a_tensor,
                           free_buffer)

    def vector_compute_process(self, dis_tensor, index_one_type, dis_dot_tensor,
                               sw_tensor, dsw_tensor, res_vec_tensor, res_vec_tensor_1,
                               res_vec_tensor_2, res_vec_tensor_3, dis_revert_tensor,
                               res_descrpt_a_deriv_tensor, sorted_coords):
        """
        the first vec compute process for descrpt and descrpt_deriv.
        """
        compute_nums = self.nnei
        nnei_mask_repeat = (compute_nums + self.nnei_once_repeat_nums - 1) // self.nnei_once_repeat_nums

        self.tik_instance.vec_dup(self.nnei_once_repeat_nums, sw_tensor, 0,
                                  nnei_mask_repeat, self.repeat_stride)

        self.tik_instance.vec_dup(self.nnei_once_repeat_nums, dsw_tensor, 0,
                                  nnei_mask_repeat, self.repeat_stride)

        self.tik_instance.vadds(compute_nums, dis_tensor,
                                dis_tensor,
                                0,
                                self.nnei_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        self.tik_instance.vadds(compute_nums, res_vec_tensor,
                                dis_tensor,
                                0,
                                self.nnei_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        self.tik_instance.vrec(self.nnei_once_repeat_nums, dis_revert_tensor, dis_tensor,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vrec(self.nnei_once_repeat_nums, res_vec_tensor_1, dis_dot_tensor,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, res_vec_tensor_2, dis_dot_tensor,
                               dis_dot_tensor,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vrec(self.nnei_once_repeat_nums, res_vec_tensor_2, res_vec_tensor_2,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, res_vec_tensor_3, res_vec_tensor_1,
                               dis_revert_tensor,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        mask_length = (compute_nums + self.nnei_once_repeat_nums - 1) // self.nnei_once_repeat_nums

        mask_tensor = self.tik_instance.Tensor("uint64", [mask_length],
                                               name="mask_tensor", scope=self.tik.scope_ubuf)

        mask_tensor_1 = self.tik_instance.Tensor("uint64", [mask_length],
                                                 name="mask_tensor_1", scope=self.tik.scope_ubuf)

        mask_tensor_2 = self.tik_instance.Tensor("uint64", [mask_length],
                                                 name="mask_tensor_2", scope=self.tik.scope_ubuf)

        mask_tensor_3 = self.tik_instance.Tensor("uint64", [mask_length],
                                                 name="mask_tensor_3", scope=self.tik.scope_ubuf)

        self.tik_instance.vcmpvs_lt(mask_tensor, res_vec_tensor, self.rcut,
                                    mask_length,
                                    self.block_stride, self.repeat_stride)

        self.tik_instance.vcmpvs_ge(mask_tensor_1, res_vec_tensor, self.rcut_smth,
                                    mask_length,
                                    self.block_stride, self.repeat_stride)

        mask_tensor_2_int16 = mask_tensor_2.reinterpret_cast_to("int16")
        mask_tensor_int16 = mask_tensor.reinterpret_cast_to("int16")

        mask_repeat_times = 1
        self.tik_instance.vnot(mask_length * 4, mask_tensor_2_int16, mask_tensor_int16,
                               mask_repeat_times,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        mask_tensor_3_int16 = mask_tensor_3.reinterpret_cast_to("int16")
        mask_tensor_1_int16 = mask_tensor_1.reinterpret_cast_to("int16")
        self.tik_instance.vnot(mask_length * 4, mask_tensor_3_int16, mask_tensor_1_int16,
                               mask_repeat_times,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vand(mask_length * 4, mask_tensor_2_int16, mask_tensor_2_int16, mask_tensor_3_int16,
                               mask_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        mask_left = self.tik_instance.Scalar(dtype="uint64", name="mask_left", init_value=0)
        mask_right = self.tik_instance.Scalar(dtype="uint64", name="mask_right", init_value=0)
        mask_right_one = self.tik_instance.Scalar(dtype="uint64", name="mask_right_one", init_value=0)
        mask_nlist_one = self.tik_instance.Scalar(dtype="uint64", name="mask_nlist_one", init_value=0)

        uu_tensor = res_descrpt_a_deriv_tensor[0]

        uu_tensor_back = res_descrpt_a_deriv_tensor[nnei_mask_repeat * self.nnei_once_repeat_nums]

        uu_tensor_pow = res_descrpt_a_deriv_tensor[nnei_mask_repeat * self.nnei_once_repeat_nums * 2]

        revert_min = self.tik_instance.Scalar(dtype="float32", name="revert_min",
                                              init_value=self.rcut * (-1))

        rev_max_sub_min = self.tik_instance.Scalar(dtype="float32", name="rev_max_sub_min",
                                                   init_value=1 / (self.rcut_smth - self.rcut))

        not_six = self.tik_instance.Scalar(dtype="float32", name="not_six",
                                           init_value=6 * (-1))

        not_twl = self.tik_instance.Scalar(dtype="float32", name="not_twl",
                                           init_value=12 * (-1))

        not_ten = self.tik_instance.Scalar(dtype="float32", name="not_ten",
                                           init_value=10 * (-1))

        fifthen = self.tik_instance.Scalar(dtype="float32", name="fifthen",
                                           init_value=15)

        with self.tik_instance.for_range(0, mask_length) as i:
            mask_right.set_as(mask_tensor_2[i])
            mask_right_one.set_as(mask_tensor[i])
            mask_nlist_one.set_as(mask_tensor_1[i])
            with self.tik_instance.if_scope(mask_right != 0):
                self.tik_instance.vadds([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                        res_vec_tensor[self.nnei_once_repeat_nums * i],
                                        revert_min,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                        uu_tensor[self.nnei_once_repeat_nums * i],
                                        rev_max_sub_min,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], sw_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                        uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                        not_six,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], uu_tensor_back[self.nnei_once_repeat_nums * i],
                                        uu_tensor[self.nnei_once_repeat_nums * i],
                                        15,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadd([mask_left, mask_right], uu_tensor_back[self.nnei_once_repeat_nums * i],
                                       uu_tensor_back[self.nnei_once_repeat_nums * i],
                                       uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadds([mask_left, mask_right], uu_tensor_back[self.nnei_once_repeat_nums * i],
                                        uu_tensor_back[self.nnei_once_repeat_nums * i],
                                        not_ten,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], sw_tensor[self.nnei_once_repeat_nums * i],
                                       sw_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor_back[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadds([mask_left, mask_right], sw_tensor[self.nnei_once_repeat_nums * i],
                                        sw_tensor[self.nnei_once_repeat_nums * i],
                                        1,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], dsw_tensor[self.nnei_once_repeat_nums * i],
                                        uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                        3,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], dsw_tensor[self.nnei_once_repeat_nums * i],
                                       dsw_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor_back[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                        uu_tensor[self.nnei_once_repeat_nums * i],
                                        not_twl,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadds([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                        uu_tensor[self.nnei_once_repeat_nums * i],
                                        fifthen,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadd([mask_left, mask_right], dsw_tensor[self.nnei_once_repeat_nums * i],
                                       dsw_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], dsw_tensor[self.nnei_once_repeat_nums * i],
                                        dsw_tensor[self.nnei_once_repeat_nums * i],
                                        rev_max_sub_min,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

            with self.tik_instance.if_scope(mask_right_one != 0):
                self.tik_instance.vadds([mask_left, mask_right_one], sw_tensor[self.nnei_once_repeat_nums * i],
                                        sw_tensor[self.nnei_once_repeat_nums * i],
                                        1,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

            with self.tik_instance.if_scope(mask_nlist_one != 0):
                self.tik_instance.vmuls([mask_left, mask_nlist_one], index_one_type[self.nnei_once_repeat_nums * i],
                                        index_one_type[self.nnei_once_repeat_nums * i],
                                        0,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadds([mask_left, mask_nlist_one], index_one_type[self.nnei_once_repeat_nums * i],
                                        index_one_type[self.nnei_once_repeat_nums * i],
                                        -1,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             sorted_coords[0][self.nnei_once_repeat_nums * i], -1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             sorted_coords[1][self.nnei_once_repeat_nums * i], -1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             sorted_coords[2][self.nnei_once_repeat_nums * i], -1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             dis_dot_tensor[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             dis_tensor[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             dis_revert_tensor[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             res_vec_tensor[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             res_vec_tensor_1[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             res_vec_tensor_2[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             res_vec_tensor_3[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)

    def extract_data_in_proposal(self, dst_proposal, dis_tensor, dis_dot_tensor,
                                 neighbour_coords, index_one_type, temp_idx):
        """
        extract data from proposal contains neighbour_coords, dis, nlist.
        """
        cur_extract_length = self.tik_instance.Scalar(self.int32_type, "cur_extract_length",
                                                      init_value=(self.cur_type_neighbour_nums + 15) // 16)
        front_tail = self.block_num - self.sel_a_list[0] % self.block_num

        dst_offset = self.tik_instance.Scalar(self.int32_type, "dst_offset",
                                              init_value=0)

        if front_tail % 8 != 0:
            with self.tik_instance.if_scope(temp_idx == 1):
                dst_offset.set_as(front_tail * 8)

        cur_move_length = self.tik_instance.Scalar(self.int32_type, "cur_move_length",
                                                   init_value=(self.cur_type_neighbour_nums + 7) // 8)

        nlist_temp_buffer = self.tik_instance.Tensor("float32", [self.max_nbor_size],
                                                     name="nlist_temp_buffer",
                                                     scope=self.tik.scope_ubuf)

        def extract_cur_index_data(index, mid_buffer, src_buffer, dst_buffer):
            """
            abstract extract data fuc.
            """
            self.tik_instance.vextract(mid_buffer[0],
                                       src_buffer[dst_offset],
                                       cur_extract_length, index)

            if front_tail % 8 != 0:
                with self.tik_instance.if_scope(temp_idx == 1):
                    with self.tik_instance.for_range(self.sel_a[0], self.sel_a[0] + front_tail) as i:
                        dst_buffer[i].set_as(src_buffer[(i - self.sel_a[0]) * 8 + index])

            self.tik_instance.vadds(cur_move_length * self.block_num, dst_buffer[self.sel_a_back[temp_idx]],
                                    mid_buffer[0],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

        index_one_type_fp32 = index_one_type.reinterpret_cast_to("float32")

        extract_cur_index_data(5, nlist_temp_buffer, dst_proposal, index_one_type_fp32)

        extract_cur_index_data(0, nlist_temp_buffer, dst_proposal, neighbour_coords[0])

        extract_cur_index_data(1, nlist_temp_buffer, dst_proposal, neighbour_coords[1])

        extract_cur_index_data(2, nlist_temp_buffer, dst_proposal, neighbour_coords[2])

        extract_cur_index_data(3, nlist_temp_buffer, dst_proposal, dis_dot_tensor)


        self.tik_instance.vextract(nlist_temp_buffer[0],
                                   dst_proposal[dst_offset],
                                   cur_extract_length, 4)

        not_one = self.tik_instance.Scalar("float32", "not_one", init_value=-1)
        front_dis_value = self.tik_instance.Scalar("float32", "front_dis_value", init_value=-1)

        if front_tail % 8 != 0:
            with self.tik_instance.if_scope(temp_idx == 1):
                with self.tik_instance.for_range(self.sel_a[0], self.sel_a[0] + front_tail) as i:
                    front_dis_value.set_as(dst_proposal[(i - self.sel_a[0]) * 8 + 4])
                    front_dis_value.set_as(front_dis_value * (-1))
                    dis_tensor[i].set_as(front_dis_value)

        self.tik_instance.vmuls(cur_move_length * 8,
                                dis_tensor[self.sel_a_back[temp_idx]],
                                nlist_temp_buffer[0], not_one,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        if self.type_num > 2:
            with self.tik_instance.for_range(self.cur_type_neighbour_nums,
                                             self.cur_type_neighbour_nums + cur_move_length * 8) as i:
                dis_dot_tensor[self.sel_a_back[temp_idx] + i].set_as(1)
                dis_tensor[self.sel_a_back[temp_idx] + i].set_as(self.rcut_smth + 1)

    def mapping_nlist(self, sorted_index, cur_sample):
        """
        mapping the nlist data to origin data.
        """
        mapping_dict = self.tik_instance.Tensor(self.type_dtype, [self.type_shape[1]],
                                                name="mapping_dict", scope=self.tik.scope_ubuf)
        mapping_offset = self.tik_instance.Scalar(self.type_dtype, "mapping_offset",
                                                  init_value=cur_sample * (1 + self.nloc * Constant.NLOC_NUM +
                                                                           self.type_shape[1]) +
                                                             1 + self.nloc * Constant.NLOC_NUM)
        mapping_burst = self.tik_instance.Scalar(self.type_dtype, "mapping_burst",
                                                 init_value=self.nall_d // self.block_num)
        self.tik_instance.data_move(mapping_dict, self.mesh_gm[mapping_offset],
                                    sid=0, nburst=1, burst=mapping_burst,
                                    src_stride=0, dst_stride=0)
        dst_inlist_length = max(self.max_nbor_size, self.nnei)
        dst_inlist = self.tik_instance.Tensor(self.type_dtype, [dst_inlist_length],
                                              name="dst_inlist", scope=self.tik.scope_ubuf)
        self.tik_instance.vmuls(self.nnei,
                                sorted_index[0],
                                sorted_index[0], 4,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")
        self.tik_instance.vgather(self.nnei,
                                  dst_inlist, mapping_dict, sorted_index,
                                  1,
                                  self.repeat_stride,
                                  0,
                                  0, "counter")
        self.tik_instance.vadds(self.nnei,
                                sorted_index[0],
                                dst_inlist[0], 0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

    def sort_distance_for_neighbour(self, dst_proposal, tensor_proposal):
        """
        sort proposal by score.
        """
        self.tik_instance.vrpsort16(dst_proposal, tensor_proposal,
                                    self.max_nbor_size // 16)

        self.tik_instance.vadds(self.max_nbor_size * self.block_num,
                                tensor_proposal[0],
                                dst_proposal[0],
                                0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        cur_range_target_nums = self.tik_instance.Scalar(self.int32_type, "cur_range_target_nums",
                                                         init_value=16 * 4)

        vector_nums = self.tik_instance.Scalar(self.int32_type, "vector_nums", init_value=0)

        src1_offset = self.tik_instance.Scalar(self.int32_type, "src1_offset", init_value=0)
        src2_offset = self.tik_instance.Scalar(self.int32_type, "src2_offset", init_value=0)
        src3_offset = self.tik_instance.Scalar(self.int32_type, "src3_offset", init_value=0)
        src4_offset = self.tik_instance.Scalar(self.int32_type, "src4_offset", init_value=0)

        dst_offset = self.tik_instance.Scalar(self.int32_type, "dst_offset", init_value=0)

        single_qune_nums = self.tik_instance.Scalar(self.int32_type, "single_qune_nums", init_value=0)
        cur_sort = self.tik_instance.Scalar(self.int32_type, "cur_sort",
                                            init_value=self.max_nbor_size + 100)

        with self.tik_instance.for_range(cur_range_target_nums, cur_sort):
            vector_nums.set_as(self.max_nbor_size // cur_range_target_nums)
            dst_offset.set_as(0)
            single_qune_nums.set_as(cur_range_target_nums // 4)

            src1_offset.set_as(0)
            src2_offset.set_as(single_qune_nums * 8)
            src3_offset.set_as(single_qune_nums * 8 * 2)
            src4_offset.set_as(single_qune_nums * 8 * 3)

            self.tik_instance.vmrgsort4(dst_proposal[dst_offset],
                                        (tensor_proposal[src1_offset:],
                                         tensor_proposal[src2_offset:],
                                         tensor_proposal[src3_offset:],
                                         tensor_proposal[src4_offset:]),
                                        (single_qune_nums, single_qune_nums,
                                         single_qune_nums, single_qune_nums),
                                        False, 15, vector_nums)

            self.tik_instance.vadds(self.max_nbor_size * self.block_num,
                                    tensor_proposal[0],
                                    dst_proposal[0],
                                    0,
                                    1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride,
                                    0, "counter")

            cur_range_target_nums.set_as(cur_range_target_nums * 4)

    def combine_proposal(self, dis_tensor, neighbour_coords, index_one_type, tensor_proposal, dis_dot_tensor):
        """
        combine dis nlist neighbour_coords data in proposal.
        """
        dis_revert_tensor = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                     name="dis_revert_tensor", scope=self.tik.scope_ubuf)

        not_one = self.tik_instance.Scalar("float32", "not_one", init_value=-1)

        self.tik_instance.vmuls(self.repeat_once_size,
                                dis_revert_tensor, dis_tensor, not_one,
                                self.max_nbor_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        with self.tik_instance.for_range(self.cur_type_neighbour_nums, self.max_nbor_size) as cur_eles:
            dis_revert_tensor[cur_eles].set_as(self.max_value)
            if self.type_num > 2:
                dis_dot_tensor[cur_eles].set_as(1)
                neighbour_coords[0][cur_eles].set_as(0)
                neighbour_coords[1][cur_eles].set_as(0)
                neighbour_coords[2][cur_eles].set_as(0)

        proposal_repeat_time = self.max_nbor_size // 16

        self.tik_instance.vconcat(tensor_proposal, dis_dot_tensor,
                                  proposal_repeat_time, 3)
        self.tik_instance.vconcat(tensor_proposal, dis_revert_tensor,
                                  proposal_repeat_time, 4)

        index_one_type_fp32 = index_one_type.reinterpret_cast_to("float32")

        self.tik_instance.vconcat(tensor_proposal, index_one_type_fp32,
                                  proposal_repeat_time, 5)

        self.tik_instance.vconcat(tensor_proposal, neighbour_coords[0],
                                  proposal_repeat_time, 0)
        self.tik_instance.vconcat(tensor_proposal, neighbour_coords[1],
                                  proposal_repeat_time, 1)
        self.tik_instance.vconcat(tensor_proposal, neighbour_coords[2],
                                  proposal_repeat_time, 2)

    def sort_index(self, index_one_type):
        """
        sort mesh data in proposal.
        """
        index_one_fp32 = index_one_type.reinterpret_cast_to("float32")
        self.tik_instance.vconv(self.nnei_once_repeat_nums, "", index_one_fp32, index_one_type,
                                self.max_nbor_size // self.nnei_once_repeat_nums,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        index_proposal = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size * 8],
                                                  name="index_proposal", scope=self.tik.scope_ubuf)
        res_proposal = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size * 8],
                                                name="res_proposal", scope=self.tik.scope_ubuf)
        index_revert = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                name="index_revert", scope=self.tik.scope_ubuf)

        self.tik_instance.vmuls(self.repeat_once_size,
                                index_revert, index_one_fp32, -1,
                                self.max_nbor_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        with self.tik_instance.for_range(self.cur_loc_nei_num, self.max_nbor_size) as cur_eles:
            index_revert[cur_eles].set_as(self.max_value)

        proposal_repeat_time = self.max_nbor_size // 16

        self.tik_instance.vconcat(index_proposal, index_revert,
                                  proposal_repeat_time, 4)

        self.sort_distance_for_neighbour(res_proposal, index_proposal)

        self.tik_instance.vextract(index_revert,
                                   res_proposal,
                                   self.max_nbor_size // 16, 4)

        self.tik_instance.vmuls(self.cur_loc_nei_num,
                                index_revert, index_revert, -1,
                                self.max_nbor_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        self.tik_instance.vconv(self.nnei_once_repeat_nums, "round", index_one_type, index_revert,
                                self.max_nbor_size // self.nnei_once_repeat_nums,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

    def compute_neighbour_dis(self, neighbour_coords, x, y, z, dis_tensor, dis_dot_tensor,
                              tensor_x, tensor_y, tensor_z):
        """
        compute neighbour distence for nloc.
        """
        x.set_as(x * -1)
        y.set_as(y * -1)
        z.set_as(z * -1)

        self.tik_instance.vadds(self.repeat_once_size,
                                neighbour_coords[0], neighbour_coords[0], x,
                                self.max_nbor_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        self.tik_instance.vadds(self.repeat_once_size,
                                neighbour_coords[1], neighbour_coords[1], y,
                                self.max_nbor_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        self.tik_instance.vadds(self.repeat_once_size,
                                neighbour_coords[2], neighbour_coords[2], z,
                                self.max_nbor_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.repeat_once_size, tensor_x, neighbour_coords[0],
                               neighbour_coords[0],
                               self.max_nbor_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.repeat_once_size, tensor_y, neighbour_coords[1],
                               neighbour_coords[1],
                               self.max_nbor_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.repeat_once_size, tensor_z, neighbour_coords[2],
                               neighbour_coords[2],
                               self.max_nbor_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vadd(self.repeat_once_size, tensor_x, tensor_x,
                               tensor_y,
                               self.max_nbor_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vadd(self.repeat_once_size, dis_dot_tensor, tensor_x,
                               tensor_z,
                               self.max_nbor_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vsqrt(self.repeat_once_size,
                                dis_tensor, dis_dot_tensor,
                                self.max_nbor_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

    def get_neighbour_coords(self, neighbour_coords, index_one_type, nums, cur_sample):
        """
        Get all neighbour coords for nloc.
        """
        gm_idx = self.tik_instance.Scalar(self.int32_type, "gm_idx", init_value=0)
        ub_idx = self.tik_instance.Scalar(self.int32_type, "ub_idx", init_value=0)

        neighbour_coord = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size * 8],
                                                   name="neighbour_coord",
                                                   scope=self.tik.scope_ubuf)
        neighbour_coord_back = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size * 8],
                                                        name="neighbour_coord_back",
                                                        scope=self.tik.scope_ubuf)

        mul3s = self.tik_instance.Tensor(self.int32_type, [self.max_nbor_size], name="muls3",
                                         scope=self.tik.scope_ubuf)

        self.tik_instance.vmuls(self.max_nbor_size,
                                mul3s, index_one_type, 3,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        coord_nsample_offset = self.tik_instance.Scalar(self.int32_type, "coord_nsample_offset",
                                                        init_value=cur_sample * self.type_shape[1] * 3)
        ub_idx.set_as(nums // 4)

        with self.tik_instance.for_range(0, ub_idx) as mesh_idx:
            gm_idx.set_as(mul3s[mesh_idx * 4])
            self.tik_instance.data_move(neighbour_coord[mesh_idx * 32], self.coord_gm[coord_nsample_offset + gm_idx],
                                        sid=0, nburst=1, burst=1,
                                        src_stride=0, dst_stride=0)

            gm_idx.set_as(mul3s[mesh_idx * 4 + 1])
            self.tik_instance.data_move(neighbour_coord[mesh_idx * 32 + 8],
                                        self.coord_gm[coord_nsample_offset + gm_idx],
                                        sid=0, nburst=1, burst=1,
                                        src_stride=0, dst_stride=0)

            gm_idx.set_as(mul3s[mesh_idx * 4 + 2])
            self.tik_instance.data_move(neighbour_coord[mesh_idx * 32 + 16],
                                        self.coord_gm[coord_nsample_offset + gm_idx],
                                        sid=0, nburst=1, burst=1,
                                        src_stride=0, dst_stride=0)

            gm_idx.set_as(mul3s[mesh_idx * 4 + 3])
            self.tik_instance.data_move(neighbour_coord[mesh_idx * 32 + 24],
                                        self.coord_gm[coord_nsample_offset + gm_idx],
                                        sid=0, nburst=1, burst=1,
                                        src_stride=0, dst_stride=0)

        with self.tik_instance.if_scope((nums % 4) != 0):
            ub_idx.set_as(nums - nums % 4)
            with self.tik_instance.for_range(ub_idx, nums) as tail_idx:
                gm_idx.set_as(mul3s[tail_idx])
                self.tik_instance.data_move(neighbour_coord[tail_idx * 8], self.coord_gm[coord_nsample_offset + gm_idx],
                                            sid=0, nburst=1, burst=1,
                                            src_stride=0, dst_stride=0)

        self.tik_instance.v4dtrans(False, neighbour_coord_back, neighbour_coord, self.max_nbor_size, 8)

        self.tik_instance.vadds(self.max_nbor_size,
                                neighbour_coords[0], neighbour_coord_back[0], 0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        self.tik_instance.vadds(self.max_nbor_size,
                                neighbour_coords[1], neighbour_coord_back[self.max_nbor_size], 0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

        self.tik_instance.vadds(self.max_nbor_size,
                                neighbour_coords[2], neighbour_coord_back[self.max_nbor_size * 2], 0,
                                1,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride,
                                0, "counter")

    def sort_for_index_by_type(self, type_idx, index_one_type, cur_type_tensor, mesh_ub):
        """
        Get all neighbours type for cur nloc.
        """
        self.tik_instance.vector_dup(self.repeat_once_size, index_one_type, -1,
                                     self.max_nbor_repeat_times, self.block_stride,
                                     self.repeat_stride)

        one_nums_bits_num = 32

        type_mask = self.tik_instance.Tensor("uint32", [self.max_nbor_size // one_nums_bits_num],
                                             name="type_mask", scope=self.tik.scope_ubuf)

        self.tik_instance.vcmpvs_eq(type_mask, cur_type_tensor, type_idx,
                                    self.max_nbor_repeat_times,
                                    self.block_stride, self.repeat_stride)

        self.tik_instance.vreduce(self.cur_loc_nei_num, index_one_type, mesh_ub,
                                  type_mask,
                                  self.max_nbor_repeat_times,
                                  1, self.repeat_stride,
                                  self.repeat_stride // self.block_num,
                                  0, self.cur_type_neighbour_nums, "counter")

    def apply_ub_tensor_phase0(self):
        """
        Apply ub buffer for mesh and type.
        """
        mesh_ub = self.tik_instance.Tensor(self.type_dtype, self.mesh_ub_shape,
                                           name="mesh_ub", scope=self.tik.scope_ubuf)

        cur_type_tensor = self.tik_instance.Tensor(self.int32_type, [self.max_nbor_size],
                                                   name="cur_type_tensor",
                                                   scope=self.tik.scope_ubuf)

        return mesh_ub, cur_type_tensor

    def apply_ub_tensor_phase1(self):
        """
        Apply ub buffer for all neighbours type.
        """
        type_ub = self.tik_instance.Tensor(self.type_dtype, [self.max_nall],
                                           name="type_ub", scope=self.tik.scope_ubuf)

        return type_ub

    def get_core_nums(self):
        """
        Get core nums for comppute process.
        """
        one_cor_nloc_num = self.tik_instance.Scalar(self.int32_type, "one_cor_nloc_num",
                                                    init_value=self.nloc_d // self.core_num_var)

        last_core_num = self.tik_instance.Scalar(self.int32_type, "last_core_num",
                                                 init_value=self.nloc_d -
                                                            one_cor_nloc_num * (self.core_num_var - 1))

        with self.tik_instance.if_scope(self.nloc_d < self.core_num_var):
            self.cur_op_core_num_scalar.set_as(self.nloc_d)
            one_cor_nloc_num.set_as(self.nloc_d - 1)
            last_core_num.set_as(1)

        return one_cor_nloc_num, last_core_num

    def compute_one_core(self, loc_coord_x, loc_coord_y, loc_coord_z, start_loc, stop_loc, block_idx, cur_sample,
                         one_cor_nloc_num):
        """
        Compute process, the main fuc.
        """
        with self.tik_instance.for_range(start_loc, stop_loc) as loc_nei_idx:
            extract_length = max(self.max_nbor_size, self.nnei)
            extract_length_align = (extract_length + self.nnei_once_repeat_nums - 1) // \
                                   self.nnei_once_repeat_nums * self.nnei_once_repeat_nums

            sorted_index = self.tik_instance.Tensor(self.type_dtype, [extract_length_align],
                                                    name="sorted_index", scope=self.tik.scope_ubuf)

            sorted_neighbour_coord_x = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                                name="sorted_neighbour_coord_x",
                                                                scope=self.tik.scope_ubuf)

            sorted_neighbour_coord_y = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                                name="sorted_neighbour_coord_y",
                                                                scope=self.tik.scope_ubuf)

            sorted_neighbour_coord_z = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                                name="sorted_neighbour_coord_z",
                                                                scope=self.tik.scope_ubuf)

            self.tik_instance.vector_dup(self.repeat_once_size, sorted_neighbour_coord_x, -1,
                                         extract_length_align // self.nnei_once_repeat_nums, self.block_stride,
                                         self.repeat_stride)

            self.tik_instance.vector_dup(self.repeat_once_size, sorted_neighbour_coord_y, -1,
                                         extract_length_align // self.nnei_once_repeat_nums, self.block_stride,
                                         self.repeat_stride)

            self.tik_instance.vector_dup(self.repeat_once_size, sorted_neighbour_coord_z, -1,
                                         extract_length_align // self.nnei_once_repeat_nums, self.block_stride,
                                         self.repeat_stride)

            sorted_dis_tensor = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                         name="sorted_dis_tensor",
                                                         scope=self.tik.scope_ubuf)

            sorted_dis_dot_tensor = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                             name="sorted_dis_dot_tensor",
                                                             scope=self.tik.scope_ubuf)

            init_dis = self.tik_instance.Scalar("float32", "init_dis",
                                                init_value=self.rcut_smth + 1)

            init_dis_dot = self.tik_instance.Scalar("float32", "init_dis",
                                                    init_value=init_dis * init_dis)

            self.tik_instance.vector_dup(self.repeat_once_size, sorted_dis_tensor, init_dis,
                                         extract_length_align // self.nnei_once_repeat_nums, self.block_stride,
                                         self.repeat_stride)

            self.tik_instance.vector_dup(self.repeat_once_size, sorted_dis_dot_tensor, init_dis_dot,
                                         extract_length_align // self.nnei_once_repeat_nums, self.block_stride,
                                         self.repeat_stride)

            cur_loc = self.tik_instance.Scalar(self.int32_type, "loc_idx",
                                               init_value=block_idx * one_cor_nloc_num + loc_nei_idx)

            for_input_cur_loc = self.tik_instance.Scalar(self.int32_type, "for_input_cur_loc",
                                                         init_value=self.nloc_offset +
                                                                    block_idx * one_cor_nloc_num + loc_nei_idx)

            cur_loc_index = self.tik_instance.Scalar(self.int32_type, "cur_loc_index",
                                                     init_value=for_input_cur_loc + 1)
            mesh_nframe_offset = self.tik_instance.Scalar(self.int32_type, "mesh_nframe_offset",
                                                          init_value=0)
            if self.mode != 3:
                mesh_nframe_offset.set_as(cur_sample * (1 + Constant.NLOC_NUM * self.nloc + self.type_shape[1]))
                self.tik_instance.data_move(sorted_index[0],
                                            self.mesh_gm[mesh_nframe_offset + cur_loc_index],
                                            sid=0, nburst=1, burst=1,
                                            src_stride=0, dst_stride=0)
            else:
                self.tik_instance.data_move(sorted_index[0], self.mesh_gm[cur_loc_index],
                                            sid=0, nburst=1, burst=1,
                                            src_stride=0, dst_stride=0)
            cur_loc_index.set_as(sorted_index[0])

            cur_loc_type_id = self.tik_instance.Scalar(self.int32_type, "cur_loc_type_id",
                                                       init_value=0)
            nei_num_index = self.tik_instance.Scalar(self.int32_type, "nei_num_index",
                                                     init_value=1 + self.total_nloc  + for_input_cur_loc)
            if self.mode != 3:
                self.tik_instance.data_move(sorted_index[0],
                                            self.mesh_gm[mesh_nframe_offset + nei_num_index],
                                            sid=0, nburst=1, burst=1,
                                            src_stride=0, dst_stride=0)
            else:
                self.tik_instance.data_move(sorted_index[0], self.mesh_gm[nei_num_index],
                                            sid=0, nburst=1, burst=1,
                                            src_stride=0, dst_stride=0)
            self.cur_loc_nei_num.set_as(sorted_index[0])

            self.tik_instance.vector_dup(self.repeat_once_size, sorted_index, -1,
                                         extract_length_align // self.nnei_once_repeat_nums, self.block_stride,
                                         self.repeat_stride)
            with self.tik_instance.if_scope(self.cur_loc_nei_num != -1):
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    mesh_ub, cur_type_tensor = self.apply_ub_tensor_phase0()
                    self.data_move_to_ub_phase0(for_input_cur_loc, mesh_ub, cur_sample)

                    if self.type_num > 2:
                        self.sort_index(mesh_ub)

                    type_ub_fp32 = cur_type_tensor.reinterpret_cast_to("float32")

                    with self.tik_instance.new_stmt_scope(disable_sync=False):
                        offsets = self.tik_instance.Tensor(self.type_dtype, [self.max_nbor_size],
                                                           name="offsets", scope=self.tik.scope_ubuf)

                        type_ub = self.apply_ub_tensor_phase1()
                        self.data_move_to_ub_phase1(cur_sample, type_ub)

                        self.tik_instance.vmuls(self.repeat_once_size,
                                                offsets, mesh_ub, 4,
                                                self.max_nbor_repeat_times,
                                                self.block_stride, self.block_stride,
                                                self.repeat_stride, self.repeat_stride)

                        self.tik_instance.vgather(self.cur_loc_nei_num,
                                                cur_type_tensor, type_ub, offsets,
                                                1,
                                                self.repeat_stride,
                                                0,
                                                0, "counter")

                        self.tik_instance.vconv(self.repeat_once_size, "", type_ub_fp32, cur_type_tensor,
                                                self.max_nbor_size // self.repeat_once_size,
                                                self.block_stride, self.block_stride,
                                                self.repeat_stride, self.repeat_stride)

                        cur_loc_type_id.set_as(type_ub[cur_loc_index])

                    with self.tik_instance.for_range(0, self.type_num) as cur_type_idx:
                        type_id = self.tik_instance.Scalar("float32", "type_id", init_value=cur_type_idx)
                        index_one_type = self.tik_instance.Tensor(self.type_dtype, [self.max_nbor_size],
                                                                  name="index_one_type",
                                                                  scope=self.tik.scope_ubuf)

                        neighbour_coord_x = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                                     name="neighbour_coord_x",
                                                                     scope=self.tik.scope_ubuf)

                        neighbour_coord_y = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                                     name="neighbour_coord_y",
                                                                     scope=self.tik.scope_ubuf)

                        neighbour_coord_z = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                                     name="neighbour_coord_z",
                                                                     scope=self.tik.scope_ubuf)

                        with self.tik_instance.new_stmt_scope(disable_sync=False):
                            self.sort_for_index_by_type(type_id, index_one_type, type_ub_fp32, mesh_ub)

                            self.get_neighbour_coords([neighbour_coord_x, neighbour_coord_y,
                                                       neighbour_coord_z],
                                                      index_one_type, self.cur_type_neighbour_nums, cur_sample)

                        temp_tensor = self.tik_instance.Tensor(self.coord_dtype, [self.block_num],
                                                               name="temp_tensor",
                                                               scope=self.tik.scope_ubuf)
                        self.tik_instance.data_move(temp_tensor[0], self.coord_gm[cur_sample * self.nall_d * 3 +
                                                                                  cur_loc_index * 3],
                                                    sid=0, nburst=1, burst=1,
                                                    src_stride=0, dst_stride=0)
                        loc_coord_x.set_as(temp_tensor[0])
                        loc_coord_y.set_as(temp_tensor[1])
                        loc_coord_z.set_as(temp_tensor[2])

                        dis_tensor = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                              name="dis_tensor",
                                                              scope=self.tik.scope_ubuf)

                        dis_dot_tensor = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                                  name="dis_dot_tensor",
                                                                  scope=self.tik.scope_ubuf)

                        with self.tik_instance.new_stmt_scope(disable_sync=False):
                            tensor_x = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                                name="tensor_x",
                                                                scope=self.tik.scope_ubuf)

                            tensor_y = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                                name="tensor_y",
                                                                scope=self.tik.scope_ubuf)

                            tensor_z = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size],
                                                                name="tensor_z",
                                                                scope=self.tik.scope_ubuf)

                            self.compute_neighbour_dis([neighbour_coord_x, neighbour_coord_y,
                                                        neighbour_coord_z], loc_coord_x,
                                                       loc_coord_y, loc_coord_z, dis_tensor,
                                                       dis_dot_tensor, tensor_x, tensor_y, tensor_z)

                        dst_proposal = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size * 8],
                                                                name="dst_proposal",
                                                                scope=self.tik.scope_ubuf)

                        with self.tik_instance.new_stmt_scope(disable_sync=False):
                            tensor_proposal = self.tik_instance.Tensor(self.coord_dtype, [self.max_nbor_size * 8],
                                                                       name="tensor_proposal",
                                                                       scope=self.tik.scope_ubuf)

                            self.combine_proposal(dis_tensor, [neighbour_coord_x, neighbour_coord_y, neighbour_coord_z],
                                                  index_one_type, tensor_proposal, dis_dot_tensor)

                            self.sort_distance_for_neighbour(dst_proposal, tensor_proposal)

                        self.extract_data_in_proposal(dst_proposal, sorted_dis_tensor, sorted_dis_dot_tensor,
                                                      [sorted_neighbour_coord_x,
                                                       sorted_neighbour_coord_y,
                                                       sorted_neighbour_coord_z],
                                                      sorted_index, cur_type_idx)

            def _rij_data_output(sorted_neighbour_coord_x, sorted_neighbour_coord_y, sorted_neighbour_coord_z):
                """
                do the copy rij data from ub to gm.
                """
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    res_rij_tensor = self.tik_instance.Tensor(self.coord_dtype, [self.nnei_align * 3],
                                                              name="res_rij_tensor",
                                                              scope=self.tik.scope_ubuf)

                    rij_trans_buffer = self.tik_instance.Tensor(self.coord_dtype,
                                                                [(self.nnei + self.repeat_once_size - 1)
                                                                 // self.repeat_once_size *
                                                                 self.repeat_once_size * 12],
                                                                name="rij_trans_buffer",
                                                                scope=self.tik.scope_ubuf)
                    with self.tik_instance.if_scope(self.cur_loc_nei_num == -1):
                        self.tik_instance.vector_dup(self.repeat_once_size, sorted_neighbour_coord_x, 0,
                                                     extract_length_align // self.nnei_once_repeat_nums,
                                                     self.block_stride,
                                                     self.repeat_stride)
                        self.tik_instance.vector_dup(self.repeat_once_size, sorted_neighbour_coord_y, 0,
                                                     extract_length_align // self.nnei_once_repeat_nums,
                                                     self.block_stride,
                                                     self.repeat_stride)
                        self.tik_instance.vector_dup(self.repeat_once_size, sorted_neighbour_coord_z, 0,
                                                     extract_length_align // self.nnei_once_repeat_nums,
                                                     self.block_stride,
                                                     self.repeat_stride)

                    self.concat_rij_result([sorted_neighbour_coord_x,
                                            sorted_neighbour_coord_y,
                                            sorted_neighbour_coord_z],
                                           res_rij_tensor, rij_trans_buffer)

                    self.data_move_rij_to_gm(res_rij_tensor, cur_sample, cur_loc, loc_nei_idx, stop_loc)

            if self.type_num == 3:
                _rij_data_output(sorted_neighbour_coord_x, sorted_neighbour_coord_y, sorted_neighbour_coord_z)

            if self.mode != 3:
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    self.mapping_nlist(sorted_index, cur_sample)

            nnei_repeat_align = (self.nnei + self.repeat_once_size - 1) // self.repeat_once_size * self.repeat_once_size
            res_descrpt_a_tensor = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_align * 4],
                                                            name="res_descrpt_a_tensor",
                                                            scope=self.tik.scope_ubuf)

            res_descrpt_a_deriv_tensor = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_align * 12],
                                                                  name="res_descrpt_a_deriv_tensor",
                                                                  scope=self.tik.scope_ubuf)
            with self.tik_instance.if_scope(self.cur_loc_nei_num == -1):
                self.tik_instance.vector_dup(self.repeat_once_size, res_descrpt_a_tensor, 0,
                                             nnei_repeat_align * 4 // self.nnei_once_repeat_nums, self.block_stride,
                                             self.repeat_stride)
                vd_repeat_times = nnei_repeat_align * 12 // self.nnei_once_repeat_nums
                max_repeat_times = 255
                if vd_repeat_times > max_repeat_times:
                    tail_repeat_times = vd_repeat_times - max_repeat_times
                    self.tik_instance.vector_dup(self.repeat_once_size, res_descrpt_a_deriv_tensor, 0,
                                                 max_repeat_times, self.block_stride,
                                                 self.repeat_stride)
                    self.tik_instance.vector_dup(self.repeat_once_size,
                                                 res_descrpt_a_deriv_tensor[max_repeat_times * self.repeat_once_size],
                                                 0,
                                                 tail_repeat_times, self.block_stride,
                                                 self.repeat_stride)
                else:
                    self.tik_instance.vector_dup(self.repeat_once_size, res_descrpt_a_deriv_tensor, 0,
                                                 vd_repeat_times, self.block_stride,
                                                 self.repeat_stride)
            with self.tik_instance.else_scope():
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    free_buffer = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_align * 12],
                                                           name="free_buffer",
                                                           scope=self.tik.scope_ubuf)

                    sw_tensor = free_buffer[0]
                    dsw_tensor = free_buffer[nnei_repeat_align]
                    res_vec_tensor = free_buffer[nnei_repeat_align * 2]
                    res_vec_tensor_1 = free_buffer[nnei_repeat_align * 3]
                    res_vec_tensor_2 = free_buffer[nnei_repeat_align * 4]
                    res_vec_tensor_3 = free_buffer[nnei_repeat_align * 5]
                    dis_revert_tensor = free_buffer[nnei_repeat_align * 6]

                    with self.tik_instance.new_stmt_scope(disable_sync=False):
                        self.vector_compute_process(sorted_dis_tensor,
                                                    sorted_index, sorted_dis_dot_tensor,
                                                    sw_tensor, dsw_tensor, res_vec_tensor, res_vec_tensor_1,
                                                    res_vec_tensor_2, res_vec_tensor_3, dis_revert_tensor,
                                                    res_descrpt_a_deriv_tensor,
                                                    [sorted_neighbour_coord_x,
                                                     sorted_neighbour_coord_y,
                                                     sorted_neighbour_coord_z])

                    with self.tik_instance.new_stmt_scope(disable_sync=False):
                        self.vector_compute_last_process(dis_revert_tensor,
                                                         [sorted_neighbour_coord_x,
                                                          sorted_neighbour_coord_y,
                                                          sorted_neighbour_coord_z],
                                                         sorted_dis_dot_tensor,
                                                         sw_tensor, dsw_tensor,
                                                         res_vec_tensor, res_vec_tensor_1,
                                                         res_vec_tensor_2, res_vec_tensor_3,
                                                         res_descrpt_a_tensor,
                                                         res_descrpt_a_deriv_tensor, free_buffer)

                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    self.do_avg_process(res_descrpt_a_tensor, res_descrpt_a_deriv_tensor, cur_loc_type_id)

            if self.type_num != 3:
                _rij_data_output(sorted_neighbour_coord_x, sorted_neighbour_coord_y, sorted_neighbour_coord_z)

            if self.nnei % 8 == 0:
                self.data_move_to_gm(res_descrpt_a_tensor,
                                     res_descrpt_a_deriv_tensor, sorted_index,
                                     cur_sample, cur_loc)
            else:
                with self.tik_instance.if_scope(loc_nei_idx != (stop_loc - 1)):
                    self.data_move_to_gm(res_descrpt_a_tensor,
                                         res_descrpt_a_deriv_tensor, sorted_index,
                                         cur_sample, cur_loc)

                with self.tik_instance.if_scope(loc_nei_idx == (stop_loc - 1)):
                    self.data_move_to_gm_last_loc(res_descrpt_a_tensor,
                                                  res_descrpt_a_deriv_tensor,
                                                  sorted_index,
                                                  cur_sample, cur_loc)

    def get_dynamic_args(self):
        """
        Get the nall and nloc value in natoms data.
        """
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            natoms_ub = self.tik_instance.Tensor(self.type_dtype, [self.block_num],
                                                 name="natoms_ub", scope=self.tik.scope_ubuf)

            self.tik_instance.data_move(natoms_ub, self.natoms_gm,
                                        sid=0, nburst=1, burst=1,
                                        src_stride=0, dst_stride=0)

            self.nloc_d.set_as(natoms_ub[0])
            self.total_nloc.set_as(natoms_ub[0])
            self.nall_d.set_as(natoms_ub[1])
            if self.mode != 3:
                self.nall_d.set_as(self.type_shape[1])
                self.nloc_d.set_as(self.nloc)
                self.total_nloc.set_as(self.nloc)
            self.nall_d.set_as((self.nall_d + self.block_num - 1) // self.block_num * self.block_num)

            if self.split_count > 1:
                if self.split_index == 0:
                    self.nloc_d.set_as(self.nloc_d - (self.nloc_d // 15 * 7))
                else:
                    self.nloc_offset.set_as(self.nloc_d - (self.nloc_d // 15 * 7))
                    self.nloc_d.set_as(self.nloc_d // 15 * 7)

            tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                        self.tiling_align // Constant.BLOCK_INT64, 0, 0)
            self.core_num_var.set_as(tiling_ub[0])

    def compute_process(self):
        """
        Get core nums and nloc nall value, then dis the compute_process to every core.
        """
        loc_coord_x = self.tik_instance.Scalar(self.coord_dtype, "loc_coord_x",
                                               init_value=0)
        loc_coord_y = self.tik_instance.Scalar(self.coord_dtype, "loc_coord_y",
                                               init_value=0)
        loc_coord_z = self.tik_instance.Scalar(self.coord_dtype, "loc_coord_z",
                                               init_value=0)

        self.get_dynamic_args()
        one_cor_nloc_num, last_core_num = self.get_core_nums()

        with self.tik_instance.for_range(0, self.nsample) as cur_sample:
            with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as block_idx:
                with self.tik_instance.if_scope(block_idx < self.cur_op_core_num_scalar - 1):
                    self.compute_one_core(loc_coord_x, loc_coord_y, loc_coord_z, 0, one_cor_nloc_num, block_idx,
                                          cur_sample, one_cor_nloc_num)

                with self.tik_instance.else_scope():
                    self.compute_one_core(loc_coord_x, loc_coord_y, loc_coord_z,
                                          0,
                                          last_core_num, block_idx, cur_sample, one_cor_nloc_num)

        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.cur_op_core_num,
        })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.coord_gm, self.type_gm, self.natoms_gm,
                                                                         self.box_gm, self.mesh_gm, self.davg_gm,
                                                                         self.dstd_gm],
                                   outputs=[self.descrpt_gm, self.descrpt_deriv_gm,
                                            self.rij_gm, self.nlist_gm],
                                   flowtable=(self.tiling_gm,), config={})


# 'pylint: disable=unused-argument
def _check_params(coord, types, natoms, box, mesh, davg, dstd, descrpt, descrpt_deriv, rij, nlist,
                  rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r, kernel_name):
    """
    Check the input args value.
    """
    coord_dtype = coord.get("dtype").lower()
    para_check.check_dtype(coord_dtype, ("float32"), param_name="coord")

    type_dtype = types.get("dtype").lower()
    para_check.check_dtype(type_dtype, ("int32"), param_name="type")

    natoms_dtype = natoms.get("dtype").lower()
    para_check.check_dtype(natoms_dtype, ("int32"), param_name="natoms")

    mesh_dtype = mesh.get("dtype").lower()
    para_check.check_dtype(mesh_dtype, ("int32"), param_name="mesh")

    davg_dtype = davg.get("dtype").lower()
    para_check.check_dtype(davg_dtype, ("float32"), param_name="davg")

    dstd_dtype = dstd.get("dtype").lower()
    para_check.check_dtype(dstd_dtype, ("float32"), param_name="dstd")

    descrpt_dtype = descrpt.get("dtype").lower()
    para_check.check_dtype(descrpt_dtype, ("float32"), param_name="descrpt")

    descrpt_deriv_dtype = descrpt_deriv.get("dtype").lower()
    para_check.check_dtype(descrpt_deriv_dtype, ("float32"), param_name="descrpt_deriv")

    rij_dtype = rij.get("dtype").lower()
    para_check.check_dtype(rij_dtype, ("float32"), param_name="rij")

    nlist_dtype = nlist.get("dtype").lower()
    para_check.check_dtype(nlist_dtype, ("int32"), param_name="nlist")

    type_shape = types.get("shape")
    para_check.check_shape(type_shape, min_rank=2, max_rank=2, param_name="type")

    mesh_shape = mesh.get("shape")
    para_check.check_shape(mesh_shape, min_rank=1, max_rank=1, param_name="mesh")

    coord_shape = coord.get("shape")
    para_check.check_shape(coord_shape, min_rank=2, max_rank=2, param_name="coord")

    davg_shape = davg.get("shape")
    para_check.check_shape(davg_shape, min_rank=2, max_rank=2, param_name="davg")

    dstd_shape = dstd.get("shape")
    para_check.check_shape(dstd_shape, min_rank=2, max_rank=2, param_name="dstd")

    natoms_shape = natoms.get("shape")
    para_check.check_shape(natoms_shape, min_rank=1, max_rank=1, min_size=3, param_name="natoms")

    nall = type_shape[1]
    max_nall_size = 32000
    if nall > max_nall_size:
        rule = "The nall value only support less than 32000."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "nall", nall)

    if any((rcut_r < 0, rcut_r_smth < 0)):
        rule = "The attributes {rcut_a, rcut_r, rcut_r_smth} can not be minus value or all 0."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "rcut_a", rcut_a)

    if rcut_r_smth > rcut_r:
        rule = "rcut_r_smth should be less than rcut_r."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "rcut_r", rcut_r)

    type_num = len(sel_a)
    if type_num == 0:
        rule = "sel_a list cant be empty."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "type_num", type_num)


# 'pylint: disable=redefined-builtin
@register_operator("ProdEnvMatA")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def prod_env_mat_a(coord, type, natoms, box, mesh, davg, dstd, descrpt, descrpt_deriv, rij, nlist,
                   rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r, split_count=1, split_index=0,
                   kernel_name="prod_env_mat_a"):
    """
    algorithm: prod_env_mat_a
    calculating distence mtr.

    Parameters
    ----------
    coord : dict. shape and dtype of input, means the all neighbour coords, only support float32
    type : dict. shape and dtype of input, means the all neighbour types, only support int32
    natoms: dict. shape and dtype of input, contains the nloc and nall value, only support int32
    box: dict. shape and dtype of input
    mesh: dict. shape and dtype of input, the input data contains the neighbour coords, only support int32.
    davg: dict. shape and dtype of input, only support float32.
    dstd: dict. shape and dtype of input, only support float32.
    descrpt: dict. shape and dtype of output, only support float32.
    descrpt_deriv: dict. shape and dtype of output, only support float32.
    rij: dict. shape and dtype of output, only support float32.
    nlist: dict. shape and dtype of output, only support int32.
    kernel_name : str cce kernel name, default value is real_div

    Returns
    -------
    None
    """
    _check_params(coord, type, natoms, box, mesh, davg, dstd, descrpt, descrpt_deriv, rij, nlist,
                  rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r, kernel_name)
    prod_env_mat_a_obj = ProdEnvMatA(coord, type, natoms, box, mesh, davg, dstd, sel_a, rcut_r, rcut_r_smth,
                                     nlist, split_count, split_index, kernel_name)

    prod_env_mat_a_obj.compute_process()
