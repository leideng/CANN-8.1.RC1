"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

trans_data_groups
"""
from functools import reduce
from typing import Iterable

from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_dynamic_input
from tbe.dsl.classifier.transdata.constants import DATA_TYPE_SIZE, REPLACE_TYPE
from tbe.dsl.classifier.transdata.constants import intrinsic_check_support
from impl.trans_data_common_func import clean_ubuf


class CST:
    """
    The class for constant
    """
    FRACTAL_Z = "FRACTAL_Z"
    N2 = 2
    N3 = 3
    N4 = 4
    N5 = 5
    N8 = 8
    N16 = 16
    N32 = 32
    N64 = 64
    N1024 = 1024

    NI = 16
    G_POS = 0
    D_POS = 1
    C1_OUT_POS = 2
    HW_POS = 3
    N_NG_POS = 4
    BYTES_PER_BLOCK = 32

    MAX_INT32 = 2 ** 31 - 1
    B32_BYTE = 4
    B16_BYTE = 2
    MAX_REPEAT = 255
    VNC_ROWS = 16
    STRIDE_LIMIT = 65535

    TILING_INFO_FZG_TO_FZ = 1
    TILING_INFO_FZ_TO_FZG = 2

    MODE_FZG_TO_FZ_ALIGN_HW = 0
    MODE_FZG_TO_FZ_ALIGN_C1OUT = 1
    MODE_FZG_TO_FZ_ALIGN_D = 2
    MODE_FZG_TO_FZ_UNALIGN_HW = 3
    MODE_FZG_TO_FZ_UNALIGN_D = 4
    MODE_FZG_TO_FZ_UNALIGN_HW_1 = 5
    MODE_FZG_TO_FZ_UNALIGN_D_1 = 6

    MODE_FZ_TO_FZG_N_N = 7
    MODE_FZ_TO_FZG_H_N = 8
    MODE_FZ_TO_FZG_C1_N = 9
    MODE_FZ_TO_FZG_D_N = 10
    MODE_FZ_TO_FZG_GROUPS_N = 11


class Message:
    """
    Message for GroupsTransData
    """

    def __init__(self, info):
        self._format, self._groups = self.analysis(info)
        self._shape, self._dtype = list(info.get("shape")), info.get("dtype")
        self._dtype = REPLACE_TYPE.get(self._dtype.lower(), self._dtype.lower())
        self._is_dynamic = is_unknown_rank_input(info) or is_dynamic_input(info)

    @property
    def format(self):
        return self._format

    @property
    def groups(self):
        return self._groups

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_dynamic(self):
        return self._is_dynamic

    @staticmethod
    def analysis(info):
        groups = info.get("sub_format", 1)
        format_ = info.get("format", None)
        return format_, groups


class Coordinate:

    def __init__(self, _iter, _stride, _offset, name):
        self._iter = _iter
        self._stride = _stride
        self._offset = _offset
        self._name = name

    def __repr__(self):
        return f'{self._name}'

    @property
    def iter(self):
        return self._iter

    @property
    def stride(self):
        return self._stride

    @property
    def offset(self):
        return self._offset


class FZ2FZCompute:
    """
        FractalZ 2 FractalZG
    """

    def __init__(self, src_info: Message, kernel_name):
        self.is_dyn = src_info.is_dynamic
        self.dtype = src_info.dtype
        self.kernel_name = kernel_name
        self.tik = tik.Tik()
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - CST.N1024
        self.type_bytes = DATA_TYPE_SIZE.get(self.dtype, None)
        self.block = int(tbe_platform.get_soc_spec("ubblock_size"))
        self.align_var = self.block // self.type_bytes
        self.vector_mask = CST.N8 * self.align_var
        self.vnchwconv_mask = CST.N16 * self.align_var

        self.input_gm = None
        self.output_gm = None
        self.tiling_gm = None
        self.tiling_shape = None

        self.input_buf = None
        self.help_a_buf = None
        self.help_b_buf = None
        self.tiling_buf = None
        self.buf_size = None
        self.data_ub = None
        self.data_ub_b16 = None

        self.dim_d = None
        self.dim_h = None
        self.dim_c = None
        self.dim_m = None
        self.dim_c1 = None
        self.dim_c0 = None
        self.dim_mx = None
        self.dim_g = None
        self.dim_c1x = None
        self.dim_nx = None
        self.lcm_e = None
        self.groups = None
        self.ori_shape = None
        self.fz_shape = None
        self.fzg_shape = None
        self.threads = None
        self.tiling_key = None
        self.ub_factor = None
        self.blk_factor = None
        self.ub_tail_factor = None
        self.blk_tail_factor = None
        self.loop_m = None
        self.loop_t = None

        self.ub_offset = None
        self.core_step_in = None
        self.core_step_out = None
        self.max_n_per_row = None
        self.g = None
        self.d = None
        self.c1_in = None
        self.c1_out = None
        self.c = None
        self.c0 = None
        self.n_ng = None
        self.n_e_align = None
        self.n_gp_align = None
        self.hw = None
        self.mc_pos = None
        self.nlc_g_lp_cnt = None
        self.nlc_d_lp_cnt = None
        self.nlc_c1_out_lp_cnt = None
        self.nlc_hw_lp_cnt = None
        self.nlc_n_ng_lp_cnt = None
        self.lc_g_lp_cnt = None
        self.lc_d_lp_cnt = None
        self.lc_c1_out_lp_cnt = None
        self.lc_hw_lp_cnt = None
        self.lc_n_ng_lp_cnt = None
        self.hw_lp_unit = None
        self.n_ng_lp_unit = None
        self.c1_out_lp_unit = None
        self.d_lp_unit = None
        self.g_lp_cnt = None
        self.d_lp_cnt = None
        self.c1_out_lp_cnt = None
        self.hw_lp_cnt = None
        self.n_ng_lp_cnt = None
        self.cur_hw_lp_unit = None
        self.cur_n_ng_lp_unit = None
        self.cur_d_lp_unit = None
        self.cur_c1_out_lp_unit = None
        self.c0_parts = None
        self.vnc_src_stride = None
        self.vnc_dst_stride = None
        self.e_in_offset = None
        self.out_ub_offset = None

        self.n = None
        self.e = None
        self.src_lower_n = None
        self.src_upper_n = None
        self.dst_lower_n = None
        self.dst_upper_n = None
        self.is_c0_not_jump = None
        self.c1_gap = None
        self.c0_gap = None
        self.max_elem_cnt = None
        self._check()

    @staticmethod
    def prod(shape):
        if shape == []:
            return 1
        return reduce(lambda x, y: x * y, shape)

    @staticmethod
    def ceil_div(a, b):
        return (a + b - 1) // b

    @staticmethod
    def calc_coordinate(shape: Iterable[Coordinate]):
        base = 0
        for i in shape:
            base += i.iter * i.stride + i.offset
        return base

    @staticmethod
    def _equal(self, a, b):
        if isinstance(a, tik.Scalar) and isinstance(b, tik.Scalar):
            return a.name == b.name
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        raise RuntimeError("Inputs is illegal in _scalar_equal")

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self._do_compute()
        tbe_context.get_context().add_compile_info("core_num", self.core_nums)
        tbe_context.get_context().add_compile_info("ub_size", self.buf_size)
        tbe_context.get_context().add_compile_info("max_elem_cnt", self.max_elem_cnt)
        tbe_context.get_context().add_compile_info("elem_per_block", self.align_var)
        tbe_context.get_context().add_compile_info("is_fz2fz", True)
        fatbin = None
        if self.is_dyn:
            fatbin = {"tiling_key": [self.tiling_key],
                      "tiling_key_value": [[CST.MODE_FZG_TO_FZ_ALIGN_HW], [CST.MODE_FZG_TO_FZ_ALIGN_C1OUT],
                                           [CST.MODE_FZG_TO_FZ_ALIGN_D], [CST.MODE_FZG_TO_FZ_UNALIGN_HW],
                                           [CST.MODE_FZG_TO_FZ_UNALIGN_D], [CST.MODE_FZG_TO_FZ_UNALIGN_HW_1],
                                           [CST.MODE_FZG_TO_FZ_UNALIGN_D_1],
                                           [CST.MODE_FZ_TO_FZG_N_N], [CST.MODE_FZ_TO_FZG_H_N],
                                           [CST.MODE_FZ_TO_FZG_C1_N], [CST.MODE_FZ_TO_FZG_D_N],
                                           [CST.MODE_FZ_TO_FZG_GROUPS_N]]}

        opt_config = {"enable_const_fold": True, "dynamic_tik": True}
        self.tik.BuildCCE(kernel_name=self.kernel_name,
                          inputs=[self.input_gm, ],
                          outputs=[self.output_gm, ],
                          flowtable=[self.tiling_gm, ],
                          config=opt_config,
                          extend_params={"build_multi_kernels": fatbin} if fatbin else None)
        return self.tik

    def _check(self):
        if self.block != CST.N32:
            raise RuntimeError("only support ubblock_size is 32 but is %d", self.block)
        if self.type_bytes not in [CST.B32_BYTE, CST.B16_BYTE]:
            raise RuntimeError("only support B32 and B16, but is %d", self.type_bytes)

    def _set_gm_buffer(self):
        self.tiling_gm = self.tik.Tensor("int64", (CST.N32,), name="tiling_gm", scope=tik.scope_gm)
        self.input_gm = self.tik.Tensor(self.dtype, (CST.MAX_INT32,), name="input_gm", scope=tik.scope_gm)
        self.output_gm = self.tik.Tensor(self.dtype, (CST.MAX_INT32,), name="output_gm", scope=tik.scope_gm,
                                         is_atomic_add=True)

    def _set_ub_buffer(self):
        # set buffer
        self.buf_size = self.ub_size_bytes // self.type_bytes // CST.N3
        self.input_buf = self.tik.Tensor(self.dtype, (self.buf_size,), name="input", scope=tik.scope_ubuf)
        self.help_a_buf = self.tik.Tensor(self.dtype, (self.buf_size,), name="helpA", scope=tik.scope_ubuf)
        self.help_b_buf = self.tik.Tensor(self.dtype, (self.buf_size,), name="helpB", scope=tik.scope_ubuf)

    def _set_ub_buffer_fzg2fz(self):
        # set buffer
        self.max_elem_cnt = self.ub_size_bytes // self.type_bytes
        self.data_ub = self.tik.Tensor(self.dtype, (self.max_elem_cnt,), name="data_ub", scope=tik.scope_ubuf)
        self.data_ub_b16 = self.data_ub.reinterpret_cast_to("float16")

    def _set_tiling_args(self):
        # OriginShape must be [D,H,C,M]
        self.dim_d = self._malloc_scalar("int64", "dimD_")
        self.dim_h = self._malloc_scalar("int64", "dimH_")
        self.dim_c = self._malloc_scalar("int64", "dimC_")
        self.dim_m = self._malloc_scalar("int64", "dimM_")
        # FZShape must be [D,C1,H,Mx,C0]
        self.dim_c1 = self._malloc_scalar("int64", "dimC1_")
        self.dim_c0 = self._malloc_scalar("int64", "dimC0_")
        self.dim_mx = self._malloc_scalar("int64", "dimMx_")
        # FzGShape must be [G,D,C1x,H,Nx,C0]
        self.dim_g = self._malloc_scalar("int64", "dimG_")
        self.dim_c1x = self._malloc_scalar("int64", "dimC1x_")
        self.dim_nx = self._malloc_scalar("int64", "dimNx_")
        self.lcm_e = self._malloc_scalar("int64", "lcmE_")
        self.groups = self._malloc_scalar("int64", "groups_")
        # Shapes
        self.ori_shape = [self.dim_d, self.dim_h, self.dim_c, self.dim_m]
        self.fz_shape = [self.dim_d, self.dim_c1, self.dim_h, self.dim_mx, self.dim_c0]
        self.fzg_shape = [self.dim_g, self.dim_d, self.dim_c1x, self.dim_h, self.dim_nx, self.dim_c0]
        # Tiling
        self.ub_factor = self._malloc_scalar("int64", "ub_factor_")
        self.blk_factor = self._malloc_scalar("int64", "blk_factor_")
        self.ub_tail_factor = self._malloc_scalar("int64", "ub_tail_factor_")
        self.blk_tail_factor = self._malloc_scalar("int64", "blk_tail_factor_")
        self.loop_m = self._malloc_scalar("int64", "loopM_")  # ub_outer
        self.loop_t = self._malloc_scalar("int64", "loopT_")  # blk_outer

    def _set_tiling_args_fzg2fz(self):
        self.ub_offset = self._malloc_scalar("int64", "ub_offset_")
        self.core_step_in = self._malloc_scalar("int64", "core_step_in_")
        self.core_step_out = self._malloc_scalar("int64", "core_step_out_")
        self.max_n_per_row = self._malloc_scalar("int64", "max_n_per_row_")

        self.g = self._malloc_scalar("int64", "g_")
        self.d = self._malloc_scalar("int64", "d_")
        self.c1_in = self._malloc_scalar("int64", "c1_in_")
        self.c1_out = self._malloc_scalar("int64", "c1_out_")
        self.c = self._malloc_scalar("int64", "c_")
        self.c0 = self._malloc_scalar("int64", "c0_")
        self.e = self._malloc_scalar("int64", "e_")
        self.n_ng = self._malloc_scalar("int64", "n_ng_")
        self.n_e_align = self._malloc_scalar("int64", "n_e_align_")
        self.n_gp_align = self._malloc_scalar("int64", "n_gp_align_")
        self.hw = self._malloc_scalar("int64", "hw_")

        self.mc_pos = self._malloc_scalar("int64", "mc_pos_")
        self.nlc_g_lp_cnt = self._malloc_scalar("int64", "nlc_g_lp_cnt_")
        self.nlc_d_lp_cnt = self._malloc_scalar("int64", "nlc_d_lp_cnt_")
        self.nlc_c1_out_lp_cnt = self._malloc_scalar("int64", "nlc_c1_out_lp_cnt_")
        self.nlc_hw_lp_cnt = self._malloc_scalar("int64", "nlc_hw_lp_cnt_")
        self.nlc_n_ng_lp_cnt = self._malloc_scalar("int64", "nlc_n_ng_lp_cnt_")
        self.lc_g_lp_cnt = self._malloc_scalar("int64", "lc_g_lp_cnt_")
        self.lc_d_lp_cnt = self._malloc_scalar("int64", "lc_d_lp_cnt_")
        self.lc_c1_out_lp_cnt = self._malloc_scalar("int64", "lc_c1_out_lp_cnt_")
        self.lc_hw_lp_cnt = self._malloc_scalar("int64", "lc_hw_lp_cnt_")
        self.lc_n_ng_lp_cnt = self._malloc_scalar("int64", "lc_n_ng_lp_cnt_")
        self.hw_lp_unit = self._malloc_scalar("int64", "hw_lp_unit_")
        self.n_ng_lp_unit = self._malloc_scalar("int64", "n_ng_lp_unit_")
        self.c1_out_lp_unit = self._malloc_scalar("int64", "c1_out_lp_unit_")
        self.d_lp_unit = self._malloc_scalar("int64", "d_lp_unit_")

        self.g_lp_cnt = self._malloc_scalar("int64", "g_lp_cnt")
        self.d_lp_cnt = self._malloc_scalar("int64", "d_lp_cnt")
        self.c1_out_lp_cnt = self._malloc_scalar("int64", "c1_out_lp_cnt")
        self.hw_lp_cnt = self._malloc_scalar("int64", "hw_lp_cnt")
        self.n_ng_lp_cnt = self._malloc_scalar("int64", "n_ng_lp_cnt")
        self.cur_hw_lp_unit = self._malloc_scalar("int64", "cur_hw_lp_unit")
        self.cur_n_ng_lp_unit = self._malloc_scalar("int64", "cur_n_ng_lp_unit")
        self.cur_d_lp_unit = self._malloc_scalar("int64", "cur_d_lp_unit")
        self.cur_c1_out_lp_unit = self._malloc_scalar("int64", "cur_c1_out_lp_unit")
        self.c0_parts = self._malloc_scalar("int64", "c0_parts", 1)
        self.vnc_src_stride = self._malloc_scalar("int64", "vnc_src_stride")
        self.vnc_dst_stride = self._malloc_scalar("int64", "vnc_dst_stride")
        self.e_in_offset = self._malloc_scalar("int64", "e_in_offset")
        self.out_ub_offset = self._malloc_scalar("int64", "out_ub_offset")

    def _init_tiling(self):
        params = [self.dim_d, self.dim_h, self.dim_c, self.dim_m, self.dim_c1,
                  self.dim_c0, self.dim_mx, self.dim_g, self.dim_c1x, self.dim_nx, self.lcm_e, self.groups,
                  self.ub_factor, self.blk_factor, self.ub_tail_factor, self.blk_tail_factor, self.loop_m, self.loop_t]
        self.tik.data_move(self.tiling_buf, self.tiling_gm, 0, 1, CST.N5, 0, 0)
        for key, value in enumerate(params):
            value.set_as(self.tiling_buf[key + 2])

    def _init_tiling_fzg2fz(self):
        params = [self.ub_offset, self.core_step_in, self.core_step_out,
                  self.max_n_per_row, self.g, self.d, self.c1_in, self.c1_out, self.c, self.c0, self.e,
                  self.n_ng, self.n_e_align, self.n_gp_align, self.hw, self.mc_pos, self.nlc_g_lp_cnt,
                  self.nlc_d_lp_cnt, self.nlc_c1_out_lp_cnt, self.nlc_hw_lp_cnt, self.nlc_n_ng_lp_cnt,
                  self.lc_g_lp_cnt, self.lc_d_lp_cnt, self.lc_c1_out_lp_cnt, self.lc_hw_lp_cnt,
                  self.lc_n_ng_lp_cnt, self.hw_lp_unit, self.n_ng_lp_unit, self.c1_out_lp_unit, self.d_lp_unit]
        self.tik.data_move(self.tiling_buf, self.tiling_gm, 0, 1, CST.N8, 0, 0)
        for key, value in enumerate(params):
            value.set_as(self.tiling_buf[key + 2])


    def _malloc_scalar(self, dtype, name, value=None):
        return self.tik.Scalar(dtype, name, init_value=value) if value is not None else self.tik.Scalar(dtype, name)

    def _addressing_core(self, block_idx, block_split_idx):
        self.n = self.dim_m // self.groups
        self.tiling_shape = [self.groups, self.dim_d, self.dim_c1, self.dim_h, self.n, self.dim_c0]
        _shape = self.tiling_shape[: block_split_idx] + [self.loop_t, ]
        return [block_idx // self.prod(_shape[key + 1:]) % value for key, value in enumerate(_shape)]

    def _calc_jump_params(self, group_idx):
        self.e = group_idx % self.lcm_e
        self.src_lower_n, self.src_upper_n = group_idx * self.n, (group_idx + 1) * self.n
        self.dst_lower_n, self.dst_upper_n = self.e * self.n, (self.e + 1) * self.n
        self.is_c0_not_jump = (self.e * self.dim_c) % self.dim_c0 == 0
        self.c1_gap = self.e * self.dim_c // self.dim_c0
        self.c0_gap = self.e * self.dim_c % self.dim_c0

    def _vec_dup(self, num, buf, address, number=0):
        inst_bound = CST.MAX_REPEAT * self.vector_mask
        dup_repeat_merchant = num // inst_bound
        dup_repeat_remainder = num % inst_bound
        dst_blk_stride = 1
        dst_rep_stride = 8

        with self.tik.for_range(0, dup_repeat_merchant) as i:
            self.tik.vector_dup(self.vector_mask,
                                buf[address + i * inst_bound],
                                number,
                                CST.MAX_REPEAT,
                                dst_blk_stride,
                                dst_rep_stride)

        with self.tik.if_scope(dup_repeat_remainder != 0):
            repeats = dup_repeat_remainder // self.vector_mask
            dup_remainder = dup_repeat_remainder % self.vector_mask
            with self.tik.if_scope(repeats != 0):
                self.tik.vector_dup(self.vector_mask,
                                    buf[address + dup_repeat_merchant * inst_bound],
                                    number,
                                    repeats,
                                    dst_blk_stride,
                                    dst_rep_stride)
            with self.tik.if_scope(dup_remainder != 0):
                self.tik.vector_dup(dup_remainder,
                                    buf[address + dup_repeat_merchant * inst_bound + repeats * self.vector_mask],
                                    number,
                                    1,
                                    dst_blk_stride,
                                    dst_rep_stride)

    def _transpose_mc02c0m(self, src_buf, dst_buf, m, c):
        """
        Customization:
        1. 910A[FP16], m' is 16X, c is 16.
        2. 910A[FP32], m' is 16X, c is 16.
        3. 910B[FP16], m' is 16X, c is 16.
        4. 910B[FP32], m' is 16X, c is 8.
        5. avoid bank conflict, make return contain invalid data in m'(m' >= m).
        """
        src_buf = src_buf.reinterpret_cast_to("float16")
        dst_buf = dst_buf.reinterpret_cast_to("float16")
        new_c = c * self.type_bytes // CST.B16_BYTE
        new_vnc_mask = self.vnchwconv_mask * self.type_bytes // CST.B16_BYTE
        align_n = self.block // CST.B16_BYTE
        align_m = new_vnc_mask // align_n
        new_m = self.ceil_div(m, align_m) * align_m

        loop = self._malloc_scalar("int32", name="vnc_loop_", value=self.ceil_div(new_c, align_n))
        repeat = self._malloc_scalar("int32", name="vnc_repeat_", value=new_m // align_m)
        with self.tik.if_scope(repeat % 2 == 0):
            repeat.set_as(repeat + 1)

        src_stride = self._malloc_scalar("int32", name="src_stride", value=align_m * new_c * CST.B16_BYTE // self.block)
        dst_stride = self._malloc_scalar("int32", name="dst_stride", value=align_m * CST.B16_BYTE // self.block)
        with self.tik.if_scope(repeat == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)

        with self.tik.for_range(0, loop) as j:
            src_list = [src_buf[i * new_c + j * align_n] for i in range(16)]
            dst_list = [dst_buf[i * repeat * align_m + j * repeat * align_m * align_n] for i in range(16)]
            self.tik.vnchwconv(False, False, dst_list, src_list, repeat, dst_stride, src_stride)
        return [c, repeat * align_m]

    def _transpose_c0m2mc0(self, src_buf, dst_buf, c, m):
        """
        Customization:
        1. 910A[FP16], m is 16X, c is 16.
        2. 910A[FP32], m is 16X, c is 16.
        3. 910B[FP16], m is 16X, c is 16.
        4. 910B[FP32], m is 16X, c is 8.
        5. existed bank-conflict depends on m.
        """
        src_buf = src_buf.reinterpret_cast_to("float16")
        dst_buf = dst_buf.reinterpret_cast_to("float16")
        new_c = c * self.type_bytes // CST.B16_BYTE
        new_vnc_mask = self.vnchwconv_mask * self.type_bytes // CST.B16_BYTE
        align_n = self.block // CST.B16_BYTE
        align_m = new_vnc_mask // align_n
        new_m = self.ceil_div(m, align_m) * align_m

        loop = self._malloc_scalar("int32", name="vnc_loop_", value=self.ceil_div(new_c, align_n))
        repeat = self._malloc_scalar("int32", name="vnc_repeat_", value=new_m // align_m)
        src_stride = self._malloc_scalar("int32", name="src_stride", value=align_m * CST.B16_BYTE // self.block)
        dst_stride = self._malloc_scalar("int32", name="dst_stride", value=align_m * new_c * CST.B16_BYTE // self.block)
        with self.tik.if_scope(repeat == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)

        with self.tik.for_range(0, loop) as j:
            src_list = [src_buf[i * new_m + j * new_m * align_n] for i in range(16)]
            dst_list = [dst_buf[i * new_c + j * align_n] for i in range(16)]
            self.tik.vnchwconv(False, False, dst_list, src_list, repeat, dst_stride, src_stride)
        return [m, c]

    def _dispatching0(self, blk_inner=None, function=None, model="mainCore"):
        # Reg: ub\block split same axes
        if model == "mainCore":
            function(bound=[0, blk_inner], ub_inner=self.ub_factor)
        else:
            function(bound=[0, blk_inner - 1], ub_inner=self.ub_factor)
            function(bound=[blk_inner - 1, blk_inner], ub_inner=self.ub_tail_factor)

    def _dispatching_1(self, blk_inner=None, function=None, model="mainCore"):
        # Reg: ub\block split diff axes
        function(bound=[0, blk_inner], ub_bound=[0, self.loop_m - 1], ub_inner=self.ub_factor)
        function(bound=[0, blk_inner], ub_bound=[self.loop_m - 1, self.loop_m], ub_inner=self.ub_tail_factor)

    def _dispatching_2(self, function=None):
        # Reg: ub\block split diff axes without blk_inner
        function(ub_bound=[0, self.loop_m - 1], ub_inner=self.ub_factor)
        function(ub_bound=[self.loop_m - 1, self.loop_m], ub_inner=self.ub_tail_factor)

    def _schedule_process(self, blk_outer, dispatch, process):
        # process: do_step_1_1, do_step_2_1, do_step_2_2, do_step_2_3
        with self.tik.if_scope(self.is_c0_not_jump):
            # Step_1_1: deal value
            with self.tik.if_scope(blk_outer < self.loop_t - 1):
                dispatch(blk_inner=self.blk_factor, function=process[0], model="mainCore")
            with self.tik.else_scope():
                dispatch(blk_inner=self.blk_tail_factor, function=process[0], model="tailCore")
        with self.tik.else_scope():
            # Step_2_1: concat zero and c1
            # Step_2_2: concat c1 - 2 and c1 - 1
            # Step_2_3: concat c1 - 1 and zero
            with self.tik.if_scope(blk_outer < self.loop_t - 1):
                dispatch(blk_inner=self.blk_factor, function=process[1], model="mainCore")
                dispatch(blk_inner=self.blk_factor, function=process[CST.N2], model="mainCore")
                dispatch(blk_inner=self.blk_factor, function=process[CST.N3], model="mainCore")
            with self.tik.else_scope():
                dispatch(blk_inner=self.blk_tail_factor, function=process[1], model="tailCore")
                dispatch(blk_inner=self.blk_tail_factor, function=process[CST.N2], model="tailCore")
                dispatch(blk_inner=self.blk_tail_factor, function=process[CST.N3], model="tailCore")

    def _ub_split_n_step_1_1(self, src_iter, dst_iter, ub_inner):
        input_addr = self.calc_coordinate(src_iter)
        output_addr = self.calc_coordinate(dst_iter)
        burst = ub_inner * self.dim_c0 * self.type_bytes // self.block
        self.tik.data_move(self.input_buf[0], self.input_gm[input_addr], 0, 1, burst, 0, 0)
        self.tik.data_move(self.output_gm[output_addr], self.input_buf[0], 0, 1, burst, 0, 0)

    def _ub_split_n_step_2_1(self, src_iter, dst_iter, ub_inner):
        input_addr = self.calc_coordinate(src_iter)
        output_addr = self.calc_coordinate(dst_iter)
        burst = ub_inner * self.dim_c0 * self.type_bytes // self.block
        self.tik.data_move(self.input_buf[0], self.input_gm[input_addr], 0, 1, burst, 0, 0)

        _, var_m = self._transpose_mc02c0m(self.input_buf, self.help_a_buf, ub_inner, self.dim_c0)
        with self.tik.new_stmt_scope(disable_sync=True):
            self._vec_dup(self.c0_gap * var_m, self.help_b_buf, 0)
            burst_b = (self.dim_c0 - self.c0_gap) * var_m // self.align_var
            self.tik.data_move(self.help_b_buf[self.c0_gap * var_m], self.help_a_buf[0], 0, 1, burst_b, 0, 0)
        _, _ = self._transpose_c0m2mc0(self.help_b_buf, self.help_a_buf, self.dim_c0, var_m)
        self.tik.data_move(self.output_gm[output_addr], self.help_a_buf[0], 0, 1, burst, 0, 0)

    def _ub_split_n_step_2_2(self, src_iter0, src_iter1, dst_iter, ub_inner):
        input_addr0 = self.calc_coordinate(src_iter0)
        input_addr1 = self.calc_coordinate(src_iter1)
        burst = ub_inner * self.dim_c0 * self.type_bytes // self.block
        self.tik.data_move(self.input_buf[0], self.input_gm[input_addr0], 0, 1, burst, 0, 0)
        self.tik.data_move(self.help_a_buf[0], self.input_gm[input_addr1], 0, 1, burst, 0, 0)

        _, var_m = self._transpose_mc02c0m(self.input_buf, self.help_b_buf, ub_inner, self.dim_c0)
        _, _ = self._transpose_mc02c0m(self.help_a_buf, self.input_buf, ub_inner, self.dim_c0)

        with self.tik.new_stmt_scope(disable_sync=True):
            burst_b, addr_b = self.c0_gap * var_m // self.align_var, (self.dim_c0 - self.c0_gap) * var_m
            self.tik.data_move(self.help_a_buf[0], self.help_b_buf[addr_b], 0, 1, burst_b, 0, 0)
            burst_i, addr_a = (self.dim_c0 - self.c0_gap) * var_m // self.align_var, self.c0_gap * var_m
            self.tik.data_move(self.help_a_buf[addr_a], self.input_buf[0], 0, 1, burst_i, 0, 0)

        _, _ = self._transpose_c0m2mc0(self.help_a_buf, self.help_b_buf, self.dim_c0, var_m)

        output_addr = self.calc_coordinate(dst_iter)
        self.tik.data_move(self.output_gm[output_addr], self.help_b_buf[0], 0, 1, burst, 0, 0)

    def _ub_split_n_step_2_3(self, src_iter, dst_iter, ub_inner):
        input_addr = self.calc_coordinate(src_iter)
        output_addr = self.calc_coordinate(dst_iter)
        burst = ub_inner * self.dim_c0 * self.type_bytes // self.block
        self.tik.data_move(self.input_buf[0], self.input_gm[input_addr], 0, 1, burst, 0, 0)

        _, var_m = self._transpose_mc02c0m(self.input_buf, self.help_a_buf, ub_inner, self.dim_c0)
        with self.tik.new_stmt_scope(disable_sync=True):
            self._vec_dup((self.dim_c0 - self.c0_gap) * var_m, self.help_b_buf, self.c0_gap * var_m)
            burst_a = self.c0_gap * var_m // self.align_var
            addr_a = (self.dim_c0 - self.c0_gap) * var_m
            self.tik.data_move(self.help_b_buf[0], self.help_a_buf[addr_a], 0, 1, burst_a, 0, 0)
        _, _ = self._transpose_c0m2mc0(self.help_b_buf, self.help_a_buf, self.dim_c0, var_m)
        self.tik.data_move(self.output_gm[output_addr], self.help_a_buf[0], 0, 1, burst, 0, 0)

    def _do_init_loop_cnt(self, blk_idx):
        with self.tik.if_scope(blk_idx != self.threads - 1):
            self.g_lp_cnt.set_as(self.nlc_g_lp_cnt)
            self.d_lp_cnt.set_as(self.nlc_d_lp_cnt)
            self.c1_out_lp_cnt.set_as(self.nlc_c1_out_lp_cnt)
            self.hw_lp_cnt.set_as(self.nlc_hw_lp_cnt)
            self.n_ng_lp_cnt.set_as(self.nlc_n_ng_lp_cnt)
        with self.tik.else_scope():
            self.g_lp_cnt.set_as(self.lc_g_lp_cnt)
            self.d_lp_cnt.set_as(self.lc_d_lp_cnt)
            self.c1_out_lp_cnt.set_as(self.lc_c1_out_lp_cnt)
            self.hw_lp_cnt.set_as(self.lc_hw_lp_cnt)
            self.n_ng_lp_cnt.set_as(self.lc_n_ng_lp_cnt)

    def _get_current_n_ng_unit(self, block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx):
        with self.tik.if_scope(tik.all(n_ng_lp_idx == self.n_ng_lp_cnt - 1, self.n_ng % self.n_ng_lp_unit > 0)):
            with self.tik.if_scope(
                    tik.any(self.mc_pos != CST.N_NG_POS,
                            tik.all(block_idx == self.threads - 1, self.mc_pos == CST.N_NG_POS))):
                self.cur_n_ng_lp_unit.set_as(self.n_ng % self.n_ng_lp_unit)
            with self.tik.else_scope():
                self.cur_n_ng_lp_unit.set_as(self.n_ng_lp_unit)
        with self.tik.else_scope():
            self.cur_n_ng_lp_unit.set_as(self.n_ng_lp_unit)

        with self.tik.if_scope(
                tik.all(
                    self.g * self.n_ng % CST.NI > 0,
                    tik.all(
                        n_ng_lp_idx == self.n_ng_lp_cnt - 1,
                        tik.any(self.mc_pos != CST.N_NG_POS,
                                tik.all(block_idx == self.threads - 1, self.mc_pos == CST.N_NG_POS))),
                    tik.any(
                        tik.all(self.mc_pos == CST.G_POS,
                                (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx == self.g - 1),
                        tik.all(self.mc_pos != CST.G_POS, g_lp_idx * self.e + e_lp_idx == self.g - 1)))):
            self.cur_n_ng_lp_unit.set_as(self.cur_n_ng_lp_unit + CST.NI - self.g * self.n_ng % CST.NI)

    def _get_current_hw_unit(self, block_idx, hw_lp_idx):
        with self.tik.if_scope(tik.all(hw_lp_idx == self.hw_lp_cnt - 1, self.hw % self.hw_lp_unit > 0)):
            with self.tik.if_scope(
                    tik.any(self.mc_pos != CST.HW_POS,
                            tik.all(block_idx == self.threads - 1, self.mc_pos == CST.HW_POS))):
                self.cur_hw_lp_unit.set_as(self.hw % self.hw_lp_unit)
            with self.tik.else_scope():
                self.cur_hw_lp_unit.set_as(self.hw_lp_unit)
        with self.tik.else_scope():
            self.cur_hw_lp_unit.set_as(self.hw_lp_unit)

    def _get_current_c1_out_unit(self, block_idx, c1_out_lp_idx):
        with self.tik.if_scope(
                tik.all(c1_out_lp_idx == self.c1_out_lp_cnt - 1, self.c1_out % self.c1_out_lp_unit > 0)):
            with self.tik.if_scope(
                    tik.any(self.mc_pos != CST.C1_OUT_POS,
                            tik.all(block_idx == self.threads - 1, self.mc_pos == CST.C1_OUT_POS))):
                self.cur_c1_out_lp_unit.set_as(self.c1_out % self.c1_out_lp_unit)
            with self.tik.else_scope():
                self.cur_c1_out_lp_unit.set_as(self.c1_out_lp_unit)
        with self.tik.else_scope():
            self.cur_c1_out_lp_unit.set_as(self.c1_out_lp_unit)

    def _get_current_d_unit(self, block_idx, d_lp_idx):
        with self.tik.if_scope(tik.all(d_lp_idx == self.d_lp_cnt - 1, self.d % self.d_lp_unit > 0)):
            with self.tik.if_scope(
                    tik.any(self.mc_pos != CST.D_POS,
                            tik.all(block_idx == self.threads - 1, self.mc_pos == CST.D_POS))):
                self.cur_d_lp_unit.set_as(self.d % self.d_lp_unit)
            with self.tik.else_scope():
                self.cur_d_lp_unit.set_as(self.d_lp_unit)
        with self.tik.else_scope():
            self.cur_d_lp_unit.set_as(self.d_lp_unit)

    def _set_vnchwconv_stride(self, repeat_cnt, src_val, dst_val):
        """
        set source and target stride for vnchwconv
        """
        with self.tik.if_scope(repeat_cnt == 1):
            self.vnc_src_stride.set_as(0)
            self.vnc_dst_stride.set_as(0)
        with self.tik.else_scope():
            self.vnc_src_stride.set_as(src_val)
            self.vnc_dst_stride.set_as(dst_val)

    def _get_c0_parts(self, block_idx, e_lp_idx, c1_out_lp_idx):
        with self.tik.if_scope(
                tik.any(
                    tik.all(
                        self.mc_pos != CST.C1_OUT_POS,
                        tik.any(
                            tik.all(c1_out_lp_idx < self.c1_out - 1, e_lp_idx * self.c % self.c0 > 0),
                            tik.all(c1_out_lp_idx == self.c1_out - 1,
                                    e_lp_idx * self.c % self.c0 + self.c % self.c0 > self.c0))),
                    tik.all(
                        self.mc_pos == CST.C1_OUT_POS,
                        tik.any(
                            tik.all(block_idx * self.nlc_c1_out_lp_cnt + c1_out_lp_idx < self.c1_out - 1,
                                    e_lp_idx * self.c % self.c0 > 0),
                            tik.all(block_idx * self.nlc_c1_out_lp_cnt + c1_out_lp_idx == self.c1_out - 1,
                                    e_lp_idx * self.c % self.c0 + self.c % self.c0 > self.c0))))):
            self.c0_parts.set_as(2)
        with self.tik.else_scope():
            self.c0_parts.set_as(1)

    def _move_to_target_layout(self, e_lp_idx):
        """
        move elements to target layout
        """
        with self.tik.if_scope(self.c0_parts == 1):
            clean_ubuf(self.tik, self.data_ub_b16, 0, self.ub_offset * self.type_bytes // 2)

        with self.tik.new_stmt_scope(disable_sync=True):
            cp2_len = e_lp_idx * self.c % self.c0 * self.type_bytes // 2
            cp1_len = self.c0 * self.type_bytes // 2 - cp2_len
            self.tik.data_move(
                self.data_ub_b16,
                self.data_ub_b16[self.ub_offset * self.type_bytes // 2 + cp2_len * CST.VNC_ROWS], 0,
                self.cur_n_ng_lp_unit, cp1_len, cp2_len, cp2_len)
            with self.tik.if_scope(self.c0_parts > 1):
                target_offset = cp1_len * CST.VNC_ROWS
                source_offset = (self.ub_offset * self.type_bytes // 2 +
                                 self.cur_n_ng_lp_unit * self.c0 * self.type_bytes // 2 * CST.VNC_ROWS)
                self.tik.data_move(self.data_ub_b16[target_offset], self.data_ub_b16[source_offset], 0,
                                        self.cur_n_ng_lp_unit, cp2_len, cp1_len, cp1_len)

    def _move_to_target_layout_4_one_row(self, e_lp_idx, n_cube_cnt):
        """
        move elements to target layout when all valid data is in one row
        """
        with self.tik.if_scope(self.c0_parts == 1):
            clean_ubuf(self.tik, self.data_ub_b16, 0, self.ub_offset * self.type_bytes // 2)

        with self.tik.new_stmt_scope(disable_sync=True):
            cp2_len = e_lp_idx * self.c % self.c0 * self.type_bytes // 2
            cp1_len = self.c0 * self.type_bytes // 2 - cp2_len
            with self.tik.for_range(0, self.cur_n_ng_lp_unit) as n_ng_idx:
                src_stride = (self.cur_n_ng_lp_unit * self.c0 * self.type_bytes // 2 * self.c0_parts -
                              cp1_len)
                dst_stride = (self.cur_n_ng_lp_unit * self.c0 * self.type_bytes // 2 - cp1_len)
                src_offset = (self.ub_offset * self.type_bytes // 2 +
                              (cp2_len + n_ng_idx * self.c0 * self.type_bytes // 2) * CST.VNC_ROWS)
                dst_offset = (n_ng_idx * self.c0 * self.type_bytes // 2 * CST.VNC_ROWS)
                self.tik.data_move(self.data_ub_b16[dst_offset], self.data_ub_b16[src_offset], 0, n_cube_cnt,
                                        cp1_len, src_stride, dst_stride)
            with self.tik.if_scope(self.c0_parts > 1):
                with self.tik.for_range(0, self.cur_n_ng_lp_unit) as n_ng_idx_1:
                    src_stride = (self.cur_n_ng_lp_unit * self.c0 * self.type_bytes // 2 * self.c0_parts -
                                  cp2_len)
                    dst_stride = (self.cur_n_ng_lp_unit * self.c0 * self.type_bytes // 2 - cp2_len)
                    src_offset = (self.ub_offset * self.type_bytes // 2 +
                                  (self.cur_n_ng_lp_unit * self.c0 * self.type_bytes // 2 +
                                   n_ng_idx_1 * self.c0 * self.type_bytes // 2) * CST.VNC_ROWS)
                    dst_offset = (n_ng_idx_1 * self.c0 * self.type_bytes // 2 + cp1_len) * CST.VNC_ROWS
                    self.tik.data_move(self.data_ub_b16[dst_offset], self.data_ub_b16[src_offset], 0, n_cube_cnt,
                                       cp2_len, src_stride, dst_stride)

    def _move_to_target_layout_b8_4_one_row(self, e_lp_idx, n_cube_cnt):
        """
        move elements to target layout for b8 when all valid data is in one row
        """
        with self.tik.if_scope(self.c0_parts == 1):
            clean_ubuf(self.tik, self.data_ub_b16, 0, self.ub_offset * self.type_bytes // 2)

        with self.tik.new_stmt_scope(disable_sync=True):
            cp2_len = e_lp_idx * self.c % self.c0
            cp1_len = self.c0 - cp2_len
            col_factor = 2
            with self.tik.for_range(0, self.cur_n_ng_lp_unit) as n_ng_idx:
                src_stride = (self.cur_n_ng_lp_unit * self.c0 * self.c0_parts - cp1_len)
                dst_stride = (self.cur_n_ng_lp_unit * self.c0 - cp1_len)
                src_offset = (self.ub_offset + (cp2_len + n_ng_idx * self.c0) * CST.VNC_ROWS * col_factor)
                dst_offset = (n_ng_idx * self.c0 * CST.VNC_ROWS * col_factor)
                self.tik.data_move(self.data_ub[dst_offset], self.data_ub[src_offset], 0, n_cube_cnt, cp1_len,
                                        src_stride, dst_stride)
            with self.tik.if_scope(self.c0_parts > 1):
                with self.tik.for_range(0, self.cur_n_ng_lp_unit) as n_ng_idx_1:
                    src_stride = (self.cur_n_ng_lp_unit * self.c0 * self.c0_parts - cp2_len)
                    dst_stride = (self.cur_n_ng_lp_unit * self.c0 - cp2_len)
                    src_offset = (self.ub_offset +
                                  (self.cur_n_ng_lp_unit * self.c0 + n_ng_idx_1 * self.c0) * CST.VNC_ROWS * col_factor)
                    dst_offset = (n_ng_idx_1 * self.c0 + cp1_len) * CST.VNC_ROWS * col_factor
                    self.tik.data_move(self.data_ub[dst_offset], self.data_ub[src_offset], 0, n_cube_cnt, cp2_len,
                                       src_stride, dst_stride)

    def _move_to_target_layout_b8(self, e_lp_idx):
        """
        move elements to target layout for b8
        """
        with self.tik.if_scope(self.c0_parts == 1):
            clean_ubuf(self.tik, self.data_ub_b16, 0, self.ub_offset * self.type_bytes // 2)

        with self.tik.new_stmt_scope(disable_sync=True):
            cp2_len = e_lp_idx * self.c % self.c0
            cp1_len = self.c0 - cp2_len
            col_factor = 2
            self.tik.data_move(self.data_ub, self.data_ub[self.ub_offset + cp2_len * CST.VNC_ROWS * col_factor],
                                    0, self.cur_n_ng_lp_unit, cp1_len, cp2_len, cp2_len)
            with self.tik.if_scope(self.c0_parts > 1):
                target_offset = cp1_len * CST.VNC_ROWS * col_factor
                source_offset = (self.ub_offset + self.cur_n_ng_lp_unit * self.c0 * CST.VNC_ROWS * col_factor)
                self.tik.data_move(self.data_ub[target_offset], self.data_ub[source_offset], 0,
                                        self.cur_n_ng_lp_unit, cp2_len, cp1_len, cp1_len)

    def _transpose_by_vnchwconv_b16(self, n_cube_cnt):
        """
        transpose two axises by vnchwconv for b16 dtype
        """
        src_addrs = [
            self.data_ub_b16[self.max_n_per_row * self.c0 * self.type_bytes // 2 * i]
            for i in range(16)
        ]
        dst_addrs = [
            self.data_ub_b16[CST.VNC_ROWS * i + self.ub_offset * self.type_bytes // 2]
            for i in range(16)
        ]
        repeat_cnt = self.ceil_div(
            n_cube_cnt * self.cur_n_ng_lp_unit * self.c0 * self.type_bytes // 2 * self.c0_parts,
            CST.VNC_ROWS)
        self._set_vnchwconv_stride(repeat_cnt, 1, CST.VNC_ROWS)
        self.tik.vnchwconv(False, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride,
                                self.vnc_src_stride)

    def _transpose_back_by_vnchwconv_b16(self, n_cube_cnt):
        """
        transpose two axises back by vnchwconv for b16 dtype
        """
        src_addrs = [self.data_ub_b16[CST.VNC_ROWS * i] for i in range(16)]
        dst_addrs = [
            self.data_ub_b16[(self.max_n_per_row * self.c0 * i + self.ub_offset) * self.type_bytes // 2]
            for i in range(16)
        ]
        repeat_cnt = self.ceil_div(n_cube_cnt * self.cur_n_ng_lp_unit * self.c0 * self.type_bytes // 2, CST.VNC_ROWS)
        self._set_vnchwconv_stride(repeat_cnt, CST.VNC_ROWS, 1)
        self.tik.vnchwconv(False, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _transpose_by_vnchwconv_b8(self, n_cube_cnt):
        """
        transpose two axises by vnchwconv for b8 dtype
        """
        src_addrs = [self.data_ub[self.max_n_per_row * self.c0 * i] for i in range(16)]
        dst_addrs = [self.data_ub[self.align_var * i + self.ub_offset] for i in range(16)]
        repeat_cnt = self.ceil_div(n_cube_cnt * self.cur_n_ng_lp_unit * self.c0 * self.c0_parts, self.align_var)
        self._set_vnchwconv_stride(repeat_cnt, 1, self.align_var)
        self.tik.vnchwconv(False, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)
        dst_addrs = [self.data_ub[self.align_var * (i + CST.VNC_ROWS) + self.ub_offset] for i in range(16)]
        self.tik.vnchwconv(False, True, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _transpose_back_by_vnchwconv_b8(self, n_cube_cnt):
        """
        transpose two axises back by vnchwconv for b8 dtype
        """
        src_addrs = [self.data_ub[self.align_var * i] for i in range(16)]
        dst_addrs = [self.data_ub[self.max_n_per_row * self.c0 * i + self.ub_offset] for i in range(16)]
        repeat_cnt = self.ceil_div(n_cube_cnt * self.cur_n_ng_lp_unit * self.c0, self.align_var)
        self._set_vnchwconv_stride(repeat_cnt, self.align_var, 1)
        self.tik.vnchwconv(False, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride,
                                self.vnc_src_stride)
        src_addrs = [self.data_ub[self.align_var * (i + CST.VNC_ROWS)] for i in range(16)]
        self.tik.vnchwconv(True, False, dst_addrs, src_addrs, repeat_cnt, self.vnc_dst_stride, self.vnc_src_stride)

    def _transform_by_vnchwconv(self, e_lp_idx, n_cube_cnt):
        """
        reorder elements by vnchwconv
        """
        if self.align_var != CST.BYTES_PER_BLOCK:
            self._transpose_by_vnchwconv_b16(n_cube_cnt)
            with self.tik.if_scope(
                tik.any(
                        self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_HW,
                        self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_D)):
                self._move_to_target_layout(e_lp_idx)
            with self.tik.else_scope():
                self._move_to_target_layout_4_one_row(e_lp_idx, n_cube_cnt)
            self._transpose_back_by_vnchwconv_b16(n_cube_cnt)
        else:
            with self.tik.if_scope(e_lp_idx * self.c % self.c0 % 2 != 0):
                self._transpose_by_vnchwconv_b8(n_cube_cnt)
                with self.tik.if_scope(
                        tik.any(self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_HW,
                                self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_D)):
                    self._move_to_target_layout_b8(e_lp_idx)
                with self.tik.else_scope():
                    self._move_to_target_layout_b8_4_one_row(e_lp_idx, n_cube_cnt)
                self._transpose_back_by_vnchwconv_b8(n_cube_cnt)
            with self.tik.else_scope():
                self._transpose_by_vnchwconv_b16(n_cube_cnt)
                with self.tik.if_scope(
                        tik.any(self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_HW,
                                self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_D)):
                    self._move_to_target_layout(e_lp_idx)
                with self.tik.else_scope():
                    self._move_to_target_layout_4_one_row(e_lp_idx, n_cube_cnt)
                self._transpose_back_by_vnchwconv_b16(n_cube_cnt)

    def _move_data_in(self, in_offset, repeat, src_stride_offset, dst_stride_offset):
        """
        The layout is: H0, NC0_1, (NC0_2,) ...
                       H1, NC0_1, (NC0_2,) ...
                       . , .    , (.    ,) ...
                       H15,NC0_1, (NC0_2,) ...
        """
        with self.tik.new_stmt_scope(disable_sync=True):
            with self.tik.for_range(0, self.c0_parts) as c0_idx:
                c0_part_gm_offset = c0_idx * self.hw * self.n_e_align * self.c0
                c0_part_ub_offset = c0_idx * self.cur_n_ng_lp_unit * self.c0
                src_stride = (src_stride_offset - self.cur_n_ng_lp_unit * self.c0) // self.align_var
                dst_stride = (dst_stride_offset - self.cur_n_ng_lp_unit * self.c0) // self.align_var
                with self.tik.if_scope(src_stride <= CST.STRIDE_LIMIT):
                    self.tik.data_move(self.data_ub[c0_part_ub_offset], self.input_gm[in_offset + c0_part_gm_offset],
                                       0, repeat, self.cur_n_ng_lp_unit * self.c0 // self.align_var,
                                       src_stride, dst_stride)
                with self.tik.else_scope():
                    with self.tik.for_range(0, repeat) as hw_idx:
                        self.tik.data_move(self.data_ub[c0_part_ub_offset + hw_idx * dst_stride_offset],
                                           self.input_gm[in_offset + c0_part_gm_offset + hw_idx * src_stride_offset],
                                           0, 1, self.cur_n_ng_lp_unit * self.c0 // self.align_var, 0, 0)

    def _move_data_out(self, out_offset, repeat, src_stride_offset, dst_stride_offset):
        with self.tik.new_stmt_scope(disable_sync=True):
            src_stride = (src_stride_offset - self.cur_n_ng_lp_unit * self.c0) // self.align_var
            dst_stride = (dst_stride_offset - self.cur_n_ng_lp_unit * self.c0) // self.align_var
            with self.tik.if_scope(dst_stride <= CST.STRIDE_LIMIT):
                self.tik.data_move(self.output_gm[out_offset], self.data_ub[self.out_ub_offset], 0, repeat,
                                        self.cur_n_ng_lp_unit * self.c0 // self.align_var, src_stride, dst_stride)
            with self.tik.else_scope():
                with self.tik.for_range(0, repeat) as hw_idx:
                    self.tik.data_move(self.output_gm[out_offset + hw_idx * dst_stride_offset],
                                       self.data_ub[self.out_ub_offset + hw_idx * src_stride_offset], 0, 1,
                                       self.cur_n_ng_lp_unit * self.c0 // self.align_var, 0, 0)

    def _do_mode_c_is_c0_align_hw(self, block_idx):
        """
        c is aligned with c0 and hw loop unit is larger than 1
        """
        with self.tik.for_range(0, self.g_lp_cnt) as g_lp_idx:
            with self.tik.for_range(0, self.e) as e_lp_idx:
                with self.tik.if_scope(e_lp_idx == 0):
                    self.e_in_offset.set_as(0)
                with self.tik.else_scope():
                    self.e_in_offset.set_as(self.e_in_offset + (
                        (e_lp_idx - 1) * self.c % self.c0 + self.c) // self.c0 * self.hw * self.n_e_align * self.c0)
                with self.tik.if_scope(
                    tik.any(
                        tik.all(self.mc_pos == CST.G_POS,
                                (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx < self.g),
                        tik.all(self.mc_pos != CST.G_POS, g_lp_idx * self.e + e_lp_idx < self.g))):
                    with self.tik.for_range(0, self.n_ng_lp_cnt) as n_ng_lp_idx:
                        self._get_current_n_ng_unit(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)

                        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
                        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
                        e_out_offset = e_lp_idx * self.n_ng * self.c0
                        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
                        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0
                        self.out_ub_offset.set_as(0)

                        with self.tik.for_range(0, self.d_lp_cnt) as d_lp_idx:
                            d_in_offset = d_lp_idx * self.c1_in * self.hw * self.n_e_align * self.c0
                            d_out_offset = d_lp_idx * self.c1_out * self.hw * self.n_gp_align * self.c0
                            with self.tik.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                                with self.tik.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                                    hw_in_offset = hw_lp_idx * self.hw_lp_unit * self.n_e_align * self.c0
                                    hw_out_offset = hw_lp_idx * self.hw_lp_unit * self.n_gp_align * self.c0
                                    self._get_current_hw_unit(block_idx, hw_lp_idx)

                                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset +
                                                n_in_offset + hw_in_offset + block_idx * self.core_step_in)
                                    in_src_stride = self.n_e_align * self.c0
                                    in_dst_stride = self.max_n_per_row * self.c0
                                    self._move_data_in(in_offset, self.cur_hw_lp_unit, in_src_stride, in_dst_stride)
                                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset +
                                                n_out_offset + hw_out_offset + block_idx * self.core_step_out)
                                    out_src_stride = in_dst_stride
                                    out_dst_stride = self.n_gp_align * self.c0
                                    self._move_data_out(out_offset, self.cur_hw_lp_unit, out_src_stride, out_dst_stride)

    def _do_mode_c_is_c0_align_c1out(self, block_idx):
        """
        c is aligned with c0 and c1_out loop unit is larger than 1
        """
        with self.tik.for_range(0, self.g_lp_cnt) as g_lp_idx:
            with self.tik.for_range(0, self.e) as e_lp_idx:
                with self.tik.if_scope(e_lp_idx == 0):
                    self.e_in_offset.set_as(0)
                with self.tik.else_scope():
                    self.e_in_offset.set_as(self.e_in_offset + (
                        (e_lp_idx - 1) * self.c % self.c0 + self.c) // self.c0 * self.hw * self.n_e_align * self.c0)
                with self.tik.if_scope(
                    tik.any(
                        tik.all(self.mc_pos == CST.G_POS,
                                (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx < self.g),
                        tik.all(self.mc_pos != CST.G_POS, g_lp_idx * self.e + e_lp_idx < self.g))):
                    with self.tik.for_range(0, self.n_ng_lp_cnt) as n_ng_lp_idx:
                        self._get_current_n_ng_unit(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
                        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
                        e_out_offset = e_lp_idx * self.n_ng * self.c0
                        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
                        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0
                        self.out_ub_offset.set_as(0)

                        with self.tik.for_range(0, self.d_lp_cnt) as d_lp_idx:
                            d_in_offset = d_lp_idx * self.c1_in * self.hw * self.n_e_align * self.c0
                            d_out_offset = d_lp_idx * self.c1_out * self.hw * self.n_gp_align * self.c0
                            with self.tik.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                                c1_in_offset = c1_out_lp_idx * self.c1_out_lp_unit * self.hw * self.n_e_align * self.c0
                                c1_out_offset = \
                                    c1_out_lp_idx * self.c1_out_lp_unit * self.hw * self.n_gp_align * self.c0
                                self._get_current_c1_out_unit(block_idx, c1_out_lp_idx)
                                with self.tik.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                                    hw_in_offset = hw_lp_idx * self.n_e_align * self.c0
                                    hw_out_offset = hw_lp_idx * self.n_gp_align * self.c0

                                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset +
                                                 n_in_offset + hw_in_offset + block_idx * self.core_step_in)
                                    in_src_stride = self.hw * self.n_e_align * self.c0
                                    in_dst_stride = self.max_n_per_row * self.c0
                                    self._move_data_in(in_offset, self.cur_c1_out_lp_unit, in_src_stride, in_dst_stride)
                                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset +
                                                  n_out_offset + hw_out_offset + block_idx * self.core_step_out)
                                    out_src_stride = in_dst_stride
                                    out_dst_stride = self.hw * self.n_gp_align * self.c0
                                    self._move_data_out(out_offset, self.cur_c1_out_lp_unit, out_src_stride,
                                                        out_dst_stride)

    def _do_mode_c_is_c0_align_d(self, block_idx):
        """
        c is aligned with c0 and d loop unit is larger than 1
        """
        with self.tik.for_range(0, self.g_lp_cnt) as g_lp_idx:
            with self.tik.for_range(0, self.e) as e_lp_idx:
                with self.tik.if_scope(e_lp_idx == 0):
                    self.e_in_offset.set_as(0)
                with self.tik.else_scope():
                    self.e_in_offset.set_as(self.e_in_offset + (
                        (e_lp_idx - 1) * self.c % self.c0 + self.c) // self.c0 * self.hw * self.n_e_align * self.c0)
                with self.tik.if_scope(
                    tik.any(
                        tik.all(self.mc_pos == CST.G_POS,
                                (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx < self.g),
                        tik.all(self.mc_pos != CST.G_POS, g_lp_idx * self.e + e_lp_idx < self.g))):
                    with self.tik.for_range(0, self.n_ng_lp_cnt) as n_ng_lp_idx:
                        self._get_current_n_ng_unit(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
                        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
                        e_out_offset = e_lp_idx * self.n_ng * self.c0
                        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
                        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0
                        self.out_ub_offset.set_as(0)

                        with self.tik.for_range(0, self.d_lp_cnt) as d_lp_idx:
                            d_in_offset = d_lp_idx * self.d_lp_unit * self.c1_in * self.hw * self.n_e_align * self.c0
                            d_out_offset = d_lp_idx * self.d_lp_unit * self.c1_out * self.hw * self.n_gp_align * self.c0
                            self._get_current_d_unit(block_idx, d_lp_idx)
                            with self.tik.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                                with self.tik.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                                    hw_in_offset = hw_lp_idx * self.n_e_align * self.c0
                                    hw_out_offset = hw_lp_idx * self.n_gp_align * self.c0

                                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset +
                                                n_in_offset + hw_in_offset + block_idx * self.core_step_in)
                                    in_src_stride = self.c1_in * self.hw * self.n_e_align * self.c0
                                    in_dst_stride = self.max_n_per_row * self.c0
                                    self._move_data_in(in_offset, self.cur_d_lp_unit, in_src_stride, in_dst_stride)
                                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset +
                                                n_out_offset + hw_out_offset + block_idx * self.core_step_out)
                                    out_src_stride = in_dst_stride
                                    out_dst_stride = self.c1_out * self.hw * self.n_gp_align * self.c0
                                    self._move_data_out(out_offset, self.cur_d_lp_unit, out_src_stride, out_dst_stride)

    def _do_mode_c_is_c0_unalign_hw(self, block_idx):
        """
        c is not aligned with c0 and hw loop unit is larger than 1
        """
        with self.tik.for_range(0, self.g_lp_cnt) as g_lp_idx:
            with self.tik.for_range(0, self.e) as e_lp_idx:
                with self.tik.if_scope(e_lp_idx == 0):
                    self.e_in_offset.set_as(0)
                with self.tik.else_scope():
                    self.e_in_offset.set_as(self.e_in_offset + (
                        (e_lp_idx - 1) * self.c % self.c0 + self.c) // self.c0 * self.hw * self.n_e_align * self.c0)
                with self.tik.if_scope(
                    tik.any(
                        tik.all(self.mc_pos == CST.G_POS,
                                (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx < self.g),
                        tik.all(self.mc_pos != CST.G_POS, g_lp_idx * self.e + e_lp_idx < self.g))):
                    with self.tik.for_range(0, self.n_ng_lp_cnt) as n_ng_lp_idx:
                        self._get_current_n_ng_unit(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
                        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
                        e_out_offset = e_lp_idx * self.n_ng * self.c0
                        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
                        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0

                        with self.tik.for_range(0, self.d_lp_cnt) as d_lp_idx:
                            d_in_offset = d_lp_idx * self.c1_in * self.hw * self.n_e_align * self.c0
                            d_out_offset = d_lp_idx * self.c1_out * self.hw * self.n_gp_align * self.c0
                            with self.tik.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                                self._get_c0_parts(block_idx, e_lp_idx, c1_out_lp_idx)
                                with self.tik.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                                    hw_in_offset = hw_lp_idx * self.hw_lp_unit * self.n_e_align * self.c0
                                    hw_out_offset = hw_lp_idx * self.hw_lp_unit * self.n_gp_align * self.c0
                                    self._get_current_hw_unit(block_idx, hw_lp_idx)

                                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset +
                                                n_in_offset + hw_in_offset + block_idx * self.core_step_in)
                                    in_src_stride = self.n_e_align * self.c0
                                    in_dst_stride = self.max_n_per_row * self.c0
                                    self._move_data_in(in_offset, self.cur_hw_lp_unit, in_src_stride, in_dst_stride)

                                    with self.tik.if_scope(e_lp_idx * self.c % self.c0 > 0):
                                        self._transform_by_vnchwconv(e_lp_idx, 1)
                                        self.out_ub_offset.set_as(self.ub_offset)
                                    with self.tik.else_scope():
                                        self.out_ub_offset.set_as(0)

                                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset +
                                                n_out_offset + hw_out_offset + block_idx * self.core_step_out)
                                    out_src_stride = in_dst_stride
                                    out_dst_stride = self.n_gp_align * self.c0
                                    self._move_data_out(out_offset, self.cur_hw_lp_unit, out_src_stride, out_dst_stride)

    def _do_mode_c_is_c0_unalign_d(self, block_idx):
        """
        c is not aligned with c0 and d loop unit is larger than 1
        """
        with self.tik.for_range(0, self.g_lp_cnt) as g_lp_idx:
            with self.tik.for_range(0, self.e) as e_lp_idx:
                with self.tik.if_scope(e_lp_idx == 0):
                    self.e_in_offset.set_as(0)
                with self.tik.else_scope():
                    self.e_in_offset.set_as(self.e_in_offset + (
                        (e_lp_idx - 1) * self.c % self.c0 + self.c) // self.c0 * self.hw * self.n_e_align * self.c0)
                with self.tik.if_scope(
                    tik.any(
                        tik.all(self.mc_pos == CST.G_POS,
                                (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx < self.g),
                        tik.all(self.mc_pos != CST.G_POS, g_lp_idx * self.e + e_lp_idx < self.g))):
                    with self.tik.for_range(0, self.n_ng_lp_cnt) as n_ng_lp_idx:
                        self._get_current_n_ng_unit(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
                        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
                        e_out_offset = e_lp_idx * self.n_ng * self.c0
                        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
                        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0

                        with self.tik.for_range(0, self.d_lp_cnt) as d_lp_idx:
                            d_in_offset = d_lp_idx * self.d_lp_unit * self.c1_in * self.hw * self.n_e_align * self.c0
                            d_out_offset = d_lp_idx * self.d_lp_unit * self.c1_out * self.hw * self.n_gp_align * self.c0
                            self._get_current_d_unit(block_idx, d_lp_idx)
                            with self.tik.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                                self._get_c0_parts(block_idx, e_lp_idx, c1_out_lp_idx)
                                with self.tik.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                                    hw_in_offset = hw_lp_idx * self.n_e_align * self.c0
                                    hw_out_offset = hw_lp_idx * self.n_gp_align * self.c0

                                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset +
                                                n_in_offset + hw_in_offset + block_idx * self.core_step_in)
                                    in_src_stride = self.c1_in * self.hw * self.n_e_align * self.c0
                                    in_dst_stride = self.max_n_per_row * self.c0
                                    self._move_data_in(in_offset, self.cur_d_lp_unit, in_src_stride, in_dst_stride)

                                    with self.tik.if_scope(e_lp_idx * self.c % self.c0 > 0):
                                        self._transform_by_vnchwconv(e_lp_idx, 1)
                                        self.out_ub_offset.set_as(self.ub_offset)
                                    with self.tik.else_scope():
                                        self.out_ub_offset.set_as(0)

                                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset +
                                                n_out_offset + hw_out_offset + block_idx * self.core_step_out)
                                    out_src_stride = in_dst_stride
                                    out_dst_stride = self.c1_out * self.hw * self.n_gp_align * self.c0
                                    self._move_data_out(out_offset, self.cur_d_lp_unit, out_src_stride, out_dst_stride)

    def _do_mode_c_is_c0_unalign_hw_1(self, block_idx):
        """
        c is not aligned with c0, hw loop unit is larger than 1 and n loop unit is small
        """
        with self.tik.for_range(0, self.g_lp_cnt) as g_lp_idx:
            with self.tik.for_range(0, self.e) as e_lp_idx:
                with self.tik.if_scope(e_lp_idx == 0):
                    self.e_in_offset.set_as(0)
                with self.tik.else_scope():
                    self.e_in_offset.set_as(self.e_in_offset + (
                        (e_lp_idx - 1) * self.c % self.c0 + self.c) // self.c0 * self.hw * self.n_e_align * self.c0)
                with self.tik.if_scope(
                    tik.any(
                        tik.all(self.mc_pos == CST.G_POS,
                                (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx < self.g),
                        tik.all(self.mc_pos != CST.G_POS, g_lp_idx * self.e + e_lp_idx < self.g))):
                    with self.tik.for_range(0, self.n_ng_lp_cnt) as n_ng_lp_idx:
                        self._get_current_n_ng_unit(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
                        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
                        e_out_offset = e_lp_idx * self.n_ng * self.c0
                        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
                        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0

                        with self.tik.for_range(0, self.d_lp_cnt) as d_lp_idx:
                            d_in_offset = d_lp_idx * self.c1_in * self.hw * self.n_e_align * self.c0
                            d_out_offset = d_lp_idx * self.c1_out * self.hw * self.n_gp_align * self.c0
                            with self.tik.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                                self._get_c0_parts(block_idx, e_lp_idx, c1_out_lp_idx)
                                with self.tik.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                                    hw_in_offset = hw_lp_idx * self.hw_lp_unit * self.n_e_align * self.c0
                                    hw_out_offset = hw_lp_idx * self.hw_lp_unit * self.n_gp_align * self.c0
                                    self._get_current_hw_unit(block_idx, hw_lp_idx)

                                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset +
                                                n_in_offset + hw_in_offset + block_idx * self.core_step_in)
                                    in_src_stride = self.n_e_align * self.c0
                                    in_dst_stride = self.c0_parts * self.cur_n_ng_lp_unit * self.c0
                                    self._move_data_in(in_offset, self.cur_hw_lp_unit, in_src_stride, in_dst_stride)

                                    with self.tik.if_scope(e_lp_idx * self.c % self.c0 > 0):
                                        self._transform_by_vnchwconv(e_lp_idx, self.cur_hw_lp_unit)
                                        self.out_ub_offset.set_as(self.ub_offset)
                                    with self.tik.else_scope():
                                        self.out_ub_offset.set_as(0)

                                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset +
                                                n_out_offset + hw_out_offset + block_idx * self.core_step_out)
                                    out_src_stride = self.cur_n_ng_lp_unit * self.c0
                                    out_dst_stride = self.n_gp_align * self.c0
                                    self._move_data_out(out_offset, self.cur_hw_lp_unit, out_src_stride, out_dst_stride)

    def _do_mode_c_is_c0_unalign_d_1(self, block_idx):
        """
        c is not aligned with c0, d loop unit is larger than 1 and n loop unit is small
        """
        with self.tik.for_range(0, self.g_lp_cnt) as g_lp_idx:
            with self.tik.for_range(0, self.e) as e_lp_idx:
                with self.tik.if_scope(e_lp_idx == 0):
                    self.e_in_offset.set_as(0)
                with self.tik.else_scope():
                    self.e_in_offset.set_as(self.e_in_offset + (
                        (e_lp_idx - 1) * self.c % self.c0 + self.c) // self.c0 * self.hw * self.n_e_align * self.c0)
                with self.tik.if_scope(
                    tik.any(
                        tik.all(self.mc_pos == CST.G_POS,
                                (block_idx * self.nlc_g_lp_cnt + g_lp_idx) * self.e + e_lp_idx < self.g),
                        tik.all(self.mc_pos != CST.G_POS, g_lp_idx * self.e + e_lp_idx < self.g))):
                    with self.tik.for_range(0, self.n_ng_lp_cnt) as n_ng_lp_idx:
                        self._get_current_n_ng_unit(block_idx, g_lp_idx, e_lp_idx, n_ng_lp_idx)
                        g_in_offset = g_lp_idx * self.d * self.c1_in * self.hw * self.n_e_align * self.c0
                        g_out_offset = g_lp_idx * self.e * self.n_ng * self.c0
                        e_out_offset = e_lp_idx * self.n_ng * self.c0
                        n_in_offset = (n_ng_lp_idx * self.n_ng_lp_unit + e_lp_idx * self.n_ng) * self.c0
                        n_out_offset = n_ng_lp_idx * self.n_ng_lp_unit * self.c0

                        with self.tik.for_range(0, self.d_lp_cnt) as d_lp_idx:
                            d_in_offset = d_lp_idx * self.d_lp_unit * self.c1_in * self.hw * self.n_e_align * self.c0
                            d_out_offset = d_lp_idx * self.d_lp_unit * self.c1_out * self.hw * self.n_gp_align * self.c0
                            self._get_current_d_unit(block_idx, d_lp_idx)
                            with self.tik.for_range(0, self.c1_out_lp_cnt) as c1_out_lp_idx:
                                c1_in_offset = c1_out_lp_idx * self.hw * self.n_e_align * self.c0
                                c1_out_offset = c1_out_lp_idx * self.hw * self.n_gp_align * self.c0
                                self._get_c0_parts(block_idx, e_lp_idx, c1_out_lp_idx)
                                with self.tik.for_range(0, self.hw_lp_cnt) as hw_lp_idx:
                                    hw_in_offset = hw_lp_idx * self.n_e_align * self.c0
                                    hw_out_offset = hw_lp_idx * self.n_gp_align * self.c0

                                    in_offset = (g_in_offset + d_in_offset + c1_in_offset + self.e_in_offset +
                                                n_in_offset + hw_in_offset + block_idx * self.core_step_in)
                                    in_src_stride = self.c1_in * self.hw * self.n_e_align * self.c0
                                    in_dst_stride = self.c0_parts * self.cur_n_ng_lp_unit * self.c0
                                    self._move_data_in(in_offset, self.cur_d_lp_unit, in_src_stride, in_dst_stride)

                                    with self.tik.if_scope(e_lp_idx * self.c % self.c0 > 0):
                                        self._transform_by_vnchwconv(e_lp_idx, self.cur_d_lp_unit)
                                        self.out_ub_offset.set_as(self.ub_offset)
                                    with self.tik.else_scope():
                                        self.out_ub_offset.set_as(0)

                                    out_offset = (g_out_offset + d_out_offset + c1_out_offset + e_out_offset +
                                                n_out_offset + hw_out_offset + block_idx * self.core_step_out)
                                    out_src_stride = self.cur_n_ng_lp_unit * self.c0
                                    out_dst_stride = self.c1_out * self.hw * self.n_gp_align * self.c0
                                    self._move_data_out(out_offset, self.cur_d_lp_unit, out_src_stride, out_dst_stride)

    def _do_mode_n_n(self, block_idx):
        i0, i1, i2, i3, i4 = self._addressing_core(block_idx, CST.N4)
        self._calc_jump_params(i0)
        # dstTensor-addressing on core-level
        di0 = Coordinate(i0 // self.lcm_e, self.prod(self.fzg_shape[1:]), 0, name="gi")
        di1 = Coordinate(i1, self.prod(self.fzg_shape[CST.N2:]), 0, name="di")
        di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N3:]),
                         self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
        di3 = Coordinate(i3, self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")
        di4 = Coordinate(i4, self.blk_factor * self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                         name="block_outer")

        # srcTensor-addressing on core-level
        si1 = Coordinate(i1, self.prod(self.fz_shape[1:]), 0, name="di")
        si2 = Coordinate(i2, self.prod(self.fz_shape[CST.N2:]), 0, name="c1i")
        si3 = Coordinate(i3, self.prod(self.fz_shape[CST.N3:]), 0, name="hi")
        si4 = Coordinate(i4, self.blk_factor * self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                         name="block_outer")

        def do_step_1_1(bound, ub_inner):
            with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i5:
                si5 = Coordinate(i5, self.ub_factor * self.dim_c0, 0, name="s_blk_inner")
                di5 = Coordinate(i5, self.ub_factor * self.dim_c0, 0, name="d_blk_inner")
                src_iter = [si1, si2, si3, si4, si5]
                dst_iter = [di0, di1, di2, di3, di4, di5]
                self._ub_split_n_step_1_1(src_iter, dst_iter, ub_inner)

        def do_step_2_1(bound, ub_inner):
            with self.tik.if_scope(i2 == 0):
                with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i5:
                    si5 = Coordinate(i5, self.ub_factor * self.dim_c0, 0, name="s_blk_inner")
                    di5 = Coordinate(i5, self.ub_factor * self.dim_c0, 0, name="d_blk_inner")
                    new_si2 = Coordinate(0, 0, 0, name="c1i")
                    new_di2 = Coordinate(0, 0, di2.offset, name="c1i")
                    src_iter = [si1, new_si2, si3, si4, si5]
                    dst_iter = [di0, di1, new_di2, di3, di4, di5]
                    self._ub_split_n_step_2_1(src_iter, dst_iter, ub_inner)

        def do_step_2_2(bound, ub_inner):
            with self.tik.if_scope(tik.all(i2 >= 1, i2 < self.dim_c1)):
                with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i5:
                    si5 = Coordinate(i5, self.ub_factor * self.dim_c0, 0, name="s_blk_inner")
                    new_si2_0 = Coordinate(i2, si2.stride, -1 * si2.stride, name="c1i_0")
                    new_si2_1 = si2
                    src_iter0 = [si1, new_si2_0, si3, si4, si5]
                    src_iter1 = [si1, new_si2_1, si3, si4, si5]
                    di5 = Coordinate(i5, self.ub_factor * self.dim_c0, 0, name="d_blk_inner")
                    dst_iter = [di0, di1, di2, di3, di4, di5]
                    self._ub_split_n_step_2_2(src_iter0, src_iter1, dst_iter, ub_inner)

        def do_step_2_3(bound, ub_inner):
            with self.tik.if_scope(self.c1_gap + self.dim_c1 < self.dim_c1x):
                with self.tik.if_scope(i2 == self.dim_c1 - 1):
                    with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i5:
                        si5 = Coordinate(i5, self.ub_factor * self.dim_c0, 0, name="s_blk_inner")
                        di5 = Coordinate(i5, self.ub_factor * self.dim_c0, 0, name="d_blk_inner")
                        new_si2 = Coordinate(self.dim_c1 - 1, si2.stride, si2.offset, name="c1i")
                        new_di2 = Coordinate(self.dim_c1, di2.stride, di2.offset, name="c1i")
                        src_iter = [si1, new_si2, si3, si4, si5]
                        dst_iter = [di0, di1, new_di2, di3, di4, di5]
                        self._ub_split_n_step_2_3(src_iter, dst_iter, ub_inner)

        process = [do_step_1_1, do_step_2_1, do_step_2_2, do_step_2_3]
        self._schedule_process(blk_outer=i4, dispatch=self._dispatching0, process=process)

    def _do_mode_h_n(self, block_idx):
        i0, i1, i2, i3 = self._addressing_core(block_idx, CST.N3)
        self._calc_jump_params(i0)
        # dstTensor-addressing on core-level
        di0 = Coordinate(i0 // self.lcm_e, self.prod(self.fzg_shape[1:]), 0, name="gi")
        di1 = Coordinate(i1, self.prod(self.fzg_shape[CST.N2:]), 0, name="di")
        di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N3:]),
                         self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
        di3 = Coordinate(i3, self.blk_factor * self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")

        # srcTensor-addressing on core-level
        si1 = Coordinate(i1, self.prod(self.fz_shape[1:]), 0, name="di")
        si2 = Coordinate(i2, self.prod(self.fz_shape[CST.N2:]), 0, name="c1i")
        si3 = Coordinate(i3, self.blk_factor * self.prod(self.fz_shape[CST.N3:]), 0, name="hi")

        def do_step_1_1(bound, ub_bound, ub_inner):
            with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i4:
                with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                    si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="s_blk_inner")
                    di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="d_blk_inner")
                    si5 = Coordinate(i5, self.ub_factor * self.dim_c0,
                                     self.src_lower_n * self.dim_c0, name="s_ub_outer")
                    di5 = Coordinate(i5, self.ub_factor * self.dim_c0,
                                     self.dst_lower_n * self.dim_c0, name="d_ub_outer")
                    src_iter = [si1, si2, si3, si4, si5]
                    dst_iter = [di0, di1, di2, di3, di4, di5]
                    self._ub_split_n_step_1_1(src_iter, dst_iter, ub_inner)

        def do_step_2_1(bound, ub_bound, ub_inner):
            with self.tik.if_scope(i2 == 0):
                with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i4:
                    with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                        si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="s_blk_inner")
                        di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="d_blk_inner")
                        si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                         name="s_ub_outer")
                        di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                         name="d_ub_outer")
                        new_si2 = Coordinate(0, 0, 0, name="c1i")
                        new_di2 = Coordinate(0, 0, di2.offset, name="c1i")
                        src_iter = [si1, new_si2, si3, si4, si5]
                        dst_iter = [di0, di1, new_di2, di3, di4, di5]
                        self._ub_split_n_step_2_1(src_iter, dst_iter, ub_inner)

        def do_step_2_2(bound, ub_bound, ub_inner):
            with self.tik.if_scope(tik.all(i2 >= 1, i2 < self.dim_c1)):
                with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i4:
                    with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                        si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="s_blk_inner")
                        si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                         name="s_ub_outer")
                        new_si2_0 = Coordinate(i2, si2.stride, -1 * si2.stride, name="c1i_0")
                        new_si2_1 = si2
                        src_iter0 = [si1, new_si2_0, si3, si4, si5]
                        src_iter1 = [si1, new_si2_1, si3, si4, si5]
                        di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="d_blk_inner")
                        di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                         name="d_ub_outer")
                        dst_iter = [di0, di1, di2, di3, di4, di5]
                        self._ub_split_n_step_2_2(src_iter0, src_iter1, dst_iter, ub_inner)

        def do_step_2_3(bound, ub_bound, ub_inner):
            with self.tik.if_scope(self.c1_gap + self.dim_c1 < self.dim_c1x):
                with self.tik.if_scope(i2 == self.dim_c1 - 1):
                    with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i4:
                        with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                            si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="s_blk_inner")
                            di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="d_blk_inner")
                            si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                             name="s_ub_outer")
                            di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                             name="d_ub_outer")
                            new_si2 = Coordinate(self.dim_c1 - 1, si2.stride, si2.offset, name="c1i")
                            new_di2 = Coordinate(self.dim_c1, di2.stride, di2.offset, name="c1i")
                            src_iter = [si1, new_si2, si3, si4, si5]
                            dst_iter = [di0, di1, new_di2, di3, di4, di5]
                            self._ub_split_n_step_2_3(src_iter, dst_iter, ub_inner)

        process = [do_step_1_1, do_step_2_1, do_step_2_2, do_step_2_3]
        self._schedule_process(blk_outer=i3, dispatch=self._dispatching_1, process=process)

    def _do_mode_c1_n(self, block_idx):
        i0, i1, i2 = self._addressing_core(block_idx, CST.N2)
        self._calc_jump_params(i0)
        # dstTensor-addressing on core-level
        di0 = Coordinate(i0 // self.lcm_e, self.prod(self.fzg_shape[1:]), 0, name="gi")
        di1 = Coordinate(i1, self.prod(self.fzg_shape[CST.N2:]), 0, name="di")
        # srcTensor-addressing on core-level
        si1 = Coordinate(i1, self.prod(self.fz_shape[1:]), 0, name="di")

        def do_step_1_1(bound, ub_bound, ub_inner):
            # need fused i2 * i3 as c1i
            with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i3:
                c1i = i2 * self.blk_factor + i3
                with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                    with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                        # i2+i3
                        sc1i = Coordinate(c1i, self.prod(self.fz_shape[CST.N2:]), 0, name="s_c1i")
                        dc1i = Coordinate(c1i, self.prod(self.fzg_shape[CST.N3:]),
                                          self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="d_c1i")
                        si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="s_hi")
                        di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="d_hi")
                        si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                         name="s_ub_outer")
                        di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                         name="d_ub_outer")
                        src_iter = [si1, sc1i, si4, si5]
                        dst_iter = [di0, di1, dc1i, di4, di5]
                        self._ub_split_n_step_1_1(src_iter, dst_iter, ub_inner)

        def do_step_2_1(bound, ub_bound, ub_inner):
            # need fused i2 * i3 as c1i
            with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i3:
                c1i = i2 * self.blk_factor + i3
                with self.tik.if_scope(c1i == 0):
                    with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                        with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                            # i2+i3: eliminate invalid scalar
                            sc1i = Coordinate(0, 0, 0, name="s_c1i")
                            dc1i = Coordinate(0, 0, self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="d_c1i")
                            si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="s_hi")
                            di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="d_hi")
                            si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                             name="s_ub_outer")
                            di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                             name="d_ub_outer")
                            src_iter = [si1, sc1i, si4, si5]
                            dst_iter = [di0, di1, dc1i, di4, di5]
                            self._ub_split_n_step_2_1(src_iter, dst_iter, ub_inner)

        def do_step_2_2(bound, ub_bound, ub_inner):
            # need fused i2 * i3 as c1i
            with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i3:
                c1i = i2 * self.blk_factor + i3
                with self.tik.if_scope(tik.all(c1i >= 1, c1i < self.dim_c1)):
                    with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                        with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                            sc1i = Coordinate(c1i, self.prod(self.fz_shape[CST.N2:]), 0, name="s_c1i")
                            si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="s_hi")
                            si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                             name="s_ub_outer")
                            new_sc1i_0 = Coordinate(c1i, sc1i.stride, -1 * sc1i.stride, name="c1i_0")
                            new_sc1i_1 = sc1i
                            src_iter0 = [si1, new_sc1i_0, si4, si5]
                            src_iter1 = [si1, new_sc1i_1, si4, si5]
                            dc1i = Coordinate(c1i, self.prod(self.fzg_shape[CST.N3:]),
                                              self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="d_c1i")
                            di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="d_hi")
                            di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                             name="d_ub_outer")
                            dst_iter = [di0, di1, dc1i, di4, di5]
                            self._ub_split_n_step_2_2(src_iter0, src_iter1, dst_iter, ub_inner)

        def do_step_2_3(bound, ub_bound, ub_inner):
            # need fused i2 * i3 as c1i
            with self.tik.if_scope(self.c1_gap + self.dim_c1 < self.dim_c1x):
                with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i3:
                    c1i = i2 * self.blk_factor + i3
                    with self.tik.if_scope(c1i == self.dim_c1 - 1):
                        with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                            with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                                sc1i = Coordinate(self.dim_c1 - 1, self.prod(self.fz_shape[CST.N2:]), 0, name="s_c1i")
                                dc1i = Coordinate(self.dim_c1, self.prod(self.fzg_shape[CST.N3:]),
                                                  self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="d_c1i")
                                si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="s_hi")
                                di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="d_hi")
                                si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                                 name="s_ub_outer")
                                di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                                 name="d_ub_outer")
                                src_iter = [si1, sc1i, si4, si5]
                                dst_iter = [di0, di1, dc1i, di4, di5]
                                self._ub_split_n_step_2_3(src_iter, dst_iter, ub_inner)

        process = [do_step_1_1, do_step_2_1, do_step_2_2, do_step_2_3]
        self._schedule_process(blk_outer=i2, dispatch=self._dispatching_1, process=process)

    def _do_mode_d_n(self, block_idx):
        i0, i1 = self._addressing_core(block_idx, 1)
        self._calc_jump_params(i0)
        # dstTensor-addressing on core-level
        di0 = Coordinate(i0 // self.lcm_e, self.prod(self.fzg_shape[1:]), 0, name="gi")
        di1 = Coordinate(i1, self.blk_factor * self.prod(self.fzg_shape[CST.N2:]), 0, name="di")
        # srcTensor-addressing on core-level
        si1 = Coordinate(i1, self.blk_factor * self.prod(self.fz_shape[1:]), 0, name="di")

        def do_step_1_1(bound, ub_bound, ub_inner):
            with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i2:
                with self.tik.for_range(0, self.dim_c1, name="c1i") as i3:
                    with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                        with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                            si2 = Coordinate(i2, self.prod(self.fz_shape[1:]), 0, name="s_blk_inner")
                            di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N2:]), 0, name="d_blk_inner")
                            si3 = Coordinate(i3, self.prod(self.fz_shape[CST.N2:]), 0, name="c1i")
                            di3 = Coordinate(i3, self.prod(self.fzg_shape[CST.N3:]),
                                             self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
                            si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="hi")
                            di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")
                            si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                             name="s_ub_outer")
                            di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                             name="d_ub_outer")
                            src_iter = [si1, si2, si3, si4, si5]
                            dst_iter = [di0, di1, di2, di3, di4, di5]
                            self._ub_split_n_step_1_1(src_iter, dst_iter, ub_inner)

        def do_step_2_1(bound, ub_bound, ub_inner):
            # i3(c1i) should be zero
            with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i2:
                with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                    with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                        si2 = Coordinate(i2, self.prod(self.fz_shape[1:]), 0, name="s_blk_inner")
                        di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N2:]), 0, name="d_blk_inner")
                        si3 = Coordinate(0, 0, 0, name="c1i")
                        di3 = Coordinate(0, 0, self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
                        si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="hi")
                        di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")
                        si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                         name="s_ub_outer")
                        di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                         name="d_ub_outer")

                        src_iter = [si1, si2, si3, si4, si5]
                        dst_iter = [di0, di1, di2, di3, di4, di5]
                        self._ub_split_n_step_2_1(src_iter, dst_iter, ub_inner)

        def do_step_2_2(bound, ub_bound, ub_inner):
            # i3(c1i) should be in [1, self.dim_c1)
            with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i2:
                with self.tik.for_range(1, self.dim_c1, name="c1i") as i3:
                    with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                        with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                            si2 = Coordinate(i2, self.prod(self.fz_shape[1:]), 0, name="s_blk_inner")
                            si3 = Coordinate(i3, self.prod(self.fz_shape[CST.N2:]), 0, name="c1i")
                            si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="hi")
                            si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                             name="s_ub_outer")
                            new_si3_0 = Coordinate(i3, si3.stride, -1 * si3.stride, name="c1i_0")
                            new_si3_1 = si3
                            src_iter0 = [si1, si2, new_si3_0, si4, si5]
                            src_iter1 = [si1, si2, new_si3_1, si4, si5]
                            di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N2:]), 0, name="d_blk_inner")
                            di3 = Coordinate(i3, self.prod(self.fzg_shape[CST.N3:]),
                                             self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
                            di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")
                            di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                             name="d_ub_outer")
                            dst_iter = [di0, di1, di2, di3, di4, di5]
                            self._ub_split_n_step_2_2(src_iter0, src_iter1, dst_iter, ub_inner)

        def do_step_2_3(bound, ub_bound, ub_inner):
            # i3(c1i) should be dimC1 - 1
            with self.tik.if_scope(self.c1_gap + self.dim_c1 < self.dim_c1x):
                with self.tik.for_range(bound[0], bound[1], name="blk_inner") as i2:
                    with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                        with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                            si2 = Coordinate(i2, self.prod(self.fz_shape[1:]), 0, name="s_blk_inner")
                            di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N2:]), 0, name="d_blk_inner")
                            si3 = Coordinate(self.dim_c1 - 1, self.prod(self.fz_shape[CST.N2:]), 0, name="c1i")
                            di3 = Coordinate(self.dim_c1, self.prod(self.fzg_shape[CST.N3:]),
                                             self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
                            si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="hi")
                            di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")
                            si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                             name="s_ub_outer")
                            di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                             name="d_ub_outer")
                            src_iter = [si1, si2, si3, si4, si5]
                            dst_iter = [di0, di1, di2, di3, di4, di5]
                            self._ub_split_n_step_2_3(src_iter, dst_iter, ub_inner)

        process = [do_step_1_1, do_step_2_1, do_step_2_2, do_step_2_3]
        self._schedule_process(blk_outer=i1, dispatch=self._dispatching_1, process=process)

    def _do_mode_groups_n(self, block_idx):
        i0 = self._addressing_core(block_idx, 0)[0]
        with self.tik.for_range(0, self.blk_factor, name="blk_inner") as i1:
            def do_step_1_1(ub_bound, ub_inner):
                with self.tik.for_range(0, self.dim_d, name="di") as i2:
                    with self.tik.for_range(0, self.dim_c1, name="c1i") as i3:
                        with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                            with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                                si2 = Coordinate(i2, self.prod(self.fz_shape[1:]), 0, name="di")
                                di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N2:]), 0, name="di")
                                si3 = Coordinate(i3, self.prod(self.fz_shape[CST.N2:]), 0, name="c1i")
                                di3 = Coordinate(i3, self.prod(self.fzg_shape[CST.N3:]),
                                                 self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
                                si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="hi")
                                di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")
                                si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                                 name="s_ub_outer")
                                di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                                 name="d_ub_outer")
                                src_iter = [si2, si3, si4, si5]
                                dst_iter = [dgi, di2, di3, di4, di5]
                                self._ub_split_n_step_1_1(src_iter, dst_iter, ub_inner)

            def do_step_2_1(ub_bound, ub_inner):
                # i3(c1i) should be zero
                with self.tik.for_range(0, self.dim_d, name="di") as i2:
                    with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                        with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                            si2 = Coordinate(i2, self.prod(self.fz_shape[1:]), 0, name="di")
                            di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N2:]), 0, name="di")
                            si3 = Coordinate(0, 0, 0, name="c1i")
                            di3 = Coordinate(0, 0, self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
                            si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="hi")
                            di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")
                            si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                             name="s_ub_outer")
                            di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                             name="d_ub_outer")
                            src_iter = [si2, si3, si4, si5]
                            dst_iter = [dgi, di2, di3, di4, di5]
                            self._ub_split_n_step_2_1(src_iter, dst_iter, ub_inner)

            def do_step_2_2(ub_bound, ub_inner):
                # i3(c1i) should be in [1, dimC1)
                with self.tik.for_range(0, self.dim_d, name="di") as i2:
                    with self.tik.for_range(1, self.dim_c1, name="c1i") as i3:
                        with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                            with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                                si2 = Coordinate(i2, self.prod(self.fz_shape[1:]), 0, name="di")
                                si3 = Coordinate(i3, self.prod(self.fz_shape[CST.N2:]), 0, name="c1i")
                                si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="hi")
                                si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                                 name="s_ub_outer")
                                new_si3_0 = Coordinate(i3, si3.stride, -1 * si3.stride, name="c1i_0")
                                new_si3_1 = si3
                                src_iter0 = [si2, new_si3_0, si4, si5]
                                src_iter1 = [si2, new_si3_1, si4, si5]
                                di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N2:]), 0, name="di")
                                di3 = Coordinate(i3, self.prod(self.fzg_shape[CST.N3:]),
                                                 self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
                                di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")
                                di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                                 name="d_ub_outer")
                                dst_iter = [dgi, di2, di3, di4, di5]
                                self._ub_split_n_step_2_2(src_iter0, src_iter1, dst_iter, ub_inner)

            def do_step_2_3(ub_bound, ub_inner):
                # i3(c1i) should be dimC1 - 1
                with self.tik.if_scope(self.c1_gap + self.dim_c1 < self.dim_c1x):
                    with self.tik.for_range(0, self.dim_d, name="di") as i2:
                        with self.tik.for_range(0, self.dim_h, name="hi") as i4:
                            with self.tik.for_range(ub_bound[0], ub_bound[1], name="ub_outer") as i5:
                                si2 = Coordinate(i2, self.prod(self.fz_shape[1:]), 0, name="di")
                                di2 = Coordinate(i2, self.prod(self.fzg_shape[CST.N2:]), 0, name="di")
                                si3 = Coordinate(self.dim_c1 - 1, self.prod(self.fz_shape[CST.N2:]), 0, name="c1i")
                                di3 = Coordinate(self.dim_c1, self.prod(self.fzg_shape[CST.N3:]),
                                                 self.c1_gap * self.prod(self.fzg_shape[CST.N3:]), name="c1i")
                                si4 = Coordinate(i4, self.prod(self.fz_shape[CST.N3:]), 0, name="hi")
                                di4 = Coordinate(i4, self.prod(self.fzg_shape[CST.N4:]), 0, name="hi")
                                si5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.src_lower_n * self.dim_c0,
                                                 name="s_ub_outer")
                                di5 = Coordinate(i5, self.ub_factor * self.dim_c0, self.dst_lower_n * self.dim_c0,
                                                 name="d_ub_outer")
                                src_iter = [si2, si3, si4, si5]
                                dst_iter = [dgi, di2, di3, di4, di5]
                                self._ub_split_n_step_2_3(src_iter, dst_iter, ub_inner)

            gi = i0 * self.blk_factor + i1
            self._calc_jump_params(gi)
            # dstTensor-addressing on core-level
            dgi = Coordinate(gi // self.lcm_e, self.prod(self.fzg_shape[1:]), 0, name="gi")

            with self.tik.if_scope(i0 < self.loop_t - 1):
                with self.tik.if_scope(self.is_c0_not_jump):
                    self._dispatching_2(function=do_step_1_1)
                with self.tik.else_scope():
                    self._dispatching_2(function=do_step_2_1)
                    self._dispatching_2(function=do_step_2_2)
                    self._dispatching_2(function=do_step_2_3)
            with self.tik.elif_scope(i1 < self.blk_tail_factor):
                with self.tik.if_scope(self.is_c0_not_jump):
                    self._dispatching_2(function=do_step_1_1)
                with self.tik.else_scope():
                    self._dispatching_2(function=do_step_2_1)
                    self._dispatching_2(function=do_step_2_2)
                    self._dispatching_2(function=do_step_2_3)

    def _do_compute(self):
        self._set_gm_buffer()
        self.tiling_buf = self.tik.Tensor("int64", (CST.N4,), name="tiling", scope=tik.scope_ubuf)
        # the tiling size is not same for fz2fzg and fzg2fz, will get full tiling data at second time
        self.tik.data_move(self.tiling_buf[0], self.tiling_gm[0], 0, 1, 1, 0, 0)
        self.tiling_key = self._malloc_scalar("int64", "tiling_key_", self.tiling_buf[0])
        self.threads = self._malloc_scalar("int64", "threads_", self.tiling_buf[1])

        with self.tik.for_range(0, self.core_nums, block_num=self.core_nums) as blk_idx:
            self.tiling_buf = self.tik.Tensor("int64", (CST.N32,), name="tiling", scope=tik.scope_ubuf)
            with self.tik.if_scope(blk_idx < self.threads):
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZG_TO_FZ_ALIGN_HW):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer_fzg2fz()
                        self._set_tiling_args_fzg2fz()
                        self._init_tiling_fzg2fz()
                        self._do_init_loop_cnt(blk_idx)
                        self._do_mode_c_is_c0_align_hw(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZG_TO_FZ_ALIGN_C1OUT):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer_fzg2fz()
                        self._set_tiling_args_fzg2fz()
                        self._init_tiling_fzg2fz()
                        self._do_init_loop_cnt(blk_idx)
                        self._do_mode_c_is_c0_align_c1out(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZG_TO_FZ_ALIGN_D):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer_fzg2fz()
                        self._set_tiling_args_fzg2fz()
                        self._init_tiling_fzg2fz()
                        self._do_init_loop_cnt(blk_idx)
                        self._do_mode_c_is_c0_align_d(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_HW):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer_fzg2fz()
                        self._set_tiling_args_fzg2fz()
                        self._init_tiling_fzg2fz()
                        self._do_init_loop_cnt(blk_idx)
                        self._do_mode_c_is_c0_unalign_hw(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_D):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer_fzg2fz()
                        self._set_tiling_args_fzg2fz()
                        self._init_tiling_fzg2fz()
                        self._do_init_loop_cnt(blk_idx)
                        self._do_mode_c_is_c0_unalign_d(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_HW_1):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer_fzg2fz()
                        self._set_tiling_args_fzg2fz()
                        self._init_tiling_fzg2fz()
                        self._do_init_loop_cnt(blk_idx)
                        self._do_mode_c_is_c0_unalign_hw_1(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZG_TO_FZ_UNALIGN_D_1):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer_fzg2fz()
                        self._set_tiling_args_fzg2fz()
                        self._init_tiling_fzg2fz()
                        self._do_init_loop_cnt(blk_idx)
                        self._do_mode_c_is_c0_unalign_d_1(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZ_TO_FZG_N_N):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer()
                        self._set_tiling_args()
                        self._init_tiling()
                        self._do_mode_n_n(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZ_TO_FZG_H_N):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer()
                        self._set_tiling_args()
                        self._init_tiling()
                        self._do_mode_h_n(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZ_TO_FZG_C1_N):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer()
                        self._set_tiling_args()
                        self._init_tiling()
                        self._do_mode_c1_n(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZ_TO_FZG_D_N):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer()
                        self._set_tiling_args()
                        self._init_tiling()
                        self._do_mode_d_n(blk_idx)
                with self.tik.if_scope(self.tiling_key == CST.MODE_FZ_TO_FZG_GROUPS_N):
                    with self.tik.new_stmt_scope():
                        self._set_ub_buffer()
                        self._set_tiling_args()
                        self._init_tiling()
                        self._do_mode_groups_n(blk_idx)


def trans_data_groups(src, dst, kernel_name):
    """
    src: dict
        shape, dtype and format of input
    dst: dict
        shape, dtype and format of output
    kernel_name: str
        kernel name
    Returns
    -------
    None
    """
    src_info = Message(src)
    return FZ2FZCompute(src_info, kernel_name).get_tik_instance()
