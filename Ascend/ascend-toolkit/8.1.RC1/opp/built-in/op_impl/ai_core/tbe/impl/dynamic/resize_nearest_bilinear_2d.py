"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

resize_nearest_bilinear_2d
"""
from abc import ABCMeta, abstractmethod
from impl import common_util
from impl.util import util_tik_comm_func
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # ting param num
    TILING_ARG_NUM = 20
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # numbers per block for b64
    B64_PER_BLOCK = 4
    # numbers per block for b32
    B32_PER_BLOCK = 8
    # numbers per block for b16
    B16_PER_BLOCK = 16
    # bytes per block
    BYTE_PER_BLOCK = 32
    # nBurst max value of `data_move`
    BURST_NUM_MAX = 4095
    # burstLen max value of `data_move`
    BURST_LEN_MAX = 65535
    # stride max value of `data_move`
    BURST_STRIDE_MAX = 65535
    # stride max value of 'data_move_pad'
    PAD_STRIDE_MAX = 2**32 - 1
    # max value of int64
    MAX_INT64_VALUE = 2**63 - 1
    # h, w: (m, n) -> (m, n)
    NO_SCALING = 0
    # h, w: (1, 1) -> (m, n)
    ONE_2_MANY = 1
    # h, w: (m, n) -> (1, 1)
    MANY_2_ONE = 2
    # vdp size
    VDP_LEN = 4096


class GetPlatformInfo():
    """
    get platform information
    """

    @staticmethod
    def get_ub_size():
        """
        get ub size
        """
        return tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,too-many-statements,unused-argument,invalid-name
class Resize2D():
    """
    Resize2D  base class
    """

    def __init__(self, images, size, y, align_corners, half_pixel_centers, kernel_name, mode):
        self.images_dtype = images.get("dtype").lower()
        self.images_dtype = self.images_dtype if self.images_dtype != "bfloat16" else "float16"
        self.size_dtype = size.get("dtype").lower()
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        self.kernel_name = kernel_name
        self.mode = 0 if mode == "nearest" else 1

        para_check.check_dtype(self.size_dtype, ("int32", ), param_name="size")
        para_check.check_dtype(self.images_dtype, ("float32", "float16"), param_name="images")

        self.ub_size_bytes = GetPlatformInfo.get_ub_size() - Constant.RESERVED_UB_SIZE
        self.dtype_size = common_util.get_data_size(self.images_dtype)
        self.ele_per_block = Constant.BYTE_PER_BLOCK // self.dtype_size
        self.ub_max_num = self.ub_size_bytes // Constant.BYTE_PER_BLOCK * self.ele_per_block

        self.tik_inst = tik.Tik()
        self.block_idx = None

        self.images_gm = self.tik_inst.Tensor(self.images_dtype, (Constant.MAX_INT64_VALUE, ), tik.scope_gm, "images")
        self.size_gm = self.tik_inst.Tensor(self.size_dtype, (2, ), tik.scope_gm, "size")
        self.out_gm = self.tik_inst.Tensor(self.images_dtype, (Constant.MAX_INT64_VALUE, ), tik.scope_gm, "y")
        self.tiling_gm = self.tik_inst.Tensor("int64", (Constant.TILING_ARG_NUM, ), tik.scope_gm, "tiling")
        self.ub_all = None
        self.ub_index = None

        self.is_support_dmp = tbe_platform.api_check_support("tik.data_move_pad")

        # runtime tiling params
        self.tiling_key = self.tik_inst.Scalar(name="tiling_key")
        self.sub_key = self.tik_inst.Scalar(name="sub_key")
        self.ub_offset = self.tik_inst.Scalar(name="ub_offset")
        self.used_core_cnt = self.tik_inst.Scalar(name="used_core_cnt")
        self.mc_pos = self.tik_inst.Scalar(name="mc_pos")
        self.core_in_offset = self.tik_inst.Scalar(name="core_in_offset")
        self.core_out_offset = self.tik_inst.Scalar(name="core_out_offset")
        self.nc_loop_unit = self.tik_inst.Scalar(name="nc_loop_unit")
        self.h_loop_unit = self.tik_inst.Scalar(name="h_loop_unit")
        self.w_loop_unit = self.tik_inst.Scalar(name="w_loop_unit")
        self.nc_per_core = self.tik_inst.Scalar(name="nc_per_core")
        self.h_per_core = self.tik_inst.Scalar(name="h_per_core")
        self.w_per_core = self.tik_inst.Scalar(name="w_per_core")
        self.nc = self.tik_inst.Scalar(name="nc")
        self.in_h = self.tik_inst.Scalar(name="in_h")
        self.in_w = self.tik_inst.Scalar(name="in_w")
        self.out_h = self.tik_inst.Scalar(name="out_h")
        self.out_w = self.tik_inst.Scalar(name="out_w")

    def _compute_branch(self):
        self._get_tiling_params()
        with self.tik_inst.for_range(0, self.used_core_cnt, block_num=self.used_core_cnt) as block_idx:
            self.block_idx = block_idx
            self.ub_all = self.tik_inst.Tensor(self.images_dtype, (self.ub_max_num, ), tik.scope_ubuf, "ub_all")
            self.ub_index = self.tik_inst.Tensor("int8", (Constant.RESERVED_UB_SIZE, ), tik.scope_ubuf, "ub_index")
            with self.tik_inst.if_scope(self.tiling_key == Constant.NO_SCALING):
                NoScaling(self, self.mode).run()
            with self.tik_inst.elif_scope(self.tiling_key == Constant.ONE_2_MANY):
                One2Many(self, self.mode).run()
            with self.tik_inst.elif_scope(self.tiling_key == Constant.MANY_2_ONE):
                Many2One(self, self.mode).run()

    def compute(self):
        """
        execution entrance
        """

        # compile information for tiling
        tbe_context.get_context().add_compile_info("vars", {"ub_max_num": self.ub_max_num,
                                                            "align_corners": self.align_corners,
                                                            "half_pixel_centers": self.half_pixel_centers,
                                                            "resize_mode": self.mode})
        self._compute_branch()
        # Build CCE
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=(self.images_gm, self.size_gm),
                               outputs=(self.out_gm, ),
                               flowtable=(self.tiling_gm, ), config={"enable_const_fold": True})

        return self.tik_inst

    def _get_tiling_params(self):
        """
        get tiling parameters
        """
        with self.tik_inst.new_stmt_scope():
            tiling_ub = self.tik_inst.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                             name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, util_tik_comm_func.ceil_div(
                Constant.TILING_ARG_NUM, Constant.B64_PER_BLOCK), 0, 0)

            tensor_tup = (self.tiling_key, self.sub_key, self.ub_offset, self.used_core_cnt, self.mc_pos,
                          self.core_in_offset, self.core_out_offset,
                          self.nc_loop_unit, self.h_loop_unit, self.w_loop_unit,
                          self.nc_per_core, self.h_per_core, self.w_per_core,
                          self.nc, self.in_h, self.in_w, self.out_h, self.out_w)
            for idx, tensor in enumerate(tensor_tup):
                tensor.set_as(tiling_ub[idx])


class ProcessorBase(metaclass=ABCMeta):
    """
    base process class
    """
    def __init__(self, resize_op_obj: Resize2D, resize_mode: str = "nearest") -> None:
        self.op = resize_op_obj
        self.tik_inst = resize_op_obj.tik_inst
        self.mode = resize_mode

        # GM in and out start per core
        self.gm_in_start = self.op.core_in_offset * self.op.block_idx
        self.gm_out_start = self.op.core_out_offset * self.op.block_idx

    @abstractmethod
    def init_loop_parameters(self) -> None:
        """
        init H/NC loop parameters once width loop length is done
        """

    @abstractmethod
    def malloc_buf(self) -> None:
        """
        malloc buffer for moving in image and reordering
        """

    @abstractmethod
    def image_process(self) -> None:
        """
        process/resize images: move in -> reorder in UB -> move out
        """

    def run(self) -> None:
        """
        main procedure for resize
        """
        self.init_loop_parameters()
        self.malloc_buf()
        self.image_process()


class NoScaling(ProcessorBase):
    """
    class of h and w have no scaling
    """

    def __init__(self, resize_op_obj: Resize2D, resize_mode: str = "nearest") -> None:
        super().__init__(resize_op_obj, resize_mode)

        self.nchw_size = self.tik_inst.Scalar(name="nchw_size")
        self.nchw_lp_unit = None
        self.nchw_loop_cnt = None
        self.nchw_tail = None
        self.ub_in = None

    def init_loop_parameters(self) -> None:
        with self.tik_inst.if_scope(self.op.block_idx < self.op.used_core_cnt - 1):
            self.nchw_size.set_as(self.op.core_in_offset)
        with self.tik_inst.else_scope():
            self.nchw_size.set_as(self.op.in_w - (self.op.used_core_cnt - 1) * self.op.core_in_offset)
        self.nchw_lp_unit = self.op.w_loop_unit
        self.nchw_loop_cnt = self.nchw_size // self.nchw_lp_unit
        self.nchw_tail = self.nchw_size % self.nchw_lp_unit

    def malloc_buf(self) -> None:
        self.ub_in = self.op.ub_all

    def image_process(self) -> None:
        with self.tik_inst.for_range(0, self.nchw_loop_cnt) as lp_idx:
            self.tik_inst.data_move(self.ub_in,
                                    self.op.images_gm[self.gm_in_start + lp_idx * self.nchw_lp_unit],
                                    0, 1, self.nchw_lp_unit // self.op.ele_per_block, 0, 0)
            self.tik_inst.data_move(self.op.out_gm[self.gm_out_start + lp_idx * self.nchw_lp_unit],
                                    self.ub_in,
                                    0, 1, self.nchw_lp_unit // self.op.ele_per_block, 0, 0)
        with self.tik_inst.if_scope(self.nchw_tail > 0):
            self.tik_inst.data_move(self.ub_in,
                                    self.op.images_gm[self.gm_in_start + self.nchw_loop_cnt * self.nchw_lp_unit],
                                    0, 1, util_tik_comm_func.ceil_div(self.nchw_tail, self.op.ele_per_block), 0, 0)

            if self.op.is_support_dmp is False:
                # to keep invalid data is zero
                pad_size = self.op.ele_per_block - self.nchw_tail % self.op.ele_per_block
                with self.tik_inst.if_scope(pad_size < self.op.ele_per_block):
                    with self.tik_inst.for_range(0, pad_size) as pad_idx:
                        self.ub_in[self.nchw_tail + pad_idx].set_as(0)
                self.tik_inst.data_move(self.op.out_gm[self.gm_out_start + self.nchw_loop_cnt * self.nchw_lp_unit],
                                        self.ub_in,
                                        0, 1, util_tik_comm_func.ceil_div(self.nchw_tail, self.op.ele_per_block), 0, 0)
            else:
                self.tik_inst.data_move_pad(self.op.out_gm[self.gm_out_start + self.nchw_loop_cnt * self.nchw_lp_unit],
                                            self.ub_in,
                                            1, self.nchw_tail * self.op.dtype_size, 0, 0)


class One2Many(ProcessorBase):
    """
    class of h and w from (1, 1) to (m, n)
    """

    def __init__(self, resize_op_obj: Resize2D, resize_mode: str = "nearest") -> None:
        super().__init__(resize_op_obj, resize_mode)

        self.nc_size = self.tik_inst.Scalar(name="nc_size")
        self.hw_size = self.tik_inst.Scalar(name="hw_size")
        self.nc_loop_unit = None
        self.nc_loop_cnt = None
        self.nc_tail = None
        self.hw_loop_unit = None
        self.hw_loop_cnt = None
        self.hw_tail = None
        self.ub_in = None
        self.ub_out = None
        self.ub_idx = None
        self.need_back_cond = self.tik_inst.Scalar(name="need_back_cond", init_value=0)

    def init_loop_parameters(self) -> None:
        """
        init loop parameters
        """
        with self.tik_inst.if_scope(self.op.block_idx < self.op.used_core_cnt - 1):
            self.nc_size.set_as(self.op.nc_per_core)
            self.hw_size.set_as(self.op.w_per_core)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(self.op.mc_pos == 0):
                self.nc_size.set_as(self.op.nc - (self.op.used_core_cnt - 1) * self.op.nc_per_core)
                self.hw_size.set_as(self.op.w_per_core)
            with self.tik_inst.else_scope():
                self.nc_size.set_as(self.op.nc_per_core)
                self.hw_size.set_as(self.op.out_w - (self.op.used_core_cnt - 1) * self.op.w_per_core)

        self.nc_loop_unit = self.op.nc_loop_unit
        self.nc_loop_cnt = self.nc_size // self.nc_loop_unit
        self.nc_tail = self.nc_size % self.nc_loop_unit
        self.hw_loop_unit = self.op.w_loop_unit
        self.hw_loop_cnt = self.hw_size // self.hw_loop_unit
        self.hw_tail = self.hw_size % self.hw_loop_unit

    def malloc_buf(self) -> None:
        self.ub_in = self.op.ub_all
        self.ub_out = self.op.ub_all[self.op.ub_offset]
        self.ub_idx = self.op.ub_index.reinterpret_cast_to("int32")

    def _process_by_vector_dup(self) -> None:
        """
        processs for hw size > one block
        """
        dup_val = self.tik_inst.Scalar(self.ub_in.dtype)

        def _vector_dup(dup_val):
            mask = 8 * self.op.ele_per_block
            loop_cnt = Constant.VDP_LEN // mask
            self.tik_inst.vector_dup(mask, self.ub_out, dup_val, loop_cnt, 1, 8)

        def _broadcast_and_move_out(nc, nc_lp_idx):
            with self.tik_inst.for_range(0, nc) as nc_idx:
                dup_val.set_as(self.ub_in[nc_idx])
                _vector_dup(dup_val)
                with self.tik_inst.for_range(0, self.hw_loop_cnt) as hw_lp_idx:
                    gm_out_offset = (self.gm_out_start + (nc_lp_idx * self.nc_loop_unit + nc_idx)
                                     * self.op.out_w + hw_lp_idx * self.hw_loop_unit)
                    if self.op.is_support_dmp is False:
                        with self.tik_inst.if_scope(tik.all(self.need_back_cond > 0,
                                                            hw_lp_idx == self.hw_loop_cnt - 1)):
                            self.tik_inst.data_move(self.op.out_gm[gm_out_offset],
                                                    self.ub_out,
                                                    0, 1, self.hw_loop_unit // self.op.ele_per_block - 1, 0, 0)
                        with self.tik_inst.else_scope():
                            self.tik_inst.data_move(self.op.out_gm[gm_out_offset],
                                                    self.ub_out,
                                                    0, 1, self.hw_loop_unit // self.op.ele_per_block, 0, 0)
                    else:
                        self.tik_inst.data_move_pad(self.op.out_gm[gm_out_offset],
                                                    self.ub_out,
                                                    1, self.hw_loop_unit * self.op.dtype_size, 0, 0)
                with self.tik_inst.if_scope(self.hw_tail > 0):
                    gm_out_offset = (self.gm_out_start + (nc_lp_idx * self.nc_loop_unit + nc_idx)
                                     * self.op.out_w + self.hw_loop_cnt * self.hw_loop_unit)
                    if self.op.is_support_dmp is False:
                        with self.tik_inst.if_scope(self.hw_tail < self.op.ele_per_block):
                            self.tik_inst.data_move(
                                self.op.out_gm[gm_out_offset - self.op.ele_per_block], self.ub_out, 0, 1, 1, 0, 0)
                        with self.tik_inst.else_scope():
                            self.tik_inst.data_move(self.op.out_gm[gm_out_offset],
                                                    self.ub_out, 0, 1, self.hw_tail // self.op.ele_per_block, 0, 0)
                        backend_len = (self.op.ele_per_block - self.hw_tail % self.op.ele_per_block)
                        floor_align_tail = self.hw_tail // self.op.ele_per_block * self.op.ele_per_block
                        with self.tik_inst.if_scope(backend_len < self.op.ele_per_block):
                            self.tik_inst.data_move(self.op.out_gm[gm_out_offset + floor_align_tail - backend_len],
                                                    self.ub_out,
                                                    0, 1, 1, 0, 0)
                    else:
                        self.tik_inst.data_move_pad(self.op.out_gm[gm_out_offset],
                                                    self.ub_out,
                                                    1, self.hw_tail * self.op.dtype_size, 0, 0)

        with self.tik_inst.if_scope(tik.all(self.op.out_w % self.op.w_per_core > 0,
                                            self.op.out_w % self.op.w_per_core < self.op.ele_per_block,
                                            self.op.mc_pos == 1,
                                            self.op.block_idx == self.op.used_core_cnt - 2)):
            self.need_back_cond.set_as(1)

        with self.tik_inst.for_range(0, self.nc_loop_cnt) as nc_lp_idx:
            self.tik_inst.data_move(self.ub_in,
                                    self.op.images_gm[self.gm_in_start + nc_lp_idx * self.nc_loop_unit],
                                    0, 1, self.nc_loop_unit // self.op.ele_per_block, 0, 0)
            _broadcast_and_move_out(self.nc_loop_unit, nc_lp_idx)
        with self.tik_inst.if_scope(self.nc_tail > 0):
            self.tik_inst.data_move(self.ub_in,
                                    self.op.images_gm[self.gm_in_start + self.nc_loop_cnt * self.nc_loop_unit],
                                    0, 1, util_tik_comm_func.ceil_div(self.nc_tail, self.op.ele_per_block), 0, 0)
            _broadcast_and_move_out(self.nc_tail, self.nc_loop_cnt)

    def _gen_vgather_idx(self, nc_cnt, w_len):
        with self.tik_inst.for_range(0, nc_cnt) as nc_idx:
            with self.tik_inst.for_range(0, w_len) as w_idx:
                self.ub_idx[nc_idx * w_len + w_idx].set_as(nc_idx * self.op.dtype_size)

    def _process_by_vgather(self) -> None:

        def _vgather_process(nc_len):
            inner_lp_cnt = nc_len // inner_nc_lp_unit
            with self.tik_inst.for_range(0, inner_lp_cnt) as nc_idx:
                vg_base_addr.set_as(inner_nc_lp_unit * nc_idx * self.op.dtype_size)
                self.tik_inst.vgather(inner_nc_lp_unit * self.op.out_w,
                                      self.ub_out[nc_idx * inner_nc_lp_unit * self.op.out_w],
                                      self.ub_in,
                                      self.ub_idx,
                                      1, 8, vg_base_addr, mask_mode="counter")
            with self.tik_inst.if_scope(nc_len % inner_nc_lp_unit > 0):
                vg_base_addr.set_as(inner_lp_cnt * inner_nc_lp_unit * self.op.dtype_size)
                self.tik_inst.vgather(nc_len % inner_nc_lp_unit * self.op.out_w,
                                      self.ub_out[inner_lp_cnt * inner_nc_lp_unit * self.op.out_w],
                                      self.ub_in,
                                      self.ub_idx,
                                      1, 8, vg_base_addr, mask_mode="counter")

        idx_gate = 256
        vg_base_addr = self.tik_inst.Scalar("uint32", init_value=0)

        if self.op.dtype_size == 2:
            inner_nc_lp_unit = idx_gate * 2 // self.op.out_w // self.op.ele_per_block * self.op.ele_per_block
        else:
            inner_nc_lp_unit = idx_gate // self.op.out_w // self.op.ele_per_block * self.op.ele_per_block
        self._gen_vgather_idx(inner_nc_lp_unit, self.op.out_w)

        # the nc_loop_unit is block align
        with self.tik_inst.for_range(0, self.nc_loop_cnt) as nc_lp_idx:
            self.tik_inst.data_move(self.ub_in,
                                    self.op.images_gm[self.gm_in_start + nc_lp_idx * self.nc_loop_unit],
                                    0, 1, self.nc_loop_unit // self.op.ele_per_block, 0, 0)
            _vgather_process(self.nc_loop_unit)
            out_offset = self.gm_out_start + nc_lp_idx * self.nc_loop_unit * self.op.out_w
            if self.op.is_support_dmp is False:
                self.tik_inst.data_move(self.op.out_gm[out_offset],
                                        self.ub_out,
                                        0, 1, self.nc_loop_unit * self.op.out_w // self.op.ele_per_block, 0, 0)
            else:
                self.tik_inst.data_move_pad(self.op.out_gm[out_offset],
                                            self.ub_out,
                                            1, self.nc_loop_unit * self.op.out_w * self.op.dtype_size, 0, 0)
        with self.tik_inst.if_scope(self.nc_tail > 0):
            self.tik_inst.data_move(self.ub_in,
                                    self.op.images_gm[self.gm_in_start + self.nc_loop_cnt * self.nc_loop_unit],
                                    0, 1, util_tik_comm_func.ceil_div(self.nc_tail, self.op.ele_per_block), 0, 0)
            _vgather_process(self.nc_tail)
            out_offset = self.gm_out_start + self.nc_loop_cnt * self.nc_loop_unit * self.op.out_w
            if self.op.is_support_dmp is False:
                self.tik_inst.data_move(self.op.out_gm[out_offset],
                                        self.ub_out,
                                        0, 1, util_tik_comm_func.ceil_div(self.nc_tail * self.op.out_w,
                                                                          self.op.ele_per_block), 0, 0)
            else:
                self.tik_inst.data_move_pad(self.op.out_gm[out_offset],
                                            self.ub_out,
                                            1, self.nc_tail * self.op.out_w * self.op.dtype_size, 0, 0)

    def image_process(self) -> None:
        with self.tik_inst.if_scope(self.op.sub_key == 0):
            self._process_by_vector_dup()
        with self.tik_inst.else_scope():
            self._process_by_vgather()


class Many2One(ProcessorBase):
    """
    class of h and w from (m, n) to (1, 1)
    """

    def __init__(self, resize_op_obj: Resize2D, resize_mode: str = "nearest") -> None:
        super().__init__(resize_op_obj, resize_mode)

        self.nc_size = self.tik_inst.Scalar(name="nc_size")
        self.nc_loop_unit = None
        self.nc_loop_cnt = None
        self.nc_tail = None
        self.ub_in = None
        self.ub_out = None
        self.ub_idx = None
        self.hw_pos = None
        self.hw_size = self.op.in_h * self.op.in_w

    def init_loop_parameters(self) -> None:
        with self.tik_inst.if_scope(self.op.block_idx < self.op.used_core_cnt - 1):
            self.nc_size.set_as(self.op.nc_per_core)
        with self.tik_inst.else_scope():
            self.nc_size.set_as(self.op.nc - (self.op.used_core_cnt - 1) * self.op.nc_per_core)
        self.nc_loop_unit = self.op.nc_loop_unit
        self.nc_loop_cnt = self.nc_size // self.nc_loop_unit
        self.nc_tail = self.nc_size % self.nc_loop_unit

        if self.op.half_pixel_centers is False:
            self.hw_pos = 0
        else:
            self.hw_pos = self.op.in_h // 2 * self.op.in_w + self.op.in_w // 2

    def malloc_buf(self) -> None:
        self.ub_in = self.op.ub_all
        self.ub_out = self.op.ub_all[self.op.ub_offset]
        self.ub_idx = self.op.ub_index.reinterpret_cast_to("int32")

    def _gen_vgather_idx(self, idx_gate):
        with self.tik_inst.for_range(0, idx_gate) as nc_idx:
            self.ub_idx[nc_idx].set_as((nc_idx * self.hw_size) * self.op.dtype_size)

    def _process_by_vgather(self) -> None:
        idx_gate = 1024 // self.op.dtype_size
        self._gen_vgather_idx(idx_gate)

        def _inner_process(nc_len):
            inner_nc_lp_unit = idx_gate
            inner_lp_cnt = nc_len // inner_nc_lp_unit
            inner_tail = nc_len % inner_nc_lp_unit
            vg_base_addr = self.tik_inst.Scalar("uint32", init_value=0)
            with self.tik_inst.for_range(0, inner_lp_cnt) as nc_idx:
                vg_base_addr.set_as(nc_idx * inner_nc_lp_unit * self.hw_size * self.op.dtype_size)
                self.tik_inst.vgather(inner_nc_lp_unit,
                                      self.ub_out[nc_idx * inner_nc_lp_unit],
                                      self.ub_in,
                                      self.ub_idx,
                                      1, 8, vg_base_addr, mask_mode="counter")
            with self.tik_inst.if_scope(inner_tail > 0):
                vg_base_addr.set_as(inner_lp_cnt * inner_nc_lp_unit * self.hw_size * self.op.dtype_size)
                self.tik_inst.vgather(inner_tail,
                                      self.ub_out[inner_lp_cnt * inner_nc_lp_unit],
                                      self.ub_in,
                                      self.ub_idx,
                                      1, 8, vg_base_addr, mask_mode="counter")

        # the nc_loop_unit is block align
        with self.tik_inst.for_range(0, self.nc_loop_cnt) as nc_lp_idx:
            in_offset = nc_lp_idx * self.nc_loop_unit * self.hw_size + self.gm_in_start + self.hw_pos
            self.tik_inst.data_move(self.ub_in,
                                    self.op.images_gm[in_offset],
                                    0, 1, self.nc_loop_unit * self.hw_size // self.op.ele_per_block, 0, 0)
            _inner_process(self.nc_loop_unit)
            out_offset = self.gm_out_start + nc_lp_idx * self.nc_loop_unit
            self.tik_inst.data_move(self.op.out_gm[out_offset],
                                    self.ub_out,
                                    0, 1, self.nc_loop_unit // self.op.ele_per_block, 0, 0)
        with self.tik_inst.if_scope(self.nc_tail > 0):
            in_offset = self.nc_loop_cnt * self.nc_loop_unit * self.hw_size + self.gm_in_start + self.hw_pos
            self.tik_inst.data_move(self.ub_in,
                                    self.op.images_gm[in_offset],
                                    0, 1, util_tik_comm_func.ceil_div(self.nc_tail * self.hw_size,
                                                                      self.op.ele_per_block), 0, 0)
            _inner_process(self.nc_tail)
            out_offset = self.gm_out_start + self.nc_loop_cnt * self.nc_loop_unit
            if self.op.is_support_dmp is False:
                self.tik_inst.data_move(self.op.out_gm[out_offset],
                                        self.ub_out,
                                        0, 1, util_tik_comm_func.ceil_div(self.nc_tail,
                                                                          self.op.ele_per_block), 0, 0)
            else:
                self.tik_inst.data_move_pad(self.op.out_gm[out_offset],
                                            self.ub_out,
                                            1, self.nc_tail * self.op.dtype_size, 0, 0)

    def _copy_data_in(self, nc_len, in_offset):
        if self.op.is_support_dmp is False:
            src_stride = self.hw_size // self.op.ele_per_block - 1
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.if_scope(tik.all(self.hw_size % self.op.ele_per_block == 0,
                                                    src_stride <= Constant.BURST_STRIDE_MAX)):
                    self.tik_inst.data_move(self.ub_in, self.op.images_gm[in_offset],
                                            0, nc_len, 1, src_stride, 0)
                with self.tik_inst.else_scope():
                    with self.tik_inst.for_range(0, nc_len) as idx:
                        self.tik_inst.data_move(self.ub_in[idx * self.op.ele_per_block],
                                                self.op.images_gm[in_offset + idx * self.hw_size],
                                                0, 1, 1, 0, 0)
        else:
            src_stride = (self.hw_size - 1) * self.op.dtype_size
            with self.tik_inst.if_scope(src_stride <= Constant.PAD_STRIDE_MAX):
                self.tik_inst.data_move_pad(self.ub_in, self.op.images_gm[in_offset],
                                            nc_len, self.op.dtype_size, 0, src_stride)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, nc_len) as idx:
                    self.tik_inst.data_move_pad(self.ub_in[idx * self.op.ele_per_block],
                                                self.op.images_gm[in_offset + idx * self.hw_size],
                                                1, self.op.dtype_size, 0, 0)

    def _process_by_vreduce(self):
        if self.op.dtype_size == 2:
            ub_idx = self.ub_idx.reinterpret_cast_to("uint16")
            for i in range(0, 8):
                ub_idx[i].set_as(1)
        else:
            ub_idx = self.ub_idx.reinterpret_cast_to("uint16")
            ub_in_b16 = self.ub_in.reinterpret_cast_to("float16")
            ub_out_b16 = self.ub_out.reinterpret_cast_to("float16")
            ele_per_block = self.op.ele_per_block * 2
            for i in range(0, 8):
                ub_idx[i].set_as(3)

        with self.tik_inst.for_range(0, self.nc_loop_cnt) as nc_lp_idx:
            in_offset = nc_lp_idx * self.nc_loop_unit * self.hw_size + self.gm_in_start + self.hw_pos
            self._copy_data_in(self.nc_loop_unit, in_offset)
            if self.op.dtype_size != 2:
                self.tik_inst.vreduce(self.nc_loop_unit * ele_per_block, ub_out_b16,
                                      ub_in_b16, ub_idx, 1, 1, 8, 0, mask_mode="counter")
            else:
                self.tik_inst.vreduce(self.nc_loop_unit * self.op.ele_per_block, self.ub_out,
                                      self.ub_in, ub_idx, 1, 1, 8, 0, mask_mode="counter")
            self.tik_inst.data_move(self.op.out_gm[self.gm_out_start + nc_lp_idx * self.nc_loop_unit],
                                    self.ub_out, 0, 1, self.nc_loop_unit // self.op.ele_per_block, 0, 0)
        with self.tik_inst.if_scope(self.nc_tail > 0):
            in_offset = self.nc_loop_cnt * self.nc_loop_unit * self.hw_size + self.gm_in_start + self.hw_pos
            self._copy_data_in(self.nc_tail, in_offset)
            if self.op.dtype_size != 2:
                self.tik_inst.vreduce(self.nc_tail * ele_per_block, ub_out_b16,
                                      ub_in_b16, ub_idx, 1, 1, 8, 0, mask_mode="counter")
            else:
                self.tik_inst.vreduce(self.nc_tail * self.op.ele_per_block, self.ub_out,
                                      self.ub_in, ub_idx, 1, 1, 8, 0, mask_mode="counter")
            if self.op.is_support_dmp is False:
                self.tik_inst.data_move(self.op.out_gm[self.gm_out_start + self.nc_loop_cnt * self.nc_loop_unit],
                                        self.ub_out,
                                        0, 1, util_tik_comm_func.ceil_div(self.nc_tail, self.op.ele_per_block), 0, 0)
            else:
                self.tik_inst.data_move_pad(self.op.out_gm[self.gm_out_start + self.nc_loop_cnt * self.nc_loop_unit],
                                            self.ub_out, 1, self.nc_tail * self.op.dtype_size, 0, 0)

    def image_process(self) -> None:
        with self.tik_inst.if_scope(self.op.sub_key == 0):
            self._process_by_vgather()
        with self.tik_inst.else_scope():
            self._process_by_vreduce()


# 'pylint: disable=huawei-too-many-arguments
def resize_2d(images, size, y, align_corners, half_pixel_centers,
              kernel_name, mode="nearest"):
    """
    Parameters
    ----------
    images: dict
        the dict of input, include shape of input_tensor which layout
        only support NCHW and dtype supports 'float16', 'float32' and 'bfloat16'
    size: dict
        the dict of input, positive height and width of output tensor
        only support 1D and dtype supports 'int32'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support NCHW and dtype supports 'float16', 'float32' and 'bfloat16'
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
        align_corners & half_pixel_centers can not be both TRUE in zoom-out scenario,
        otherwise, the input index will be out of range
    kernel_name: str
        cce kernel name, default value is `resize_nearest_neighbor`
    mode: str
        method selection, it can be 'bilinear' or 'nearest', default value is 'nearest'

    Returns
    -------
    tik_inst
    """

    obj = Resize2D(images, size, y, align_corners, half_pixel_centers, kernel_name, mode)
    return obj.compute()
