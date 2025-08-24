# Copyright 2023 Huawei Technologies Co., Ltd
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
grid_sampler_3d
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl import constant_util as constant


# 'pylint: disable=too-many-return-statements,too-many-arguments,unused-argument,huawei-too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-few-public-methods
def check_supported(x, grid, y, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False,
                    kernel_name="grid_sampler_3d"):
    """
    Parameters
    ----------
    x : dict. shape and dtype of input data x
    grid : dict. shape and dtype of input data grid
    y : dict. shape and dtype of input data y
    interpolation_mode : value of attr interpolation_mode
    padding_mode : value of attr padding_mode
    align_corners : value of attr align_corners
    kernel_name : str. cce kernel name, default value is "grid_sampler_3d"

    Returns
    -------
    True or False
    """
    # platform check
    if not tbe_platform.api_check_support("tik.vgather"):
        return False, "No support for this platform"

    # dtype check
    x_dtype = x.get("dtype").lower()
    grid_dtype = grid.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    if any((x_dtype not in ("float32"), grid_dtype != x_dtype, y_dtype != x_dtype)):
        return False, "Only support float32"

    # shape check
    x_shape = x.get("shape")
    grid_shape = grid.get("shape")
    if len(x_shape) != 5:
        return False, "x shape length should be 5"
    if len(grid_shape) != 5:
        return False, "grid shape length should be 5"

    # attr check
    if interpolation_mode != 'bilinear':
        return False, "interpolation_mode only support bilinear"

    if padding_mode not in ('zeros', 'border'):
        return False, "padding_mode only support zeros and border"

    return True, ""


class Constant:
    ELEMENT_PER_LOOP = 2048
    ELEMENT_PER_LOOP_MINI = 1024
    CHANNEL_PER_LOOP = 4 * 1024
    UNROLL_NUM = 16
    TILING_PARAMS_NUM = 16
    UB_SIZE = 262144


class Gm:
    def __init__(self, tik_instance, d_type):
        self.tiling = tik_instance.Tensor(constant.DATA_TYPE_INT64, (Constant.TILING_PARAMS_NUM,), name="tiling_gm",
                                          scope=tik.scope_gm)
        self.x = tik_instance.Tensor(d_type, (constant.SHAPE_SIZE_LIMIT,), name="x_gm", scope=tik.scope_gm)
        self.grid = tik_instance.Tensor(d_type, (constant.SHAPE_SIZE_LIMIT,), name="grid_gm", scope=tik.scope_gm)
        self.y = tik_instance.Tensor(d_type, (constant.SHAPE_SIZE_LIMIT,), name="y_gm", scope=tik.scope_gm,
                                     is_atomic_add=True)


class Shape:
    def __init__(self, tik_instance, tiling):
        """"""
        self.batch = tik_instance.Scalar(constant.DATA_TYPE_INT64, "N")
        self.channel = tik_instance.Scalar(constant.DATA_TYPE_INT64, "C")
        self.input_d = tik_instance.Scalar(constant.DATA_TYPE_INT64, "ID")
        self.input_h = tik_instance.Scalar(constant.DATA_TYPE_INT64, "IH")
        self.input_w = tik_instance.Scalar(constant.DATA_TYPE_INT64, "IW")
        self.output_d = tik_instance.Scalar(constant.DATA_TYPE_INT64, "OD")
        self.output_h = tik_instance.Scalar(constant.DATA_TYPE_INT64, "OH")
        self.output_w = tik_instance.Scalar(constant.DATA_TYPE_INT64, "OW")
        self.input_stride_d = tik_instance.Scalar(constant.DATA_TYPE_INT32, "ISZ")
        self.input_stride_h = tik_instance.Scalar(constant.DATA_TYPE_INT32, "ISY")
        self.input_stride_c = tik_instance.Scalar(constant.DATA_TYPE_INT32, "ISC")
        self.input_stride_n = tik_instance.Scalar(constant.DATA_TYPE_INT64, "ISN")
        self.output_stride_c = tik_instance.Scalar(constant.DATA_TYPE_INT32, "OSC")
        self.output_stride_n = tik_instance.Scalar(constant.DATA_TYPE_INT32, "OSN")
        self.batch.set_as(tiling[1])
        self.channel.set_as(tiling[2])
        self.input_d.set_as(tiling[3])
        self.input_h.set_as(tiling[4])
        self.input_w.set_as(tiling[5])
        self.output_d.set_as(tiling[6])
        self.output_h.set_as(tiling[7])
        self.output_w.set_as(tiling[8])
        self.input_stride_d.set_as(self.input_h * self.input_w)
        self.input_stride_h.set_as(self.input_w)
        self.input_stride_c.set_as(self.input_d * self.input_stride_d)
        self.input_stride_n.set_as(self.input_stride_c * self.channel)
        self.output_stride_c.set_as(self.output_d * self.output_h * self.output_w)
        self.output_stride_n.set_as(self.output_stride_c * self.channel)


class Ub:
    def __init__(self, tik_instance, d_type, ele_num, dim_num=3):
        self.mask1 = tik_instance.Tensor(constant.DATA_TYPE_UINT16, (ele_num * 2,), name="mask1", scope=tik.scope_ubuf)
        self.mask2 = tik_instance.Tensor(constant.DATA_TYPE_UINT16, (ele_num * 2,), name="mask2", scope=tik.scope_ubuf)
        self.grid = tik_instance.Tensor(d_type, (ele_num * dim_num,), name="grid_ub", scope=tik.scope_ubuf)
        self.grid_trans = tik_instance.Tensor(d_type, (ele_num * dim_num,), name="grid_trans_ub", scope=tik.scope_ubuf)
        self.tmp_int = tik_instance.Tensor(constant.DATA_TYPE_INT32, (ele_num,), name="tmp_int_ub",
                                           scope=tik.scope_ubuf)
        self.tmp_fp = tik_instance.Tensor(d_type, (ele_num,), name="tmp_fp_ub", scope=tik.scope_ubuf)
        self.weight = tik_instance.Tensor(d_type, (ele_num,), name="weight_ub", scope=tik.scope_ubuf)
        self.location = tik_instance.Tensor(constant.DATA_TYPE_INT32, (ele_num,), name="location_ub",
                                            scope=tik.scope_ubuf)
        self.out = tik_instance.Tensor(d_type, (ele_num,), name="out_ub", scope=tik.scope_ubuf)

        # ix -> ix - floor(ix)
        self.ix = tik_instance.Tensor(d_type, (ele_num,), name="ix_ub", scope=tik.scope_ubuf)
        self.iy = tik_instance.Tensor(d_type, (ele_num,), name="iy_ub", scope=tik.scope_ubuf)
        self.iz = tik_instance.Tensor(d_type, (ele_num,), name="iz_ub", scope=tik.scope_ubuf)

        # ix_1 -> 1 - ix
        self.ix_1 = tik_instance.Tensor(d_type, (ele_num,), name="ix_1_ub", scope=tik.scope_ubuf)
        self.iy_1 = tik_instance.Tensor(d_type, (ele_num,), name="iy_1_ub", scope=tik.scope_ubuf)
        self.iz_1 = tik_instance.Tensor(d_type, (ele_num,), name="iz_1_ub", scope=tik.scope_ubuf)

        # ix_tnw -> int(ix)
        self.ix_tnw = tik_instance.Tensor(constant.DATA_TYPE_INT32, (ele_num,), name="ix_tnw_ub", scope=tik.scope_ubuf)
        self.iy_tnw = tik_instance.Tensor(constant.DATA_TYPE_INT32, (ele_num,), name="iy_tnw_ub", scope=tik.scope_ubuf)
        self.iz_tnw = tik_instance.Tensor(constant.DATA_TYPE_INT32, (ele_num,), name="iz_tnw_ub", scope=tik.scope_ubuf)

        # ix_tnw_1 -> ix_tnw + 1
        self.ix_tnw_1 = tik_instance.Tensor(constant.DATA_TYPE_INT32, (ele_num,), name="ix_tnw_1_ub",
                                            scope=tik.scope_ubuf)
        self.iy_tnw_1 = tik_instance.Tensor(constant.DATA_TYPE_INT32, (ele_num,), name="iy_tnw_1_ub",
                                            scope=tik.scope_ubuf)
        self.iz_tnw_1 = tik_instance.Tensor(constant.DATA_TYPE_INT32, (ele_num,), name="iz_tnw_1_ub",
                                            scope=tik.scope_ubuf)


class Reg:
    def __init__(self, tik_instance):
        self.num = tik_instance.Scalar(constant.DATA_TYPE_INT64, "num")
        self.repeat_times = tik_instance.Scalar(constant.DATA_TYPE_INT64, "repeat_times")
        self.burst_len = tik_instance.Scalar(constant.DATA_TYPE_INT64, "burst_len")
        self.input_offset = tik_instance.Scalar(constant.DATA_TYPE_INT64, "input_offset")
        self.out_offset = tik_instance.Scalar(constant.DATA_TYPE_INT64, "out_offset")
        self.offset = tik_instance.Scalar(constant.DATA_TYPE_INT64, "offset")
        self.batch_idx = tik_instance.Scalar(constant.DATA_TYPE_INT64, "batch_idx")


class Sc:
    def __init__(self, tik_instance, data_type):
        self.loc = [tik_instance.Scalar(constant.DATA_TYPE_INT32, "index") for _ in range(Constant.UNROLL_NUM)]
        self.start = tik_instance.Scalar(constant.DATA_TYPE_INT64, "start")
        self.tmp_int = tik_instance.Scalar(constant.DATA_TYPE_INT32, "tmp_int")
        self.tmp_int64 = tik_instance.Scalar(constant.DATA_TYPE_INT32, "tmp_int64")
        self.tmp_fp = tik_instance.Scalar(data_type, "tmp_fp")


class Util:
    @staticmethod
    def data_type_size(d_type):
        return tbe_platform.get_bit_len(d_type) // constant.DATA_SIZE_EIGHT

    @staticmethod
    def burst_size(num, d_type):
        return (num * Util.data_type_size(d_type) + constant.BLOCK_SIZE - 1) // constant.BLOCK_SIZE

    @staticmethod
    def ceil(x, y):
        return (x + y - 1) // y

    @staticmethod
    def align(x, y):
        return Util.ceil(x, y) * y


class Tiling:
    def __init__(self, tik_instance):
        self.tik = tik_instance
        self.ub = self.tik.Tensor(constant.DATA_TYPE_INT64, (Constant.TILING_PARAMS_NUM,), name="tiling",
                                  scope=tik.scope_ubuf)
        self.core_num = self.tik.Scalar(constant.DATA_TYPE_INT64, "core_num", init_value=8)
        self.element_total = self.tik.Scalar(constant.DATA_TYPE_INT64, "element_total")
        self.element_per_core = self.tik.Scalar(constant.DATA_TYPE_INT64, "element_per_core")
        self.element_per_loop = self.tik.Scalar(constant.DATA_TYPE_INT64, "element_per_loop")


class GridSampler3D:
    # 'pylint: disable=too-many-return-statements,too-many-arguments,disable=unused-argument
    def __init__(self, x, grid, y, interpolation_mode, padding_mode, align_corners, kernel_name):
        self.x = x
        self.grid = grid
        self.kernel_name = kernel_name

        self.tik = tik.Tik(tik.Dprofile)
        self.d_type = x.get("dtype").lower()
        self.block_item_size = constant.BLOCK_SIZE / Util.data_type_size(self.d_type)
        self.default_mask = constant.VECTOR_BYTE_SIZE // Util.data_type_size(self.d_type)
        self.support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad")
        self.gm = Gm(self.tik, self.d_type)
        self.sc = Sc(self.tik, self.d_type)
        self.tiling = Tiling(self.tik)
        # register parameters
        self.reg = Reg(self.tik)
        self.ub = None
        self.shape = None
        self.channel_last = 0
        x_format = x.get("format").upper()
        if x_format == "NDHWC":
            self.channel_last = 1
        self.align_corners = align_corners
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode

        self.ele_num_per_loop = Constant.ELEMENT_PER_LOOP
        if tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) < Constant.UB_SIZE:
            self.ele_num_per_loop = Constant.ELEMENT_PER_LOOP_MINI

    def compute(self):
        self._tiling()

        with self.tik.for_range(0, self.tiling.core_num, block_num=self.tiling.core_num) as core_id:
            self.tik.set_atomic_add(self.d_type)

            # Bind cores to batch axis when batch is much greater than DxHxW
            with self.tik.for_range(0, self.shape.batch) as n:
                self.sc.start.set_as(core_id * self.tiling.element_per_core)
                self.reg.batch_idx.set_as(n)
                self._process_core(self.sc.start)
            self.tik.set_atomic_add(0)

        tbe_context.get_context().add_compile_info(
            "vars", {"core_num": tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik.BuildCCE(kernel_name=self.kernel_name,
                          inputs=[self.gm.x, self.gm.grid],
                          outputs=[self.gm.y],
                          flowtable=(self.gm.tiling,),
                          config=opt_config)

    def _data_move(self, dst, src, num=None, d_type=None):
        d_type = self.d_type if d_type is None else d_type
        burst_len = Util.burst_size(num, d_type) if num is not None else self.reg.burst_len
        self.tik.data_move(dst, src, 0, 1, burst_len, 0, 0)

    def _data_move_pad(self, dst, src, num=None, d_type=None):
        d_type = self.d_type if d_type is None else d_type
        if self.support_data_move_pad:
            _num = num if num is not None else self.reg.num
            self.tik.data_move_pad(dst, src, 1, _num * Util.data_type_size(d_type), 0, 0)
        else:
            burst_len = Util.burst_size(num, d_type) if num is not None else self.reg.burst_len
            self.tik.data_move(dst, src, 0, 1, burst_len, 0, 0)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _vector_s(self, func, dst, src, scalar, num=None):
        repeat_times = Util.ceil(num, self.default_mask) if num is not None else self.reg.repeat_times
        func(self.default_mask, dst, src, scalar, repeat_times, 1, 1, 8, 8)
        return dst

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _vector_v(self, func, dst, src1, src2, num=None, mask=None):
        repeat_times = Util.ceil(num, self.default_mask) if num is not None else self.reg.repeat_times
        vec_mask = self.default_mask if mask is None else mask
        func(vec_mask, dst, src1, src2, repeat_times, 1, 1, 1, 8, 8, 8)
        return dst

    def _conv(self, mode, dst, src, num=None):
        repeat_times = Util.ceil(num, self.default_mask) if num is not None else self.reg.repeat_times
        self.tik.vconv(self.default_mask, mode, dst, src, repeat_times, 1, 1, 8, 8)
        return dst

    def _simple_trans_fp32_2k_3(self, dst, src, num):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.tik.v4dtrans(False, dst, src, num, 3)
        else:  # only support fp32 (2048, 3) -> (3, 2048)
            ub1 = self.tik.Tensor("float32", (2048 * 3,), tik.scope_ubuf, "ub1")
            ub2 = self.tik.Tensor("float32", (2048 * 3,), tik.scope_ubuf, "ub2")
            with self.tik.for_range(0, 6) as ii:
                src_list = [src[i * 48 + ii * 8] for i in range(16)]
                dst_list = [ub1[i * 8 + ii * 128] for i in range(16)]
                self.tik.vnchwconv(False, False, dst_list, src_list, 8, 96, 96)
            with self.tik.for_range(0, 3) as ii:
                src_list = [ub1[i * 48 + ii * 16] for i in range(16)]
                dst_list = [ub2[i * 8 + ii * 2048] for i in range(16)]
                self.tik.vnchwconv(False, False, dst_list, src_list, 8, 32, 96)
            with self.tik.for_range(0, 3) as ii:
                src_list = [ub1[i * 48 + ii * 16 + 8] for i in range(16)]
                dst_list = [ub2[i * 8 + ii * 2048 + 128] for i in range(16)]
                self.tik.vnchwconv(False, False, dst_list, src_list, 8, 32, 96)
            with self.tik.for_range(0, 3) as ii:
                self.tik.vadds(self.default_mask, dst[num * ii], ub2[2048 * ii], 0,
                               num // self.default_mask, 1, 1, 8, 8)

    def _simple_trans_fp32_64_3(self, dst, src):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.tik.v4dtrans(False, dst, src, self.default_mask, 3)
        elif self.default_mask == 64:
            ub1 = self.tik.Tensor("float32", (256 * 3,), tik.scope_ubuf, "ub1")
            ub2 = self.tik.Tensor("float32", (256 * 3,), tik.scope_ubuf, "ub2")

            src_list = [src[i * 48] for i in range(16)]
            dst_list = [ub1[i * 8] for i in range(16)]
            self.tik.vnchwconv(False, False, dst_list, src_list, 6, 16, 1)

            src_list = [ub1[i * 48] for i in range(16)]
            dst_list = [ub2[i * 8] for i in range(16)]
            self.tik.vnchwconv(False, False, dst_list, src_list, 3, 16, 2)

            self.tik.vadds(self.default_mask, dst, ub2, 0, 3, 1, 1, 8, 16)

    def _simple_trans_fp32_64_8x(self, dst, src, channel_align):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.tik.v4dtrans(False, dst, src, self.default_mask, channel_align)
        elif self.default_mask == 64:  # only support channel_align le 64
            with self.tik.if_scope(channel_align == 8):
                src_list = [src[i * 8] for i in range(16)]
                dst_list = []
                for ii in range(8):
                    dst_list = dst_list + [dst[64 * ii], dst[64 * ii + 8]]
                self.tik.vnchwconv(False, False, dst_list, src_list, 4, 2, 16)
            with self.tik.else_scope():
                with self.tik.for_range(0, 4) as ii:
                    src_list = [src[i * channel_align + ii * 16 * channel_align] for i in range(16)]
                    dst_list = []
                    for i in range(8):
                        dst_list = dst_list + [dst[64 * i + ii * 16], dst[64 * i + 8 + ii * 16]]
                    self.tik.vnchwconv(False, False, dst_list, src_list, channel_align // 8, 64, 1)

    def _simple_trans_fp32_x_64(self, dst, src, channel):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.tik.v4dtrans(True, dst, src, self.default_mask, channel)
        elif self.default_mask == 64:  # only support channel_align le 64
            channel_align = Util.align(channel, self.block_item_size)
            with self.tik.if_scope(channel < channel_align):
                self.tik.vec_dup(self.default_mask, src[self.default_mask * channel], 0, channel_align - channel, 8)
                self.tik.vec_dup(self.default_mask, dst, 0, channel_align, 8)
            with self.tik.if_scope(channel_align == 8):
                src_list = [src[i * 64] for i in range(8)] + [src[i * 64 + 8] for i in range(8)]
                dst_list = []
                for ii in range(8):
                    dst_list = dst_list + [dst[8 * ii], dst[8 * ii + 64]]
                self.tik.vnchwconv(False, False, dst_list, src_list, 4, 16, 2)
            with self.tik.else_scope():
                with self.tik.for_range(0, 4) as ii:
                    src_list = [src[i * 64 + ii * 16] for i in range(8)] + [src[i * 64 + 8 + ii * 16] for i in range(8)]
                    dst_list = []
                    for i in range(8):
                        dst_list = dst_list + [dst[channel_align * (i + ii * 16)],
                                               dst[channel_align * (i + 8 + ii * 16)]]
                    self.tik.vnchwconv(False, False, dst_list, src_list, channel_align // 8, 1, 64)

    def _move_to_gm(self, t, channel_align, out_val):
        out_offset = self.reg.out_offset
        per_items = self.default_mask

        if tbe_platform.api_check_support("tik.v4dtrans"):
            self._data_move_pad(self.gm.y[out_offset + t * per_items * self.shape.channel], out_val,
                                per_items * self.shape.channel)
        else:
            with self.tik.if_scope(self.shape.channel == channel_align):
                self._data_move(self.gm.y[out_offset + t * per_items * self.shape.channel], out_val,
                                per_items * self.shape.channel)
            with self.tik.else_scope():
                with self.tik.for_range(0, per_items) as ii:
                    self._data_move_pad(self.gm.y[out_offset + t * per_items * self.shape.channel +
                                                  ii * self.shape.channel],
                                        out_val[ii * channel_align], self.shape.channel)

    def _unnormalize(self, ix, iy, iz, align_corners):
        if align_corners:
            # Calc `((coord + 1) * (size - 1)) / 2`
            self.ub.ix = self._vector_s(self.tik.vmuls, self.ub.ix, ix, 0.5 * (self.shape.input_w - 1))
            self.ub.iy = self._vector_s(self.tik.vmuls, self.ub.iy, iy, 0.5 * (self.shape.input_h - 1))
            self.ub.iz = self._vector_s(self.tik.vmuls, self.ub.iz, iz, 0.5 * (self.shape.input_d - 1))
        else:
            # Calc `((coord + 1) * size - 1)) / 2`
            ix = self._vector_s(self.tik.vmuls, ix, ix, 0.5 * self.shape.input_w)
            self.ub.ix = self._vector_s(self.tik.vadds, self.ub.ix, ix, -0.5)
            iy = self._vector_s(self.tik.vmuls, iy, iy, 0.5 * self.shape.input_h)
            self.ub.iy = self._vector_s(self.tik.vadds, self.ub.iy, iy, -0.5)
            iz = self._vector_s(self.tik.vmuls, iz, iz, 0.5 * self.shape.input_d)
            self.ub.iz = self._vector_s(self.tik.vadds, self.ub.iz, iz, -0.5)
        return self.ub.ix, self.ub.iy, self.ub.iz

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _compute_index(self, ix, ix_1, ix_tnw, ix_tnw_1, size):
        if self.padding_mode == "border":
            tmp_sc = self.sc.tmp_fp
            tmp_sc.set_as(size - 1)
            self._vector_s(self.tik.vmins, ix, ix, tmp_sc)
            self._vector_s(self.tik.vmaxs, ix, ix, 0.0)
        ix_tnw = self._conv("floor", ix_tnw, ix)
        tmp = self._conv("none", self.ub.tmp_fp, ix_tnw)
        # ix -> ix - ix_tnw
        ix = self._vector_v(self.tik.vsub, ix, ix, tmp)
        # ix_tnw_1 -> ix_tnw + 1
        ix_tnw_1 = self._vector_s(self.tik.vadds, ix_tnw_1, ix_tnw, 1)
        # ix_1 -> 1 - ix
        ix_1 = self._vector_s(self.tik.vmuls, ix_1, ix, -1.0)
        ix_1 = self._vector_s(self.tik.vadds, ix_1, ix_1, 1.0)

    def _process_point_channel_last(self, location, tnw):
        """channel last"""
        self.sc.tmp_int.set_as(self.shape.channel)
        location = self._vector_s(self.tik.vmuls, location, location, self.sc.tmp_int)
        self.sc.tmp_int.set_as(self.reg.input_offset)
        location = self._vector_s(self.tik.vadds, location, location, self.sc.tmp_int)

        with self.tik.if_scope(self.shape.channel > 64):
            self._process_point_large_channel_last(location, tnw)
        with self.tik.else_scope():
            out_offset = self.reg.out_offset

            # Process [128/64 x channel] items each time.
            loc = self.sc.loc[0]
            tnw_val = self.sc.tmp_fp
            per_items = self.default_mask
            channel_align = Util.align(self.shape.channel, self.block_item_size)
            out_val = self.tik.Tensor(self.d_type, (per_items * channel_align,), name="out_val",
                                      scope=tik.scope_ubuf)
            x_tmp = self.tik.Tensor(self.d_type, (per_items * channel_align,), name="x_tmp",
                                    scope=tik.scope_ubuf)
            # Process 128/64 align parts.
            with self.tik.for_range(0, self.reg.num / per_items) as t:
                # [1] Move in 128or64 x [channel] @gm --> [128or64, channel_align] @ub
                with self.tik.for_range(0, per_items) as i:
                    loc.set_as(location[t * per_items + i])
                    self._data_move_pad(x_tmp[channel_align * i], self.gm.x[loc], self.shape.channel)

                # [2] [128or64, channel_align] --> [channel_align, 128or64]
                self._simple_trans_fp32_64_8x(out_val, x_tmp, channel_align)

                # [3] x_value x weight, [channel, 128or64] x [128or64,], repeat channel times.
                self.tik.vmul(self.default_mask, x_tmp, out_val, tnw[per_items * t], self.shape.channel,
                              1, 1, 1, 8, 8, 0)

                # [4] [channel_align, 128or64] --> [128or64, channel]
                self._simple_trans_fp32_x_64(out_val, x_tmp, self.shape.channel)

                # [5] Move out [128or64, channel]
                self._move_to_gm(t, channel_align, out_val)

            # Process tail parts which less than 128/64.
            current_offset = self.reg.num / per_items * per_items
            with self.tik.if_scope(self.reg.num > current_offset):
                self.tik.vec_dup(self.default_mask, out_val, 0.0, channel_align, 8)
                with self.tik.for_range(current_offset, self.reg.num) as i:
                    loc.set_as(location[i])
                    self._data_move_pad(x_tmp, self.gm.x[loc], self.shape.channel)
                    tnw_val.set_as(tnw[i])
                    self._vector_s(self.tik.vmuls, x_tmp, x_tmp, tnw_val, self.shape.channel)
                    with self.tik.for_range(0, self.shape.channel) as j:
                        out_val[i * self.shape.channel + j].set_as(x_tmp[j])
                self._data_move_pad(self.gm.y[out_offset + current_offset * self.shape.channel], out_val,
                                    (self.reg.num - current_offset) * self.shape.channel)

    def _process_point_large_channel_last(self, location, tnw):
        out_offset = self.reg.out_offset
        loc = self.sc.loc[0]
        channel_mask_align = Util.align(self.shape.channel, self.default_mask)

        channel_loop = channel_mask_align // Constant.CHANNEL_PER_LOOP
        repeat_times_loop = Constant.CHANNEL_PER_LOOP // self.default_mask
        channel_loop_tail = channel_mask_align % Constant.CHANNEL_PER_LOOP
        channel_loop_tail_no_pad = self.shape.channel % Constant.CHANNEL_PER_LOOP
        repeat_times_tail = channel_loop_tail // self.default_mask

        tnw_scalar = self.tik.Scalar(self.d_type, "tnw_scalar")
        out_val = self.tik.Tensor(self.d_type, (Constant.CHANNEL_PER_LOOP,), name="out_val", scope=tik.scope_ubuf)

        with self.tik.for_range(0, self.reg.num) as t:
            loc.set_as(location[t])
            tnw_scalar.set_as(tnw[t])
            with self.tik.for_range(0, channel_loop) as l:
                self.tik.vec_dup(self.default_mask, out_val, 0.0, repeat_times_loop, 8)
                # [1] Move in [channel_align,] @gm --> [channel_align,] @ub
                self._data_move(out_val, self.gm.x[loc + l * Constant.CHANNEL_PER_LOOP], Constant.CHANNEL_PER_LOOP)
                # [2] x_value x weight, [channel_align,] x weight
                self.tik.vmuls(self.default_mask, out_val, out_val, tnw_scalar, repeat_times_loop, 1, 1, 8, 8)
                # [3] Move out [channel_align,]
                self._data_move(self.gm.y[out_offset + t * self.shape.channel + l * Constant.CHANNEL_PER_LOOP], out_val,
                                Constant.CHANNEL_PER_LOOP)
            with self.tik.if_scope(channel_loop_tail > 0):
                offset = channel_loop * Constant.CHANNEL_PER_LOOP
                self.tik.vec_dup(self.default_mask, out_val, 0.0, repeat_times_tail, 8)
                # [1] Move in [channel_tail,] @gm --> [channel_tail,] @ub
                self._data_move_pad(out_val, self.gm.x[loc + offset], channel_loop_tail_no_pad)
                # [2] x_value x weight, [channel_align,] x weight
                self.tik.vmuls(self.default_mask, out_val, out_val, tnw_scalar, repeat_times_tail, 1, 1, 8, 8)
                # [3] Process tail parts which less than 64/128x
                with self.tik.for_range(self.shape.channel - offset, channel_loop_tail) as i:
                    out_val[i].set_as(0)
                # [4] Move out [channel_tail,]
                self._data_move_pad(self.gm.y[out_offset + t * self.shape.channel + offset],
                                    out_val, channel_loop_tail_no_pad)

    def _process_point_channel_first(self, location, tnw):
        """Channel not last"""
        out_val = self.ub.out
        with self.tik.for_range(0, self.shape.channel) as channel:
            offset = channel * self.shape.input_stride_c
            self.tik.vec_dup(self.default_mask, out_val, 0.0, self.reg.repeat_times, 8)
            with self.tik.for_range(0, self.reg.num / Constant.UNROLL_NUM) as i:
                for j in range(Constant.UNROLL_NUM):
                    self.sc.loc[j].set_as(location[i * Constant.UNROLL_NUM + j])
                for j in range(Constant.UNROLL_NUM):
                    out_val[i * Constant.UNROLL_NUM + j].set_as(self.gm.x[self.reg.input_offset + offset +
                                                                          self.sc.loc[j]])
            with self.tik.for_range(self.reg.num / Constant.UNROLL_NUM * Constant.UNROLL_NUM, self.reg.num) as i:
                # The tail block of unroll.
                self.sc.loc[0].set_as(location[i])
                out_val[i].set_as(self.gm.x[self.reg.input_offset + offset + self.sc.loc[0]])
            out_val = self._vector_v(self.tik.vmul, out_val, out_val, tnw)
            self._data_move_pad(self.gm.y[self.reg.out_offset + channel * self.shape.output_stride_c], out_val)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _process_boundary(self, location, tnw, ix_tnw, iy_tnw, iz_tnw):
        """Ensure the address is valid"""
        tmp_sc = self.sc.tmp_fp
        tmp_fp = self.ub.tmp_fp
        # ix >= 0 && ix < input_h
        ix_fp = self._conv("none", tmp_fp, ix_tnw)
        self.tik.vcmpvs_ge(self.ub.mask1, ix_fp, 0, self.reg.repeat_times, 1, 8)
        tmp_sc.set_as(self.shape.input_w)
        self.tik.vcmpvs_lt(self.ub.mask2, ix_fp, tmp_sc, self.reg.repeat_times, 1, 8)
        self._vector_v(self.tik.vand, self.ub.mask1, self.ub.mask1, self.ub.mask2, mask=constant.MASK128)

        # iy >= 0 && iy < input_h
        iy_fp = self._conv("none", tmp_fp, iy_tnw)
        self.tik.vcmpvs_ge(self.ub.mask2, iy_fp, 0, self.reg.repeat_times, 1, 8)
        self._vector_v(self.tik.vand, self.ub.mask1, self.ub.mask1, self.ub.mask2, mask=constant.MASK128)
        tmp_sc.set_as(self.shape.input_h)
        self.tik.vcmpvs_lt(self.ub.mask2, iy_fp, tmp_sc, self.reg.repeat_times, 1, 8)
        self._vector_v(self.tik.vand, self.ub.mask1, self.ub.mask1, self.ub.mask2, mask=constant.MASK128)

        # iz >= 0 && iz < input_d
        iz_fp = self._conv("none", tmp_fp, iz_tnw)
        self.tik.vcmpvs_ge(self.ub.mask2, iz_fp, 0, self.reg.repeat_times, 1, 8)
        self._vector_v(self.tik.vand, self.ub.mask1, self.ub.mask1, self.ub.mask2, mask=constant.MASK128)
        tmp_sc.set_as(self.shape.input_d)
        self.tik.vcmpvs_lt(self.ub.mask2, iz_fp, tmp_sc, self.reg.repeat_times, 1, 8)
        self._vector_v(self.tik.vand, self.ub.mask1, self.ub.mask1, self.ub.mask2, mask=constant.MASK128)

        # Set location to zero with mask
        tmp_int = self.ub.tmp_int
        self.tik.vec_dup(self.default_mask, tmp_fp, 1.0, self.reg.repeat_times, 8)
        self.tik.vsel(self.default_mask, 1, tmp_fp, self.ub.mask1, tmp_fp, 0.0,
                      self.reg.repeat_times, 1, 1, 1, 8, 8, 8)
        tnw = self._vector_v(self.tik.vmul, tnw, tnw, tmp_fp)
        self._conv('round', tmp_int, tmp_fp)
        location = self._vector_v(self.tik.vmul, location, location, tmp_int)
        return location, tnw

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _process_point(self, ix_1, iy_1, iz_1, ix_tnw, iy_tnw, iz_tnw):
        # Calculate weight
        # tnw : `(1-ix) * (1-iy) * (1-iz) = ix_1 * iy_1 * iz_1``
        tnw = self.ub.weight
        tnw = self._vector_v(self.tik.vmul, tnw, ix_1, iy_1)
        tnw = self._vector_v(self.tik.vmul, tnw, tnw, iz_1)

        # Calculate location
        # location channel first: iz_tnw * self.shape.IH * self.shape.IW + iy_tnw * self.shape.IW + ix_tnw
        # location channel last:     location * self.shape.C
        location = self.ub.location
        tmp_int = self.ub.tmp_int
        location = self._vector_s(self.tik.vmuls, location, iz_tnw, self.shape.input_stride_d)
        tmp_int = self._vector_s(self.tik.vmuls, tmp_int, iy_tnw, self.shape.input_stride_h)
        location = self._vector_v(self.tik.vadd, location, location, tmp_int)
        location = self._vector_v(self.tik.vadd, location, location, ix_tnw)

        location, tnw = self._process_boundary(location, tnw, ix_tnw, iy_tnw, iz_tnw)
        if self.channel_last == 1:
            self._process_point_channel_last(location, tnw)
        else:
            self._process_point_channel_first(location, tnw)

    def _process_loop(self):
        self._compute_index(self.ub.ix, self.ub.ix_1, self.ub.ix_tnw, self.ub.ix_tnw_1, self.shape.input_w)
        self._compute_index(self.ub.iy, self.ub.iy_1, self.ub.iy_tnw, self.ub.iy_tnw_1, self.shape.input_h)
        self._compute_index(self.ub.iz, self.ub.iz_1, self.ub.iz_tnw, self.ub.iz_tnw_1, self.shape.input_d)

        ix, ix_1, ix_tnw, ix_tnw_1 = self.ub.ix, self.ub.ix_1, self.ub.ix_tnw, self.ub.ix_tnw_1
        iy, iy_1, iy_tnw, iy_tnw_1 = self.ub.iy, self.ub.iy_1, self.ub.iy_tnw, self.ub.iy_tnw_1
        iz, iz_1, iz_tnw, iz_tnw_1 = self.ub.iz, self.ub.iz_1, self.ub.iz_tnw, self.ub.iz_tnw_1

        ix_tne, iy_tne, iz_tne = [ix_tnw_1, iy_tnw, iz_tnw]
        ix_tsw, iy_tsw, iz_tsw = [ix_tnw, iy_tnw_1, iz_tnw]
        ix_tse, iy_tse, iz_tse = [ix_tnw_1, iy_tnw_1, iz_tnw]
        ix_bnw, iy_bnw, iz_bnw = [ix_tnw, iy_tnw, iz_tnw_1]
        ix_bne, iy_bne, iz_bne = [ix_tnw_1, iy_tnw, iz_tnw_1]
        ix_bsw, iy_bsw, iz_bsw = [ix_tnw, iy_tnw_1, iz_tnw_1]
        ix_bse, iy_bse, iz_bse = [ix_tnw_1, iy_tnw_1, iz_tnw_1]

        # Calculate tnw
        self._process_point(ix_1, iy_1, iz_1, ix_tnw, iy_tnw, iz_tnw)
        # Calculate tne
        self._process_point(ix, iy_1, iz_1, ix_tne, iy_tne, iz_tne)
        # Calculate tsw
        self._process_point(ix_1, iy, iz_1, ix_tsw, iy_tsw, iz_tsw)
        # Calculate tse
        self._process_point(ix, iy, iz_1, ix_tse, iy_tse, iz_tse)
        # Calculate bnw
        self._process_point(ix_1, iy_1, iz, ix_bnw, iy_bnw, iz_bnw)
        # Calculate bne
        self._process_point(ix, iy_1, iz, ix_bne, iy_bne, iz_bne)
        # Calculate bsw
        self._process_point(ix_1, iy, iz, ix_bsw, iy_bsw, iz_bsw)
        # Calculate bse
        self._process_point(ix, iy, iz, ix_bse, iy_bse, iz_bse)

    def _process_loop_align(self, start, num):
        # calc reg params
        self.reg.num.set_as(num)
        self.reg.repeat_times.set_as(num // self.default_mask)
        self.reg.offset.set_as(self.reg.batch_idx * self.shape.output_stride_c + start)
        self.reg.input_offset.set_as(self.reg.batch_idx * self.shape.input_stride_n)
        if self.channel_last == 1:
            self.reg.out_offset.set_as(self.reg.batch_idx * self.shape.output_stride_n + start * self.shape.channel)
        else:
            self.reg.out_offset.set_as(self.reg.batch_idx * self.shape.output_stride_n + start)
        self.reg.burst_len.set_as(Util.burst_size(num, self.d_type))

        # [1] Move in grid
        self._data_move(self.ub.grid, self.gm.grid[self.reg.offset * 3], num * 3)

        # [2] grid + 1.0
        self._vector_s(self.tik.vadds, self.ub.grid, self.ub.grid, 1.0, num * 3)

        # [3] [grid_index, 3] --> [3, grid_index]
        self._simple_trans_fp32_2k_3(self.ub.grid_trans, self.ub.grid, num)

        # [4] unnormalize x/y/z
        ix = self.ub.grid_trans[0]
        iy = self.ub.grid_trans[self.reg.num]
        iz = self.ub.grid_trans[self.reg.num * 2]
        self._unnormalize(ix, iy, iz, self.align_corners)
        self._process_loop()

    def _process_loop_tail(self, start, num):
        """num less than 128/64"""
        self.reg.num.set_as(num)
        self.reg.repeat_times.set_as(1)
        self.reg.offset.set_as(self.reg.batch_idx * self.shape.output_stride_c + start)
        self.reg.input_offset.set_as(self.reg.batch_idx * self.shape.input_stride_n)
        if self.channel_last == 1:
            self.reg.out_offset.set_as(self.reg.batch_idx * self.shape.output_stride_n + start * self.shape.channel)
        else:
            self.reg.out_offset.set_as(self.reg.batch_idx * self.shape.output_stride_n + start)
        self.reg.burst_len.set_as(Util.burst_size(num, self.d_type))

        # [1] Move in grid
        self.tik.vec_dup(self.default_mask, self.ub.grid, 0.0, self.reg.repeat_times * 3, 8)
        self._data_move_pad(self.ub.grid, self.gm.grid[self.reg.offset * 3], self.reg.num * 3)

        # [2] grid + 1.0
        self._vector_s(self.tik.vadds, self.ub.grid, self.ub.grid, 1.0, self.reg.num * 3)

        # [3] [grid_index, 3] --> [3, grid_index]
        self._simple_trans_fp32_64_3(self.ub.grid_trans, self.ub.grid)

        # [4] unnormalize x/y/z
        ix = self.ub.grid_trans[0]
        iy = self.ub.grid_trans[self.default_mask]
        iz = self.ub.grid_trans[self.default_mask * 2]
        self._unnormalize(ix, iy, iz, self.align_corners)
        self._process_loop()

    def _process_core(self, start):
        element_per_core = self.tik.Scalar(constant.DATA_TYPE_INT64, "element_per_core_local")
        element_per_core.set_as(self.tiling.element_per_core)
        with self.tik.if_scope(self.tiling.element_per_core > self.tiling.element_total - start):
            element_per_core.set_as(self.tiling.element_total - start)

        # Process normal align blocks.
        with self.tik.for_range(0, element_per_core / self.tiling.element_per_loop):
            self._process_loop_align(start, self.tiling.element_per_loop)
            start.set_as(start + self.tiling.element_per_loop)

        # Process the align(128/64) parts in tail blocks.
        element_per_core.set_as(element_per_core % self.tiling.element_per_loop)
        element_per_loop_align = self.tik.Scalar(constant.DATA_TYPE_INT64, "element_per_loop_align")
        with self.tik.if_scope(element_per_core >= self.default_mask):
            element_per_loop_align.set_as(element_per_core - element_per_core % self.default_mask)
            self._process_loop_align(start, element_per_loop_align)
            start.set_as(start + element_per_loop_align)

        # Process the parts less than 128/64 in tail blocks.
        element_per_core.set_as(element_per_core - element_per_loop_align)
        with self.tik.if_scope(element_per_core > 0):
            self._process_loop_tail(start, element_per_core)

    def _tiling(self):
        """Tiling"""
        self._data_move(self.tiling.ub, self.gm.tiling, Constant.TILING_PARAMS_NUM, constant.DATA_TYPE_INT64)
        # Update params with tiling data.
        self.shape = Shape(self.tik, self.tiling.ub)
        self.tiling.core_num.set_as(self.tiling.ub[0])
        self.tiling.element_total.set_as(self.shape.output_d * self.shape.output_h * self.shape.output_w)
        # Align to 128 or 64
        self.tiling.element_per_core.set_as(
            Util.align(Util.ceil(self.tiling.element_total, self.tiling.core_num), self.default_mask))
        self.tiling.core_num.set_as(Util.ceil(self.tiling.element_total, self.tiling.element_per_core))
        # Element_per_loop must be aligned to 128 or 64(depending on the data type)
        # The tail block is handled separately.
        self.tiling.element_per_loop.set_as(self.ele_num_per_loop)
        self.ub = Ub(self.tik, self.d_type, self.tiling.element_per_loop)


# 'pylint: disable=redefined-builtin,huawei-too-many-arguments
@register_operator("GridSampler3D")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def grid_sampler_3d(x, grid, y, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False,
                    kernel_name="grid_sampler_3d"):
    """
    Compute GridSampler3D

    Parameters
    ----------
    x : dict. shape and dtype of input data x
    grid : dict. shape and dtype of input data grid
    y : dict. shape and dtype of input data y
    interpolation_mode : value of attr interpolation_mode
    padding_mode : value of attr padding_mode
    align_corners : value of attr align_corners
    kernel_name : str. cce kernel name, default value is "grid_sampler_3d"

    Returns
    -------
    None
    """
    obj = GridSampler3D(x, grid, y, interpolation_mode, padding_mode, align_corners, kernel_name)
    obj.compute()
