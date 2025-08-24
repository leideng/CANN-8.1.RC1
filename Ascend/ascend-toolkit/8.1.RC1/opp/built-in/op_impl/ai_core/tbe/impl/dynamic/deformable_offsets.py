#!/usr/bin/python
# -*- coding: utf-8 -*-
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
deformable_offsets
"""
from impl import constant_util as constant
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from tbe.common.platform.platform_info import get_soc_spec
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_unknown
from impl.util.util_common import is_dynamic_input


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    MAX_INT64 = 2 ** 63 - 1
    TILING_ALIGN_SIZE = 160
    TILING_PARAM_NUM = 17
    INT64 = "int64"
    BLOCK_BYTES_SIZE = 32
    VECTOR_BYTES_SIZE = 256
    BLOCK_SIZE = 32
    MAX_REPEAT = 255
    MAX_MASK = 64
    TILING_MODE_1 = 1
    TILING_MODE_2 = 2
    TILING_MODE_3 = 3
    TILING_MODE_4 = 4
    THREAD_NUM_1 = 1
    THREAD_NUM_2 = 2
    MIN_PERF_LIMMIT = 1024
    DYNAMIC_MIN_PERF_LIMMIT = 2048
    TYPE_LEN_DICT = {"bfloat16": 2, "float16": 2, "float32": 4, "int64": 8, "int32": 4}


def is_unknown_attr(attr_list):
    if None in attr_list:
        return True
    return False


def _ceil_div(value, block):
    """
    upper division
    """
    return (value + block - 1) // block


def _ceil_align(value, block):
    """
    upper align
    """
    return _ceil_div(value, block) * block


def get_dtype_size(datatype):
    """
    get data size unit, means one element of this input datatype takes up nbyte space

    Parameters
    ----------
    datatype: datatype supports float32,float16,bfloat16

    Returns
    -------
    data_size: one element of this input datatype takes up nbyte space
    """
    datatype_map = {
        "float32": constant.DATA_SIZE_FOUR,
        "float16": constant.DATA_SIZE_TWO,
        "bfloat16": constant.DATA_SIZE_TWO,
    }
    data_size = datatype_map.get(datatype)
    if data_size is None:
        raise RuntimeError("datatype %s is not support!" % (datatype))

    return data_size


# 'pylint: disable=unused-argument,unused-variable
# 'pylint: disable=too-many-arguments,too-many-locals,too-many-return-statements
def check_supported(x,
                    offsets,
                    y,
                    strides,
                    pads,
                    ksize,
                    dilations=(1, 1, 1, 1),
                    data_format="NHWC",
                    deformable_groups=1,
                    modulated=True,
                    kernel_name="deformable_offsets"):
    """
    check whether ai_core is supported
    """
    check_list = ("float32", "float16", "bfloat16")
    x_dtype = x.get("dtype").lower()
    x_format = x.get("format")
    if not modulated:
        reason = "modulated is False"
        return False, reason
    if x_dtype not in check_list:
        reason = "dtype of x is not supported, x_dtype is %s, supported list is %s" % (x_dtype, check_list)
        return False, reason
    if x_format != "NHWC":
        reason = "x_format is not NHWC"
        return False, reason
    if deformable_groups != 1:
        reason = "deformable_groups is not 1"
        return False, reason

    x_shape = x.get("shape")
    core_num = get_soc_spec("CORE_NUM")
    # dynamic shape
    if (is_dynamic_input(x) and len(x_shape) == 4 and x_shape[3] > 0):
        if core_num * x_shape[3] <= Constant.DYNAMIC_MIN_PERF_LIMMIT:
            reason = "dim_c[%s] is not supported now because of dynamic perf" % (str(x_shape[3]),)
            return False, reason
    if is_unknown([x, offsets]):
        return "Unknown"
    # static shape
    if len(x_shape) != 4:
        reason = "len of x_shape is not 4, x_shape is %s" % (str(x_shape),)
        return False, reason
    group_c = x_shape[3] // deformable_groups
    d_size = get_dtype_size(x_dtype)
    block_size = Constant.BLOCK_BYTES_SIZE // d_size
    if group_c % block_size != 0:
        reason = "group_c[%s] is not multiple of block_size[%s]" % (str(group_c), str(block_size))
        return False, reason

    dim_kh = ksize[0]
    dim_kw = ksize[1]
    dim_c = x_shape[3]
    total_ub = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    elem_num_offsets_filter = deformable_groups * dim_kh * dim_kw * 3
    elem_num_aligned = _ceil_align(elem_num_offsets_filter, block_size)
    ub_elem = dim_kw * dim_c + elem_num_aligned
    ub_global = 4 * dim_c
    if x_dtype in ("float16", "bfloat16"):
        ub_global = ub_global + ub_global * 2
        ub_elem = ub_elem + ub_elem * 2
    max_ub_elem = (total_ub - Constant.TILING_ALIGN_SIZE) // d_size
    ub_seg_size = (max_ub_elem - ub_global) // ub_elem
    if ub_seg_size <= 0:
        reason = "size needed exceed ub_size"
        return False, reason

    if core_num * dim_c < Constant.MIN_PERF_LIMMIT:
        reason = "dim_c[%s] is not supported now because of perf" % (str(dim_c),)
        return False, reason

    return True, ""


# 'pylint: disable=too-many-instance-attributes,too-many-arguments
class DeformableOffsets:
    """
    initialize some properties
    """

    # 'pylint: disable=too-many-statements
    def __init__(self, x, offsets, y, strides, pads, ksize, dilations, data_format, deformable_groups,
                 kernel_name):
        self.dtype = x.get("dtype").lower()
        check_list = ("float32", "float16", "bfloat16")
        para_check.check_dtype(self.dtype, check_list, param_name="x")
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.tiling_dtype = Constant.INT64
        self.is_unknown_rank = False
        if is_unknown_rank_input((x, offsets)) or is_unknown_attr([strides, pads, ksize, dilations]):
            self.is_unknown_rank = True
            x["shape"] = (-1, -1, -1, -1)
            x["range"] = ((1, None), (1, None), (1, None), (1, None))
            offsets["shape"] = (-1, -1, -1, -1)
            offsets["range"] = ((1, None), (1, None), (1, None), (1, None))
            self.stride_h = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="stride_h")
            self.stride_w = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="stride_w")
            self.pads_h = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="pads_h")
            self.pads_w = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="stride_h")
            self.dim_kh = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_kh")
            self.dim_kw = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_kw")
            self.dilation_h = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dilation_h")
            self.dilation_w = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dilation_w")
            self.x_shape = x.get("shape")
            self.offsets_shape = offsets.get("shape")
        else:
            self.ksize = ksize
            self.strides = strides
            self.pads = pads
            self.x_shape = x.get("shape")
            self.offsets_shape = offsets.get("shape")
            self._check_param()
            if data_format == "NHWC":
                _, self.stride_h, self.stride_w, _ = strides
                _, self.dilation_h, self.dilation_w, _ = dilations
            else:
                _, _, self.stride_h, self.stride_w = strides
                _, _, self.dilation_h, self.dilation_w = dilations
            self.pads_h = pads[0]
            self.pads_w = pads[2]
            self.dim_kh = ksize[0]
            self.dim_kw = ksize[1]
        self.tiling_dsize = Constant.TYPE_LEN_DICT.get(self.tiling_dtype)
        self.dsize = Constant.TYPE_LEN_DICT.get(self.dtype)
        self.kernel_name = kernel_name
        self.dim_group = deformable_groups
        self.data_in_one_block = Constant.BLOCK_BYTES_SIZE // self.dsize
        self.elem_num_offsets_filter = self.dim_group * self.dim_kh * self.dim_kw * 3
        self.elem_num_aligned = _ceil_align(self.elem_num_offsets_filter, self.data_in_one_block)
        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.core_num = tik.Dprofile().get_aicore_num()

        self.x_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT64, ], name="x_gm",
                                             scope=tbe_platform.scope_gm)
        self.offsets_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT64, ], name="offsets_gm",
                                                   scope=tbe_platform.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT64, ], name="y_gm",
                                             scope=tbe_platform.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(
            self.tiling_dtype, (Constant.TILING_PARAM_NUM,), name="tiling_gm", scope=tik.scope_gm
        )

        self.x_scope = None
        self.ub_x = None

        self.ub_out_tmp = None
        self.ub_offsets_ori_tmp = None
        self.ub_lrt_x_tmp = None
        self.ub_lrb_x_tmp = None

        self.ub_out = None
        self.ub_offsets_ori = None
        self.ub_lrt_x = None
        self.ub_lrb_x = None

        self.tiling_mode = 0
        self.real_core_num = 0
        self.total_wc_num = 0
        self.dim_offsets_h = 0
        self.dim_offsets_w = 0
        self.dim_group_c = 0
        self.dim_c = 0
        self.dim_h_in = 0
        self.dim_w_in = 0
        self.ub_seg_size = 1
        self.loop_seg = 0
        self.ub_seg_res = 0

    def _check_param(self):
        """
        Check if the shape and dtype of input be right.

        Parameters:
        ----------
        None.(Get from class member.)

        Returns:
        -------
        None.
        Error will report when exception happened.
        """
        para_check.check_shape(self.x_shape, min_rank=4, max_rank=4, param_name="x")
        para_check.check_shape(self.offsets_shape, min_rank=4, max_rank=4, param_name="offsets")
        if len(self.ksize) != 2:
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, "the length of ksize should be 2",
                                                              "strides", self.ksize)
        if len(self.strides) != 4:
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, "the length of strides should be 4",
                                                              "strides", self.strides)
        if len(self.pads) != 4:
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, "the length of pads should be 4",
                                                              "pads", self.pads)

    def deformable_offset_compute(self):
        """
        The main comute func.
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            # get tiling data
            tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_PARAM_NUM,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)

            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                        Constant.TILING_ALIGN_SIZE // Constant.BLOCK_BYTES_SIZE, 0, 0)

            # get run info
            self.get_tiling_args(tiling_ub)
            with self.tik_instance.if_scope(block_id < self.real_core_num):
                # get each_cycle and block_offset
                wc_per_core = self.total_wc_num // self.real_core_num
                tail = self.total_wc_num % self.real_core_num
                each_cycle = self.tik_instance.Scalar(init_value=wc_per_core,
                                                        dtype=Constant.INT64, name="each_cycle")
                block_offset = self.tik_instance.Scalar(init_value=wc_per_core * block_id,
                                                        dtype=Constant.INT64, name="block_offset")
                with self.tik_instance.if_scope(tail > 0):
                    with self.tik_instance.if_scope(block_id < tail):
                        each_cycle.set_as(wc_per_core + 1)
                        block_offset.set_as(each_cycle * block_id)
                    with self.tik_instance.else_scope():
                        block_offset.set_as(wc_per_core * block_id + tail)

                # different branches are used based on the value of tiling_mode
                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
                    self.compute_mode_1(each_cycle, block_offset, Constant.THREAD_NUM_2)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_2):
                        self.compute_mode_2(each_cycle, block_offset, Constant.THREAD_NUM_2)
                    with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_3):
                        self.compute_mode_2(each_cycle, block_offset, Constant.THREAD_NUM_1)
                    with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_4):
                        self.compute_mode_1(each_cycle, block_offset, Constant.THREAD_NUM_1)

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }
        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                                            "ub_size": self.total_ub,
                                                            "dsize": self.dsize,
                                                            })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.x_gm, self.offsets_gm],
                              outputs=[self.y_gm],
                              flowtable=(self.tiling_gm,),
                              enable_l2=True, config=opt_config)

    def get_tiling_args(self, tiling_ub):
        self.tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
        self.tiling_mode.set_as(tiling_ub[0])
        self.real_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="real_core_num")
        self.real_core_num.set_as(tiling_ub[1])
        self.total_wc_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="total_wc_num")
        self.total_wc_num.set_as(tiling_ub[2])
        self.dim_offsets_h = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_offsets_h")
        self.dim_offsets_h.set_as(tiling_ub[3])
        self.dim_offsets_w = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_offsets_w")
        self.dim_offsets_w.set_as(tiling_ub[4])
        self.dim_c = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_c")
        self.dim_c.set_as(tiling_ub[5])
        self.dim_h_in = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_h_in")
        self.dim_h_in.set_as(tiling_ub[6])
        self.dim_w_in = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_w_in")
        self.dim_w_in.set_as(tiling_ub[7])
        self.ub_seg_size = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="ub_seg_size")
        self.ub_seg_size.set_as(tiling_ub[8])
        self.dim_group_c = self.dim_c // self.dim_group
        if self.is_unknown_rank:
            self.stride_h.set_as(tiling_ub[9])
            self.stride_w.set_as(tiling_ub[10])
            self.dilation_h.set_as(tiling_ub[11])
            self.dilation_w.set_as(tiling_ub[12])
            self.pads_h.set_as(tiling_ub[13])
            self.pads_w.set_as(tiling_ub[14])
            self.dim_kh.set_as(tiling_ub[15])
            self.dim_kw.set_as(tiling_ub[16])

    def compute_mode_1(self, each_cycle, block_offset, thread_num):
        """
        Compute mode 1.
        """
        with self.tik_instance.for_range(0, each_cycle, thread_num=thread_num) as wc_index:
            sw_start = block_offset + wc_index
            offsets_w_start = sw_start // self.dim_kh
            kh_idx = sw_start % self.dim_kh
            n_idx = offsets_w_start // self.dim_offsets_h
            self.x_scope = self.x_gm
            self.compute_one_w(n_idx, kh_idx, offsets_w_start, sw_start)

    def compute_mode_2(self, each_cycle, block_offset, thread_num):
        """
        Compute mode 2.
        """
        n_idx_loaded = self.tik_instance.Scalar(init_value=-1, dtype=Constant.INT64, name="n_idx_loaded")
        self.ub_x = self.tik_instance.Tensor(self.dtype, (self.dim_h_in * self.dim_w_in * self.dim_c,),
                        scope=tbe_platform.scope_ubuf, name="ub_x")
        self.x_scope = self.ub_x
        with self.tik_instance.for_range(0, each_cycle, thread_num=thread_num) as wc_index:
            sw_start = block_offset + wc_index
            offsets_w_start = sw_start // self.dim_kh
            kh_idx = sw_start % self.dim_kh
            n_idx = offsets_w_start // self.dim_offsets_h
            self.load_x(n_idx, n_idx_loaded)
            # n_idx is set as 0 after load_x
            n_idx = 0
            self.compute_one_w(n_idx, kh_idx, offsets_w_start, sw_start)

    def compute_one_w(self, n_idx, kh_idx, offsets_w_start, out_gm_start):
        """
        Compute one w dim unit.
        """
        h_idx = offsets_w_start % self.dim_offsets_h
        self.alloc_all_ub()
        with self.tik_instance.if_scope(self.loop_seg != 0):
            with self.tik_instance.for_range(0, self.loop_seg) as ub_w_idx:
                offsets_f_start = offsets_w_start * self.dim_offsets_w + ub_w_idx * self.ub_seg_size
                helper_f_start = h_idx * self.dim_offsets_w + ub_w_idx * self.ub_seg_size
                self.compute_one_filter(n_idx, offsets_f_start, helper_f_start, kh_idx, self.ub_seg_size)
                out_gm_addr = (out_gm_start * self.dim_offsets_w + \
                               ub_w_idx * self.ub_seg_size) * self.dim_c * self.dim_kw
                self.move_out(out_gm_addr, self.ub_seg_size * self.dim_kw * self.dim_c)
        with self.tik_instance.if_scope(self.ub_seg_res != 0):
            offsets_f_start = offsets_w_start * self.dim_offsets_w + self.loop_seg * self.ub_seg_size
            helper_f_start = h_idx * self.dim_offsets_w + self.loop_seg * self.ub_seg_size
            self.compute_one_filter(n_idx, offsets_f_start, helper_f_start, kh_idx, self.ub_seg_res)
            out_gm_addr = (out_gm_start * self.dim_offsets_w + \
                           self.loop_seg * self.ub_seg_size) * self.dim_c * self.dim_kw
            self.move_out(out_gm_addr, self.ub_seg_res * self.dim_kw * self.dim_c)

    def alloc_all_ub(self):
        """
        Allocate some global ub tensors.
        """
        self.loop_seg = self.dim_offsets_w // self.ub_seg_size
        self.ub_seg_res = self.dim_offsets_w % self.ub_seg_size

        self.ub_offsets_ori_tmp = self.tik_instance.Tensor("float32", (self.ub_seg_size * self.elem_num_aligned,),
                                                       scope=tbe_platform.scope_ubuf, name="ub_offsets_ori_tmp")
        self.ub_out_tmp = self.tik_instance.Tensor("float32", (self.ub_seg_size * self.dim_kw * self.dim_c,),
                                               scope=tbe_platform.scope_ubuf, name="ub_out_tmp")
        self.ub_lrt_x_tmp = self.tik_instance.Tensor("float32", (2 * self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="ub_lrt_x_tmp")
        self.ub_lrb_x_tmp = self.tik_instance.Tensor("float32", (2 * self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="ub_lrb_x_tmp")
        self.ub_offsets_ori = self.tik_instance.Tensor(self.dtype, (self.ub_seg_size * self.elem_num_aligned,),
                                                            scope=tbe_platform.scope_ubuf,
                                                            name="ub_offsets_ori")
        self.ub_out = self.tik_instance.Tensor(self.dtype, (self.ub_seg_size * self.dim_kw * self.dim_c,),
                                                    scope=tbe_platform.scope_ubuf, name="ub_out")
        self.ub_lrt_x = self.tik_instance.Tensor(self.dtype, (2 * self.dim_c,),
                                                        scope=tbe_platform.scope_ubuf,
                                                        name="ub_lrt_x")
        self.ub_lrb_x = self.tik_instance.Tensor(self.dtype, (2 * self.dim_c,),
                                                        scope=tbe_platform.scope_ubuf,
                                                        name="ub_lrb_x")

    # 'pylint: disable=too-many-locals,too-many-statements,disable=too-many-arguments
    def compute_one_filter(self, n_idx, f_start, hlp_start, kh_idx, out_c_num):
        """
        Compute each w unit.
        """
        offsets_start = f_start * self.elem_num_offsets_filter
        self.load_offsets(offsets_start, out_c_num * self.elem_num_offsets_filter)
        with self.tik_instance.for_range(0, out_c_num * self.dim_kw * self.dim_group) as inner_index:
            cur_w_index = inner_index // (self.dim_kw * self.dim_group)
            kw_idx = inner_index % (self.dim_kw * self.dim_group) // self.dim_group
            group_idx = inner_index % (self.dim_kw * self.dim_group) % self.dim_group

            k_hw = self.dim_kh * self.dim_kw
            offsets_1wc_start = cur_w_index * self.elem_num_offsets_filter
            w_start = offsets_1wc_start + (0 * self.dim_group + group_idx) * k_hw + kh_idx * self.dim_kw + kw_idx
            h_start = offsets_1wc_start + (1 * self.dim_group + group_idx) * k_hw + kh_idx * self.dim_kw + kw_idx
            s_start = offsets_1wc_start + (2 * self.dim_group + group_idx) * k_hw + kh_idx * self.dim_kw + kw_idx

            offset_s = self.tik_instance.Scalar(
                dtype="float32", name="offset_s", init_value=self.ub_offsets_ori_tmp[s_start]
            )
            offset_h = self.tik_instance.Scalar(
                dtype="float32", name="offset_h", init_value=self.ub_offsets_ori_tmp[h_start]
            )
            offset_w = self.tik_instance.Scalar(
                dtype="float32", name="offset_w", init_value=self.ub_offsets_ori_tmp[w_start]
            )

            helper_1wc_start = hlp_start + cur_w_index
            helper_w = (helper_1wc_start % self.dim_offsets_w) * self.stride_w - self.pads_w + \
                        kw_idx * self.dilation_w
            helper_h = (helper_1wc_start // self.dim_offsets_w) * self.stride_h - self.pads_h + \
                        kh_idx * self.dilation_h
            offset_h.set_as(offset_h + helper_h)
            offset_w.set_as(offset_w + helper_w)

            low_h = self.tik_instance.Scalar(dtype="int32", name="low_h", init_value=0)
            low_w = self.tik_instance.Scalar(dtype="int32", name="low_w", init_value=0)
            high_h = self.tik_instance.Scalar(dtype="int32", name="high_h", init_value=0)
            high_w = self.tik_instance.Scalar(dtype="int32", name="high_w", init_value=0)
            ceil_sub_x = self.tik_instance.Scalar(dtype="float32", name="ceil_sub_x", init_value=0.0)
            ceil_sub_y = self.tik_instance.Scalar(dtype="float32", name="ceil_sub_y", init_value=0.0)
            sub_floor_x = self.tik_instance.Scalar(dtype="float32", name="sub_floor_x", init_value=0.0)
            sub_floor_y = self.tik_instance.Scalar(dtype="float32", name="sub_floor_y", init_value=0.0)

            with self.tik_instance.if_scope(tik.all(offset_h <= 2147483648, offset_h >= -2147483647)):
                self.tik_instance.scalar_conv("floor", low_h, offset_h)
                high_h.set_as(low_h + 1)
                ceil_sub_y.set_as(offset_h - low_h)
                sub_floor_y.set_as(high_h - offset_h)
    
            with self.tik_instance.if_scope(tik.all(offset_w <= 2147483648, offset_w >= -2147483647)):
                self.tik_instance.scalar_conv("floor", low_w, offset_w)
                high_w.set_as(low_w + 1)
                ceil_sub_x.set_as(offset_w - low_w)
                sub_floor_x.set_as(high_w - offset_w)

            self.get_input_x([low_h, low_w],
                             [n_idx * self.dim_h_in * self.dim_w_in * self.dim_c + \
                              low_h * self.dim_w_in * self.dim_c + low_w * self.dim_c + \
                              group_idx * self.dim_group_c,
                              n_idx * self.dim_h_in * self.dim_w_in * self.dim_c + \
                              low_h * self.dim_w_in * self.dim_c + high_w * self.dim_c + \
                              group_idx * self.dim_group_c,
                              n_idx * self.dim_h_in * self.dim_w_in * self.dim_c + \
                              high_h * self.dim_w_in * self.dim_c + low_w * self.dim_c + \
                              group_idx * self.dim_group_c,
                              n_idx * self.dim_h_in * self.dim_w_in * self.dim_c + \
                              high_h * self.dim_w_in * self.dim_c + high_w * self.dim_c + \
                              group_idx * self.dim_group_c],
                             [sub_floor_x * sub_floor_y * offset_s,
                              ceil_sub_x * sub_floor_y * offset_s,
                              ceil_sub_y * sub_floor_x * offset_s,
                              ceil_sub_y * ceil_sub_x * offset_s])

            out_ub_addr = cur_w_index * self.dim_kw * self.dim_c + kw_idx * self.dim_c + \
                          group_idx * self.dim_group_c
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.vector_binary_op(
                    self.dim_group_c,
                    [self.ub_out_tmp[out_ub_addr], self.ub_lrt_x_tmp, self.ub_lrt_x_tmp[self.dim_group_c]]
                )
                self.vector_binary_op(
                    self.dim_group_c, [self.ub_lrb_x_tmp, self.ub_lrb_x_tmp, self.ub_lrb_x_tmp[self.dim_group_c]]
                )
            if self.dtype != "float32":
                self.vector_binary_op(
                    self.dim_group_c, [self.ub_out_tmp[out_ub_addr], self.ub_out_tmp[out_ub_addr], self.ub_lrb_x_tmp]
                )
            else:
                self.vector_binary_op(
                    self.dim_group_c, [self.ub_out[out_ub_addr], self.ub_out_tmp[out_ub_addr], self.ub_lrb_x_tmp]
                )

    def get_input_x(self, pos_list, gm_offset_list, weight_list):
        self.get_input_x_top(pos_list, gm_offset_list[:2], weight_list[:2])
        self.get_input_x_bottom(pos_list, gm_offset_list[2:], weight_list[2:])

    def get_input_x_top(self, pos_list, gm_offset_list, weight_list):
        burst_len = self.dim_group_c // self.data_in_one_block
        low_h, low_w = pos_list
        gm_start_l_t, gm_start_r_t = gm_offset_list
        weight_l_t, weight_r_t = weight_list

        with self.tik_instance.if_scope(tik.all(low_h > -1, low_h < self.dim_h_in, \
                                                low_w > -1, low_w < self.dim_w_in - 1)):
            self.tik_instance.data_move(
                self.ub_lrt_x, self.x_scope[gm_start_l_t], constant.SID, constant.DEFAULT_NBURST, burst_len * 2,
                constant.STRIDE_ZERO, constant.STRIDE_ZERO
            )

        with self.tik_instance.elif_scope(tik.all(low_h > -1, low_h < self.dim_h_in, low_w == -1)):
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.tik_instance.data_move(
                    self.ub_lrt_x[self.dim_group_c], self.x_scope[gm_start_r_t], constant.SID, constant.DEFAULT_NBURST,
                    burst_len, constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
                self.vector_dup(self.ub_lrt_x, self.dim_group_c, 0, self.dtype)
        with self.tik_instance.elif_scope(tik.all(low_h > -1, low_h < self.dim_h_in, low_w == self.dim_w_in - 1)):
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.tik_instance.data_move(
                    self.ub_lrt_x, self.x_scope[gm_start_l_t], constant.SID, constant.DEFAULT_NBURST, burst_len,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
                self.vector_dup(self.ub_lrt_x[self.dim_group_c], self.dim_group_c, 0, self.dtype)
        with self.tik_instance.else_scope():
            self.vector_dup(self.ub_lrt_x, 2 * self.dim_group_c, 0, self.dtype)

        if self.dtype != "float32":
            self.vector_conv(self.ub_lrt_x_tmp, self.ub_lrt_x, 2 * self.dim_group_c)
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.vector_muls(self.ub_lrt_x_tmp, self.ub_lrt_x_tmp, weight_l_t, self.dim_group_c)
                self.vector_muls(
                    self.ub_lrt_x_tmp[self.dim_group_c], self.ub_lrt_x_tmp[self.dim_group_c], weight_r_t,
                    self.dim_group_c
                )
        else:
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.vector_muls(self.ub_lrt_x_tmp, self.ub_lrt_x, weight_l_t, self.dim_group_c)
                self.vector_muls(
                    self.ub_lrt_x_tmp[self.dim_group_c], self.ub_lrt_x[self.dim_group_c], weight_r_t, self.dim_group_c
                )

    def get_input_x_bottom(self, pos_list, gm_offset_list, weight_list):
        burst_len = self.dim_group_c // self.data_in_one_block
        low_h, low_w = pos_list
        gm_start_l_b, gm_start_r_b = gm_offset_list
        weight_l_b, weight_r_b = weight_list

        with self.tik_instance.if_scope(tik.all(low_h > -2, low_h < self.dim_h_in - 1, \
                                                low_w > -1, low_w < self.dim_w_in - 1)):
            self.tik_instance.data_move(
                self.ub_lrb_x, self.x_scope[gm_start_l_b], constant.SID, constant.DEFAULT_NBURST, burst_len * 2,
                constant.STRIDE_ZERO, constant.STRIDE_ZERO
            )
        with self.tik_instance.elif_scope(tik.all(low_h > -2, low_h < self.dim_h_in - 1, low_w == -1)):
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.tik_instance.data_move(
                    self.ub_lrb_x[self.dim_group_c], self.x_scope[gm_start_r_b], constant.SID, constant.DEFAULT_NBURST,
                    burst_len, constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
                self.vector_dup(self.ub_lrb_x, self.dim_group_c, 0, self.dtype)
        with self.tik_instance.elif_scope(tik.all(low_h > -2, low_h < self.dim_h_in - 1, low_w == self.dim_w_in - 1)):
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.tik_instance.data_move(
                    self.ub_lrb_x, self.x_scope[gm_start_l_b], constant.SID, constant.DEFAULT_NBURST, burst_len,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
                self.vector_dup(self.ub_lrb_x[self.dim_group_c], self.dim_group_c, 0, self.dtype)
        with self.tik_instance.else_scope():
            self.vector_dup(self.ub_lrb_x, 2 * self.dim_group_c, 0, self.dtype)

        if self.dtype != "float32":
            self.vector_conv(self.ub_lrb_x_tmp, self.ub_lrb_x, 2 * self.dim_group_c)
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.vector_muls(self.ub_lrb_x_tmp, self.ub_lrb_x_tmp, weight_l_b, self.dim_group_c)
                self.vector_muls(
                    self.ub_lrb_x_tmp[self.dim_group_c], self.ub_lrb_x_tmp[self.dim_group_c], weight_r_b,
                    self.dim_group_c
                )
        else:
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.vector_muls(self.ub_lrb_x_tmp, self.ub_lrb_x, weight_l_b, self.dim_group_c)
                self.vector_muls(
                    self.ub_lrb_x_tmp[self.dim_group_c], self.ub_lrb_x[self.dim_group_c], weight_r_b, self.dim_group_c
                )

    def load_x(self, n_idx, n_idx_loaded):
        """
        Load x from gm.
        -------
        Parameters:
        ----------
        n_idx: The index n of x to load.
        n_idx_loaded: The index n has be loaded.
        """
        with self.tik_instance.if_scope(n_idx != n_idx_loaded):
            n_idx_loaded = n_idx
            burst_len = self.dim_h_in * self.dim_w_in * self.dim_c // self.data_in_one_block
            self.tik_instance.data_move(self.ub_x[0], self.x_gm[n_idx * self.dim_h_in * self.dim_w_in * self.dim_c],
                                        constant.SID, 1, burst_len, constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def load_offsets(self, offsets_start, load_num):
        """
        Load offsets from gm tensor and prepare for ceil and floor offsets tensor.
        -------
        Parameters:
        ----------
        offsets_start: The index w of offsets to load.
        load_num: The length will be loaded.
        """
        align_num = _ceil_align(load_num, self.data_in_one_block)
        burst_len = align_num // self.data_in_one_block

        if self.dtype in ("float16", "bfloat16"):
            if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                self.tik_instance.data_move_pad(
                    self.ub_offsets_ori[0], self.offsets_gm[offsets_start], 1, load_num * self.dsize,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
            else:
                self.tik_instance.data_move(
                    self.ub_offsets_ori[0], self.offsets_gm[offsets_start], constant.SID, 1, burst_len,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
            # fp16 -> fp32
            self.vector_conv(self.ub_offsets_ori_tmp, self.ub_offsets_ori, align_num)
        else:
            if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                self.tik_instance.data_move_pad(
                    self.ub_offsets_ori, self.offsets_gm[offsets_start], 1, load_num * self.dsize, constant.STRIDE_ZERO,
                    constant.STRIDE_ZERO
                )
            else:
                self.tik_instance.data_move(
                    self.ub_offsets_ori, self.offsets_gm[offsets_start], constant.SID, 1, burst_len,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
            self.tik_instance.data_move(
                self.ub_offsets_ori_tmp, self.ub_offsets_ori, constant.SID, 1, burst_len, constant.STRIDE_ZERO,
                constant.STRIDE_ZERO
            )

    def move_out(self, out_gm_addr, load_num):
        bur_len = load_num // self.data_in_one_block
        if self.dtype in ("float16", "bfloat16"):
            # fp32 -> fp16
            self.vector_conv(self.ub_out, self.ub_out_tmp, load_num)
        self.tik_instance.data_move(self.y_gm[out_gm_addr],
                                    self.ub_out[0],
                                    constant.SID,
                                    constant.DEFAULT_NBURST,
                                    bur_len,
                                    constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)

    # 'pylint: disable=too-many-locals,disable=too-many-arguments
    def vector_binary_op(self, process_num, ub_list, op_name="add", dtype="float32", blk_stride_list=None,
                         rep_stride_list=None):
        """
        vadd, vsub func
        """
        func_map = {
            "add": self.tik_instance.vadd,
            "sub": self.tik_instance.vsub
        }

        dst_ub, src0_ub, src1_ub = ub_list
        if blk_stride_list is None:
            blk_stride_list = (1, 1, 1)
        if rep_stride_list is None:
            rep_stride_list = (8, 8, 8)
        dst_blk_stride, src0_blk_stride, src1_blk_stride = blk_stride_list
        dst_rep_stride, src0_rep_stride, src1_rep_stride = rep_stride_list
        d_size = get_dtype_size(dtype)
        one_cnt = Constant.VECTOR_BYTES_SIZE // d_size
        block_cnt = Constant.BLOCK_SIZE // d_size
        repeat = process_num // one_cnt
        remainder = process_num % one_cnt
        cycles_num = repeat // Constant.MAX_REPEAT
        cycles_tail = repeat % Constant.MAX_REPEAT

        with self.tik_instance.for_range(0, cycles_num) as cycles_index:
            func_map.get(op_name)(one_cnt,
                                  dst_ub[cycles_index * Constant.MAX_REPEAT * one_cnt],
                                  src0_ub[cycles_index * Constant.MAX_REPEAT * one_cnt],
                                  src1_ub[cycles_index * Constant.MAX_REPEAT * one_cnt],
                                  Constant.MAX_REPEAT,
                                  dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride, src1_rep_stride)

        with self.tik_instance.if_scope(cycles_tail > 0):
            func_map.get(op_name)(one_cnt,
                                  dst_ub[cycles_num * Constant.MAX_REPEAT * one_cnt],
                                  src0_ub[cycles_num * Constant.MAX_REPEAT * one_cnt],
                                  src1_ub[cycles_num * Constant.MAX_REPEAT * one_cnt],
                                  cycles_tail,
                                  dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride, src1_rep_stride)

        with self.tik_instance.if_scope(remainder > 0):
            func_map.get(op_name)(remainder,
                                  dst_ub[repeat * dst_rep_stride * block_cnt],
                                  src0_ub[repeat * src0_rep_stride * block_cnt],
                                  src1_ub[repeat * src1_rep_stride * block_cnt],
                                  1,
                                  dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride, src1_rep_stride)

    def vector_dup(self, ub_to_dup, process_num, const_val, dtype="float32"):
        """
        Duplicate data to a tensor.

        Parameters:
        ----------
        ub_to_dup: Dst tensor to store the result.
        process_num: The length to operate.
        const_val: A scalar value.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in ub_to_dup.
        """
        d_size = get_dtype_size(dtype)
        one_cnt = Constant.VECTOR_BYTES_SIZE // d_size
        repeat = process_num // one_cnt
        remainder = process_num % one_cnt
        cycles_num = repeat // Constant.MAX_REPEAT
        cycles_tail = repeat % Constant.MAX_REPEAT

        with self.tik_instance.for_range(0, cycles_num) as cycles_index:
            self.tik_instance.vec_dup(one_cnt,
                                      ub_to_dup[cycles_index * Constant.MAX_REPEAT * one_cnt],
                                      const_val,
                                      Constant.MAX_REPEAT, 8)

        with self.tik_instance.if_scope(cycles_tail > 0):
            self.tik_instance.vec_dup(one_cnt,
                                      ub_to_dup[cycles_num * Constant.MAX_REPEAT * one_cnt],
                                      const_val,
                                      cycles_tail, 8)

        with self.tik_instance.if_scope(remainder > 0):
            self.tik_instance.vec_dup(remainder, ub_to_dup[repeat * one_cnt], const_val, 1, 8)

    def vector_muls(self, dst, src, const, process_num, dtype="float32"):
        """
        Vector operator, Mul each elem in src tensor with a scalar value const.

        Parameters:
        ----------
        dst: Dst tensor to store the result.
        const: Scalar value.
        process_num: The length to operate.
        src: src tensor.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in dst.
        """
        d_size = get_dtype_size(dtype)
        one_cnt = Constant.VECTOR_BYTES_SIZE // d_size
        repeat = process_num // one_cnt
        remainder = process_num % one_cnt
        cycles_num = repeat // Constant.MAX_REPEAT
        cycles_tail = repeat % Constant.MAX_REPEAT

        with self.tik_instance.for_range(0, cycles_num) as cycles_index:
            self.tik_instance.vmuls(one_cnt,
                                    dst[cycles_index * Constant.MAX_REPEAT * one_cnt],
                                    src[cycles_index * Constant.MAX_REPEAT * one_cnt],
                                    const, Constant.MAX_REPEAT, 1, 1, 8, 8)

        with self.tik_instance.if_scope(cycles_tail > 0):
            self.tik_instance.vmuls(one_cnt,
                                    dst[cycles_num * Constant.MAX_REPEAT * one_cnt],
                                    src[cycles_num * Constant.MAX_REPEAT * one_cnt],
                                    const, cycles_tail, 1, 1, 8, 8)

        with self.tik_instance.if_scope(remainder > 0):
            self.tik_instance.vmuls(remainder,
                                    dst[repeat * one_cnt],
                                    src[repeat * one_cnt],
                                    const, 1, 1, 1, 8, 8)

    def vector_conv(self, dst, src, process_num):
        """
        vector_conv operator

        Parameters:
        ----------
        dst: Dst tensor to store the result.
        process_num: The length to operate.
        src: src tensor.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in dst.
        """
        src_dtype = src.dtype
        dst_dtype = dst.dtype
        if src_dtype in ("float32",) and dst_dtype in ("float16", "bfloat16"):
            dst_repeat_stride, src_repeat_stride = 4, 8
        else:
            dst_repeat_stride, src_repeat_stride = 8, 4
        repeat = process_num // Constant.MAX_MASK
        remainder = process_num % Constant.MAX_MASK
        cycles_num = repeat // Constant.MAX_REPEAT
        cycles_tail = repeat % Constant.MAX_REPEAT

        if dst_dtype == "bfloat16":
            with self.tik_instance.for_range(0, cycles_num) as cycles_index:
                self.tik_instance.vconv(Constant.MAX_MASK, "round",
                                        dst[cycles_index * Constant.MAX_REPEAT * Constant.MAX_MASK],
                                        src[cycles_index * Constant.MAX_REPEAT * Constant.MAX_MASK],
                                        Constant.MAX_REPEAT, 1, 1, dst_repeat_stride, src_repeat_stride)

            with self.tik_instance.if_scope(cycles_tail > 0):
                self.tik_instance.vconv(Constant.MAX_MASK, "round",
                                        dst[cycles_num * Constant.MAX_REPEAT * Constant.MAX_MASK],
                                        src[cycles_num * Constant.MAX_REPEAT * Constant.MAX_MASK], cycles_tail, 1, 1,
                                        dst_repeat_stride, src_repeat_stride)

            with self.tik_instance.if_scope(remainder > 0):
                self.tik_instance.vconv(remainder, "round", dst[repeat * Constant.MAX_MASK],
                                        src[repeat * Constant.MAX_MASK], 1, 1, 1,
                                        dst_repeat_stride, src_repeat_stride)
        else:
            with self.tik_instance.for_range(0, cycles_num) as cycles_index:
                self.tik_instance.vconv(Constant.MAX_MASK, "",
                                        dst[cycles_index * Constant.MAX_REPEAT * Constant.MAX_MASK],
                                        src[cycles_index * Constant.MAX_REPEAT * Constant.MAX_MASK],
                                        Constant.MAX_REPEAT, 1, 1, dst_repeat_stride, src_repeat_stride)

            with self.tik_instance.if_scope(cycles_tail > 0):
                self.tik_instance.vconv(Constant.MAX_MASK, "",
                                        dst[cycles_num * Constant.MAX_REPEAT * Constant.MAX_MASK],
                                        src[cycles_num * Constant.MAX_REPEAT * Constant.MAX_MASK], cycles_tail, 1, 1,
                                        dst_repeat_stride, src_repeat_stride)

            with self.tik_instance.if_scope(remainder > 0):
                self.tik_instance.vconv(remainder, "", dst[repeat * Constant.MAX_MASK],
                                        src[repeat * Constant.MAX_MASK], 1, 1, 1,
                                        dst_repeat_stride, src_repeat_stride)


@register_operator("DeformableOffsets")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def deformable_offsets(x,
                       offsets,
                       y,
                       strides,
                       pads,
                       ksize,
                       dilations=(1, 1, 1, 1),
                       data_format="NHWC",
                       deformable_groups=1,
                       modulated=True,
                       kernel_name="deformable_offsets"):
    """
    Computes the deformed convolution output with the expected input
    Parameters:
    ----------
    x: A Tensor of type float16,float32
    offsets: A Tensor of type float16,float32.Deformation offset parameter.
    Required Attributes:
    strides: A tuple/list of 4 integers.The stride of the sliding window for
             height and width for H/W dimension.
    pads: A tuple/list of 4 integers.Padding added to H/W dimension
          of the input.
    ksize: A tuple/list of 2 integers.kernel size.
    Attributes:
    dilations: A tuple/list of 4 integers, The dilation factor for each dimension
               of input.  Defaults to [1, 1, 1, 1]
    data_format: An optional string from: "NCHW", "NHWC". Defaults to "NCHW". Specify the data format of the input x.
    deformable_groups: Specify the c-axis grouping number of input x.
    modulated: Specify version of DeformableConv2D, true means v2, false means v1
    Outputs:
    y: A Tensor. A Tensor of type float16, float32.
    """
    dfm_inst = DeformableOffsets(x, offsets, y,
                                 strides, pads, ksize, dilations,
                                 data_format, deformable_groups,
                                 kernel_name)
    dfm_inst.deformable_offset_compute()
