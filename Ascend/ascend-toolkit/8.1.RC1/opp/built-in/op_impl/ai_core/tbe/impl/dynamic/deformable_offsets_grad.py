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
from impl import common_util
from impl import constant_util as constant
from impl.dynamic.deformable_offsets_grad_helper import DeformableOffsetsGradHelper
from impl.dynamic.deformable_offsets_grad_5hd import DeformableOffsetsGradV2
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_common import is_unknown
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_dynamic_input


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    MAX_INT64 = 2 ** 63 - 1
    TILING_ALIGN_SIZE = 160
    TILING_ARG_NUM = 17
    INT64 = "int64"
    BLOCK_BYTES_SIZE = 32
    VECTOR_BYTES_SIZE = 256
    BLOCK_SIZE = 32
    MAX_REPEAT = 255
    MAX_MASK = 64
    BURST_DOUBLE = 2
    TYPE_LEN_DICT = {"bfloat16": 2, "float16": 2, "float32": 4, "int64": 8, "int32": 4}
    # vector fp32 size
    VECTOR_FP32_SIZE = 64
    # block fp32 size
    BLOCK_FP32_SIZE = 8
    # block fp16 size
    BLOCK_FP16_SIZE = 16
    # 3 for offsets' w h scale
    W_H_SCALE = 3
    # MAX INIT GM is 256 * 1024
    MAX_INIT_GM = 262144


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


def is_unknown_attr(attr_list):
    if None in attr_list:
        return True
    return False


# 'pylint: disable=unused-argument,unused-variable
# 'pylint: disable=too-many-arguments,too-many-locals,too-many-return-statements
# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
def check_supported(grad,
                    x,
                    offsets,
                    grad_x,
                    grad_offsets,
                    strides,
                    pads,
                    ksize,
                    dilations=(1, 1, 1, 1),
                    data_format="NHWC",
                    deformable_groups=1,
                    modulated=True,
                    kernel_name="deformable_offsets_grad"):
    """
    check whether ai_core is supported
    """
    check_list = ("float32", "float16", "bfloat16")
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    x_format = x.get("format")
    if len(x_shape) == 5:
        return True
    if not modulated:
        reason = "modulated is False, not supported"
        return False, reason
    if x_dtype not in check_list:
        reason = "dtype is x is not supported, x_dtype is %s, supported list is %s" % (x_dtype, check_list)
        return False, reason
    if x_format != "NHWC":
        reason = "x_format is not NHWC, not supported"
        return False, reason

    # dynamic shape
    if is_unknown([grad, x, offsets]):
        return "Unknown"
    # static shape
    if len(x_shape) != 4:
        reason = "len of x_shape is not 4, x_shape is %s" % (str(x_shape),)
        return False, reason
    group_c = x_shape[3] // deformable_groups
    d_size = common_util.get_data_size(x_dtype)
    block_size = Constant.BLOCK_BYTES_SIZE // d_size

    if group_c % block_size != 0:
        reason = "group_c[%s] is not multiple of block_size[%s]" % (str(group_c), str(block_size))
        return False, reason

    dim_kh = ksize[0]
    dim_kw = ksize[1]
    total_ub = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    max_ub_elem = (total_ub - Constant.TILING_ALIGN_SIZE) // d_size
    elem_num_offsets_filter = deformable_groups * dim_kh * dim_kw * 3
    elem_num_aligned = _ceil_align(elem_num_offsets_filter, block_size)
    ub_global = 9 * group_c + Constant.W_H_SCALE * group_c
    ub_elem = 2 * elem_num_aligned
    if x_dtype in ("float16", "bfloat16"):
        ub_global = 12 * group_c + Constant.W_H_SCALE * group_c
        ub_elem = ub_elem + elem_num_aligned
    seg_size = (max_ub_elem - ub_global) // ub_elem
    if seg_size <= 0:
        reason = "size needed exceed ub_size"
        return False, reason

    return True, ""


# 'pylint: disable=too-many-instance-attributes,too-many-arguments
class DeformableOffsetsGrad:
    """
    initialize some properties
    """

    # 'pylint: disable=too-many-statements
    def __init__(self, grad, x, offsets, grad_x, grad_offsets, strides, pads, ksize, dilations,
                 data_format, deformable_groups, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.dtype = x.get("dtype").lower()
        self.tiling_dtype = Constant.INT64
        self.tiling_dsize = Constant.TYPE_LEN_DICT.get(self.tiling_dtype)
        self.is_unknown_rank = False
        if is_unknown_rank_input((grad, x, offsets)) or is_unknown_attr([strides, pads, ksize, dilations]):
            self.is_unknown_rank = True
            x["shape"] = (-1, -1, -1, -1)
            x["range"] = ((1, None), (1, None), (1, None), (1, None))
            offsets["shape"] = (-1, -1, -1, -1)
            offsets["range"] = ((1, None), (1, None), (1, None), (1, None))
            self.stride_h = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="stride_h")
            self.stride_w = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="stride_w")
            self.pads_h = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="pads_h")
            self.pads_w = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="pads_w")
            self.dim_kh = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_kh")
            self.dim_kw = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_kw")
            self.dilation_h = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dilation_h")
            self.dilation_w = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dilation_w")
            self.dim_group = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="dim_group")
            self.x_shape = x.get("shape")
            self.offsets_shape = offsets.get("shape")
        else:
            self.ksize = ksize
            self.strides = strides
            self.pads = pads
            self.x_shape = x.get("shape")
            self.offsets_shape = offsets.get("shape")
            self.check_param()
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
            self.dim_group = deformable_groups
        self.dsize = Constant.TYPE_LEN_DICT.get(self.dtype)
        self.kernel_name = kernel_name
        self.data_in_one_block = Constant.BLOCK_BYTES_SIZE // self.dsize
        self.elem_num_offsets_filter = self.dim_group * self.dim_kh * self.dim_kw * 3
        self.elem_num_aligned = _ceil_align(self.elem_num_offsets_filter, self.data_in_one_block)
        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.core_num = tik.Dprofile().get_aicore_num()

        self.grad_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT64, ], name="grad_gm",
                                             scope=tbe_platform.scope_gm)
        self.x_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT64, ], name="x_gm",
                                             scope=tbe_platform.scope_gm)
        self.offsets_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT64, ], name="offsets_gm",
                                                   scope=tbe_platform.scope_gm)
        self.grad_x_gm = None
        self.grad_offsets_gm = None

        self.grad_x_gm = self.tik_instance.Tensor("float32", [Constant.MAX_INT64, ], name="grad_x_gm",
                                            scope=tbe_platform.scope_gm, is_atomic_add=True)
        self.grad_offsets_gm = self.tik_instance.Tensor("float32", [Constant.MAX_INT64, ],
                                                        name="grad_offsets_gm",
                                                        scope=tbe_platform.scope_gm, is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)

        self.out_shape = None

        self.x_scope = None
        self.ub_x = None
        self.ub_out = None
        self.ub_offsets_ori = None
        self.ub_lt_x = None
        self.ub_lb_x = None
        self.ub_rt_x = None
        self.ub_rb_x = None
        self.weight_ub_lt = None
        self.weight_ub_lb = None
        self.weight_ub_rb = None
        self.weight_ub_rt = None
        self.grad_ub = None
        self.ub_out_offset = None

        self.ub_grad_fp16 = None
        self.ub_offsets_ori_fp16 = None
        self.ub_lt_x_fp16 = None
        self.ub_lb_x_fp16 = None
        self.ub_rt_x_fp16 = None
        self.ub_rb_x_fp16 = None
        self.ub_out_offset_fp16 = None

        self.tiling_mode = 0
        self.real_core_num = 0
        self.total_wc_num = 0
        self.dim_offsets_h = 0
        self.dim_offsets_w = 0
        self.dim_group_c = 0
        self.dim_c = 0
        self.dim_h_in = 0
        self.dim_w_in = 0
        self.ub_seg_size = 0
        self.loop_seg = 0
        self.ub_seg_res = 0

    def check_param(self):
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
        check_list = ("float32", "float16", "bfloat16")
        para_check.check_shape(self.x_shape, min_rank=4, max_rank=4, param_name="x")
        para_check.check_shape(self.offsets_shape, min_rank=4, max_rank=4, param_name="offsets")
        para_check.check_dtype(self.dtype, check_list, param_name="x")
        if len(self.ksize) != 2:
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, "the length of ksize should be 2",
                                                              "strides", self.ksize)
        if len(self.strides) != 4:
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, "the length of strides should be 4",
                                                              "strides", self.strides)
        if len(self.pads) != 4:
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, "the length of pads should be 4",
                                                              "pads", self.pads)

    def _get_tiling_args(self, tiling_ub):
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
        if self.is_unknown_rank:
            self.stride_h.set_as(tiling_ub[8])
            self.stride_w.set_as(tiling_ub[9])
            self.dilation_h.set_as(tiling_ub[10])
            self.dilation_w.set_as(tiling_ub[11])
            self.pads_h.set_as(tiling_ub[12])
            self.pads_w.set_as(tiling_ub[13])
            self.dim_kh.set_as(tiling_ub[14])
            self.dim_kw.set_as(tiling_ub[15])
            self.dim_group.set_as(tiling_ub[16])
        self.dim_group_c = self.dim_c // self.dim_group

    def alloc_all_ub(self):
        """
        Allocate some global ub tensors.
        """
        fp32_size = Constant.TYPE_LEN_DICT.get("float32")
        max_ub_elem = (self.total_ub - Constant.TILING_ALIGN_SIZE) // fp32_size // 2
        ub_global = 9 * self.dim_c + Constant.W_H_SCALE * self.dim_c
        ub_elem = 2 * self.elem_num_aligned
        # fp16 size is fp32 size // 2
        if self.dtype in ("float16", "bfloat16"):
            ub_global = 12 * self.dim_c + Constant.W_H_SCALE * self.dim_c
            ub_elem = ub_elem + self.elem_num_aligned
        seg_size = (max_ub_elem - ub_global) // ub_elem
        self.ub_seg_size = seg_size
        self.loop_seg = self.dim_offsets_w // self.ub_seg_size
        self.ub_seg_res = self.dim_offsets_w % self.ub_seg_size

        self.ub_offsets_ori = self.tik_instance.Tensor("float32", (seg_size * self.elem_num_aligned,),
                                                       scope=tbe_platform.scope_ubuf, name="ub_offsets_ori")
        self.ub_out = self.tik_instance.Tensor("float32", (Constant.W_H_SCALE * self.dim_c,),
                                               scope=tbe_platform.scope_ubuf, name="ub_out")
        self.ub_lt_x = self.tik_instance.Tensor("float32", (self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="ub_lt_x")
        self.ub_lb_x = self.tik_instance.Tensor("float32", (self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="ub_lb_x")
        self.ub_rt_x = self.tik_instance.Tensor("float32", (self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="ub_rt_x")
        self.ub_rb_x = self.tik_instance.Tensor("float32", (self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="ub_rb_x")

        self.weight_ub_lt = self.tik_instance.Tensor("float32", (self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="weight_ub_lt")
        self.weight_ub_lb = self.tik_instance.Tensor("float32", (self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="weight_ub_lb")
        self.weight_ub_rt = self.tik_instance.Tensor("float32", (self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="weight_ub_rt")
        self.weight_ub_rb = self.tik_instance.Tensor("float32", (self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="weight_ub_rb")

        self.grad_ub = self.tik_instance.Tensor("float32", (self.dim_c,), scope=tbe_platform.scope_ubuf,
                                                name="grad_ub")
        self.out_shape = _ceil_align(seg_size * self.elem_num_aligned, self.data_in_one_block)
        self.ub_out_offset = self.tik_instance.Tensor("float32", (self.out_shape,),
                                                      scope=tbe_platform.scope_ubuf, name="ub_out_offset")
        if self.dtype in ("float16", "bfloat16"):
            self.ub_offsets_ori_fp16 = self.tik_instance.Tensor(self.dtype, (seg_size * self.elem_num_aligned,),
                                                                scope=tbe_platform.scope_ubuf,
                                                                name="ub_offsets_ori_fp16")
            self.ub_grad_fp16 = self.tik_instance.Tensor(self.dtype, (self.dim_c,),
                                                        scope=tbe_platform.scope_ubuf, name="ub_grad_fp16")
            self.ub_lt_x_fp16 = self.tik_instance.Tensor(self.dtype, (self.dim_c,),
                                                         scope=tbe_platform.scope_ubuf,
                                                         name="ub_lt_x_fp16")
            self.ub_lb_x_fp16 = self.tik_instance.Tensor(self.dtype, (self.dim_c,),
                                                         scope=tbe_platform.scope_ubuf,
                                                         name="ub_lb_x_fp16")
            self.ub_rt_x_fp16 = self.tik_instance.Tensor(self.dtype, (self.dim_c,),
                                                         scope=tbe_platform.scope_ubuf,
                                                         name="ub_rt_x_fp16")
            self.ub_rb_x_fp16 = self.tik_instance.Tensor(self.dtype, (self.dim_c,),
                                                         scope=tbe_platform.scope_ubuf,
                                                         name="ub_rb_x_fp16")
            self.ub_out_offset_fp16 = self.tik_instance.Tensor(self.dtype, (self.out_shape,),
                                                        scope=tbe_platform.scope_ubuf, name="ub_out_offset_fp16")

    def deformable_offset_grad_compute(self):
        """
        The main comute func.
        """
        tik_instance = self.tik_instance
        with tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            # get tiling data
            tiling_ub = tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                            name="tiling_ub", scope=tik.scope_ubuf)

            tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 
                                   Constant.TILING_ALIGN_SIZE // Constant.BLOCK_BYTES_SIZE, 0, 0)
            # get run info
            self._get_tiling_args(tiling_ub)
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
                with self.tik_instance.for_range(0, each_cycle, thread_num = 2) as wc_index:
                    sw_start = block_offset + wc_index
                    offsets_w_start = sw_start // self.dim_kh
                    kh_idx = sw_start % self.dim_kh

                    self._compute_mode_1(kh_idx, offsets_w_start, sw_start)

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }
        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                                            "ub_size": self.total_ub,
                                                            "dsize": self.dsize,
                                                            })

        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.grad_gm, self.x_gm, self.offsets_gm],
                              outputs=[self.grad_x_gm, self.grad_offsets_gm],
                              flowtable=(self.tiling_gm,),
                              enable_l2=True, config=opt_config)

    def _compute_mode_1(self, kh_idx, offsets_w_start, out_gm_start):
        """
        Compute mode 1.
        """
        h_idx = offsets_w_start % self.dim_offsets_h
        n_idx = offsets_w_start // self.dim_offsets_h

        self.alloc_all_ub()
        with self.tik_instance.if_scope(self.loop_seg != 0):
            with self.tik_instance.for_range(0, self.loop_seg) as ub_w_idx:
                offsets_f_start = offsets_w_start * self.dim_offsets_w + ub_w_idx * self.ub_seg_size
                helper_f_start = h_idx * self.dim_offsets_w + ub_w_idx * self.ub_seg_size
                grad_f_start = (out_gm_start * self.dim_offsets_w + \
                                ub_w_idx * self.ub_seg_size) * self.dim_kw * self.dim_group * self.dim_group_c
                self.compute_one_filter(n_idx, [offsets_f_start, helper_f_start, grad_f_start],
                                        kh_idx, self.ub_seg_size)
                out_gm_addr = (offsets_w_start * self.dim_offsets_w + \
                                ub_w_idx * self.ub_seg_size) * self.dim_kh * self.dim_kw * 3 * self.dim_group
                self.move_out(out_gm_addr, self.ub_seg_size * self.dim_kh * self.dim_kw * 3 * self.dim_group,
                              self.ub_out_offset)
        with self.tik_instance.if_scope(self.ub_seg_res != 0):
            offsets_f_start = offsets_w_start * self.dim_offsets_w + self.loop_seg * self.ub_seg_size
            helper_f_start = h_idx * self.dim_offsets_w + self.loop_seg * self.ub_seg_size
            grad_f_start = (out_gm_start * self.dim_offsets_w + \
                            self.loop_seg * self.ub_seg_size) * self.dim_kw * self.dim_group * self.dim_group_c
            self.compute_one_filter(n_idx, [offsets_f_start, helper_f_start, grad_f_start], kh_idx, self.ub_seg_res)
            out_gm_addr = (offsets_w_start * self.dim_offsets_w + \
                            self.loop_seg * self.ub_seg_size) * self.dim_kh * self.dim_kw * 3 * self.dim_group
            self.move_out(out_gm_addr, self.ub_seg_res * self.dim_kh * self.dim_kw * 3 * self.dim_group,
                          self.ub_out_offset)

    # 'pylint: disable=too-many-locals,too-many-statements
    def compute_one_filter(self, n_idx, start_list, kh_idx, out_c_num):
        """
        Compute each w unit.
        """
        f_start, hlp_start, grad_f_start = start_list
        dx_start = n_idx * self.dim_h_in * self.dim_w_in * self.dim_group * self.dim_group_c

        offsets_start = f_start * self.elem_num_offsets_filter
        self.load_offsets(offsets_start, out_c_num * self.elem_num_offsets_filter)

        # clear ub_out_offset
        self.vector_dup(self.ub_out_offset, self.out_shape, 0.0)

        with self.tik_instance.for_range(0, out_c_num * self.dim_kw * self.dim_group) as inner_index:
            cur_w_index = inner_index // (self.dim_kw * self.dim_group)
            kw_idx = inner_index % (self.dim_kw * self.dim_group) // self.dim_group
            group_idx = inner_index % (self.dim_kw * self.dim_group) % self.dim_group

            k_hw = self.dim_kh * self.dim_kw
            offsets_1wc_start = cur_w_index * self.elem_num_offsets_filter
            w_start = offsets_1wc_start + (
                    0 * self.dim_group + group_idx) * k_hw + kh_idx * self.dim_kw + kw_idx
            h_start = offsets_1wc_start + (
                    1 * self.dim_group + group_idx) * k_hw + kh_idx * self.dim_kw + kw_idx
            s_start = offsets_1wc_start + (
                    2 * self.dim_group + group_idx) * k_hw + kh_idx * self.dim_kw + kw_idx

            offset_s = self.tik_instance.Scalar(dtype="float32", name="offset_s",
                                                init_value=self.ub_offsets_ori[s_start])
            offset_h = self.tik_instance.Scalar(dtype="float32", name="offset_h",
                                                init_value=self.ub_offsets_ori[h_start])
            offset_w = self.tik_instance.Scalar(dtype="float32", name="offset_w",
                                                init_value=self.ub_offsets_ori[w_start])

            helper_1wc_start = hlp_start + cur_w_index
            helper_w = (helper_1wc_start % self.dim_offsets_w) * self.stride_w - self.pads_w + \
                        kw_idx * self.dilation_w
            helper_h = (helper_1wc_start // self.dim_offsets_w) * self.stride_h - self.pads_h + \
                        kh_idx * self.dilation_h
            offset_h.set_as(offset_h + helper_h)
            offset_w.set_as(offset_w + helper_w)

            low_h = self.tik_instance.Scalar(dtype="int32", name="low_h")
            low_w = self.tik_instance.Scalar(dtype="int32", name="low_w")
            self.tik_instance.scalar_conv("floor", low_h, offset_h)
            self.tik_instance.scalar_conv("floor", low_w, offset_w)
            high_h = low_h + 1
            high_w = low_w + 1

            ceil_sub_x = offset_w - low_w
            ceil_sub_y = offset_h - low_h
            sub_floor_x = high_w - offset_w
            sub_floor_y = high_h - offset_h

            dx_lt_index = dx_start + \
                        low_h * self.dim_w_in * self.dim_group * self.dim_group_c + \
                        low_w * self.dim_group * self.dim_group_c + \
                        group_idx * self.dim_group_c
            dx_lb_index = dx_start + \
                        high_h * self.dim_w_in * self.dim_group * self.dim_group_c + \
                        low_w * self.dim_group * self.dim_group_c + \
                        group_idx * self.dim_group_c
            dx_rt_index = dx_start + \
                        low_h * self.dim_w_in * self.dim_group * self.dim_group_c + \
                        high_w * self.dim_group * self.dim_group_c + \
                        group_idx * self.dim_group_c
            dx_rb_index = dx_start + \
                        high_h * self.dim_w_in * self.dim_group * self.dim_group_c + \
                        high_w * self.dim_group * self.dim_group_c + \
                        group_idx * self.dim_group_c

            ub_out_w_index = 0
            ub_out_h_index = self.dim_group_c
            ub_out_s_index = 2 * self.dim_group_c

            grad_ub_index = grad_f_start + \
                            cur_w_index * self.dim_kw * self.dim_group * self.dim_group_c + \
                            kw_idx * self.dim_group * self.dim_group_c + \
                            group_idx * self.dim_group_c

            self.load_grad(self.dim_group_c, grad_ub_index)

            # 'grad_scale_lb = grad * scale * l_b_w * l_b_h
            scale_lb = offset_s * sub_floor_x * ceil_sub_y
            self.vector_muls(self.weight_ub_lb, self.grad_ub, scale_lb, self.dim_group_c, "float32")
            # 'l_b -> high_h < h_in && low_w >= 0
            self.input_x_out_weight(tik.all(offset_h > -1, offset_h < self.dim_h_in,
                                    offset_w > -1, offset_w < self.dim_w_in,
                                    high_h < self.dim_h_in, low_w >= 0),
                            [self.ub_lb_x, self.ub_lb_x_fp16, self.weight_ub_lb],
                            dx_lb_index)

            # 'grad_scale_lt = grad * scale * l_t_w * l_t_h
            scale_lt = offset_s * sub_floor_x * sub_floor_y
            self.vector_muls(self.weight_ub_lt, self.grad_ub, scale_lt, self.dim_group_c, "float32")
            # 'l_t -> low_h >= 0 && low_w >= 0
            self.input_x_out_weight(tik.all(offset_h > -1, offset_h < self.dim_h_in,
                                    offset_w > -1, offset_w < self.dim_w_in,
                                    low_h >= 0, low_w >= 0),
                            [self.ub_lt_x, self.ub_lt_x_fp16, self.weight_ub_lt],
                            dx_lt_index)

            # 'grad_scale_rt = grad * scale * r_t_w * r_t_h
            scale_rt = offset_s * ceil_sub_x * sub_floor_y
            self.vector_muls(self.weight_ub_rt, self.grad_ub, scale_rt, self.dim_group_c, "float32")
            # 'r_t -> low_h >= 0 && high_w < w_in
            self.input_x_out_weight(tik.all(offset_h > -1, offset_h < self.dim_h_in,
                                    offset_w > -1, offset_w < self.dim_w_in,
                                    low_h >= 0, high_w < self.dim_w_in),
                            [self.ub_rt_x, self.ub_rt_x_fp16, self.weight_ub_rt],
                            dx_rt_index)

            # 'grad_scale_rb = grad * scale * r_b_w * r_b_h
            scale_rb = offset_s * ceil_sub_x * ceil_sub_y
            self.vector_muls(self.weight_ub_rb, self.grad_ub, scale_rb, self.dim_group_c, "float32")
            # 'r_b -> high_h < h_in && high_w < w_in
            self.input_x_out_weight(tik.all(offset_h > -1, offset_h < self.dim_h_in,
                                    offset_w > -1, offset_w < self.dim_w_in,
                                    high_h < self.dim_h_in, high_w < self.dim_w_in),
                            [self.ub_rb_x, self.ub_rb_x_fp16, self.weight_ub_rb],
                            dx_rb_index)

            self.get_offset_w(self.ub_out, ub_out_w_index,
                            [ceil_sub_x, ceil_sub_y, sub_floor_x, sub_floor_y, offset_s])
            self.get_offset_h(self.ub_out, ub_out_h_index,
                            [ceil_sub_x, ceil_sub_y, sub_floor_x, sub_floor_y, offset_s])
            self.get_offset_s(self.ub_out, ub_out_s_index,
                            [ceil_sub_x, ceil_sub_y, sub_floor_x, sub_floor_y, offset_s])

            self.reduce_sum(self.dim_group_c, self.ub_out[:ub_out_h_index])

            self.ub_out_offset[w_start].set_as(self.ub_out[0])

            self.reduce_sum(self.dim_group_c, self.ub_out[ub_out_h_index:ub_out_s_index])

            self.ub_out_offset[h_start].set_as(self.ub_out[self.dim_group_c])

            self.reduce_sum(self.dim_group_c, self.ub_out[ub_out_s_index:])

            self.ub_out_offset[s_start].set_as(self.ub_out[2 * self.dim_group_c])


    def input_x_out_weight(self, flag, ub_tensor, gm_start):
        x_ub, fp16_tensor, weight_ub = ub_tensor
        group_c_burst = self.dim_group_c // self.data_in_one_block
        group_c_burst_pad = self.dim_group_c * self.dsize
        with self.tik_instance.if_scope(flag):
            if self.dtype in ("float16", "bfloat16"):
                if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                    self.tik_instance.data_move_pad(fp16_tensor, self.x_gm[gm_start], 1, group_c_burst_pad, 0, 0)
                    self.vector_conv(x_ub, fp16_tensor, self.dim_group_c)
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move_pad(self.grad_x_gm[gm_start], weight_ub, 1,
                                                    group_c_burst_pad * Constant.BURST_DOUBLE, 0, 0)
                    self.tik_instance.set_atomic_add(0)
                else:
                    self.tik_instance.data_move(fp16_tensor, self.x_gm[gm_start], 0, 1, group_c_burst, 0, 0)
                    # fp16 -> fp32
                    self.vector_conv(x_ub, fp16_tensor, self.dim_group_c)

                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(self.grad_x_gm[gm_start], weight_ub, 0, 1,
                                                group_c_burst * Constant.BURST_DOUBLE, 0, 0)
                    self.tik_instance.set_atomic_add(0)
            else:
                if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                    self.tik_instance.data_move_pad(x_ub, self.x_gm[gm_start], 1, group_c_burst_pad, 0, 0)
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move_pad(self.grad_x_gm[gm_start], weight_ub, 1, group_c_burst_pad, 0, 0)
                    self.tik_instance.set_atomic_add(0)
                else:
                    self.tik_instance.data_move(x_ub, self.x_gm[gm_start], 0, 1, group_c_burst, 0, 0)
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(self.grad_x_gm[gm_start], weight_ub, 0, 1, group_c_burst, 0, 0)
                    self.tik_instance.set_atomic_add(0)

        with self.tik_instance.else_scope():
            self.vector_dup(x_ub, self.dim_group_c, 0)

    def load_grad(self, size, grad_start):
        grad_burst_len = size // self.data_in_one_block
        grad_burst_len_pad = size * self.dsize
        if self.dtype in ("float16", "bfloat16"):
            if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                self.tik_instance.data_move_pad(self.ub_grad_fp16, self.grad_gm[grad_start], 
                                                1, grad_burst_len_pad, 0, 0)
            else:
                self.tik_instance.data_move(self.ub_grad_fp16, self.grad_gm[grad_start], 0, 1, grad_burst_len, 0, 0)

            self.vector_conv(self.grad_ub, self.ub_grad_fp16, size)
        else:
            if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                self.tik_instance.data_move_pad(self.grad_ub, self.grad_gm[grad_start], 1, grad_burst_len_pad, 0, 0)
            else:
                self.tik_instance.data_move(self.grad_ub, self.grad_gm[grad_start], 0, 1, grad_burst_len, 0, 0)

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
        burst_len_pad = load_num * self.dsize
        if self.dtype in ("float16", "bfloat16"):
            if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                self.tik_instance.data_move_pad(self.ub_offsets_ori_fp16[0], self.offsets_gm[offsets_start],
                                                1, burst_len_pad, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            else:
                self.tik_instance.data_move(self.ub_offsets_ori_fp16[0], self.offsets_gm[offsets_start],
                                            constant.SID, 1, burst_len, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            # fp16 -> fp32
            self.vector_conv(self.ub_offsets_ori, self.ub_offsets_ori_fp16, load_num)
        else:
            if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                self.tik_instance.data_move_pad(self.ub_offsets_ori[0], self.offsets_gm[offsets_start],
                                                1, burst_len_pad, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            else:
                self.tik_instance.data_move(self.ub_offsets_ori[0], self.offsets_gm[offsets_start],
                                            constant.SID, 1, burst_len, constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def get_offset_w(self, dst_tensor, dst_index, val_list):
        ceil_sub_x, ceil_sub_y, sub_floor_x, sub_floor_y, offset_s = val_list
        self.vector_muls(self.weight_ub_lt, self.ub_lt_x, sub_floor_y * offset_s, self.dim_group_c, "float32")
        self.vector_muls(self.weight_ub_lb, self.ub_lb_x, ceil_sub_y * offset_s, self.dim_group_c, "float32")
        self.vector_muls(self.weight_ub_rt, self.ub_rt_x, sub_floor_y * offset_s, self.dim_group_c, "float32")
        self.vector_muls(self.weight_ub_rb, self.ub_rb_x, ceil_sub_y * offset_s, self.dim_group_c, "float32")
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, self.weight_ub_rt, self.weight_ub_rb], "add", "float32", [dst_index, 0, 0])
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, dst_tensor, self.weight_ub_lb], "sub", "float32", [dst_index, dst_index, 0])
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, dst_tensor, self.weight_ub_lt], "sub", "float32", [dst_index, dst_index, 0])
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, dst_tensor, self.grad_ub], "mul", "float32", [dst_index, dst_index, 0])

    def get_offset_h(self, dst_tensor, dst_index, val_list):
        ceil_sub_x, ceil_sub_y, sub_floor_x, sub_floor_y, offset_s = val_list
        self.vector_muls(self.weight_ub_lt, self.ub_lt_x, sub_floor_x * offset_s, self.dim_group_c, "float32")
        self.vector_muls(self.weight_ub_lb, self.ub_lb_x, sub_floor_x * offset_s, self.dim_group_c, "float32")
        self.vector_muls(self.weight_ub_rt, self.ub_rt_x, ceil_sub_x * offset_s, self.dim_group_c, "float32")
        self.vector_muls(self.weight_ub_rb, self.ub_rb_x, ceil_sub_x * offset_s, self.dim_group_c, "float32")
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, self.weight_ub_lb, self.weight_ub_rb], "add", "float32", [dst_index, 0, 0])
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, dst_tensor, self.weight_ub_rt], "sub", "float32", [dst_index, dst_index, 0])
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, dst_tensor, self.weight_ub_lt], "sub", "float32", [dst_index, dst_index, 0])
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, dst_tensor, self.grad_ub], "mul", "float32", [dst_index, dst_index, 0])

    def get_offset_s(self, dst_tensor, dst_index, val_list):
        ceil_sub_x, ceil_sub_y, sub_floor_x, sub_floor_y, offset_s = val_list
        self.vector_muls(self.weight_ub_lt, self.ub_lt_x, sub_floor_x * sub_floor_y, self.dim_group_c, "float32")
        self.vector_muls(self.weight_ub_lb, self.ub_lb_x, sub_floor_x * ceil_sub_y, self.dim_group_c, "float32")
        self.vector_muls(self.weight_ub_rt, self.ub_rt_x, ceil_sub_x * sub_floor_y, self.dim_group_c, "float32")
        self.vector_muls(self.weight_ub_rb, self.ub_rb_x, ceil_sub_x * ceil_sub_y, self.dim_group_c, "float32")
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, self.weight_ub_lb, self.weight_ub_rb], "add", "float32", [dst_index, 0, 0])
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, dst_tensor, self.weight_ub_rt], "add", "float32", [dst_index, dst_index, 0])
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, dst_tensor, self.weight_ub_lt], "add", "float32", [dst_index, dst_index, 0])
        self.vector_binary_op(self.dim_group_c,
                              [dst_tensor, dst_tensor, self.grad_ub], "mul", "float32", [dst_index, dst_index, 0])

    def move_out(self, out_index, size, out_ub):
        burst_len = (size + self.data_in_one_block - 1) // self.data_in_one_block
        burst_len_pad = size * self.dsize
        self.tik_instance.set_atomic_add(1)
        if self.dtype in ("float16", "bfloat16"):
            if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                self.tik_instance.data_move_pad(self.grad_offsets_gm[out_index],
                                            out_ub, 1, burst_len_pad * Constant.BURST_DOUBLE, 0, 0)
            else:
                self.tik_instance.data_move(self.grad_offsets_gm[out_index],
                                            out_ub, 0, 1, burst_len * Constant.BURST_DOUBLE, 0, 0)
        else:
            if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
                self.tik_instance.data_move_pad(self.grad_offsets_gm[out_index], out_ub, 1, burst_len_pad, 0, 0)
            else:
                self.tik_instance.data_move(self.grad_offsets_gm[out_index], out_ub, 0, 1, burst_len, 0, 0)
        self.tik_instance.set_atomic_add(0)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
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
        d_size = common_util.get_data_size(dtype)
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

    # 'pylint: disable=too-many-locals
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def vector_binary_op(self, process_num, ub_list, op_name="add", dtype="float32",
                         start_list = None,
                         blk_stride_list=None,
                         rep_stride_list=None):
        """
        vadd, vsub func
        """
        func_map = {
            "add": self.tik_instance.vadd,
            "sub": self.tik_instance.vsub,
            "mul": self.tik_instance.vmul,
            "min": self.tik_instance.vmin,
            "max": self.tik_instance.vmax
        }

        dst_ub, src0_ub, src1_ub = ub_list
        if start_list is None:
            start_list = (0, 0, 0)
        if blk_stride_list is None:
            blk_stride_list = (1, 1, 1)
        if rep_stride_list is None:
            rep_stride_list = (8, 8, 8)
        dst_start, src0_start, src1_start = start_list
        dst_blk_stride, src0_blk_stride, src1_blk_stride = blk_stride_list
        dst_rep_stride, src0_rep_stride, src1_rep_stride = rep_stride_list

        d_size = common_util.get_data_size(dtype)
        one_cnt = Constant.VECTOR_BYTES_SIZE // d_size
        block_cnt = Constant.BLOCK_SIZE // d_size

        repeat = process_num // one_cnt
        remainder = process_num % one_cnt
        cycles_num = repeat // Constant.MAX_REPEAT
        cycles_tail = repeat % Constant.MAX_REPEAT

        with self.tik_instance.if_scope(cycles_num > 0):
            with self.tik_instance.for_range(0, cycles_num) as cycles_index:
                func_map.get(op_name)(one_cnt,
                                    dst_ub[dst_start + cycles_index * Constant.MAX_REPEAT * one_cnt],
                                    src0_ub[src0_start + cycles_index * Constant.MAX_REPEAT * one_cnt],
                                    src1_ub[src1_start + cycles_index * Constant.MAX_REPEAT * one_cnt],
                                    Constant.MAX_REPEAT,
                                    dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                    dst_rep_stride, src0_rep_stride, src1_rep_stride)

        with self.tik_instance.if_scope(cycles_tail > 0):
            func_map.get(op_name)(one_cnt,
                                  dst_ub[dst_start + cycles_num * Constant.MAX_REPEAT * one_cnt],
                                  src0_ub[src0_start + cycles_num * Constant.MAX_REPEAT * one_cnt],
                                  src1_ub[src1_start + cycles_num * Constant.MAX_REPEAT * one_cnt],
                                  cycles_tail,
                                  dst_blk_stride, src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride, src1_rep_stride)

        with self.tik_instance.if_scope(remainder > 0):
            func_map.get(op_name)(remainder,
                                  dst_ub[dst_start + repeat * dst_rep_stride * block_cnt],
                                  src0_ub[src0_start + repeat * src0_rep_stride * block_cnt],
                                  src1_ub[src1_start + repeat * src1_rep_stride * block_cnt],
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
        d_size = common_util.get_data_size(dtype)
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

    def reduce_sum(self, size, out_ub):
        """
        summation of group_c axis, [...,kh,kw,group_c] -> [...,kh,kw]
        """
        with self.tik_instance.if_scope(self.dim_group_c > 64):
            group_c_num = size // self.dim_group_c
            loop_remainder = group_c_num % Constant.MAX_REPEAT

            if loop_remainder > 0:
                self.vcadd_stride_group_c(loop_remainder, out_ub, [0, 0])

            max_repeat = Constant.MAX_REPEAT // Constant.BLOCK_FP32_SIZE * Constant.BLOCK_FP32_SIZE
            loop_rem = group_c_num % max_repeat

            if loop_rem > 0:
                self.tik_instance.vcadd(Constant.VECTOR_FP32_SIZE,
                                        out_ub,
                                        out_ub,
                                        loop_rem,
                                        1, 1, 64 // Constant.BLOCK_FP32_SIZE)
        with self.tik_instance.else_scope():
            self.tik_instance.vcadd(self.dim_group_c,
                                    out_ub,
                                    out_ub,
                                    1,
                                    1, 1, self.dim_group_c // Constant.BLOCK_FP32_SIZE)

    def vcadd_stride_group_c(self, repeat, out_ub, start_list):
        """
        using repeat to sum group_c
        """
        dst_start, src_start = start_list
        one_cnt = Constant.VECTOR_FP32_SIZE
        num = self.dim_group_c // Constant.VECTOR_FP32_SIZE
        tail = self.dim_group_c % Constant.VECTOR_FP32_SIZE
        with self.tik_instance.if_scope(tail > 0):
            self.tik_instance.vadd(tail,
                                   out_ub[dst_start],
                                   out_ub[src_start],
                                   out_ub[src_start + num * one_cnt],
                                   repeat,
                                   1, 1, 1, self.dim_group_c // 8, self.dim_group_c // 8, self.dim_group_c // 8)
        with self.tik_instance.if_scope(num > 1):
            with self.tik_instance.for_range(1, num) as l_i:
                self.tik_instance.vadd(one_cnt,
                                      out_ub[dst_start],
                                      out_ub[src_start],
                                      out_ub[src_start + l_i * one_cnt],
                                      repeat,
                                      1, 1, 1, self.dim_group_c // 8, self.dim_group_c // 8, self.dim_group_c // 8)
        self.tik_instance.vadds(one_cnt,
                                out_ub[dst_start],
                                out_ub[src_start],
                                0,
                                repeat,
                                1, 1, 8, self.dim_group_c // 8)

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

        with self.tik_instance.for_range(0, cycles_num) as cycles_index:
            self.tik_instance.vconv(Constant.MAX_MASK, "",
                                    dst[cycles_index * Constant.MAX_REPEAT * Constant.MAX_MASK],
                                    src[cycles_index * Constant.MAX_REPEAT * Constant.MAX_MASK],
                                    Constant.MAX_REPEAT, 1, 1, dst_repeat_stride, src_repeat_stride)

        with self.tik_instance.if_scope(cycles_tail > 0):
            self.tik_instance.vconv(Constant.MAX_MASK, "", dst[cycles_num * Constant.MAX_REPEAT * Constant.MAX_MASK],
                                    src[cycles_num * Constant.MAX_REPEAT * Constant.MAX_MASK], cycles_tail, 1, 1,
                                    dst_repeat_stride, src_repeat_stride)

        with self.tik_instance.if_scope(remainder > 0):
            self.tik_instance.vconv(remainder, "", dst[repeat * Constant.MAX_MASK],
                                    src[repeat * Constant.MAX_MASK], 1, 1, 1,
                                    dst_repeat_stride, src_repeat_stride)


@register_operator("DeformableOffsetsGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
# 'pylint: disable=unused-argument,unused-variable
# 'pylint: disable=too-many-arguments,too-many-locals,too-many-return-statements
# 'pylint: disable=unused-argument,too-many-arguments
def deformable_offsets_grad(grad,
                       x,
                       offsets,
                       grad_x,
                       grad_offsets,
                       strides,
                       pads,
                       ksize,
                       dilations,
                       data_format="NHWC",
                       deformable_groups=1,
                       modulated=True,
                       kernel_name="deformable_offsets_grad"):
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
    x_shape = x.get("shape")
    if len(x_shape) == 5:
        dfm_inst = DeformableOffsetsGradV2(grad, x, offsets, grad_x, grad_offsets,
                                            strides, pads, ksize, dilations,
                                            data_format, deformable_groups,
                                            kernel_name)
        dfm_inst.deformable_offset_grad_v2_compute()
    else:
        if is_unknown_rank_input((grad, x, offsets)) or is_unknown_attr([strides, pads, ksize, dilations]):
            x["shape"] = (-1, -1, -1, -1)
            x["range"] = ((1, None), (1, None), (1, None), (1, None))
            grad["shape"] = (-1, -1, -1, -1)
            grad["range"] = ((1, None), (1, None), (1, None), (1, None))
        x_shape = x.get("shape")
        x_dtype = x.get("dtype").lower()
        grad_shape = grad.get("shape")
        helper_size = 0
        group_c = 0
        if not is_unknown((grad, x, offsets)):
            k_h, k_w = ksize
            w_out = grad_shape[2] // k_w
            helper_size = w_out * Constant.W_H_SCALE * deformable_groups * k_h * k_w
            group_c = x_shape[3]
        lesshelper = False
        block_size = Constant.BLOCK_FP16_SIZE if x_dtype in ("float16", "bfloat16") else Constant.BLOCK_FP32_SIZE
        if group_c <= 64 and deformable_groups == 1 and helper_size % block_size == 0:
            lesshelper = True
        if not is_unknown((grad, x, offsets)) and grad_shape[1] * \
            grad_shape[2] * deformable_groups * Constant.W_H_SCALE <= Constant.MAX_INIT_GM and lesshelper:

            x_shape = x.get("shape")
            x_dtype = x.get("dtype").lower()
            grad_shape = grad.get("shape")
            offsets_shape = offsets.get("shape")

            if data_format == "NHWC":
                _, stride_h, stride_w, _ = strides
                _, dilation_h, dilation_w, _ = dilations
            else:
                _, _, stride_h, stride_w = strides
                _, _, dilation_h, dilation_w = dilations

            k_h, k_w = ksize
            group_c = x_shape[3] // deformable_groups
            h_out, w_out = grad_shape[1] // k_h, grad_shape[2] // k_w

            x_shape_new = [x_shape[0], x_shape[1], x_shape[2], deformable_groups, group_c]
            grad_shape_new = [grad_shape[0], h_out, k_h, w_out, k_w, deformable_groups, group_c]
            offsets_shape_new = [offsets_shape[0], h_out, w_out, 3, deformable_groups, k_h, k_w]

            group_c_only = group_c > 64

            input_params = {
                "dtype": x_dtype,
                "grad_shape": grad_shape_new,
                "x_shape": x_shape_new,
                "offsets_shape": offsets_shape_new,
                "grad_x_shape": grad_x.get("shape"),
                "grad_offsets_shape": grad_offsets.get("shape"),
                "batch": x_shape[0],
                "h_in": x_shape[1],
                "w_in": x_shape[2],
                "group_c": group_c,
                "h_out": h_out,
                "w_out": w_out,
                "stride_h": stride_h,
                "stride_w": stride_w,
                "k_h": k_h,
                "k_w": k_w,
                "dilation_h": dilation_h,
                "dilation_w": dilation_w,
                "pads": pads,
                "deformable_groups": deformable_groups,
                "group_c_only": group_c_only,
                "kernel_name": kernel_name
            }
            grad = DeformableOffsetsGradHelper(input_params)
            grad.offsets_grad()
            grad.instance.BuildCCE(kernel_name=kernel_name,
                                inputs=[grad.grad_gm,
                                        grad.x_gm,
                                        grad.offsets_gm],
                                outputs=[grad.grad_x_gm,
                                            grad.grad_offsets_gm])

        else:
            dfm_inst = DeformableOffsetsGrad(grad, x, offsets, grad_x, grad_offsets,
                                            strides, pads, ksize, dilations,
                                            data_format, deformable_groups,
                                            kernel_name)
            dfm_inst.deformable_offset_grad_compute()
