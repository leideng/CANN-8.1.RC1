#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
iou
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from tbe.common.platform.platform_info import get_soc_spec
from impl.dynamic.common_iou import CommonIoU


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # MAX ELIMENT NUM OF FP16 IN 1BLOCK
    FP16_ELIMENTS_BLOCK = 16
    # MAX ELIMENT NUM OF FP32 IN 1BLOCK
    FP32_ELIMENTS_BLOCK = 8
    # CONST GTBOX SLICE SEGMENT
    GTBOX_SEGMENT = 4096 * 4
    # CONST BBOX SLICE SEGMENT
    BBOX_SEGMENT = 4096 * 4

    MAX_INT32 = 2 ** 31 - 1
    TILING_SCALAR_DTYPE = "int64"
    TILING_PARAMS_NUM = 12

    AREA_UB_SIZE = 4096

    MASK_BLOCK_32 = 64
    MASK_BLOCK_16 = 128
    BLOCK_32 = 8
    BLOCK_16 = 16
    BYTE_PER_DATA_32 = 4
    BYTE_PER_DATA_16 = 2
    UB_NUM = 40


def _apply_mem(tik_instance, dtype,
               shape, name, scope=tik.scope_ubuf):
    """apply mem fuc

    Parameters
    ----------
    tik_instance: tik_instance
        tik_instance
    dtype: str
        ub dtype
    shape: list
        ub shape
    name: str
        ub name
    scope: scope
        scope_ubuf or scope_gm
    Returns
    -------
    Tensor: Tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    """
    Get Ceil Int

    Parameters
    ----------
    int1: int
        input int 1
    int2: int
        input int 2

    Returns
    -------
    ceil_int: int
    """

    return (int1 + int2 - 1) // int2


def _get_align_int(int1, int2):
    return _get_ceil_int(int1, int2) * int2


# 'pylint: disable=too-many-instance-attributes,invalid-name
class Iou:
    """Function: use to finish Iou main functions
    """

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, bboxes, gtboxes, mode, eps, aligned):
        """
        init Iou parameters

        Parameters
        ----------
        bboxes : dict
            data of bboxes.
            source data type, support "float16" and "float32"
        gtboxes : dict
            data of gtboxes.
            source data type, support "float16" and "float32"
        overlap : dict
            shape and dtype of overlap
            result shape is [m, n]
        mode :  str
            ('iou','iof')
            iou : the output is inter_area / total_area
            iof : the output is inter_area / gtboxes_area
        eps : float
            prevent division by 0
            default value is 1.0
        aligned : bool
            (False, True)
            False : the output shape is [m, n]
            True : the output shape is [n, 1], m equals to n

        Returns
        -------
        None
        """
        self.bboxes_shape = bboxes.get("shape")
        self.bboxes_dtype = bboxes.get("dtype").lower()
        self.gtboxes_shape = gtboxes.get("shape")
        self.gtboxes_dtype = gtboxes.get("dtype").lower()
        self.dtype = self.bboxes_dtype
        self.eps = eps
        self.aligned = aligned
        self.mode = mode.lower()
        self.tik_instance = tik.Tik()
        self.full_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.available_ub_size = get_soc_spec("UB_SIZE")
        self.product = tbe_platform.api_check_support("tik.vdiv", "float32")
        # input and output tensor in gm
        self.bboxes_gm = self.tik_instance.Tensor(
            self.bboxes_dtype,
            (Constant.MAX_INT32,),
            name="bboxes_gm",
            scope=tik.scope_gm)
        self.gtboxes_gm = self.tik_instance.Tensor(
            self.gtboxes_dtype,
            (Constant.MAX_INT32,),
            name="gtboxes_gm",
            scope=tik.scope_gm)
        self.overlap_gm = self.tik_instance.Tensor(
            self.bboxes_dtype,
            (Constant.MAX_INT32,),
            name="overlap_gm",
            scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            name="tiling_gm",
            scope=tik.scope_gm)

        self.bboxes_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "bboxes_num")
        self.gtboxes_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "gtboxes_num")

        self.core_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "core_num")
        self.tiling_core_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tiling_core_num",
                                                        init_value=self.full_core_num)

        # init attr in objext
        self.tiling_mode = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tiling_mode")
        self.point_per_core = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "point_per_core")
        self.core_tail_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "core_tail_num")
        self.area_x0_size = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "area_x0_size")
        self.area_ub_size = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "area_ub_size")
        self.gt_area_ub_size = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "gt_area_ub_size")
        self.bb_loop = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "bb_loop")
        self.bb_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "bb_tail")
        self.bb_tail_offset = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "bb_tail_offset")
        self.dst_gm_offset = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dst_gm_offset")
        self.gm_point_offset = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "gm_point_offset")
        self.get_tiling_args()
        
        self.area_x0 = None
        self.area_x1 = None
        self.area_y0 = None
        self.area_y1 = None
        self.inter_area_x0 = None
        self.inter_area_x1 = None
        self.inter_area_y0 = None
        self.inter_area_y1 = None
        self.area_y1_y0 = None
        self.gtboxes_ub = None
        self.gt_boxes_area_ub = None
        self.bboxes_ub = None
        self.bboxes_area_ub = None
        self.out_ub = None
        self.inter_area_ub = None
        self.zero_ub = None
        self.gtboxes_x0 = None
        self.gtboxes_x1 = None
        self.gtboxes_y0 = None
        self.gtboxes_y1 = None
        self.rec_1 = None
        self.rec_2 = None
        if self.bboxes_dtype == "float16":
            self.gt_ub_segment = Constant.GTBOX_SEGMENT
            self.bb_ub_segment = Constant.BBOX_SEGMENT
            self.max_eliments = Constant.FP16_ELIMENTS_BLOCK * 8
            self.min_point_per_core = Constant.FP16_ELIMENTS_BLOCK
            self.eliments_per_block = Constant.FP16_ELIMENTS_BLOCK
            self.AREA_UB_SIZE = 4096
        else:
            self.gt_ub_segment = Constant.GTBOX_SEGMENT // 4
            self.bb_ub_segment = Constant.BBOX_SEGMENT // 4
            self.max_eliments = Constant.FP32_ELIMENTS_BLOCK * 8
            self.min_point_per_core = Constant.FP32_ELIMENTS_BLOCK
            self.eliments_per_block = Constant.FP32_ELIMENTS_BLOCK
            self.AREA_UB_SIZE = 2048
        if self.product is False:
            self.bb_ub_segment = self.bb_ub_segment // 2

    def get_tiling_args(self):
        """get runtime tiling data and set for scalar

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        tiling_ub = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            tik.scope_ubuf,
            "tiling_ub"
        )
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 3, 0, 0)
        self.tiling_mode.set_as(tiling_ub[0])
        self.point_per_core.set_as(tiling_ub[1])
        self.core_tail_num.set_as(tiling_ub[2])
        self.core_num.set_as(tiling_ub[3])
        self.area_x0_size.set_as(tiling_ub[4])
        self.bboxes_num.set_as(tiling_ub[5])
        self.gtboxes_num.set_as(tiling_ub[6])
        self.bb_loop.set_as(tiling_ub[7])
        self.bb_tail.set_as(tiling_ub[8])
        self.bb_tail_offset.set_as(tiling_ub[9])
        self.tiling_core_num.set_as(tiling_ub[10])

    # 'pylint: disable=too-many-statements
    def iou_process(self, core_id):
        """
        do process and schedule
        main function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(core_id < self.core_num):
            # calcu gt area

            run_gt_point = self.gtboxes_num
            run_gt_point_segment = run_gt_point * 4
            # global
            nbust = _get_ceil_int(run_gt_point_segment, self.eliments_per_block)
            self.tik_instance.data_move(self.gtboxes_ub, self.gtboxes_gm, 0, 1, nbust, 0, 0)
            # [n,4] --> 4*[n,1]  by scalar
            self.data_rerange(run_gt_point, self.gtboxes_ub)
            # calcu area
            self.calcu_area(run_gt_point, self.gt_boxes_area_ub)

            # one time output bb_ub_segment_point values
            with  self.tik_instance.if_scope(self.core_tail_num != 0):
                with self.tik_instance.if_scope(core_id == (self.core_num - 1)):
                    self.dst_gm_offset.set_as(self.point_per_core * core_id - self.point_per_core + self.core_tail_num)
                    with self.tik_instance.if_scope(self.core_num == 1):
                        self.dst_gm_offset.set_as(0)
                    with self.tik_instance.for_range(0, self.bb_loop, thread_num=2) as bb_loop_index:
                        self.gm_point_offset.set_as((bb_loop_index * self.bb_ub_segment) // 4 + self.dst_gm_offset)
                        self._run_segment(self.bb_ub_segment, self.gm_point_offset, self.gtboxes_num)
                    with self.tik_instance.if_scope(self.bb_tail != 0):
                        self.gm_point_offset.set_as(self.bb_tail_offset // 4 + self.dst_gm_offset)
                        self._run_segment(self.bb_tail, self.gm_point_offset, self.gtboxes_num)
                with self.tik_instance.else_scope():
                    self.dst_gm_offset.set_as(self.point_per_core * core_id)
                    with self.tik_instance.for_range(0, self.bb_loop, thread_num=2) as bb_loop_index:
                        self.gm_point_offset.set_as((bb_loop_index * self.bb_ub_segment) // 4 + self.dst_gm_offset)
                        self._run_segment(self.bb_ub_segment, self.gm_point_offset, self.gtboxes_num)
                    with self.tik_instance.if_scope(self.bb_tail != 0):
                        self.gm_point_offset.set_as(self.bb_tail_offset // 4 + self.dst_gm_offset)
                        self._run_segment(self.bb_tail, self.gm_point_offset, self.gtboxes_num)
            with self.tik_instance.else_scope():
                self.dst_gm_offset.set_as(self.point_per_core * core_id)
                with self.tik_instance.for_range(0, self.bb_loop, thread_num=2) as bb_loop_index:
                    self.gm_point_offset.set_as((bb_loop_index * self.bb_ub_segment) // 4 + self.dst_gm_offset)
                    self._run_segment(self.bb_ub_segment, self.gm_point_offset, self.gtboxes_num)
                with self.tik_instance.if_scope(self.bb_tail != 0):
                    self.gm_point_offset.set_as(self.bb_tail_offset // 4 + self.dst_gm_offset)
                    self._run_segment(self.bb_tail, self.gm_point_offset, self.gtboxes_num)

    def iou_process_cut_by_gt(self, core_id):
        """
        do process and schedule by gt
        main function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(core_id < self.core_num):
            def _run(gt_len):
                gt_loop = (gt_len) // self.gt_ub_segment
                gt_tail = (gt_len) % self.gt_ub_segment
                with self.tik_instance.for_range(0, gt_loop) as _gt_loop:
                    self.dst_gm_offset.set_as(\
                        (self.point_per_core * core_id + _gt_loop * self.gt_ub_segment // 4) * self.bboxes_num)

                    # global
                    nbust = _get_ceil_int(self.gt_ub_segment, self.eliments_per_block)
                    gt_gm_offset = core_id * self.point_per_core * 4 + _gt_loop * self.gt_ub_segment
                    self.tik_instance.data_move(self.gtboxes_ub, self.gtboxes_gm[gt_gm_offset], 0, 1, nbust, 0, 0)
                    # [n,4] --> 4*[n,1]  by scalar
                    self.data_rerange(self.gt_ub_segment // 4, self.gtboxes_ub)
                    # calcu area
                    self.calcu_area(self.gt_ub_segment // 4, self.gt_boxes_area_ub)
                    gtbox_num = self.gt_ub_segment // 4
                    with self.tik_instance.for_range(0, self.bb_loop, thread_num=2) as bb_loop_index:
                        self.gm_point_offset.set_as((bb_loop_index * self.bb_ub_segment) // 4)
                        self._run_segment(self.bb_ub_segment, self.gm_point_offset, gtbox_num, self.dst_gm_offset)

                    with self.tik_instance.if_scope(self.bb_tail != 0):
                        self.gm_point_offset.set_as(self.bb_tail_offset // 4)

                        with self.tik_instance.if_scope(
                                tik.any((self.bb_tail // 4) % self.eliments_per_block == 0, self.core_num == 1)):
                            self._run_segment(self.bb_tail, self.gm_point_offset, self.dst_gm_offset)

                        with self.tik_instance.else_scope():
                            bb_tail_half = _get_align_int(self.bb_tail // 8, self.eliments_per_block)
                            self._run_segment(bb_tail_half * 4, self.gm_point_offset, self.dst_gm_offset)
                            self.gm_point_offset.set_as(self.gm_point_offset + bb_tail_half - \
                                            (bb_tail_half * 2 - self.bb_tail // 4))
                            self._run_segment(bb_tail_half * 4, self.gm_point_offset, self.dst_gm_offset)


                with self.tik_instance.if_scope(gt_tail != 0):
                    self.dst_gm_offset.set_as((self.point_per_core * core_id +
                                               gt_loop * self.gt_ub_segment // 4) * self.bboxes_num)

                    # global
                    nbust = _get_ceil_int(gt_tail, self.eliments_per_block)
                    gt_gm_offset = core_id * self.point_per_core * 4 + gt_loop * self.gt_ub_segment
                    self.tik_instance.data_move(self.gtboxes_ub, self.gtboxes_gm[gt_gm_offset], 0, 1, nbust, 0, 0)
                    # [n,4] --> 4*[n,1]  by scalar
                    self.data_rerange(gt_tail // 4, self.gtboxes_ub)
                    # calcu area
                    self.calcu_area(gt_tail // 4, self.gt_boxes_area_ub)
                    gtbox_num = gt_tail // 4
                    with self.tik_instance.for_range(0, self.bb_loop, thread_num=2) as bb_loop_index:
                        self.gm_point_offset.set_as((bb_loop_index * self.bb_ub_segment) // 4)
                        self._run_segment(self.bb_ub_segment, self.gm_point_offset, gtbox_num, self.dst_gm_offset)

                    with self.tik_instance.if_scope(self.bb_tail != 0):
                        self.gm_point_offset.set_as(self.bb_tail_offset // 4)

                        with self.tik_instance.if_scope(
                                tik.any((self.bb_tail // 4) % self.eliments_per_block == 0, self.core_num == 1)):
                            self._run_segment(self.bb_tail, self.gm_point_offset, gtbox_num, self.dst_gm_offset)

                        with self.tik_instance.else_scope():
                            bb_tail_half = _get_align_int(self.bb_tail // 8, self.eliments_per_block)
                            self._run_segment(bb_tail_half * 4, self.gm_point_offset, gtbox_num, self.dst_gm_offset)
                            self.gm_point_offset.set_as(self.gm_point_offset + bb_tail_half - \
                                                        (bb_tail_half * 2 - self.bb_tail // 4))
                            self._run_segment(bb_tail_half * 4, self.gm_point_offset, gtbox_num, self.dst_gm_offset)

            with self.tik_instance.if_scope(self.core_tail_num == 0):
                _run(self.point_per_core * 4)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(core_id == (self.core_num - 1)):
                    _run(self.core_tail_num * 4)
                with self.tik_instance.else_scope():
                    _run(self.point_per_core * 4)

    def run_tik(self, kernel_name):
        """
        run_tik start tik process, and buid cce

        Parameters
        ----------
        kernel_name : str
            bbox segment len

        Returns
        -------
        result: tik_instance
            tik_instance
        """

        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_id:
            self._apply_all_ub()
            with self.tik_instance.if_scope(self.tiling_mode == 1):
                self.iou_process(core_id)

            with self.tik_instance.else_scope():
                self.iou_process_cut_by_gt(core_id)

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "full_core_num": self.full_core_num,
                "ub_size": self.available_ub_size,
                "product": self.product
                })
        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.bboxes_gm, self.gtboxes_gm],
            outputs=[self.overlap_gm],
            flowtable=[self.tiling_gm],
            config=opt_config
        )
        return self.tik_instance

    def data_rerange(self, run_point, point_ub):
        """
        data rerange

        Parameters
        ----------
        run_point : int
            data range len
        point_ub : ub tensor
            UB addr

        Returns
        -------
        None
        """
        for_range = _get_ceil_int(run_point, 2)
        index_reg = self._get_scalar_list(8)
        with self.tik_instance.for_range(0, for_range) as conv_index:
            for i in range(8):
                index_reg[i].set_as(point_ub[conv_index * 8 + i])
            for i in range(2):
                self.area_x0[conv_index * 2 + i].set_as(index_reg[i * 4 + 0])
                self.area_y0[conv_index * 2 + i].set_as(index_reg[i * 4 + 1])
                self.area_x1[conv_index * 2 + i].set_as(index_reg[i * 4 + 2])
                self.area_y1[conv_index * 2 + i].set_as(index_reg[i * 4 + 3])

    def calcu_area(self, run_point, area_ub, inter_mode=False):
        """
        calcu area

        Parameters
        ----------
        run_point : int
            data range len
        area_ub : ub tensor
            UB addr
        inter_mode: bool
            calcu mode

        Returns
        -------
        None
        """
        if inter_mode:
            x0_ub = self.inter_area_x0
            x1_ub = self.inter_area_x1
            y0_ub = self.inter_area_y0
            y1_ub = self.inter_area_y1
        else:
            x0_ub = self.area_x0
            x1_ub = self.area_x1
            y0_ub = self.area_y0
            y1_ub = self.area_y1
        repeat_time = _get_ceil_int(run_point, self.max_eliments)
        # cala x1 - x0

        self.tik_instance.vsub(self.max_eliments, area_ub,
                               x1_ub,
                               x0_ub, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vsub(self.max_eliments, self.area_y1_y0,
                               y1_ub,
                               y0_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        if self.eps < 1.0:
            if inter_mode is False:
                self.tik_instance.vadds(self.max_eliments, area_ub,
                                        area_ub, self.eps, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vadds(self.max_eliments, self.area_y1_y0,
                                        self.area_y1_y0, self.eps, repeat_time, 1, 1, 8, 8)
        else:
            self.tik_instance.vadds(self.max_eliments, area_ub,
                                    area_ub, 1, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vadds(self.max_eliments, self.area_y1_y0,
                                    self.area_y1_y0, 1, repeat_time, 1, 1, 8, 8)
        # vmuls 0.2 to evade fp16 overflows
        self.tik_instance.vmuls(self.max_eliments, area_ub,
                                area_ub, 0.2, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmuls(self.max_eliments, self.area_y1_y0,
                                self.area_y1_y0, 0.2, repeat_time, 1, 1, 8, 8)
        if inter_mode:
            self.tik_instance.vmax(self.max_eliments, area_ub,
                                   self.zero_ub, area_ub,
                                   repeat_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vmax(self.max_eliments, self.area_y1_y0,
                                   self.zero_ub, self.area_y1_y0,
                                   repeat_time, 1, 0, 1, 8, 0, 8)
        self.tik_instance.vmul(self.max_eliments, area_ub,
                               self.area_y1_y0,
                               area_ub, repeat_time, 1, 1, 1, 8, 8, 8)

    def _apply_all_ub(self):
        """
        apply_all_ub
        """
        self.area_x0 = _apply_mem(self.tik_instance, self.dtype,
                                  [self.AREA_UB_SIZE], "area_x0")
        self.area_x1 = _apply_mem(self.tik_instance, self.dtype,
                                  [self.AREA_UB_SIZE], "area_x1")
        self.area_y0 = _apply_mem(self.tik_instance, self.dtype,
                                  [self.AREA_UB_SIZE], "area_y0")
        self.area_y1 = _apply_mem(self.tik_instance, self.dtype,
                                  [self.AREA_UB_SIZE], "area_y1")
        self.inter_area_x0 = _apply_mem(self.tik_instance, self.dtype,
                                        [self.AREA_UB_SIZE], "inter_area_x0")
        self.inter_area_x1 = _apply_mem(self.tik_instance, self.dtype,
                                        [self.AREA_UB_SIZE], "inter_area_x1")
        self.inter_area_y0 = _apply_mem(self.tik_instance, self.dtype,
                                        [self.AREA_UB_SIZE], "inter_area_y0")
        self.inter_area_y1 = _apply_mem(self.tik_instance, self.dtype,
                                        [self.AREA_UB_SIZE], "inter_area_y1")
        self.area_y1_y0 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.AREA_UB_SIZE], "area_y1_y0")
        if self.bboxes_dtype == "float16":
            self.gtboxes_ub = _apply_mem(self.tik_instance, self.dtype,
                                         [self.gt_ub_segment], "gtboxes_ub")
        else:
            self.gtboxes_ub = _apply_mem(self.tik_instance, self.dtype,
                                         [self.gt_ub_segment * 2], "gtboxes_ub")
        self.gt_boxes_area_ub = _apply_mem(self.tik_instance, self.dtype,
                                           [self.AREA_UB_SIZE], "gt_boxes_area_ub")
        self.zero_ub = _apply_mem(self.tik_instance, self.dtype,
                                  [self.eliments_per_block], "zero_ub")
        self.out_ub = _apply_mem(self.tik_instance, self.dtype,
                                 [self.AREA_UB_SIZE], "out_ub")
        self.inter_area_ub = _apply_mem(self.tik_instance, self.dtype,
                                        [self.AREA_UB_SIZE], "inter_area_ub")
        self.bboxes_area_ub = _apply_mem(self.tik_instance, self.dtype,
                                         [self.AREA_UB_SIZE], "bboxes_area_ub")
        if self.bboxes_dtype == "float16":
            self.bboxes_ub = _apply_mem(self.tik_instance, self.dtype,
                                        [self.bb_ub_segment], "bboxes_ub")
        else:
            self.bboxes_ub = _apply_mem(self.tik_instance, self.dtype,
                                        [self.bb_ub_segment * 2], "bboxes_ub")
        self.gtboxes_x0 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.eliments_per_block], "gtboxes_x0")
        self.gtboxes_x1 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.eliments_per_block], "gtboxes_x1")
        self.gtboxes_y0 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.eliments_per_block], "gtboxes_y0")
        self.gtboxes_y1 = _apply_mem(self.tik_instance, self.dtype,
                                     [self.eliments_per_block], "gtboxes_y1")
        if not self.product:
            self.rec_1 = _apply_mem(self.tik_instance, self.dtype,
                                    [self.AREA_UB_SIZE], "rec_1")
            self.rec_2 = _apply_mem(self.tik_instance, self.dtype,
                                    [self.AREA_UB_SIZE], "rec_2")
        _repeat = _get_ceil_int(self.AREA_UB_SIZE, self.max_eliments)
        self.tik_instance.vector_dup(self.max_eliments,
                                     self.area_x0, 0.0,
                                     _repeat, 1, 8)
        self.tik_instance.vector_dup(self.max_eliments,
                                     self.area_x1, 0.0,
                                     _repeat, 1, 8)
        self.tik_instance.vector_dup(self.max_eliments,
                                     self.area_y0, 0.0,
                                     _repeat, 1, 8)
        self.tik_instance.vector_dup(self.max_eliments,
                                     self.area_y1, 0.0,
                                     _repeat, 1, 8)
        self.tik_instance.vector_dup(self.eliments_per_block,
                                     self.zero_ub, 0.0,
                                     1, 1, 8)

    def _get_scalar_list(self, count):
        result = []
        for _ in range(count):
            result.append(self.tik_instance.Scalar(dtype=self.dtype))
        return result

    def _run_segment(self, run_bb_point_segment, gm_offset, gtbox_num, gm_out_offset=0):
        """
        do a segment of bbox compute

        Parameters
        ----------
        run_bb_point_segment : int
            bbox segment len
        gm_offset : int
            gm offset

        Returns
        -------
        None
        """
        run_bb_point = run_bb_point_segment // 4
        src_gm_offset = gm_offset * 4
        # copy gm to ub
        nbust = _get_ceil_int(run_bb_point_segment, self.eliments_per_block)
        self.tik_instance.data_move(
            self.bboxes_ub, self.bboxes_gm[src_gm_offset], 0, 1, nbust, 0, 0)

        # [n,4] --> 4*[n,1]  by scalar
        self.data_rerange(run_bb_point, self.bboxes_ub)
        # calcu area
        self.calcu_area(run_bb_point, self.bboxes_area_ub)

        scalar_addr = self._get_scalar_list(4)
        scalar_area = self.tik_instance.Scalar(dtype=self.dtype)
        with self.tik_instance.for_range(
                0, gtbox_num) as gt_global_index:
            scalar_area.set_as(self.gt_boxes_area_ub[gt_global_index])
            for i in range(4):
                scalar_addr[i].set_as(self.gtboxes_ub[gt_global_index * 4 + i])
            # `scalar_area = (scalar_addr[2]-scalar_addr[0]) * (scalar_addr[3]-scalar_addr[1])`
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.gtboxes_x0, scalar_addr[0],
                                         1, 1, 8)
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.gtboxes_y0, scalar_addr[1],
                                         1, 1, 8)
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.gtboxes_x1, scalar_addr[2],
                                         1, 1, 8)
            self.tik_instance.vector_dup(self.eliments_per_block,
                                         self.gtboxes_y1, scalar_addr[3],
                                         1, 1, 8)
            # vmin vmax
            repeat_time = _get_ceil_int(run_bb_point, self.max_eliments)
            self.tik_instance.vmax(self.max_eliments,
                                   self.inter_area_x0,
                                   self.area_x0,
                                   self.gtboxes_x0,
                                   repeat_time, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmax(self.max_eliments,
                                   self.inter_area_y0,
                                   self.area_y0,
                                   self.gtboxes_y0,
                                   repeat_time, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmin(self.max_eliments,
                                   self.inter_area_x1,
                                   self.area_x1,
                                   self.gtboxes_x1,
                                   repeat_time, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmin(self.max_eliments,
                                   self.inter_area_y1,
                                   self.area_y1,
                                   self.gtboxes_y1,
                                   repeat_time, 1, 1, 0, 8, 8, 0)
            self.calcu_area(run_bb_point, self.inter_area_ub, True)
            if self.mode == "iou":
                self.tik_instance.vsub(self.max_eliments, self.out_ub,
                                       self.bboxes_area_ub,
                                       self.inter_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vadds(self.max_eliments, self.out_ub, 
                                        self.out_ub,
                                        scalar_area, repeat_time, 1, 1, 8, 8)
            elif self.mode == "iof":
                self.tik_instance.vector_dup(self.max_eliments, self.out_ub,
                                             scalar_area, repeat_time, 1, 8)

            if self.product:
                self.tik_instance.vdiv(
                    self.max_eliments, self.out_ub, self.inter_area_ub,
                    self.out_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            else:
                # for mini
                self.tik_instance.vrec(self.max_eliments, self.rec_1,
                                       self.out_ub,
                                       repeat_time, 1, 1, 8, 8)
                self.tik_instance.vmul(self.max_eliments, self.rec_2,
                                       self.rec_1,
                                       self.out_ub, repeat_time, 1,
                                       1, 1, 8, 8, 8)
                self.tik_instance.vmuls(self.max_eliments, self.rec_2,
                                        self.rec_2,
                                        -1, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vadds(self.max_eliments, self.rec_2,
                                        self.rec_2,
                                        2, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vmul(self.max_eliments, self.rec_2,
                                       self.rec_2,
                                       self.rec_1, repeat_time, 1,
                                       1, 1, 8, 8, 8)
                self.tik_instance.vmul(self.max_eliments, self.rec_1,
                                       self.rec_2,
                                       self.out_ub, repeat_time, 1,
                                       1, 1, 8, 8, 8)
                self.tik_instance.vmuls(self.max_eliments, self.rec_1,
                                        self.rec_1,
                                        -1, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vadds(self.max_eliments, self.rec_1,
                                        self.rec_1,
                                        2, repeat_time, 1, 1, 8, 8)
                self.tik_instance.vmul(self.max_eliments, self.rec_1,
                                       self.rec_1,
                                       self.rec_2, repeat_time, 1,
                                       1, 1, 8, 8, 8)

                self.tik_instance.vmul(self.max_eliments, self.out_ub,
                                       self.rec_1,
                                       self.inter_area_ub, repeat_time, 1,
                                       1, 1, 8, 8, 8)

            iou_gm_offset = gt_global_index * self.bboxes_num + gm_offset + gm_out_offset
            nbust = _get_ceil_int(run_bb_point, self.eliments_per_block)
            self.tik_instance.data_move(self.overlap_gm[iou_gm_offset],
                                        self.out_ub, 0, 1, nbust, 0, 0)

    
class AlignedIoU(CommonIoU):
    """Function: use to finish Iou main functions
    """

    # pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, bboxes, gtboxes, trans, is_cross, mode, eps):
        super().__init__(bboxes, gtboxes, trans, is_cross, mode)
        self.eps = eps

    def calcu_area(self, area_ub, inter_mode=False, gt_mode=False):
        if inter_mode:
            x0_ub = self.inter_area_x0
            x1_ub = self.inter_area_x1
            y0_ub = self.inter_area_y0
            y1_ub = self.inter_area_y1
        elif gt_mode:
            x0_ub = self.gtboxes_x0
            x1_ub = self.gtboxes_x1
            y0_ub = self.gtboxes_y0
            y1_ub = self.gtboxes_y1
        else:
            x0_ub = self.bboxes_x0
            x1_ub = self.bboxes_x1
            y0_ub = self.bboxes_y0
            y1_ub = self.bboxes_y1
        # cala x1 - x0
        area_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "area_y1_y0")
        self.tik_instance.vsub(self.mask, area_y1_y0, y1_ub, y0_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(self.mask, area_y1_y0, area_y1_y0, self.eps, self.dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vsub(self.mask, area_ub, x1_ub, x0_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(self.mask, area_ub, area_ub, self.eps, self.dup_rep_time, 1, 1, 8, 8)
        if inter_mode:
            zero_ub = _apply_mem(self.tik_instance, self.dtype, [self.eliments_per_block], "zero_ub")
            self.tik_instance.vector_dup(self.eliments_per_block, zero_ub, 0.0, 1, 1, 8)
            self.tik_instance.vmax(self.mask, area_ub, zero_ub, area_ub, self.dup_rep_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vmax(self.mask, area_y1_y0, zero_ub, area_y1_y0, 
                                   self.dup_rep_time, 1, 0, 1, 8, 0, 8)
        self.tik_instance.vmul(self.mask, area_ub, area_y1_y0, area_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)

    def _run_segment(self, task_idx):
        """
        do a segment of bbox compute
        """
        self._apply_all_ub(self.data_align)

        if not self.trans:
            self.data_move_in(task_idx, self.data_align, self.mov_rep_time)
        else:
            self.data_move_in_and_trans(task_idx, self.mask, self.dup_rep_time, self.data_align, self.mov_rep_time)

        gtboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "gtboxes_area_ub")
        bboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "bboxes_area_ub")
        inter_area_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "inter_area_ub")
        out_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "out_ub")

        # calcu bboxes area
        self.calcu_area(bboxes_area_ub)

        # calcu gtboxes area
        self.calcu_area(gtboxes_area_ub, gt_mode=True)

        # vmin vmax: get inter x0 x1 y0 y1, outer x0 x1 y0 y1
        self.get_inter_outer_area()

        # calcu inter area
        self.calcu_area(inter_area_ub, inter_mode=True)

        if self.mode == "iou":
            self.tik_instance.vadd(self.mask, out_ub, bboxes_area_ub,
                                   gtboxes_area_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(self.mask, out_ub, out_ub, inter_area_ub, 
                                   self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        else:
            self.tik_instance.data_move(out_ub, gtboxes_area_ub, 0, 1, self.mov_rep_time, 0, 0)

        if self.product is True:
            self.tik_instance.vdiv(self.mask, out_ub, inter_area_ub, 
                                   out_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        else:
            # for mini
            self._rev_div(out_ub, inter_area_ub, out_ub)
        move_times = self.mov_rep_time
        # data move process for tasks which are not the last one
        with self.tik_instance.if_scope(task_idx != (self.task_num - 1)):
            self.tik_instance.data_move(self.overlap_gm[self.data_align * task_idx], 
                                    out_ub, 0, 1, move_times, 0, 0)
        # data move process for the last task, move_times changes
        with self.tik_instance.else_scope():
            # calculate number of elements in the last task
            data_align_tail = self.all_num - (self.task_num - 1) * self.data_align
            # calculate move_times in the last task
            move_times = (data_align_tail + self.eliments_per_block - 1) // self.eliments_per_block
            self.tik_instance.data_move(self.overlap_gm[self.data_align * task_idx], 
                                    out_ub, 0, 1, move_times, 0, 0)


# 'pylint: disable = unused-argument,too-many-arguments
@register_operator("Iou")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def iou(bboxes, gtboxes, overlap, mode="iou", eps=1.0, aligned=False, kernel_name="iou"):
    """
    calculating data

    Parameters
    ----------
    bboxes : dict
        shape and dtype of bboxes, the coordinates of bbox
        shape must be [n, 4]
        [x1, y1, x2, y2]
    gtboxes : dict
        shape and dtype of gtboxes, the coordinates of bbox
        shape must be [m, 4]
        [x1, y1, x2, y2]
    overlap : dict
        shape and dtype of overlap
        result shape is [m, n]
    mode :  str
        ('iou','iof')
        iou : the output is gtbox and bbox iou
        iof : the output is inter_area / gtboxes_area
    eps : float
        prevent division by 0
        default value is 1.0
    aligned : bool
        (False, True)
        False : the output shape is [m, n]
        True : the output shape is [n, 1], m equals to n
    kernel_name : str
        kernel name, default value is "iou"

    Returns
    -------
    None
    """
    bboxes_shape = bboxes.get("shape")
    gtboxes_shape = gtboxes.get("shape")

    para_check.check_shape(bboxes_shape, param_name="bboxes")
    para_check.check_shape(gtboxes_shape, param_name="gtboxes")

    bboxes_dtype = bboxes.get("dtype").lower()
    shape_util.compare_tensor_dict_key(bboxes, gtboxes, "dtype")
    check_list = ("float16", "float32")
    para_check.check_dtype(bboxes_dtype, check_list, param_name="bboxes")

    # check whether mode is valid
    check_list = ("iou", "iof")
    if mode not in check_list:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "mode", "iou,iof", mode)

    if not aligned:
        res = Iou(bboxes, gtboxes, mode, eps, aligned).run_tik(kernel_name)
    else:
        res = AlignedIoU(bboxes, gtboxes, False, False, mode, eps).run_tik(kernel_name)

    return res
