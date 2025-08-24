# Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.
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
# =============================================================================
"""
Dynamic roi align compute class for RoiExtractor.
"""
from abc import ABCMeta, abstractmethod
from functools import partial
from impl import common_util
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.util_tik_comm_func import ceil_div, tik_func_vconv, tik_func_vcomple


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # 5HD C0
    C0 = 16
    # reserved ub size for temporary variables, including RESERVED_UB_BYTES_FOR_OUT_MIN
    RESERVED_UB_BYTES_FOR_OUT_MIN = 8 * 1024
    # batch size, 64 align for float32 or 128 align for float16
    BATCH_SIZE = 128
    # stride max value of `data_move`
    BURST_STRIDE_MAX = 65535
    # instr max repeat value
    INSTR_REPEAT_MAX = 255
    # C1 * C0 temporary tensors used in calculation
    TEMP_C1xC0_COUNT = 8


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,huawei-too-many-arguments
class RoiAlignCompute():
    """RoiAlign operator information only"""

    def __init__(self, op, feats_gm, roi_buf, roi_feats_buf, roi_end_mode, rois_per_core, idx):
        """Init."""
        # Init parameters
        self.tik = op.tik
        self.dtype = op.x_dtype
        self.support_vbi = op.support_vbi
        self.ub_size_bytes = op.ub_size
        self.l1_size_bytes = op.l1_size_bytes
        self.data_size = op.data_size
        self.block_element = op.block_element
        self.max_ub_element = op.max_ub_element
        self.c0_blocks = op.c0_blocks
        self.roi_end_mode = roi_end_mode
        self.aligned = op.aligned
        self.mode_compute = dict()

        # Init input and output
        self.features_gm = feats_gm
        self.rois_gm = roi_buf
        self.y_gm = roi_feats_buf

        # Init tiling scalar
        self.pool_h = op.pooled_h
        self.pool_w = op.pooled_h
        self.samples = op.sample_num
        self.scale = op.spatial_scale[idx]
        self.pool_h_reciprocal = op.pooled_h_reciprocal
        self.pool_w_reciprocal = op.pooled_w_reciprocal
        self.samples_reciprocal = op.sample_num_reciprocal
        self.tiling_key = op.tiliing_key[idx]
        self.rois_row_size = op.rois_last_dim
        self.feature_n = op.feats_n_dim
        self.c1 = op.feats_c1_dim
        self.feature_h = op.feats_h_dims[idx]
        self.feature_w = op.feats_w_dims[idx]
        self.rois_per_core = rois_per_core
        self.c1_per_core = op.c1_per_core
        self.out_c1_per_loop = op.out_c1_per_loop[idx]
        self.out_w_per_loop = op.out_w_per_loop[idx]
        self.out_h_per_loop = op.out_h_per_loop

        # Roi this core processes
        self.core_roi_offset = self.tik.Scalar("int64", name="core_roi_offset", init_value=0)
        self.core_rois = self.tik.Scalar("int64", name="core_rois", init_value=self.rois_per_core)
        self.core_c1_offset = self.tik.Scalar("int64", name="core_c1_offset", init_value=0)

        # processor dictionary for different tiling keys
        self._processors = {
            100: MoveWholeFeatureMapToUb,
            300: MoveFeatureMap4Points,
            400: MoveFeatureMap4Points,
        }

    def roi_align_compute(self):
        """
        Main compute entrance.
        """
        # register compute based on tiling_key
        register_func = partial(self.regist_compute, tiling_func=self._functions)
        for k in self._processors:
            register_func(k, key=k)
        # run all registered compute based tiling key
        self.run_compute(self.tiling_key)
        return self.tik

    def regist_compute(self, tiling_key, tiling_func, *var_tuple, **var_key_value):
        """
        Regist compute functions.
        """
        compute_classify = "default"
        if "compute_classify" in var_key_value.keys():
            compute_classify = var_key_value.get(compute_classify)
        if compute_classify not in self.mode_compute.keys():
            self.mode_compute[compute_classify] = dict()
        self.mode_compute[compute_classify][tiling_key] = [tiling_func, var_tuple, var_key_value]

    def run_compute(self, tiling_key, compute_classify=None):
        """
        Run all the regist_compute base on tiling_key.
        """
        for classify_key, compute_info in self.mode_compute.items():
            if compute_classify is not None and classify_key != compute_classify:
                continue
            for key, key_func in compute_info.items():
                with self.tik.if_scope(tiling_key == key):
                    with self.tik.new_stmt_scope():
                        key_func[0](*key_func[1], **key_func[2])

    def _functions(self, key: int):
        """
        Invoke each tiling functions.
        """
        processor = self._processors.get(key)
        processor(self).run()


class ProcessorBase(metaclass=ABCMeta):
    def __init__(self, op) -> None:
        self.op = op
        self.tik: tik.Tik = op.tik
        self.out_init_val = 0.0
        self.pool_func = op.tik.vadd
        self.aligned = op.aligned

        # whole feature map count
        self.feature_hwc0 = self.tik.Scalar(dtype="int64", name="feature_hwc0",
                                            init_value=op.feature_h * op.feature_w * Constant.C0)
        self.feature_c1hwc0 = self.tik.Scalar(dtype="int64", name="feature_c1hwc0",
                                              init_value=op.c1_per_core * self.feature_hwc0)

        # wether cache whole feature map in L1
        self.cache_all_fm_in_l1 = self.tik.Scalar(dtype="int32", name="cache_all_fm_in_l1", init_value=0)
        if op.l1_size_bytes > 0:
            self.fm_in_l1 = self.tik.Tensor(dtype=op.dtype, name="fm_in_l1", scope=tik.scope_cbuf,
                                            shape=[op.l1_size_bytes // op.data_size])
            with self.tik.if_scope(tik.all(self.feature_c1hwc0 <= op.l1_size_bytes // op.data_size,
                                           op.rois_per_core > 1)):
                self.cache_all_fm_in_l1.set_as(1)

        self.roi_in_ub = self.tik.Tensor(dtype="float32", shape=(5, Constant.BATCH_SIZE),
                                         name="roi_in_ub", scope=tik.scope_ubuf)

        # x0, y0
        self.cur_roi_start_x = self.tik.Scalar(dtype="float32", name="cur_roi_start_x")
        self.cur_roi_start_y = self.tik.Scalar(dtype="float32", name="cur_roi_start_y")
        # roi width / height
        self.roi_wh = self.tik.Tensor(dtype="float32", shape=(2, Constant.BATCH_SIZE),
                                      name="roi_wh", scope=tik.scope_ubuf)
        # grid width / height
        self.grid_wh = self.tik.Tensor(dtype="float32", shape=(2, Constant.BATCH_SIZE),
                                       name="grid_wh", scope=tik.scope_ubuf)
        # current roi grid height / width
        self.cur_roi_grid_h = self.tik.Scalar(dtype="float32", name="cur_roi_grid_h")
        self.cur_roi_grid_w = self.tik.Scalar(dtype="float32", name="cur_roi_grid_w")
        # height / width sample count
        self.sample_wh = self.tik.Tensor(dtype="int32", shape=(2, Constant.BATCH_SIZE),
                                         name="sample_wh", scope=tik.scope_ubuf)
        # current roi samples
        self.cur_roi_sample_h = self.tik.Scalar(dtype="int32", name="cur_roi_sample_h")
        self.cur_roi_sample_w = self.tik.Scalar(dtype="int32", name="cur_roi_sample_w")
        self.cur_roi_sample_h_vec = self.tik.Tensor(dtype="float32", name="cur_roi_sample_h_vec",
                                                    scope=tik.scope_ubuf, shape=[8])
        self.cur_roi_sample_w_vec = self.tik.Tensor(dtype="float32", name="cur_roi_sample_w_vec",
                                                    scope=tik.scope_ubuf, shape=[8])

        # feature map index specified in ROI
        self.fm_idx = self.tik.Tensor(dtype="int32", shape=(Constant.BATCH_SIZE,),
                                      name="fm_idx", scope=tik.scope_ubuf)
        self.cur_fm_idx = self.tik.Scalar(dtype="int32", name="cur_fm_idx")
        # output y index
        self.cur_out_n = self.tik.Scalar(dtype="int32", name="cur_out_n")
        # pool width index to find elements in y
        self.pool_w_idx = self.tik.Tensor(dtype="int32", shape=(Constant.BATCH_SIZE,),
                                          name="pool_w_idx", scope=tik.scope_ubuf)
        # 4 points around, x_low, x_high, y_low, y_high
        self.x_lh = self.tik.Tensor(dtype="int32", shape=(2, Constant.BATCH_SIZE),
                                    name="x_lh", scope=tik.scope_ubuf)
        self.y_lh = self.tik.Tensor(dtype="int32", shape=(2, Constant.BATCH_SIZE),
                                    name="y_lh", scope=tik.scope_ubuf)
        # length of 4 points, hx, lx, hy, ly
        self.hlx = self.tik.Tensor(dtype="float32", shape=(2, Constant.BATCH_SIZE),
                                   name="hlx", scope=tik.scope_ubuf)
        self.hly = self.tik.Tensor(dtype="float32", shape=(2, Constant.BATCH_SIZE),
                                   name="hly", scope=tik.scope_ubuf)
        # weight of 4 points, w1=hy*hx, w2=hy*lx, w3=ly*hx, w4=ly*lx
        self.w1234 = self.tik.Tensor(dtype=op.dtype, shape=[4 * Constant.BATCH_SIZE],
                                     name="w1234", scope=tik.scope_ubuf)

        # output row loop scalars
        self.h_grid_remain = self.tik.Scalar(dtype="int64", name="h_grid_remain", init_value=0)
        self.h_grid_idx_all = self.tik.Scalar(dtype="int64", name="h_grid_idx_all", init_value=0)
        self.h_grid_h1 = self.tik.Scalar(dtype="int64", name="h_grid_h1")
        self.h_grid_h2 = self.tik.Scalar(dtype="int64", name="h_grid_h2")

        # cache row / column weights in scenario: sample * pooled <= Constant.BATCH_SIZE
        self.x_weight_cache_idx = self.tik.Scalar("int64", "x_weight_cache_idx")
        self.y_weight_cache_idx = self.tik.Scalar("int64", "y_weight_cache_idx")

        # whole feature map cached index
        self.cache_fm_idx = self.tik.Scalar(dtype="int32", name="cache_fm_idx", init_value=-1)

        # help tensors
        # filled with 0.0 in tensor, total count: one block
        self.const_zero_fp32 = self.tik.Tensor(dtype="float32", shape=(8,),
                                               name="const_zero_fp32", scope=tik.scope_ubuf)
        # filled with 1.0 in tensor, total count: one block
        self.const_one_fp32 = self.tik.Tensor(dtype="float32", shape=(8,),
                                              name="const_one_fp32", scope=tik.scope_ubuf)
        # filled with 1 in tensor, total count: one block
        self.const_one_int = self.tik.Tensor(dtype="int32", shape=(8,),
                                             name="const_one_int", scope=tik.scope_ubuf)
        # filled 0...127 in tensor
        self.const_0_127_fp32 = self.tik.Tensor(dtype="float32", shape=(Constant.BATCH_SIZE,),
                                                name="const_0_127_fp32", scope=tik.scope_ubuf)
        # filled with -1.0 in tensor, total count: one block
        self.const_minus_1_fp32 = self.tik.Tensor(dtype="float32", shape=(8,),
                                                  name="const_minus_1_fp32", scope=tik.scope_ubuf)
        # filled with height/weight of feature_map shape in tensor, total count: one block
        self.const_fm_h_fp = self.tik.Tensor(dtype="float32", shape=(8,),
                                             name="const_fm_h_fp", scope=tik.scope_ubuf)
        self.const_fm_w_fp = self.tik.Tensor(dtype="float32", shape=(8,),
                                             name="const_fm_w_fp", scope=tik.scope_ubuf)
        # filled with height-1/weight-1 of feature_map shape in tensor, total count: 8
        self.const_fm_h_minus_1 = self.tik.Tensor(dtype="int32", shape=(8,),
                                                  name="const_fm_h_minus_1", scope=tik.scope_ubuf)
        self.const_fm_w_minus_1 = self.tik.Tensor(dtype="int32", shape=(8,),
                                                  name="const_fm_w_minus_1", scope=tik.scope_ubuf)
        # filled with height-1/weight-1 of feature_map shape in tensor, total count: one block
        self.const_fm_h_minus_1_fp = self.tik.Tensor(dtype="float32", shape=(8,),
                                                     name="const_fm_h_minus_1_fp", scope=tik.scope_ubuf)
        self.const_fm_w_minus_1_fp = self.tik.Tensor(dtype="float32", shape=(8,),
                                                     name="const_fm_w_minus_1_fp", scope=tik.scope_ubuf)
        # temporary fp tensor
        self.tmp_fp32 = self.tik.Tensor(dtype="float32", shape=(2, Constant.BATCH_SIZE),
                                        name="tmp_fp32", scope=tik.scope_ubuf)
        # memory left for vbi
        int32_bytes = 4
        self.vbi_memory = self.tik.Scalar(dtype="int32", name="vbi_memory", init_value=0)
        self.vbi_mem_factor = self.tik.Scalar(dtype="int32", name="vbi_mem_factor",
                                              init_value=4 * int32_bytes + (4 + Constant.C0) * op.data_size)
        # memory left max for vbi
        self.memory_for_vbi_max = self.tik.Scalar(dtype="int32", name="memory_for_vbi_max")

    @abstractmethod
    def tiling_compute_one_roi(self, roi_idx: tik.Scalar, c1_offset: tik.Scalar, c1_num: tik.Scalar) -> None:
        pass

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    @abstractmethod
    def bilinear_interpolate_one_grid_row(self, fm_in_ub: tik.Tensor, out_y_ub: tik.Tensor,
                                          relative_c1_offset: tik.Scalar,
                                          absolute_c1_offset: tik.Scalar, c1_num: tik.Scalar,
                                          out_w_offset: tik.Scalar, out_w_num: tik.Scalar,
                                          y_low: tik.Scalar, y_high: tik.Scalar,
                                          grid_h_idx: tik.Scalar, grid_w_num: tik.Scalar) -> None:
        pass

    def run(self) -> None:
        self._prepare()
        roi_loops = ceil_div(self.op.core_rois, Constant.BATCH_SIZE)
        with self.tik.for_range(0, roi_loops) as _roi_loop_idx:
            rois_num_per_loop, roi_offset = self._calc_segment(self.op.core_rois, _roi_loop_idx, Constant.BATCH_SIZE)
            roi_offset.set_as(self.op.core_roi_offset + roi_offset)
            self.compute_roi_batch(roi_offset, rois_num_per_loop)

    def compute_roi_batch(self, roi_offset: tik.Scalar, rois_num_per_loop: tik.Scalar) -> None:
        self._move_roi_data_to_ub(rois_num_per_loop, roi_offset)
        self._calc_fm_start_end_coordinate()
        self._calc_roi_size()
        self._calc_grid_size()
        self._convert_fm_idx()
        self.before_loop_roi_batch()
        with self.tik.for_range(0, rois_num_per_loop) as _roi_idx:
            self.compute_one_roi(roi_offset, _roi_idx, self.op.core_c1_offset, self.op.c1_per_core)

    def before_loop_roi_batch(self) -> None:
        pass

    def compute_one_roi(self, roi_offset: tik.Scalar, roi_idx: tik.Scalar,
                        core_c1_offset: tik.Scalar, core_c1_num: tik.Scalar) -> None:
        self.cur_fm_idx.set_as(self.fm_idx[roi_idx])
        self.cur_roi_start_x.set_as(self.roi_in_ub[1, roi_idx])
        self.cur_roi_start_y.set_as(self.roi_in_ub[2, roi_idx])
        self.cur_roi_grid_w.set_as(self.grid_wh[0, roi_idx])
        self.cur_roi_grid_h.set_as(self.grid_wh[1, roi_idx])
        self.cur_roi_sample_w.set_as(self.sample_wh[0, roi_idx])
        self.cur_roi_sample_h.set_as(self.sample_wh[1, roi_idx])
        self.cur_out_n.set_as(roi_offset + roi_idx)

        self.tik.vec_dup(mask=8, dst=self.cur_roi_sample_w_vec, scalar=1.0 * self.cur_roi_sample_w,
                         repeat_times=1, dst_rep_stride=8)
        self.tik.vec_dup(mask=8, dst=self.cur_roi_sample_h_vec, scalar=1.0 * self.cur_roi_sample_h,
                         repeat_times=1, dst_rep_stride=8)

        with self.tik.if_scope(tik.all(self.cur_fm_idx >= 0, self.cur_fm_idx < self.op.feature_n)):
            self.x_weight_cache_idx.set_as(-1)
            self.y_weight_cache_idx.set_as(-1)
            self.tiling_compute_one_roi(roi_idx, core_c1_offset, core_c1_num)

    def tiling_compute_output(self, fm_in_ub: tik.Tensor, c1_offset: tik.Scalar, c1_num: tik.Scalar) -> None:
        # cut output W
        out_w_loops = ceil_div(self.op.pool_w, self.op.out_w_per_loop)
        with self.tik.for_range(0, out_w_loops) as _out_w_loop_idx:
            out_w_num, out_w_offset = self._calc_segment(self.op.pool_w, _out_w_loop_idx, self.op.out_w_per_loop)
            # cut output C1
            out_c1_loops = ceil_div(c1_num, self.op.out_c1_per_loop)
            with self.tik.for_range(0, out_c1_loops) as _out_c1_loop_idx:
                out_c1_num, relative_c1_offset = self._calc_segment(c1_num, _out_c1_loop_idx,
                                                                    self.op.out_c1_per_loop)
                out_c1_offset = self.tik.Scalar(dtype="int64", name="out_c1_offset",
                                                init_value=c1_offset + relative_c1_offset)
                out_y_ub = self.tik.Tensor(dtype=self.op.dtype, name="out_y_ub", scope=tik.scope_ubuf,
                                           shape=[out_c1_num, out_w_num, Constant.C0])
                self._clear_ub(out_y_ub, out_c1_num * out_w_num * Constant.C0, self.out_init_val)
                self.h_grid_remain.set_as(0)
                self.h_grid_idx_all.set_as(0)
                # cut h samples with 128 batch
                grid_h_loops = ceil_div(self.cur_roi_sample_h * self.op.pool_h, Constant.BATCH_SIZE)
                with self.tik.for_range(0, grid_h_loops) as _grid_h_loop_idx:
                    grid_h_num, grid_h_offset = self._calc_segment(self.cur_roi_sample_h * self.op.pool_h,
                                                                   _grid_h_loop_idx, Constant.BATCH_SIZE)
                    self.compute_grid_batch_h(fm_in_ub,
                                              out_y_ub, relative_c1_offset,
                                              out_c1_offset, out_c1_num,
                                              out_w_offset, out_w_num,
                                              grid_h_offset, grid_h_num)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def compute_grid_batch_h(self, fm_in_ub: tik.Tensor,
                             out_y_ub: tik.Tensor, relative_c1_offset: tik.Scalar,
                             out_c1_offset: tik.Scalar, out_c1_num: tik.Scalar,
                             out_w_offset: tik.Scalar, out_w_num: tik.Scalar,
                             grid_h_offset: tik.Scalar, grid_h_num: tik.Scalar) -> None:
        with self.tik.if_scope(self.y_weight_cache_idx != grid_h_offset):
            self.y_weight_cache_idx.set_as(grid_h_offset)
            self._calc_bilinear_interpolate_coordinate(grid_offset=grid_h_offset, is_width=False)
        self.h_grid_h1.set_as(0)
        tmp_unfinished_grid = self.tik.Scalar(dtype="int64", name="tmp_unfinished_grid",
                                              init_value=self.cur_roi_sample_h - self.h_grid_remain)
        self.tik.scalar_min(dst=self.h_grid_h2, src0=tmp_unfinished_grid, src1=grid_h_num)
        pool_h_loops = ceil_div(grid_h_num, self.cur_roi_sample_h) + 1
        # loop each output row
        with self.tik.for_range(0, pool_h_loops) as _pool_h_loop_idx:
            # loop grid width
            grid_w_loops = ceil_div(self.cur_roi_sample_w * out_w_num, Constant.BATCH_SIZE)
            with self.tik.for_range(0, grid_w_loops) as _grid_w_loop_idx:
                grid_w_num, grid_w_offset = self._calc_segment(self.cur_roi_sample_w * out_w_num,
                                                               _grid_w_loop_idx, Constant.BATCH_SIZE)
                grid_w_offset.set_as(out_w_offset * self.cur_roi_sample_w + grid_w_offset)
                self.compute_grid_batch_w(fm_in_ub, out_y_ub, relative_c1_offset,
                                          out_c1_offset, out_c1_num, out_w_offset, out_w_num,
                                          grid_w_offset, grid_w_num)
            # check whether to move out to GM
            self.h_grid_idx_all.set_as(self.h_grid_h2 + grid_h_offset)
            with self.tik.if_scope(self.h_grid_idx_all % self.cur_roi_sample_h == 0):
                # move y out
                gm_offset = (((self.cur_out_n * self.op.c1 + out_c1_offset) * self.op.pool_h +
                              self.h_grid_idx_all // self.cur_roi_sample_h - 1) * self.op.pool_w +
                             out_w_offset) * Constant.C0
                self._move_out_y_ub_to_gm(out_y_ub, out_c1_num, out_w_num, gm_offset)
                # clean up output ub
                self._clear_ub(out_y_ub, out_c1_num * out_w_num * Constant.C0, self.out_init_val)
                # reset remain grid
                self.h_grid_remain.set_as(0)
            with self.tik.else_scope():
                # remain grid for next batch grid height
                self.h_grid_remain.set_as(grid_h_num - self.h_grid_h1 + self.h_grid_remain)
            self.h_grid_h1.set_as(self.h_grid_h2)
            tmp_next_grid = self.tik.Scalar(dtype="int64", name="tmp_next_grid",
                                            init_value=self.h_grid_h1 + self.cur_roi_sample_h)
            self.tik.scalar_min(dst=self.h_grid_h2, src0=tmp_next_grid, src1=grid_h_num)
            with self.tik.if_scope(self.h_grid_h1 == self.h_grid_h2):
                self.tik.tik_break()

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def compute_grid_batch_w(self, fm_in_ub: tik.Tensor, out_y_ub: tik.Tensor, relative_c1_offset: tik.Scalar,
                             out_c1_offset: tik.Scalar, out_c1_num: tik.Scalar,
                             out_w_offset: tik.Scalar, out_w_num: tik.Scalar,
                             grid_w_offset: tik.Scalar, grid_w_num: tik.Scalar):
        with self.tik.if_scope(self.x_weight_cache_idx != grid_w_offset):
            self.x_weight_cache_idx.set_as(grid_w_offset)
            self._calc_pool_idx_of_grid(self.pool_w_idx, grid_w_offset, self.cur_roi_sample_w)
            self._calc_bilinear_interpolate_coordinate(grid_offset=grid_w_offset, is_width=True)
        with self.tik.for_range(self.h_grid_h1, self.h_grid_h2) as _grid_h_idx:  # _grid_h_idx: [0, grid_h_num)
            y_low = self.tik.Scalar(dtype="int32", name="y_low", init_value=self.y_lh[0, _grid_h_idx])
            y_high = self.tik.Scalar(dtype="int32", name="y_high", init_value=self.y_lh[1, _grid_h_idx])
            self.bilinear_interpolate_one_grid_row(fm_in_ub, out_y_ub, relative_c1_offset, out_c1_offset, out_c1_num,
                                                   out_w_offset, out_w_num, y_low, y_high, _grid_h_idx, grid_w_num)

    def _calc_segment(self, total_seg: tik.Scalar, seg_index: tik.Scalar, seg_len: tik.Scalar):
        ret_seg_len = self.tik.Scalar(dtype="int64", name="ret_seg_len", init_value=seg_len)
        ret_offset = self.tik.Scalar(dtype="int64", name="ret_offset", init_value=seg_index * seg_len)
        with self.tik.if_scope(total_seg - seg_index * seg_len < seg_len):
            ret_seg_len.set_as(total_seg - seg_index * seg_len)
        return ret_seg_len, ret_offset

    def _move_roi_data_to_ub(self, roi_num: tik.Scalar, roi_offset: tik.Scalar) -> None:
        """move roi data from GM to UB"""
        with self.tik.new_stmt_scope():
            # incase of dst address overlap
            blocks = ceil_div(roi_num * self.op.rois_row_size, self.op.block_element)
            rois_data_tmp = self.tik.Tensor(self.op.dtype, [Constant.BATCH_SIZE * self.op.rois_row_size],
                                            name="rois_data_tmp", scope=tik.scope_ubuf)
            self.tik.data_move(dst=rois_data_tmp, src=self.op.rois_gm[roi_offset * self.op.rois_row_size],
                               sid=0, nburst=1, burst=blocks, src_stride=0, dst_stride=0)
            if self.op.dtype == "float32":
                src = rois_data_tmp
            else:
                # convert from float16 to float32. because of coordinate calculation precision
                roi_data_fp32_tmp = self.tik.Tensor("float32", [Constant.BATCH_SIZE * self.op.rois_row_size],
                                                    name="roi_data_fp32_tmp", scope=tik.scope_ubuf)
                self.tik.vconv(mask=64, round_mode='', dst=roi_data_fp32_tmp, src=rois_data_tmp,
                               repeat_times=Constant.BATCH_SIZE * self.op.rois_row_size // 64,
                               dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=8, src_rep_stride=4)
                src = roi_data_fp32_tmp
            with self.tik.for_range(0, roi_num) as idx:
                self.roi_in_ub[0, idx].set_as(src[idx * self.op.rois_row_size + 0])
                self.roi_in_ub[1, idx].set_as(src[idx * self.op.rois_row_size + 1])
                self.roi_in_ub[2, idx].set_as(src[idx * self.op.rois_row_size + 2])
                self.roi_in_ub[3, idx].set_as(src[idx * self.op.rois_row_size + 3])
                self.roi_in_ub[4, idx].set_as(src[idx * self.op.rois_row_size + 4])

    def _calc_fm_start_end_coordinate(self) -> None:
        """calculate feature map coordinate: x0', y0', x1', y1'"""
        # 4 instructions in one
        # x0' = x0 * scale, y0' = y0 * scale, x1' = x1 * scale, y1' = y1 * scale
        self.tik.vmuls(mask=64, dst=self.roi_in_ub[1, 0], src=self.roi_in_ub[1, 0],
                       scalar=self.op.scale, repeat_times=2 * 4,
                       dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=8, src_rep_stride=8)

    def _calc_roi_size(self) -> None:
        """calculate roi width and height"""
        # For pytorch/mmcv
        if self.aligned:
            self.tik.vadds(mask=64, dst=self.roi_in_ub[1, 0], src=self.roi_in_ub[1, 0], scalar=-0.5,
                           repeat_times=2 * 4, dst_blk_stride=1, src_blk_stride=1,
                           dst_rep_stride=8, src_rep_stride=8)
        # 2 instructions in one
        # roi_w = x1' - x0', roi_h = y1' - y0'
        self.tik.vsub(mask=64, dst=self.roi_wh, src0=self.roi_in_ub[3, 0], src1=self.roi_in_ub[1, 0],
                      repeat_times=2 * 2, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                      dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)
        # `roi_w = max(roi_w, 1)`, `roi_h = max(roi_h, 1)`
        self.tik.vmax(mask=64, dst=self.roi_wh, src0=self.roi_wh, src1=self.const_one_fp32,
                      repeat_times=2 * 2, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=0,
                      dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=0)

    def _calc_grid_size(self) -> None:
        """calculate grid width and height"""
        op = self.op
        # `bin_w = roi_w / pool_w`
        self.tik.vmuls(mask=64, dst=self.grid_wh[0, 0], src=self.roi_wh[0, 0],
                       scalar=op.pool_w_reciprocal, repeat_times=2,
                       dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=8, src_rep_stride=8)
        # `bin_h = roi_h / pool_h`
        self.tik.vmuls(mask=64, dst=self.grid_wh[1, 0], src=self.roi_wh[1, 0],
                       scalar=op.pool_h_reciprocal, repeat_times=2,
                       dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=8, src_rep_stride=8)
        # count sample counts if not positive
        with self.tik.if_scope(op.samples <= 0):
            # `sample_w = ceil(bin_w)`, `sample_h = ceil(bin_h)`
            self._convert_fpx_to_int32(round_mode='ceil', dst=self.sample_wh, src=self.grid_wh,
                                       max_num=2 * Constant.BATCH_SIZE, instr_num=2)
            # sample(int32) -> sample(float32)
            self._convert_int32_to_fpx(round_mode='', dst=self.tmp_fp32, src=self.sample_wh,
                                       max_num=2 * Constant.BATCH_SIZE, instr_num=2)
            # `grid_w = bin_w / samples_w`
            self._vdiv(dst=self.grid_wh, src0=self.grid_wh, src1=self.tmp_fp32,
                       max_num=Constant.BATCH_SIZE * 2, instr_num=2)
        with self.tik.else_scope():
            # `grid_w = bin_w * samples_reciprocal`
            # 2 instructions in one
            self.tik.vmuls(mask=64, dst=self.grid_wh, src=self.grid_wh, scalar=op.samples_reciprocal,
                           repeat_times=2 * 2, dst_blk_stride=1, src_blk_stride=1,
                           dst_rep_stride=8, src_rep_stride=8)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _vdiv(self, dst: tik.Tensor, src0: tik.Tensor, src1: tik.Tensor,
              max_num: int, instr_num: int = 1):
        if not tbe_platform.api_check_support("tik.vdiv", dst.dtype):
            tik_func_vcomple(self.tik, "vdiv", out_dst=dst, src0=src0, src1=src1, copy_num=max_num)
        else:
            mask = 64 if dst.dtype == "float32" else 128
            repeat = 2 if dst.dtype == "float32" else 1
            self.tik.vdiv(mask=mask, dst=dst, src0=src0, src1=src1, repeat_times=repeat * instr_num,
                          dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                          dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _convert_fpx_to_int32(self, round_mode: str, dst: tik.Tensor, src: tik.Tensor,
                              max_num: int, instr_num: int = 1):
        if src.dtype == "float32" and not tbe_platform.api_check_support("tik.vconv", "f322s32r"):
            tik_func_vconv(self.tik, dst_ub=dst, src_ub=src, do_len=max_num, mode=round_mode)
        else:
            if src.dtype == "float32":
                self.tik.vconv(mask=64, round_mode=round_mode, dst=dst, src=src,
                               repeat_times=2 * instr_num, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=8, src_rep_stride=8)
            else:
                self.tik.vconv(mask=64, round_mode=round_mode, dst=dst, src=src,
                               repeat_times=2 * instr_num, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=8, src_rep_stride=4)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _convert_int32_to_fpx(self, round_mode: str, dst: tik.Tensor, src: tik.Tensor,
                              max_num: int, instr_num: int = 1):
        if dst.dtype == "float32" and not tbe_platform.api_check_support("tik.vconv", "s322f32"):
            tik_func_vconv(self.tik, dst_ub=dst, src_ub=src, do_len=max_num, mode=round_mode)
        else:
            if dst.dtype == "float32":
                self.tik.vconv(mask=64, round_mode=round_mode, dst=dst, src=src,
                               repeat_times=2 * instr_num, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=8, src_rep_stride=8)
            else:
                self.tik.vconv(mask=64, round_mode=round_mode, dst=dst, src=src,
                               repeat_times=2 * instr_num, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=4, src_rep_stride=8, deqscale=1.0)

    def _convert_fm_idx(self) -> None:
        """convert indices in ROI from float to int"""
        self._convert_fpx_to_int32(round_mode='floor', dst=self.fm_idx, src=self.roi_in_ub[0, 0],
                                   max_num=Constant.BATCH_SIZE)

    def _clear_ub(self, ub_to_clear: tik.Tensor, clear_len: tik.Scalar, clear_value: float = 0):
        """clear ub to zero"""
        block_elements = common_util.get_block_element(ub_to_clear.dtype)
        mask_max = block_elements * 8

        one_loop_offset = mask_max * Constant.INSTR_REPEAT_MAX
        repeat = clear_len // one_loop_offset
        with self.tik.if_scope(repeat > 0):
            with self.tik.for_range(0, repeat) as index:
                tmp_offset = index * one_loop_offset
                self.tik.vec_dup(mask=mask_max, dst=ub_to_clear[tmp_offset], scalar=clear_value,
                                 repeat_times=Constant.INSTR_REPEAT_MAX, dst_rep_stride=8)

        one_loop_repeat = (clear_len % one_loop_offset) // mask_max
        with self.tik.if_scope(one_loop_repeat > 0):
            tmp_offset = repeat * one_loop_offset
            self.tik.vec_dup(mask=mask_max, dst=ub_to_clear[tmp_offset], scalar=clear_value,
                             repeat_times=one_loop_repeat, dst_rep_stride=8)

        last_num = clear_len % mask_max
        with self.tik.if_scope(last_num > 0):
            tmp_offset = repeat * one_loop_offset + one_loop_repeat * mask_max
            self.tik.vec_dup(mask=last_num, dst=ub_to_clear[tmp_offset], scalar=clear_value,
                             repeat_times=1, dst_rep_stride=8)

    def _calc_bilinear_interpolate_coordinate(self, grid_offset: tik.Scalar, is_width: bool = True) -> None:
        """
        calculate feature map coordinates
        grid_x = start_coordinate + grid_distance * (grid_offset + grid_idx + 0.5) where grid_idx is 0 - 127
        """
        if is_width:
            fm_border_fp = self.const_fm_w_fp
            fm_border_minus_1 = self.const_fm_w_minus_1
            fm_border_minus_1_fp = self.const_fm_w_minus_1_fp
            dst_x_low_high = self.x_lh
            dst_hx_lx = self.hlx
            grid_distance = self.cur_roi_grid_w
            start_coordinate = self.cur_roi_start_x
        else:
            fm_border_fp = self.const_fm_h_fp
            fm_border_minus_1 = self.const_fm_h_minus_1
            fm_border_minus_1_fp = self.const_fm_h_minus_1_fp
            dst_x_low_high = self.y_lh
            dst_hx_lx = self.hly
            grid_distance = self.cur_roi_grid_h
            start_coordinate = self.cur_roi_start_y
        x = self.tik.Tensor("float32", shape=(Constant.BATCH_SIZE,), name="x", scope=tik.scope_ubuf)
        # `grid_idx * grid_distance`
        self.tik.vmuls(mask=64, dst=self.tmp_fp32[0, 0], src=self.const_0_127_fp32, scalar=grid_distance,
                       repeat_times=2, dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=8, src_rep_stride=8)
        grid_start = start_coordinate + grid_distance * grid_offset + 0.5 * grid_distance
        self.tik.vadds(mask=64, dst=self.tmp_fp32[0, 0], src=self.tmp_fp32[0, 0], scalar=grid_start,
                       repeat_times=2, dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=8, src_rep_stride=8)
        # `if x <= 0: x = 0`
        self.tik.vmax(mask=64, dst=x, src0=self.tmp_fp32[0, 0], src1=self.const_zero_fp32,
                      repeat_times=2, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=0,
                      dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=0)
        # `x_low = floor(x)`
        self._convert_fpx_to_int32(round_mode='floor', dst=dst_x_low_high[0, 0], src=x, max_num=Constant.BATCH_SIZE)
        # `x_high = x_low + 1`. INT32
        self.tik.vadd(mask=64, dst=dst_x_low_high[1, 0], src0=dst_x_low_high[0, 0], src1=self.const_one_int,
                      repeat_times=2, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=0,
                      dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=0)
        # `if x_low >= width - 1: x_low = width - 1;  if x_high >= width - 1: x_high = width - 1`. INT32
        self.tik.vmin(mask=64, dst=dst_x_low_high, src0=dst_x_low_high, src1=fm_border_minus_1,
                      repeat_times=2 * 2, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=0,
                      dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=0)
        # `if x >= width - 1: x = width - 1`
        self.tik.vmin(mask=64, dst=x, src0=x, src1=fm_border_minus_1_fp,
                      repeat_times=2, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=0,
                      dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=0)
        # `lx = x - x_low`
        self._convert_int32_to_fpx(round_mode='', dst=self.tmp_fp32[1, 0], src=dst_x_low_high[0, 0],
                                   max_num=Constant.BATCH_SIZE)
        self.tik.vsub(mask=64, dst=dst_hx_lx[1, 0], src0=x, src1=self.tmp_fp32[1, 0],
                      repeat_times=2, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                      dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)
        # `hx = 1 - lx`
        self.tik.vsub(mask=64, dst=dst_hx_lx[0, 0], src0=self.const_one_fp32, src1=dst_hx_lx[1, 0],
                      repeat_times=2, dst_blk_stride=1, src0_blk_stride=0, src1_blk_stride=1,
                      dst_rep_stride=8, src0_rep_stride=0, src1_rep_stride=8)
        # `if x < -1.0 or x > width: lx = 0, hx = 0`
        tmp_cmp_mask = self.tik.Tensor("uint16", (32,), name="tmp_cmp_mask", scope=tik.scope_ubuf)
        # Ascend310 does not support dynamic.
        if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in ("Ascend910",):
            # 910 vsel only support mode=0
            # mov_tensor_to_cmpmask supports 2*uint64 = 128bits each time
            with self.tik.for_range(0, 2) as _repeat_idx:
                # each time 64*fp32 / 128*fp16 numbers
                num_per_loop = 64
                # vcmpv_lt compares 256bytes. (128fp16, 64fp32), dst must be 32 bytes align
                self.tik.vcmpv_lt(dst=tmp_cmp_mask, src0=self.tmp_fp32[0, _repeat_idx * num_per_loop],
                                  src1=self.const_minus_1_fp32, repeat_times=1,
                                  src0_blk_stride=1, src1_blk_stride=0,
                                  src0_rep_stride=8, src1_rep_stride=0)
                self.tik.vcmpv_gt(dst=tmp_cmp_mask[16], src0=self.tmp_fp32[0, _repeat_idx * num_per_loop],
                                  src1=fm_border_fp, repeat_times=1,
                                  src0_blk_stride=1, src1_blk_stride=0,
                                  src0_rep_stride=8, src1_rep_stride=0)
                # vor only support uint16/int16, mask=4 means 64 bit for fp32, 8 means 128 bit for fp16
                vor_mask = 4
                self.tik.vor(mask=vor_mask, dst=tmp_cmp_mask, src0=tmp_cmp_mask[0], src1=tmp_cmp_mask[16],
                             repeat_times=1, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                             dst_rep_stride=2, src0_rep_stride=2, src1_rep_stride=2)
                cmpmask = self.tik.mov_tensor_to_cmpmask(tmp_cmp_mask)
                # 2 instructions in one
                self.tik.vsel(mask=64, mode=0, dst=dst_hx_lx[_repeat_idx * num_per_loop], sel=cmpmask,
                              src0=self.const_zero_fp32, src1=dst_hx_lx[_repeat_idx * num_per_loop],
                              repeat_times=2, dst_blk_stride=1, src0_blk_stride=0, src1_blk_stride=1,
                              dst_rep_stride=16, src0_rep_stride=0, src1_rep_stride=16)
        else:
            # vcmpv_lt compares 256bytes. (128fp16, 64fp32). if repeat=2, 2*64fp32 will be done.
            self.tik.vcmpv_lt(dst=tmp_cmp_mask, src0=self.tmp_fp32, src1=self.const_minus_1_fp32,
                              repeat_times=2, src0_blk_stride=1, src1_blk_stride=0,
                              src0_rep_stride=8, src1_rep_stride=0)
            self.tik.vcmpv_gt(dst=tmp_cmp_mask[16], src0=self.tmp_fp32, src1=fm_border_fp,
                              repeat_times=2, src0_blk_stride=1, src1_blk_stride=0,
                              src0_rep_stride=8, src1_rep_stride=0)
            # 8: 128 bit together
            self.tik.vor(mask=8, dst=tmp_cmp_mask, src0=tmp_cmp_mask[0], src1=tmp_cmp_mask[16],
                         repeat_times=1, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                         dst_rep_stride=2, src0_rep_stride=2, src1_rep_stride=2)
            self.tik.vsel(mask=64, mode=2, dst=dst_hx_lx[0, 0], sel=tmp_cmp_mask,
                          src0=self.const_zero_fp32, src1=dst_hx_lx[0, 0], repeat_times=2,
                          dst_blk_stride=1, src0_blk_stride=0, src1_blk_stride=1,
                          dst_rep_stride=8, src0_rep_stride=0, src1_rep_stride=8)
            self.tik.vsel(mask=64, mode=2, dst=dst_hx_lx[1, 0], sel=tmp_cmp_mask,
                          src0=self.const_zero_fp32, src1=dst_hx_lx[1, 0], repeat_times=2,
                          dst_blk_stride=1, src0_blk_stride=0, src1_blk_stride=1,
                          dst_rep_stride=8, src0_rep_stride=0, src1_rep_stride=8)

        if is_width:
            tmp = self.tik.Scalar(dtype="float32", name="tmp", init_value=self.cur_roi_sample_w)
            self._vdivs_fp32(dst=dst_hx_lx, divided=dst_hx_lx, fp_divisor=tmp,
                             fp_divisor_tensor=self.cur_roi_sample_w_vec, instr_num=2)
        else:
            tmp = self.tik.Scalar(dtype="float32", name="tmp", init_value=self.cur_roi_sample_h)
            self._vdivs_fp32(dst=dst_hx_lx, divided=dst_hx_lx, fp_divisor=tmp,
                             fp_divisor_tensor=self.cur_roi_sample_h_vec, instr_num=2)

    def _vdivs_fp32(self, dst: tik.Tensor, divided: tik.Tensor, fp_divisor: tik.Scalar,
                    fp_divisor_tensor: tik.Tensor = None, instr_num: int = 1):
        if tbe_platform.api_check_support("tik.vdiv", "float32"):
            if fp_divisor_tensor is None:
                fp_divisor_tensor = self.tik.Tensor("float32", name="fp_divisor_tensor",
                                                    shape=[8], scope=tik.scope_ubuf)
                self.tik.vector_dup(mask=8, dst=fp_divisor_tensor, scalar=fp_divisor,
                                    repeat_times=1, dst_blk_stride=1, dst_rep_stride=8)
            self.tik.vdiv(mask=64, dst=dst, src0=divided, src1=fp_divisor_tensor,
                          repeat_times=2 * instr_num, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=0,
                          dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=0)
        else:
            self.tik.vmuls(mask=64, dst=dst, src=divided, scalar=1.0 / fp_divisor,
                           repeat_times=2 * instr_num, dst_blk_stride=1, src_blk_stride=1,
                           dst_rep_stride=8, src_rep_stride=8)

    def _calc_pool_idx_of_grid(self, dst_pool_idx: tik.Tensor,
                               grid_offset: tik.Scalar, sample_int: tik.Scalar) -> None:
        """pool_idx = (grid_offset + grid_idx) // sample_count"""
        # grid_offset + grid_idx where grid_idx is 0..127
        self.tik.vadds(mask=64, dst=self.tmp_fp32[0, 0], src=self.const_0_127_fp32, scalar=grid_offset * 1.0,
                       repeat_times=2, dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=8, src_rep_stride=8)
        # grid index / sample
        tmp = self.tik.Scalar(dtype="float32", name="tmp", init_value=sample_int)
        self._vdivs_fp32(dst=self.tmp_fp32[0, 0], divided=self.tmp_fp32[0, 0], fp_divisor=tmp)
        # floor(grid_idx / sample), float16/float32 -> int32
        self._convert_fpx_to_int32(round_mode='floor', dst=dst_pool_idx, src=self.tmp_fp32[0, 0],
                                   max_num=Constant.BATCH_SIZE)

    def _move_out_y_ub_to_gm(self, out_y_ub: tik.Tensor, c1_num: tik.Scalar,
                             w_num: tik.Scalar, gm_offset: tik.Scalar) -> None:
        """move result from UB to GM"""
        op = self.op
        burst_num = c1_num
        burst_len = w_num * op.c0_blocks
        burst_dst_gap = op.pool_h * op.pool_w * op.c0_blocks - burst_len
        with self.tik.new_stmt_scope(disable_sync=True):
            with self.tik.if_scope(burst_dst_gap <= Constant.BURST_STRIDE_MAX):
                self.tik.data_move(dst=op.y_gm[gm_offset], src=out_y_ub,
                                   sid=0, nburst=burst_num, burst=burst_len,
                                   src_stride=0, dst_stride=burst_dst_gap)
            with self.tik.else_scope():
                with self.tik.for_range(0, burst_num) as _c1_idx:
                    dst_offset = gm_offset + _c1_idx * op.pool_h * op.pool_w * Constant.C0
                    src_offset = _c1_idx * w_num * Constant.C0
                    self.tik.data_move(dst=op.y_gm[dst_offset], src=out_y_ub[src_offset],
                                       sid=0, nburst=1, burst=burst_len,
                                       src_stride=0, dst_stride=0)

    def _prepare(self) -> None:
        """prepares"""
        self._clear_roi_in_ub()
        with self.tik.if_scope(self.op.samples > 0):
            # int32 vector_dup
            self.tik.vector_dup(mask=64, dst=self.sample_wh, scalar=self.op.samples, repeat_times=2 * 2,
                                dst_blk_stride=1, dst_rep_stride=8)
        # constant vectors prepare, all are one block only
        self.tik.vector_dup(8, self.const_zero_fp32, 0, 1, 0, 0)
        self.tik.vector_dup(8, self.const_one_fp32, 1, 1, 0, 0)  # fp32
        self.tik.vector_dup(8, self.const_one_int, 1, 1, 0, 0)  # int32
        self.tik.vector_dup(8, self.const_minus_1_fp32, -1.0, 1, 0, 0)
        self.tik.vector_dup(8, self.const_fm_h_fp, self.op.feature_h + 0, 1, 0, 0)
        self.tik.vector_dup(8, self.const_fm_w_fp, self.op.feature_w + 0, 1, 0, 0)
        self.tik.vector_dup(8, self.const_fm_h_minus_1, self.op.feature_h - 1, 1, 0, 0)  # int32
        self.tik.vector_dup(8, self.const_fm_w_minus_1, self.op.feature_w - 1, 1, 0, 0)  # int32
        self.tik.vector_dup(8, self.const_fm_h_minus_1_fp, self.op.feature_h - 1, 1, 0, 0)
        self.tik.vector_dup(8, self.const_fm_w_minus_1_fp, self.op.feature_w - 1, 1, 0, 0)
        # 0-127 const index
        with self.tik.for_range(0, Constant.BATCH_SIZE) as i:
            self.const_0_127_fp32[i] = i

    def _clear_roi_in_ub(self) -> None:
        """clear duty data"""
        mask = 64 if self.roi_in_ub.dtype == "float32" else 128
        self.tik.vector_dup(mask=mask, dst=self.roi_in_ub, scalar=0.0,
                            repeat_times=ceil_div(5 * Constant.BATCH_SIZE, mask),
                            dst_blk_stride=1, dst_rep_stride=8)

    def _calc_bilinear_interpolate_weight(self, grid_h_idx: tik.Scalar, disable_vbi: bool = False) -> None:
        """w1=hy*hx, w2=hy*lx, w3=ly*hx, w4=ly*lx"""
        hy = self.tik.Scalar(dtype="float32", name="hy", init_value=self.hly[0, grid_h_idx])
        ly = self.tik.Scalar(dtype="float32", name="ly", init_value=self.hly[1, grid_h_idx])
        if not self.op.support_vbi or disable_vbi:
            if self.op.dtype == "float32":
                # 2 instructions in one. w1=hy*hx, w2=hy*lx, w3=ly*hx, w4=ly*lx
                with self.tik.new_stmt_scope(disable_sync=True):
                    self.tik.vmuls(mask=64, dst=self.w1234[0:], src=self.hlx, scalar=hy,
                                   repeat_times=2 * 2, dst_blk_stride=1, src_blk_stride=1,
                                   dst_rep_stride=8, src_rep_stride=8)
                    self.tik.vmuls(mask=64, dst=self.w1234[2 * Constant.BATCH_SIZE:], src=self.hlx, scalar=ly,
                                   repeat_times=2 * 2, dst_blk_stride=1, src_blk_stride=1,
                                   dst_rep_stride=8, src_rep_stride=8)
            else:
                w_fp32 = self.tik.Tensor(dtype="float32", shape=[4 * Constant.BATCH_SIZE],
                                         name="w1234_tmp", scope=tik.scope_ubuf)
                with self.tik.new_stmt_scope(disable_sync=True):
                    self.tik.vmuls(mask=64, dst=w_fp32[0:], src=self.hlx, scalar=hy,
                                   repeat_times=2 * 2, dst_blk_stride=1, src_blk_stride=1,
                                   dst_rep_stride=8, src_rep_stride=8)
                    self.tik.vmuls(mask=64, dst=w_fp32[2 * Constant.BATCH_SIZE:], src=self.hlx, scalar=ly,
                                   repeat_times=2 * 2, dst_blk_stride=1, src_blk_stride=1,
                                   dst_rep_stride=8, src_rep_stride=8)
                # float32 -> float16
                self.tik.vconv(mask=64, round_mode='', dst=self.w1234, src=w_fp32,
                               repeat_times=2 * 4, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=4, src_rep_stride=8)
        elif self.op.dtype == 'float32':
            with self.tik.new_stmt_scope(disable_sync=True):
                # 8*w1|8*w2|8*w3|8*w4|8*w1|8*w2|8*w3|8*w4|...
                self.tik.vmuls(mask=8, dst=self.w1234[0:], src=self.hlx[0, 0], scalar=hy, repeat_times=16,
                               dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=4, src_rep_stride=1)
                self.tik.vmuls(mask=8, dst=self.w1234[8:], src=self.hlx[1, 0], scalar=hy, repeat_times=16,
                               dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=4, src_rep_stride=1)
                self.tik.vmuls(mask=8, dst=self.w1234[16:], src=self.hlx[0, 0], scalar=ly, repeat_times=16,
                               dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=4, src_rep_stride=1)
                self.tik.vmuls(mask=8, dst=self.w1234[24:], src=self.hlx[1, 0], scalar=ly, repeat_times=16,
                               dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=4, src_rep_stride=1)
        else:  # float16
            w_fp32 = self.tik.Tensor(dtype="float32", name="w_fp32", scope=tik.scope_ubuf,
                                     shape=[4 * Constant.BATCH_SIZE])
            with self.tik.new_stmt_scope(disable_sync=True):
                self.tik.vmuls(mask=8, dst=w_fp32[0:], src=self.hlx[0, 0], scalar=hy, repeat_times=16,
                               dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=4, src_rep_stride=1)
                self.tik.vmuls(mask=8, dst=w_fp32[8:], src=self.hlx[1, 0], scalar=hy, repeat_times=16,
                               dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=4, src_rep_stride=1)
                self.tik.vmuls(mask=8, dst=w_fp32[16:], src=self.hlx[0, 0], scalar=ly, repeat_times=16,
                               dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=4, src_rep_stride=1)
                self.tik.vmuls(mask=8, dst=w_fp32[24:], src=self.hlx[1, 0], scalar=ly, repeat_times=16,
                               dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=4, src_rep_stride=1)
            # float32 -> float16
            self.tik.vconv(mask=64, round_mode='', dst=self.w1234, src=w_fp32,
                           repeat_times=2 * 4, dst_blk_stride=1, src_blk_stride=1,
                           dst_rep_stride=4, src_rep_stride=8)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _gather_bilinear_interpolate_4_points(self, dst_ub: tik.Tensor, fm_tensor: tik.Tensor,
                                              fm_offset: tik.Scalar, fm_hwc0: tik.Scalar, fm_row_elements: tik.Scalar,
                                              c1_num: tik.Scalar, x_low: tik.Scalar, x_high: tik.Scalar,
                                              y_low: tik.Scalar, y_high: tik.Scalar,
                                              check_src_burst_gap: bool = True) -> None:
        """gather bilinear interpolate 4 points"""

        def _move_one_row(dst_offset, src_offset):
            if check_src_burst_gap:
                with self.tik.if_scope(burst_src_gap <= Constant.BURST_STRIDE_MAX):
                    self.tik.data_move(dst=dst_ub[dst_offset], src=fm_tensor[src_offset],
                                       sid=0, nburst=c1_num, burst=burst_len,
                                       src_stride=burst_src_gap, dst_stride=burst_dst_gap)
                with self.tik.else_scope():
                    with self.tik.for_range(0, c1_num) as _c1_idx:
                        _dst_offset = dst_offset + _c1_idx * 4 * Constant.C0
                        _src_offset = src_offset + _c1_idx * fm_hwc0
                        self.tik.data_move(dst=dst_ub[_dst_offset], src=fm_tensor[_src_offset],
                                           sid=0, nburst=1, burst=burst_len, src_stride=0, dst_stride=0)
            else:
                self.tik.data_move(dst=dst_ub[dst_offset], src=fm_tensor[src_offset],
                                   sid=0, nburst=c1_num, burst=burst_len,
                                   src_stride=burst_src_gap, dst_stride=burst_dst_gap)

        burst_len = self.tik.Scalar("int32", "burst_len", init_value=2 * self.op.c0_blocks)
        with self.tik.if_scope(x_low == x_high):
            burst_len.set_as(1 * self.op.c0_blocks)
        burst_dst_gap = 4 * self.op.c0_blocks - burst_len
        burst_src_gap = fm_hwc0 // self.op.block_element - burst_len
        with self.tik.new_stmt_scope(disable_sync=True):
            _move_one_row(0, fm_offset)
            with self.tik.if_scope(y_low != y_high):
                _move_one_row(2 * Constant.C0, fm_offset + 1 * fm_row_elements)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _bilinear_interpolate_without_vbi(self, grid_value: tik.Tensor, fm_4_points: tik.Tensor, c1_num: tik.Scalar,
                                          grid_w_idx: tik.Scalar, x_low: tik.Scalar, x_high: tik.Scalar,
                                          y_low: tik.Scalar, y_high: tik.Scalar) -> None:
        op = self.op
        with self.tik.new_stmt_scope():
            w1 = self.tik.Scalar(dtype=op.dtype, name="w1", init_value=self.w1234[grid_w_idx])
            w2 = self.tik.Scalar(dtype=op.dtype, name="w2", init_value=self.w1234[Constant.BATCH_SIZE + grid_w_idx])
            w3 = self.tik.Scalar(dtype=op.dtype, name="w3", init_value=self.w1234[2 * Constant.BATCH_SIZE + grid_w_idx])
            w4 = self.tik.Scalar(dtype=op.dtype, name="w4", init_value=self.w1234[3 * Constant.BATCH_SIZE + grid_w_idx])

            tmp_result = self.tik.Tensor(dtype=op.dtype, shape=[3, c1_num, Constant.C0],
                                         name="tmp_result", scope=tbe_platform.scope_ubuf)
            p2_offset = self.tik.Scalar(dtype="int32", name="p2_offset", init_value=x_high - x_low)
            p3_offset = self.tik.Scalar(dtype="int32", name="p3_offset", init_value=(y_high - y_low) * 2)
            p4_offset = self.tik.Scalar(dtype="int32", name="p4_offset", init_value=p3_offset + p2_offset)

            # c1_num must <= 255 has been ensured when tiling
            mask, repeat = Constant.C0, c1_num
            dst_rep_stride, src_rep_stride = op.c0_blocks, 4 * op.c0_blocks
            with self.tik.new_stmt_scope(disable_sync=True):
                # `v1=p1*w1`
                self.tik.vmuls(mask=mask, dst=grid_value, src=fm_4_points[0:],
                               scalar=w1, repeat_times=repeat, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride)
                # `v2=p2*w2`
                self.tik.vmuls(mask=mask, dst=tmp_result[0, 0, 0], src=fm_4_points[p2_offset * Constant.C0:],
                               scalar=w2, repeat_times=repeat, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride)
                # `v3=p3*w3`
                self.tik.vmuls(mask=mask, dst=tmp_result[1, 0, 0], src=fm_4_points[p3_offset * Constant.C0:],
                               scalar=w3, repeat_times=repeat, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride)
                # `v4=p4*w4`
                self.tik.vmuls(mask=mask, dst=tmp_result[2, 0, 0], src=fm_4_points[p4_offset * Constant.C0:],
                               scalar=w4, repeat_times=repeat, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride)
            with self.tik.new_stmt_scope(disable_sync=True):
                # `v1 + v2`
                self.tik.vadd(mask=mask, dst=grid_value, src0=grid_value, src1=tmp_result[0, 0, 0],
                              repeat_times=repeat, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                              dst_rep_stride=op.c0_blocks, src0_rep_stride=op.c0_blocks,
                              src1_rep_stride=op.c0_blocks)
                # `v3 + v4`
                self.tik.vadd(mask=mask, dst=tmp_result[1, 0, 0], src0=tmp_result[1, 0, 0], src1=tmp_result[2, 0, 0],
                              repeat_times=repeat, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                              dst_rep_stride=op.c0_blocks, src0_rep_stride=op.c0_blocks,
                              src1_rep_stride=op.c0_blocks)
            # `v1 + v2` + `v3 + v4`
            self.tik.vadd(mask=mask, dst=grid_value, src0=grid_value, src1=tmp_result[1, 0, 0],
                          repeat_times=repeat, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                          dst_rep_stride=op.c0_blocks, src0_rep_stride=op.c0_blocks, src1_rep_stride=op.c0_blocks)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _bilinear_interpolate_with_vbi(self, out_y_ub: tik.Tensor, out_w_offset: tik.Scalar, out_w_num: tik.Scalar,
                                       fm_in_ub: tik.Tensor, fm_offset_base: tik.Scalar, fm_in_ub_h: tik.Scalar,
                                       fm_in_ub_w: tik.Scalar, c1_num: tik.Scalar, grid_w_num: tik.Scalar,
                                       y_low: tik.Scalar, y_high: tik.Scalar):
        op = self.op
        aligned_grid_w_num = ceil_div(grid_w_num, 8) * 8  # range: [8, 128]
        rep_times = aligned_grid_w_num // 8  # range: [1, 16]
        src0_offset_backup = self.tik.Tensor(dtype="int32", name="src0_offset_backup", scope=tik.scope_ubuf,
                                             shape=[4, aligned_grid_w_num])
        self._calc_fm_offset_used_in_vbi(dst_offset=src0_offset_backup, fm_offset_base=fm_offset_base,
                                         fm_in_ub_w=fm_in_ub_w, grid_w_num=grid_w_num, y_low=y_low, y_high=y_high)
        vbi_c1_per_loop = self.tik.Scalar(dtype="int32", name="vbi_c1_per_loop", init_value=255 // rep_times)
        vbi_c1_limit = self.tik.Scalar(dtype="int32", name="vbi_c1_limit",
                                       init_value=self.vbi_memory // (self.vbi_mem_factor * aligned_grid_w_num))
        with self.tik.if_scope(vbi_c1_per_loop > vbi_c1_limit):
            vbi_c1_per_loop.set_as(vbi_c1_limit)
        c1_loops = ceil_div(c1_num, vbi_c1_per_loop)
        with self.tik.for_range(0, c1_loops) as _c1_loop_idx:
            c1, c1_offset = self._calc_segment(c1_num, _c1_loop_idx, vbi_c1_per_loop)
            src0_offset = self.tik.Tensor(dtype="int32", name="src0_offset", scope=tik.scope_ubuf,
                                          shape=[c1, 4, aligned_grid_w_num])
            w1234_dup = self.tik.Tensor(dtype=op.dtype, name="w1234_dup", scope=tik.scope_ubuf,
                                        shape=[c1, 4, aligned_grid_w_num])
            grid_value = self.tik.Tensor(dtype=op.dtype, name="grid_value", scope=tik.scope_ubuf,
                                         shape=[c1, aligned_grid_w_num, Constant.C0])
            with self.tik.for_range(0, c1) as _c1_idx:
                self.tik.vadds(mask=32, dst=src0_offset[_c1_idx, 0, 0], src=src0_offset_backup,
                               scalar=(c1_offset + _c1_idx) * fm_in_ub_h * fm_in_ub_w * Constant.C0 * op.data_size,
                               repeat_times=rep_times, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=4, src_rep_stride=4)
                self.tik.vadds(mask=32, dst=w1234_dup[_c1_idx, 0, 0], src=self.w1234, scalar=0.0,
                               repeat_times=rep_times, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=4 if op.dtype == 'float32' else 2,
                               src_rep_stride=4 if op.dtype == 'float32' else 2)
            if op.dtype == 'float32':
                # `C0[0:8]`
                self.tik.vbi(mask=64, dst=grid_value[0:],
                             src0=fm_in_ub, src1=w1234_dup, src0_offset=src0_offset, dst_blk_stride=2,
                             vertical_repeat_times=rep_times * c1,
                             horizontal_repeat_times=4, repeat_mode=1,
                             vertical_repeat_offset=64 * 2)
                # update src0_offset
                self.tik.vadds(mask=32, dst=src0_offset, src=src0_offset,
                               scalar=op.data_size * Constant.C0 // 2,
                               repeat_times=rep_times * c1, dst_blk_stride=1, src_blk_stride=1,
                               dst_rep_stride=4, src_rep_stride=4)
                # `C0[8:16]`
                self.tik.vbi(mask=64, dst=grid_value[8:],
                             src0=fm_in_ub, src1=w1234_dup, src0_offset=src0_offset, dst_blk_stride=2,
                             vertical_repeat_times=rep_times * c1,
                             horizontal_repeat_times=4, repeat_mode=1,
                             vertical_repeat_offset=64 * 2)
            else:  # float16. actually hardware vbi only supports float16.
                self.tik.vbi(mask=128, dst=grid_value,
                             src0=fm_in_ub, src1=w1234_dup, src0_offset=src0_offset, dst_blk_stride=1,
                             vertical_repeat_times=rep_times * c1,
                             horizontal_repeat_times=4, repeat_mode=1,
                             vertical_repeat_offset=128)
            # add grid value together
            self._pool_output_value_with_vbi(out_y_ub=out_y_ub, grid_value=grid_value, c1_offset=c1_offset, c1_num=c1,
                                             out_w_offset=out_w_offset, out_w_num=out_w_num, grid_w_num=grid_w_num)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _calc_fm_offset_used_in_vbi(self, dst_offset: tik.Tensor, fm_offset_base: tik.Scalar, fm_in_ub_w: tik.Scalar,
                                    grid_w_num: tik.Scalar, y_low: tik.Scalar, y_high: tik.Scalar) -> None:
        """
        calculate feature map offset used in vbi
        ----
        offset for w1 = (fm_offset_base + x_low) * C0 * data_size,
        offset for w2 = (fm_offset_base + x_high) * C0 * data_size,
        offset for w3 = (fm_offset_base + 1 * op_feature_w + x_low) * C0 * data_size,
        offset for w4 = (fm_offset_base + 1 * op_feature_w + x_high) * C0 * data_size.
        """
        op = self.op
        rep_times = ceil_div(grid_w_num, 8)
        # initialize in case of tensor overlap
        self.tik.vadds(mask=8, dst=dst_offset[0:], src=self.x_lh[0, 0], scalar=fm_offset_base,
                       repeat_times=rep_times, dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=4, src_rep_stride=1)
        self.tik.vadds(mask=8, dst=dst_offset[8:], src=self.x_lh[1, 0], scalar=fm_offset_base,
                       repeat_times=rep_times, dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=4, src_rep_stride=1)
        tmp_scalar = self.tik.Scalar(dtype="int32", name="tmp_scalar", init_value=fm_offset_base)
        with self.tik.if_scope(y_low != y_high):
            tmp_scalar.set_as(fm_offset_base + fm_in_ub_w)
        self.tik.vadds(mask=8, dst=dst_offset[16:], src=self.x_lh[0, 0], scalar=tmp_scalar,
                       repeat_times=rep_times, dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=4, src_rep_stride=1)
        self.tik.vadds(mask=8, dst=dst_offset[24:], src=self.x_lh[1, 0], scalar=tmp_scalar,
                       repeat_times=rep_times, dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=4, src_rep_stride=1)
        self.tik.vmuls(mask=32, dst=dst_offset, src=dst_offset, scalar=Constant.C0 * op.data_size,
                       repeat_times=rep_times, dst_blk_stride=1, src_blk_stride=1,
                       dst_rep_stride=4, src_rep_stride=4)
        # reset tail to 0 in case of tensor overlap
        start_1, start_2 = grid_w_num // 8, grid_w_num % 8
        for idx in range(0, 4):
            with self.tik.for_range(start_2, 8) as inner_idx:
                dst_offset[4 * 8 * start_1 + idx * 8 + inner_idx] = 0

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _pool_output_value_with_vbi(self, out_y_ub: tik.Tensor, grid_value: tik.Tensor, c1_offset: tik.Scalar,
                                    c1_num: tik.Scalar, out_w_offset: tik.Scalar, out_w_num: tik.Scalar,
                                    grid_w_num: tik.Scalar) -> None:
        """add grid value together in 'AVG' scenario"""

        relative_out_w = self.tik.Scalar(dtype="int64", name="relative_out_w")
        relative_out_w_max = self.tik.Scalar(dtype="int64", name="relative_out_w_max", init_value=out_w_num - 1)
        aligned_grid_w_num = ceil_div(grid_w_num, 8) * 8
        with self.tik.for_range(0, grid_w_num) as _grid_w_idx:
            relative_out_w.set_as(self.pool_w_idx[_grid_w_idx])
            relative_out_w.set_as(relative_out_w - out_w_offset)
            # in case of memory out of range due to coordinate calculation error
            self.tik.scalar_max(relative_out_w, relative_out_w, 0)
            self.tik.scalar_min(relative_out_w, relative_out_w, relative_out_w_max)
            dst_rep_stride = out_w_num * self.op.c0_blocks
            self.pool_func(mask=Constant.C0, dst=out_y_ub[c1_offset, relative_out_w, 0],
                           src0=grid_value[0, _grid_w_idx, 0], src1=out_y_ub[c1_offset, relative_out_w, 0],
                           repeat_times=c1_num, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                           dst_rep_stride=dst_rep_stride,
                           src0_rep_stride=aligned_grid_w_num * self.op.c0_blocks, src1_rep_stride=dst_rep_stride)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments

    def _pool_output_value_without_vbi(self, out_y_ub: tik.Tensor, grid_value: tik.Tensor, c1_num: tik.Scalar,
                                       out_w_offset: tik.Scalar, out_w_num: tik.Scalar, grid_w_idx: tik.Scalar) -> None:
        relative_out_w = self.tik.Scalar(dtype="int64", name="relative_out_w", init_value=self.pool_w_idx[grid_w_idx])
        relative_out_w.set_as(relative_out_w - out_w_offset)
        # in case of memory out of range due to coordinate calculation error
        relative_out_w_max = self.tik.Scalar(dtype="int64", name="relative_out_w_max", init_value=out_w_num - 1)
        self.tik.scalar_max(relative_out_w, relative_out_w, 0)
        self.tik.scalar_min(relative_out_w, relative_out_w, relative_out_w_max)
        rep_stride = out_w_num * self.op.c0_blocks  # out_w_num < 128 has been ensured when tiling
        self.pool_func(mask=Constant.C0, dst=out_y_ub[0, relative_out_w, 0],
                       src0=out_y_ub[0, relative_out_w, 0], src1=grid_value,
                       repeat_times=c1_num, dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                       dst_rep_stride=rep_stride, src0_rep_stride=rep_stride, src1_rep_stride=self.op.c0_blocks)

    def _cache_whole_feature_map(self, cache_dst: tik.Tensor) -> None:
        """
        Whole feature map GM -> UB/L1.
        """
        with self.tik.if_scope(self.cache_fm_idx != self.cur_fm_idx):
            op = self.op
            burst_len = self.feature_c1hwc0 // op.block_element
            gm_offset = (self.cur_fm_idx * op.c1 + op.core_c1_offset) * op.feature_h * op.feature_w * Constant.C0
            self.tik.data_move(dst=cache_dst,
                               src=op.features_gm[gm_offset],
                               sid=0, nburst=1, burst=burst_len,
                               src_stride=0, dst_stride=0)
            self.cache_fm_idx.set_as(self.cur_fm_idx)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _move_fm_to_ub(self, dst: tik.Tensor, src: tik.Tensor, src_offset: tik.Scalar, c1_num: tik.Scalar,
                       burst_len: tik.Scalar, src_hwc0: tik.Scalar, dst_hwc0: tik.Scalar,
                       check_src_gap: bool = True):
        op = self.op
        burst_dst_gap = dst_hwc0 // op.block_element - burst_len
        burst_src_gap = src_hwc0 // op.block_element - burst_len
        with self.tik.new_stmt_scope(disable_sync=True):
            if check_src_gap:
                with self.tik.if_scope(burst_src_gap <= Constant.BURST_STRIDE_MAX):
                    self.tik.data_move(dst=dst, src=src[src_offset],
                                       sid=0, nburst=c1_num, burst=burst_len,
                                       src_stride=burst_src_gap, dst_stride=burst_dst_gap)
                with self.tik.else_scope():
                    with self.tik.for_range(0, c1_num) as _c1_idx:
                        _dst_offset = _c1_idx * dst_hwc0
                        _src_offset = src_offset + _c1_idx * src_hwc0
                        self.tik.data_move(dst=dst[_dst_offset], src=src[_src_offset], sid=0, nburst=1, burst=burst_len,
                                           src_stride=burst_src_gap, dst_stride=0)
            else:
                self.tik.data_move(dst=dst, src=src[src_offset], sid=0, nburst=c1_num, burst=burst_len,
                                   src_stride=burst_src_gap, dst_stride=burst_dst_gap)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _gather_4_points_from_gm(self, dst_ub: tik.Tensor, relative_c1_offset: tik.Scalar,
                                 absolute_c1_offset: tik.Scalar, c1_num: tik.Scalar,
                                 x_low: tik.Scalar, x_high: tik.Scalar,
                                 y_low: tik.Scalar, y_high: tik.Scalar) -> None:
        op = self.op
        fm_hwc0 = op.feature_h * op.feature_w * Constant.C0
        fm_row_elements = op.feature_w * Constant.C0
        if op.l1_size_bytes > 0:
            with self.tik.if_scope(self.cache_all_fm_in_l1 == 1):
                # no need to check burst_src_gap vs 65535.
                fm_offset = ((relative_c1_offset * op.feature_h + y_low) * op.feature_w + x_low) * Constant.C0
                self._gather_bilinear_interpolate_4_points(dst_ub=dst_ub, fm_tensor=self.fm_in_l1,
                                                           fm_offset=fm_offset, fm_hwc0=fm_hwc0,
                                                           fm_row_elements=fm_row_elements, c1_num=c1_num,
                                                           x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high,
                                                           check_src_burst_gap=False)
            with self.tik.else_scope():
                fm_offset = (((self.cur_fm_idx * op.c1 + absolute_c1_offset
                               ) * op.feature_h + y_low) * op.feature_w + x_low) * Constant.C0
                self._gather_bilinear_interpolate_4_points(dst_ub=dst_ub, fm_tensor=op.features_gm,
                                                           fm_offset=fm_offset, fm_hwc0=fm_hwc0,
                                                           fm_row_elements=fm_row_elements, c1_num=c1_num,
                                                           x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high,
                                                           check_src_burst_gap=True)
        else:
            fm_offset = (((self.cur_fm_idx * op.c1 + absolute_c1_offset
                           ) * op.feature_h + y_low) * op.feature_w + x_low) * Constant.C0
            self._gather_bilinear_interpolate_4_points(dst_ub=dst_ub, fm_tensor=op.features_gm,
                                                       fm_offset=fm_offset, fm_hwc0=fm_hwc0,
                                                       fm_row_elements=fm_row_elements, c1_num=c1_num,
                                                       x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high,
                                                       check_src_burst_gap=True)

    def _calc_vbi_memory(self):
        op = self.op
        # calculate memory for vbi first. this instruction is heavy. less will be better.
        aligned_grid_w_max = self.tik.Scalar(dtype="int32", name="aligned_grid_w_max",
                                             init_value=ceil_div(self.cur_roi_sample_w * op.out_w_per_loop, 8) * 8)
        with self.tik.if_scope(aligned_grid_w_max > Constant.BATCH_SIZE):
            aligned_grid_w_max.set_as(Constant.BATCH_SIZE)
        # vbi vertical_repeat_times range: [1, 255]
        vbi_rep_times = aligned_grid_w_max // 8
        vbi_c1_per_loop = self.tik.Scalar(dtype="int32", name="vbi_c1_per_loop", init_value=255 // vbi_rep_times)
        with self.tik.if_scope(vbi_c1_per_loop > op.out_c1_per_loop):
            vbi_c1_per_loop.set_as(op.out_c1_per_loop)

        self.vbi_memory.set_as(vbi_c1_per_loop * aligned_grid_w_max * self.vbi_mem_factor)
        with self.tik.if_scope(self.vbi_memory > self.memory_for_vbi_max):
            vbi_c1_per_loop.set_as(self.memory_for_vbi_max // (aligned_grid_w_max * self.vbi_mem_factor))
            self.vbi_memory.set_as(vbi_c1_per_loop * aligned_grid_w_max * self.vbi_mem_factor)


class MoveWholeFeatureMapToUb(ProcessorBase):
    def __init__(self, op) -> None:
        super().__init__(op)
        self.fm_in_ub = self.tik.Tensor(dtype=op.dtype, name="fm_in_ub", scope=tik.scope_ubuf,
                                        shape=[op.c1_per_core, op.feature_h, op.feature_w, Constant.C0])

    def tiling_compute_one_roi(self, roi_idx: tik.Scalar, c1_offset: tik.Scalar, c1_num: tik.Scalar) -> None:
        op = self.op
        # move whole feature map to UB
        self._cache_whole_feature_map(self.fm_in_ub)
        if op.support_vbi:
            self.memory_for_vbi_max.set_as(op.max_ub_element * op.data_size + Constant.RESERVED_UB_BYTES_FOR_OUT_MIN -
                                           op.out_w_per_loop * op.out_c1_per_loop * Constant.C0 * op.data_size -
                                           self.feature_c1hwc0 * op.data_size)
            self._calc_vbi_memory()
        self.tiling_compute_output(self.fm_in_ub, c1_offset, c1_num)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def bilinear_interpolate_one_grid_row(self, fm_in_ub: tik.Tensor, out_y_ub: tik.Tensor,
                                          relative_c1_offset: tik.Scalar, absolute_c1_offset: tik.Scalar,
                                          c1_num: tik.Scalar, out_w_offset: tik.Scalar, out_w_num: tik.Scalar,
                                          y_low: tik.Scalar, y_high: tik.Scalar, grid_h_idx: tik.Scalar,
                                          grid_w_num: tik.Scalar) -> None:
        op = self.op
        fm_offset_base = (relative_c1_offset * op.feature_h + y_low) * op.feature_w
        # calculate bilinear interpolation each row.
        self._calc_bilinear_interpolate_weight(grid_h_idx)
        if op.support_vbi:
            self._bilinear_interpolate_with_vbi(out_y_ub=out_y_ub, out_w_offset=out_w_offset, out_w_num=out_w_num,
                                                fm_in_ub=fm_in_ub, fm_offset_base=fm_offset_base,
                                                fm_in_ub_h=op.feature_h, fm_in_ub_w=op.feature_w,
                                                c1_num=c1_num, grid_w_num=grid_w_num, y_low=y_low, y_high=y_high)
        else:
            # calculate sample point one by one
            fm_4_points = self.tik.Tensor(dtype=op.dtype, name="fm_4_points", scope=tik.scope_ubuf,
                                          shape=[c1_num, 2, 2, Constant.C0])
            grid_value = self.tik.Tensor(dtype=op.dtype, name="grid_value", scope=tik.scope_ubuf,
                                         shape=[c1_num, Constant.C0])
            with self.tik.for_range(0, grid_w_num) as _grid_w_idx:
                # calculate sample grid one by one
                x_low = self.tik.Scalar(dtype="int32", name="x_low", init_value=self.x_lh[0, _grid_w_idx])
                x_high = self.tik.Scalar(dtype="int32", name="x_high", init_value=self.x_lh[1, _grid_w_idx])
                # no need to check `fm_hwc0 <= 65535`, bcz: c1*fh*fw*c0*data_size < 256KB. max(fh*fw) = 8192
                fm_hwc0 = op.feature_h * op.feature_w * Constant.C0
                fm_offset = (fm_offset_base + x_low) * Constant.C0
                self._gather_bilinear_interpolate_4_points(dst_ub=fm_4_points, fm_tensor=fm_in_ub,
                                                           fm_offset=fm_offset, fm_hwc0=fm_hwc0,
                                                           fm_row_elements=op.feature_w * Constant.C0,
                                                           c1_num=c1_num, x_low=x_low, x_high=x_high,
                                                           y_low=y_low, y_high=y_high, check_src_burst_gap=False)
                self._bilinear_interpolate_without_vbi(grid_value=grid_value, fm_4_points=fm_4_points,
                                                       c1_num=c1_num, grid_w_idx=_grid_w_idx,
                                                       x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high)
                self._pool_output_value_without_vbi(out_y_ub=out_y_ub, grid_value=grid_value, c1_num=c1_num,
                                                    out_w_offset=out_w_offset, out_w_num=out_w_num,
                                                    grid_w_idx=_grid_w_idx)


class MoveFeatureMapMultipleRows(ProcessorBase):
    def __init__(self, op) -> None:
        super().__init__(op)
        self.fm_in_ub = None
        # move fm of roi region rows batch to ub rather than whole roi region
        self.roi_rows_to_ub_batch = self.tik.Scalar(dtype="int32", name="roi_rows_to_ub_batch")
        self.cache_c1 = self.tik.Scalar(dtype="int64", name="cache_c1")
        self.cache_y0_low = self.tik.Scalar(dtype="int32", name="cache_y0_low")
        self.cache_y1_high = self.tik.Scalar(dtype="int32", name="cache_y1_high")
        if op.support_vbi:
            self.memory_for_vbi_max.set_as(op.max_ub_element * op.data_size + Constant.RESERVED_UB_BYTES_FOR_OUT_MIN -
                                           op.out_w_per_loop * op.out_c1_per_loop * Constant.C0 * op.data_size -
                                           2 * op.out_c1_per_loop * op.feature_w * Constant.C0 * op.data_size)
        else:
            self.memory_for_vbi_max.set_as(op.max_ub_element * op.data_size + Constant.RESERVED_UB_BYTES_FOR_OUT_MIN -
                                           op.out_w_per_loop * op.out_c1_per_loop * Constant.C0 * op.data_size -
                                           2 * op.out_c1_per_loop * op.feature_w * Constant.C0 * op.data_size -
                                           op.out_c1_per_loop * Constant.TEMP_C1xC0_COUNT * Constant.C0 * op.data_size)

    def tiling_compute_one_roi(self, roi_idx: tik.Scalar, c1_offset: tik.Scalar, c1_num: tik.Scalar) -> None:
        op = self.op
        # move whole feature map to L1
        if op.l1_size_bytes > 0:
            with self.tik.if_scope(self.cache_all_fm_in_l1 == 1):
                self._cache_whole_feature_map(self.fm_in_l1)
        self.cache_c1.set_as(-1)
        if op.support_vbi:
            self._calc_vbi_memory()
        # input height batch size
        self.roi_rows_to_ub_batch.set_as(
            (self.memory_for_vbi_max - self.vbi_memory +
             2 * op.out_c1_per_loop * op.feature_w * Constant.C0 * op.data_size) //
            (op.data_size * Constant.C0) // (op.out_c1_per_loop * op.feature_w)
        )
        self.fm_in_ub = self.tik.Tensor(dtype=op.dtype, name="fm_in_ub", scope=tik.scope_ubuf,
                                        shape=[op.out_c1_per_loop, self.roi_rows_to_ub_batch,
                                               op.feature_w, Constant.C0])
        self.tiling_compute_output(self.fm_in_ub, c1_offset, c1_num)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def bilinear_interpolate_one_grid_row(self, fm_in_ub: tik.Tensor, out_y_ub: tik.Tensor,
                                          relative_c1_offset: tik.Scalar, absolute_c1_offset: tik.Scalar,
                                          c1_num: tik.Scalar, out_w_offset: tik.Scalar, out_w_num: tik.Scalar,
                                          y_low: tik.Scalar, y_high: tik.Scalar, grid_h_idx: tik.Scalar,
                                          grid_w_num: tik.Scalar) -> None:
        op = self.op
        self._cache_fm_rows_to_ub(fm_in_ub, relative_c1_offset, absolute_c1_offset, c1_num, y_low, y_high)
        fm_offset_base = (y_low - self.cache_y0_low) * op.feature_w
        # calculate bilinear interpolation each row.
        self._calc_bilinear_interpolate_weight(grid_h_idx)
        if op.support_vbi:
            self._bilinear_interpolate_with_vbi(out_y_ub=out_y_ub, out_w_offset=out_w_offset, out_w_num=out_w_num,
                                                fm_in_ub=fm_in_ub, fm_offset_base=fm_offset_base,
                                                fm_in_ub_h=self.roi_rows_to_ub_batch,
                                                fm_in_ub_w=op.feature_w,
                                                c1_num=c1_num, grid_w_num=grid_w_num, y_low=y_low, y_high=y_high)
        else:
            # calculate sample point one by one
            fm_4_points = self.tik.Tensor(dtype=op.dtype, name="fm_4_points", scope=tik.scope_ubuf,
                                          shape=[c1_num, 2, 2, Constant.C0])
            grid_value = self.tik.Tensor(dtype=op.dtype, name="grid_value", scope=tik.scope_ubuf,
                                         shape=[c1_num, Constant.C0])
            with self.tik.for_range(0, grid_w_num) as _grid_w_idx:
                # calculate sample grid one by one
                x_low = self.tik.Scalar(dtype="int32", name="x_low", init_value=self.x_lh[0, _grid_w_idx])
                x_high = self.tik.Scalar(dtype="int32", name="x_high", init_value=self.x_lh[1, _grid_w_idx])
                # no need to check `fm_hwc0 <= 65535`, bcz: c1*h*w*c0*data_size < 256KB. max(h*w) = 8192
                fm_hwc0 = self.roi_rows_to_ub_batch * op.feature_w * Constant.C0
                fm_offset = (fm_offset_base + x_low) * Constant.C0
                self._gather_bilinear_interpolate_4_points(dst_ub=fm_4_points, fm_tensor=fm_in_ub,
                                                           fm_offset=fm_offset, fm_hwc0=fm_hwc0,
                                                           fm_row_elements=op.feature_w * Constant.C0,
                                                           c1_num=c1_num, x_low=x_low, x_high=x_high,
                                                           y_low=y_low, y_high=y_high, check_src_burst_gap=False)
                self._bilinear_interpolate_without_vbi(grid_value, fm_4_points, c1_num,
                                                       _grid_w_idx, x_low, x_high, y_low, y_high)
                self._pool_output_value_without_vbi(out_y_ub=out_y_ub, grid_value=grid_value, c1_num=c1_num,
                                                    out_w_offset=out_w_offset, out_w_num=out_w_num,
                                                    grid_w_idx=_grid_w_idx)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _cache_fm_rows_to_ub(self, dst_ub: tik.Tensor,
                             relative_c1_offset: tik.Scalar, absolute_c1_offset: tik.Scalar, c1_num: tik.Scalar,
                             y_low: tik.Scalar, y_high: tik.Scalar) -> None:
        """
        Move fm rows to ub.
        ----------
        dst_ub: tik.Tensor, destination ub to move in
        relative_c1_offset : tik.Scalar, c1 offset relative to beginning of c1 current core need to process
        absolute_c1_offset : tik.Scalar, absolute c1 offset of feature map
        c1_num: tik.Scalar, number of c1
        y_low : tik.Scalar, current row index of feature map
        y_high : tik.Scalar, current row index of feature map
        """
        op = self.op
        with self.tik.if_scope(tik.any(self.cache_c1 != absolute_c1_offset,
                                       self.cache_y1_high < y_high,
                                       self.cache_y0_low > y_low)):
            rows = self.tik.Scalar(dtype="int64", name="rows", init_value=self.roi_rows_to_ub_batch)
            with self.tik.if_scope(y_low + rows > op.feature_h):
                rows.set_as(op.feature_h - y_low)
            self.cache_c1.set_as(absolute_c1_offset)
            self.cache_y1_high.set_as(y_low + rows - 1)
            self.cache_y0_low.set_as(y_low)
            # no need to check burst_len vs 65535.
            burst_len = rows * op.feature_w * op.c0_blocks
            src_hwc0 = op.feature_h * op.feature_w * Constant.C0
            dst_hwc0 = self.roi_rows_to_ub_batch * op.feature_w * Constant.C0
            if op.l1_size_bytes > 0:
                with self.tik.if_scope(self.cache_all_fm_in_l1 == 1):
                    # no need to check burst_src_gap vs 65535.
                    gm_offset = (((0 * op.c1 + relative_c1_offset) * op.feature_h + y_low) * op.feature_w) * Constant.C0
                    self._move_fm_to_ub(dst_ub, self.fm_in_l1, gm_offset, c1_num, burst_len, src_hwc0, dst_hwc0, False)
                with self.tik.else_scope():
                    # need to check burst_src_gap vs 65535.
                    gm_offset = (((self.cur_fm_idx * op.c1 + absolute_c1_offset) * op.feature_h + y_low
                                  ) * op.feature_w) * Constant.C0
                    self._move_fm_to_ub(dst_ub, op.features_gm, gm_offset, c1_num, burst_len, src_hwc0, dst_hwc0, True)
            else:
                # need to check burst_src_gap vs 65535.
                gm_offset = (((self.cur_fm_idx * op.c1 + absolute_c1_offset) * op.feature_h + y_low
                              ) * op.feature_w) * Constant.C0
                self._move_fm_to_ub(dst_ub, op.features_gm, gm_offset, c1_num, burst_len, src_hwc0, dst_hwc0, True)


class MoveFeatureMap4Points(ProcessorBase):
    def __init__(self, op) -> None:
        super().__init__(op)

    def tiling_compute_one_roi(self, roi_idx: tik.Scalar, c1_offset: tik.Scalar, c1_num: tik.Scalar) -> None:
        op = self.op
        # move whole feature map to L1
        if op.l1_size_bytes > 0:
            with self.tik.if_scope(self.cache_all_fm_in_l1 == 1):
                self._cache_whole_feature_map(self.fm_in_l1)
        self.tiling_compute_output(None, c1_offset, c1_num)

    def bilinear_interpolate_one_grid_row(self, fm_in_ub: tik.Tensor, out_y_ub: tik.Tensor,
                                          relative_c1_offset: tik.Scalar,
                                          absolute_c1_offset: tik.Scalar, c1_num: tik.Scalar,
                                          out_w_offset: tik.Scalar, out_w_num: tik.Scalar,
                                          y_low: tik.Scalar, y_high: tik.Scalar,
                                          grid_h_idx: tik.Scalar, grid_w_num: tik.Scalar) -> None:
        op = self.op
        # calculate bilinear interpolation each row.
        self._calc_bilinear_interpolate_weight(grid_h_idx, disable_vbi=True)
        # calculate sample point one by one
        fm_4_points = self.tik.Tensor(dtype=op.dtype, name="fm_4_points", scope=tik.scope_ubuf,
                                      shape=[c1_num, 2, 2, Constant.C0])
        grid_value = self.tik.Tensor(dtype=op.dtype, name="grid_value", scope=tik.scope_ubuf,
                                     shape=[c1_num, Constant.C0])
        with self.tik.for_range(0, grid_w_num) as _grid_w_idx:
            # calculate sample grid one by one
            x_low = self.tik.Scalar(dtype="int32", name="x_low", init_value=self.x_lh[0, _grid_w_idx])
            x_high = self.tik.Scalar(dtype="int32", name="x_high", init_value=self.x_lh[1, _grid_w_idx])
            self._gather_4_points_from_gm(fm_4_points, relative_c1_offset,
                                          absolute_c1_offset, c1_num, x_low, x_high, y_low, y_high)
            self._bilinear_interpolate_without_vbi(grid_value, fm_4_points, c1_num,
                                                   _grid_w_idx, x_low, x_high, y_low, y_high)
            self._pool_output_value_without_vbi(out_y_ub=out_y_ub, grid_value=grid_value, c1_num=c1_num,
                                                out_w_offset=out_w_offset, out_w_num=out_w_num,
                                                grid_w_idx=_grid_w_idx)
