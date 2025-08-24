# Copyright 2021 Huawei Technologies Co., Ltd
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
roi_align_true
"""

import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform


class RoiAlign(object):
    """
    roi_align op
    """
    def __init__(self, feature_map_dict, rois_dict, roisn_dict, output, scale, pool_h, pool_w, sample_ratio=2,
                 roi_end_mode=0, pool_mode='avg', kernel_name="roi_align"):
        self.dtype = feature_map_dict.get("dtype")
        self.roi_shape = rois_dict.get("shape")
        self.fm_shape = feature_map_dict.get("shape")
        self.out_shape = output.get("shape")
        self.roisn_exist = True if roisn_dict else False
        self.scale = scale
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.sample_ratio = sample_ratio
        self.roi_end_mode = roi_end_mode
        self.pool_mode = pool_mode
        self.kernel_name = kernel_name
        self.roi_offset = -0.5 if self.roi_end_mode else 0
        self.channels = self.fm_shape[1]
        self.height = self.fm_shape[2]
        self.width = self.fm_shape[3]
        self.roi_num = self.roi_shape[0]
        self.proposal_num = self.roi_shape[1]
        self.c0 = 16
        self.block_byte_size = 32
        self.c1hwc0 = self.channels * self.height * self.width * self.c0
        self.hwc0 = 2 * self.channels * self.width * self.c0

        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.avg_roi = math.ceil(self.roi_num / self.core_num)
        self.block_num = math.ceil(self.roi_num / self.avg_roi)
        self.last_roi = self.roi_num - self.avg_roi * (self.block_num - 1)
        self.dtype_byte_size = self.get_dtype_size(self.dtype)
        self.data_each_block = self.block_byte_size // self.dtype_byte_size
        self.c_burst = self.c0 // self.data_each_block
        self.max_mask = 64
        self.tiling_mode = 0

        self.fm_gm = self.tik_instance.Tensor(self.dtype, self.fm_shape, scope=tik.scope_gm, name="fm_gm")
        self.roi_gm = self.tik_instance.Tensor(self.dtype, self.roi_shape, scope=tik.scope_gm, name="roi_gm")
        self.out_gm = self.tik_instance.Tensor(self.dtype, self.out_shape, scope=tik.scope_gm, name="out_gm")
        if self.roisn_exist:
            roisn_dtype = roisn_dict.get("dtype", "int32")
            roisn_shape = roisn_dict.get("shape", (self.roi_shape[0],))
            self.roin_gm = self.tik_instance.Tensor(roisn_dtype, roisn_shape, scope=tik.scope_gm, name="roin_gm")
        if self.pool_mode == 'avg':
            self.out_init_val = 0.0
        else:
            self.out_init_val = -65504 if self.dtype == 'float16' else -3.4028235e+38

    class CommonScalar(object):
        """
        define some scalar
        """

        def __init__(self, tik_instance):
            self.start_h = tik_instance.Scalar("float32")
            self.start_w = tik_instance.Scalar("float32")
            self.grid_h = tik_instance.Scalar("int32")
            self.grid_w = tik_instance.Scalar("int32")
            self.size_h = tik_instance.Scalar("float32")
            self.size_w = tik_instance.Scalar("float32")
            self.fm_batch = tik_instance.Scalar("int32")
            self.val_n = tik_instance.Scalar("int32")
            self.x = tik_instance.Scalar("float32")
            self.y = tik_instance.Scalar("float32")
            self.x_low = tik_instance.Scalar("int32")
            self.y_low = tik_instance.Scalar("int32")
            self.x_high = tik_instance.Scalar("int32")
            self.y_high = tik_instance.Scalar("int32")
            self.w1 = tik_instance.Scalar("float32")
            self.w2 = tik_instance.Scalar("float32")
            self.w3 = tik_instance.Scalar("float32")
            self.w4 = tik_instance.Scalar("float32")

    def compute(self):
        """
        compute func
        """
        with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as block_idx:
            with self.tik_instance.if_scope(block_idx < self.block_num - 1):
                self.compute_per_core(self.avg_roi, block_idx)
            with self.tik_instance.else_scope():
                self.compute_per_core(self.last_roi, block_idx)

        inputs_list = [self.fm_gm, self.roi_gm]
        if self.roisn_exist:
            inputs_list.append(self.roin_gm)
        self.tik_instance.BuildCCE(inputs=inputs_list, outputs=[self.out_gm],
                                   kernel_name=self.kernel_name)
        return self.tik_instance

    def extract_roi(self, roi_ub, batch_ub, x0_ub, y0_ub, x1_ub, y1_ub, roi_num):
        """
        extract roi
        """
        with self.tik_instance.for_range(0, roi_num) as idx:
            batch_ub[idx].set_as(roi_ub[idx, 0])
            x0_ub[idx].set_as(roi_ub[idx, 1])
            y0_ub[idx].set_as(roi_ub[idx, 2])
            x1_ub[idx].set_as(roi_ub[idx, 3])
            y1_ub[idx].set_as(roi_ub[idx, 4])

    def scale_roi(self, x0_ub, y0_ub, x1_ub, y1_ub, roi_num):
        """
        scale roi
        """
        self.data_muls(x0_ub, x0_ub, self.scale, [0, 0], num=roi_num)
        self.data_muls(y0_ub, y0_ub, self.scale, [0, 0], num=roi_num)
        self.data_muls(x1_ub, x1_ub, self.scale, [0, 0], num=roi_num)
        self.data_muls(y1_ub, y1_ub, self.scale, [0, 0], num=roi_num)
        if self.roi_end_mode:
            self.data_adds(x0_ub, x0_ub, self.roi_offset, [0, 0], num=roi_num)
            self.data_adds(y0_ub, y0_ub, self.roi_offset, [0, 0], num=roi_num)
            self.data_adds(x1_ub, x1_ub, self.roi_offset, [0, 0], num=roi_num)
            self.data_adds(y1_ub, y1_ub, self.roi_offset, [0, 0], num=roi_num)

    def get_bin_size(self, x0_ub, y0_ub, x1_ub, y1_ub, one_ub, roi_num):
        """
        get bin size
        """
        self.data_sub(x1_ub, x1_ub, x0_ub, [0, 0, 0], num=roi_num)
        self.data_sub(y1_ub, y1_ub, y0_ub, [0, 0, 0], num=roi_num)

        if self.roi_end_mode < 2:
            self.data_max(x1_ub, x1_ub, one_ub, [0, 0, 0], num=roi_num)
            self.data_max(y1_ub, y1_ub, one_ub, [0, 0, 0], num=roi_num)

        self.data_muls(x1_ub, x1_ub, 1 / self.pool_w, [0, 0])
        self.data_muls(y1_ub, y1_ub, 1 / self.pool_h, [0, 0])

    def get_roi_grid(self, grid_w, grid_h, x1_ub, y1_ub, roi_num):
        """
        get number of roi grid
        """
        if self.sample_ratio:
            self.dup_value(grid_w, roi_num, self.sample_ratio)
            self.dup_value(grid_h, roi_num, self.sample_ratio)
        else:
            self.data_conv(grid_w, x1_ub, [0, 0], num=roi_num)
            self.data_conv(grid_h, y1_ub, [0, 0], num=roi_num)

    def get_count(self, count_ub, grid_w, grid_h, one_ub, roi_num):
        """
        get number of grid point
        """
        self.data_mul(count_ub, grid_w, grid_h, [0, 0, 0], num=roi_num)
        self.data_max(count_ub, count_ub, one_ub, [0, 0, 0], num=roi_num)

    def l1_support(self):
        """
        check if support l1 buffer or not
        :return:
        """
        soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
        if soc_version == "Ascend910B":
            return 0
        else:
            return 1

    def get_tiling_mode(self):
        """
        get tiling mode by input shape
        """
        left_ub = self.tik_instance.get_available_buffer_size(tik.scope_ubuf)
        left_size = left_ub // self.get_dtype_size(self.dtype)
        l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
        l1_num = l1_size // self.get_dtype_size(self.dtype)
        l1_support = self.l1_support()

        bilinear_size = self.channels * self.pool_w * self.c0 + 8 * self.channels * self.c0
        if self.c1hwc0 + bilinear_size < left_size:
            return 0

        if self.c1hwc0 < l1_num and bilinear_size < left_size and l1_support:
            return 1

        if self.hwc0 + bilinear_size < left_size:
            return 2

        if self.hwc0 < l1_num and bilinear_size < left_size and l1_support:
            return 3
        return 4

    def set_roi(self, scalar, x0_ub, y0_ub, x1_ub, y1_ub, grid_w_ub, grid_h_ub, batch_ub, count_ub, n):
        """
        set roi parameter to scalar
        """
        scalar.start_w.set_as(x0_ub[n])
        scalar.start_h.set_as(y0_ub[n])
        scalar.grid_w.set_as(grid_w_ub[n])
        scalar.grid_h.set_as(grid_h_ub[n])
        scalar.size_w.set_as(x1_ub[n])
        scalar.size_h.set_as(y1_ub[n])
        scalar.fm_batch.set_as(batch_ub[n])
        scalar.val_n.set_as(count_ub[n])

    def get_idx(self, index, index_low, index_high, edge):
        """
        get idx of bilinear interpolate
        """
        with self.tik_instance.if_scope(index < 0):
            index.set_as(0)

        self.tik_instance.scalar_conv("floor", index_low, index)
        with self.tik_instance.if_scope(index_low >= edge - 1):
            index_high.set_as(edge - 1)
            index_low.set_as(edge - 1)
            index.set_as(edge - 1)
        with self.tik_instance.else_scope():
            index_high.set_as(index_low + 1)

    def get_weight(self, x, y, x_low, y_low, w1, w2, w3, w4):
        """
        get weight
        """
        ly = y - y_low
        lx = x - x_low
        hy = 1 - ly
        hx = 1 - lx

        w1.set_as(hy * hx)
        w2.set_as(hy * lx)
        w3.set_as(ly * hx)
        w4.set_as(ly * lx)

    def load_fm_h(self, fm_cache, batch, y_low, y_high):
        """
        load feature_map from gm to ub or l1
        """
        src_stride = (self.height * self.width - self.width) * self.c_burst
        if src_stride <= 65535:
            self.tik_instance.data_move(fm_cache[0, 0, 0, 0], self.fm_gm[batch, 0, y_low, 0, 0], 0, self.channels,
                                        self.width * self.c_burst, src_stride, 0)
            self.tik_instance.data_move(fm_cache[1, 0, 0, 0], self.fm_gm[batch, 0, y_high, 0, 0], 0, self.channels,
                                        self.width * self.c_burst, src_stride, 0)

        else:
            with self.tik_instance.for_range(0, self.channels) as c_idx:
                self.tik_instance.data_move(fm_cache[0, c_idx, 0, 0], self.fm_gm[batch, c_idx, y_low, 0, 0], 0, 1,
                                            self.width * self.c_burst, 1, 1)
                self.tik_instance.data_move(fm_cache[1, c_idx, 0, 0], self.fm_gm[batch, c_idx, y_high, 0, 0], 0, 1,
                                            self.width * self.c_burst, 1, 1)

    def load_fm_grid(self, fm_grid, feature_map, fm_batch, x_low, y_low, x_high, y_high):
        """
        load one grid feature_map
        """
        if self.tiling_mode in (0, 1, 4):
            self.load_fm_grid1(fm_grid, feature_map, fm_batch, x_low, y_low, x_high, y_high)
        else:
            self.load_fm_grid2(fm_grid, feature_map, x_low, x_high)

    def load_fm_grid1(self, fm_grid, feature_map, fm_batch, x_low, y_low, x_high, y_high):
        """
        load one grid feature_map
        """
        if self.tiling_mode in (0, 1):
            fm_batch.set_as(0)

        if (self.height * self.width - 1) * self.c_burst <= 65535:
            burst_len = (16 + self.data_each_block - 1) // self.data_each_block
            src_stride = (self.height * self.width - 1) * self.c_burst
            dst_stride = (2 * 2 - 1) * self.c_burst
            self.tik_instance.data_move(fm_grid[0, 0, 0, 0], feature_map[fm_batch, 0, y_low, x_low, 0], 0,
                                        self.channels, burst_len, src_stride=src_stride,
                                        dst_stride=dst_stride)
            self.tik_instance.data_move(fm_grid[0, 0, 1, 0], feature_map[fm_batch, 0, y_low, x_high, 0], 0,
                                        self.channels, burst_len, src_stride=src_stride,
                                        dst_stride=dst_stride)
            self.tik_instance.data_move(fm_grid[0, 1, 0, 0], feature_map[fm_batch, 0, y_high, x_low, 0], 0,
                                        self.channels, burst_len, src_stride=src_stride,
                                        dst_stride=dst_stride)
            self.tik_instance.data_move(fm_grid[0, 1, 1, 0], feature_map[fm_batch, 0, y_high, x_high, 0], 0,
                                        self.channels, burst_len, src_stride=src_stride,
                                        dst_stride=dst_stride)
        else:
            with self.tik_instance.for_range(0, self.channels) as c_idx:
                self.data_move(fm_grid[c_idx, 0, 0, 0], feature_map[fm_batch, c_idx, y_low, x_low, 0], num=16)
                self.data_move(fm_grid[c_idx, 0, 1, 0], feature_map[fm_batch, c_idx, y_low, x_high, 0], num=16)
                self.data_move(fm_grid[c_idx, 1, 0, 0], feature_map[fm_batch, c_idx, y_high, x_low, 0], num=16)
                self.data_move(fm_grid[c_idx, 1, 1, 0], feature_map[fm_batch, c_idx, y_high, x_high, 0], num=16)

    def load_fm_grid2(self, fm_grid, fm_cache, x_low, x_high):
        """
        load one grid feature_map
        """
        src_stride = (self.width - 1) * self.c_burst
        dst_stride = (2 * 2 - 1) * self.c_burst

        self.tik_instance.data_move(fm_grid[0, 0, 0, 0], fm_cache[0, 0, x_low, 0], 0, self.channels, self.c_burst,
                                    src_stride, dst_stride)
        self.tik_instance.data_move(fm_grid[0, 0, 1, 0], fm_cache[0, 0, x_high, 0], 0, self.channels, self.c_burst,
                                    src_stride, dst_stride)
        self.tik_instance.data_move(fm_grid[0, 1, 0, 0], fm_cache[1, 0, x_low, 0], 0, self.channels, self.c_burst,
                                    src_stride, dst_stride)
        self.tik_instance.data_move(fm_grid[0, 1, 1, 0], fm_cache[1, 0, x_high, 0], 0, self.channels, self.c_burst,
                                    src_stride, dst_stride)

    def move_result(self, val_ub, roi_idx, ph):
        """
        move result from ub to gm
        """
        dst_stride = (self.pool_h * self.pool_w - self.pool_w) * self.c_burst
        if dst_stride <= 65535:
            self.tik_instance.data_move(self.out_gm[roi_idx, 0, ph, 0, 0], val_ub[0, 0, 0], 0,
                                        self.channels, self.pool_w * self.c_burst, 0, dst_stride)

        else:
            with self.tik_instance.for_range(0, self.channels) as c_idx:
                self.tik_instance.data_move(self.out_gm[roi_idx, c_idx, ph, 0, 0],
                                            val_ub[c_idx, 0, 0], 0, 1, self.pool_w * self.c_burst, 0, 0)

    def calculate_val(self, val1, val2, val3, val4, fm_grid, w1, w2, w3, w4):
        """
        calculate value of grid
        """
        self.tik_instance.vec_muls(16, val1[0, 0], fm_grid[0, 0, 0, 0], w1, self.channels, self.c_burst,
                                   4 * self.c_burst)
        self.tik_instance.vec_muls(16, val2[0, 0], fm_grid[0, 0, 1, 0], w2, self.channels, self.c_burst,
                                   4 * self.c_burst)
        self.tik_instance.vec_muls(16, val3[0, 0], fm_grid[0, 1, 0, 0], w3, self.channels, self.c_burst,
                                   4 * self.c_burst)
        self.tik_instance.vec_muls(16, val4[0, 0], fm_grid[0, 1, 1, 0], w4, self.channels, self.c_burst,
                                   4 * self.c_burst)

        self.tik_instance.vec_add(16, val1[0, 0], val1[0, 0], val2[0, 0], self.channels, self.c_burst, self.c_burst,
                                  self.c_burst)
        self.tik_instance.vec_add(16, val1[0, 0], val1[0, 0], val3[0, 0], self.channels, self.c_burst, self.c_burst,
                                  self.c_burst)
        self.tik_instance.vec_add(16, val1[0, 0], val1[0, 0], val4[0, 0], self.channels, self.c_burst, self.c_burst,
                                  self.c_burst)

    def compute_per_core(self, roi_num, block_idx):
        """
        compute per core
        """
        batch_ub_fp = self.tik_instance.Tensor(self.dtype, [roi_num], scope=tik.scope_ubuf, name="batch_ub_fp")
        batch_ub = self.tik_instance.Tensor("int32", [roi_num], scope=tik.scope_ubuf, name="batch_ub")
        x0_ub = self.tik_instance.Tensor(self.dtype, [roi_num], scope=tik.scope_ubuf, name="x0_ub")
        y0_ub = self.tik_instance.Tensor(self.dtype, [roi_num], scope=tik.scope_ubuf, name="y0_ub")
        x1_ub = self.tik_instance.Tensor(self.dtype, [roi_num], scope=tik.scope_ubuf, name="x1_ub")
        y1_ub = self.tik_instance.Tensor(self.dtype, [roi_num], scope=tik.scope_ubuf, name="y1_ub")

        with self.tik_instance.new_stmt_scope():
            roi_ub = self.tik_instance.Tensor(self.dtype, [roi_num, self.proposal_num], scope=tik.scope_ubuf,
                                              name="roi_ub")
            self.data_move(roi_ub[0, 0], self.roi_gm[block_idx * self.avg_roi, 0],
                           num=roi_num * self.proposal_num)
            self.extract_roi(roi_ub, batch_ub_fp, x0_ub, y0_ub, x1_ub, y1_ub, roi_num)
            self.data_conv(batch_ub, batch_ub_fp, [0, 0], "round", num=roi_num)

        grid_w_ub = self.tik_instance.Tensor("int32", [roi_num], scope=tik.scope_ubuf, name="w_ub")
        grid_h_ub = self.tik_instance.Tensor("int32", [roi_num], scope=tik.scope_ubuf, name="h_ub")
        count_ub = self.tik_instance.Tensor("int32", [roi_num], scope=tik.scope_ubuf, name="count_ub")

        self.scale_roi(x0_ub, y0_ub, x1_ub, y1_ub, roi_num)
        with self.tik_instance.new_stmt_scope():
            # x1_ub, y1_ub --> bin_size_w, bin_size_h
            one_ub = self.tik_instance.Tensor(self.dtype, [roi_num], scope=tik.scope_ubuf, name="one_ub")
            self.dup_value(one_ub, roi_num, 1)
            self.get_bin_size(x0_ub, y0_ub, x1_ub, y1_ub, one_ub, roi_num)

        self.get_roi_grid(grid_w_ub, grid_h_ub, x1_ub, y1_ub, roi_num)
        with self.tik_instance.new_stmt_scope():
            one_ub_int = self.tik_instance.Tensor("int32", [roi_num], scope=tik.scope_ubuf, name="one_ub_int")
            self.dup_value(one_ub_int, roi_num, 1)
            self.get_count(count_ub, grid_w_ub, grid_h_ub, one_ub_int, roi_num)

        self.tiling_mode = self.get_tiling_mode()
        scalar = self.CommonScalar(self.tik_instance)
        if self.tiling_mode in (0, 1, 4):
            if self.tiling_mode == 0:
                fm_cache = self.tik_instance.Tensor(self.dtype, [1, self.channels, self.height, self.width, self.c0],
                                                    scope=tik.scope_ubuf, name="fm_ub")
            elif self.tiling_mode == 1:
                fm_cache = self.tik_instance.Tensor(self.dtype, [1, self.channels, self.height, self.width, self.c0],
                                                    scope=tik.scope_cbuf, name="fm_l1")
            else:
                fm_cache = self.fm_gm
            self.get_val_c1hwc0(x0_ub, y0_ub, x1_ub, y1_ub, grid_w_ub, grid_h_ub, batch_ub, count_ub, roi_num,
                                block_idx, scalar, fm_cache)

        if self.tiling_mode in (2, 3):
            if self.tiling_mode == 2:
                fm_cache = self.tik_instance.Tensor(self.dtype, [2, self.channels, self.width, self.c0],
                                                    scope=tik.scope_ubuf, name="fm_ub")
            else:
                fm_cache = self.tik_instance.Tensor(self.dtype, [2, self.channels, self.width, self.c0],
                                                    scope=tik.scope_cbuf, name="fm_l1")
            self.get_val_c1wc0(x0_ub, y0_ub, x1_ub, y1_ub, grid_w_ub, grid_h_ub, batch_ub, count_ub, roi_num, block_idx,
                               scalar, fm_cache)

    def get_val_c1hwc0(self, x0_ub, y0_ub, x1_ub, y1_ub, grid_w_ub, grid_h_ub, batch_ub, count_ub, roi_num, block_idx,
                       scalar, fm_cache):
        """
        get value when tiling_mode in (0, 1, 4)
        """
        cache_idx = self.tik_instance.Scalar("int32", init_value=-1)
        fp_idx = self.tik_instance.Scalar("float32")

        with self.tik_instance.for_range(0, roi_num) as n:
            self.set_roi(scalar, x0_ub, y0_ub, x1_ub, y1_ub, grid_w_ub, grid_h_ub, batch_ub, count_ub, n)

            if self.tiling_mode != 4:
                with self.tik_instance.if_scope(scalar.fm_batch != cache_idx):
                    self.data_move(fm_cache[0, 0, 0, 0, 0], self.fm_gm[scalar.fm_batch, 0, 0, 0, 0], num=self.c1hwc0)
                    cache_idx.set_as(scalar.fm_batch)

            val_ub = self.tik_instance.Tensor(self.dtype, [self.channels, self.pool_w, self.c0], scope=tik.scope_ubuf,
                                              name="val_ub")
            self.dup_value(val_ub, num=self.channels * self.pool_w * self.c0, dup_value=self.out_init_val)

            with self.tik_instance.for_range(0, self.pool_h) as ph:
                with self.tik_instance.for_range(0, self.pool_w) as pw:
                    roi_scalars = [scalar.start_h, scalar.start_w, scalar.grid_h, scalar.grid_w, scalar.size_h,
                                   scalar.size_w]
                    xy_scalars = [scalar.x, scalar.y, scalar.x_low, scalar.y_low, scalar.x_high, scalar.y_high]
                    weight_scalars = [scalar.w1, scalar.w2, scalar.w3, scalar.w4]
                    self.bilinear_interpolate(val_ub, fm_cache, scalar.fm_batch, ph, pw, roi_scalars, xy_scalars,
                                              weight_scalars)

                fp_idx.set_as(scalar.val_n)
                if self.pool_mode == 'avg':
                    self.data_muls(val_ub, val_ub, 1 / fp_idx, [0, 0], num=self.channels * self.pool_w * self.c0)
                self.move_result(val_ub, block_idx * self.avg_roi + n, ph)
                self.dup_value(val_ub, num=self.channels * self.pool_w * self.c0, dup_value=self.out_init_val)

    def get_val_c1wc0(self, x0_ub, y0_ub, x1_ub, y1_ub, grid_w_ub, grid_h_ub, batch_ub, count_ub, roi_num, block_idx,
                      scalar, fm_cache):
        """
        get value when tiling_mode in (2, 3)
        """
        fp_idx = self.tik_instance.Scalar("float32")
        with self.tik_instance.for_range(0, roi_num) as n:
            self.set_roi(scalar, x0_ub, y0_ub, x1_ub, y1_ub, grid_w_ub, grid_h_ub, batch_ub, count_ub, n)
            val_ub = self.tik_instance.Tensor(self.dtype, [self.channels, self.pool_w, self.c0], scope=tik.scope_ubuf,
                                              name="val_ub")
            self.dup_value(val_ub, num=self.channels * self.pool_w * self.c0, dup_value=self.out_init_val)

            with self.tik_instance.for_range(0, self.pool_h) as ph:
                with self.tik_instance.for_range(0, scalar.grid_h) as iy:
                    fp_idx.set_as(iy)
                    y_idx = scalar.start_h + ph * scalar.size_h + (fp_idx + 0.5) * scalar.size_h / scalar.grid_h
                    with self.tik_instance.if_scope(tik.all(y_idx >= -1.0, y_idx <= self.height)):
                        scalar.y.set_as(y_idx)
                        self.get_idx(scalar.y, scalar.y_low, scalar.y_high, self.height)
                        self.load_fm_h(fm_cache, scalar.fm_batch, scalar.y_low, scalar.y_high)

                        with self.tik_instance.for_range(0, self.pool_w) as pw:
                            val1 = self.tik_instance.Tensor(self.dtype, [self.channels, self.c0],
                                                            scope=tik.scope_ubuf, name="val1")
                            val2 = self.tik_instance.Tensor(self.dtype, [self.channels, self.c0],
                                                            scope=tik.scope_ubuf, name="val2")
                            val3 = self.tik_instance.Tensor(self.dtype, [self.channels, self.c0],
                                                            scope=tik.scope_ubuf, name="val3")
                            val4 = self.tik_instance.Tensor(self.dtype, [self.channels, self.c0],
                                                            scope=tik.scope_ubuf, name="val4")
                            roi_scalars = [scalar.start_h, scalar.start_w, scalar.grid_h, scalar.grid_w, scalar.size_h,
                                           scalar.size_w]
                            xy_scalars = [scalar.x, scalar.y, scalar.x_low, scalar.y_low, scalar.x_high, scalar.y_high]
                            weight_scalars = [scalar.w1, scalar.w2, scalar.w3, scalar.w4]

                            self.bilinear_for_w(val1, val2, val3, val4, fm_cache, val_ub, pw, scalar.fm_batch,
                                                roi_scalars, xy_scalars, weight_scalars)

                fp_idx.set_as(scalar.val_n)
                if self.pool_mode == 'avg':
                    self.data_muls(val_ub, val_ub, 1 / fp_idx, [0, 0], num=self.channels * self.pool_w * self.c0)
                self.move_result(val_ub, block_idx * self.avg_roi + n, ph)
                self.dup_value(val_ub, num=self.channels * self.pool_w * self.c0, dup_value=self.out_init_val)

    def bilinear_interpolate(self, val_ub, fm_cache, fm_batch, ph, pw, roi_scalars, xy_scalars, weight_scalars):
        """
        bilinear interpolate
        """
        start_h, start_w, grid_h, grid_w, size_h, size_w = roi_scalars
        x, y, x_low, y_low, x_high, y_high = xy_scalars
        fp_idx = self.tik_instance.Scalar("float32")

        val1 = self.tik_instance.Tensor(self.dtype, [self.channels, self.c0], scope=tik.scope_ubuf, name="val1")
        val2 = self.tik_instance.Tensor(self.dtype, [self.channels, self.c0], scope=tik.scope_ubuf, name="val2")
        val3 = self.tik_instance.Tensor(self.dtype, [self.channels, self.c0], scope=tik.scope_ubuf, name="val3")
        val4 = self.tik_instance.Tensor(self.dtype, [self.channels, self.c0], scope=tik.scope_ubuf, name="val4")
        with self.tik_instance.for_range(0, grid_h) as iy:
            fp_idx.set_as(iy)
            y_idx = start_h + ph * size_h + (fp_idx + 0.5) * size_h / grid_h
            with self.tik_instance.if_scope(tik.all(y_idx >= -1.0, y_idx <= self.height)):
                y.set_as(y_idx)
                self.get_idx(y, y_low, y_high, self.height)

                self.bilinear_for_w(val1, val2, val3, val4, fm_cache, val_ub, pw, fm_batch, roi_scalars, xy_scalars,
                                    weight_scalars)
            with self.tik_instance.else_scope():
                self.dup_value(val1, num=self.channels * self.c0, dup_value=0.0)
                pool_func = self.tik_instance.vec_add if self.pool_mode == 'avg' else self.tik_instance.vec_max
                pool_func(16, val_ub[0, pw, 0], val_ub[0, pw, 0],
                          val1[0, 0], self.channels,
                          self.pool_w * self.c_burst, self.pool_w * self.c_burst, self.c_burst)

    def bilinear_for_w(self, val1, val2, val3, val4, fm_cache, val_ub, pw, fm_batch, roi_scalars, xy_scalars,
                       weight_scalars):
        """
        bilinear interpolate for w
        """
        start_h, start_w, grid_h, grid_w, size_h, size_w = roi_scalars
        x, y, x_low, y_low, x_high, y_high = xy_scalars
        w1, w2, w3, w4 = weight_scalars
        fp_idx = self.tik_instance.Scalar("float32")
        with self.tik_instance.for_range(0, grid_w) as ix:
            fp_idx.set_as(ix)
            x_idx = start_w + pw * size_w + (fp_idx + 0.5) * size_w / grid_w
            with self.tik_instance.if_scope(tik.all(x_idx >= -1.0, x_idx <= self.width)):
                fm_grid = self.tik_instance.Tensor(self.dtype, (self.channels, 2, 2, 16),
                                                   name="fm_grid", scope=tbe_platform.scope_ubuf)
                x.set_as(x_idx)
                self.get_idx(x, x_low, x_high, self.width)

                self.get_weight(x, y, x_low, y_low, w1, w2, w3, w4)
                self.load_fm_grid(fm_grid, fm_cache, fm_batch, x_low, y_low, x_high, y_high)
                self.calculate_val(val1, val2, val3, val4, fm_grid, w1, w2, w3, w4)
            with self.tik_instance.else_scope():
                self.dup_value(val1, num=self.channels * self.c0, dup_value=0.0)
            pool_func = self.tik_instance.vec_add if self.pool_mode == 'avg' else self.tik_instance.vec_max
            pool_func(16, val_ub[0, pw, 0], val_ub[0, pw, 0],
                        val1[0, 0], self.channels,
                        self.pool_w * self.c_burst, self.pool_w * self.c_burst, self.c_burst)

    def get_dtype_size(self, dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2}
        return dtype_dict.get(dtype)

    def data_move(self, dst, src, num, src_stride=0, dst_stride=0):
        """
        move data
        """
        sid = 0
        nburst = 1
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        burst_len = (num + data_each_block - 1) // data_each_block
        self.tik_instance.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride,
                                    dst_stride=dst_stride)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        stride = 8

        loop = num // (mask * 255)
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        if repeat_time > 0:
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset += repeat_time * mask

        last_num = num % mask
        if last_num > 0:
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)

    def single_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        vector_mask_max = 64
        dst_offset, src_offset = offsets

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], scalar, 255,
                       dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_stride, src_stride)

    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8,
                                 src1_stride=8):
        """
        tik api template
        """
        vector_mask_max = 64
        dst_offset, src0_offset, src1_offset = offsets

        tensor_size = num if num else src1.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src0_offset = src0_offset + index * vector_mask_max * 255
                tmp_src1_offset = src1_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src0[tmp_src0_offset], src1[tmp_src1_offset], 255,
                       dst_stride, src0_stride, src1_stride)

            dst_offset += loop * vector_mask_max * 255
            src0_offset += loop * vector_mask_max * 255
            src1_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            op_obj(vector_mask_max, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time, dst_stride,
                   src0_stride, src1_stride)
            dst_offset += repeat_time * vector_mask_max
            src0_offset += repeat_time * vector_mask_max
            src1_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            op_obj(last_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, dst_stride, src0_stride,
                   src1_stride)

    def data_add(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik add
        """
        self.double_operator_template(self.tik_instance.vec_add, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_adds(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik adds
        """
        self.single_operator_template(self.tik_instance.vec_adds, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)

    def data_mul(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik mul
        """
        self.double_operator_template(self.tik_instance.vec_mul, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_muls(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik muls
        """
        self.single_operator_template(self.tik_instance.vec_muls, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)

    def data_conv(self, dst, src, offsets, mode="ceil", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        vector_mask_max = 64
        dst_offset, src_offset = offsets

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                self.tik_instance.vec_conv(vector_mask_max, mode, dst[tmp_dst_offset], src[tmp_src_offset], 255,
                                           dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            self.tik_instance.vec_conv(vector_mask_max, mode, dst[dst_offset], src[src_offset], repeat_time,
                                       dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    def data_sub(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik sub
        """
        self.double_operator_template(self.tik_instance.vec_sub, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_max(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik max
        """
        self.double_operator_template(self.tik_instance.vec_max, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)


def check_param(feature_map, rois):
    """
    check input parameter
    """
    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    fm_shape = feature_map.get("shape")
    rois_shape = rois.get("shape")
    fm_dtype = feature_map.get("dtype").lower()
    rois_dtype = feature_map.get("dtype").lower()
    roi_num = rois_shape[0]
    c1 = fm_shape[1]

    if fm_dtype != "float32" or rois_dtype != "float32":
        raise RuntimeError("roi_align only support float32")

    if roi_num // core_num > 1000:
        raise RuntimeError("roi_num is too large")

    if c1 > 255:
        raise RuntimeError("c1 should be smaller than 255")


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments
def roi_align_true(feature_map_dict, rois_dict, roisn_dict, output, scale, pool_h, pool_w, sample_ratio, roi_end_mode,
                   pool_mode, kernel_name):
    """
    roi_align func
    """
    check_param(feature_map_dict, rois_dict)
    instance = RoiAlign(feature_map_dict, rois_dict, roisn_dict, output, scale, pool_h, pool_w, sample_ratio,
                        roi_end_mode, pool_mode, kernel_name)
    return instance.compute()
