# Copyright 2022 Huawei Technologies Co., Ltd
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
deformable_roi_pool_grad
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


class Constant:
    """
    The class for constant
    """
    # max uint16
    PARAMS_SIZE = 2**31 - 1
    TILING_ARG_NUM = 256
    # C0 size
    C0_SIZE = 16
    # batch size
    BATCH_SIZE = 128
    # data type of int64
    INT64 = "int64"
    # one block size takes up 32b
    BLOCK_SIZE = 32
    TYPE_LEN_DICT = {"float16": 2, "float32": 4, "int64": 8}


def ceil_value(value, dtype):
    """
    if not divide exactly then plus 1
    """
    value *= Constant.TYPE_LEN_DICT.get(dtype)

    return (value + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE


def align_value(value, factor):
    """
    Alignment based on factor.
    """
    return (value + factor - 1) // factor * factor


class DeformableRoiPoolGrad():
    """
    define deformable_roi_pool_grad object
    """

    def __init__(self):
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.core_num = profile.get_aicore_num()
        self.ub_size = profile.get_unified_buffer_size()
        self.l1_size = profile.get_l1_buffer_size()
        self.feature_map_to_ub_verify = 0
        self.feature_map_to_l1_verify = 0
        self.w_number_ub = 0
        self.w_number_l1 = 0
        self.tiling_mode = 1
        self.real_core_num = 0
        self.tiling_dtype = Constant.INT64
        self.rois_n = 0
        self.rois_row_length = 0
        self.c1_num = 0

        self.grad_output = None
        self.feature_map = None
        self.rois = None
        self.offset = None
        self.grad_fm = None
        self.grad_offset = None
        self.tiling_gm = None

        self.pooled_width = 0
        self.pooled_height = 0
        self.x_height = 0
        self.x_width = 0
        self.spatial_scale = 1.0
        self.sample_ratio = 2
        self.gamma = 0.1
        self.kernel_name = None
        self.has_offset = True
        self.dtype = "float32"
        self.available_c1_num = None
        self.offset_shape = None

    def deformable_roi_pool_grad_compute(self):
        if self.offset_shape:
            numel = 1
            for i in range(len(self.offset_shape)):
                numel *= self.offset_shape[i]
            if numel == 0:
                self.has_offset = False

        tik_instance = self.tik_instance
        self.grad_output = tik_instance.Tensor(self.dtype,
                                               (Constant.PARAMS_SIZE, ),
                                               name="grad_output_data",
                                               scope=tbe_platform.scope_gm)
        self.feature_map = tik_instance.Tensor(self.dtype,
                                               (Constant.PARAMS_SIZE, ),
                                               name="input_data",
                                               scope=tbe_platform.scope_gm)
        self.rois = tik_instance.Tensor(self.dtype, (Constant.PARAMS_SIZE, ),
                                        name="rois_data",
                                        scope=tbe_platform.scope_gm)
        self.offset = tik_instance.Tensor(self.dtype,
                                          (Constant.PARAMS_SIZE, ),
                                          name="offset_data",
                                          scope=tbe_platform.scope_gm)
        inputs = [self.grad_output, self.feature_map, self.rois, self.offset]

        self.grad_fm = tik_instance.Tensor(self.dtype,
                                           (Constant.PARAMS_SIZE, ),
                                           name="grad_input_data",
                                           scope=tbe_platform.scope_gm,
                                           is_atomic_add=True)
        self.grad_offset = tik_instance.Tensor(self.dtype,
                                               (Constant.PARAMS_SIZE, ),
                                               name="grad_offset_data",
                                               scope=tbe_platform.scope_gm,
                                               is_atomic_add=True)
        self.tiling_gm = tik_instance.Tensor(self.tiling_dtype,
                                             (Constant.TILING_ARG_NUM, ),
                                             name="tiling_gm",
                                             scope=tik.scope_gm)

        tiling_ub = tik_instance.Tensor(self.tiling_dtype,
                                        (Constant.TILING_ARG_NUM, ),
                                        name="tiling_ub",
                                        scope=tik.scope_ubuf)
        tik_instance.data_move(
            tiling_ub, self.tiling_gm, 0, 1,
            ceil_value(Constant.TILING_ARG_NUM, self.tiling_dtype), 0, 0)
        self._get_tiling_args(tiling_ub)

        self._deformable_roi_pool_grad_compute_tiling()

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }

        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.core_num,
            "ub_size": self.ub_size
        })

        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=inputs,
                              outputs=[self.grad_fm, self.grad_offset],
                              flowtable=[self.tiling_gm],
                              config=opt_config)

    def _deformable_roi_pool_grad_compute_tiling(self):
        """
        define deformable_roi_pool_grad tiling method
        """
        tik_instance = self.tik_instance
        with tik_instance.for_range(0, self.real_core_num,
                                    block_num=self.real_core_num) as block_id:
            core_rois_n = self.tik_instance.Scalar(
                init_value=self.rois_n // self.real_core_num,
                dtype="int32",
                name="core_rois_n")
            core_tail = self.tik_instance.Scalar(init_value=self.rois_n %
                                                 self.real_core_num,
                                                 dtype="int32",
                                                 name="core_tail")
            core_bias = self.tik_instance.Scalar(init_value=core_rois_n *
                                                 block_id,
                                                 dtype="int32",
                                                 name="core_bias")
            with self.tik_instance.if_scope(core_tail != 0):
                with self.tik_instance.if_scope(block_id < core_tail):
                    core_rois_n.set_as(core_rois_n + 1)
                    core_bias.set_as(core_rois_n * block_id)
                with self.tik_instance.else_scope():
                    core_bias.set_as(core_rois_n * block_id + core_tail)

            with tik_instance.new_stmt_scope():
                self._compute_mode_1(core_rois_n, core_bias)

    def _get_tiling_args(self, tiling_ub):
        self.tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                    name="tiling_mode")
        self.tiling_mode.set_as(tiling_ub[0])
        self.real_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                      name="real_core_num")
        self.real_core_num.set_as(tiling_ub[1])
        self.rois_n = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                               name="rois_n")
        self.rois_n.set_as(tiling_ub[2])
        self.rois_row_length = self.tik_instance.Scalar(
            dtype=self.tiling_dtype, name="rois_row_length")
        self.rois_row_length.set_as(tiling_ub[3])
        self.c1_num = self.tik_instance.Scalar(dtype="int32", name="c1_num")
        self.c1_num.set_as(tiling_ub[4])
        self.x_height = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                 name="x_height")
        self.x_height.set_as(tiling_ub[5])
        self.x_width = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                name="x_width")
        self.x_width.set_as(tiling_ub[6])

    def _compute_mode_1(self, core_rois_n, core_bias):
        """
        deformable_roi_pool_grad_tik
        """
        tik_instance = self.tik_instance
        n_bust = 2

        rois_ub = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE, 8],
                                      name="rois_ub",
                                      scope=tbe_platform.scope_ubuf)
        proposals_ub_x0 = tik_instance.Tensor(self.dtype,
                                              [Constant.BATCH_SIZE],
                                              name="proposals_ub_x0",
                                              scope=tbe_platform.scope_ubuf)
        proposals_ub_y0 = tik_instance.Tensor(self.dtype,
                                              [Constant.BATCH_SIZE],
                                              name="proposals_ub_y0",
                                              scope=tbe_platform.scope_ubuf)
        proposals_ub_x1 = tik_instance.Tensor(self.dtype,
                                              [Constant.BATCH_SIZE],
                                              name="proposals_ub_x1",
                                              scope=tbe_platform.scope_ubuf)
        proposals_ub_y1 = tik_instance.Tensor(self.dtype,
                                              [Constant.BATCH_SIZE],
                                              name="proposals_ub_y1",
                                              scope=tbe_platform.scope_ubuf)
        roi_float_fm_index = tik_instance.Tensor(self.dtype,
                                                 [Constant.BATCH_SIZE],
                                                 name="roi_float_fm_index",
                                                 scope=tbe_platform.scope_ubuf)
        roi_int32_fm_index = tik_instance.Tensor("int32",
                                                 [Constant.BATCH_SIZE],
                                                 name="roi_int32_fm_index",
                                                 scope=tbe_platform.scope_ubuf)

        tik_instance.vector_dup(64, rois_ub, 0.0, 16, 1, 8)
        rois_valid = tik_instance.Scalar(dtype="int32", init_value=core_rois_n)
        rois_batch_num = (core_rois_n + 127) // Constant.BATCH_SIZE

        with tik_instance.if_scope(rois_valid != 0):
            with tik_instance.for_range(0, rois_batch_num) as roi_128_number:
                rois_valid_in_block = tik_instance.Scalar(
                    dtype="int32", init_value=Constant.BATCH_SIZE)
                with tik_instance.if_scope(roi_128_number == (rois_batch_num -
                                                              1)):
                    rois_valid_in_block.set_as(rois_valid - roi_128_number *
                                               Constant.BATCH_SIZE)
                with tik_instance.if_scope(self.rois_row_length == 5):
                    rois_ub_n5 = tik_instance.Tensor(
                        self.dtype, [Constant.BATCH_SIZE, 5],
                        name="rois_ub_n5",
                        scope=tbe_platform.scope_ubuf)
                    burst_num = (rois_valid_in_block * 5 * n_bust + 15) // 16
                    tik_instance.data_move(
                        rois_ub_n5[0, 0],
                        self.rois[(core_bias +
                                   roi_128_number * Constant.BATCH_SIZE) * 5],
                        0, 1, burst_num, 0, 0)

                    self._tf_n52n8(rois_ub, rois_ub_n5, rois_valid_in_block)

                with tik_instance.else_scope():
                    burst_num = (rois_valid_in_block * 8 * n_bust + 15) // 16
                    tik_instance.data_move(
                        rois_ub[0, 0],
                        self.rois[(core_bias +
                                   roi_128_number * Constant.BATCH_SIZE) * 8],
                        0, 1, burst_num, 0, 0)

                with tik_instance.for_range(0, rois_valid_in_block) as j:
                    roi_float_fm_index[j].set_as(rois_ub[j, 0])
                    proposals_ub_x0[j].set_as(rois_ub[j, 1])
                    proposals_ub_y0[j].set_as(rois_ub[j, 2])
                    proposals_ub_x1[j].set_as(rois_ub[j, 3])
                    proposals_ub_y1[j].set_as(rois_ub[j, 4])

                tik_instance.vec_conv(64, "ceil", roi_int32_fm_index[0],
                                      roi_float_fm_index[0], 2, 8, 4 * n_bust)

                roi_bin_h_sample, roi_bin_w_sample, proposals_ub_x0, proposals_ub_x1, grid_h, grid_w, roi_h, roi_w = \
                    self._get_deformable_roi_pool_grad_perf_scale_for_zero(proposals_ub_x0, proposals_ub_y0,
                                                                           proposals_ub_x1, proposals_ub_y1)

                self._common_compute(proposals_ub_x0, proposals_ub_y0,
                                     roi_128_number, rois_valid_in_block,
                                     n_bust, roi_int32_fm_index, grid_h,
                                     grid_w, core_bias, roi_h, roi_w,
                                     roi_bin_h_sample, roi_bin_w_sample)

    def _tf_n52n8(self, rois_ub, rois_n5, block_num):
        """
        transform ROIS from N5 to N8
        """
        tik_instance = self.tik_instance
        with tik_instance.for_range(0, block_num) as rois_num:
            rois_ub[rois_num, 0].set_as(rois_n5[rois_num, 0])
            rois_ub[rois_num, 1].set_as(rois_n5[rois_num, 1])
            rois_ub[rois_num, 2].set_as(rois_n5[rois_num, 2])
            rois_ub[rois_num, 3].set_as(rois_n5[rois_num, 3])
            rois_ub[rois_num, 4].set_as(rois_n5[rois_num, 4])

    # 'pylint: disable=too-many-locals,too-many-arguments,too-many-return-values
    def _get_deformable_roi_pool_grad_perf_scale_for_zero(self, proposals_ub_x0,
                                                          proposals_ub_y0,
                                                          proposals_ub_x1,
                                                          proposals_ub_y1):
        """
        get start point, bin_size and sample number
        """
        tik_instance = self.tik_instance

        roi_h_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                         name="roi_h_fp32",
                                         scope=tbe_platform.scope_ubuf)
        roi_w_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                         name="roi_w_fp32",
                                         scope=tbe_platform.scope_ubuf)

        self._get_deformable_roi_pool_start_point(proposals_ub_x0, proposals_ub_y0,
                                                  proposals_ub_x1, proposals_ub_y1,
                                                  roi_h_fp32, roi_w_fp32)

        roi_bin_h_fp32_value = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE],
            name="roi_bin_h_fp32_value",
            scope=tbe_platform.scope_ubuf)
        roi_bin_w_fp32_value = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE],
            name="roi_bin_w_fp32_value",
            scope=tbe_platform.scope_ubuf)
        grid_h = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                     name="grid_h",
                                     scope=tbe_platform.scope_ubuf)
        grid_w = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                     name="grid_w",
                                     scope=tbe_platform.scope_ubuf)

        tik_instance.vec_muls(64, roi_bin_h_fp32_value[0], roi_h_fp32[0],
                              1.0 / self.pooled_height, 2, 8, 8)
        tik_instance.vec_muls(64, roi_bin_w_fp32_value[0], roi_w_fp32[0],
                              1.0 / self.pooled_width, 2, 8, 8)

        h_sample_div, w_sample_div, roi_bin_h_sample, roi_bin_w_sample =\
            self._get_defor_roi_pool_roi_bin_number(roi_bin_h_fp32_value,
                                                    roi_bin_w_fp32_value)

        tik_instance.vec_mul(64, grid_h[0], roi_bin_h_fp32_value[0],
                             h_sample_div, 2, 8, 8, 8)
        tik_instance.vec_mul(64, grid_w[0], roi_bin_w_fp32_value[0],
                             w_sample_div, 2, 8, 8, 8)

        return roi_bin_h_sample, roi_bin_w_sample, proposals_ub_x0, proposals_ub_y0, \
            grid_h, grid_w, roi_h_fp32, roi_w_fp32

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _get_deformable_roi_pool_start_point(self, proposals_ub_x0,
                                             proposals_ub_y0, proposals_ub_x1,
                                             proposals_ub_y1, roi_h_fp32,
                                             roi_w_fp32):
        """
        get start point and bin_size
        """
        tik_instance = self.tik_instance
        mask = 64
        repeat_times = 2

        scaler_offset = tik_instance.Scalar(dtype=self.dtype, init_value=-0.5)
        # get start point [x0,x1,y0,y1]
        tik_instance.vec_muls(mask, proposals_ub_x0[0], proposals_ub_x0[0],
                              self.spatial_scale, repeat_times, 8, 8)
        tik_instance.vec_adds(mask, proposals_ub_x0[0], proposals_ub_x0[0],
                              scaler_offset, repeat_times, 8, 8)
        tik_instance.vec_muls(mask, proposals_ub_y0[0], proposals_ub_y0[0],
                              self.spatial_scale, repeat_times, 8, 8)
        tik_instance.vec_adds(mask, proposals_ub_y0[0], proposals_ub_y0[0],
                              scaler_offset, repeat_times, 8, 8)
        tik_instance.vec_muls(mask, proposals_ub_x1[0], proposals_ub_x1[0],
                              self.spatial_scale, repeat_times, 8, 8)
        tik_instance.vec_adds(mask, proposals_ub_x1[0], proposals_ub_x1[0],
                              scaler_offset, repeat_times, 8, 8)
        tik_instance.vec_muls(mask, proposals_ub_y1[0], proposals_ub_y1[0],
                              self.spatial_scale, repeat_times, 8, 8)
        tik_instance.vec_adds(mask, proposals_ub_y1[0], proposals_ub_y1[0],
                              scaler_offset, repeat_times, 8, 8)

        # get bin size
        tik_instance.vec_sub(mask, roi_h_fp32, proposals_ub_y1[0],
                             proposals_ub_y0[0], repeat_times, 8, 8, 8)
        tik_instance.vec_sub(mask, roi_w_fp32, proposals_ub_x1[0],
                             proposals_ub_x0[0], repeat_times, 8, 8, 8)

    # 'pylint: disable=too-many-return-values
    def _get_defor_roi_pool_roi_bin_number(self, roi_bin_h_fp32_value,
                                           roi_bin_w_fp32_value):
        """
        get sample number
        """
        tik_instance = self.tik_instance
        n_bust = 2
        roi_bin_h_sample = tik_instance.Tensor("int32", [Constant.BATCH_SIZE],
                                               name="roi_bin_h_sample",
                                               scope=tbe_platform.scope_ubuf)
        roi_bin_w_sample = tik_instance.Tensor("int32", [Constant.BATCH_SIZE],
                                               name="roi_bin_w_sample",
                                               scope=tbe_platform.scope_ubuf)
        h_sample_div = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                           name="h_sample_div",
                                           scope=tbe_platform.scope_ubuf)
        w_sample_div = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                           name="w_sample_div",
                                           scope=tbe_platform.scope_ubuf)
        tik_instance.vec_dup(64, roi_bin_h_sample, self.sample_ratio, 2, 0)
        tik_instance.vec_dup(64, roi_bin_w_sample, self.sample_ratio, 2, 0)

        with tik_instance.if_scope(self.sample_ratio == 0):
            tik_instance.vec_conv(64, "ceil", roi_bin_h_sample,
                                  roi_bin_h_fp32_value, 2, 8, 4 * n_bust)
            tik_instance.vec_conv(64, "ceil", roi_bin_w_sample,
                                  roi_bin_w_fp32_value, 2, 8, 4 * n_bust)

        work_tensor_ub = tik_instance.Tensor("float32", (256, ),
                                             name="work_tensor_ub",
                                             scope=tbe_platform.scope_ubuf)
        tmp_h_sample = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                           name="tmp_h_sample",
                                           scope=tbe_platform.scope_ubuf)
        tmp_w_sample = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                           name="tmp_w_sample",
                                           scope=tbe_platform.scope_ubuf)

        tik_instance.vec_conv(64, "", tmp_h_sample, roi_bin_h_sample, 2, 8,
                              4 * n_bust)
        tik_instance.vec_conv(64, "", tmp_w_sample, roi_bin_w_sample, 2, 8,
                              4 * n_bust)

        tik_instance.vec_rec_high_preci(64, h_sample_div, tmp_h_sample,
                                        work_tensor_ub, 2, 8, 8)
        tik_instance.vec_rec_high_preci(64, w_sample_div, tmp_w_sample,
                                        work_tensor_ub, 2, 8, 8)

        return h_sample_div, w_sample_div, roi_bin_h_sample, roi_bin_w_sample

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _common_compute(self, proposals_ub_x0, proposals_ub_y0, roi_128_number,
                        rois_valid_in_block, n_bust, roi_int32_fm_index,
                        grid_h, grid_w, core_bias, roi_h, roi_w,
                        roi_bin_grid_h_vec, roi_bin_grid_w_vec):
        tik_instance = self.tik_instance
        grid_w_cur = tik_instance.Scalar(dtype=self.dtype)
        grid_h_cur = tik_instance.Scalar(dtype=self.dtype)
        roi_start_w_cur = tik_instance.Scalar(dtype=self.dtype)
        roi_start_h_cur = tik_instance.Scalar(dtype=self.dtype)
        roi_h_cur = tik_instance.Scalar(dtype=self.dtype)
        roi_w_cur = tik_instance.Scalar(dtype=self.dtype)
        roi_bin_grid_w_count = tik_instance.Scalar(dtype="int32")
        roi_bin_grid_h_count = tik_instance.Scalar(dtype="int32")
        index = tik_instance.Scalar(dtype="int32")

        with tik_instance.for_range(0, rois_valid_in_block) as curr_roi:
            index.set_as(roi_int32_fm_index[curr_roi])
            grid_w_cur.set_as(grid_w[curr_roi])
            grid_h_cur.set_as(grid_h[curr_roi])
            roi_start_w_cur.set_as(proposals_ub_x0[curr_roi])
            roi_start_h_cur.set_as(proposals_ub_y0[curr_roi])
            roi_h_cur.set_as(roi_h[curr_roi])
            roi_w_cur.set_as(roi_w[curr_roi])
            roi_bin_grid_h_count.set_as(roi_bin_grid_h_vec[curr_roi])
            roi_bin_grid_w_count.set_as(roi_bin_grid_w_vec[curr_roi])
            with tik_instance.for_range(0, self.pooled_height) as bin_h_index:
                grad_output_one_line = self._get_grad_output_one_line_(
                    core_bias, roi_128_number, curr_roi, bin_h_index)
                offset_one_line, offset_offset = self._get_offset_input(
                    core_bias, roi_128_number, curr_roi, bin_h_index)
                grad_offset_one_line_tmp = tik_instance.Tensor(
                    self.dtype, [self.pooled_width, 16],
                    name="grad_offset_one_line_tmp",
                    scope=tbe_platform.scope_ubuf)
                bin_start_h_cur = roi_start_h_cur + grid_h_cur * bin_h_index * roi_bin_grid_h_count
                with tik_instance.for_range(0,
                                            self.pooled_width) as bin_w_index:
                    bin_start_w_cur = roi_start_w_cur + grid_w_cur * bin_w_index * roi_bin_grid_w_count
                    bin_start_w_cur_, bin_start_h_cur_ = self._compute_start_point(
                        offset_one_line, roi_w_cur, roi_h_cur, bin_start_w_cur,
                        bin_start_h_cur, bin_w_index)
                    x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi, raw_x, raw_y = self._get_grid_weight(
                        grid_w_cur, grid_h_cur, bin_start_w_cur_,
                        bin_start_h_cur_)
                    with tik_instance.new_stmt_scope():
                        self._bilinear_interpolate_all_in_ub(
                            x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo,
                            y_hi, raw_x, raw_y, roi_w_cur, roi_h_cur, n_bust,
                            index, bin_w_index, grad_output_one_line,
                            grad_offset_one_line_tmp, roi_bin_grid_h_count,
                            roi_bin_grid_w_count)

                if self.has_offset:
                    tik_instance.data_move(self.grad_offset[offset_offset],
                                           grad_offset_one_line_tmp, 0, 1,
                                           self.pooled_width * n_bust, 0, 0)

    def _get_grad_output_one_line_(self, core_bias, roi_128_number, curr_roi,
                                   bin_h_index):
        tik_instance = self.tik_instance
        n_bust = 2
        grad_output_one_line = tik_instance.Tensor(
            self.dtype, [self.c1_num * self.pooled_width, 16],
            name="grad_output_one_line",
            scope=tbe_platform.scope_ubuf)
        with tik_instance.for_range(0, self.pooled_width) as i:
            grad_output_offset = self._get_grad_output_offset(
                core_bias + 128 * roi_128_number + curr_roi, 0, bin_h_index, i)
            tik_instance.data_move(
                grad_output_one_line[i * self.c1_num, 0],
                self.grad_output[grad_output_offset], 0, self.c1_num, n_bust,
                (self.pooled_height * self.pooled_width - 1) * n_bust, 0)
        return grad_output_one_line

    def _get_offset_input(self, core_bias, roi_128_number, curr_roi,
                          bin_h_index):
        tik_instance = self.tik_instance
        n_bust = 2

        if self.has_offset:
            offset_one_line = tik_instance.Tensor(self.dtype,
                                                  [self.pooled_width, 16],
                                                  name="offset_one_line",
                                                  scope=tbe_platform.scope_ubuf)
            offset_offset = self._get_offset_offset(
                core_bias + 128 * roi_128_number + curr_roi, 0, bin_h_index, 0)
            tik_instance.data_move(offset_one_line[0, 0],
                                   self.offset[offset_offset], 0, 1,
                                   self.pooled_width * n_bust, 0, 0)
            return offset_one_line, offset_offset
        else:
            return None, None

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _compute_start_point(self, offset_one_line, roi_w_cur, roi_h_cur,
                             bin_start_w_cur, bin_start_h_cur, bin_w_index):
        tik_instance = self.tik_instance
        if self.has_offset:
            delta_w_offset = tik_instance.Scalar(
                dtype=self.dtype, init_value=offset_one_line[bin_w_index, 0])
            delta_w_offset.set_as(delta_w_offset * self.gamma * roi_w_cur)
            delta_h_offset = tik_instance.Scalar(
                dtype=self.dtype, init_value=offset_one_line[bin_w_index, 1])
            delta_h_offset.set_as(delta_h_offset * self.gamma * roi_h_cur)
            bin_start_w_cur_ = bin_start_w_cur + delta_w_offset
            bin_start_h_cur_ = bin_start_h_cur + delta_h_offset
        else:
            bin_start_w_cur_ = bin_start_w_cur
            bin_start_h_cur_ = bin_start_h_cur
        return bin_start_w_cur_, bin_start_h_cur_

    def _get_offset_offset(self, n, c, h, w):
        """
        calc offset's offset
        """
        n_offset = n * 1 * self.pooled_height * self.pooled_width * 16
        c_offset = c * self.pooled_height * self.pooled_width * 16
        h_offset = h * self.pooled_width * 16
        w_offset = w * 16
        return n_offset + c_offset + h_offset + w_offset

    # 'pylint: disable=too-many-locals,too-many-arguments,too-many-statements
    def _get_start_point(self, grid_w_roi, grid_h_roi, rois_start_w,
                         rois_start_h):
        tik_instance = self.tik_instance
        dtype_num = 1
        raw_x = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                    name="raw_x",
                                    scope=tbe_platform.scope_ubuf)
        raw_y = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                    name="raw_y",
                                    scope=tbe_platform.scope_ubuf)
        const_value_0_127 = tik_instance.Tensor(self.dtype,
                                                (Constant.BATCH_SIZE, ),
                                                name="const_value_0_127",
                                                scope=tbe_platform.scope_ubuf)

        with tik_instance.for_range(0, 128) as i:
            const_value_0_127[i] = i

        grid_w_vector = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                            name="grid_w_vector",
                                            scope=tbe_platform.scope_ubuf)
        grid_h_vector = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                            name="grid_h_vector",
                                            scope=tbe_platform.scope_ubuf)

        tik_instance.vec_muls(64 * dtype_num, grid_w_vector, const_value_0_127,
                              grid_w_roi, 2 // dtype_num, 8, 8)
        tik_instance.vec_muls(64 * dtype_num, grid_h_vector, const_value_0_127,
                              grid_h_roi, 2 // dtype_num, 8, 8)

        half_grid = 0.5 * grid_w_roi + rois_start_w
        tik_instance.vec_adds(64 * dtype_num, raw_x, grid_w_vector, half_grid,
                              2 // dtype_num, 8, 8)
        half_grid = 0.5 * grid_h_roi + rois_start_h
        tik_instance.vec_adds(64 * dtype_num, raw_y, grid_h_vector, half_grid,
                              2 // dtype_num, 8, 8)
        return raw_x, raw_y

    # 'pylint: disable=too-many-locals,too-many-arguments,too-many-statements,too-many-return-values
    def _get_four_point_for_bilinear(self, x_lo, x_output, y_lo, y_output,
                                     const_value_fp32):
        tik_instance = self.tik_instance
        dtype_num = 1
        x_lo_w = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                     name="x_lo_w",
                                     scope=tbe_platform.scope_ubuf)
        x_hi_w = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                     name="x_hi_w",
                                     scope=tbe_platform.scope_ubuf)
        y_lo_w = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                     name="y_lo_w",
                                     scope=tbe_platform.scope_ubuf)
        y_hi_w = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                     name="y_hi_w",
                                     scope=tbe_platform.scope_ubuf)
        tmp_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                       name="tmp_fp32",
                                       scope=tbe_platform.scope_ubuf)

        tik_instance.vec_conv(64, "", tmp_fp32, x_lo, 2, 8, 8)
        tik_instance.vec_sub(64 * dtype_num, x_lo_w, x_output, tmp_fp32,
                             2 // dtype_num, 8, 8, 8)
        tik_instance.vec_conv(64, "", tmp_fp32, y_lo, 2, 8, 8)
        tik_instance.vec_sub(64 * dtype_num, y_lo_w, y_output, tmp_fp32,
                             2 // dtype_num, 8, 8, 8)
        tik_instance.vec_dup(64 * dtype_num, const_value_fp32, 1.0, 1, 0)
        tik_instance.vec_sub(64 * dtype_num, x_hi_w, const_value_fp32, x_lo_w,
                             2 // dtype_num, 8, 0, 8)
        tik_instance.vec_sub(64 * dtype_num, y_hi_w, const_value_fp32, y_lo_w,
                             2 // dtype_num, 8, 0, 8)
        return x_lo_w, x_hi_w, y_lo_w, y_hi_w

    def _get_grid_weight(self, grid_w, grid_h, bin_start_w, bin_start_h):
        tik_instance = self.tik_instance
        dtype_num = 1
        x_lo = tik_instance.Tensor("int32", [Constant.BATCH_SIZE],
                                   name="x_lo",
                                   scope=tbe_platform.scope_ubuf)
        x_hi = tik_instance.Tensor("int32", [Constant.BATCH_SIZE],
                                   name="x_hi",
                                   scope=tbe_platform.scope_ubuf)
        y_lo = tik_instance.Tensor("int32", [Constant.BATCH_SIZE],
                                   name="y_lo",
                                   scope=tbe_platform.scope_ubuf)
        y_hi = tik_instance.Tensor("int32", [Constant.BATCH_SIZE],
                                   name="y_hi",
                                   scope=tbe_platform.scope_ubuf)
        x_output = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                       name="x_output",
                                       scope=tbe_platform.scope_ubuf)
        y_output = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                       name="y_output",
                                       scope=tbe_platform.scope_ubuf)
        raw_x, raw_y = self._get_start_point(grid_w, grid_h, bin_start_w,
                                             bin_start_h)

        const_zero = tik_instance.Tensor(self.dtype, [64 * dtype_num],
                                         name="const_zero",
                                         scope=tbe_platform.scope_ubuf)

        tik_instance.vec_dup(64 * dtype_num, const_zero, 0, 1, 0)

        tik_instance.vec_max(64 * dtype_num, x_output, raw_x, const_zero,
                             2 // dtype_num, 8, 8, 0)
        tik_instance.vec_max(64 * dtype_num, y_output, raw_y, const_zero,
                             2 // dtype_num, 8, 8, 0)

        tik_instance.vec_conv(64, "floor", x_lo, x_output, 2, 8,
                              8 // dtype_num)
        tik_instance.vec_conv(64, "floor", y_lo, y_output, 2, 8,
                              8 // dtype_num)

        const_one = tik_instance.Tensor("int32", [64],
                                        name="const_one",
                                        scope=tbe_platform.scope_ubuf)
        tik_instance.vec_dup(64, const_one, 1, 1, 0)
        tik_instance.vec_add(64, x_hi, x_lo, const_one, 2, 8, 8, 0)
        tik_instance.vec_add(64, y_hi, y_lo, const_one, 2, 8, 8, 0)

        const_value_fp32 = tik_instance.Tensor(self.dtype, [64 * dtype_num],
                                               name="const_value_fp32",
                                               scope=tbe_platform.scope_ubuf)
        const_value_int32 = tik_instance.Tensor("int32", [64],
                                                name="const_value_int32",
                                                scope=tbe_platform.scope_ubuf)

        tik_instance.vec_dup(64 * dtype_num, const_value_fp32,
                             self.x_width - 1, 1, 0)
        tik_instance.vec_dup(64, const_value_int32, self.x_width - 1, 1, 0)
        tik_instance.vec_min(64, x_lo, x_lo, const_value_int32, 2, 8, 8, 0)
        tik_instance.vec_min(64, x_hi, x_hi, const_value_int32, 2, 8, 8, 0)
        tik_instance.vec_min(64 * dtype_num, x_output, x_output,
                             const_value_fp32, 2 // dtype_num, 8, 8, 0)

        tik_instance.vec_dup(64, const_value_int32, self.x_height - 1, 1, 0)
        tik_instance.vec_dup(64 * dtype_num, const_value_fp32,
                             self.x_height - 1, 1, 0)
        tik_instance.vec_min(64, y_lo, y_lo, const_value_int32, 2, 8, 8, 0)
        tik_instance.vec_min(64, y_hi, y_hi, const_value_int32, 2, 8, 8, 0)
        tik_instance.vec_min(64 * dtype_num, y_output, y_output,
                             const_value_fp32, 2 // dtype_num, 8, 8, 0)

        x_lo_w, x_hi_w, y_lo_w, y_hi_w = self._get_four_point_for_bilinear(
            x_lo, x_output, y_lo, y_output, const_value_fp32)

        return x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi, raw_x, raw_y

    def _bilinear_interpolate_all_in_ub(
            self, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi,
            raw_x, raw_y, roi_w_cur, roi_h_cur, n_bust, index, bin_w_index,
            grad_output_one_line, grad_offset_one_line_tmp,
            roi_bin_grid_h_count, roi_bin_grid_w_count):
        """
        bilinear interpolate
        """
        tik_instance = self.tik_instance

        grad_output_val = tik_instance.Tensor(self.dtype, [self.c1_num * 16],
                                              name="grad_output_val",
                                              scope=tbe_platform.scope_ubuf)
        tik_instance.vec_dup(16, grad_output_val, 0.0, self.c1_num, n_bust)

        tik_instance.data_move(
            grad_output_val[0], grad_output_one_line[self.c1_num * bin_w_index,
                                                     0], 0, self.c1_num,
            n_bust, 0, 0)
        bin_count = tik_instance.Scalar(dtype=self.dtype,
                                        name="bin_count",
                                        init_value=roi_bin_grid_h_count *
                                        roi_bin_grid_w_count)

        delta_x_vector_val = tik_instance.Tensor(self.dtype,
                                                 [self.c1_num * 16],
                                                 name="delta_x_vector_val",
                                                 scope=tbe_platform.scope_ubuf)
        tik_instance.vec_dup(16, delta_x_vector_val, 0.0, self.c1_num, n_bust)
        delta_y_vector_val = tik_instance.Tensor(self.dtype,
                                                 [self.c1_num * 16],
                                                 name="delta_y_vector_val",
                                                 scope=tbe_platform.scope_ubuf)
        tik_instance.vec_dup(16, delta_y_vector_val, 0.0, self.c1_num, n_bust)

        roi_y_floor = tik_instance.Tensor("int32", [Constant.BATCH_SIZE],
                                          name="roi_y_floor",
                                          scope=tbe_platform.scope_ubuf)
        roi_x_floor = tik_instance.Tensor("int32", [Constant.BATCH_SIZE],
                                          name="roi_x_floor",
                                          scope=tbe_platform.scope_ubuf)
        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_y[0], 2, 8, 8)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_x[0], 2, 8, 8)

        with tik_instance.for_range(0, roi_bin_grid_h_count) as grid_num_h:
            verify = tik_instance.Scalar(dtype="int32", init_value=0)
            y = tik_instance.Scalar(dtype=self.dtype,
                                    init_value=raw_y[grid_num_h])
            with tik_instance.if_scope(y < -1):
                verify.set_as(1)
            with tik_instance.if_scope(y > self.x_height):
                verify.set_as(1)
            with tik_instance.if_scope(verify == 0):
                y_low = tik_instance.Scalar(dtype="int32",
                                            init_value=y_lo[grid_num_h])
                y_high = tik_instance.Scalar(dtype="int32",
                                             init_value=y_hi[grid_num_h])
                with tik_instance.for_range(
                        0, roi_bin_grid_w_count) as grid_num_w:
                    verify.set_as(0)
                    x = tik_instance.Scalar(dtype=self.dtype,
                                            init_value=raw_x[grid_num_w])
                    with tik_instance.if_scope(x < -1):
                        verify.set_as(1)
                    with tik_instance.if_scope(x > self.x_width):
                        verify.set_as(1)
                    with tik_instance.if_scope(verify == 0):
                        x_low = tik_instance.Scalar(
                            dtype="int32", init_value=x_lo[grid_num_w])
                        x_high = tik_instance.Scalar(
                            dtype="int32", init_value=x_hi[grid_num_w])
                        self._compute_grad_feature_map(
                            y_hi_w, y_lo_w, x_hi_w, x_lo_w, grid_num_h,
                            grid_num_w, grad_output_val, bin_count, index,
                            y_low, x_low, x_high, y_high, delta_x_vector_val,
                            delta_y_vector_val, y, x)
        self._compute_offset_grad(delta_x_vector_val, grad_output_val,
                                  roi_w_cur, bin_count,
                                  grad_offset_one_line_tmp, bin_w_index,
                                  delta_y_vector_val, roi_h_cur)

    def _compute_offset_grad(self, delta_x_vector_val, grad_output_val,
                             roi_w_cur, bin_count, grad_offset_one_line_tmp,
                             bin_w_index, delta_y_vector_val, roi_h_cur):
        tik_instance = self.tik_instance
        n_bust = 2
        dst_reduce = tik_instance.Tensor(self.dtype, (32, ),
                                         name="dst_reduce",
                                         scope=tbe_platform.scope_ubuf)
        tik_instance.vec_dup(32, dst_reduce, 0.0, 1, n_bust)
        work_tensor = tik_instance.Tensor(self.dtype, (self.c1_num, ),
                                          name="work_tensor",
                                          scope=tbe_platform.scope_ubuf)

        tik_instance.vec_mul(16, delta_x_vector_val, delta_x_vector_val,
                             grad_output_val, self.c1_num, n_bust, n_bust,
                             n_bust)
        tik_instance.vec_reduce_add(16, dst_reduce, delta_x_vector_val,
                                    work_tensor, self.c1_num, n_bust)
        tik_instance.vec_muls(16, dst_reduce, dst_reduce, self.gamma, 2,
                              n_bust, n_bust)
        tik_instance.vec_muls(16, dst_reduce, dst_reduce, roi_w_cur, 2, n_bust,
                              n_bust)
        tik_instance.vec_muls(16, dst_reduce, dst_reduce, 1.0 / bin_count, 2,
                              n_bust, n_bust)
        grad_offset_one_line_tmp[bin_w_index, 0].set_as(dst_reduce[0])

        tik_instance.vec_dup(32, dst_reduce, 0.0, 1, n_bust)
        tik_instance.vec_mul(16, delta_y_vector_val, delta_y_vector_val,
                             grad_output_val, self.c1_num, n_bust, n_bust,
                             n_bust)
        tik_instance.vec_reduce_add(16, dst_reduce, delta_y_vector_val,
                                    work_tensor, self.c1_num, n_bust)
        tik_instance.vec_muls(16, dst_reduce, dst_reduce, self.gamma, 2,
                              n_bust, n_bust)
        tik_instance.vec_muls(16, dst_reduce, dst_reduce, roi_h_cur, 2, n_bust,
                              n_bust)
        tik_instance.vec_muls(16, dst_reduce, dst_reduce, 1.0 / bin_count, 2,
                              n_bust, n_bust)
        grad_offset_one_line_tmp[bin_w_index, 1].set_as(dst_reduce[0])

    def _compute_grad_feature_map(self, y_hi_w, y_lo_w, x_hi_w, x_lo_w,
                                  grid_num_h, grid_num_w, grad_output_val,
                                  bin_count, index, y_low, x_low, x_high,
                                  y_high, delta_x_vector_val,
                                  delta_y_vector_val, y, x):
        tik_instance = self.tik_instance
        n_bust = 2
        hy = tik_instance.Scalar(self.dtype, init_value=y_hi_w[grid_num_h])
        ly = tik_instance.Scalar(self.dtype, init_value=y_lo_w[grid_num_h])
        hx = tik_instance.Scalar(self.dtype, init_value=x_hi_w[grid_num_w])
        lx = tik_instance.Scalar(self.dtype, init_value=x_lo_w[grid_num_w])

        w_1 = tik_instance.Scalar(self.dtype, init_value=hy * hx)
        w_2 = tik_instance.Scalar(self.dtype, init_value=hy * lx)
        w_3 = tik_instance.Scalar(self.dtype, init_value=ly * hx)
        w_4 = tik_instance.Scalar(self.dtype, init_value=ly * lx)

        stride = (self.x_height * self.x_width - 1) * n_bust
        tmp_feature_map_grad = tik_instance.Tensor(
            self.dtype, [self.c1_num * 16],
            name="grad_feature_map_val",
            scope=tbe_platform.scope_ubuf)
        tik_instance.vec_muls(16, tmp_feature_map_grad, grad_output_val,
                              w_1 / bin_count, self.c1_num, n_bust, n_bust)

        self._move_one_point_to_gm(tmp_feature_map_grad, index, y_low, x_low,
                                   stride, n_bust)
        input_00 = tik_instance.Tensor(self.dtype, [self.c1_num * 16],
                                       name="input_00",
                                       scope=tbe_platform.scope_ubuf)
        self._move_one_point_to_ub(input_00, n_bust, stride, index, y_low,
                                   x_low)

        tik_instance.vec_muls(16, tmp_feature_map_grad, grad_output_val,
                              w_2 / bin_count, self.c1_num, n_bust, n_bust)
        self._move_one_point_to_gm(tmp_feature_map_grad, index, y_low, x_high,
                                   stride, n_bust)
        input_01 = tik_instance.Tensor(self.dtype, [self.c1_num * 16],
                                       name="input_01",
                                       scope=tbe_platform.scope_ubuf)
        self._move_one_point_to_ub(input_01, n_bust, stride, index, y_low,
                                   x_high)

        tik_instance.vec_muls(16, tmp_feature_map_grad, grad_output_val,
                              w_3 / bin_count, self.c1_num, n_bust, n_bust)
        self._move_one_point_to_gm(tmp_feature_map_grad, index, y_high, x_low,
                                   stride, n_bust)
        input_10 = tik_instance.Tensor(self.dtype, [self.c1_num * 16],
                                       name="input_10",
                                       scope=tbe_platform.scope_ubuf)
        self._move_one_point_to_ub(input_10, n_bust, stride, index, y_high,
                                   x_low)

        tik_instance.vec_muls(16, tmp_feature_map_grad, grad_output_val,
                              w_4 / bin_count, self.c1_num, n_bust, n_bust)
        self._move_one_point_to_gm(tmp_feature_map_grad, index, y_high, x_high,
                                   stride, n_bust)
        input_11 = tik_instance.Tensor(self.dtype, [self.c1_num * 16],
                                       name="input_11",
                                       scope=tbe_platform.scope_ubuf)
        self._move_one_point_to_ub(input_11, n_bust, stride, index, y_high,
                                   x_high)

        tmp_val = tik_instance.Tensor(self.dtype, [self.c1_num * 16],
                                      name="tmp_val",
                                      scope=tbe_platform.scope_ubuf)
        tik_instance.vec_muls(16, tmp_val, input_00, y - y_high, self.c1_num,
                              n_bust, n_bust)
        tik_instance.vec_add(16, delta_x_vector_val, delta_x_vector_val,
                             tmp_val, self.c1_num, n_bust, n_bust, n_bust)
        tik_instance.vec_muls(16, tmp_val, input_01, y_high - y, self.c1_num,
                              n_bust, n_bust)
        tik_instance.vec_add(16, delta_x_vector_val, delta_x_vector_val,
                             tmp_val, self.c1_num, n_bust, n_bust, n_bust)
        tik_instance.vec_muls(16, tmp_val, input_10, y_low - y, self.c1_num,
                              n_bust, n_bust)
        tik_instance.vec_add(16, delta_x_vector_val, delta_x_vector_val,
                             tmp_val, self.c1_num, n_bust, n_bust, n_bust)
        tik_instance.vec_muls(16, tmp_val, input_11, y - y_low, self.c1_num,
                              n_bust, n_bust)
        tik_instance.vec_add(16, delta_x_vector_val, delta_x_vector_val,
                             tmp_val, self.c1_num, n_bust, n_bust, n_bust)

        tik_instance.vec_muls(16, tmp_val, input_00, x - x_high, self.c1_num,
                              n_bust, n_bust)
        tik_instance.vec_add(16, delta_y_vector_val, delta_y_vector_val,
                             tmp_val, self.c1_num, n_bust, n_bust, n_bust)
        tik_instance.vec_muls(16, tmp_val, input_01, x_low - x, self.c1_num,
                              n_bust, n_bust)
        tik_instance.vec_add(16, delta_y_vector_val, delta_y_vector_val,
                             tmp_val, self.c1_num, n_bust, n_bust, n_bust)
        tik_instance.vec_muls(16, tmp_val, input_10, x_high - x, self.c1_num,
                              n_bust, n_bust)
        tik_instance.vec_add(16, delta_y_vector_val, delta_y_vector_val,
                             tmp_val, self.c1_num, n_bust, n_bust, n_bust)
        tik_instance.vec_muls(16, tmp_val, input_11, x - x_low, self.c1_num,
                              n_bust, n_bust)
        tik_instance.vec_add(16, delta_y_vector_val, delta_y_vector_val,
                             tmp_val, self.c1_num, n_bust, n_bust, n_bust)

    def _move_one_point_to_gm(self, tmp_feature_map_grad, index, y, x, stride,
                              n_bust):
        tik_instance = self.tik_instance
        tik_instance.set_atomic_add(1)
        with tik_instance.if_scope(stride <= 65535):
            feature_map_offset = self._get_feature_map_offset(index, 0, y, x)
            tik_instance.data_move(self.grad_fm[feature_map_offset],
                                   tmp_feature_map_grad[0], 0, self.c1_num,
                                   n_bust, 0,
                                   (self.x_height * self.x_width - 1) * n_bust)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, self.c1_num) as c_iter_i:
                feature_map_offset = self._get_feature_map_offset(
                    index, c_iter_i, y, x)
                tik_instance.data_move(self.grad_fm[feature_map_offset],
                                       tmp_feature_map_grad[c_iter_i * 16], 0,
                                       1, n_bust, 0, 0)
        tik_instance.set_atomic_add(0)

    def _move_one_point_to_ub(self, input_tmp, n_bust, stride, index, y, x):
        tik_instance = self.tik_instance
        with tik_instance.if_scope(stride <= 65535):
            feature_map_offset = self._get_feature_map_offset(index, 0, y, x)
            tik_instance.data_move(input_tmp[0],
                                   self.feature_map[feature_map_offset], 0,
                                   self.c1_num, n_bust,
                                   (self.x_height * self.x_width - 1) * n_bust,
                                   0)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, self.c1_num) as c_iter_i:
                feature_map_offset = self._get_feature_map_offset(
                    index, c_iter_i, y, x)
                tik_instance.data_move(input_tmp[16 * c_iter_i],
                                       self.feature_map[feature_map_offset], 0,
                                       1, n_bust, 0, 0)

    def _get_feature_map_offset(self, n, c1, h, w):
        """
        calc feature_map's offset
        """
        n_offset = n * self.c1_num * self.x_height * self.x_width * Constant.C0_SIZE
        c1_offset = c1 * self.x_height * self.x_width * Constant.C0_SIZE
        h_offset = h * self.x_width * Constant.C0_SIZE
        w_offset = w * Constant.C0_SIZE
        return n_offset + c1_offset + h_offset + w_offset

    def _get_grad_output_offset(self, n, c1, h, w):
        """
        calc grad_output offset
        """
        n_offset = n * self.c1_num * self.pooled_height * self.pooled_width * Constant.C0_SIZE
        c1_offset = c1 * self.pooled_height * self.pooled_width * Constant.C0_SIZE
        h_offset = h * self.pooled_width * Constant.C0_SIZE
        w_offset = w * Constant.C0_SIZE
        return n_offset + c1_offset + h_offset + w_offset


# 'pylint: disable=too-many-locals,too-many-arguments
@register_operator("DeformableRoiPoolGrad")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_FLOAT,
    para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
    para_check.KERNEL_NAME)
def deformable_roi_pool_grad(grad_output,
                             feature_map,
                             rois,
                             offset,
                             grad_fm,
                             grad_offset,
                             output_size,
                             spatial_scale,
                             sample_ratio=2,
                             gamma=0.1,
                             kernel_name="deformable_roi_pool_grad"):
    """
    DeformableRoiPoolGrad operator
    """
    deformable_roi_pool_grad_obj = DeformableRoiPoolGrad()
    deformable_roi_pool_grad_obj.pooled_height = output_size[0]
    deformable_roi_pool_grad_obj.pooled_width = output_size[1]
    deformable_roi_pool_grad_obj.spatial_scale = spatial_scale
    deformable_roi_pool_grad_obj.sample_ratio = sample_ratio
    deformable_roi_pool_grad_obj.gamma = gamma
    deformable_roi_pool_grad_obj.kernel_name = kernel_name
    if offset:
        deformable_roi_pool_grad_obj.offset_shape = offset.get('shape')

    return deformable_roi_pool_grad_obj.deformable_roi_pool_grad_compute()
