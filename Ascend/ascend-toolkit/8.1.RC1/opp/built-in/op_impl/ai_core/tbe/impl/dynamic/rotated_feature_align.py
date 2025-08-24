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
rotated_feature_align
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


def ceil_block(value, tiling_dtype):
    """
    if not divide exactly then plus 1
    """
    value *= Constant.TYPE_LEN_DICT.get(tiling_dtype)
    return (value + Constant.BLOCK_BYTE_SIZE - 1) // Constant.BLOCK_BYTE_SIZE


def ceil_value(value_x, value_n):
    """
    if not divide exactly then plus 1
    """
    return (value_x + value_n - 1) // value_n


class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 8
    INT32 = "int32"
    UINT16 = "uint16"
    FLOAT32 = "float32"
    BLOCK_BYTE_SIZE = 32
    TYPE_LEN_DICT = {
        "uint16": 2,
        "float32": 4,
        "int32": 4
    }
    TYPE_NUM_EACH_BLOCK = {
        "float32": 8,
        "int32": 8
    }
    PARAM_NUM_EACH_BOX = 5
    POINT_NUM_EACH_BOX = 5
    CHANNEL = 4
    MAX_FP32 = 2 ** 31 - 1
    HALF = 0.5


class RotatedFeatureAlign():
    """
    define rotated_feature_align object
    """

    def __init__(self, spatial_scale, points, kernel_name):
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.core_num = profile.get_aicore_num()
        self.ub_size = profile.get_unified_buffer_size()
        self.kernel_name = kernel_name
        self.number = 0
        self.channel = 0
        self.height = 0
        self.width = 0
        self.spatial_scale = None
        self.points = None
        self.dtype = Constant.FLOAT32
        self.scalar_dtype = Constant.INT32
        self.tiling_dtype = Constant.INT32
        self.init_scalar()
        self.init_inputs_and_output_gm()
        self.get_tiling_args()
        self.bilinear_nthread = self.points * self.tilem
        self.init_ub_tensor()

    def init_scalar(self):
        self.number = self.tik_instance.Scalar(self.tiling_dtype, name='number')
        self.channel = self.tik_instance.Scalar(self.tiling_dtype, name='channel')
        self.height = self.tik_instance.Scalar(self.tiling_dtype, name='height')
        self.width = self.tik_instance.Scalar(self.tiling_dtype, name='width')
        self.real_core_num = self.tik_instance.Scalar(self.tiling_dtype, name='real_core_num')
        self.tilem = self.tik_instance.Scalar(self.tiling_dtype, name='tilem')
        self.spatial_scale = self.tik_instance.Scalar(self.dtype, name='spatial_scale')
        self.points = self.tik_instance.Scalar(self.tiling_dtype, name='points')
        self.loop_total = self.tik_instance.Scalar(self.scalar_dtype, name='loop_total')
        self.cur_m = self.tik_instance.Scalar(self.scalar_dtype, name='cur_m')
        self.cur_mup = self.tik_instance.Scalar(self.scalar_dtype, name='cur_mup')
        self.m_start = self.tik_instance.Scalar(self.scalar_dtype, name='m_start')
        self.width_sub_one = self.tik_instance.Scalar(self.dtype, name='width_sub_one')
        self.height_sub_one = self.tik_instance.Scalar(self.dtype, name='height_sub_one')
        self.channel_start = self.tik_instance.Scalar(self.scalar_dtype, name='channel_start')
        self.channel_loop = self.tik_instance.Scalar(self.scalar_dtype, name='channel_loop')
        self.x_max = self.tik_instance.Scalar(self.dtype, name='x_max')
        self.x_min = self.tik_instance.Scalar(self.dtype, name='x_min')
        self.y_max = self.tik_instance.Scalar(self.dtype, name='y_max')
        self.y_min = self.tik_instance.Scalar(self.dtype, name='y_min')
        self.load_input_or_not = self.tik_instance.Scalar(self.scalar_dtype, name='load_input_or_not', init_value=1)
        self.x_min_int32 = self.tik_instance.Scalar(self.scalar_dtype, name='x_min_int32')
        self.x_max_int32 = self.tik_instance.Scalar(self.scalar_dtype, name='x_max_int32')
        self.y_min_int32 = self.tik_instance.Scalar(self.scalar_dtype, name='y_min_int32')
        self.y_max_int32 = self.tik_instance.Scalar(self.scalar_dtype, name='y_max_int32')
        self.h_tmp = self.tik_instance.Scalar(self.scalar_dtype, name='h_tmp')
        self.w_tmp = self.tik_instance.Scalar(self.scalar_dtype, name='w_tmp')
        self.w_tmp_32 = self.tik_instance.Scalar(self.scalar_dtype, name='w_tmp_32')
        self.src_stride = self.tik_instance.Scalar(self.scalar_dtype, name='src_stride')
        self.dst_stride = self.tik_instance.Scalar(self.scalar_dtype, name='dst_stride')
        self.param_num_each_box = self.tik_instance.Scalar(self.scalar_dtype, name='param_num_each_box')
        self.buffer_limit = self.tik_instance.Scalar(self.scalar_dtype, name='buffer_limit')

    def init_inputs_and_output_gm(self):
        self.input_x_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_FP32,), name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.input_bboxes_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_FP32,), name="input_bboxes_gm",
                                                        scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_FP32,), name="output_y_gm",
                                                    scope=tik.scope_gm)

    def get_tiling_args(self):
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1,
                                    ceil_block(Constant.TILING_ARG_NUM, self.tiling_dtype), 0, 0)
        self.number.set_as(self.tiling_ub[0])
        self.channel.set_as(self.tiling_ub[1])
        self.height.set_as(self.tiling_ub[2])
        self.width.set_as(self.tiling_ub[3])
        self.real_core_num.set_as(self.tiling_ub[4])
        self.tilem.set_as(self.tiling_ub[5])
        self.spatial_scale.set_as(self.tiling_ub[6])
        self.points.set_as(self.tiling_ub[7])
        
        with self.tik_instance.if_scope(self.points == 1):
            self.param_num_each_box.set_as(2)
            self.buffer_limit.set_as(43000)
        with self.tik_instance.else_scope():
            self.param_num_each_box.set_as(Constant.PARAM_NUM_EACH_BOX)
            self.buffer_limit.set_as(23000)

    def init_ub_tensor(self):
        self.bbox_ub = self.tik_instance.Tensor(self.dtype, (self.param_num_each_box, self.tilem), name="bbox_ub",
                                                scope=tik.scope_ubuf)
        self.input_feature_ub = self.tik_instance.Tensor(self.dtype, (self.buffer_limit,), name="input_feature_ub",
                                                         scope=tik.scope_ubuf)
        self.p_ub = self.tik_instance.Tensor(self.dtype, (16, self.bilinear_nthread), name="p_ub", scope=tik.scope_ubuf)
        self.p_int32_ub = self.p_ub.reinterpret_cast_to(Constant.INT32)[
                          2 * self.bilinear_nthread: 4 * self.bilinear_nthread]
        self.p_fp32_ub = self.tik_instance.Tensor(self.dtype, (2, self.bilinear_nthread), name="p_fp32_ub",
                                                  scope=tik.scope_ubuf)
        self.n_ub = self.p_ub[6 * self.bilinear_nthread: 10 * self.bilinear_nthread]
        self.kn_4phw_ub = self.tik_instance.Tensor(self.dtype, (4, self.bilinear_nthread), name='kn_4phw_ub',
                                                   scope=tik.scope_ubuf)
        self.x_fp32_ub_total = self.p_fp32_ub[0: self.bilinear_nthread]
        self.y_fp32_ub_total = self.p_fp32_ub[self.bilinear_nthread: 2 * self.bilinear_nthread]
        self.x_fp32_ub = self.p_ub[4 * self.bilinear_nthread: 5 * self.bilinear_nthread]
        self.y_fp32_ub = self.p_ub[5 * self.bilinear_nthread: 6 * self.bilinear_nthread]
        self.x_int32_ub = self.p_int32_ub[0: self.bilinear_nthread]
        self.y_int32_ub = self.p_int32_ub[self.bilinear_nthread: 2 * self.bilinear_nthread]
        self.flag_int_ub = self.bbox_ub.reinterpret_cast_to(Constant.UINT16)[0: 96]

    def rotated_feature_align_compute(self):
        self.rotated_feature_align_compute_tiling()

        inputs = [self.input_x_gm,
                  self.input_bboxes_gm]
        outputs = [self.output_y_gm]
        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }
        # add compile info
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.core_num,
            "ub_size": self.ub_size
        })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=inputs,
                                   outputs=outputs,
                                   flowtable=(self.tiling_gm,),
                                   config=opt_config)
        return self.tik_instance

    def rotated_feature_align_compute_tiling(self):
        """
        define rotated_feature_align tiling method
        """
        self.loop_total.set_as((self.width * self.height + self.tilem - 1) / self.tilem)

        core_task_num = self.loop_total
        per_core_task_num = core_task_num // self.core_num
        core_task_tail = core_task_num % self.core_num

        with self.tik_instance.for_range(0, self.real_core_num, block_num=self.real_core_num) as i:
            with self.tik_instance.for_range(0, per_core_task_num, name='j') as j:
                loopi = i * per_core_task_num + j
                self.compute_per_core(loopi)

            with self.tik_instance.if_scope(i < core_task_tail):
                loop_idx = self.core_num * per_core_task_num + i
                self.compute_per_core(loop_idx)

    def compute_per_core(self, loopi):
        """
        define single core computing logic
        """
        with self.tik_instance.if_scope(loopi == self.loop_total - 1):
            self.cur_m.set_as(self.height * self.width - loopi * self.tilem)
        with self.tik_instance.else_scope():
            self.cur_m.set_as(self.tilem)
        self.cur_mup.set_as(self.tilem)
        self.m_start.set_as(loopi * self.tilem)
        with self.tik_instance.for_range(0, self.number) as n:
            self.mul_spatial_scale(n)
            self.sample_points_coor_compute()
            self.bilinear_interpolate(n)

    def mul_spatial_scale(self, n):
        """
        spatial_scale map coordinates in bbox to feature maps
        """
        offset = self.tik_instance.Scalar(self.scalar_dtype)
        offset.set_as(n * Constant.PARAM_NUM_EACH_BOX * self.height * self.width + self.m_start)

        self.src_stride.set_as((self.height * self.width - self.tilem) // 8)
        with self.tik_instance.if_scope(self.src_stride < 0):
            self.src_stride.set_as(0)

        with self.tik_instance.if_scope(tik.all(self.cur_m == self.cur_mup, self.height * self.width % 8 == 0)):
            self.tik_instance.data_move(self.bbox_ub, self.input_bboxes_gm[offset], 0, self.param_num_each_box,
                                        self.tilem // 8, self.src_stride, 0)
        with self.tik_instance.elif_scope(
                tik.all(self.height * self.width > self.tilem, self.cur_m < self.cur_mup)):
            with self.tik_instance.for_range(0, self.param_num_each_box) as i:
                self.tik_instance.data_move(self.bbox_ub[i * self.tilem],
                                            self.input_bboxes_gm[(n * Constant.PARAM_NUM_EACH_BOX + i + 1) * \
                                                                 self.height * self.width - self.tilem],
                                            0, 1, self.tilem // 8, 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.param_num_each_box) as i:
                self.tik_instance.data_move(self.bbox_ub[i * self.tilem],
                                            self.input_bboxes_gm[offset + i * self.width * self.height],
                                            0, 1, self.tilem // 8, 0, 0)

        with self.tik_instance.if_scope(self.points == 5):
            self.tik_instance.vmuls(64, self.bbox_ub, self.bbox_ub, self.spatial_scale,
                                    (self.param_num_each_box - 1) * self.tilem // 64, 1, 1, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vmuls(64, self.bbox_ub, self.bbox_ub, self.spatial_scale,
                                    self.param_num_each_box * self.tilem // 64, 1, 1, 8, 8)

    def sample_points_coor_compute(self):
        """
        calculate the specific coordinates of the sampling point
        """
        self.tik_instance.data_move(self.p_ub, self.bbox_ub[self.tilem], 0, 1, self.tilem // 8, 0, 0)
        self.tik_instance.data_move(self.p_ub[self.bilinear_nthread], self.bbox_ub, 0, 1, self.tilem // 8, 0, 0)

        with self.tik_instance.if_scope(self.points == 5):
            self.corner_points_coor_compute()
        self.hw_bound_process()

    def corner_points_coor_compute(self):
        """
        calculate the coordinates of four corner points
        """
        self.tik_instance.vmuls(64, self.bbox_ub[2 * self.tilem], self.bbox_ub[2 * self.tilem], Constant.HALF,
                                2 * self.tilem // 64, 1, 1, 8, 8)
        a_ub = self.bbox_ub[4 * self.tilem:]
        cosa_ub = self.bbox_ub[0: self.tilem]
        sina_ub = self.bbox_ub[self.tilem: 2 * self.tilem]
        self.tik_instance.h_cos(cosa_ub, a_ub)
        self.tik_instance.h_sin(sina_ub, a_ub)

        wx_wy_hx_hy_ub = self.p_ub[2 * self.bilinear_nthread: (2 * self.bilinear_nthread + 4 * self.tilem)]
        self.tik_instance.vmul(64, wx_wy_hx_hy_ub, cosa_ub, self.bbox_ub[2 * self.tilem], self.tilem // 64,
                               1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(64, wx_wy_hx_hy_ub[self.tilem], sina_ub, self.bbox_ub[2 * self.tilem],
                               self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(64, wx_wy_hx_hy_ub[2 * self.tilem], sina_ub, self.bbox_ub[3 * self.tilem],
                               self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(64, wx_wy_hx_hy_ub[2 * self.tilem], wx_wy_hx_hy_ub[2 * self.tilem], -1,
                                self.tilem // 64, 1, 1, 8, 8)
        self.tik_instance.vmul(64, wx_wy_hx_hy_ub[3 * self.tilem], cosa_ub, self.bbox_ub[3 * self.tilem],
                               self.tilem // 64, 1, 1, 1, 8, 8, 8)

        roi_x_add_wx_ub = self.p_ub[3 * self.bilinear_nthread: (3 * self.bilinear_nthread + self.tilem)]
        roi_y_add_wy_ub = self.p_ub[(3 * self.bilinear_nthread + self.tilem): (
                3 * self.bilinear_nthread + 2 * self.tilem)]
        roi_x_red_wx_ub = self.p_ub[(3 * self.bilinear_nthread + 2 * self.tilem): (
                3 * self.bilinear_nthread + 3 * self.tilem)]
        roi_y_red_wy_ub = self.p_ub[(3 * self.bilinear_nthread + 3 * self.tilem): (
                3 * self.bilinear_nthread + 4 * self.tilem)]

        self.tik_instance.vadd(64, roi_x_add_wx_ub, self.p_ub, wx_wy_hx_hy_ub, self.tilem // 64, 1, 1, 1, 8, 8,
                               8)
        self.tik_instance.vadd(64, roi_y_add_wy_ub, self.p_ub[self.bilinear_nthread],
                               wx_wy_hx_hy_ub[self.tilem], self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, roi_x_red_wx_ub, self.p_ub, wx_wy_hx_hy_ub, self.tilem // 64, 1, 1, 1, 8, 8,
                               8)
        self.tik_instance.vsub(64, roi_y_red_wy_ub, self.p_ub[self.bilinear_nthread],
                               wx_wy_hx_hy_ub[self.tilem], self.tilem // 64, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadd(64, self.p_ub[self.tilem], roi_x_add_wx_ub, wx_wy_hx_hy_ub[2 * self.tilem],
                               self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(64, self.p_ub[2 * self.tilem], roi_x_red_wx_ub,
                               wx_wy_hx_hy_ub[2 * self.tilem], self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, self.p_ub[3 * self.tilem], roi_x_red_wx_ub,
                               wx_wy_hx_hy_ub[2 * self.tilem], self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, self.p_ub[4 * self.tilem], roi_x_add_wx_ub,
                               wx_wy_hx_hy_ub[2 * self.tilem], self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(64, self.p_ub[(Constant.POINT_NUM_EACH_BOX + 1) * self.tilem], roi_y_add_wy_ub,
                               wx_wy_hx_hy_ub[3 * self.tilem], self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(64, self.p_ub[(Constant.POINT_NUM_EACH_BOX + 2) * self.tilem], roi_y_red_wy_ub,
                               wx_wy_hx_hy_ub[3 * self.tilem], self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, self.p_ub[(Constant.POINT_NUM_EACH_BOX + 3) * self.tilem], roi_y_red_wy_ub,
                               wx_wy_hx_hy_ub[3 * self.tilem], self.tilem // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, self.p_ub[(Constant.POINT_NUM_EACH_BOX + 4) * self.tilem], roi_y_add_wy_ub,
                               wx_wy_hx_hy_ub[3 * self.tilem], self.tilem // 64, 1, 1, 1, 8, 8, 8)

    def hw_bound_process(self):
        """
        boundary processing on the h and w dimensions
        """
        width_fp32 = self.tik_instance.Scalar(self.dtype, init_value=self.width)
        height_fp32 = self.tik_instance.Scalar(self.dtype, init_value=self.height)
        self.width_sub_one.set_as(self.width - 1)
        self.height_sub_one.set_as(self.height - 1)
        flag_bound_ub = self.bbox_ub.reinterpret_cast_to(Constant.UINT16)[0:192]
        self.tik_instance.vcmpvs_lt(flag_bound_ub, self.p_ub, -1, 2 * self.bilinear_nthread // 64, 1, 8)
        self.tik_instance.vcmpvs_ge(flag_bound_ub[96], self.p_ub, 0, 2 * self.bilinear_nthread // 64, 1, 8)
        self.tik_instance.vec_or(80, flag_bound_ub, flag_bound_ub, flag_bound_ub[96], 1, 8, 8, 8)
        self.tik_instance.vec_sel(64, 1, self.p_ub, flag_bound_ub, self.p_ub, 0, 2 * self.bilinear_nthread // 64, 8, 8,
                                  8)

        self.tik_instance.vcmpvs_gt(flag_bound_ub, self.p_ub, width_fp32, self.bilinear_nthread // 64, 1, 8)
        self.tik_instance.vcmpvs_le(flag_bound_ub[48], self.p_ub, self.width_sub_one, self.bilinear_nthread // 64, 1, 8)
        self.tik_instance.vec_or(40, flag_bound_ub, flag_bound_ub, flag_bound_ub[48], 1, 8, 8, 8)
        self.tik_instance.vec_sel(64, 1, self.p_ub, flag_bound_ub, self.p_ub, self.width_sub_one,
                                  self.bilinear_nthread // 64, 8, 8, 8)

        self.tik_instance.vcmpvs_gt(flag_bound_ub, self.p_ub[self.bilinear_nthread], height_fp32,
                                    self.bilinear_nthread // 64, 1, 8)
        self.tik_instance.vcmpvs_le(flag_bound_ub[48], self.p_ub[self.bilinear_nthread],
                                    self.height_sub_one, self.bilinear_nthread // 64, 1, 8)
        self.tik_instance.vec_or(40, flag_bound_ub, flag_bound_ub, flag_bound_ub[48], 1, 8, 8, 8)
        self.tik_instance.vec_sel(64, 1, self.p_ub[self.bilinear_nthread], flag_bound_ub,
                                  self.p_ub[self.bilinear_nthread], self.height_sub_one, self.bilinear_nthread // 64, 8,
                                  8, 8)

    def bilinear_interpolate(self, n):
        """
        calculate eigenvalues using bilinear interpolation method
        """
        self.cal_kn_matrix()
        self.cal_move_area(n)
        channel_loops = self.tik_instance.Scalar(self.scalar_dtype)
        channel_loops.set_as(ceil_value(self.channel, Constant.CHANNEL))
        with self.tik_instance.for_range(0, channel_loops) as c_index:
            self.channel_start.set_as(c_index * Constant.CHANNEL)
            with self.tik_instance.if_scope(c_index == (channel_loops - 1)):
                self.channel_loop.set_as(self.channel - c_index * Constant.CHANNEL)
            with self.tik_instance.else_scope():
                self.channel_loop.set_as(Constant.CHANNEL)
            pixel_tmp_ub = self.gather_from_image(n)
            pixel_val_fp32_ub = self.dot_product_and_reduce(pixel_tmp_ub)
            self.move_out(n, pixel_val_fp32_ub)

    def cal_kn_matrix(self):
        """
        calculate the multiplication of matrix k and matrix n
        """
        self.tik_instance.vec_conv(64, "floor", self.p_int32_ub, self.p_ub, 2 * self.bilinear_nthread // 64, 8, 8)

        self.tik_instance.vec_dup(64, self.n_ub, 1, self.bilinear_nthread // 64, 8)
        self.tik_instance.vec_conv(64, "none", self.p_fp32_ub, self.p_int32_ub, 2 * self.bilinear_nthread // 64, 8, 8)

        self.tik_instance.vsub(64, self.n_ub[self.bilinear_nthread], self.p_ub, self.p_fp32_ub,
                               2 * self.bilinear_nthread // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(64, self.n_ub[3 * self.bilinear_nthread], self.n_ub[self.bilinear_nthread],
                               self.n_ub[2 * self.bilinear_nthread], self.bilinear_nthread // 64, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(self.kn_4phw_ub, self.n_ub, 0, 1, 4 * self.bilinear_nthread // 8, 0, 0)
        self.tik_instance.vsub(64, self.kn_4phw_ub, self.kn_4phw_ub, self.n_ub[self.bilinear_nthread],
                               self.bilinear_nthread // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, self.kn_4phw_ub, self.kn_4phw_ub, self.n_ub[2 * self.bilinear_nthread],
                               self.bilinear_nthread // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(64, self.kn_4phw_ub, self.kn_4phw_ub, self.n_ub[3 * self.bilinear_nthread],
                               self.bilinear_nthread // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, self.kn_4phw_ub[self.bilinear_nthread], self.kn_4phw_ub[self.bilinear_nthread],
                               self.n_ub[3 * self.bilinear_nthread], self.bilinear_nthread // 64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, self.kn_4phw_ub[2 * self.bilinear_nthread],
                               self.kn_4phw_ub[2 * self.bilinear_nthread],
                               self.n_ub[3 * self.bilinear_nthread], self.bilinear_nthread // 64, 1, 1, 1, 8, 8, 8)

    def cal_move_area(self, n):
        """
        calculate the area to be moved
        """
        offset1_fp32 = self.tik_instance.Scalar(Constant.FLOAT32)
        offset2_fp32 = self.tik_instance.Scalar(Constant.FLOAT32)
        offset1_fp32.set_as(self.x_fp32_ub_total)
        offset2_fp32.set_as(self.y_fp32_ub_total)
        with self.tik_instance.if_scope(self.height * self.width < self.tilem):
            with self.tik_instance.for_range(0, self.points) as p:
                with self.tik_instance.for_range(self.cur_m, self.tilem) as i:
                    self.x_fp32_ub_total[p * self.tilem + i].set_as(offset1_fp32)
                    self.y_fp32_ub_total[p * self.tilem + i].set_as(offset2_fp32)

        x_max_fp32_ub = self.bbox_ub[0: 20]
        x_min_fp32_ub = self.bbox_ub[20: 40]
        y_max_fp32_ub = self.bbox_ub[40: 60]
        y_min_fp32_ub = self.bbox_ub[60: 80]
        work_tensor_ub = self.bbox_ub[80: 100]
        self.tik_instance.vec_reduce_max(64, x_max_fp32_ub, self.x_fp32_ub_total, work_tensor_ub,
                                         self.bilinear_nthread // 64, 8)
        self.tik_instance.vec_reduce_min(64, x_min_fp32_ub, self.x_fp32_ub_total, work_tensor_ub,
                                         self.bilinear_nthread // 64, 8)
        self.tik_instance.vec_reduce_max(64, y_max_fp32_ub, self.y_fp32_ub_total, work_tensor_ub,
                                         self.bilinear_nthread // 64, 8)
        self.tik_instance.vec_reduce_min(64, y_min_fp32_ub, self.y_fp32_ub_total, work_tensor_ub,
                                         self.bilinear_nthread // 64, 8)
        self.x_max.set_as(x_max_fp32_ub[0])
        self.x_min.set_as(x_min_fp32_ub[0])
        self.y_max.set_as(y_max_fp32_ub[0])
        self.y_min.set_as(y_min_fp32_ub[0])
        self.x_max.set_as(self.x_max + 1.0)
        self.y_max.set_as(self.y_max + 1.0)
        self.tik_instance.scalar_min(self.x_max, self.width_sub_one, self.x_max)
        self.tik_instance.scalar_max(self.x_min, 0, self.x_min)
        self.tik_instance.scalar_min(self.y_max, self.height_sub_one, self.y_max)
        self.tik_instance.scalar_max(self.y_min, 0, self.y_min)

        with self.tik_instance.if_scope(tik.any(self.x_min > (self.width - 1),
                                                self.x_max < 0,
                                                self.y_min > (self.height - 1),
                                                self.y_max < 0)):
            self.load_input_or_not.set_as(0)

    def gather_from_image(self, n):
        """
        Generate a new feature map from the original feature map
        """
        self.tik_instance.vec_dup(self.bilinear_nthread // 16, self.flag_int_ub, 0, 1, 8)
        gather_index = self.p_ub.reinterpret_cast_to(Constant.INT32)
        gather_index_fp32 = self.p_ub
        pixel_tmp_ub = self.tik_instance.Tensor(self.dtype, (4, self.channel_loop, self.bilinear_nthread),
                                                name='pixel_tmp_ub', scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(self.load_input_or_not == 1):
            self.tik_instance.scalar_conv('floor', self.x_min_int32, self.x_min)
            self.tik_instance.scalar_conv('floor', self.x_max_int32, self.x_max)
            self.tik_instance.scalar_conv('floor', self.y_min_int32, self.y_min)
            self.tik_instance.scalar_conv('floor', self.y_max_int32, self.y_max)

            self.h_tmp.set_as(self.y_max_int32 - self.y_min_int32 + 1)
            self.w_tmp.set_as(self.x_max_int32 - self.x_min_int32 + 1)
            self.w_tmp_32.set_as(ceil_block(self.w_tmp, self.dtype) * Constant.TYPE_NUM_EACH_BLOCK.get(self.dtype))

            with self.tik_instance.if_scope(self.h_tmp * self.w_tmp_32 * self.channel_loop <= self.buffer_limit):
                pixel_tmp_ub = self.move_input_feature_gather(n, gather_index, gather_index_fp32, pixel_tmp_ub)
            with self.tik_instance.else_scope():
                pixel_tmp_ub = self.move_input_feature_set_as(n, gather_index, gather_index_fp32, pixel_tmp_ub)
        return pixel_tmp_ub

    def move_input_feature_gather(self, n, gather_index, gather_index_fp32, pixel_tmp_ub):
        """
        move feature map from gm through vgather
        """
        self.tik_instance.vec_adds(64, self.x_fp32_ub, self.x_fp32_ub_total, -1 * self.x_min,
                                   self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.vec_adds(64, self.y_fp32_ub, self.y_fp32_ub_total, -1 * self.y_min,
                                   self.bilinear_nthread // 64, 8, 8)

        self.move_input_feature(n)

        cmp_ub = self.p_ub[10 * self.bilinear_nthread: 13 * self.bilinear_nthread]
        self.tik_instance.vec_dup(64, cmp_ub, 0, ceil_value(self.bilinear_nthread, 64), 8)
        self.tik_instance.vec_dup(64, cmp_ub[self.bilinear_nthread], self.h_tmp - 1,
                                  ceil_value(self.bilinear_nthread, 64), 8)
        self.tik_instance.vec_dup(64, cmp_ub[2 * self.bilinear_nthread], self.w_tmp - 1,
                                  ceil_value(self.bilinear_nthread, 64), 8)

        self.tik_instance.vec_cmpv_ge(self.flag_int_ub, self.y_fp32_ub, cmp_ub, self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.vec_cmpv_le(self.flag_int_ub[48], self.y_fp32_ub, cmp_ub[self.bilinear_nthread],
                                      self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.vec_and(40, self.flag_int_ub, self.flag_int_ub, self.flag_int_ub[48], 1, 8, 8, 8)
        self.tik_instance.vec_cmpv_ge(self.flag_int_ub[48], self.x_fp32_ub, cmp_ub, self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.vec_and(40, self.flag_int_ub, self.flag_int_ub, self.flag_int_ub[48], 1, 8, 8, 8)
        self.tik_instance.vec_cmpv_le(self.flag_int_ub[48], self.x_fp32_ub, cmp_ub[2 * self.bilinear_nthread],
                                      self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.vec_and(40, self.flag_int_ub, self.flag_int_ub, self.flag_int_ub[48], 1, 8, 8, 8)

        # get index of (x, y),(xp1, y),(x, yp1),(xp1, yp1)
        self.tik_instance.vec_sel(64, 1, self.y_fp32_ub, self.flag_int_ub, self.y_fp32_ub, 0,
                                  self.bilinear_nthread // 64, 8, 8, 8)
        self.tik_instance.vec_sel(64, 1, self.x_fp32_ub, self.flag_int_ub, self.x_fp32_ub, 0,
                                  self.bilinear_nthread // 64, 8, 8, 8)

        w_tmp_32_fp32 = self.tik_instance.Scalar(Constant.FLOAT32, init_value=self.w_tmp_32)
        h_tmp_fp32 = self.tik_instance.Scalar(Constant.FLOAT32, init_value=self.h_tmp)
        self.tik_instance.vmuls(64, self.y_fp32_ub, self.y_fp32_ub, w_tmp_32_fp32, self.bilinear_nthread // 64, 1, 1, 8,
                                8)
        self.tik_instance.vadd(64, gather_index_fp32, self.y_fp32_ub, self.x_fp32_ub, self.bilinear_nthread // 64, 1, 1,
                               1, 8, 8, 8)
        self.tik_instance.vadds(64, gather_index_fp32[self.channel_loop * self.bilinear_nthread], gather_index_fp32, 1,
                                self.bilinear_nthread // 64, 1, 1, 8, 8)
        self.tik_instance.vadds(64, gather_index_fp32[2 * self.channel_loop * self.bilinear_nthread], gather_index_fp32,
                                w_tmp_32_fp32, self.bilinear_nthread // 64, 1, 1, 8, 8)
        self.tik_instance.vadds(64, gather_index_fp32[3 * self.channel_loop * self.bilinear_nthread],
                                gather_index_fp32[2 * self.channel_loop * self.bilinear_nthread],
                                1, self.bilinear_nthread // 64, 1, 1, 8, 8)
        with self.tik_instance.for_range(0, 4) as i:
            with self.tik_instance.for_range(1, self.channel_loop) as c:
                self.tik_instance.vadds(64, gather_index_fp32[((i * self.channel_loop) + c) * self.bilinear_nthread],
                                        gather_index_fp32[i * self.channel_loop * self.bilinear_nthread],
                                        c * h_tmp_fp32 * w_tmp_32_fp32, self.bilinear_nthread // 64, 1, 1, 8, 8)
        self.tik_instance.vmuls(64, gather_index_fp32, gather_index_fp32, 4,
                                4 * self.channel_loop * self.bilinear_nthread // 64, 1, 1, 8, 8)
        self.tik_instance.vec_conv(64, 'round', gather_index, gather_index_fp32,
                                   4 * self.channel_loop * self.bilinear_nthread // 64, 8, 8)

        self.tik_instance.vgather(64, pixel_tmp_ub, self.input_feature_ub, gather_index,
                                  4 * self.channel_loop * self.bilinear_nthread // 64, 8, 0, 0)
        return pixel_tmp_ub

    def move_input_feature(self, n):
        gm_in_offset = self.tik_instance.Scalar(self.scalar_dtype)
        gm_in_offset.set_as(
            ((n * self.channel + self.channel_start) * self.height + self.y_min_int32) * self.width + self.x_min_int32)

        self.src_stride.set_as(ceil_block(self.width - self.w_tmp_32, self.dtype))
        with self.tik_instance.if_scope(self.src_stride < 0):
            self.src_stride.set_as(0)

        offset1 = self.tik_instance.Scalar(self.scalar_dtype)
        offset2 = self.tik_instance.Scalar(self.scalar_dtype)
        with self.tik_instance.if_scope(self.width % Constant.TYPE_NUM_EACH_BLOCK.get(self.dtype) == 0):
            with self.tik_instance.for_range(0, self.channel_loop) as c:
                offset1.set_as(c * self.h_tmp * self.w_tmp_32)
                offset2.set_as(gm_in_offset + c * self.height * self.width)
                self.tik_instance.data_move(self.input_feature_ub[offset1], self.input_x_gm[offset2], 0, self.h_tmp,
                                            ceil_block(self.w_tmp_32, self.dtype),
                                            self.src_stride, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.channel_loop) as c:
                with self.tik_instance.for_range(0, self.h_tmp) as h:
                    offset1.set_as((c * self.h_tmp + h) * self.w_tmp_32)
                    offset2.set_as(gm_in_offset + (c * self.height + h) * self.width)
                    self.tik_instance.data_move(self.input_feature_ub[offset1], self.input_x_gm[offset2],
                                                0, 1, ceil_block(self.w_tmp_32, self.dtype), 0, 0)

    def move_input_feature_set_as(self, n, gather_index, gather_index_fp32, pixel_tmp_ub):
        """
        move feature map from gm through set_as
        """
        self.tik_instance.data_move(self.x_fp32_ub, self.x_fp32_ub_total, 0, 1,
                                    ceil_block(self.bilinear_nthread, self.dtype), 0, 0)
        self.tik_instance.data_move(self.y_fp32_ub, self.y_fp32_ub_total, 0, 1,
                                    ceil_block(self.bilinear_nthread, self.dtype), 0, 0)

        width_fp32 = self.tik_instance.Scalar(self.dtype, init_value=self.width)
        height_fp32 = self.tik_instance.Scalar(self.dtype, init_value=self.height)
        cmp_ub = self.p_ub[10 * self.bilinear_nthread: 13 * self.bilinear_nthread]
        self.tik_instance.vec_dup(64, cmp_ub, 0, ceil_value(self.bilinear_nthread, 64), 8)
        self.tik_instance.vec_dup(64, cmp_ub[self.bilinear_nthread], height_fp32 - 1,
                                  ceil_value(self.bilinear_nthread, 64), 8)
        self.tik_instance.vec_dup(64, cmp_ub[2 * self.bilinear_nthread], width_fp32 - 1,
                                  ceil_value(self.bilinear_nthread, 64), 8)

        self.tik_instance.vec_cmpv_ge(self.flag_int_ub, self.y_fp32_ub, cmp_ub, self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.vec_cmpv_le(self.flag_int_ub[48], self.y_fp32_ub, cmp_ub[self.bilinear_nthread],
                                      self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.vec_and(40, self.flag_int_ub, self.flag_int_ub, self.flag_int_ub[48], 1, 8, 8, 8)
        self.tik_instance.vec_cmpv_ge(self.flag_int_ub[48], self.x_fp32_ub, cmp_ub, self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.vec_and(40, self.flag_int_ub, self.flag_int_ub, self.flag_int_ub[48], 1, 8, 8, 8)
        self.tik_instance.vec_cmpv_le(self.flag_int_ub[48], self.x_fp32_ub, cmp_ub[2 * self.bilinear_nthread],
                                      self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.vec_and(40, self.flag_int_ub, self.flag_int_ub, self.flag_int_ub[48], 1, 8, 8, 8)

        self.tik_instance.vec_sel(64, 1, self.y_fp32_ub, self.flag_int_ub, self.y_fp32_ub,
                                  0, self.bilinear_nthread // 64, 8, 8, 8)
        self.tik_instance.vec_sel(64, 1, self.x_fp32_ub, self.flag_int_ub, self.x_fp32_ub,
                                  0, self.bilinear_nthread // 64, 8, 8, 8)

        # get index of (x, y),(xp1, y),(x, yp1),(xp1, yp1)
        self.tik_instance.vmuls(64, self.y_fp32_ub, self.y_fp32_ub, width_fp32, self.bilinear_nthread // 64, 1, 1,
                                8, 8)
        self.tik_instance.vadd(64, gather_index_fp32, self.y_fp32_ub, self.x_fp32_ub, self.bilinear_nthread // 64, 1, 1,
                               1, 8, 8, 8)
        self.tik_instance.vadds(64, gather_index_fp32[self.channel_loop * self.bilinear_nthread], gather_index_fp32, 1,
                                self.bilinear_nthread // 64, 1, 1, 8, 8)
        self.tik_instance.vadds(64, gather_index_fp32[2 * self.channel_loop * self.bilinear_nthread], gather_index_fp32,
                                width_fp32, self.bilinear_nthread // 64, 1, 1, 8, 8)
        self.tik_instance.vadds(64, gather_index_fp32[3 * self.channel_loop * self.bilinear_nthread],
                                gather_index_fp32[2 * self.channel_loop * self.bilinear_nthread],
                                1, self.bilinear_nthread // 64, 1, 1, 8, 8)
        with self.tik_instance.for_range(0, 4) as i:
            with self.tik_instance.for_range(1, self.channel_loop) as c:
                self.tik_instance.vadds(64, gather_index_fp32[((i * self.channel_loop) + c) * self.bilinear_nthread],
                                        gather_index_fp32[i * self.channel_loop * self.bilinear_nthread],
                                        c * height_fp32 * width_fp32, self.bilinear_nthread // 64, 1, 1, 8, 8)
        self.tik_instance.vec_conv(64, 'round', gather_index, gather_index_fp32,
                                   4 * self.channel_loop * self.bilinear_nthread // 64, 8, 8)
        self.tik_instance.h_add(gather_index, gather_index,
                                (n * self.channel + self.channel_start) * self.height * self.width)
        pixel_tmp_ub = self.get_feature_value_set_as(pixel_tmp_ub, gather_index)
        return pixel_tmp_ub

    def get_feature_value_set_as(self, pixel_tmp_ub, gather_index):
        """
        get feature value by set_as
        """
        index = self.tik_instance.Scalar(self.scalar_dtype)
        with self.tik_instance.if_scope(self.height * self.width < self.tilem):
            pixel_tmp_ub = self.get_feature_value_set_as_short_case(index, pixel_tmp_ub, gather_index)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, 4 * self.channel_loop * self.bilinear_nthread) as offset:
                index.set_as(gather_index[offset])
                pixel_tmp_ub[offset].set_as(self.input_x_gm[index])
        return pixel_tmp_ub

    def get_feature_value_set_as_short_case(self, index, pixel_tmp_ub, gather_index):
        with self.tik_instance.for_range(0, 4 * self.channel_loop) as i:
            with self.tik_instance.for_range(0, self.points) as p:
                with self.tik_instance.for_range(0, self.height * self.width) as j:
                    offset = i * self.bilinear_nthread + p * self.tilem + j
                    index.set_as(gather_index[offset])
                    pixel_tmp_ub[offset].set_as(self.input_x_gm[index])
        return pixel_tmp_ub

    def dot_product_and_reduce(self, pixel_tmp_ub):
        """
        point multiplication and reduce operations for the new feature map
        """
        offset1 = self.tik_instance.Scalar(self.scalar_dtype)
        offset2 = self.tik_instance.Scalar(self.scalar_dtype)
        with self.tik_instance.for_range(0, 4) as i:
            offset2.set_as(i * self.bilinear_nthread)
            with self.tik_instance.for_range(0, self.channel_loop) as c:
                offset1.set_as(((i * self.channel_loop) + c) * self.bilinear_nthread)
                self.tik_instance.vmul(64, pixel_tmp_ub[offset1], pixel_tmp_ub[offset1], self.kn_4phw_ub[offset2],
                                       self.bilinear_nthread // 64, 1, 1, 1, 8, 8, 8)

        # reduce(4,c,phw)->(1,c,phw)
        with self.tik_instance.for_range(1, 4) as i:
            offset1.set_as(i * self.channel_loop * self.bilinear_nthread)
            self.tik_instance.vadd(64, pixel_tmp_ub, pixel_tmp_ub, pixel_tmp_ub[offset1],
                                   self.channel_loop * self.bilinear_nthread // 64, 1, 1, 1, 8, 8, 8)

        pixel_val_fp32_ub = self.p_ub[0: 4 * self.bilinear_nthread]
        with self.tik_instance.for_range(0, self.channel_loop) as c:
            offset1.set_as(c * self.bilinear_nthread)
            self.tik_instance.vec_sel(64, 1, pixel_val_fp32_ub[offset1], self.flag_int_ub, pixel_tmp_ub[offset1],
                                      0, self.bilinear_nthread // 64, 8, 8, 8)

        # reduce(c,p,h,w)->(c,1,h,w)
        with self.tik_instance.for_range(1, self.points) as i:
            offset1.set_as(i * self.tilem)
            with self.tik_instance.if_scope(self.tilem == 128):
                self.tik_instance.vadd(64, pixel_val_fp32_ub, pixel_val_fp32_ub, pixel_val_fp32_ub[offset1],
                                       self.channel_loop, 1, 1, 1, self.bilinear_nthread // 8,
                                       self.bilinear_nthread // 8, self.bilinear_nthread // 8)
                self.tik_instance.vadd(64, pixel_val_fp32_ub[64], pixel_val_fp32_ub[64],
                                       pixel_val_fp32_ub[offset1 + 64],
                                       self.channel_loop, 1, 1, 1, self.bilinear_nthread // 8,
                                       self.bilinear_nthread // 8, self.bilinear_nthread // 8)
            with self.tik_instance.else_scope():
                self.tik_instance.vadd(self.tilem, pixel_val_fp32_ub, pixel_val_fp32_ub, pixel_val_fp32_ub[offset1],
                                       self.channel_loop, 1, 1, 1, self.bilinear_nthread // 8,
                                       self.bilinear_nthread // 8, self.bilinear_nthread // 8)
        return pixel_val_fp32_ub

    def move_out(self, n, pixel_val_fp32_ub):
        """
        move output feature map from ub to gm
        """
        self.src_stride.set_as((self.width * self.height - self.tilem) // 8)
        with self.tik_instance.if_scope(self.src_stride < 0):
            self.src_stride.set_as(0)

        gm_offset_hw = self.tik_instance.Scalar(self.scalar_dtype)
        gm_offset_hw.set_as((n * self.channel + self.channel_start) * self.height * self.width + self.m_start)
        with self.tik_instance.if_scope(tik.all(self.cur_m == self.cur_mup, self.height * self.width % 8 == 0)):
            self.tik_instance.data_move(self.input_feature_ub, self.input_x_gm[gm_offset_hw], 0,
                                        self.channel_loop,
                                        self.tilem // Constant.TYPE_NUM_EACH_BLOCK.get(self.dtype),
                                        self.src_stride, 0)
        with self.tik_instance.elif_scope(
                tik.all(self.height * self.width > self.tilem, self.cur_m < self.cur_mup)):
            with self.tik_instance.for_range(0, self.channel_loop) as c:
                self.tik_instance.data_move(self.input_feature_ub[c * self.tilem],
                                            self.input_x_gm[(n * self.channel + self.channel_start + c + 1) * \
                                                            self.height * self.width - self.tilem],
                                            0, 1, self.tilem // Constant.TYPE_NUM_EACH_BLOCK.get(self.dtype), 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.channel_loop) as c:
                self.tik_instance.data_move(self.input_feature_ub[c * self.tilem],
                                            self.input_x_gm[gm_offset_hw + c * self.width * self.height],
                                            0, 1, self.tilem // Constant.TYPE_NUM_EACH_BLOCK.get(self.dtype), 0, 0)

        pixel_val_fp32_ub = self.accumulate_feature(pixel_val_fp32_ub)

        self.src_stride.set_as(ceil_block(self.bilinear_nthread - self.tilem, self.dtype))
        with self.tik_instance.if_scope(self.src_stride < 0):
            self.src_stride.set_as(0)
        self.dst_stride.set_as(ceil_block(self.height * self.width - self.tilem, self.dtype))
        with self.tik_instance.if_scope(self.dst_stride < 0):
            self.dst_stride.set_as(0)

        with self.tik_instance.if_scope(tik.all(self.cur_m == self.cur_mup, self.height * self.width % 8 == 0)):
            self.tik_instance.data_move(self.output_y_gm[gm_offset_hw], pixel_val_fp32_ub, 0, self.channel_loop,
                                        ceil_block(self.tilem, self.dtype), self.src_stride, self.dst_stride)
        with self.tik_instance.elif_scope(
                tik.all(self.height * self.width > self.tilem, self.cur_m < self.cur_mup)):
            with self.tik_instance.for_range(0, self.channel_loop) as c:
                self.tik_instance.data_move(self.output_y_gm[(n * self.channel + self.channel_start + c + 1) * \
                                                             self.height * self.width - self.tilem],
                                            pixel_val_fp32_ub[c * self.bilinear_nthread], 0, 1,
                                            ceil_block(self.cur_mup, self.dtype), 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.channel_loop) as c:
                self.tik_instance.data_move(self.output_y_gm[gm_offset_hw + c * self.height * self.width],
                                            pixel_val_fp32_ub[c * self.bilinear_nthread], 0, 1,
                                            ceil_block(self.cur_m, self.dtype), 0, 0)

    def accumulate_feature(self, pixel_val_fp32_ub):
        with self.tik_instance.if_scope(self.tilem == 128):
            self.tik_instance.vadd(64, pixel_val_fp32_ub, pixel_val_fp32_ub, self.input_feature_ub,
                                   self.channel_loop, 1, 1, 1, self.bilinear_nthread // 8, self.bilinear_nthread // 8,
                                   self.tilem // 8)
            self.tik_instance.vadd(64, pixel_val_fp32_ub[64], pixel_val_fp32_ub[64], self.input_feature_ub[64],
                                   self.channel_loop, 1, 1, 1, self.bilinear_nthread // 8, self.bilinear_nthread // 8,
                                   self.tilem // 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vadd(self.tilem, pixel_val_fp32_ub, pixel_val_fp32_ub, self.input_feature_ub,
                                   self.channel_loop, 1, 1, 1, self.bilinear_nthread // 8, self.bilinear_nthread // 8,
                                   self.tilem // 8)
        return pixel_val_fp32_ub


# 'pylint: disable=unused-argument,too-many-argument
@register_operator("RotatedFeatureAlign")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def rotated_feature_align(x,
                          bboxes,
                          y,
                          spatial_scale,
                          points=1,
                          kernel_name="rotated_feature_align"):
    """
    RotatedFeatureAlign operator
    """

    rotated_feature_align_obj = RotatedFeatureAlign(spatial_scale, points, kernel_name)
    return rotated_feature_align_obj.rotated_feature_align_compute()
