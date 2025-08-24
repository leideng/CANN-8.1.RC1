# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
ball_query
"""
from tbe.common.platform.platform_info import get_soc_spec

from impl import common_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # MAX ELEMENT NUM OF FP16 IN 1BLOCK
    FP16_ELEMENTS_BLOCK = 16
    # MAX ELEMENT NUM OF FP32 IN 1BLOCK
    FP32_ELEMENTS_BLOCK = 8
    # MAX ELEMENT NUM OF INT32 IN 1BLOCK
    INT32_ELEMENTS_BLOCK = 8
    # CONST CENTER_XYZ SLICE SEGMENT, EQUAL TO NUMBER OF POINTS
    CENTER_XYZ_SEGMENT = 2048 * 3
    # CONST XYZ SLICE SEGMENT, EQUAL TO NUMBER OF CLUSTERS
    XYZ_SEGMENT = 2048
    # CONST RESULT SLICE SEGMENT
    RESULT_SEGMENT = 2048
    MAX_INT32 = 2 ** 31 - 1
    TILING_SCALAR_DTYPE = "int32"
    DATA_TYPE_INT32 = "int32"
    DATA_TYPE_FLOAT32 = "float32"
    MAX_ELEMENTS_INT32 = 64
    TILING_PARAMS_NUM = 8
    SEL_MAX_ELEMENTS = 64


def _apply_mem(tik_instance, dtype,
               shape, name, scope=tik.scope_ubuf):
    """
    apply mem fuc
    :param tik_instance: tik_instance
        tik_instance
    :param dtype: str
        ub dtype
    :param shape: list
        ub shape
    :param name: str
        ub name
    :param scope: scope
        scope_ubuf or scope_gm
    :return: the result tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    """
    Get Ceil Int
    :param int1:int
        input int 1
    :param int2: int
        input int 2
    :return: ceil_int: int
    """
    return (int1 + int2 - 1) // int2


# 'pylint: disable=too-many-instance-attributes,invalid-name
class BallQuery:
    """Function: use to finish Iou main functions
    """

    def __init__(self, xyz, center_xyz, min_radius, max_radius, sample_num):
        """
        init BallQuery
        :param xyz: dict
            shape and dtype of xyz
            shape must be [B, 3, N]
        :param center_xyz: dict
            shape and dtype of center_xyz
            shape must be [M, B, 3]
        :param min_radius: float
            minimum radius of the balls
        :param max_radius: float
            maximum radius of the balls
        :param sample_num: int
            maximum number if features in the balls
        :return: None
        """
        # square the distance to avoid squaring when calculating Euclidean distance
        self.min_radius = min_radius * min_radius
        self.max_radius = max_radius * max_radius

        self.sample_num = sample_num
        self.dtype = xyz.get("dtype").lower()
        self.tik_instance = tik.Tik(disable_debug=True)

        self.full_core_num = get_soc_spec(tbe_platform.CORE_NUM)
        self.available_ub_size = get_soc_spec("UB_SIZE")
        self.product = tbe_platform.api_check_support("tik.vdiv", "float32")

        # generate input and output gm
        self.xyz_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,), name="xyz_gm", scope=tik.scope_gm)
        self.center_xyz_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,),
                                                      name="center_xyz_gm", scope=tik.scope_gm)
        self.idx_gm = self.tik_instance.Tensor(Constant.DATA_TYPE_INT32, (Constant.MAX_INT32,), name="idx_gm",
                                               scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)

        # generate tiling data
        self.input_b = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "input_b")
        self.input_m = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "input_m")
        self.input_n = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "input_n")
        self.m_per_core = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "m_per_core")
        self.tiling_core_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tiling_core_num")
        self.core_tail_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "core_tail_num")
        self.xyz_segment_loop = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "xyz_segment_loop")
        self.xyz_segment_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "xyz_segment_tail")

        # the current number of results when each center_xyz point is calculated
        self.res_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "res_num")
        # for each center_xyz point, the first xyz index value that satisfies the requirement
        self.first_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "first_num")
        # the current number of results in each core
        self.result_offset = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "result_offset", init_value=0)
        # the starting M value of each core calculation
        self.m_start = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "m_start", init_value=0)
        # the number of results that need to be sent to gm
        self.len_segment = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "len_segment", init_value=0)
        # the offset of result gm
        self.gm_offset = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "gm_offset")

        # the center_xyz that needs to be computed in each loop
        self.center_x = self.tik_instance.Scalar(self.dtype, "center_x", init_value=0.0)
        self.center_y = self.tik_instance.Scalar(self.dtype, "center_y", init_value=0.0)
        self.center_z = self.tik_instance.Scalar(self.dtype, "center_z", init_value=0.0)

        self.get_tiling_args()

        if self.dtype == "float16":
            # the length of each center_xyz segment
            self.center_xyz_ub_segment_len = 2 * Constant.CENTER_XYZ_SEGMENT
            # the length of each xyz segment
            self.xyz_ub_segment_len = 2 * Constant.XYZ_SEGMENT
            # the maximum number of elements of fp16 in a block
            self.elements_per_block = Constant.FP16_ELEMENTS_BLOCK
            self.max_elements = Constant.FP16_ELEMENTS_BLOCK * 8
        else:
            self.center_xyz_ub_segment_len = Constant.CENTER_XYZ_SEGMENT
            self.xyz_ub_segment_len = Constant.XYZ_SEGMENT
            self.elements_per_block = Constant.FP32_ELEMENTS_BLOCK
            self.max_elements = Constant.FP32_ELEMENTS_BLOCK * 8

        # init params with none
        self.x = None
        self.y = None
        self.z = None
        self.each_segment_result = None
        self.center_xyz_ub = None
        self.distance_each_segment = None
        self.distance_x_tmp = None
        self.distance_y_tmp = None
        self.distance_z_tmp = None
        self.repeat_center_x_tmp = None
        self.repeat_center_y_tmp = None
        self.repeat_center_z_tmp = None
        self.ub_0_dtype = None
        self.ub_0_float32 = None
        self.ub_1_float32 = None
        self.ub_min_radius = None
        self.ub_max_radius = None
        self.ub_dst_eq = None
        self.ub_dst_ge = None
        self.ub_dst_lt = None
        self.ub_result_eq = None
        self.ub_result_ge = None
        self.ub_result_lt = None

    def get_tiling_args(self):
        """
        get runtime tiling data and set for scalar
        :return: None
        """
        tiling_ub = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            scope=tik.scope_ubuf,
            name="tiling_ub"
        )
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)

        self.input_b.set_as(tiling_ub[0])
        self.input_m.set_as(tiling_ub[1])
        self.input_n.set_as(tiling_ub[2])
        self.m_per_core.set_as(tiling_ub[3])
        self.tiling_core_num.set_as(tiling_ub[4])
        self.core_tail_num.set_as(tiling_ub[5])
        self.xyz_segment_loop.set_as(tiling_ub[6])
        self.xyz_segment_tail.set_as(tiling_ub[7])

    def get_center_xyz_slice(self, segment_loop_index, len_of_segment):
        """ get the xyz value for each segment
        :param segment_loop_index: the index of segment
        :param len_of_segment: len of current segment
        :return: None
        """
        center_xyz_gm_offset = self.m_start * self.input_b * 3 + segment_loop_index * self.center_xyz_ub_segment_len
        burst = _get_ceil_int(len_of_segment, self.elements_per_block)
        self.tik_instance.data_move(self.center_xyz_ub,
                                    self.center_xyz_gm[center_xyz_gm_offset], 0, 1, burst, 0, 0)

    def get_xyz_slice(self, current_b, xyz_segment_loop_index, segment_len):
        """ get the xyz value for each segment
        :param current_b: index of current batch
        :param xyz_segment_loop_index: index of current xyz segment loop
        :param segment_len: len of this segment
        :return: None
        """
        start_of_xyz_gm = current_b * self.input_n * 3
        xyz_gm_offset_x = start_of_xyz_gm + xyz_segment_loop_index * self.xyz_ub_segment_len
        xyz_gm_offset_y = start_of_xyz_gm + xyz_segment_loop_index * self.xyz_ub_segment_len + self.input_n
        xyz_gm_offset_z = start_of_xyz_gm + xyz_segment_loop_index * self.xyz_ub_segment_len + 2 * self.input_n

        burst = _get_ceil_int(segment_len, self.elements_per_block)
        self.tik_instance.data_move(self.x, self.xyz_gm[xyz_gm_offset_x], 0, 1, burst, 0, 0)
        self.tik_instance.data_move(self.y, self.xyz_gm[xyz_gm_offset_y], 0, 1, burst, 0, 0)
        self.tik_instance.data_move(self.z, self.xyz_gm[xyz_gm_offset_z], 0, 1, burst, 0, 0)

    def calculate_distance(self):
        """
        calculate_distance between xyz and center_xyz in each segment
        :return: None
        """
        repeat_time = _get_ceil_int(self.xyz_ub_segment_len, self.max_elements)
        self.tik_instance.vector_dup(self.max_elements, self.repeat_center_x_tmp, self.center_x, repeat_time, 1, 8)
        self.tik_instance.vector_dup(self.max_elements, self.repeat_center_y_tmp, self.center_y, repeat_time, 1, 8)
        self.tik_instance.vector_dup(self.max_elements, self.repeat_center_z_tmp, self.center_z, repeat_time, 1, 8)

        self.tik_instance.vsub(self.max_elements, self.distance_x_tmp, self.x, self.repeat_center_x_tmp,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.max_elements, self.distance_x_tmp, self.distance_x_tmp, self.distance_x_tmp,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.max_elements, self.distance_y_tmp, self.y, self.repeat_center_y_tmp,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.max_elements, self.distance_y_tmp, self.distance_y_tmp, self.distance_y_tmp,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.max_elements, self.distance_z_tmp, self.z, self.repeat_center_z_tmp,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.max_elements, self.distance_z_tmp, self.distance_z_tmp, self.distance_z_tmp,
                               repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadd(self.max_elements, self.distance_each_segment, self.distance_x_tmp, self.distance_y_tmp,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(self.max_elements, self.distance_each_segment, self.distance_each_segment,
                               self.distance_z_tmp, repeat_time, 1, 1, 1, 8, 8, 8)

    def send_result_to_gm(self, force=False):
        """
        send the result to GM
        :param force: if the upper limit is NOT reached, still sent to gm
        :return: None
        """
        with self.tik_instance.if_scope(
                tik.any(force is True, self.result_offset % Constant.RESULT_SEGMENT == 0)):
            # send result from ub to gm
            self.len_segment.set_as(Constant.RESULT_SEGMENT)
            with self.tik_instance.if_scope(self.result_offset % Constant.RESULT_SEGMENT != 0):
                self.len_segment.set_as(self.result_offset % Constant.RESULT_SEGMENT)

            self.gm_offset.set_as(0)
            self.gm_offset.set_as(self.sample_num * self.m_start * self.input_b
                                  + self.result_offset - self.len_segment)

            input_dict = {
                "instance": self.tik_instance,
                "out_ub": self.each_segment_result,
                "out_gm": self.idx_gm,
                "gm_offset": self.gm_offset,
                "element_num": self.len_segment,
                "dsize": 4,
            }
            common_util.move_out_non32_alignment(input_dict)

    def set_result_and_try_send(self, current_n):
        """
        add the result to ub, and if it reaches the maximum length, sent the result to gm
        :param current_n: current value of N
        :return: None
        """
        self.each_segment_result[self.result_offset % Constant.RESULT_SEGMENT].set_as(current_n)
        self.result_offset.set_as(self.result_offset + 1)
        self.send_result_to_gm()

    def get_ball_query_fp32(self, current_n_start, current_segment_len):
        """
        determine whether the distance is within the required range, used when dtype is float32
        :param current_n_start: index of current n
        :param current_segment_len: len of current segment
        :return: None
        """
        with self.tik_instance.for_range(0, current_segment_len, thread_num=2) as cluster_num_index:
            with self.tik_instance.if_scope(self.res_num < self.sample_num):
                distance_scalar = self.distance_each_segment[cluster_num_index]
                current_n = current_n_start + cluster_num_index
                with self.tik_instance.if_scope(tik.any(
                        distance_scalar == 0.0,
                        tik.all(distance_scalar >= self.min_radius,
                                distance_scalar < self.max_radius))):
                    with self.tik_instance.if_scope(self.res_num == 0):
                        self.first_num.set_as(current_n)

                    self.set_result_and_try_send(current_n)
                    self.res_num.set_as(self.res_num + 1)

    def get_ball_query_fp16(self, current_n_start, current_segment_len):
        """
        determine whether the distance is within the required range, used when dtype is float16
        :param current_n_start: index of current n
        :param current_segment_len: len of current segment
        :return: None
        """
        sel_loop_num = _get_ceil_int(current_segment_len, Constant.SEL_MAX_ELEMENTS)
        with self.tik_instance.for_range(0, sel_loop_num) as sel_idx:
            with self.tik_instance.if_scope(self.res_num >= self.sample_num):
                self.tik_instance.tik_break()
            self.tik_instance.vec_cmpv_eq(self.ub_dst_eq,
                                          self.distance_each_segment[sel_idx * Constant.SEL_MAX_ELEMENTS],
                                          self.ub_0_dtype[sel_idx * Constant.SEL_MAX_ELEMENTS], 1, 8, 8)
            self.tik_instance.vec_cmpv_ge(self.ub_dst_ge,
                                          self.distance_each_segment[sel_idx * Constant.SEL_MAX_ELEMENTS],
                                          self.ub_min_radius[sel_idx * Constant.SEL_MAX_ELEMENTS], 1, 8, 8)
            self.tik_instance.vec_cmpv_lt(self.ub_dst_lt,
                                          self.distance_each_segment[sel_idx * Constant.SEL_MAX_ELEMENTS],
                                          self.ub_max_radius[sel_idx * Constant.SEL_MAX_ELEMENTS], 1, 8, 8)

            self.tik_instance.vec_sel(Constant.SEL_MAX_ELEMENTS, 0, self.ub_result_eq, self.ub_dst_eq,
                                      self.ub_1_float32, self.ub_0_float32, 1, 8, 8, 8)
            self.tik_instance.vec_sel(Constant.SEL_MAX_ELEMENTS, 0, self.ub_result_ge, self.ub_dst_ge,
                                      self.ub_1_float32, self.ub_0_float32, 1, 8, 8, 8)
            self.tik_instance.vec_sel(Constant.SEL_MAX_ELEMENTS, 0, self.ub_result_lt, self.ub_dst_lt,
                                      self.ub_1_float32, self.ub_0_float32, 1, 8, 8, 8)

            with self.tik_instance.for_range(0, Constant.SEL_MAX_ELEMENTS) as internal_sel_idx:
                current_cal_num = internal_sel_idx + sel_idx * Constant.SEL_MAX_ELEMENTS
                with self.tik_instance.if_scope(tik.all(current_cal_num < current_segment_len,
                                                        self.res_num < self.sample_num)):
                    each_result = tik.any(self.ub_result_eq[internal_sel_idx] == 1.,
                                          tik.all(self.ub_result_ge[internal_sel_idx] == 1.,
                                                  self.ub_result_lt[internal_sel_idx] == 1.))
                    current_n = current_n_start + current_cal_num
                    with self.tik_instance.if_scope(each_result):
                        with self.tik_instance.if_scope(self.res_num == 0):
                            self.first_num.set_as(current_n)

                        self.set_result_and_try_send(current_n)
                        self.res_num.set_as(self.res_num + 1)

    def get_xyz_slice_and_cal_dis(self, current_b):
        """
        calculate the distance and result between center_xyz and a segment of xyz
        :param current_b: the current value of B
        :return: None
        """
        # input_n may exceed the length of UB, process several times by dividing the segment
        self.res_num.set_as(0)
        self.first_num.set_as(0)  # when none of the values are within the distance range, the default value is 0
        with self.tik_instance.for_range(0, self.xyz_segment_loop) as xyz_segment_loop_index:
            with self.tik_instance.if_scope(self.res_num < self.sample_num):
                segment_len = self.xyz_ub_segment_len
                current_n_start = xyz_segment_loop_index * self.xyz_ub_segment_len
                # get self.x、self.y、self.z of each segment
                self.get_xyz_slice(current_b, xyz_segment_loop_index, segment_len)
                # already get a segment of xyz and self.center_x, self.center_y, self.center_y
                # calculate the distance for each segment
                self.calculate_distance()
                # calculate the ball query results for each segment
                if self.dtype == "float16":
                    self.get_ball_query_fp16(current_n_start, segment_len)
                else:
                    self.get_ball_query_fp32(current_n_start, segment_len)
            with self.tik_instance.else_scope():
                self.tik_instance.tik_break()
        with self.tik_instance.if_scope(self.xyz_segment_tail != 0):
            with self.tik_instance.if_scope(self.res_num < self.sample_num):
                segment_len = self.xyz_segment_tail
                xyz_segment_loop_index = self.xyz_segment_loop
                current_n_start = xyz_segment_loop_index * self.xyz_ub_segment_len
                self.get_xyz_slice(current_b, xyz_segment_loop_index, segment_len)
                self.calculate_distance()
                if self.dtype == "float16":
                    self.get_ball_query_fp16(current_n_start, segment_len)
                else:
                    self.get_ball_query_fp32(current_n_start, segment_len)

        # fill the result so that the number equal to sample_num
        with self.tik_instance.for_range(self.res_num, self.sample_num):
            self.set_result_and_try_send(self.first_num)

    def run_per_cluster(self, segment_loop_index, cluster_index):
        """
        process each center_xyz point
        :param segment_loop_index: index of segment loop
        :param cluster_index: index of cluster
        :return: None
        """
        # get center_x, center_y, center_z in this cluster
        self.center_x.set_as(self.center_xyz_ub[cluster_index * 3 + 0])
        self.center_y.set_as(self.center_xyz_ub[cluster_index * 3 + 1])
        self.center_z.set_as(self.center_xyz_ub[cluster_index * 3 + 2])

        # gets the current value of B, using B to get the corresponding xyz
        current_points_offset = (segment_loop_index * self.center_xyz_ub_segment_len) + cluster_index * 3
        current_b = (current_points_offset // 3) % self.input_b
        # and the current_m is self.m_start + current_points_offset // self.input_b // 3

        # calculate the distance and result between xyz and center_xyz
        self.get_xyz_slice_and_cal_dis(current_b)

    def run_per_core(self, core_id, m_current_core):
        """
        ball query computation for each core
        :param core_id id of each core
        :param m_current_core: the size of M in the current core
        :return: None
        """
        self.m_start.set_as(core_id * self.m_per_core)
        self.result_offset.set_as(0)

        # since m_current_core*b*3 may exceed the length of UB,
        # it needs to be processed several times by dividing the segment
        points_num = m_current_core * self.input_b * 3
        segment_loop = points_num // self.center_xyz_ub_segment_len
        segment_loop_tail = points_num % self.center_xyz_ub_segment_len

        with self.tik_instance.for_range(0, segment_loop) as segment_loop_index:
            # the code guarantees that self.center_xyz_ub_segment_len is divisible by 3
            each_segment_cluster_num = self.center_xyz_ub_segment_len // 3
            # get the center_xyz value of a segment
            self.get_center_xyz_slice(segment_loop_index, self.center_xyz_ub_segment_len)

            with self.tik_instance.for_range(0, each_segment_cluster_num) as cluster_index:
                # solve for each center_xyz points
                self.run_per_cluster(segment_loop_index, cluster_index)

        with self.tik_instance.if_scope(segment_loop_tail != 0):
            # the remainder is also divisible by 3
            each_segment_cluster_num = segment_loop_tail // 3
            segment_loop_index = segment_loop
            self.get_center_xyz_slice(segment_loop_index, segment_loop_tail)

            with self.tik_instance.for_range(0, each_segment_cluster_num) as cluster_index:
                self.run_per_cluster(segment_loop_index, cluster_index)

        # when the loop ends and there is still unsent data in UB, forced to send data
        with self.tik_instance.if_scope(self.result_offset % Constant.RESULT_SEGMENT != 0):
            self.send_result_to_gm(force=True)

    def start_query_cut_by_center_xyz_m(self, core_id):
        """
        cut the M of center_xyz and start multi-core calculations
        :param core_id: int, index of core
        :return:None
        """
        with self.tik_instance.if_scope(core_id < self.tiling_core_num):
            with self.tik_instance.if_scope(self.core_tail_num == 0):
                self.run_per_core(core_id, self.m_per_core)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(core_id == (self.tiling_core_num - 1)):
                    self.run_per_core(core_id, self.core_tail_num)
                with self.tik_instance.else_scope():
                    self.run_per_core(core_id, self.m_per_core)

    def apply_all_ub(self):
        """
        apply for space on all ub data
        :return: None
        """
        # the xyz that needs to be computed in each segment
        self.x = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "x")
        self.y = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "y")
        self.z = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "z")
        # results for each segment
        self.each_segment_result = _apply_mem(
            self.tik_instance, Constant.DATA_TYPE_INT32, [Constant.RESULT_SEGMENT], "each_segment_result")

        # the center_xyz that needs to be computed in each segment
        self.center_xyz_ub = _apply_mem(
            self.tik_instance, self.dtype, [self.center_xyz_ub_segment_len], "center_xyz_ub")
        # the distance between center_xyz and xyz in each segment
        self.distance_each_segment = _apply_mem(
            self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "distance_each_segment")

        self.distance_x_tmp = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "distance_x_tmp")
        self.distance_y_tmp = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "distance_y_tmp")
        self.distance_z_tmp = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "distance_z_tmp")

        self.repeat_center_x_tmp = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len],
                                              "repeat_center_x_tmp")
        self.repeat_center_y_tmp = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len],
                                              "repeat_center_y_tmp")
        self.repeat_center_z_tmp = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len],
                                              "repeat_center_z_tmp")

        self.ub_0_dtype = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "ub_0_dtype")
        self.ub_0_float32 = _apply_mem(self.tik_instance, "float32", [Constant.SEL_MAX_ELEMENTS], "ub_0_float32")
        self.ub_1_float32 = _apply_mem(self.tik_instance, "float32", [Constant.SEL_MAX_ELEMENTS], "ub_1_float32")
        self.ub_min_radius = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "ub_min_radius")
        self.ub_max_radius = _apply_mem(self.tik_instance, self.dtype, [self.xyz_ub_segment_len], "ub_max_radius")

        self.ub_dst_eq = _apply_mem(self.tik_instance, "uint16", [8], "ub_dst_eq")
        self.ub_dst_ge = _apply_mem(self.tik_instance, "uint16", [8], "ub_dst_ge")
        self.ub_dst_lt = _apply_mem(self.tik_instance, "uint16", [8], "ub_dst_lt")

        self.ub_result_eq = _apply_mem(self.tik_instance, "float32", [Constant.SEL_MAX_ELEMENTS], "ub_result_eq")
        self.ub_result_ge = _apply_mem(self.tik_instance, "float32", [Constant.SEL_MAX_ELEMENTS], "ub_result_ge")
        self.ub_result_lt = _apply_mem(self.tik_instance, "float32", [Constant.SEL_MAX_ELEMENTS], "ub_result_lt")

        _repeat_xyz = _get_ceil_int(self.xyz_ub_segment_len, self.max_elements)
        _repeat_center_xyz_ub = _get_ceil_int(self.center_xyz_ub_segment_len, self.max_elements)
        _repeat_segment_result = _get_ceil_int(Constant.RESULT_SEGMENT, Constant.MAX_ELEMENTS_INT32)

        self.tik_instance.vector_dup(self.max_elements, self.x, 0.0, _repeat_xyz, 1, 8)
        self.tik_instance.vector_dup(self.max_elements, self.y, 0.0, _repeat_xyz, 1, 8)
        self.tik_instance.vector_dup(self.max_elements, self.z, 0.0, _repeat_xyz, 1, 8)
        self.tik_instance.vector_dup(self.max_elements, self.center_xyz_ub, 0.0, _repeat_center_xyz_ub, 1, 8)
        self.tik_instance.vector_dup(self.max_elements, self.distance_each_segment, 0.0, _repeat_xyz, 1, 8)
        self.tik_instance.vector_dup(Constant.MAX_ELEMENTS_INT32, self.each_segment_result,
                                     0, _repeat_segment_result, 1, 8)

        self.tik_instance.vector_dup(Constant.SEL_MAX_ELEMENTS, self.ub_0_float32, 0, 1, 1, 8)
        self.tik_instance.vector_dup(Constant.SEL_MAX_ELEMENTS, self.ub_1_float32, 1, 1, 1, 8)
        self.tik_instance.vector_dup(self.max_elements, self.ub_0_dtype, 0.0, _repeat_xyz, 1, 8)
        self.tik_instance.vector_dup(self.max_elements, self.ub_min_radius, self.min_radius, _repeat_xyz, 1, 8)
        self.tik_instance.vector_dup(self.max_elements, self.ub_max_radius, self.max_radius, _repeat_xyz, 1, 8)

    def run_tik(self, kernel_name):
        """
        start tik process and build cce
        :param kernel_name: str
            name of kernel
        :return: tik_instance
            tik instance of ball_query
        """
        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_id:
            self.apply_all_ub()
            self.start_query_cut_by_center_xyz_m(core_id)

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "full_core_num": self.full_core_num,
                "ub_size": self.available_ub_size,
                "product": self.product
            })
        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.xyz_gm, self.center_xyz_gm],
            outputs=[self.idx_gm],
            flowtable=[self.tiling_gm],
            config=opt_config
        )
        return self.tik_instance


# 'pylint: disable = unused-argument,too-many-arguments
@register_operator("BallQuery")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def ball_query(xyz, center_xyz, idx, min_radius, max_radius, sample_num, kernel_name="ball_query"):
    """
    calculating data
    :param xyz: dict
        shape and dtype of xyz
        shape must be [B, 3, N]
    :param center_xyz: dict
        shape and dtype of center_xyz
        shape must be [M, B, 3]
    :param idx: dict
        shape and dtype of output
        shape must be [M, B, sample_num]
    :param min_radius: float
        minimum radius of the balls
    :param max_radius: float
        maximum radius of the balls
    :param sample_num: int
        maximum number if features in the balls
    :param kernel_name: str
        kernel name, default value is "ball_query"
    :return: None
    """
    xyz_shape = xyz.get("shape")
    center_xyz_shape = center_xyz.get("shape")

    para_check.check_shape(xyz_shape, param_name="xyz")
    para_check.check_shape(center_xyz_shape, param_name="center_xyz")

    xyz_dtype = xyz.get("dtype").lower()
    shape_util.compare_tensor_dict_key(xyz, center_xyz, "dtype")
    check_list = ("float16", "float32")
    para_check.check_dtype(xyz_dtype, check_list, param_name="xyz")

    res = BallQuery(xyz, center_xyz, min_radius, max_radius, sample_num).run_tik(kernel_name)

    return res
