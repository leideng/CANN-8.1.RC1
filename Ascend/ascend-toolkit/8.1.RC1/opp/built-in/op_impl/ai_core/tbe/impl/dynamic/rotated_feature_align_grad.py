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
rotated_feature_align_grad
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
    PARAMS_SIZE = 2 ** 31 - 1
    TILING_ARG_NUM = 64
    # data type of int32
    INT32 = "int32"
    # data type of float32
    FLOAT32 = "float32"
    # one block size takes up 32b
    BLOCK_BYTE_SIZE = 32
    TYPE_LEN_DICT = {"float32": 4, "int32": 4}
    # mask of float32
    MASK = 64
    # param number of each box
    PARAM_NUM_EACH_BOX = 5
    # point number of each box
    POINT_NUM_EACH_BOX = 5
    # max burst and nburst
    BURST_MAX = 65535
    NBURST_MAX = 4095
    # max repeat_times
    REPEAT_TIMES_MAX = 255
    # 30K size
    UB_30K_SIZE = 30 * 1024
    HALF = 0.5
    # number of each number
    DY_SHAPE_NUM = 1
    BBOXES_CORE_SHAPE_NUM = 1
    NHW_CORE_SHAPE_NUM = 10
    PY_CORE_SHAPE_NUM = 24
    CHANNEL_SHAPE_NUM = 5


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


def data_move_dynamic(self, data_len, dst, src):
    """
    data move from src to dst based on data_len
    """
    nburst = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="nburst", init_value=0)
    burst = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="burst", init_value=0)
    offset = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="offset", init_value=0)
    loop_nburst = data_len // self.data_each_block // Constant.BURST_MAX // Constant.NBURST_MAX
    with self.tik_instance.for_range(0, loop_nburst) as loop_id:
        nburst.set_as(Constant.NBURST_MAX)
        burst.set_as(Constant.BURST_MAX)
        offset.set_as(loop_id * Constant.NBURST_MAX * Constant.BURST_MAX * self.data_each_block)
        self.tik_instance.data_move(dst[offset], src[offset], 0, nburst, burst, 0, 0)
    loop_burst = data_len // self.data_each_block // Constant.BURST_MAX % Constant.NBURST_MAX
    with self.tik_instance.if_scope(loop_burst > 0):
        nburst.set_as(loop_burst)
        burst.set_as(Constant.BURST_MAX)
        offset.set_as(loop_nburst * Constant.NBURST_MAX * Constant.BURST_MAX * self.data_each_block)
        self.tik_instance.data_move(dst[offset], src[offset], 0, nburst, burst, 0, 0)
    loop_block = data_len // self.data_each_block % Constant.BURST_MAX
    with self.tik_instance.if_scope(loop_block > 0):
        nburst.set_as(1)
        burst.set_as(loop_block)
        offset.set_as((loop_nburst * Constant.NBURST_MAX + loop_burst) * Constant.BURST_MAX * self.data_each_block)
        self.tik_instance.data_move(dst[offset], src[offset], 0, nburst, burst, 0, 0)
    loop_tail = data_len % self.data_each_block
    with self.tik_instance.if_scope(loop_tail > 0):
        nburst.set_as(1)
        burst.set_as(1)
        offset.set_as(
            ((loop_nburst * Constant.NBURST_MAX + loop_burst) * Constant.BURST_MAX + loop_block) * self.data_each_block)
        self.tik_instance.data_move(dst[offset], src[offset], 0, nburst, burst, 0, 0)


def vec_dup_dynamic(self, data_len, dst, dup_num):
    """
    duplicate dup_num to tensor dst based on data_len
    """
    repeat_times = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="repeat_times", init_value=0)
    offset = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="offset", init_value=0)
    mask_last = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="mask_last", init_value=0)
    loop_repeats = data_len // self.mask // Constant.REPEAT_TIMES_MAX
    with self.tik_instance.for_range(0, loop_repeats) as loop_id:
        repeat_times.set_as(Constant.REPEAT_TIMES_MAX)
        offset.set_as(loop_id * Constant.REPEAT_TIMES_MAX * self.mask)
        self.tik_instance.vec_dup(self.mask, dst[offset], dup_num, repeat_times, 8)
    loop_mask = data_len // self.mask % Constant.REPEAT_TIMES_MAX
    with self.tik_instance.if_scope(loop_mask > 0):
        repeat_times.set_as(loop_mask)
        offset.set_as(loop_repeats * Constant.REPEAT_TIMES_MAX * self.mask)
        self.tik_instance.vec_dup(self.mask, dst[offset], dup_num, repeat_times, 8)
    loop_tail = data_len % self.mask
    with self.tik_instance.if_scope(loop_tail > 0):
        offset.set_as(self.mask * (loop_repeats * Constant.REPEAT_TIMES_MAX + loop_mask))
        mask_last.set_as(data_len % self.mask)
        repeat_times.set_as(1)
        self.tik_instance.vec_dup(mask_last, dst[offset], dup_num, repeat_times, 8)


def vmuls_dynamic(self, data_len, dst, src, mul_num):
    """
    multiply tensor src and mul_num to tensor dst based on data_len
    """
    repeat_times = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="repeat_times", init_value=0)
    offset = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="offset", init_value=0)
    mask_last = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="mask_last", init_value=0)
    loop_repeats = data_len // self.mask // Constant.REPEAT_TIMES_MAX
    with self.tik_instance.for_range(0, loop_repeats) as loop_id:
        repeat_times.set_as(Constant.REPEAT_TIMES_MAX)
        offset.set_as(loop_id * Constant.REPEAT_TIMES_MAX * self.mask)
        self.tik_instance.vmuls(self.mask, dst[offset], src[offset], mul_num, repeat_times, 1, 1, 8, 8)
    loop_mask = data_len // self.mask % Constant.REPEAT_TIMES_MAX
    with self.tik_instance.if_scope(loop_mask > 0):
        repeat_times.set_as(loop_mask)
        offset.set_as(loop_repeats * Constant.REPEAT_TIMES_MAX * self.mask)
        self.tik_instance.vmuls(self.mask, dst[offset], src[offset], mul_num, repeat_times, 1, 1, 8, 8)
    loop_tail = data_len % self.mask
    with self.tik_instance.if_scope(loop_tail > 0):
        offset.set_as(self.mask * (loop_repeats * Constant.REPEAT_TIMES_MAX + loop_mask))
        mask_last.set_as(data_len % self.mask)
        repeat_times.set_as(1)
        self.tik_instance.vmuls(mask_last, dst[offset], src[offset], mul_num, repeat_times, 1, 1, 8, 8)


# 'pylint: disable=too-many-arguments,too-many-statements,too-many-locals,too-many-return-values
def vmul_vadd_vsub_dynamic(self, mode, data_len, dst, src0, src1):
    """
    mode 1 corresponds to vmul, mode 2 corresponds to vadd, mode 3 corresponds to vsub.
    """
    repeat_times = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="repeat_times", init_value=0)
    offset = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="offset", init_value=0)
    mask_last = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="mask_last", init_value=0)
    loop_repeats = data_len // self.mask // Constant.REPEAT_TIMES_MAX
    with self.tik_instance.for_range(0, loop_repeats) as loop_id:
        repeat_times.set_as(Constant.REPEAT_TIMES_MAX)
        offset.set_as(loop_id * Constant.REPEAT_TIMES_MAX * self.mask)
        with self.tik_instance.if_scope(mode == 1):
            self.tik_instance.vmul(self.mask, dst[offset], src0[offset], src1[offset], repeat_times, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.elif_scope(mode == 2):
            self.tik_instance.vadd(self.mask, dst[offset], src0[offset], src1[offset], repeat_times, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vsub(self.mask, dst[offset], src0[offset], src1[offset], repeat_times, 1, 1, 1, 8, 8, 8)
    loop_mask = data_len // self.mask % Constant.REPEAT_TIMES_MAX
    with self.tik_instance.if_scope(loop_mask > 0):
        repeat_times.set_as(loop_mask)
        offset.set_as(loop_repeats * Constant.REPEAT_TIMES_MAX * self.mask)
        with self.tik_instance.if_scope(mode == 1):
            self.tik_instance.vmul(self.mask, dst[offset], src0[offset], src1[offset], repeat_times, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.elif_scope(mode == 2):
            self.tik_instance.vadd(self.mask, dst[offset], src0[offset], src1[offset], repeat_times, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vsub(self.mask, dst[offset], src0[offset], src1[offset], repeat_times, 1, 1, 1, 8, 8, 8)
    loop_tail = data_len % self.mask
    with self.tik_instance.if_scope(loop_tail > 0):
        offset.set_as(self.mask * (loop_repeats * Constant.REPEAT_TIMES_MAX + loop_mask))
        mask_last.set_as(data_len % self.mask)
        repeat_times.set_as(1)
        with self.tik_instance.if_scope(mode == 1):
            self.tik_instance.vmul(mask_last, dst[offset], src0[offset], src1[offset], repeat_times, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.elif_scope(mode == 2):
            self.tik_instance.vadd(mask_last, dst[offset], src0[offset], src1[offset], repeat_times, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vsub(mask_last, dst[offset], src0[offset], src1[offset], repeat_times, 1, 1, 1, 8, 8, 8)


def vec_conv_dynamic(self, round_mode, data_len, dst, src):
    """
    convert tensor src to tensor dst according to round_mode based on data_len
    """
    repeat_times = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="repeat_times", init_value=0)
    offset = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="offset", init_value=0)
    mask_last = self.tik_instance.Scalar(dtype=self.scalar_dtype, name="mask_last", init_value=0)
    loop_repeats = data_len // self.mask // Constant.REPEAT_TIMES_MAX
    with self.tik_instance.for_range(0, loop_repeats) as loop_id:
        repeat_times.set_as(Constant.REPEAT_TIMES_MAX)
        offset.set_as(loop_id * Constant.REPEAT_TIMES_MAX * self.mask)
        self.tik_instance.vec_conv(self.mask, round_mode, dst[offset], src[offset], repeat_times, 8, 8)
    loop_mask = data_len // self.mask % Constant.REPEAT_TIMES_MAX
    with self.tik_instance.if_scope(loop_mask > 0):
        repeat_times.set_as(loop_mask)
        offset.set_as(loop_repeats * Constant.REPEAT_TIMES_MAX * self.mask)
        self.tik_instance.vec_conv(self.mask, round_mode, dst[offset], src[offset], repeat_times, 8, 8)
    loop_tail = data_len % self.mask
    with self.tik_instance.if_scope(loop_tail > 0):
        offset.set_as(self.mask * (loop_repeats * Constant.REPEAT_TIMES_MAX + loop_mask))
        mask_last.set_as(data_len % self.mask)
        repeat_times.set_as(1)
        self.tik_instance.vec_conv(mask_last, round_mode, dst[offset], src[offset], repeat_times, 8, 8)


class RotatedFeatureAlignGrad:
    """
    define rotated_feature_align_grad object
    """

    def __init__(self, spatial_scale, points, kernel_name):
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile)
        self.core_num = profile.get_aicore_num()
        self.ub_size = profile.get_unified_buffer_size()
        self.kernel_name = kernel_name
        self.spatial_scale = None
        self.points = None
        self.dtype = Constant.FLOAT32
        self.scalar_dtype = Constant.INT32
        self.mask = Constant.MASK
        self.tiling_dtype = Constant.INT32
        self.dtype_bytes_size = Constant.TYPE_LEN_DICT.get(self.dtype)
        self.data_each_block = Constant.BLOCK_BYTE_SIZE // self.dtype_bytes_size
        self.dy_shape = None
        self.nhw_thread = None
        self.init_scalar()
        self.init_inputs_and_output_gm()
        self.get_tiling_args()

    def init_scalar(self):
        self.core_bbox_n = self.tik_instance.Scalar(self.scalar_dtype, name="core_bbox_n")
        self.core_tail = self.tik_instance.Scalar(self.scalar_dtype, name="core_tail")
        self.core_bias = self.tik_instance.Scalar(self.scalar_dtype, name="core_bias")
        self.number = self.tik_instance.Scalar(self.tiling_dtype, name="number")
        self.height = self.tik_instance.Scalar(self.tiling_dtype, name="height")
        self.width = self.tik_instance.Scalar(self.tiling_dtype, name="width")
        self.channel = self.tik_instance.Scalar(self.tiling_dtype, name="channel")
        self.real_core_num = self.tik_instance.Scalar(self.tiling_dtype, name="real_core_num")
        self.spatial_scale = self.tik_instance.Scalar(self.dtype, name='spatial_scale')
        self.points = self.tik_instance.Scalar(self.tiling_dtype, name='points')
        self.dy_core_bias = self.tik_instance.Scalar(self.scalar_dtype, name="dy_core_bias")
        self.bboxes_core_thread = self.tik_instance.Scalar(self.scalar_dtype, name="bboxes_core_thread")
        self.bboxes_core_bias = self.tik_instance.Scalar(self.scalar_dtype, name="bboxes_core_bias")
        self.py_core_thread = self.tik_instance.Scalar(self.scalar_dtype, name="py_core_thread")
        self.channel_thread_task = self.tik_instance.Scalar(self.scalar_dtype, name="channel_thread_task")
        self.core_bbox_n_task = self.tik_instance.Scalar(self.scalar_dtype, name="core_bbox_n_task")
        self.core_bbox_n_task_real = self.tik_instance.Scalar(self.scalar_dtype, name="core_bbox_n_task_real")
        self.task_bias = self.tik_instance.Scalar(self.scalar_dtype, name="task_bias")
        self.nhw_core_shape_task = self.tik_instance.Scalar(self.scalar_dtype, name="nhw_core_shape_task")
        self.py_core_shape_task = self.tik_instance.Scalar(self.scalar_dtype, name="py_core_shape_task")
        self.py_core_thread_task = self.tik_instance.Scalar(self.scalar_dtype, name="py_core_thread_task")
        self.height_float = self.tik_instance.Scalar(self.dtype, name="height_float")
        self.width_float = self.tik_instance.Scalar(self.dtype, name="width_float")
        self.minus_one_float = self.tik_instance.Scalar(self.dtype, name="minus_one_float", init_value=-1)
        self.zero_float = self.tik_instance.Scalar(self.dtype, name="zero_float", init_value=0)
        self.w1_scalar = self.tik_instance.Scalar(self.dtype, name="w1_scalar")
        self.w2_scalar = self.tik_instance.Scalar(self.dtype, name="w2_scalar")
        self.w3_scalar = self.tik_instance.Scalar(self.dtype, name="w3_scalar")
        self.w4_scalar = self.tik_instance.Scalar(self.dtype, name="w4_scalar")
        self.y_low_scalar = self.tik_instance.Scalar(self.scalar_dtype, name="y_low_scalar")
        self.y_hight_scalar = self.tik_instance.Scalar(self.scalar_dtype, name="y_hight_scalar")
        self.x_low_scalar = self.tik_instance.Scalar(self.scalar_dtype, name="x_low_scalar")
        self.x_hight_scalar = self.tik_instance.Scalar(self.scalar_dtype, name="x_hight_scalar")
        self.number_index = self.tik_instance.Scalar(self.scalar_dtype, name="number_index")
        self.height_index = self.tik_instance.Scalar(self.scalar_dtype, name="height_index")
        self.width_index = self.tik_instance.Scalar(self.scalar_dtype, name="width_index")

    def init_inputs_and_output_gm(self):
        self.input_dy_gm = self.tik_instance.Tensor(self.dtype, (Constant.PARAMS_SIZE,), name="input_dy_gm",
                                                    scope=tik.scope_gm)
        self.input_bboxes_gm = self.tik_instance.Tensor(self.dtype, (Constant.PARAMS_SIZE,), name="input_bboxes_gm",
                                                        scope=tik.scope_gm)
        self.output_dx_gm = self.tik_instance.Tensor(self.dtype, (Constant.PARAMS_SIZE,), name="output_dx_gm",
                                                     is_atomic_add=True, scope=tik.scope_gm)

    def get_tiling_args(self):
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1,
                                    ceil_block(Constant.TILING_ARG_NUM, self.tiling_dtype), 0, 0)
        self.number.set_as(self.tiling_ub[0])
        self.height.set_as(self.tiling_ub[1])
        self.width.set_as(self.tiling_ub[2])
        self.channel.set_as(self.tiling_ub[3])
        self.real_core_num.set_as(self.tiling_ub[4])
        self.spatial_scale.set_as(self.tiling_ub[5])
        self.points.set_as(self.tiling_ub[6])

    def rotated_feature_align_grad_compute(self):
        inputs = [self.input_dy_gm, self.input_bboxes_gm]
        outputs = [self.output_dx_gm]

        self.rotated_feature_align_compute_tiling()

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }
        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num, "ub_size": self.ub_size})

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=inputs,
                                   outputs=outputs,
                                   flowtable=(self.tiling_gm,),
                                   config=opt_config)
        return self.tik_instance

    def rotated_feature_align_compute_tiling(self):
        """
        define rotated_feature_align tiling method, divide the data volume into multiple cores
        """
        self.dy_shape = (self.number, self.height, self.width, self.channel)
        self.nhw_thread = self.number * self.height * self.width

        self.tik_instance.set_atomic_add(1)
        with self.tik_instance.for_range(0, self.real_core_num, block_num=self.real_core_num) as block_id:
            self.core_bbox_n.set_as(self.nhw_thread // self.real_core_num)
            self.core_tail.set_as(self.nhw_thread % self.real_core_num)
            self.core_bias.set_as(self.core_bbox_n * block_id)

            with self.tik_instance.if_scope(self.core_tail != 0):
                with self.tik_instance.if_scope(block_id < self.core_tail):
                    self.core_bbox_n.set_as(self.core_bbox_n + 1)
                    self.core_bias.set_as(self.core_bbox_n * block_id)
                with self.tik_instance.else_scope():
                    self.core_bias.set_as(self.core_bbox_n * block_id + self.core_tail)

            self.rotated_feature_align_grad_compute_kernel()
        self.tik_instance.set_atomic_add(0)

    def rotated_feature_align_grad_compute_kernel(self):
        """
        divide the computational workload of a single core into multiple tasks based on the size of the ub space
        """
        self.dy_core_bias.set_as(self.core_bias * self.channel)
        self.bboxes_core_thread.set_as(self.core_bbox_n * Constant.PARAM_NUM_EACH_BOX)
        self.bboxes_core_bias.set_as(self.core_bias * Constant.PARAM_NUM_EACH_BOX)
        self.py_core_thread.set_as(Constant.POINT_NUM_EACH_BOX * self.core_bbox_n)
        self.channel_thread_task = ceil_block(self.channel, self.dtype) * self.data_each_block

        # compute task number of each core
        ub_size_available = self.ub_size - Constant.UB_30K_SIZE
        ub_ele_size_available = ub_size_available // self.dtype_bytes_size
        total_ele_var_core = self.core_bbox_n * self.channel * Constant.DY_SHAPE_NUM \
                             + self.bboxes_core_thread * Constant.BBOXES_CORE_SHAPE_NUM \
                             + self.core_bbox_n * Constant.NHW_CORE_SHAPE_NUM \
                             + self.py_core_thread * Constant.PY_CORE_SHAPE_NUM
        total_ele_fix_core = self.channel_thread_task * Constant.CHANNEL_SHAPE_NUM

        task_core = self.tik_instance.Scalar(self.scalar_dtype, name="task_core")
        task_core.set_as(self.core_bbox_n)

        with self.tik_instance.for_range(0, task_core) as task_core_id:
            self.core_bbox_n_task_real.set_as(self.core_bbox_n // task_core)

            with self.tik_instance.if_scope(
                    tik.all((task_core_id == (task_core - 1)), ((self.core_bbox_n % task_core) > 0))):
                self.core_bbox_n_task_real.set_as(self.core_bbox_n_task_real + (self.core_bbox_n % task_core))

            self.core_bbox_n_task.set_as(ceil_block(self.core_bbox_n_task_real, self.dtype) * self.data_each_block)
            bboxes_core_shape_task = (self.core_bbox_n_task_real, Constant.PARAM_NUM_EACH_BOX)
            bboxes_core_thread_task = self.core_bbox_n_task_real * Constant.PARAM_NUM_EACH_BOX
            self.task_bias = (self.core_bbox_n // task_core) * task_core_id
            bboxes_task_bias = self.task_bias * Constant.PARAM_NUM_EACH_BOX
            bboxes_core_task_bias = self.bboxes_core_bias + bboxes_task_bias
            self.nhw_core_shape_task = [self.core_bbox_n_task_real, ]
            self.py_core_shape_task = [Constant.POINT_NUM_EACH_BOX, self.core_bbox_n_task]
            self.py_core_thread_task = Constant.POINT_NUM_EACH_BOX * self.core_bbox_n_task

            dy_core_thread_task = ceil_block(self.core_bbox_n_task_real * self.channel,
                                             self.dtype) * self.data_each_block
            dy_task_bias = self.task_bias * self.channel
            dy_core_task_bias = self.dy_core_bias + dy_task_bias

            bboxes_ub = self.tik_instance.Tensor(self.dtype, bboxes_core_shape_task, name="bboxes_ub",
                                                 scope=tik.scope_ubuf)
            with self.tik_instance.new_stmt_scope():
                dy_data_move_task_ub = self.tik_instance.Tensor(self.dtype, [dy_core_thread_task, ],
                                                            name="dy_data_move_task_ub", scope=tik.scope_ubuf)

                self.dy_data_move_to_dx(dy_core_thread_task, dy_core_task_bias, dy_data_move_task_ub)
            self.mul_spatial_scale(bboxes_core_thread_task, bboxes_core_task_bias, bboxes_ub)
            px_ub, py_ub = self.sample_points_coor_compute(bboxes_ub)
            self.input_feature_map_grad_compute(py_ub, px_ub)

    def dy_data_move_to_dx(self, dy_core_thread_task, dy_core_task_bias, dy_data_move_task_ub):
        data_move_dynamic(self, dy_core_thread_task, dy_data_move_task_ub, self.input_dy_gm[dy_core_task_bias])
        with self.tik_instance.for_range(self.core_bbox_n_task_real * self.channel, dy_core_thread_task) as dy_index:
            dy_data_move_task_ub[dy_index].set_as(0)
        data_move_dynamic(self, dy_core_thread_task, self.output_dx_gm[dy_core_task_bias], dy_data_move_task_ub)

    def mul_spatial_scale(self, bboxes_core_thread_task, bboxes_core_task_bias, bboxes_ub):
        """
        spatial_scale map coordinates in bbox to feature maps
        """
        data_move_dynamic(self, bboxes_core_thread_task, bboxes_ub, self.input_bboxes_gm[bboxes_core_task_bias])
        vmuls_dynamic(self, bboxes_core_thread_task, bboxes_ub, bboxes_ub, self.spatial_scale)

    def sample_points_coor_compute(self, bboxes_ub):
        """
        Calculate the specific coordinates of the sampling point
        """
        py_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="py_ub", scope=tik.scope_ubuf)
        px_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="px_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.core_bbox_n_task_real) as box_index:
            points_offset = box_index * Constant.PARAM_NUM_EACH_BOX
            py_ub[box_index].set_as(bboxes_ub[points_offset + 0])
            px_ub[box_index].set_as(bboxes_ub[points_offset + 1])

        with self.tik_instance.if_scope(self.points > 1):
            with self.tik_instance.new_stmt_scope():
                roi_w_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="roi_w_ub",
                                                    scope=tik.scope_ubuf)
                roi_h_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="roi_h_ub",
                                                    scope=tik.scope_ubuf)
                roi_a_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="roi_a_ub",
                                                    scope=tik.scope_ubuf)
                cosa_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="cosa_ub",
                                                   scope=tik.scope_ubuf)
                sina_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="sina_ub",
                                                   scope=tik.scope_ubuf)
                temp1 = self.tik_instance.Scalar(self.dtype)
                with self.tik_instance.for_range(0, self.core_bbox_n_task_real) as box_index:
                    points_offset = box_index * Constant.PARAM_NUM_EACH_BOX
                    roi_w_ub[box_index].set_as(bboxes_ub[points_offset + 2])
                    roi_h_ub[box_index].set_as(bboxes_ub[points_offset + 3])
                    temp1.set_as(bboxes_ub[points_offset + 4])
                    roi_a_ub[box_index].set_as(temp1 / self.spatial_scale)

                self.tik_instance.h_cos(cosa_ub, roi_a_ub)
                self.tik_instance.h_sin(sina_ub, roi_a_ub)

                w_2_ub = roi_w_ub
                h_2_ub = roi_h_ub
                vmuls_dynamic(self, self.core_bbox_n_task_real, w_2_ub, roi_w_ub, Constant.HALF)
                vmuls_dynamic(self, self.core_bbox_n_task_real, h_2_ub, roi_h_ub, Constant.HALF)

                wx_ub, wy_ub, hx_ub, hy_ub = self.sample_points_coor_compute_matmul(cosa_ub, sina_ub, w_2_ub, h_2_ub)
                self.sample_points_coor_compute_vadd(wx_ub, wy_ub, hx_ub, hy_ub, px_ub, py_ub)
        return px_ub, py_ub

    # 'pylint: disable=too-many-return-values
    def sample_points_coor_compute_matmul(self, cosa_ub, sina_ub, w_2_ub, h_2_ub):
        """
        Calculation of Matrix multiplication in Coordinate Calculation of Sampling Points
        """
        wx_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="wx_ub", scope=tik.scope_ubuf)
        wy_ub = w_2_ub
        hx_ub = sina_ub
        hy_ub = h_2_ub

        vmul_vadd_vsub_dynamic(self, 1, self.core_bbox_n_task_real, wx_ub, cosa_ub, w_2_ub)
        vmul_vadd_vsub_dynamic(self, 1, self.core_bbox_n_task_real, wy_ub, sina_ub, w_2_ub)
        vmul_vadd_vsub_dynamic(self, 1, self.core_bbox_n_task_real, hx_ub, sina_ub, h_2_ub)
        vmuls_dynamic(self, self.core_bbox_n_task_real, hx_ub, hx_ub, -1)
        vmul_vadd_vsub_dynamic(self, 1, self.core_bbox_n_task_real, hy_ub, cosa_ub, h_2_ub)

        return wx_ub, wy_ub, hx_ub, hy_ub

    # 'pylint: disable=too-many-arguments,too-many-statements,too-many-locals,too-many-return-values
    def sample_points_coor_compute_vadd(self, wx_ub, wy_ub, hx_ub, hy_ub, px_ub, py_ub):
        """
        Calculation of Matrix addition in Coordinate Calculation of Sampling Points
        """
        roi_x_add_wx_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="roi_x_add_wx_ub",
                                                   scope=tik.scope_ubuf)
        roi_y_add_wy_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="roi_y_add_wy_ub",
                                                   scope=tik.scope_ubuf)
        roi_x_red_wx_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="roi_x_red_wx_ub",
                                                   scope=tik.scope_ubuf)
        roi_y_red_wy_ub = self.tik_instance.Tensor(self.dtype, self.nhw_core_shape_task, name="roi_y_red_wy_ub",
                                                   scope=tik.scope_ubuf)

        vmul_vadd_vsub_dynamic(self, 2, self.core_bbox_n_task_real, roi_x_add_wx_ub, px_ub, wx_ub)
        vmul_vadd_vsub_dynamic(self, 2, self.core_bbox_n_task_real, roi_y_add_wy_ub, py_ub, wy_ub)
        vmul_vadd_vsub_dynamic(self, 3, self.core_bbox_n_task_real, roi_x_red_wx_ub, px_ub, wx_ub)
        vmul_vadd_vsub_dynamic(self, 3, self.core_bbox_n_task_real, roi_y_red_wy_ub, py_ub, wy_ub)
        vmul_vadd_vsub_dynamic(self, 2, self.core_bbox_n_task_real, px_ub[self.core_bbox_n_task], roi_x_add_wx_ub,
                               hx_ub)
        vmul_vadd_vsub_dynamic(self, 2, self.core_bbox_n_task_real, py_ub[self.core_bbox_n_task], roi_y_add_wy_ub,
                               hy_ub)
        vmul_vadd_vsub_dynamic(self, 2, self.core_bbox_n_task_real, px_ub[2 * self.core_bbox_n_task], roi_x_red_wx_ub,
                               hx_ub)
        vmul_vadd_vsub_dynamic(self, 2, self.core_bbox_n_task_real, py_ub[2 * self.core_bbox_n_task], roi_y_red_wy_ub,
                               hy_ub)
        vmul_vadd_vsub_dynamic(self, 3, self.core_bbox_n_task_real, px_ub[3 * self.core_bbox_n_task], roi_x_red_wx_ub,
                               hx_ub)
        vmul_vadd_vsub_dynamic(self, 3, self.core_bbox_n_task_real, py_ub[3 * self.core_bbox_n_task], roi_y_red_wy_ub,
                               hy_ub)
        vmul_vadd_vsub_dynamic(self, 3, self.core_bbox_n_task_real, px_ub[4 * self.core_bbox_n_task], roi_x_add_wx_ub,
                               hx_ub)
        vmul_vadd_vsub_dynamic(self, 3, self.core_bbox_n_task_real, py_ub[4 * self.core_bbox_n_task], roi_y_add_wy_ub,
                               hy_ub)

    # 'pylint: disable=too-many-arguments,too-many-statements,too-many-locals,too-many-return-values,duplicate-string
    def input_feature_map_grad_compute(self, py_ub, px_ub):
        """
        calculate the gradient of input feature maps through bilinear interpolation
        """
        y_low_int_ub, x_low_int_ub, y_low_ub, x_low_ub, y_hight_ub, x_hight_ub, w1_ub, \
            w2_ub, w3_ub, w4_ub = self.bilinear_interpolate_gradient(py_ub, px_ub)

        # Accumulate gradient values onto the output feature map gradient
        value_dy_diff_ub = self.tik_instance.Tensor(self.dtype, (self.channel_thread_task,), name="value_dx_diff_ub",
                                                    scope=tik.scope_ubuf)
        g1_ub = self.tik_instance.Tensor(self.dtype, (self.channel_thread_task,), name="g1_ub", scope=tik.scope_ubuf)
        g2_ub = self.tik_instance.Tensor(self.dtype, (self.channel_thread_task,), name="g2_ub", scope=tik.scope_ubuf)
        g3_ub = self.tik_instance.Tensor(self.dtype, (self.channel_thread_task,), name="g3_ub", scope=tik.scope_ubuf)
        g4_ub = self.tik_instance.Tensor(self.dtype, (self.channel_thread_task,), name="g4_ub", scope=tik.scope_ubuf)
        y_hight_int_ub = self.tik_instance.Tensor(self.scalar_dtype, self.py_core_shape_task, name="y_hight_int_ub",
                                                  scope=tik.scope_ubuf)
        x_hight_int_ub = self.tik_instance.Tensor(self.scalar_dtype, self.py_core_shape_task, name="x_hight_int_ub",
                                                  scope=tik.scope_ubuf)

        vec_conv_dynamic(self, 'floor', self.py_core_thread_task, y_low_int_ub, y_low_ub)
        vec_conv_dynamic(self, 'floor', self.py_core_thread_task, y_hight_int_ub, y_hight_ub)
        vec_conv_dynamic(self, 'floor', self.py_core_thread_task, x_low_int_ub, x_low_ub)
        vec_conv_dynamic(self, 'floor', self.py_core_thread_task, x_hight_int_ub, x_hight_ub)

        with self.tik_instance.for_range(0, self.points) as points_index:
            with self.tik_instance.for_range(0, self.core_bbox_n_task_real) as box_index:
                self.w1_scalar.set_as(w1_ub[points_index, box_index])
                self.w2_scalar.set_as(w2_ub[points_index, box_index])
                self.w3_scalar.set_as(w3_ub[points_index, box_index])
                self.w4_scalar.set_as(w4_ub[points_index, box_index])

                box_index_all = self.core_bias + self.task_bias + box_index

                self.width_index.set_as(box_index_all % self.width)
                self.height_index.set_as((box_index_all // self.width) % self.height)
                self.number_index.set_as(box_index_all // self.width // self.height)

                data_move_dynamic(self, self.channel, value_dy_diff_ub, self.input_dy_gm[
                    ((self.number_index * self.height + self.height_index) * self.width + self.width_index) \
                    * self.channel])

                vmuls_dynamic(self, self.channel, g1_ub, value_dy_diff_ub, self.w1_scalar)
                vmuls_dynamic(self, self.channel, g2_ub, value_dy_diff_ub, self.w2_scalar)
                vmuls_dynamic(self, self.channel, g3_ub, value_dy_diff_ub, self.w3_scalar)
                vmuls_dynamic(self, self.channel, g4_ub, value_dy_diff_ub, self.w4_scalar)

                self.y_low_scalar.set_as(y_low_int_ub[points_index, box_index])
                self.y_hight_scalar.set_as(y_hight_int_ub[points_index, box_index])
                self.x_low_scalar.set_as(x_low_int_ub[points_index, box_index])
                self.x_hight_scalar.set_as(x_hight_int_ub[points_index, box_index])

                self.move_out_to_dx(g1_ub, g2_ub, g3_ub, g4_ub)

    def bilinear_interpolate_gradient(self, py_ub, px_ub):
        """
        calculate bilinear interpolate gradient, verify boundary conditions
        """
        compare_temp_zero_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task,
                                                        name="compare_temp_zero_ub", scope=tik.scope_ubuf)
        compare_temp_ones_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task,
                                                        name="compare_temp_ones_ub", scope=tik.scope_ubuf)
        vec_dup_dynamic(self, self.py_core_thread_task, compare_temp_zero_ub, 0)
        vec_dup_dynamic(self, self.py_core_thread_task, compare_temp_ones_ub, 1)

        mask_tensor_bool_ub = self.tik_instance.Tensor('bool', self.py_core_shape_task, name="mask_tensor1_bool_ub",
                                                       scope=tik.scope_ubuf)
        mask_tensor1_float_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task,
                                                         name="mask_tensor1_float_ub", scope=tik.scope_ubuf)
        mask_tensor2_float_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task,
                                                         name="mask_tensor2_float_ub", scope=tik.scope_ubuf)

        self.tik_instance.scalar_conv("none", self.height_float, self.height)
        self.tik_instance.scalar_conv("none", self.width_float, self.width)
        self.tik_instance.h_cmpv(mask_tensor_bool_ub, py_ub, self.minus_one_float, 'LT')
        self.tik_instance.h_sel(mask_tensor1_float_ub, compare_temp_ones_ub, compare_temp_zero_ub, mask_tensor_bool_ub)
        self.tik_instance.h_cmpv(mask_tensor_bool_ub, py_ub, self.height_float, 'GT')
        self.tik_instance.h_sel(mask_tensor2_float_ub, compare_temp_ones_ub, compare_temp_zero_ub, mask_tensor_bool_ub)
        vmul_vadd_vsub_dynamic(self, 2, self.py_core_thread_task, mask_tensor1_float_ub, mask_tensor1_float_ub,
                               mask_tensor2_float_ub)
        self.tik_instance.h_cmpv(mask_tensor_bool_ub, px_ub, self.minus_one_float, 'LT')
        self.tik_instance.h_sel(mask_tensor2_float_ub, compare_temp_ones_ub, compare_temp_zero_ub, mask_tensor_bool_ub)
        vmul_vadd_vsub_dynamic(self, 2, self.py_core_thread_task, mask_tensor1_float_ub, mask_tensor1_float_ub,
                               mask_tensor2_float_ub)
        self.tik_instance.h_cmpv(mask_tensor_bool_ub, px_ub, self.width_float, 'GT')
        self.tik_instance.h_sel(mask_tensor2_float_ub, compare_temp_ones_ub, compare_temp_zero_ub, mask_tensor_bool_ub)
        vmul_vadd_vsub_dynamic(self, 2, self.py_core_thread_task, mask_tensor1_float_ub, mask_tensor1_float_ub,
                               mask_tensor2_float_ub)

        self.tik_instance.h_cmpv(mask_tensor_bool_ub, mask_tensor1_float_ub, compare_temp_zero_ub, 'GT')

        y_low_int_ub, x_low_int_ub, w1_ub, w2_ub, w3_ub, w4_ub, x_low_ub, y_low_ub, x_hight_ub, y_hight_ub \
            = self.bilinear_interpolate_gradient_compute(py_ub, px_ub, compare_temp_zero_ub, compare_temp_ones_ub,
                                                         mask_tensor_bool_ub)

        return y_low_int_ub, x_low_int_ub, y_low_ub, x_low_ub, y_hight_ub, x_hight_ub, w1_ub, w2_ub, w3_ub, w4_ub

    # 'pylint: disable=too-many-arguments,too-many-statements,too-many-locals,too-many-return-values
    def bilinear_interpolate_gradient_compute(self, py_ub, px_ub, compare_temp_zero_ub, compare_temp_ones_ub,
                                              mask_tensor_bool_ub):
        """
        calculate gradient values for sampling points that meet the conditions
        """

        x_low_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="x_low_ub", scope=tik.scope_ubuf)
        x_hight_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="x_hight_ub",
                                              scope=tik.scope_ubuf)
        y_low_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="y_low_ub", scope=tik.scope_ubuf)
        y_hight_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="y_hight_ub",
                                              scope=tik.scope_ubuf)
        mask_tensor1_bool_ub = self.tik_instance.Tensor('bool', self.py_core_shape_task, name="mask_tensor1_bool_ub",
                                                        scope=tik.scope_ubuf)
        y_low_int_ub = self.tik_instance.Tensor(self.scalar_dtype, self.py_core_shape_task, name="y_low_int_ub",
                                                scope=tik.scope_ubuf)
        x_low_int_ub = self.tik_instance.Tensor(self.scalar_dtype, self.py_core_shape_task, name="x_low_int_ub",
                                                scope=tik.scope_ubuf)

        self.tik_instance.h_cmpv(mask_tensor1_bool_ub, py_ub, self.zero_float, 'LE')
        self.tik_instance.h_sel(py_ub, compare_temp_zero_ub, py_ub, mask_tensor1_bool_ub)
        self.tik_instance.h_cmpv(mask_tensor1_bool_ub, px_ub, self.zero_float, 'LE')
        self.tik_instance.h_sel(px_ub, compare_temp_zero_ub, px_ub, mask_tensor1_bool_ub)

        vec_conv_dynamic(self, 'floor', self.py_core_thread_task, y_low_int_ub, py_ub)
        vec_conv_dynamic(self, 'floor', self.py_core_thread_task, x_low_int_ub, px_ub)
        vec_conv_dynamic(self, 'none', self.py_core_thread_task, y_low_ub, y_low_int_ub)
        vec_conv_dynamic(self, 'none', self.py_core_thread_task, x_low_ub, x_low_int_ub)

        y_low_add_one_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="y_low_add_one_ub",
                                                    scope=tik.scope_ubuf)
        x_low_add_one_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="x_low_add_one_ub",
                                                    scope=tik.scope_ubuf)
        vmul_vadd_vsub_dynamic(self, 2, self.py_core_thread_task, y_low_add_one_ub, y_low_ub, compare_temp_ones_ub)
        vmul_vadd_vsub_dynamic(self, 2, self.py_core_thread_task, x_low_add_one_ub, x_low_ub, compare_temp_ones_ub)

        height_minus_one_float = self.tik_instance.Scalar(self.dtype, name="height_minus_one_float",
                                                          init_value=self.height - 1)
        width_minus_one_float = self.tik_instance.Scalar(self.dtype, name="width_minus_one_float",
                                                         init_value=self.width - 1)

        self.tik_instance.h_cmpv(mask_tensor1_bool_ub, y_low_ub, height_minus_one_float, 'GE')
        self.tik_instance.h_sel(y_hight_ub, height_minus_one_float, y_low_add_one_ub, mask_tensor1_bool_ub)
        self.tik_instance.h_sel(y_low_ub, height_minus_one_float, y_low_ub, mask_tensor1_bool_ub)
        self.tik_instance.h_sel(py_ub, y_low_ub, py_ub, mask_tensor1_bool_ub)

        self.tik_instance.h_cmpv(mask_tensor1_bool_ub, x_low_ub, width_minus_one_float, 'GE')
        self.tik_instance.h_sel(x_hight_ub, width_minus_one_float, x_low_add_one_ub, mask_tensor1_bool_ub)
        self.tik_instance.h_sel(x_low_ub, width_minus_one_float, x_low_ub, mask_tensor1_bool_ub)
        self.tik_instance.h_sel(px_ub, x_low_ub, px_ub, mask_tensor1_bool_ub)

        ly_ub = y_low_add_one_ub
        lx_ub = x_low_add_one_ub
        w1_ub, w2_ub, w3_ub, w4_ub = self.get_area(lx_ub, ly_ub, py_ub, px_ub, y_low_ub, x_low_ub, mask_tensor_bool_ub,
                                                   compare_temp_zero_ub, compare_temp_ones_ub)

        self.tik_instance.h_sel(x_low_ub, self.minus_one_float, x_low_ub, mask_tensor_bool_ub)
        self.tik_instance.h_sel(x_hight_ub, self.minus_one_float, x_hight_ub, mask_tensor_bool_ub)
        self.tik_instance.h_sel(y_low_ub, self.minus_one_float, y_low_ub, mask_tensor_bool_ub)
        self.tik_instance.h_sel(y_hight_ub, self.minus_one_float, y_hight_ub, mask_tensor_bool_ub)

        return y_low_int_ub, x_low_int_ub, w1_ub, w2_ub, w3_ub, w4_ub, x_low_ub, y_low_ub, x_hight_ub, y_hight_ub

    # 'pylint: disable=too-many-arguments,too-many-statements,too-many-locals,too-many-return-values
    def get_area(self, lx_ub, ly_ub, py_ub, px_ub, y_low_ub, x_low_ub, mask_tensor_bool_ub, compare_temp_zero_ub,
                 compare_temp_ones_ub):
        """
        calculate domain area
        """
        w1_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="w1_ub", scope=tik.scope_ubuf)
        w2_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="w2_ub", scope=tik.scope_ubuf)
        w3_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="w3_ub", scope=tik.scope_ubuf)
        w4_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="w4_ub", scope=tik.scope_ubuf)
        hy_bilinear_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="hy_bilinear_ub",
                                                  scope=tik.scope_ubuf)
        hx_bilinear_ub = self.tik_instance.Tensor(self.dtype, self.py_core_shape_task, name="hx_bilinear_ub",
                                                  scope=tik.scope_ubuf)

        vmul_vadd_vsub_dynamic(self, 3, self.py_core_thread_task, ly_ub, py_ub, y_low_ub)
        vmul_vadd_vsub_dynamic(self, 3, self.py_core_thread_task, lx_ub, px_ub, x_low_ub)
        vmul_vadd_vsub_dynamic(self, 3, self.py_core_thread_task, hy_bilinear_ub, compare_temp_ones_ub, ly_ub)
        vmul_vadd_vsub_dynamic(self, 3, self.py_core_thread_task, hx_bilinear_ub, compare_temp_ones_ub, lx_ub)

        vmul_vadd_vsub_dynamic(self, 1, self.py_core_thread_task, w1_ub, hy_bilinear_ub, hx_bilinear_ub)
        vmul_vadd_vsub_dynamic(self, 1, self.py_core_thread_task, w2_ub, hy_bilinear_ub, lx_ub)
        vmul_vadd_vsub_dynamic(self, 1, self.py_core_thread_task, w3_ub, ly_ub, hx_bilinear_ub)
        vmul_vadd_vsub_dynamic(self, 1, self.py_core_thread_task, w4_ub, ly_ub, lx_ub)

        self.tik_instance.h_sel(w1_ub, compare_temp_zero_ub, w1_ub, mask_tensor_bool_ub)
        self.tik_instance.h_sel(w2_ub, compare_temp_zero_ub, w2_ub, mask_tensor_bool_ub)
        self.tik_instance.h_sel(w3_ub, compare_temp_zero_ub, w3_ub, mask_tensor_bool_ub)
        self.tik_instance.h_sel(w4_ub, compare_temp_zero_ub, w4_ub, mask_tensor_bool_ub)
        return w1_ub, w2_ub, w3_ub, w4_ub

    def move_out_to_dx(self, g1_ub, g2_ub, g3_ub, g4_ub):
        """
        accumulate gradients to output
        """
        with self.tik_instance.if_scope(tik.all((self.x_low_scalar >= 0), (self.x_hight_scalar >= 0),
                                                (self.y_low_scalar >= 0), (self.y_hight_scalar >= 0))):
            with self.tik_instance.for_range(self.channel, self.channel_thread_task) as gn_index:
                g1_ub[gn_index].set_as(0)
                g2_ub[gn_index].set_as(0)
                g3_ub[gn_index].set_as(0)
                g4_ub[gn_index].set_as(0)

            data_move_dynamic(self, self.channel,
                              self.output_dx_gm[((self.number_index * self.height + self.y_low_scalar) \
                                                 * self.width + self.x_low_scalar) * self.channel], g1_ub)
            data_move_dynamic(self, self.channel,
                              self.output_dx_gm[((self.number_index * self.height + self.y_low_scalar) \
                                                 * self.width + self.x_hight_scalar) * self.channel], g2_ub)
            data_move_dynamic(self, self.channel,
                              self.output_dx_gm[((self.number_index * self.height + self.y_hight_scalar) \
                                                 * self.width + self.x_low_scalar) * self.channel], g3_ub)
            data_move_dynamic(self, self.channel,
                              self.output_dx_gm[((self.number_index * self.height + self.y_hight_scalar) \
                                                 * self.width + self.x_hight_scalar) * self.channel], g4_ub)


# 'pylint: disable=too-many-locals,too-many-arguments
@register_operator("RotatedFeatureAlignGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def rotated_feature_align_grad(dy,
                               bboxes,
                               dx,
                               spatial_scale,
                               points=1,
                               kernel_name="rotated_feature_align_grad"):
    """
    RotatedFeatureAlignGrad operator
    """
    rotated_feature_align_grad_obj = RotatedFeatureAlignGrad(spatial_scale, points, kernel_name)
    return rotated_feature_align_grad_obj.rotated_feature_align_grad_compute()

