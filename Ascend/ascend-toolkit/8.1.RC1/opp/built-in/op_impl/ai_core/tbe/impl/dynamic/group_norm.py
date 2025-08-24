# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
group_norm
"""

from impl import common_util
from impl.util.platform_adapter import tik
import tbe.common.platform as tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.platform_adapter import para_check
from impl import constant_util
from impl.util.util_tik_comm_func import tik_func_vector_support_tensor
from impl.util.util_tik_comm_func import tik_func_double_input_new


MAX_INT64 = 2 ** 63 - 1
TILING_NUM = 33
MASK = 64
MAX_REPEAT_NUM = 255

TILING_MODE0 = 0
TILING_MODE1 = 1
TILING_MODE2 = 2

NOT_SUPPORT_5HD = 0
SUPPORT_5HD_CASE_1 = 1
SUPPORT_5HD_CASE_2 = 2

BLOCK_LEN_FP32 = 8
BLOCK_LEN_FP16 = 16


def support_5hd(shape_x, num_groups, soc_version):
    """
    check if support 5HD or not
    """
    if len(shape_x) != 4:
        return NOT_SUPPORT_5HD

    c = shape_x[1]
    c0 = 16
    if c % c0 != 0:
        return NOT_SUPPORT_5HD
    else:
        c1 = c // c0
        if c1 and (c1 % num_groups == 0):
            return SUPPORT_5HD_CASE_1
        elif (c1 and num_groups / c1 == 2 and
              soc_version in ["Ascend910B", tbe_platform.ASCEND_910, "Ascend910_93"]):
            return SUPPORT_5HD_CASE_2
        else:
            return NOT_SUPPORT_5HD


# 'pylint: disable=unused-argument,too-many-locals
def op_select_format(x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-4,
                     is_training=False, kernel_name="group_norm"):
    """
    op_select format func for dynamic format
    """

    shape_x = x.get("ori_shape")
    format_x = x.get("ori_format")
    dtype_x = x.get("dtype")

    soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    dtype_list = ["float32"]
    format_list0 = ["ND"]
    format_list1 = ["ND"]

    is_support_5hd = support_5hd(shape_x, num_groups, soc_version)
    if soc_version in ["Ascend310B", "AS31XM1", tbe_platform.ASCEND_310P, "Ascend910B", "Ascend910_93"]:
        dtype_list = ["float16", "float32", "float16", "float32"]
        format_list0 = ["NC1HWC0", "NC1HWC0", "ND", "ND"]
        format_list1 = ["ND", "ND", "ND", "ND"]

        if (is_support_5hd == SUPPORT_5HD_CASE_2):
            dtype_list = ["float32", "float16", "float32"]
            format_list0 = ["NC1HWC0", "ND", "ND"]
            format_list1 = ["ND", "ND", "ND"]

        if (-2 in shape_x) or (-1 in shape_x) or (is_support_5hd == NOT_SUPPORT_5HD):
            dtype_list = ["float16", "float32"]
            format_list0 = ["ND", "ND"]
            format_list1 = ["ND", "ND"]

    if soc_version == tbe_platform.ASCEND_910:
        dtype_list = ["float16", "float32", "float32"]
        format_list0 = ["NC1HWC0", "NC1HWC0", "ND"]
        format_list1 = ["ND", "ND", "ND"]
        only_support_nd = dtype_x == "float16" and format_x == "ND"

        if (is_training or (is_support_5hd == SUPPORT_5HD_CASE_2)):
            dtype_list = ["float32", "float32"]
            format_list0 = ["NC1HWC0", "ND"]
            format_list1 = ["ND", "ND"]

        if ((-2 in shape_x) or (-1 in shape_x) or (is_support_5hd == NOT_SUPPORT_5HD) or only_support_nd):
            dtype_list = ["float32"]
            format_list0 = ["ND"]
            format_list1 = ["ND"]

    input0 = gen_param(classify="input0", name="x", datatype=",".join(dtype_list), format=",".join(format_list0))
    input1 = gen_param(classify="input1", name="gamma", datatype=",".join(dtype_list), format=",".join(format_list1))
    input2 = gen_param(classify="input2", name="beta", datatype=",".join(dtype_list), format=",".join(format_list1))
    output0 = gen_param(classify="output0", name="y", datatype=",".join(dtype_list), format=",".join(format_list0))
    output1 = gen_param(classify="output1", name="mean", datatype=",".join(dtype_list), format=",".join(format_list1))
    output2 = gen_param(classify="output2", name="variance", datatype=",".join(dtype_list),
                        format=",".join(format_list1))

    param_dynamic_in_json = get_dynamic_param_in_json([input0, input1, input2, output0, output1, output2])
    return param_dynamic_in_json


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_supported(x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-4,
                    is_training=False, kernel_name="group_norm"):
    """
    check_supported
    """
    format_x = x.get("format")
    shape_x = x.get("shape")
    gamma_shape = scale.get("shape")
    beta_shape = offset.get("shape")
    c0 = 16

    if len(shape_x) < 2:
        return False, "the dim of input x can't be smaller than 2"

    if shape_x[1] != -1 and shape_x[0] != -2 and gamma_shape[0] != -1 and beta_shape[0] != -1:
        if format_x == "NC1HWC0":
            channel = shape_x[1] * c0
        else:
            channel = shape_x[1]

        if gamma_shape[0] != channel:
            return False, "gamma shape is not equal to channel"

        if beta_shape[0] != channel:
            return False, "beta shape is not equal to channel"

        if channel % num_groups != 0:
            return False, "channel must can be divided by num_groups"

    return True, ""


# 'pylint: disable=unused-argument,too-many-locals
class GroupNorm5HD(object):
    """
    object of GroupNorm5HD
    """

    def __init__(self, x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-4,
                 is_training=False, kernel_name="group_norm"):
        self.tik_instance = tik.Tik()
        self.dtype = x.get("dtype")
        self.x_shape = x.get("shape")
        self.is_fp16 = self.dtype == "float16"
        self.fp32 = "float32"
        self.int64 = "int64"
        self.kernel_name = kernel_name
        self.is_training = is_training
        self.format = x.get("format")
        self.c0 = 16
        self.block_byte_size = 32
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_byte_size = common_util.get_data_size(self.fp32)
        self.data_each_block = self.block_byte_size // self.dtype_byte_size
        self.c_burst = self.c0 // self.data_each_block
        self.max_mask = MASK
        self.ub_n = 512
        self.iter_num = 9
        self.scale_n = 512
        self.offset_n = 512
        self.atomic_num = 2 if self.is_fp16 else 1
        self.is_all_same = self.tik_instance.Scalar("int32", init_value=0)
        self.soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        self.support_move_pad = tbe_platform.api_check_support("tik.data_move_pad")

        self.input_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="input_gm")
        self.scale_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="scale_gm")
        self.offset_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="offset_gm")
        self.output_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="output_gm")
        self.mean_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="mean_gm",
                                                is_atomic_add=True)
        self.var_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="var_gm",
                                               is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor("int32", [TILING_NUM], scope=tik.scope_gm, name="tiling_gm")
        self.tmp_ub = None
        self.tiling_mode = None
        self.elem_num = None
        self.elem_num_fp = None
        self.hw_num = None
        self.group_c = None
        self.loop_m = None
        self.last_m = None
        self.loop_w = None
        self.last_w = None
        self.avg_ng = None
        self.block_num = None
        self.last_ng = None
        self.shape_c = None
        self.group_hw = None
        self.hw = None
        self.back_m = None
        self.back_w = None
        self.group_batch, self.loop_batch, self.last_batch = None, None, None
        self.loop_c0, self.last_c0, self.back_loop, self.back_last = None, None, None, None
        self.core_num_var = None
        self.num_groups = None
        self.epsilon = None

    def get_tiling_params(self):
        """
        get runtime params from tiling
        :return: None
        """
        self.tiling_mode = self.tik_instance.Scalar(self.int64)
        self.elem_num = self.tik_instance.Scalar(self.int64)
        self.hw_num = self.tik_instance.Scalar(self.int64)
        self.group_c = self.tik_instance.Scalar(self.int64)
        self.loop_m = self.tik_instance.Scalar(self.int64)
        self.last_m = self.tik_instance.Scalar(self.int64)
        self.loop_w = self.tik_instance.Scalar(self.int64)
        self.last_w = self.tik_instance.Scalar(self.int64)
        self.avg_ng = self.tik_instance.Scalar(self.int64)
        self.block_num = self.tik_instance.Scalar(self.int64)
        self.last_ng = self.tik_instance.Scalar(self.int64)
        self.shape_c = self.tik_instance.Scalar(self.int64)
        self.group_hw = self.tik_instance.Scalar(self.int64)
        self.hw = self.tik_instance.Scalar(self.int64)
        self.elem_num_fp = self.tik_instance.Scalar(self.fp32)
        self.group_batch = self.tik_instance.Scalar(self.int64)
        self.loop_batch = self.tik_instance.Scalar(self.int64)
        self.last_batch = self.tik_instance.Scalar(self.int64)
        self.loop_c0 = self.tik_instance.Scalar(self.int64)
        self.last_c0 = self.tik_instance.Scalar(self.int64)
        self.back_loop = self.tik_instance.Scalar(self.int64)
        self.back_last = self.tik_instance.Scalar(self.int64)
        self.core_num_var = self.tik_instance.Scalar(self.int64)
        self.num_groups = self.tik_instance.Scalar(self.int64)
        self.epsilon = self.tik_instance.Scalar(self.fp32)

        with self.tik_instance.new_stmt_scope():
            tiling_num_block = ceil_div(TILING_NUM, BLOCK_LEN_FP32)
            tiling_ub = self.tik_instance.Tensor("int32", shape=(tiling_num_block * BLOCK_LEN_FP32,),
                                                 scope=tik.scope_ubuf, name="tiling_ub")
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, tiling_num_block, 0, 0)
            tiling_ub_int64 = tiling_ub.reinterpret_cast_to(self.int64)
            self.tiling_mode.set_as(tiling_ub_int64[0])
            self.elem_num.set_as(tiling_ub_int64[1])
            self.hw_num.set_as(tiling_ub_int64[2])
            self.group_c.set_as(tiling_ub_int64[3])
            self.loop_m.set_as(tiling_ub_int64[4])
            self.last_m.set_as(tiling_ub_int64[5])
            self.loop_w.set_as(tiling_ub_int64[6])
            self.last_w.set_as(tiling_ub_int64[7])
            self.avg_ng.set_as(tiling_ub_int64[8])
            self.block_num.set_as(tiling_ub_int64[9])
            self.last_ng.set_as(tiling_ub_int64[10])
            self.shape_c.set_as(tiling_ub_int64[11])
            self.group_hw.set_as(tiling_ub_int64[12])
            self.hw.set_as(tiling_ub_int64[13])
            self.core_num_var.set_as(tiling_ub_int64[14])
            self.num_groups.set_as(tiling_ub_int64[15])
            self.epsilon.set_as(tiling_ub[32])
            self.elem_num_fp.set_as(self.elem_num)
            self.calc_batch_param()

    def calc_batch_param(self):
        self.group_batch.set_as(self.ub_n * self.c0 // self.hw_num)
        with self.tik_instance.if_scope(self.group_batch > self.group_c):
            self.group_batch.set_as(self.group_c)

        with self.tik_instance.if_scope(self.group_batch > 1):
            self.loop_batch.set_as((self.group_c + self.group_batch - 1) // self.group_batch)
            self.last_batch.set_as(self.group_c - (self.loop_batch - 1) * self.group_batch)
            self.loop_c0.set_as((self.group_batch * self.hw_num + self.c0 - 1) // self.c0)
            self.last_c0.set_as((self.last_batch * self.hw_num + self.c0 - 1) // self.c0)
            self.back_loop.set_as(self.loop_c0 * self.c0 - self.group_batch * self.hw_num)
            self.back_last.set_as(self.last_c0 * self.c0 - self.last_batch * self.hw_num)

    def compute(self):
        """
        main compute func
        """
        self.get_tiling_params()
        with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as block_idx:
            ng_num = self.tik_instance.Scalar("int64")
            with self.tik_instance.if_scope(block_idx < self.block_num - 1):
                ng_num.set_as(self.avg_ng)
            with self.tik_instance.if_scope(block_idx == self.block_num - 1):
                ng_num.set_as(self.last_ng)
            self.compute_per_core(block_idx, ng_num)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        outputs = [self.output_gm, self.mean_gm, self.var_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm, self.scale_gm, self.offset_gm],
                                   outputs=outputs,
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    def compute_per_core(self, block_idx, ng_num):
        """
        compute per ai_core
        """
        self.normalize_input(block_idx, ng_num)

    def normalize_input(self, block_idx, ng_num):
        """
        normalization
        """
        ng_idx = self.tik_instance.Scalar("int64")
        g_idx = self.tik_instance.Scalar("int64")
        move_offset = self.tik_instance.Scalar("int64")
        offset = self.tik_instance.Scalar("int64")
        # |--- c1 < num_groups && if_5HD :
        # |   |---  c1 <= 512 : tiling_mode = 3
        # |   |---  group_c <= 512 : tiling_mode = 4
        # |--- else:
        #     |--- c1 <= 512 : tiling_mode = 0
        #     |--- group_c <= 512 : tiling_mode = 1
        #     |--- else : tiling_mode = 2
        with self.tik_instance.if_scope(self.tiling_mode > 2):
            half_c0_status = True
            self.tmp_ub = self.tik_instance.Tensor(self.dtype, [self.ub_n, self.c0 // 2], scope=tik.scope_ubuf,
                                                   name="conv_ub")
            loop_ub = [
                self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0 // 2], scope=tik.scope_ubuf, name="loop_ub_1"),
                self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0 // 2], scope=tik.scope_ubuf, name="loop_ub_2")
            ]
            mean_scalar_1 = self.tik_instance.Scalar(self.fp32)
            var_scalar_1 = self.tik_instance.Scalar(self.fp32)
            mean_scalar_2 = self.tik_instance.Scalar(self.fp32)
            var_scalar_2 = self.tik_instance.Scalar(self.fp32)
            mean_sum_1 = self.tik_instance.Tensor(self.fp32, [self.c0 // 2], scope=tik.scope_ubuf, name="mean_ub_1")
            var_sum_1 = self.tik_instance.Tensor(self.fp32, [self.c0 // 2], scope=tik.scope_ubuf, name="var_ub_1")
            sum_ub0_1 = self.tik_instance.Tensor(self.fp32, [self.c0 // 2], scope=tik.scope_ubuf, name="sum_ub0_1")
            sum_ub1_1 = self.tik_instance.Tensor(self.fp32, [self.c0 // 2], scope=tik.scope_ubuf, name="sum_ub1_1")
            mean_sum_2 = self.tik_instance.Tensor(self.fp32, [self.c0 // 2], scope=tik.scope_ubuf, name="mean_ub_2")
            var_sum_2 = self.tik_instance.Tensor(self.fp32, [self.c0 // 2], scope=tik.scope_ubuf, name="var_ub_2")
            sum_ub0_2 = self.tik_instance.Tensor(self.fp32, [self.c0 // 2], scope=tik.scope_ubuf, name="sum_ub0_2")
            sum_ub1_2 = self.tik_instance.Tensor(self.fp32, [self.c0 // 2], scope=tik.scope_ubuf, name="sum_ub1_2")
            with self.tik_instance.if_scope(self.tiling_mode == 3):
                scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                                    name="scale_ub")
                offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                                     name="offset_ub")
                self.data_move(scale_ub, self.scale_gm, num=self.shape_c,
                               src_stride=0, dst_stride=0, need_conv=True)
                self.data_move(offset_ub, self.offset_gm, num=self.shape_c,
                               src_stride=0, dst_stride=0, need_conv=True)

                with self.tik_instance.for_range(0, ng_num) as n_idx:
                    #  `[block | n * group_num / 2 / block] | 2 | group_c(1), H, W, C0 / 2`
                    ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                    with self.tik_instance.if_scope(self.num_groups != 1):
                        g_idx.set_as(ng_idx % (self.num_groups // 2))
                    with self.tik_instance.else_scope():
                        g_idx.set_as(ng_idx % self.num_groups)
                    self.get_mean_var(loop_ub[0], sum_ub0_1, sum_ub1_1, mean_sum_1, var_sum_1, var_scalar_1,
                                      mean_scalar_1, move_offset, ng_idx, False, move_src_stride=1,
                                      half_c0_status=half_c0_status)
                    with self.tik_instance.for_range(0, self.group_c) as group_idx:
                        offset.set_as((g_idx * self.group_c + group_idx) * self.c0)
                        self.calc_out(loop_ub[0], scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                      mean_scalar_1, var_scalar_1, half_c0_status=half_c0_status)
                    self.get_mean_var(loop_ub[1], sum_ub0_2, sum_ub1_2, mean_sum_2, var_sum_2, var_scalar_2,
                                      mean_scalar_2, move_offset, ng_idx, False,
                                      move_offset_plus=8, move_src_stride=1, out_offset=1,
                                      half_c0_status=half_c0_status)
                    with self.tik_instance.for_range(0, self.group_c) as group_idx:
                        # `[block | n * group_num / 2 / block | 2 | group_c(1)] | H, W, C0 / 2`
                        offset.set_as((g_idx * self.group_c + group_idx) * self.c0)
                        self.calc_out(loop_ub[1], scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                      mean_scalar_2, var_scalar_2, move_offset_plus=8, half_c0_status=half_c0_status)

            with self.tik_instance.if_scope(self.tiling_mode == 4):
                scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                                    name="scale_ub")
                offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                                     name="offset_ub")
                with self.tik_instance.for_range(0, ng_num) as n_idx:
                    ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                    with self.tik_instance.if_scope(self.num_groups != 1):
                        g_idx.set_as(ng_idx % (self.num_groups // 2))
                    with self.tik_instance.else_scope():
                        g_idx.set_as(ng_idx % self.num_groups)
                    offset.set_as(g_idx * self.group_c * self.c0)
                    self.data_move(scale_ub, self.scale_gm[offset], num=self.group_c * self.c0,
                                   src_stride=0, dst_stride=0, need_conv=True)
                    self.data_move(offset_ub, self.offset_gm[offset], num=self.group_c * self.c0,
                                   src_stride=0, dst_stride=0, need_conv=True)

                    self.get_mean_var(loop_ub[0], sum_ub0_1, sum_ub1_1, mean_sum_1, var_sum_1,
                                      var_scalar_1, mean_scalar_1, move_offset,
                                      ng_idx, False, move_src_stride=1, half_c0_status=half_c0_status)
                    with self.tik_instance.for_range(0, self.group_c) as group_idx:
                        offset.set_as(group_idx * self.c0)
                        self.calc_out(loop_ub[0], scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                      mean_scalar_1, var_scalar_1, half_c0_status=half_c0_status)

                    self.get_mean_var(loop_ub[1], sum_ub0_2, sum_ub1_2, mean_sum_2, var_sum_2, var_scalar_2,
                                      mean_scalar_2, move_offset, ng_idx, False,
                                      move_offset_plus=8, move_src_stride=1, out_offset=1,
                                      half_c0_status=half_c0_status)
                    with self.tik_instance.for_range(0, self.group_c) as group_idx:
                        offset.set_as(group_idx * self.c0)
                        self.calc_out(loop_ub[1], scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                      mean_scalar_2, var_scalar_2, move_offset_plus=8, half_c0_status=half_c0_status)

        with self.tik_instance.if_scope(self.tiling_mode < 3):
            self.tmp_ub = self.tik_instance.Tensor("float16", [self.ub_n, self.c0], scope=tik.scope_ubuf,
                                                   name="conv_ub")
            loop_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="loop_ub")
            mean_scalar = self.tik_instance.Scalar(self.fp32)
            var_scalar = self.tik_instance.Scalar(self.fp32)
            mean_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="mean_ub")
            var_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="var_ub")
            sum_ub0 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub0")
            sum_ub1 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub1")

            with self.tik_instance.if_scope(self.tiling_mode == 0):
                scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                                    name="scale_ub")
                offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                                     name="offset_ub")
                self.data_move(scale_ub, self.scale_gm, num=self.shape_c, need_conv=True)
                self.data_move(offset_ub, self.offset_gm, num=self.shape_c, need_conv=True)
                with self.tik_instance.for_range(0, ng_num) as n_idx:
                    ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                    g_idx.set_as(ng_idx % self.num_groups)

                    self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                      move_offset, ng_idx, False)
                    with self.tik_instance.for_range(0, self.group_c) as group_idx:
                        offset.set_as((g_idx * self.group_c + group_idx) * self.c0)
                        self.calc_out(loop_ub, scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                      mean_scalar, var_scalar)

            with self.tik_instance.elif_scope(self.tiling_mode == 1):
                scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                                    name="scale_ub")
                offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                                     name="offset_ub")
                with self.tik_instance.for_range(0, ng_num) as n_idx:
                    ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                    g_idx.set_as(ng_idx % self.num_groups)
                    offset.set_as(g_idx * self.group_c * self.c0)
                    self.data_move(scale_ub, self.scale_gm[offset], num=self.group_c * self.c0, need_conv=True)
                    self.data_move(offset_ub, self.offset_gm[offset], num=self.group_c * self.c0, need_conv=True)

                    self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                      move_offset, ng_idx, False)
                    with self.tik_instance.for_range(0, self.group_c) as group_idx:
                        offset.set_as(group_idx * self.c0)
                        self.calc_out(loop_ub, scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                      mean_scalar, var_scalar)

            with self.tik_instance.elif_scope(self.tiling_mode == 2):
                scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                                    name="scale_ub")
                offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                                     name="offset_ub")
                with self.tik_instance.for_range(0, ng_num) as n_idx:
                    ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                    g_idx.set_as(ng_idx % self.num_groups)

                    self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                      move_offset, ng_idx, False)
                    with self.tik_instance.for_range(0, self.group_c) as group_idx:
                        offset.set_as((g_idx * self.group_c + group_idx) * self.c0)
                        self.data_move(scale_ub, self.scale_gm[offset], num=self.c0, need_conv=True)
                        self.data_move(offset_ub, self.offset_gm[offset], num=self.c0, need_conv=True)
                        offset.set_as(0)
                        self.calc_out(loop_ub, scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                      mean_scalar, var_scalar)

    def calc_out(self, loop_ub, scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                 mean_scalar, var_scalar, move_offset_plus=0, half_c0_status=False):
        """
        sub mean and divide variance, then mul with scale and add offset
        """
        offset.set_as(offset + move_offset_plus)
        with self.tik_instance.for_range(0, self.loop_w) as w_idx:
            move_offset.set_as(ng_idx * self.elem_num + group_idx * self.hw_num +
                               w_idx * self.ub_n * self.c0 + move_offset_plus)
            with self.tik_instance.if_scope(w_idx != self.loop_w - 1):
                if half_c0_status:
                    burst_num = self.c0 // 2
                    self.data_move(loop_ub, self.input_gm[move_offset], num=burst_num,
                                   nburst=self.ub_n * self.c0 // 2 // burst_num, src_stride=8 // 8,
                                   dst_stride=0, need_conv=True)
                    self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.ub_n * self.c0 // 2)
                    self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.ub_n * self.c0 // 2)
                    self.mul_add(loop_ub, scale_ub[offset], offset_ub[offset],
                                 num=self.ub_n * self.c0 // 2, mask=8)
                    self.data_move(self.output_gm[move_offset], loop_ub, num=burst_num,
                                   nburst=self.ub_n * self.c0 // 2 // burst_num, src_stride=0, dst_stride=8 // 8,
                                   need_conv=True, out=True)
                else:
                    self.data_move(loop_ub, self.input_gm[move_offset], num=self.ub_n * self.c0, need_conv=True)
                    self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.ub_n * self.c0)
                    self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.ub_n * self.c0)
                    self.mul_add(loop_ub, scale_ub[offset], offset_ub[offset], num=self.ub_n * self.c0)
                    self.data_move(self.output_gm[move_offset], loop_ub, num=self.ub_n * self.c0,
                                   need_conv=True, out=True)

            with self.tik_instance.else_scope():
                if half_c0_status:
                    burst_num = self.c0 // 2
                    self.data_move(loop_ub, self.input_gm[move_offset], num=burst_num,
                                   nburst=self.last_w * self.c0 // 2 // burst_num, src_stride=8 // 8,
                                   dst_stride=0, need_conv=True)
                    self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.last_w * self.c0 // 2)
                    self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.last_w * self.c0 // 2)
                    self.mul_add(loop_ub, scale_ub[offset], offset_ub[offset],
                                 num=self.last_w * self.c0 // 2, mask=8)
                    self.data_move(self.output_gm[move_offset], loop_ub, num=burst_num,
                                   nburst=self.last_w * self.c0 // 2 // burst_num, src_stride=0, dst_stride=8 // 8,
                                   need_conv=True, out=True)
                else:
                    self.data_move(loop_ub, self.input_gm[move_offset], num=self.last_w * self.c0, need_conv=True)
                    self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.last_w * self.c0)
                    self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.last_w * self.c0)
                    self.mul_add(loop_ub, scale_ub[offset], offset_ub[offset], num=self.last_w * self.c0)
                    self.data_move(self.output_gm[move_offset], loop_ub, num=self.last_w * self.c0,
                                   need_conv=True, out=True)

    def get_data_0(self, data_temp, loop_ub):
        with self.tik_instance.new_stmt_scope():
            loop_ub_temp = self.tik_instance.Tensor(self.fp32, [BLOCK_LEN_FP32], scope=tik.scope_ubuf,
                                                    name="loop_ub_temp")
            if self.is_fp16:
                loop_ub_fp16 = self.tik_instance.Tensor("float16", [BLOCK_LEN_FP16], scope=tik.scope_ubuf,
                                                        name="loop_ub_fp16")
                self.tik_instance.data_move(loop_ub_fp16, loop_ub, 0, 1, 1, 8, 8)
                self.tik_instance.vec_conv(1, "", loop_ub_temp, loop_ub_fp16, 1, 8, 4)
            else:
                self.tik_instance.data_move(loop_ub_temp, loop_ub, 0, 1, 1, 8, 8)
            data_temp.set_as(loop_ub_temp[0])

    def h_reduce_max(self, loop_max_pre, loop_ub, work_tensor, num, src_rep_stride=8):
        repeat_times = self.tik_instance.Scalar("int32", init_value=(num + self.max_mask - 1) // self.max_mask)
        reduce_max_ub = self.tik_instance.Tensor(self.fp32, [2], scope=tik.scope_ubuf, name="reduce_max_ub")
        self.tik_instance.vec_reduce_max(self.max_mask, reduce_max_ub, loop_ub, work_tensor, repeat_times,
                                         src_rep_stride)
        loop_max_pre.set_as(reduce_max_ub[0])

    def h_reduce_min(self, loop_min_pre, loop_ub, work_tensor, num, src_rep_stride=8):
        repeat_times = self.tik_instance.Scalar("int32", init_value=(num + self.max_mask - 1) // self.max_mask)
        reduce_min_ub = self.tik_instance.Tensor(self.fp32, [2], scope=tik.scope_ubuf, name="reduce_min_ub")
        self.tik_instance.vec_reduce_min(self.max_mask, reduce_min_ub, loop_ub, work_tensor, repeat_times,
                                         src_rep_stride)
        loop_min_pre.set_as(reduce_min_ub[0])

    def get_mean_scalar_impl(self, loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum,
                             move_offset, ng_idx, is_nd, act_c0, num, nburst, last_num, last_nburst,
                             elem_num_fp, iter_num, mask, move_offset_plus=0, move_src_stride=0,
                             move_dst_stride=0, out_offset=0):
        tik_func_vector_support_tensor(self.tik_instance, mean_sum, 0, act_c0)
        tik_func_vector_support_tensor(self.tik_instance, var_sum, 0, act_c0)
        tik_func_vector_support_tensor(self.tik_instance, sum_ub0, 0, act_c0)
        tik_func_vector_support_tensor(self.tik_instance, sum_ub1, 0, act_c0)
        tmp_var = self.tik_instance.Tensor(self.fp32, [act_c0], scope=tik.scope_ubuf, name="tmp_ub")
        loop_ub2 = self.tik_instance.Tensor(self.fp32, [self.ub_n, act_c0], scope=tik.scope_ubuf,
                                            name="loop_ub2")
        tik_func_vector_support_tensor(self.tik_instance, loop_ub2, 0, self.ub_n * act_c0)
        work_tensor = self.tik_instance.Tensor(self.fp32, [256], scope=tik.scope_ubuf, name="work_ub")
        is_910b_or_910_93 = self.soc_version == "Ascend910B" or self.soc_version == "Ascend910_93"

        if is_910b_or_910_93:
            data_temp = self.tik_instance.Scalar(self.fp32)
            loop_max = self.tik_instance.Scalar(self.fp32)
            loop_min = self.tik_instance.Scalar(self.fp32)
            loop_max_pre = self.tik_instance.Scalar(self.fp32)
            loop_min_pre = self.tik_instance.Scalar(self.fp32)

        with self.tik_instance.for_range(0, self.loop_m) as m_idx:
            move_offset.set_as(ng_idx * self.elem_num + m_idx * self.ub_n * self.c0 + move_offset_plus)
            with self.tik_instance.if_scope(m_idx != self.loop_m - 1):
                self.data_move(loop_ub, self.input_gm[move_offset], num, nburst,
                               src_stride=move_src_stride, dst_stride=move_dst_stride, need_conv=True)
                self.data_mul(loop_ub2, loop_ub, loop_ub, [0, 0, 0], num=self.ub_n * act_c0)
                if is_910b_or_910_93:
                    with self.tik_instance.if_scope(m_idx == 0):
                        loop_max.set_as(loop_ub[0])
                        loop_min.set_as(loop_ub[0])
                    self.get_data_0(data_temp, loop_ub)
                    loop_max_pre.set_as(data_temp)
                    loop_min_pre.set_as(data_temp)
                    self.h_reduce_max(loop_max_pre, loop_ub, work_tensor, num)
                    self.h_reduce_min(loop_min_pre, loop_ub, work_tensor, num)
            with self.tik_instance.else_scope():
                tik_func_vector_support_tensor(self.tik_instance, loop_ub2, 0, self.ub_n * act_c0)
                self.data_move(loop_ub, self.input_gm[move_offset], last_num, last_nburst,
                               src_stride=move_src_stride, dst_stride=move_dst_stride, need_conv=True)
                if is_910b_or_910_93:
                    with self.tik_instance.if_scope(m_idx == 0):
                        loop_max.set_as(loop_ub[0])
                        loop_min.set_as(loop_ub[0])
                    self.get_data_0(data_temp, loop_ub)
                    loop_max_pre.set_as(data_temp)
                    loop_min_pre.set_as(data_temp)
                    tik_func_vector_support_tensor(self.tik_instance, loop_ub[self.last_m * self.c0], data_temp,
                                                   self.ub_n * act_c0 - self.last_m * self.c0)
                    if is_nd:
                        self.back_value(loop_ub, self.back_m, self.last_m * self.c0, data_temp)
                    self.h_reduce_max(loop_max_pre, loop_ub, work_tensor, last_num)
                    self.h_reduce_min(loop_min_pre, loop_ub, work_tensor, last_num)

                tik_func_vector_support_tensor(self.tik_instance, loop_ub[self.last_m * self.c0], 0,
                                               self.ub_n * act_c0 - self.last_m * self.c0)

                if is_nd:
                    self.back_value(loop_ub, self.back_m, self.last_m * self.c0, 0)
                self.data_mul(loop_ub2, loop_ub, loop_ub, [0, 0, 0], num=self.last_m * act_c0)

            self.data_sum(loop_ub, self.ub_n * act_c0, iter_num)
            self.tik_instance.vcadd(16, sum_ub0, loop_ub, 1, 1, 1, 1)
            self.tik_instance.vec_add(mask, mean_sum, mean_sum, sum_ub0, 1, 1, 1, 1)
            self.data_sum(loop_ub2, self.ub_n * act_c0, iter_num)
            self.tik_instance.vcadd(16, sum_ub1, loop_ub2, 1, 1, 1, 1)
            self.tik_instance.vec_add(mask, var_sum, var_sum, sum_ub1, 1, 1, 1, 1)

            if is_910b_or_910_93:
                with self.tik_instance.if_scope(loop_max < loop_max_pre):
                    loop_max.set_as(loop_max_pre)
                with self.tik_instance.if_scope(loop_min > loop_min_pre):
                    loop_min.set_as(loop_min_pre)

        if is_910b_or_910_93:
            with self.tik_instance.if_scope(loop_max == loop_min):
                self.is_all_same.set_as(1)
        self.tik_instance.vec_muls(mask, mean_sum, mean_sum, 1 / elem_num_fp, 1, 1, 1)
        return tmp_var, work_tensor

    def get_var_scalar_impl(self, var_sum, mean_sum, elem_num_fp, mask, act_c0):
        self.tik_instance.vec_muls(mask, var_sum, var_sum, 1 / elem_num_fp, 1, 1, 1)
        self.tik_instance.vec_mul(mask, mean_sum, mean_sum, mean_sum, 1, 1, 1, 1)
        self.tik_instance.vec_muls(mask, mean_sum, mean_sum, -1, 1, 1, 1)
        self.data_add(var_sum, var_sum, mean_sum, [0, 0, 0], num=act_c0)

    def get_mean_var(self, loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                     move_offset, ng_idx, is_nd, move_offset_plus=0, move_src_stride=0, move_dst_stride=0,
                     out_offset=0, half_c0_status=False):
        if half_c0_status:
            tmp_var, work_tensor = self.get_mean_scalar_impl(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum,
                                                             move_offset, ng_idx, is_nd, self.c0 // 2,
                                                             8, self.ub_n * self.c0 // 2 // 8, 8,
                                                             self.last_m * self.c0 // 2 // 8, self.elem_num_fp / 2,
                                                             self.iter_num - 1, 8, move_offset_plus, move_src_stride,
                                                             move_dst_stride, out_offset)
        else:
            tmp_var, work_tensor = self.get_mean_scalar_impl(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum,
                                                             move_offset, ng_idx, is_nd, self.c0,
                                                             self.ub_n * self.c0, 1, self.last_m * self.c0, 1,
                                                             self.elem_num_fp, self.iter_num, 16, move_offset_plus,
                                                             move_src_stride, move_dst_stride, out_offset)

        if self.is_training:
            self.tik_instance.set_atomic_add(self.atomic_num)
            if half_c0_status:
                self.data_move(self.mean_gm[2 * ng_idx + out_offset], mean_sum, num=1, nburst=self.c0 // 2 // 8,
                               src_stride=move_dst_stride, dst_stride=move_src_stride, need_conv=True, out=True,
                               conv_num=BLOCK_LEN_FP32)
            elif self.is_fp16:
                self.data_move(self.mean_gm[ng_idx], mean_sum, 1, need_conv=True, out=True, conv_num=BLOCK_LEN_FP16)
            else:
                self.data_move(self.mean_gm[ng_idx], mean_sum, 1, need_conv=True, out=True, conv_num=BLOCK_LEN_FP32)
            self.tik_instance.set_atomic_add(0)
        mean_scalar.set_as(mean_sum[0])
        mean_scalar.set_as(-1 * mean_scalar)

        with self.tik_instance.if_scope(self.is_all_same == 0):
            if half_c0_status:
                self.get_var_scalar_impl(var_sum, mean_sum, self.elem_num_fp / 2, 8, self.c0 // 2)
            else:
                self.get_var_scalar_impl(var_sum, mean_sum, self.elem_num_fp, 16, self.c0)
            if self.is_training:
                self.tik_instance.set_atomic_add(self.atomic_num)
                if half_c0_status:
                    self.data_move(self.var_gm[2 * ng_idx + out_offset], var_sum, num=1, nburst=self.c0 // 2 // 8,
                                   src_stride=move_dst_stride, dst_stride=move_src_stride, need_conv=True, out=True,
                                   conv_num=BLOCK_LEN_FP32)
                else:
                    self.data_move(self.var_gm[ng_idx], var_sum, 1, need_conv=True, out=True, conv_num=BLOCK_LEN_FP32)
                self.tik_instance.set_atomic_add(0)
            if half_c0_status:
                self.tik_instance.vec_adds(8, var_sum, var_sum, self.epsilon, 1, 1, 1)
                self.tik_instance.vec_rsqrt_high_preci(8, tmp_var, var_sum, work_tensor, 1, 1, 1)
            else:
                self.tik_instance.vec_adds(16, var_sum, var_sum, self.epsilon, 1, 1, 1)
                self.tik_instance.vec_rsqrt_high_preci(16, tmp_var, var_sum, work_tensor, 1, 1, 1)
            var_scalar.set_as(tmp_var[0])
        with self.tik_instance.else_scope():
            var_scalar.set_as(0)

    def data_move(self, dst, src, num, nburst=1, src_stride=0, dst_stride=0, need_conv=False, out=False,
                  conv_num=-1):
        """
        move data
        """
        if self.is_fp16 and need_conv:
            if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") not in\
                ["Ascend910B", tbe_platform.ASCEND_910, "Ascend910_93"]:
                vconv_num = num if conv_num == -1 else conv_num
                if not out:
                    self.select_data_move(self.tmp_ub, src, count=num, repeat=1, src_stride=src_stride,
                                          dst_stride=dst_stride, out=out)
                    self.data_conv(dst, self.tmp_ub, [0, 0], mode="", num=vconv_num, dst_stride=8, src_stride=4)
                else:
                    self.data_conv(self.tmp_ub, src, [0, 0], mode="", num=vconv_num, dst_stride=4, src_stride=8)
                    self.select_data_move(dst, self.tmp_ub, count=num, repeat=1, src_stride=src_stride,
                                          dst_stride=dst_stride, out=out)
            else:
                if not out:
                    self.select_data_move(self.tmp_ub, src, count=num, out=out)
                    self.data_conv(dst, self.tmp_ub, [0, 0], mode="", num=num, dst_stride=8, src_stride=4)
                else:
                    self.data_conv(self.tmp_ub, src, [0, 0], mode="", num=num, dst_stride=4, src_stride=8)
                    self.select_data_move(dst, self.tmp_ub, count=num, out=out)
        else:
            self.select_data_move(dst, src, num, nburst, src_stride, dst_stride, out)
    
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def select_data_move(self, dst, src, count, repeat=1, src_stride=0, dst_stride=0, out=False):
        dtype_size = common_util.get_data_size(src.dtype)
        if self.support_move_pad:
            burst = dtype_size * count
            if out:
                dst_gap = dst_stride * constant_util.BLOCK_SIZE
                src_gap = src_stride
            else:
                dst_gap = dst_stride
                src_gap = src_stride * constant_util.BLOCK_SIZE
            self.tik_instance.data_move_pad(dst, src, repeat, burst, dst_gap, src_gap)
        else:
            block_element = constant_util.BLOCK_SIZE // dtype_size
            burst = ceil_div(count, block_element)
            self.tik_instance.data_move(dst, src, 0, repeat, burst, src_stride, dst_stride)

    def data_sum(self, src, num, iter_num):
        """
        sum data
        """
        for _ in range(iter_num):
            num = num // 2
            if num // self.max_mask > 0:
                mask = self.max_mask
                repeat_time = num // self.max_mask
            else:
                mask = num
                repeat_time = 1

            src_stride = mask // self.data_each_block
            self.tik_instance.vec_add(mask, src, src[num], src, repeat_time, 0, src_stride, 0)

    def mul_add(self, loop_ub, scale_ub, offset_ub, num, mask=16):
        """
        mul and add
        """
        loop = num // (mask * MAX_REPEAT_NUM)
        stride = mask // self.data_each_block

        offset = self.tik_instance.Scalar("int64", init_value=0)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = index * mask * MAX_REPEAT_NUM
                self.tik_instance.vec_mul(mask, loop_ub[tmp_offset], loop_ub[tmp_offset], scale_ub, MAX_REPEAT_NUM,
                                          stride, stride, 0)
                self.tik_instance.vec_add(mask, loop_ub[tmp_offset], loop_ub[tmp_offset], offset_ub, MAX_REPEAT_NUM,
                                          stride, stride, 0)

            offset.set_as(loop * mask * MAX_REPEAT_NUM)

        repeat_time = (num % (mask * MAX_REPEAT_NUM)) // mask

        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_mul(mask, loop_ub[offset], loop_ub[offset], scale_ub, repeat_time,
                                      stride, stride, 0)
            self.tik_instance.vec_add(mask, loop_ub[offset], loop_ub[offset], offset_ub, repeat_time,
                                      stride, stride, 0)
            offset.set_as(offset + repeat_time * mask)

        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_mul(last_num, loop_ub[offset], loop_ub[offset], scale_ub, 1,
                                      stride, stride, 0)
            self.tik_instance.vec_add(last_num, loop_ub[offset], loop_ub[offset], offset_ub, 1,
                                      stride, stride, 0)

    def back_value(self, loop_ub, back_num, ub_num, value):
        """
        when format is ND, need set certain elements to a value
        """
        with self.tik_instance.if_scope(back_num > 0):
            with self.tik_instance.for_range(0, back_num) as idx:
                loop_ub[ub_num - 1 - idx].set_as(value)

    def single_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        dst_offset = self.tik_instance.Scalar("int64", init_value=offsets[0])
        src_offset = self.tik_instance.Scalar("int64", init_value=offsets[1])
        vector_mask_max = 256 // self.dtype_byte_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * MAX_REPEAT_NUM)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * MAX_REPEAT_NUM
                tmp_src_offset = src_offset + index * vector_mask_max * MAX_REPEAT_NUM
                op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], scalar, MAX_REPEAT_NUM,
                       dst_stride, src_stride)

            dst_offset.set_as(dst_offset + loop * vector_mask_max * MAX_REPEAT_NUM)
            src_offset.set_as(src_offset + loop * vector_mask_max * MAX_REPEAT_NUM)

        repeat_time = (tensor_size % (vector_mask_max * MAX_REPEAT_NUM)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_stride, src_stride)
            dst_offset.set_as(dst_offset + repeat_time * vector_mask_max)
            src_offset.set_as(src_offset + repeat_time * vector_mask_max)

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_stride, src_stride)

    def data_conv(self, dst, src, offsets, mode="ceil", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        dst_offset = self.tik_instance.Scalar("int64", init_value=offsets[0])
        src_offset = self.tik_instance.Scalar("int64", init_value=offsets[1])

        tensor_size = num
        loop = tensor_size // (MASK * MAX_REPEAT_NUM)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * MASK * MAX_REPEAT_NUM
                tmp_src_offset = src_offset + index * MASK * MAX_REPEAT_NUM
                self.tik_instance.vec_conv(MASK, mode, dst[tmp_dst_offset], src[tmp_src_offset], MAX_REPEAT_NUM,
                                           dst_stride, src_stride)

            dst_offset.set_as(dst_offset + loop * MASK * MAX_REPEAT_NUM)
            src_offset.set_as(src_offset + loop * MASK * MAX_REPEAT_NUM)

        repeat_time = (tensor_size % (MASK * MAX_REPEAT_NUM)) // MASK

        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_conv(MASK, mode, dst[dst_offset], src[src_offset], repeat_time,
                                       dst_stride, src_stride)
            dst_offset.set_as(dst_offset + repeat_time * MASK)
            src_offset.set_as(src_offset + repeat_time * MASK)

        last_num = tensor_size % MASK
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    def data_adds(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik adds
        """
        self.single_operator_template(self.tik_instance.vec_adds, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)

    def data_add(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik add
        """
        tik_func_double_input_new(self.tik_instance, "vadd", out_dst=dst, src0=src0, src1=src1, copy_num=num)

    def data_mul(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik mul
        """
        tik_func_double_input_new(self.tik_instance, "vmul", out_dst=dst, src0=src0, src1=src1, copy_num=num)

    def data_muls(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik muls
        """
        self.single_operator_template(self.tik_instance.vec_muls, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)


class GroupNormND(GroupNorm5HD):
    """
    object of GroupNorm when format is ND
    """

    def __init__(self, x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-4,
                 is_training=False, kernel_name="group_norm"):
        super(GroupNormND, self).__init__(x, scale, offset, y, mean, variance, num_groups, data_format, epsilon,
                                          is_training, kernel_name)
        self.input_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="input_gm_nd")
        self.scale_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="scale_gm_nd")
        self.offset_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="offset_gm_nd")
        self.output_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="output_gm_nd",
                                                  is_atomic_add=True)
        self.mean_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="mean_gm_nd",
                                                is_atomic_add=self.is_training)
        self.var_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT64], scope=tik.scope_gm, name="var_gm_nd",
                                               is_atomic_add=self.is_training)
        self.tiling_gm = self.tik_instance.Tensor(self.int64, [TILING_NUM], scope=tik.scope_gm, name="tiling_gm_nd")

    def compute(self):
        """
        main compute func
        """
        self.get_tiling_params()
        with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as block_idx:
            self.compute_back()
            ng_num = self.tik_instance.Scalar("int64")
            with self.tik_instance.if_scope(block_idx < self.block_num - 1):
                ng_num.set_as(self.avg_ng)
            with self.tik_instance.if_scope(block_idx == self.block_num - 1):
                ng_num.set_as(self.last_ng)

            self.compute_per_core(block_idx, ng_num)

        outputs = [self.output_gm, self.mean_gm, self.var_gm]
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm, self.scale_gm, self.offset_gm],
                                   outputs=outputs,
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    def normalize_input(self, block_idx, ng_num):
        """
        normalize x
        """
        self.tmp_ub = self.tik_instance.Tensor("float16", [self.ub_n, self.c0], scope=tik.scope_ubuf,
                                               name="conv_ub")
        with self.tik_instance.if_scope(tik.all(self.group_batch > 1, self.hw_num % 16 == 0, self.tiling_mode == 0)):
            self.normalize_batch_hw(block_idx, ng_num)
        with self.tik_instance.else_scope():
            loop_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="loop_ub")
            with self.tik_instance.if_scope(self.elem_num == 1):
                self.normalize_one_num(block_idx, ng_num, loop_ub)
            with self.tik_instance.elif_scope(tik.all(self.group_batch > 1, self.hw_num == 1)):
                self.normalize_batch(block_idx, ng_num, loop_ub)
            with self.tik_instance.else_scope():
                self.normalize_c(block_idx, ng_num, loop_ub)

    def normalize_batch(self, block_idx, ng_num, loop_ub):
        ng_idx = self.tik_instance.Scalar("int64")
        g_idx = self.tik_instance.Scalar("int64")
        move_offset = self.tik_instance.Scalar("int64")
        offset = self.tik_instance.Scalar("int64")
        mean_scalar = self.tik_instance.Scalar(self.fp32)
        var_scalar = self.tik_instance.Scalar(self.fp32)
        mean_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="mean_ub")
        var_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="var_ub")
        sum_ub0 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub0")
        sum_ub1 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub1")
        scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                            name="scale_ub")
        offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                             name="offset_ub")

        with self.tik_instance.for_range(0, ng_num) as n_idx:
            ng_idx.set_as(block_idx * self.avg_ng + n_idx)
            g_idx.set_as(ng_idx % self.num_groups)

            self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                              move_offset, ng_idx, True)
            self.calc_batch(loop_ub, scale_ub, offset_ub, move_offset, ng_idx, g_idx, mean_scalar, var_scalar, offset)

    def normalize_c(self, block_idx, ng_num, loop_ub):
        ng_idx = self.tik_instance.Scalar("int64")
        g_idx = self.tik_instance.Scalar("int64")
        move_offset = self.tik_instance.Scalar("int64")
        offset = self.tik_instance.Scalar("int64")
        scale_scalar = self.tik_instance.Scalar(self.fp32)
        offset_scalar = self.tik_instance.Scalar(self.fp32)
        mean_scalar = self.tik_instance.Scalar(self.fp32)
        var_scalar = self.tik_instance.Scalar(self.fp32)
        last_hw = self.tik_instance.Scalar("int64", init_value=(self.hw_num - (self.loop_w - 1) * self.ub_n * self.c0))
        mean_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="mean_ub")
        var_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="var_ub")
        sum_ub0 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub0")
        sum_ub1 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub1")
        scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                            name="scale_ub")
        offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                             name="offset_ub")

        with self.tik_instance.if_scope(self.tiling_mode == 0):
            self.data_move(scale_ub, self.scale_gm, num=self.shape_c, need_conv=True)
            self.data_move(offset_ub, self.offset_gm, num=self.shape_c, need_conv=True)

            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)

                self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                  move_offset, ng_idx, True)
                with self.tik_instance.if_scope(self.is_all_same == 0):
                    with self.tik_instance.for_range(0, self.group_c) as group_idx:
                        offset.set_as(g_idx * self.group_c + group_idx)
                        scale_scalar.set_as(scale_ub[offset])
                        offset_scalar.set_as(offset_ub[offset])
                        self.calc_out_nd(loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx,
                                         mean_scalar, var_scalar, last_hw)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.group_c) as group_idx:
                        offset.set_as(g_idx * self.group_c + group_idx)
                        scale_scalar.set_as(scale_ub[offset])
                        offset_scalar.set_as(offset_ub[offset])
                        self.calc_out_nd_all_same(loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx,
                                                  mean_scalar, var_scalar, last_hw)

        with self.tik_instance.elif_scope(self.tiling_mode == 1):
            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)
                offset.set_as(g_idx * self.group_c)
                self.data_move(scale_ub, self.scale_gm[offset], num=self.group_c, need_conv=True)
                self.data_move(offset_ub, self.offset_gm[offset], num=self.group_c, need_conv=True)

                self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                  move_offset, ng_idx, True)
                with self.tik_instance.for_range(0, self.group_c) as group_idx:
                    scale_scalar.set_as(scale_ub[group_idx])
                    offset_scalar.set_as(offset_ub[group_idx])
                    self.calc_out_nd(loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx, mean_scalar,
                                     var_scalar, last_hw)

        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)

                self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                  move_offset, ng_idx, True)
                with self.tik_instance.for_range(0, self.group_c) as group_idx:
                    offset.set_as(g_idx * self.group_c + group_idx)
                    self.data_move(scale_ub, self.scale_gm[offset], num=self.c0, need_conv=True)
                    self.data_move(offset_ub, self.offset_gm[offset], num=self.c0, need_conv=True)
                    scale_scalar.set_as(scale_ub[0])
                    offset_scalar.set_as(offset_ub[0])
                    self.calc_out_nd(loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx, mean_scalar,
                                     var_scalar, last_hw)

    def normalize_one_num(self, block_idx, ng_num, loop_ub):
        ng_idx = self.tik_instance.Scalar("int64")
        g_idx = self.tik_instance.Scalar("int64")
        tmp_idx = self.tik_instance.Scalar("int64")
        move_offset = self.tik_instance.Scalar("int64")
        move_loop = self.tik_instance.Scalar("int64")
        move_value = self.tik_instance.Scalar(self.fp32)
        move_last = self.tik_instance.Scalar("int64")
        mean_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="mean_ub")
        sum_ub0 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub0")
        offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                             name="offset_ub")
        out_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                          name="out_ub")

        if self.is_training:
            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                tik_func_vector_support_tensor(self.tik_instance, mean_sum, 0, self.c0)
                tik_func_vector_support_tensor(self.tik_instance, loop_ub, 0, self.c0)

                move_offset.set_as(ng_idx * self.elem_num)
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.c0, need_conv=True)
                self.tik_instance.set_atomic_add(self.atomic_num)
                mean_sum[0].set_as(loop_ub[0])
                self.data_move(self.mean_gm[ng_idx], mean_sum, 8, need_conv=True, out=True)
                self.tik_instance.set_atomic_add(0)

        self.tik_instance.set_atomic_add(self.atomic_num)

        move_loop.set_as((ng_num + self.scale_n * self.c0 - 1) // (self.scale_n * self.c0))
        move_last.set_as(ng_num - (move_loop - 1) * self.scale_n * self.c0)
        with self.tik_instance.if_scope(self.num_groups == 1):
            self.data_move(offset_ub, self.offset_gm[0], num=self.group_c, need_conv=True)
            move_value.set_as(offset_ub[0])
            ng_idx.set_as(block_idx * self.avg_ng)
            with self.tik_instance.for_range(0, move_loop) as loop_idx:
                move_offset.set_as(ng_idx + loop_idx * self.scale_n * self.c0)
                with self.tik_instance.if_scope(loop_idx != move_loop - 1):
                    tik_func_vector_support_tensor(self.tik_instance, out_ub, move_value, self.scale_n * self.c0)
                    self.data_move(self.output_gm[move_offset], out_ub, num=self.scale_n * self.c0,
                                   need_conv=True, out=True)

                with self.tik_instance.else_scope():
                    tik_func_vector_support_tensor(self.tik_instance, out_ub, 0, self.scale_n * self.c0)
                    tik_func_vector_support_tensor(self.tik_instance, out_ub, move_value, move_last)
                    self.data_move(self.output_gm[move_offset], out_ub, num=move_last, need_conv=True, out=True)
        with self.tik_instance.elif_scope(self.num_groups < self.scale_n * self.c0):
            self.move_offset(offset_ub, out_ub, ng_idx, block_idx, g_idx, tmp_idx, move_offset, move_loop, move_last)
        with self.tik_instance.else_scope():
            tik_func_vector_support_tensor(self.tik_instance, sum_ub0, 0, self.c0)
            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)
                self.data_move(offset_ub, self.offset_gm[g_idx], num=self.group_c, need_conv=True)

                move_offset.set_as(ng_idx * self.elem_num)
                sum_ub0[0].set_as(offset_ub[0])

                self.data_move(self.output_gm[move_offset], sum_ub0, num=8, need_conv=True, out=True)

        self.tik_instance.set_atomic_add(0)

    def get_mean_var_pre(self, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar, move_offset, ng_idx, is_nd,
                         move_offset_plus=0, move_src_stride=0, move_dst_stride=0, out_offset=0, half_c0_status=False):
        """
        apply new loop_ub for func: get_mean_var
        """
        with self.tik_instance.new_stmt_scope():
            loop_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="loop_ub")
            self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar, move_offset,
                              ng_idx, True)

    def normalize_batch_hw(self, block_idx, ng_num):
        """
        normalize input data when hw can be divided by 16
        """
        ng_idx = self.tik_instance.Scalar("int64")
        g_idx = self.tik_instance.Scalar("int64")
        move_offset = self.tik_instance.Scalar("int64")
        offset = self.tik_instance.Scalar("int64")
        mean_scalar = self.tik_instance.Scalar(self.fp32)
        var_scalar = self.tik_instance.Scalar(self.fp32)
        scale_scalar = self.tik_instance.Scalar(self.fp32)
        offset_scalar = self.tik_instance.Scalar(self.fp32)
        mean_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="mean_ub")
        var_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="var_ub")
        sum_ub0 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub0")
        sum_ub1 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub1")
        scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                            name="scale_ub")
        offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                             name="offset_ub")
        self.data_move(scale_ub, self.scale_gm, num=self.shape_c, need_conv=True)
        self.data_move(offset_ub, self.offset_gm, num=self.shape_c, need_conv=True)

        with self.tik_instance.for_range(0, ng_num) as n_idx:
            ng_idx.set_as(block_idx * self.avg_ng + n_idx)
            g_idx.set_as(ng_idx % self.num_groups)

            self.get_mean_var_pre(sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar, move_offset, ng_idx,
                                  True)
            self.calc_batch_hw(scale_ub, offset_ub, scale_scalar, offset_scalar, move_offset, ng_idx, g_idx,
                               mean_scalar, var_scalar, offset)

    def calc_batch_hw(self, scale_ub, offset_ub, scale_scalar, offset_scalar, move_offset, ng_idx, g_idx, mean_scalar,
                      var_scalar, offset):
        """
        batch hw sub mean and div var
        """
        with self.tik_instance.for_range(0, self.loop_batch, thread_num=2) as loop_idx:
            loop_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="loop_ub")

            move_offset.set_as(ng_idx * self.elem_num + loop_idx * self.group_batch * self.hw_num)
            offset.set_as(g_idx * self.group_c + self.group_batch * loop_idx)

            with self.tik_instance.if_scope(loop_idx != self.loop_batch - 1):
                self.calc_hw(loop_ub, scale_ub, offset_ub, mean_scalar, var_scalar, scale_scalar, offset_scalar,
                             move_offset, offset, self.group_batch)
            with self.tik_instance.else_scope():
                self.calc_hw(loop_ub, scale_ub, offset_ub, mean_scalar, var_scalar, scale_scalar, offset_scalar,
                             move_offset, offset, self.last_batch)

    def calc_hw(self, loop_ub, scale_ub, offset_ub, mean_scalar, var_scalar, scale_scalar, offset_scalar, move_offset,
                offset, batch_num):
        """
        hw sub mean and div var
        """
        self.data_move(loop_ub, self.input_gm[move_offset], num=batch_num * self.hw_num, need_conv=True)
        self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=batch_num * self.hw_num)
        self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=batch_num * self.hw_num)

        with self.tik_instance.for_range(0, batch_num) as batch_idx:
            scale_scalar.set_as(scale_ub[offset + batch_idx])
            offset_scalar.set_as(offset_ub[offset + batch_idx])

            self.data_muls(loop_ub, loop_ub, scale_scalar, [batch_idx * self.hw_num, batch_idx * self.hw_num],
                           num=self.hw_num)
            self.data_adds(loop_ub, loop_ub, offset_scalar, [batch_idx * self.hw_num, batch_idx * self.hw_num],
                           num=self.hw_num)

        self.data_move(self.output_gm[move_offset], loop_ub, num=batch_num * self.hw_num,
                       need_conv=True, out=True)

    def move_offset(self, offset_ub, out_ub, ng_idx, block_idx, g_idx, tmp_idx, move_offset, move_loop, move_last):
        self.data_move(offset_ub, self.offset_gm, num=self.num_groups, need_conv=True)
        ng_idx.set_as(block_idx * self.avg_ng)
        with self.tik_instance.for_range(0, move_loop) as loop_idx:
            move_offset.set_as(ng_idx + loop_idx * self.scale_n * self.c0)
            with self.tik_instance.if_scope(loop_idx != move_loop - 1):
                with self.tik_instance.for_range(0, self.scale_n * self.c0) as idx:
                    tmp_idx.set_as(move_offset + idx)
                    g_idx.set_as(tmp_idx % self.num_groups)
                    out_ub[idx].set_as(offset_ub[g_idx])

                self.data_move(self.output_gm[move_offset], out_ub, num=self.scale_n * self.c0, need_conv=True,
                               out=True)

            with self.tik_instance.else_scope():
                tik_func_vector_support_tensor(self.tik_instance, out_ub, 0, self.scale_n * self.c0)
                with self.tik_instance.for_range(0, move_last) as idx:
                    tmp_idx.set_as(move_offset + idx)
                    g_idx.set_as(tmp_idx % self.num_groups)
                    out_ub[idx].set_as(offset_ub[g_idx])

                self.data_move(self.output_gm[move_offset], out_ub, num=move_last, need_conv=True,
                               out=True)

    def compute_back(self):
        """
        compute number of back
        """
        self.back_m = self.tik_instance.Scalar("int64")
        self.back_w = self.tik_instance.Scalar("int64")
        self.back_m.set_as(self.group_hw * self.c0 - self.elem_num)
        self.back_w.set_as(self.hw * self.c0 - self.hw_num)

    def calc_out_nd(self, loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx,
                    mean_scalar, var_scalar, last_hw):
        """
        calculate output when format is ND
        """
        with self.tik_instance.for_range(0, self.loop_w) as w_idx:
            move_offset.set_as(ng_idx * self.elem_num + group_idx * self.hw_num +
                               w_idx * self.ub_n * self.c0)
            with self.tik_instance.if_scope(w_idx != self.loop_w - 1):
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.ub_n * self.c0, need_conv=True)
                self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.ub_n * self.c0)
                self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.ub_n * self.c0)
                self.mul_add_nd(loop_ub, scale_scalar, offset_scalar, num=self.ub_n * self.c0)
                self.data_move(self.output_gm[move_offset], loop_ub, num=self.ub_n * self.c0, need_conv=True,
                               out=True)
            with self.tik_instance.else_scope():
                self.tik_instance.set_atomic_add(self.atomic_num)
                tik_func_vector_support_tensor(self.tik_instance, loop_ub, 0, self.ub_n * self.c0)
                self.data_move(loop_ub, self.input_gm[move_offset], num=last_hw, need_conv=True)
                self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.last_w * self.c0)
                self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.last_w * self.c0)
                self.mul_add_nd(loop_ub, scale_scalar, offset_scalar, num=self.last_w * self.c0)
                self.back_value(loop_ub, self.back_w, self.last_w * self.c0, 0)
                if self.is_fp16:
                    self.data_conv(self.tmp_ub, loop_ub, [0, 0], mode="", num=self.last_w * self.c0,
                                   dst_stride=4, src_stride=8)
                    self.select_data_move(self.output_gm[move_offset], self.tmp_ub, last_hw, out=True)
                else:
                    self.select_data_move(self.output_gm[move_offset], loop_ub, last_hw, out=True)
                self.tik_instance.set_atomic_add(0)

    def calc_out_nd_all_same(self, loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx,
                             mean_scalar, var_scalar, last_hw):
        """
        calculate output when format is ND
        """
        with self.tik_instance.for_range(0, self.loop_w) as w_idx:
            move_offset.set_as(ng_idx * self.elem_num + group_idx * self.hw_num +
                               w_idx * self.ub_n * self.c0)
            with self.tik_instance.if_scope(w_idx != self.loop_w - 1):
                self.tik_instance.vec_dup(MASK, loop_ub, offset_scalar, self.ub_n * self.c0 // MASK,
                                          MASK // BLOCK_LEN_FP32)
                self.data_move(self.output_gm[move_offset], loop_ub, num=self.ub_n * self.c0, need_conv=True,
                               out=True)
            with self.tik_instance.else_scope():
                self.tik_instance.set_atomic_add(self.atomic_num)
                self.tik_instance.vec_dup(MASK, loop_ub, offset_scalar, self.last_w * self.c0 // MASK,
                                          MASK // BLOCK_LEN_FP32)
                if self.is_fp16:
                    self.data_conv(self.tmp_ub, loop_ub, [0, 0], mode="", num=self.last_w * self.c0,
                                   dst_stride=4, src_stride=8)
                    self.back_value(self.tmp_ub, self.back_w, self.last_w * self.c0, 0)
                    self.select_data_move(self.output_gm[move_offset], self.tmp_ub, last_hw, out=True)
                else:
                    self.back_value(loop_ub, self.back_w, self.last_w * self.c0, 0)
                    self.select_data_move(self.output_gm[move_offset], loop_ub, last_hw, out=True)
                self.tik_instance.set_atomic_add(0)

    def calc_batch(self, loop_ub, scale_ub, offset_ub, move_offset, ng_idx, g_idx, mean_scalar, var_scalar, offset):
        self.tik_instance.set_atomic_add(self.atomic_num)
        with self.tik_instance.for_range(0, self.loop_batch) as loop_idx:
            move_offset.set_as(ng_idx * self.elem_num + loop_idx * self.group_batch * self.hw_num)
            with self.tik_instance.if_scope(loop_idx != self.loop_batch - 1):
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.group_batch * self.hw_num, need_conv=True)
                offset.set_as(g_idx * self.group_c + self.group_batch * loop_idx)
                self.data_move(scale_ub, self.scale_gm[offset], num=self.group_batch, need_conv=True)
                self.data_move(offset_ub, self.offset_gm[offset], num=self.group_batch, need_conv=True)

                self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.ub_n * self.c0)
                self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.ub_n * self.c0)
                self.data_mul(loop_ub, loop_ub, scale_ub, [0, 0, 0], num=self.group_batch)
                self.data_add(loop_ub, loop_ub, offset_ub, [0, 0, 0], num=self.group_batch)
                self.back_value(loop_ub, self.back_loop, self.loop_c0 * self.c0, 0)
                self.data_move(self.output_gm[move_offset], loop_ub, num=self.group_batch, need_conv=True, out=True)

            with self.tik_instance.else_scope():
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.last_batch * self.hw_num,
                               need_conv=True)
                offset.set_as(g_idx * self.group_c + self.group_batch * loop_idx)
                self.data_move(scale_ub, self.scale_gm[offset], num=self.last_batch, need_conv=True)
                self.data_move(offset_ub, self.offset_gm[offset], num=self.last_batch, need_conv=True)
                self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.ub_n * self.c0)
                self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.ub_n * self.c0)
                self.data_mul(loop_ub, loop_ub, scale_ub, [0, 0, 0], num=self.last_batch)
                self.data_add(loop_ub, loop_ub, offset_ub, [0, 0, 0], num=self.last_batch)
                if self.is_fp16:
                    self.data_conv(self.tmp_ub, loop_ub, [0, 0], mode="", num=self.last_batch,
                                   dst_stride=4, src_stride=8)
                    self.back_value(self.tmp_ub, self.back_last, self.last_c0 * self.c0, 0)
                    self.select_data_move(self.output_gm[move_offset], self.tmp_ub, self.last_batch, out=True)
                else:
                    self.back_value(loop_ub, self.back_last, self.last_c0 * self.c0, 0)
                    self.select_data_move(self.output_gm[move_offset], loop_ub, self.last_batch, out=True)
        self.tik_instance.set_atomic_add(0)

    def mul_add_nd(self, loop_ub, scale_scalar, offset_scalar, num):
        """
        mul and add
        """
        self.data_muls(loop_ub, loop_ub, scale_scalar, [0, 0], num)
        self.data_adds(loop_ub, loop_ub, offset_scalar, [0, 0], num=num)


def check_params(x, scale, offset):
    """
    check params of GroupNorm
    """
    dtype_x = x.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")

    if dtype_x != dtype_scale or dtype_x != dtype_offset:
        raise RuntimeError("dtype of x, scale, offset must be same")

    if dtype_x not in ("float16", "float32"):
        raise RuntimeError("only support float16 and float32")


def ceil_div(int1, int2):
    """
    ceil for (int1 / int2)
    :param int1: Scalar variable or an immediate
    :param int2: Scalar variable or an immediate
    :return: ceil for (int1 / int2)
    """
    return (int1 + int2 - 1) // int2


# 'pylint: disable=unused-argument,too-many-locals
@register_operator("GroupNorm")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def group_norm(x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-4, is_training=False,
               kernel_name="group_norm"):
    """
    :param x: input_data, support ND and 5HD of float16 or float32
    :param scale: scale_factor
    :param offset: offset_factor
    :param y: The result of GroupNorm
    :param mean: mean of x
    :param variance: variance of x
    :param num_groups: number of groups
    :param data_format: data_format, default to NCHW
    :param epsilon: epsilon avoid divided by zero, default to 1e-4
    :param is_training: is_training
    :param kernel_name: kernel_name, default to group_norm
    :return: instance
    """
    check_params(x, scale, offset)
    format_x = x.get("format")
    if format_x == "NC1HWC0":
        instance = GroupNorm5HD(x, scale, offset, y, mean, variance, num_groups, data_format, epsilon, is_training,
                                kernel_name)
    else:
        instance = GroupNormND(x, scale, offset, y, mean, variance, num_groups, data_format, epsilon, is_training,
                               kernel_name)
    return instance.compute()
