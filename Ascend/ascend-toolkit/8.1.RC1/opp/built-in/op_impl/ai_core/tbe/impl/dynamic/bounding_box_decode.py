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
dynamic bounding_box_decode
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl import constant_util as constant


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    NUMBER_FOUR = 4
    # the number of blocks skipped per repeat
    STRIDE_EIGHT = 8
    # the number of blocks skipped per repeat
    STRIDE_FOUR = 4
    # the number of blocks skipped per repeat
    STRIDE_ONE = 1
    # the number of blocks per transposition
    LIST_NUMBER = 16
    # the number of transposes per repeat
    NUMBER_TWO = 2
    # max int64
    MAX_INT64 = 2 ** 63 - 1
    # tiling param num
    TILING_ARG_NUM = 32
    TILING_SCALAR_DTYPE = "int64"
    RESERVED_UB = 20480
    NUMBER_SIXTEEN = 16


# 'pylint: disable=useless-object-inheritance,too-many-instance-attributes
class BoundingBoxDecode(object):

    # 'pylint: disable=too-many-arguments,invalid-name
    def __init__(self, rois, deltas,
                 means, stds, max_shape, wh_ratio_clip,
                 kernel_name):
        """
        Init BoundingBoxDecode base parameters

        Parameters
        ----------
        rois : dict
            shape and dtype of input rois
        deltas : dict
            shape and dtype of input deltas
        means : list
            the result of the calculation is normalized, default is [0,0,0,0]
        stds : list
            the result of the calculation is normalized, default is [1,1,1,1]
        max_shape : list or tuple
            max_shape of bboxes, default is None
        wh_ratio_clip : scalar
            limit the size of deltas[:,4] and deltas[:,3] between negative
            wh_ratio_clip and positive wh_ratio_clip, default is 0.016
        kernel_name : str
            kernel name, default value is "bounding_box_decode"

        Returns
        -------
        None
        """
        byte_size = 8
        block_number_fp16 = 32
        repeat_number_max = 128
        self.tik_instance = tik.Tik()
        self.rois_dtype = rois.get("dtype").lower()
        self.deltas_dtype = deltas.get("dtype").lower()
        self.kernel_name = kernel_name
        self.rois_dtype_bytes_size = tbe_platform.get_bit_len(self.rois_dtype) // byte_size
        self.rois_data_each_block = constant.BLOCK_SIZE // self.rois_dtype_bytes_size
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.each_repeat_block_number = block_number_fp16
        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.available_ub_size = (self.total_ub - Constant.RESERVED_UB) // self.rois_dtype_bytes_size // 118 \
                                 // repeat_number_max * repeat_number_max

        # init gm data
        self.rois_gm = self.tik_instance.Tensor(self.rois_dtype, [Constant.MAX_INT64],
                                                name="rois_gm", scope=tik.scope_gm)
        self.deltas_gm = self.tik_instance.Tensor(self.rois_dtype, [Constant.MAX_INT64],
                                                  name="deltas_gm", scope=tik.scope_gm)
        self.temp_gm = self.tik_instance.Tensor(self.rois_dtype, [Constant.MAX_INT64],
                                                name="temp_gm", scope=tik.scope_gm, is_workspace=True)
        self.bboxes_out_gm = self.tik_instance.Tensor(self.rois_dtype, [Constant.MAX_INT64],
                                                      name="bboxes_out_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        # init tiling data
        self.box_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="box_num")
        self.core_data = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="core_data")
        self.core_used = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="core_used")
        self.copy_loop = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="copy_loop")
        self.copy_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="copy_tail")
        self.last_copy_loop = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="last_copy_loop")
        self.last_copy_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="last_copy_tail")
        self.max_shape = self.tik_instance.ScalarArray(dtype=self.rois_dtype, length=2)
        self.means = self.tik_instance.ScalarArray(dtype=self.rois_dtype, length=4)
        self.stds = self.tik_instance.ScalarArray(dtype=self.rois_dtype, length=4)
        self.wh_ratio_clip = self.tik_instance.Scalar(self.rois_dtype, name="wh_ratio_clip")
        self.core_num_var = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="core_name_var", 
                                                     init_value=self.core_num)

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    def _tiling_args(self, tiling_ub):
        """
        get runtime tiling params from tiling

        Parameters
        ----------

        Returns
        -------
        None
        """
        # read tiling int32 scalar
        tiling_ub_int64 = tiling_ub.reinterpret_cast_to("int64")
        self.box_num.set_as(tiling_ub_int64[0])
        self.core_data.set_as(tiling_ub_int64[1])
        self.core_used.set_as(tiling_ub_int64[2])
        self.copy_loop.set_as(tiling_ub_int64[3])
        self.copy_tail.set_as(tiling_ub_int64[4])
        self.last_copy_loop.set_as(tiling_ub_int64[5])
        self.last_copy_tail.set_as(tiling_ub_int64[6])
        self.set_running_core_num(tiling_ub_int64[7])
        conv_scalar = self.tik_instance.Scalar("float16", name="conv_scalar")
        temp_scalar = self.tik_instance.Scalar("float32", name="temp_scalar")
        temp_scalar_INT32 = self.tik_instance.Scalar("int32", name="temp_scalar_INT32")
        with self.tik_instance.if_scope(self.rois_dtype == "float16"):
            temp_scalar_INT32.set_as(tiling_ub[18])
            temp_scalar_INT32.set_as(temp_scalar_INT32 - 1)
            self.tik_instance.scalar_conv('', temp_scalar, temp_scalar_INT32)
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.max_shape[0].set_as(conv_scalar)
            temp_scalar_INT32.set_as(tiling_ub[19])
            temp_scalar_INT32.set_as(temp_scalar_INT32 - 1)
            self.tik_instance.scalar_conv('', temp_scalar, temp_scalar_INT32)
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.max_shape[1].set_as(conv_scalar)

            temp_scalar.set_as(tiling_ub[20])
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.means[0].set_as(conv_scalar)
            temp_scalar.set_as(tiling_ub[21])
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.means[1].set_as(conv_scalar)
            temp_scalar.set_as(tiling_ub[22])
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.means[2].set_as(conv_scalar)
            temp_scalar.set_as(tiling_ub[23])
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.means[3].set_as(conv_scalar)

            temp_scalar.set_as(tiling_ub[24])
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.stds[0].set_as(conv_scalar)
            temp_scalar.set_as(tiling_ub[25])
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.stds[1].set_as(conv_scalar)
            temp_scalar.set_as(tiling_ub[26])
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.stds[2].set_as(conv_scalar)
            temp_scalar.set_as(tiling_ub[27])
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.stds[3].set_as(conv_scalar)
            
            temp_scalar.set_as(tiling_ub[28])
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.wh_ratio_clip.set_as(conv_scalar)
        with self.tik_instance.else_scope():
            temp_scalar_INT32.set_as(tiling_ub[18])
            temp_scalar_INT32.set_as(temp_scalar_INT32 - 1)
            self.tik_instance.scalar_conv('', temp_scalar, temp_scalar_INT32)
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.max_shape[0].set_as(conv_scalar)
            temp_scalar_INT32.set_as(tiling_ub[19])
            temp_scalar_INT32.set_as(temp_scalar_INT32 - 1)
            self.tik_instance.scalar_conv('', temp_scalar, temp_scalar_INT32)
            self.tik_instance.scalar_conv('', conv_scalar, temp_scalar)
            self.max_shape[1].set_as(conv_scalar)
            self.means.set_as([tiling_ub[20], tiling_ub[21], tiling_ub[22], tiling_ub[23]])
            self.stds.set_as([tiling_ub[24], tiling_ub[25], tiling_ub[26], tiling_ub[27]])
            self.wh_ratio_clip.set_as(tiling_ub[28])

    def init_cmpmask(self):
        zero_ub2 = \
            self.tik_instance.Tensor("float16",
                                     (128,),
                                     name="zero_ub2",
                                     scope=tik.scope_ubuf)
        one_ub = \
            self.tik_instance.Tensor("float16",
                                     (128,),
                                     name="one_ub",
                                     scope=tik.scope_ubuf)
        one_ub2 = \
            self.tik_instance.Tensor("float16",
                                     (128,),
                                     name="one_ub2",
                                     scope=tik.scope_ubuf)
        is_le = self.tik_instance.Tensor("uint16", (16,), name="is_le", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(128, zero_ub2, 0.0, 1, 1, 8)
        self.tik_instance.vector_dup(128, one_ub, 1.0, 1, 1, 8)
        self.tik_instance.vector_dup(128, one_ub2, 0.0, 1, 1, 8)
        # 12297829382473034410 = Binary representation：64 bit [10101010....101010]
        self.tik_instance.vec_add([12297829382473034410, 12297829382473034410],
                                  one_ub2, zero_ub2, one_ub, 1, 8, 8, 8)
        self.tik_instance.vec_cmpv_le(is_le, one_ub2, zero_ub2, 1, 8, 8)
        # mask is_le is used to get means_ub_front:[means[0],means[1],means[0],means[1],......]
        # mask is_le is used to get means_ub_tail:[means[2],means[3],means[2],means[3],......]
        # mask is_le is used to get stds_ub_front:[stds[0],stds[1],stds[0],stds[1],......]
        # mask is_le is used to get stds_ub_tail:[stds[2],stds[3],stds[2],stds[3],......]
        # mask is_le is used to get max_shape_ub:[max_shape[1],max_shape[0],max_shape[1],max_shape[0],......]
        return is_le

    def init_cmpmask2(self):
        zero_ub3 = \
            self.tik_instance.Tensor("float16",
                                     (128,),
                                     name="zero_ub3",
                                     scope=tik.scope_ubuf)
        one_ub4 = \
            self.tik_instance.Tensor("float16",
                                     (128,),
                                     name="one_ub4",
                                     scope=tik.scope_ubuf)
        one_ub5 = \
            self.tik_instance.Tensor("float16",
                                     (128,),
                                     name="one_ub5",
                                     scope=tik.scope_ubuf)
        is_le2 = self.tik_instance.Tensor("uint16", (16,), name="is_le", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(128, zero_ub3, 0.0, 1, 1, 8)
        self.tik_instance.vector_dup(128, one_ub4, 1.0, 1, 1, 8)
        self.tik_instance.vector_dup(128, one_ub5, 0.0, 1, 1, 8)
        # 14757395258967641292 = Binary representation：64 bit [11001100.....11001100]
        self.tik_instance.vec_add([14757395258967641292, 14757395258967641292], one_ub5, zero_ub3,
                                  one_ub4, 1, 8, 8, 8)
        self.tik_instance.vec_cmpv_le(is_le2, one_ub5, zero_ub3, 1, 8, 8)
        # mask is_le2 is used to get final result:
        # [rois_ub_front[index],rois_ub_front[index],rois_ub_tail[index],rois_ub_tail[index],......]
        return is_le2

    def set_attribute_tensor(self, means, stds, repeat_times, masknum, cmpmask):
        means_ub_front = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="means_ub_front",
                                     scope=tik.scope_ubuf)
        means_ub_front2 = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="means_ub_front2",
                                     scope=tik.scope_ubuf)
        means_ub_tail = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="means_ub_tail",
                                     scope=tik.scope_ubuf)
        means_ub_tail2 = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="means_ub_tail2",
                                     scope=tik.scope_ubuf)
        stds_ub_front = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="stds_ub_front",
                                     scope=tik.scope_ubuf)
        stds_ub_front2 = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="stds_ub_front2",
                                     scope=tik.scope_ubuf)
        stds_ub_tail = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="stds_ub_tail",
                                     scope=tik.scope_ubuf)
        stds_ub_tail2 = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="stds_ub_tail2",
                                     scope=tik.scope_ubuf)
        # set means value
        self.tik_instance.vec_dup(masknum, means_ub_front, means[0], repeat_times, 8)
        self.tik_instance.vec_dup(masknum, means_ub_front2, means[1], repeat_times, 8)
        self.tik_instance.vec_sel(masknum, 0, means_ub_front, cmpmask, means_ub_front,
                                  means_ub_front2, repeat_times, 8, 8, 8)

        self.tik_instance.vec_dup(masknum, means_ub_tail, means[2], repeat_times, 8)
        self.tik_instance.vec_dup(masknum, means_ub_tail2, means[3], repeat_times, 8)
        self.tik_instance.vec_sel(masknum, 0, means_ub_tail, cmpmask, means_ub_tail,
                                  means_ub_tail2, repeat_times, 8, 8, 8)

        # set stds value
        self.tik_instance.vec_dup(masknum, stds_ub_front, stds[0], repeat_times, 8)
        self.tik_instance.vec_dup(masknum, stds_ub_front2, stds[1], repeat_times, 8)
        self.tik_instance.vec_sel(masknum, 0, stds_ub_front, cmpmask, stds_ub_front,
                                  stds_ub_front2, repeat_times, 8, 8, 8)

        self.tik_instance.vec_dup(masknum, stds_ub_tail, stds[2], repeat_times, 8)
        self.tik_instance.vec_dup(masknum, stds_ub_tail2, stds[3], repeat_times, 8)
        self.tik_instance.vec_sel(masknum, 0, stds_ub_tail, cmpmask, stds_ub_tail,
                                  stds_ub_tail2, repeat_times, 8, 8, 8)

        # set max_ratio
        max_ratio_ub = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="max_ratio_ub",
                                     scope=tik.scope_ubuf)
        max_ratio_ub2 = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="max_ratio_ub2",
                                     scope=tik.scope_ubuf)

        self.tik_instance.vector_dup(masknum, max_ratio_ub,
                                     self.wh_ratio_clip, repeat_times,
                                     Constant.STRIDE_ONE, Constant.STRIDE_EIGHT)
        self.tik_instance.vln(masknum, max_ratio_ub,
                              max_ratio_ub, repeat_times, Constant.STRIDE_ONE,
                              Constant.STRIDE_ONE, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)
        self.tik_instance.vabs(masknum, max_ratio_ub,
                               max_ratio_ub, repeat_times, Constant.STRIDE_ONE,
                               Constant.STRIDE_ONE, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

        self.tik_instance.vmuls(
            masknum, max_ratio_ub2, max_ratio_ub, -1.0,
            repeat_times, Constant.STRIDE_ONE, Constant.STRIDE_ONE, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

        return [means_ub_front, means_ub_tail, stds_ub_front, stds_ub_tail, max_ratio_ub, max_ratio_ub2]

    # 'pylint: disable=too-many-arguments
    def calculate_denorm_delta(self, repeat_times, deltas_ub_front, deltas_ub_tail,
                               means_ub_front, means_ub_tail, stds_ub_front, stds_ub_tail, masknum):
        """
        calculate denorm_delta using formula:
        dx = delta[..., 0]*target_stds[0] + target_means[0]
        dy = delta[..., 1]*target_stds[1] + target_means[1]
        dw = delta[..., 2]*target_stds[2] + target_means[2]
        dh = delta[..., 3]*target_stds[3] + target_means[3]

        Parameters
        ----------
        deltas_ub_front_vconv: the ub tensor with deltas
        deltas_ub_tail_vconv: the temporary ub tensor for calculation
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_detal_dst_ub: the ub tensor with denorm_delta
        """

        self.tik_instance.vec_mul(
            masknum, deltas_ub_front, deltas_ub_front,
            stds_ub_front, repeat_times, 8, 8, 8)
        self.tik_instance.vec_add(
            masknum, deltas_ub_front, deltas_ub_front,
            means_ub_front, repeat_times, 8, 8, 8)

        self.tik_instance.vec_mul(
            masknum, deltas_ub_tail, deltas_ub_tail,
            stds_ub_tail, repeat_times, 8, 8, 8)
        self.tik_instance.vec_add(
            masknum, deltas_ub_tail, deltas_ub_tail,
            means_ub_tail, repeat_times, 8, 8, 8)

        return deltas_ub_front, deltas_ub_tail

    def clamp_denorm_detal(self, repeat_times, deltas_ub_tail, max_ratio_ub, max_ratio_ub2, masknum):
        """
        clamp denorm_delta using formula:
        max_ratio = abs(log(max_ratio))
        dw = -max_ratio<= dw <= max_ratio
        dh = -max_ratio<= dh <= max_ratio

        Parameters
        ----------
        deltas_ub_tail_vconv: the ub tensor with denorm_delta
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_detal_dst_ub: the ub tensor with denorm_delta
        """

        self.tik_instance.vec_min(masknum, deltas_ub_tail,
                                  deltas_ub_tail, max_ratio_ub,
                                  repeat_times, 8, 8, 8)

        self.tik_instance.vec_max(masknum, deltas_ub_tail,
                                  deltas_ub_tail, max_ratio_ub2,
                                  repeat_times, 8, 8, 8)

        return deltas_ub_tail

    def calculate_denorm_rois(self, repeat_times, rois_ub_front, rois_ub_tail, temp_ub_front, temp_ub_tail, masknum):
        """
        calculate denorm_rois using formula:
        ax = (rois[..., 2] + rois[..., 0])*0.5
        ay = (rois[..., 3] + rois[..., 1])*0.5
        aw = rois[..., 2] - rois[..., 0] + 1
        ah = rois[..., 3] - rois[..., 1] + 1

        Parameters
        ----------
        rois_src_ub: the ub tensor with rois
        rois_dst_ub: the temporary ub tensor for calculation
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        """

        # calculate denorm_rois_dst_ub == (px, py, pw, ph) ==>(px, py)
        self.tik_instance.vec_add(
            masknum, temp_ub_front, rois_ub_tail, rois_ub_front,
            repeat_times, 8, 8, 8)
        self.tik_instance.vec_muls(
            masknum, temp_ub_front, temp_ub_front, 0.5,
            repeat_times, 8, 8)

        # calculate denorm_rois_dst_ub == (px, py, pw, ph) ==>(pw, ph)
        self.tik_instance.vec_sub(
            masknum, temp_ub_tail, rois_ub_tail, rois_ub_front,
            repeat_times, 8, 8, 8)
        self.tik_instance.vec_adds(
            masknum, temp_ub_tail, temp_ub_tail, 1.0,
            repeat_times, 8, 8)

        return temp_ub_front, temp_ub_tail

    def addcmul_demorm_rois(self, repeat_times, rois_ub_front, rois_ub_tail,
                            temp_ub_front, temp_ub_tail, deltas_ub_front, deltas_ub_tail, masknum):
        """
        addcmul denorm_rois using formula:
        px = dx * aw + ax
        py = dy * ah + ay
        pw = exp(dw)*aw
        ph = exp(dh)*ah

        Parameters
        ----------
        denorm_detal_dst_ub: the ub tensor with denorm_rois
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_detal_dst_ub: the ub tensor with denorm_rois
        """
        # calculate denorm_rois_dst_ub == (px, py, pw, ph) ==>px, py
        self.tik_instance.vec_mul(
            masknum, rois_ub_front, temp_ub_tail,
            deltas_ub_front, repeat_times, 8, 8, 8)
        self.tik_instance.vec_add(
            masknum, rois_ub_front, rois_ub_front,
            temp_ub_front, repeat_times, 8, 8, 8)

        # calculate denorm_rois_dst_ub == (px, py, pw, ph) ==>pw, ph
        self.tik_instance.vec_exp(
            masknum, rois_ub_tail, deltas_ub_tail,
            repeat_times, 8, 8)
        self.tik_instance.vec_mul(
            masknum, rois_ub_tail, rois_ub_tail,
            temp_ub_tail, repeat_times, 8, 8, 8)

        return rois_ub_front, rois_ub_tail

    def calculate_result(self, repeat_times, rois_ub_front, rois_ub_tail,
                         temp_ub_front, temp_ub_tail, masknum):
        """
        calculate the result using formula:
        x1 = px - pw*0.5 + 0.5
        y1 = py - ph*0.5 + 0.5
        x2 = px + pw*0.5 - 0.5
        y2 = py + ph*0.5 - 0.5

        Parameters
        ----------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        denorm_detal_dst_ub: the ub tensor with denorm_detal
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        """
        self.tik_instance.vec_muls(
            masknum, rois_ub_tail, rois_ub_tail,
            0.5, repeat_times, 8, 8)

        # calculate (x1, y1, x2, y2) ==>x1, y1
        self.tik_instance.vec_sub(
            masknum, temp_ub_front, rois_ub_front,
            rois_ub_tail, repeat_times, 8, 8, 8)
        self.tik_instance.vec_adds(
            masknum, temp_ub_front, temp_ub_front, 0.5,
            repeat_times, 8, 8)

        # calculate (x1, y1, x2, y2) ==>x2, y2
        self.tik_instance.vec_add(
            masknum, temp_ub_tail, rois_ub_front,
            rois_ub_tail, repeat_times, 8, 8, 8)
        self.tik_instance.vec_adds(
            masknum, temp_ub_tail, temp_ub_tail,
            -0.5, repeat_times, 8, 8)

        return temp_ub_front, temp_ub_tail

    def clamp_result(self, repeat_times, rois_ub_front, rois_ub_tail, temp_ub_front, temp_ub_tail,
                     max_shape_ub_1, masknum, cmpmask):
        """
        clamp the result using formula if max_shape is not none:
        x1 = 0 <= x1 <= max_shape[1] - 1
        y1 = 0 <= y1 <= max_shape[0] - 1
        x2 = 0 <= x2 <= max_shape[1] - 1
        y2 = 0 <= y2 <= max_shape[0] - 1

        Parameters
        ----------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        """
        if self.max_shape is not None:
            zero_ub = \
                self.tik_instance.Tensor(self.rois_dtype,
                                         (self.available_ub_size, 4),
                                         name="zero_ub",
                                         scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(masknum, zero_ub, 0.0,
                                         repeat_times, Constant.STRIDE_ONE, Constant.STRIDE_EIGHT)
            self.tik_instance.vec_min(masknum, rois_ub_front,
                                      temp_ub_front, max_shape_ub_1,
                                      repeat_times, 8, 8, 8)
            self.tik_instance.vec_min(masknum, rois_ub_tail,
                                      temp_ub_tail, max_shape_ub_1,
                                      repeat_times, 8, 8, 8)
            self.tik_instance.vec_max(masknum, rois_ub_front,
                                      rois_ub_front, zero_ub,
                                      repeat_times, 8, 8, 8)
            self.tik_instance.vec_max(masknum, rois_ub_tail,
                                      rois_ub_tail, zero_ub,
                                      repeat_times, 8, 8, 8)

        return rois_ub_front, rois_ub_tail

    def data_move_method(self, dst, src, burst, burst_pad):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(dst, src, constant.DEFAULT_NBURST, burst_pad, 
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO,
                                            right_padding=0, left_padding=0, padding_value=None)
        else:
            self.tik_instance.data_move(dst, src, constant.SID, constant.DEFAULT_NBURST, burst, 
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            
    def data_move_mte2_function(self, loop_input, burst, burst_pad):
        """
        move data of rois/deltas from gm to ub with each pinpang of each core

        Parameters
        ----------
        loop_input: the starting address of the gm data
        burst: the burst of each repeat

        Returns
        -------
        rois_src_ub: the ub tensor with rois
        deltas_src_ub: the ub tensor with deltas
        """
        rois_ub_front = self.tik_instance.Tensor(
            self.rois_dtype, (self.available_ub_size, 4),
            name="rois_ub_front",
            scope=tik.scope_ubuf)
        rois_ub_tail = self.tik_instance.Tensor(
            self.rois_dtype, (self.available_ub_size, 4),
            name="rois_ub_tail",
            scope=tik.scope_ubuf)
        self.data_move_method(rois_ub_front, self.rois_gm[loop_input],
                              burst, burst_pad)
        self.data_move_method(rois_ub_tail, self.rois_gm[loop_input + 2],
                              burst, burst_pad - 2 * self.rois_dtype_bytes_size)

        deltas_ub_front = self.tik_instance.Tensor(
            self.deltas_dtype, (self.available_ub_size, 4),
            name="deltas_ub_front",
            scope=tik.scope_ubuf)
        deltas_ub_tail = self.tik_instance.Tensor(
            self.deltas_dtype, (self.available_ub_size, 4),
            name="deltas_ub_tail",
            scope=tik.scope_ubuf)
        self.data_move_method(deltas_ub_front, self.deltas_gm[loop_input],
                              burst, burst_pad)
        self.data_move_method(deltas_ub_tail, self.deltas_gm[loop_input + 2],
                              burst, burst_pad - 2 * self.rois_dtype_bytes_size)

        return [rois_ub_front, rois_ub_tail, deltas_ub_front, deltas_ub_tail]

    def data_move_mte3_function(self, loop_input, burst,
                                denorm_rois_dst_ub):
        """
        move output data of bboxes from gm to ub with each pinpang of each core

        Parameters
        ----------
        loop_input: the starting address of the gm data
        burst: the burst of each repeat
        denorm_rois_dst_ub: the ub tensor of output data of bboxes

        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.bboxes_out_gm[loop_input],
                                    denorm_rois_dst_ub, constant.SID,
                                    constant.DEFAULT_NBURST, burst,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def max_shape_init(self, masknum, repeat_times, cmpmask, max_shape_ub_1):
        max_shape_ub_2 = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="max_shape_ub_2",
                                     scope=tik.scope_ubuf)

        self.tik_instance.vector_dup(masknum, max_shape_ub_1,
                                     self.max_shape[1], repeat_times,
                                     Constant.STRIDE_ONE, Constant.STRIDE_EIGHT)
        self.tik_instance.vector_dup(masknum, max_shape_ub_2,
                                     self.max_shape[0], repeat_times,
                                     Constant.STRIDE_ONE, Constant.STRIDE_EIGHT)
        self.tik_instance.vec_sel(masknum, 0, max_shape_ub_1, cmpmask, max_shape_ub_1, max_shape_ub_2, repeat_times,
                                  8, 8, 8)
        return max_shape_ub_1
    
    # 'pylint: disable=too-many-locals
    def bounding_box_decode_compute(self, repeat_times, loop_input, rois_ub_front, rois_ub_tail,
                                    deltas_ub_front, deltas_ub_tail, means_ub_front, means_ub_tail,
                                    stds_ub_front, stds_ub_tail, max_ratio_ub, max_ratio_ub2,
                                    max_shape_ub_1, masknum, cmpmask, cmpmask2, burst):
        """
        describe the bounding_box_decode calculation process

        Parameters
        ----------
        repeat_times: the vector calculation repeat times
        loop_input: the loop number in each repeat times
        rois_ub_front,rois_ub_tail: the ub tensor with rois
        deltas_ub_front,deltas_ub_tail: the ub tensor with deltas
        means_ub_front, means_ub_tail: the ub tensor with means
        stds_ub_front, stds_ub_tail: the ub tensor with stds
        max_ratio_ub, max_ratio_ub2: the ub tensor with max_ratio
        max_shape_ub_1:
        masknum:
        cmpmask, cmpmask2:
        burst:
        Returns
        -------
        denorm_rois_dst_ub: the ub tensor for output data of bboxes

        """
        temp_ub_front = \
            self.tik_instance.Tensor(self.rois_dtype, (self.available_ub_size, 4),
                                     name="temp_ub_front",
                                     scope=tik.scope_ubuf)
        temp_ub_tail = \
            self.tik_instance.Tensor(self.rois_dtype, (self.available_ub_size, 4),
                                     name="temp_ub_tail",
                                     scope=tik.scope_ubuf)

        deltas_ub_front, deltas_ub_tail = self.calculate_denorm_delta(
            repeat_times, deltas_ub_front, deltas_ub_tail,
            means_ub_front, means_ub_tail, stds_ub_front, stds_ub_tail, masknum)
        deltas_ub_tail = self.clamp_denorm_detal(repeat_times, deltas_ub_tail,
                                                 max_ratio_ub, max_ratio_ub2, masknum)

        temp_ub_front, temp_ub_tail = self.calculate_denorm_rois(
            repeat_times, rois_ub_front, rois_ub_tail, temp_ub_front, temp_ub_tail, masknum)

        rois_ub_front, rois_ub_tail = self.addcmul_demorm_rois(repeat_times, rois_ub_front, rois_ub_tail,
                                                               temp_ub_front, temp_ub_tail, deltas_ub_front,
                                                               deltas_ub_tail, masknum)

        temp_ub_front, temp_ub_tail = self.calculate_result(
            repeat_times, rois_ub_front, rois_ub_tail, temp_ub_front, temp_ub_tail, masknum)
        rois_ub_front, rois_ub_tail = \
            self.clamp_result(repeat_times, rois_ub_front, rois_ub_tail, temp_ub_front, temp_ub_tail,
                              max_shape_ub_1, masknum, cmpmask)

        self.tik_instance.data_move(self.temp_gm[loop_input + 2],
                                    rois_ub_tail, constant.SID,
                                    constant.DEFAULT_NBURST, burst,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        self.tik_instance.data_move(rois_ub_tail,
                                    self.temp_gm[loop_input], constant.SID,
                                    constant.DEFAULT_NBURST, burst,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        self.tik_instance.vec_sel(masknum, 0, rois_ub_front, cmpmask2, rois_ub_front, rois_ub_tail, repeat_times,
                                  8, 8, 8)
        return rois_ub_front

    def calculation_process(self, loop_input, repeat_times, means_ub_front, means_ub_tail,
                            stds_ub_front, stds_ub_tail, max_ratio_ub, max_ratio_ub2, max_shape_ub_1,
                            masknum, cmpmask, cmpmasl2, input_param_size):
        """
        decide whether to enable pingpang according to different loop_cycle of
        the core

        Parameters
        ----------
        block_id: identifies the number of cores

        Returns
        -------
        None
        """

        burst = (input_param_size * 4 + self.rois_data_each_block - 1) // self.rois_data_each_block
        burst_pad = input_param_size * 4 * self.rois_dtype_bytes_size 

        rois_ub_front, rois_ub_tail, deltas_ub_front, deltas_ub_tail = \
            self.data_move_mte2_function(loop_input, burst, burst_pad)
        rois_ub_front = \
            self.bounding_box_decode_compute(repeat_times, loop_input, rois_ub_front, rois_ub_tail,
                                             deltas_ub_front, deltas_ub_tail, means_ub_front, means_ub_tail,
                                             stds_ub_front, stds_ub_tail, max_ratio_ub, max_ratio_ub2,
                                             max_shape_ub_1, masknum, cmpmask, cmpmasl2, burst)
        self.data_move_mte3_function(loop_input, burst, rois_ub_front)

    def copy_only(self, core_index, loop_num, tail_num):
        
        mask_64 = 64
        mask_128 = 128
        mask_num = mask_128 if self.rois_dtype == 'float16' else mask_64
        repeat_times = (self.available_ub_size * 4 + mask_num - 1) // mask_num
        cmpmask = self.init_cmpmask()
        cmpmask2 = self.init_cmpmask2()
        means_ub_front, means_ub_tail, stds_ub_front, stds_ub_tail, max_ratio_ub, max_ratio_ub2 = \
            self.set_attribute_tensor(self.means, self.stds, repeat_times, mask_num, cmpmask)
        max_shape_ub_1 = \
            self.tik_instance.Tensor(self.rois_dtype,
                                     (self.available_ub_size, 4),
                                     name="max_shape_ub_1",
                                     scope=tik.scope_ubuf)
        if self.max_shape is not None:
            max_shape_ub_1 = self.max_shape_init(mask_num, repeat_times, cmpmask, max_shape_ub_1)
        with self.tik_instance.for_range(0, loop_num, thread_num=2) as loop_idx:
            loop_input = core_index * self.core_data * 4 + loop_idx * self.available_ub_size * 4
            input_param_size = self.available_ub_size
            burst = (self.available_ub_size * 4 + self.rois_data_each_block - 1) // self.rois_data_each_block
            self.calculation_process(loop_input, repeat_times, means_ub_front,
                                     means_ub_tail, stds_ub_front, stds_ub_tail,
                                     max_ratio_ub, max_ratio_ub2, max_shape_ub_1,
                                     mask_num, cmpmask, cmpmask2, input_param_size)

        with self.tik_instance.if_scope(tail_num > 0):
            with self.tik_instance.if_scope((tail_num * 4) < mask_num):
                repeat_times2 = 1
                mask_num2 = tail_num * 4
            with self.tik_instance.else_scope():
                repeat_times2 = (tail_num * 4 + mask_num - 1) // mask_num
                mask_num2 = mask_num
            loop_input = core_index * self.core_data * 4 + loop_num * self.available_ub_size * 4
            input_param_size = tail_num
            self.calculation_process(loop_input, repeat_times2, means_ub_front,
                                     means_ub_tail, stds_ub_front, stds_ub_tail,
                                     max_ratio_ub, max_ratio_ub2, max_shape_ub_1,
                                     mask_num2, cmpmask, cmpmask2, input_param_size)

    def tik_instance_function(self):
        """
        the entry of bounding_box_decode calculation

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
            self._tiling_args(tiling_ub)

        # core process
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_index:
            with self.tik_instance.if_scope(core_index < (self.core_used - 1)):
                self.copy_only(core_index, self.copy_loop, self.copy_tail)
            with self.tik_instance.elif_scope(core_index == (self.core_used - 1)):
                self.copy_only(core_index, self.last_copy_loop, self.last_copy_tail)

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars",
                                                   {"core_num": self.core_num,
                                                    "rois_data_each_block": self.rois_data_each_block,
                                                    "each_repeat_block_number": self.each_repeat_block_number,
                                                    "ub_max_size": self.available_ub_size})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.rois_gm, self.deltas_gm],
                                   outputs=[self.bboxes_out_gm],
                                   flowtable=[self.tiling_gm], config=opt_config)
        return self.tik_instance


# 'pylint: disable=unused-argument, too-many-locals, too-many-lines
@register_operator("BoundingBoxDecode")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def bounding_box_decode(rois,
                        deltas,
                        bboxes,
                        means=(0.0, 0.0, 0.0, 0.0),
                        stds=(1.0, 1.0, 1.0, 1.0),
                        max_shape=None,
                        wh_ratio_clip=0.016,
                        kernel_name="bounding_box_decode"):
    """
    calculating data

    Parameters
    ----------
    rois : dict
        shape and dtype of input rois
    deltas : dict
        shape and dtype of input deltas
    bboxes : dict
        shape and dtype of output, should be same shape and type as input
    means : list
        the result of the calculation is normalized, default is [0,0,0,0]
    stds : list
        the result of the calculation is normalized, default is [1,1,1,1]
    max_shape : list or tuple
        max_shape of bboxes, default is None
    wh_ratio_clip : scalar
        limit the size of deltas[:,4] and deltas[:,3] between negative
        wh_ratio_clip and positive wh_ratio_clip, default is 0.016
    kernel_name : str
        kernel name, default value is "bounding_box_decode"

    Returns
    -------
    None
    """
    rois_dtype = rois.get("dtype").lower()
    deltas_dtype = deltas.get("dtype").lower()
    para_check.check_dtype(rois_dtype, ["float16", "float32"])
    para_check.check_dtype(deltas_dtype, ["float16", "float32"])
    if rois_dtype != deltas_dtype:
        error_detail = "dtype of rois and deltas should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "rois", "deltas", error_detail)

    bboxes_instance = BoundingBoxDecode(rois, deltas, means, stds, max_shape,
                                        wh_ratio_clip, kernel_name)
    instance = bboxes_instance.tik_instance_function()
    return instance
