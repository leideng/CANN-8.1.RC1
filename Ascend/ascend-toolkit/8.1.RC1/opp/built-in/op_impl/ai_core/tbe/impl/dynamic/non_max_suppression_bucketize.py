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
non_max_suppression_bucketize
"""
# 'pylint: disable=too-many-lines
import te.platform as tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tik
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    SCALAR_MIN_INT32 = -(2**31 - 1)
    INT32_DTYPE = "int32"
    INT64_DTYPE = "int64"
    MASK64 = 64
    MASK32 = 32
    MASK16 = 16
    MASK8 = 8
    DESC_SIZE = 128
    INT64_BYTE_SIZE = 8

    INDEX_0 = 0
    INDEX_1 = 1
    INDEX_2 = 2
    INDEX_3 = 3
    INDEX_4 = 4
    INDEX_5 = 5
    INDEX_6 = 6
    INDEX_7 = 7
    BLOCK_STRIDE = 1
    REPEAT_STRIDE = 8


# 'pylint: disable=unused-argument
# 'pylint: disable=invalid-name,too-many-arguments
def get_op_support_info(input_nmsed_boxes, input_nmsed_score, input_nmsed_class, input_nmsed_num,
                        output_nmsed_boxes, output_nmsed_score, output_nmsed_class,
                        kernel_name="non_max_suppression_bucketize"):
    """
    get_op_support_info
    """
    axis_split_list = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_list, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def prod(values):
    """
    Prod the input values by multiply.
    """
    res = 1
    for value in values:
        res *= value
    return res


def ceil_div(value, block):
    """
    integrate the input value by block.
    """
    return (value + block - 1) // block


def check_param(ori_boxes_shape, ori_score_shape, ori_class_shape, ori_num_shape,
                boxes_dtype, score_dtype, class_dtype, num_dtype):
    # boxes, score and class input's dtype must be fp16, num's dtype must be int32
    para_check.check_dtype(boxes_dtype, ("float16", ), param_name="ori_boxes")
    para_check.check_dtype(score_dtype, ("float16", ), param_name="ori_score")
    para_check.check_dtype(class_dtype, ("float16", ), param_name="ori_class")
    para_check.check_dtype(num_dtype, ("int32", ), param_name="ori_num")

    # check input shape
    para_check.check_shape(ori_boxes_shape, param_name="ori_boxes")
    para_check.check_shape(ori_score_shape, param_name="ori_score")
    para_check.check_shape(ori_class_shape, param_name="ori_class")
    para_check.check_shape(ori_num_shape, param_name="ori_num")

    def check_shape_len(param, shape_len, valid_len):
        if shape_len != valid_len:
            error_manager_vector.raise_error_input_param_range_invalid("non_max_suppression_bucketize", param,
                                                                       valid_len, valid_len, shape_len)

    boxes_dims = 3
    score_class_dims = 2
    check_shape_len("ori_boxes", len(ori_boxes_shape), boxes_dims)
    check_shape_len("ori_score", len(ori_score_shape), score_class_dims)
    check_shape_len("ori_class", len(ori_class_shape), score_class_dims)
    check_shape_len("ori_num", len(ori_num_shape), 1)

    boxes_last_dim = 4
    if ori_boxes_shape[-1] != boxes_last_dim:
        error_manager_vector.raise_err_check_params_rules("non_max_suppression_bucketize",
                                                          "the last dim must be 4",
                                                          "ori_boxes", ori_boxes_shape)

    if ori_score_shape != ori_class_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal("non_max_suppression_bucketize", "ori_score", "ori_class",
                                                              ori_score_shape, ori_class_shape, ori_class_shape)

    if ori_boxes_shape[0] != ori_score_shape[0] or ori_boxes_shape[1] != ori_score_shape[1]:
        error_manager_vector.raise_err_check_params_rules(
            "non_max_suppression_bucketize",
            "the first 2 dims must be the same, where ori_boxes_shape is %s" % ori_boxes_shape,
            "ori_score", ori_score_shape)

    if ori_boxes_shape[0] != ori_num_shape[0]:
        error_manager_vector.raise_err_check_params_rules(
            "non_max_suppression_bucketize",
            "the first dim must be the same, where ori_boxes_shape is %s" % ori_boxes_shape,
            "ori_num", ori_num_shape)


class NMSBucketize:
    """
    Function: use to store basic parameters
    """
    def __init__(self, shape_list, dtype, num_dtype, kernel_name):
        self.ori_boxes_shape = shape_list[Constant.INDEX_0]
        self.ori_score_shape = shape_list[Constant.INDEX_1]
        self.ori_class_shape = shape_list[Constant.INDEX_2]
        self.ori_num_shape = shape_list[Constant.INDEX_3]
        self.batch = self.ori_boxes_shape[Constant.INDEX_0]
        self.nms_num = self.ori_boxes_shape[Constant.INDEX_1]
        self.boxes_last_axis = self.ori_boxes_shape[Constant.INDEX_2]
        self.dtype = dtype
        self.num_dtype = num_dtype
        self.out_dtype = "float32"
        self.kernel_name = kernel_name
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        block_byte_size = 32
        self.fp16_data_size = 2
        self.fp32_data_size = 4
        self.data_each_block_int32 = 8
        self.data_each_block_fp16 = block_byte_size // self.fp16_data_size
        self.data_each_block_fp32 = block_byte_size // self.fp32_data_size
        self.data_each_vector = 64
        self.segment = 2048 * 7
        self.burst_len = Constant.DESC_SIZE * Constant.INT64_BYTE_SIZE // block_byte_size
        self.max_result_ub = None
        self.out_boxes_desc_ub = None
        self.out_score_desc_ub = None
        self.out_class_desc_ub = None
        self.out_boxes_addr_ub = None
        self.out_score_addr_ub = None
        self.out_class_addr_ub = None
        self.max_nmsed_num = self.tik_instance.Scalar(dtype=self.num_dtype, name="max_nmsed_num",
                                                      init_value=Constant.SCALAR_MIN_INT32)
        self.boxes_ub_loop = self.tik_instance.Scalar(dtype=Constant.INT32_DTYPE,
                                                      name="boxes_ub_loop", init_value=0)
        self.boxes_ub_remain = self.tik_instance.Scalar(dtype=Constant.INT32_DTYPE,
                                                        name="boxes_ub_remain")
        self.boxes_data_size = self.tik_instance.Scalar(dtype=Constant.INT32_DTYPE,
                                                        name="boxes_data_size")
        self.score_class_ub_loop = self.tik_instance.Scalar(dtype=Constant.INT32_DTYPE,
                                                            name="score_class_ub_loop", init_value=0)
        self.score_class_ub_remain = self.tik_instance.Scalar(dtype=Constant.INT32_DTYPE,
                                                              name="score_class_ub_remain")
        self.score_class_data_size = self.tik_instance.Scalar(dtype=Constant.INT32_DTYPE,
                                                              name="score_class_data_size")

        self.ori_boxes_gm = None
        self.ori_score_gm = None
        self.ori_class_gm = None
        self.ori_num_gm = None
        self.out_boxes_gm = None
        self.out_score_gm = None
        self.out_class_gm = None

    def compute(self):
        """
        compute
        """
        self._set_src_dst_tensor()
        self._get_tensor_addr()

        # find max nmsed_num
        with self.tik_instance.new_stmt_scope():
            self._reduce_max()

        # max nmsed_num up to 32B align
        self.max_nmsed_num.set_as(ceil_div(self.max_nmsed_num, self.data_each_block_fp32) * self.data_each_block_fp32)

        # `return original nmsed_num if max_nmsed_num > ori_nmsed_num`
        with self.tik_instance.if_scope(self.max_nmsed_num > self.nms_num):
            self.max_nmsed_num.set_as(self.nms_num)

        # `return 8 if max_nmsed_num == 0`
        with self.tik_instance.if_scope(self.max_nmsed_num == 0):
            self.max_nmsed_num.set_as(self.data_each_block_fp32)

        # set output shape
        self._set_tensor_desc_shape(self.ori_boxes_shape, self.out_boxes_desc_ub)
        self._set_tensor_desc_shape(self.ori_score_shape, self.out_score_desc_ub)
        self._set_tensor_desc_shape(self.ori_class_shape, self.out_class_desc_ub)

        # slice boxes, score and class data
        with self.tik_instance.new_stmt_scope():
            self._slice_data()

        # move tensor desc from ub to gm
        self.tik_instance.data_move(self.out_boxes_gm, self.out_boxes_desc_ub, 0, 1, self.burst_len, 0, 0)
        self.tik_instance.data_move(self.out_score_gm, self.out_score_desc_ub, 0, 1, self.burst_len, 0, 0)
        self.tik_instance.data_move(self.out_class_gm, self.out_class_desc_ub, 0, 1, self.burst_len, 0, 0)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.ori_boxes_gm, self.ori_score_gm,
                                           self.ori_class_gm, self.ori_num_gm),
                                   outputs=(self.out_boxes_gm, self.out_score_gm,
                                            self.out_class_gm))
        return self.tik_instance

    def _set_src_dst_tensor(self):
        """
        set input and output gm tensor
        """
        self.ori_boxes_gm = self.tik_instance.Tensor(self.dtype, self.ori_boxes_shape,
                                                     name="ori_boxes_gm", scope=tik.scope_gm)
        self.ori_score_gm = self.tik_instance.Tensor(self.dtype, self.ori_score_shape,
                                                     name="ori_score_gm", scope=tik.scope_gm)
        self.ori_class_gm = self.tik_instance.Tensor(self.dtype, self.ori_class_shape,
                                                     name="ori_class_gm", scope=tik.scope_gm)
        self.ori_num_gm = self.tik_instance.Tensor(self.num_dtype, self.ori_num_shape,
                                                   name="ori_num_gm", scope=tik.scope_gm)
        self.out_boxes_gm = self.tik_instance.Tensor(Constant.INT64_DTYPE, (Constant.DESC_SIZE,),
                                                     name="out_boxes_gm", scope=tik.scope_gm)
        self.out_score_gm = self.tik_instance.Tensor(Constant.INT64_DTYPE, (Constant.DESC_SIZE,),
                                                     name="out_score_gm", scope=tik.scope_gm)
        self.out_class_gm = self.tik_instance.Tensor(Constant.INT64_DTYPE, (Constant.DESC_SIZE,),
                                                     name="out_class_gm", scope=tik.scope_gm)

    def _get_tensor_addr(self):
        """
        get input and output desc tensor and data address
        """
        # output desc
        self.out_boxes_desc_ub = self.tik_instance.Tensor(Constant.INT64_DTYPE, (Constant.DESC_SIZE,),
                                                          name="out_boxes_desc_ub", scope=tik.scope_ubuf)
        self.out_score_desc_ub = self.tik_instance.Tensor(Constant.INT64_DTYPE, (Constant.DESC_SIZE,),
                                                          name="out_score_desc_ub", scope=tik.scope_ubuf)
        self.out_class_desc_ub = self.tik_instance.Tensor(Constant.INT64_DTYPE, (Constant.DESC_SIZE,),
                                                          name="out_class_desc_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.out_boxes_desc_ub, self.out_boxes_gm, 0, 1, self.burst_len, 0, 0)
        self.tik_instance.data_move(self.out_score_desc_ub, self.out_score_gm, 0, 1, self.burst_len, 0, 0)
        self.tik_instance.data_move(self.out_class_desc_ub, self.out_class_gm, 0, 1, self.burst_len, 0, 0)

        # get output tensor data address
        self.out_boxes_addr_ub = self.tik_instance.TensorAddrList(1, name="out_boxes_addr_ub",
                                                                  scope=tik.scope_ubuf)
        self.out_score_addr_ub = self.tik_instance.TensorAddrList(1, name="out_score_addr_ub",
                                                                  scope=tik.scope_ubuf)
        self.out_class_addr_ub = self.tik_instance.TensorAddrList(1, name="out_class_addr_ub",
                                                                  scope=tik.scope_ubuf)
        self.out_boxes_addr_ub[0].set_as(self.out_boxes_desc_ub[0])
        self.out_score_addr_ub[0].set_as(self.out_score_desc_ub[0])
        self.out_class_addr_ub[0].set_as(self.out_class_desc_ub[0])

    def _reduce_max(self):
        """
        compute reduce max of nmsed_num
        """
        buf_size, loop_times, over_size, align_flag = self._get_reduce_max_tiling_info()
        max_result_size = 64
        self.max_result_ub = self.tik_instance.Tensor(self.num_dtype, (max_result_size,),
                                                      name="max_result_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.data_each_vector, self.max_result_ub,
                                     Constant.SCALAR_MIN_INT32, 1, 1, 8)

        if loop_times != 0:
            thread_num = 1
            # 2 indicates for double buffer
            if loop_times > 2:
                thread_num = 2
            with self.tik_instance.for_range(0, loop_times, thread_num=thread_num) as loop:
                self._do_max_last_axis(buf_size, loop)
        if align_flag:
            self._do_max_last_axis(over_size, loop_times)

        self._get_one_from_64()
        self.max_nmsed_num.set_as(self.max_result_ub[0])

    # 'pylint: disable=too-many-return-values
    def _get_reduce_max_tiling_info(self):
        """
        get_tiling_info of reduce max operation
        """
        align_flag = ((self.batch % self.segment) != 0)
        if self.segment <= self.batch:
            buf_size = self.segment
            loop_times = self.batch // self.segment
            over_size = self.batch - (loop_times * self.segment)
        else:
            loop_times = 0
            buf_size = self.batch
            over_size = buf_size
        return buf_size, loop_times, over_size, align_flag

    def _do_max_last_axis(self, ub_buf_size, loop):
        """
        compute vmax operation
        """
        nmsed_num_ub_size = [self.data_each_block_int32 * ceil_div(self.segment, self.data_each_block_int32)]
        nmsed_num_ub = self.tik_instance.Tensor(self.num_dtype, nmsed_num_ub_size,
                                                name="nmsed_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(nmsed_num_ub, self.ori_num_gm[loop * self.segment], 0, 1,
                                    ceil_div(ub_buf_size, self.data_each_block_int32), 0, 0)
        tail = ub_buf_size % self.data_each_vector
        if tail != 0:
            max_mask_int64 = 2**64 - 1
            mask_h = 0
            mask = 2**tail - 1
            mask_l = max_mask_int64 - mask
            _offset = ub_buf_size // self.data_each_vector
            self.tik_instance.vector_dup([mask_h, mask_l], nmsed_num_ub[_offset * self.data_each_vector],
                                         Constant.SCALAR_MIN_INT32, 1, Constant.BLOCK_STRIDE, Constant.REPEAT_STRIDE)
        total_block = ceil_div(ub_buf_size, self.data_each_vector)
        while total_block > 1:
            total_block = self._get_half_max(nmsed_num_ub, total_block)

        self.tik_instance.vmax(self.data_each_vector, self.max_result_ub, nmsed_num_ub,
                               self.max_result_ub, 1, 1, 1, 1, 0, Constant.REPEAT_STRIDE, 0)

    def _get_half_max(self, vmax_ub, vector_num):
        """
        get_half_max
        """
        if vector_num == 1:
            return 1
        # 2 indicates half of output
        output_num = vector_num // 2
        ub2_offset = output_num * Constant.MASK64

        self.tik_instance.vmax(Constant.MASK64, vmax_ub, vmax_ub, vmax_ub[ub2_offset], output_num,
                               Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                               Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)

        # 2 indicates for remain calculation
        if vector_num % 2 == 1:
            self.tik_instance.vmax(Constant.MASK64, vmax_ub, vmax_ub, vmax_ub[ub2_offset * 2], 1,
                                   Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                                   Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
        return output_num

    def _get_one_from_64(self):
        """
        get_one_from_64
        """
        process_num = 8
        ub_block_0 = self.max_result_ub[process_num * Constant.INDEX_0]
        ub_block_1 = self.max_result_ub[process_num * Constant.INDEX_1]
        ub_block_2 = self.max_result_ub[process_num * Constant.INDEX_2]
        ub_block_3 = self.max_result_ub[process_num * Constant.INDEX_3]
        ub_block_4 = self.max_result_ub[process_num * Constant.INDEX_4]
        ub_block_5 = self.max_result_ub[process_num * Constant.INDEX_5]
        ub_block_6 = self.max_result_ub[process_num * Constant.INDEX_6]
        ub_block_7 = self.max_result_ub[process_num * Constant.INDEX_7]
        self.tik_instance.vmax(Constant.MASK32, ub_block_0, ub_block_0, ub_block_4, 1,
                               Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                               Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
        self.tik_instance.vmax(Constant.MASK16, ub_block_4, ub_block_0, ub_block_2, 1,
                               Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                               Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
        self.tik_instance.vmax(Constant.MASK8, ub_block_0, ub_block_4, ub_block_5, 1,
                               Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                               Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)
        index_reg = [self.tik_instance.Scalar(dtype=self.num_dtype) for _ in range(process_num)]
        for i in range(process_num - 1):
            index_reg[i + 1].set_as(self.max_result_ub[i + 1])
        ub_block_1.set_as(index_reg[Constant.INDEX_1])
        ub_block_2.set_as(index_reg[Constant.INDEX_2])
        ub_block_3.set_as(index_reg[Constant.INDEX_3])
        ub_block_4.set_as(index_reg[Constant.INDEX_4])
        ub_block_5.set_as(index_reg[Constant.INDEX_5])
        ub_block_6.set_as(index_reg[Constant.INDEX_6])
        ub_block_7.set_as(index_reg[Constant.INDEX_7])
        self._get_max_last(ub_block_0, ub_block_0, ub_block_1)
        self._get_max_last(ub_block_2, ub_block_2, ub_block_3)
        self._get_max_last(ub_block_4, ub_block_4, ub_block_5)
        self._get_max_last(ub_block_6, ub_block_6, ub_block_7)
        self._get_max_last(ub_block_0, ub_block_0, ub_block_2)
        self._get_max_last(ub_block_4, ub_block_6, ub_block_4)
        self._get_max_last(ub_block_0, ub_block_0, ub_block_4)

    def _get_max_last(self, dst, src0, src1):
        self.tik_instance.vmax(1, dst, src0, src1, 1,
                               Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE, Constant.BLOCK_STRIDE,
                               Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE, Constant.REPEAT_STRIDE)

    def _set_tensor_desc_shape(self, shape, desc_ub):
        shape_idx = 3
        batch_scalar = self.tik_instance.Scalar(dtype=Constant.INT64_DTYPE, init_value=self.batch)
        dim_num = self.tik_instance.Scalar(dtype=Constant.INT64_DTYPE, init_value=len(shape))
        nmsed_num = self.tik_instance.Scalar(dtype=Constant.INT64_DTYPE, init_value=self.max_nmsed_num)
        desc_ub[shape_idx].set_as(dim_num)
        desc_ub[shape_idx + 1].set_as(batch_scalar)  # 1 indicates first dim of shape
        desc_ub[shape_idx + 2].set_as(nmsed_num)  # 2 indicates second dim of shape
        boxes_dims = 3
        if len(shape) == boxes_dims:
            last_dim = self.tik_instance.Scalar(dtype=Constant.INT64_DTYPE, init_value=self.boxes_last_axis)
            desc_ub[shape_idx + 3].set_as(last_dim)  # 3 indicates third dim of shape

    def _slice_data(self):
        """
        compute slice
        """
        loop_num = self.tik_instance.Scalar(dtype=Constant.INT32_DTYPE, name="loop_num", init_value=1)
        actual_batch = self.tik_instance.Scalar(dtype=Constant.INT32_DTYPE, name="actual_batch")
        is_multi_core = self.tik_instance.Scalar(dtype=Constant.INT32_DTYPE, name="is_multi_core")
        with self.tik_instance.if_scope(tik.any(self.max_nmsed_num < self.data_each_block_fp32, self.batch == 1)):
            is_multi_core.set_as(0)
            actual_batch.set_as(1)
            self._get_slice_tiling_info(self.batch)
        with self.tik_instance.else_scope():
            is_multi_core.set_as(1)
            actual_batch.set_as(self.batch)
            loop_num.set_as(ceil_div(self.batch, self.core_num))
            self._get_slice_tiling_info(1)

        nmsed_fp16_ub = self.tik_instance.Tensor(self.dtype, (self.boxes_data_size,),
                                                 name="nmsed_fp16_ub", scope=tik.scope_ubuf)
        nmsed_fp32_ub = self.tik_instance.Tensor(self.out_dtype, (self.boxes_data_size,),
                                                 name="nmsed_fp32_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as outer_loop:
            with self.tik_instance.for_range(0, loop_num) as inner_loop:
                block_idx = outer_loop * loop_num + inner_loop
                with self.tik_instance.if_scope(block_idx < actual_batch):
                    # compute boxes
                    self._data_move(nmsed_fp16_ub, nmsed_fp32_ub, block_idx, self.boxes_ub_loop,
                                    self.boxes_ub_remain, self.boxes_data_size, self.ori_boxes_gm,
                                    self.out_boxes_addr_ub, prod(self.ori_boxes_shape[1:]),
                                    self.max_nmsed_num * self.boxes_last_axis, is_multi_core)
                    # compute score
                    self._data_move(nmsed_fp16_ub, nmsed_fp32_ub, block_idx, self.score_class_ub_loop,
                                    self.score_class_ub_remain, self.score_class_data_size,
                                    self.ori_score_gm, self.out_score_addr_ub,
                                    self.ori_score_shape[-1], self.max_nmsed_num, is_multi_core)
                    # compute class
                    self._data_move(nmsed_fp16_ub, nmsed_fp32_ub, block_idx, self.score_class_ub_loop,
                                    self.score_class_ub_remain, self.score_class_data_size,
                                    self.ori_class_gm, self.out_class_addr_ub,
                                    self.ori_class_shape[-1], self.max_nmsed_num, is_multi_core)

    def _get_slice_tiling_info(self, batch):
        # 3 indicates number of desc_ub and addr_ub
        used_ub_byte = (Constant.DESC_SIZE * 3 + 3) * Constant.INT64_BYTE_SIZE
        max_ub_size = (self.ub_size - used_ub_byte) // (self.fp16_data_size + self.fp32_data_size) // \
                      self.data_each_block_fp16 * self.data_each_block_fp16

        boxes_size = batch * self.max_nmsed_num * self.boxes_last_axis
        score_class_size = batch * self.max_nmsed_num
        with self.tik_instance.if_scope(boxes_size > max_ub_size):
            self.boxes_data_size.set_as(max_ub_size)
            self.boxes_ub_loop.set_as(boxes_size // self.boxes_data_size)
            self.boxes_ub_remain.set_as(boxes_size % self.boxes_data_size)
            with self.tik_instance.if_scope(score_class_size > max_ub_size):
                self.score_class_data_size.set_as(max_ub_size)
                self.score_class_ub_loop.set_as(score_class_size // self.score_class_data_size)
                self.score_class_ub_remain.set_as(score_class_size % self.score_class_data_size)
            with self.tik_instance.else_scope():
                self.score_class_data_size.set_as(score_class_size)
                self.score_class_ub_remain.set_as(score_class_size)
        with self.tik_instance.else_scope():
            self.boxes_data_size.set_as(boxes_size)
            self.score_class_data_size.set_as(score_class_size)
            self.boxes_ub_remain.set_as(boxes_size)
            self.score_class_ub_remain.set_as(score_class_size)

    def _data_move(self, input_fp16_ub, input_fp32_ub, block_idx, ub_loop, ub_remain, ub_size, ori_gm, out_gm,
                   ori_size, out_size, is_multi_core):
        """
        do slice of boxes, score and class
        """
        with self.tik_instance.for_range(0, ub_loop) as loop_idx:
            in_burst = ub_size // self.data_each_block_fp16
            out_burst = ub_size // self.data_each_block_fp32
            ori_gm_offset = block_idx * ori_size + loop_idx * ub_size
            out_gm_offset = block_idx * out_size + loop_idx * ub_size
            self.tik_instance.data_move(input_fp16_ub, ori_gm[ori_gm_offset], 0, 1, in_burst, 0, 0)
            self._vconv(input_fp16_ub, 0, input_fp32_ub, 0, ub_size)
            self.tik_instance.data_move(out_gm[0].value + out_gm_offset, input_fp32_ub, 0, 1, out_burst, 0, 0)
        with self.tik_instance.if_scope(ub_remain > 0):
            in_burst = ceil_div(ub_remain, self.data_each_block_fp16)
            ori_gm_offset = block_idx * ori_size + ub_loop * ub_size
            out_gm_offset = block_idx * out_size + ub_loop * ub_size
            self.tik_instance.data_move(input_fp16_ub, ori_gm[ori_gm_offset], 0, 1, in_burst, 0, 0)
            self._vconv(input_fp16_ub, 0, input_fp32_ub, 0, ub_remain)
            with self.tik_instance.if_scope(is_multi_core == 0):
                out_burst = ceil_div(ub_remain, self.data_each_block_fp32)
                self.tik_instance.data_move(out_gm[0].value + out_gm_offset, input_fp32_ub, 0, 1, out_burst, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(out_gm[0].value + out_gm_offset, input_fp32_ub, 0, 1,
                                            ub_remain // self.data_each_block_fp32, 0, 0)
                with self.tik_instance.if_scope(ub_remain % self.data_each_block_fp32 != 0):
                    for i in range(self.data_each_block_fp32):
                        input_fp32_ub[i] = input_fp32_ub[ub_remain - self.data_each_block_fp32 + i]
                    out_gm_offset += ub_remain - self.data_each_block_fp32
                    self.tik_instance.data_move(out_gm[0].value + out_gm_offset, input_fp32_ub, 0, 1, 1, 0, 0)

    def _vconv(self, src, src_start, dst, dst_start, ele_num):
        MAX_VECTOR_REPEAT_TIME = 255
        VECTOR_FP32_SIZE = 64
        total_repeat_time = self.tik_instance.Scalar(Constant.INT32_DTYPE)
        remain_ele = self.tik_instance.Scalar(Constant.INT32_DTYPE)
        total_repeat_time.set_as(ele_num // VECTOR_FP32_SIZE)
        remain_ele.set_as(ele_num % VECTOR_FP32_SIZE)
        mask = VECTOR_FP32_SIZE

        repeat_max_time = self.tik_instance.Scalar(Constant.INT32_DTYPE)
        remain_repeat_time = self.tik_instance.Scalar(Constant.INT32_DTYPE)
        repeat_max_time.set_as(total_repeat_time // MAX_VECTOR_REPEAT_TIME)
        remain_repeat_time.set_as(total_repeat_time % MAX_VECTOR_REPEAT_TIME)

        src_stride, dst_stride = 4, 8
        with self.tik_instance.if_scope(repeat_max_time > 0):
            with self.tik_instance.for_range(0, repeat_max_time) as loop1:
                self.tik_instance.vconv(mask, "",
                                        dst[dst_start + loop1 * MAX_VECTOR_REPEAT_TIME * mask],
                                        src[src_start + loop1 * MAX_VECTOR_REPEAT_TIME * mask],
                                        MAX_VECTOR_REPEAT_TIME, 1, 1, dst_stride, src_stride)
        with self.tik_instance.if_scope(remain_repeat_time > 0):
            self.tik_instance.vconv(mask, "",
                                    dst[dst_start + repeat_max_time * MAX_VECTOR_REPEAT_TIME * mask],
                                    src[src_start + repeat_max_time * MAX_VECTOR_REPEAT_TIME * mask],
                                    remain_repeat_time, 1, 1, dst_stride, src_stride)
        with self.tik_instance.if_scope(remain_ele > 0):
            self.tik_instance.vconv(
                remain_ele, "",
                dst[dst_start + repeat_max_time * MAX_VECTOR_REPEAT_TIME * mask + remain_repeat_time * mask],
                src[src_start + repeat_max_time * MAX_VECTOR_REPEAT_TIME * mask + remain_repeat_time * mask],
                1, 1, 1, dst_stride, src_stride)


# 'pylint: disable=unused-argument
@register_operator("NonMaxSuppressionBucketize")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def non_max_suppression_bucketize(input_nmsed_boxes, input_nmsed_score, input_nmsed_class, input_nmsed_num,
                                  output_nmsed_boxes, output_nmsed_score, output_nmsed_class,
                                  kernel_name="non_max_suppression_bucketize"):
    """
    main function of non_max_suppression_bucketize

    Parameters
    ----------
    input_nmsed_boxes: dict
        shape and data type of batch_multi_class_non_max_suppression's 1st output
    input_nmsed_score: dict
        shape and data type of batch_multi_class_non_max_suppression's 2nd output
    input_nmsed_class: dict
        shape and data type of batch_multi_class_non_max_suppression's 3rd output
    input_nmsed_num: dict
        shape and data type of batch_multi_class_non_max_suppression's 4th output
    output_nmsed_boxes: dict
        shape of sliced input_nmsed_boxes, data type must be fp32
    output_nmsed_score: dict
        shape of sliced input_nmsed_score, data type must be fp32
    output_nmsed_class: dict
        shape of sliced input_nmsed_class, data type must be fp32
    kernel_name: str
        cce kernel name, default value is "non_max_suppression_bucketize"
    """
    ori_boxes_shape = list(input_nmsed_boxes.get("shape"))
    ori_score_shape = list(input_nmsed_score.get("shape"))
    ori_class_shape = list(input_nmsed_class.get("shape"))
    ori_num_shape = list(input_nmsed_num.get("shape"))
    boxes_dtype = input_nmsed_boxes.get("dtype")
    score_dtype = input_nmsed_score.get("dtype")
    class_dtype = input_nmsed_class.get("dtype")
    num_dtype = input_nmsed_num.get("dtype")
    check_param(ori_boxes_shape, ori_score_shape, ori_class_shape, ori_num_shape,
                boxes_dtype, score_dtype, class_dtype, num_dtype)

    shape_list = [ori_boxes_shape, ori_score_shape, ori_class_shape, ori_num_shape]
    result = NMSBucketize(shape_list, boxes_dtype, num_dtype, kernel_name)
    return result.compute()
