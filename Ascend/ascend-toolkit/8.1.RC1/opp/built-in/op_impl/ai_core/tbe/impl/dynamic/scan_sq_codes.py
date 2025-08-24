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
scan_sq_codes
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util import util_common


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-few-public-methods
class ScanSQCodes():
    """
    ScanSQCodes class
    """
    MAX_INT32 = 2 ** 31 - 1
    MAX_INT64 = 2 ** 63 - 1
    MAX_FP16 = 65504
    MIN_FP16 = -65504
    MAX_NPROBE = 1024
    TILING_ARG_NUM = 8
    MASK_FP16 = 128
    MASK_INT32 = 64
    BLOCK_INT64 = 4
    BLOCK_INT32 = 8
    BLOCK_FP16 = 16
    BLOCK_U8 = 32
    SLICE_SIZE = 1024

    def __init__(self, args_list, total_limit, group_size, extreme_mode):
        """
        args_list is (ivf, query, bucket_list, bucket_limits, bucket_offsets, vmin, vdiff)
        """
        self.tik_inst = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        (ivf, query, bucket_list, bucket_limits, bucket_offsets, vmin, vdiff) = args_list
        self.group_size = group_size
        self.total_limit = util_common.align(total_limit, self.SLICE_SIZE)
        self.extreme_mode = extreme_mode
        self.groups = self.total_limit // self.group_size

        self.tiling_align = util_common.align(self.TILING_ARG_NUM, self.BLOCK_INT64)
        self.tiling_dtype = "int64"
        self.ivf_dtype = ivf.get("dtype").lower()
        self.query_dtype = query.get("dtype").lower()
        self.bucket_list_dtype = bucket_list.get("dtype").lower()
        self.bucket_limits_dtype = bucket_limits.get("dtype").lower()
        self.bucket_offsets_dtype = bucket_offsets.get("dtype").lower()
        self.vmin_dtype = vmin.get("dtype").lower()
        self.vdiff_dtype = vdiff.get("dtype").lower()
        query_shape = query.get("shape")
        self.d = query_shape[0]

        # input gm
        self.ivf_gm = None
        self.query_gm = None
        self.bucket_list_gm = None
        self.bucket_limits_gm = None
        self.bucket_offsets_gm = None
        self.vmin_gm = None
        self.vdiff_gm = None
        self.tiling_gm = None
        # output gm
        self.actual_count_gm = None
        self.sq_distance_gm = None
        self.grouped_extreme_distance_gm = None
        self.sq_ivf_gm = None
        self.sq_index_gm = None
        self._init_gm_tensor()

        # tiling params
        self.bucket_num_total = None
        self.bucket_idx_offset = None
        self.need_core_num = None
        self.low_core_num = None
        self.bucket_num_low = None
        self.core_num_var = None

    def scan_sq_codes_compute(self):
        """
        scan_sq_codes compute
        """
        # get tiling data
        self._get_tiling_args()
        with self.tik_inst.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_id:
            with self.tik_inst.if_scope(core_id < self.need_core_num):
                with self.tik_inst.if_scope(core_id < self.low_core_num):
                    with self.tik_inst.new_stmt_scope():
                        start_idx = self.bucket_num_low * core_id
                        self._one_core_compute(core_id, start_idx, self.bucket_num_low)
                with self.tik_inst.else_scope():
                    with self.tik_inst.new_stmt_scope():
                        bucket_num = self.bucket_num_low - 1
                        start_idx = self.bucket_num_low * self.low_core_num + \
                                    bucket_num * (core_id - self.low_core_num)
                        self._one_core_compute(core_id, start_idx, bucket_num)

    def get_inputs_outputs_gm(self):
        inputs_gm = (self.ivf_gm, self.query_gm, self.bucket_list_gm, self.bucket_limits_gm, self.bucket_offsets_gm,
                     self.vmin_gm, self.vdiff_gm)
        outputs_gm = (self.actual_count_gm, self.sq_distance_gm, self.grouped_extreme_distance_gm,
                      self.sq_ivf_gm, self.sq_index_gm)

        return inputs_gm, outputs_gm

    def _init_gm_tensor(self):
        """
        init gm tensor
        """
        self.ivf_gm = self.tik_inst.Tensor(self.ivf_dtype, (self.MAX_INT64,), name="ivf_gm", scope=tik.scope_gm)
        self.query_gm = self.tik_inst.Tensor(self.query_dtype, (self.d,), name="query_gm", scope=tik.scope_gm)
        self.bucket_list_gm = self.tik_inst.Tensor(self.bucket_list_dtype, (self.MAX_INT32,),
                                                   name="bucket_list_gm", scope=tik.scope_gm)
        self.bucket_limits_gm = self.tik_inst.Tensor(self.bucket_limits_dtype, (self.MAX_INT32,),
                                                     name="bucket_limits_gm", scope=tik.scope_gm)
        self.bucket_offsets_gm = self.tik_inst.Tensor(self.bucket_offsets_dtype, (self.MAX_INT32,),
                                                      name="bucket_offsets_gm", scope=tik.scope_gm)
        self.vmin_gm = self.tik_inst.Tensor(self.vmin_dtype, (self.d,), name="vmin_gm", scope=tik.scope_gm)
        self.vdiff_gm = self.tik_inst.Tensor(self.vdiff_dtype, (self.d,), name="vdiff_gm", scope=tik.scope_gm)

        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_gm",
                                              scope=tik.scope_gm)

        self.actual_count_gm = self.tik_inst.Tensor(self.bucket_list_dtype, (self.BLOCK_INT32,),
                                                    name="actual_count_gm", scope=tik.scope_gm)
        self.sq_distance_gm = self.tik_inst.Tensor(self.query_dtype, (self.total_limit,), name="sq_distance_gm",
                                                   scope=tik.scope_gm)
        self.grouped_extreme_distance_gm = self.tik_inst.Tensor(
            self.query_dtype, (self.groups,), name="grouped_extreme_distance_gm", scope=tik.scope_gm)
        self.sq_ivf_gm = self.tik_inst.Tensor(self.bucket_list_dtype, (self.total_limit,),
                                              name="sq_ivf_gm", scope=tik.scope_gm)
        self.sq_index_gm = self.tik_inst.Tensor(self.bucket_list_dtype, (self.total_limit,),
                                                name="sq_index_gm", scope=tik.scope_gm)

    def _init_ub_tensor(self, bucket_idx_start, bucket_num):
        """
        init ub tensor
        """
        query_ub = self.tik_inst.Tensor(self.query_dtype, (self.MASK_FP16,), name="query_ub", scope=tik.scope_ubuf)
        vmin_ub = self.tik_inst.Tensor(self.vmin_dtype, (self.MASK_FP16,), name="vmin_ub", scope=tik.scope_ubuf)
        vdiff_ub = self.tik_inst.Tensor(self.vdiff_dtype, (self.MASK_FP16,), name="vdiff_ub", scope=tik.scope_ubuf)
        bucket_list_ub = self.tik_inst.Tensor(self.bucket_list_dtype, (self.MAX_NPROBE,),
                                              name="bucket_list_ub", scope=tik.scope_ubuf)
        bucket_limits_ub = self.tik_inst.Tensor(self.bucket_limits_dtype, (self.MAX_NPROBE,),
                                                name="bucket_limits_ub", scope=tik.scope_ubuf)
        bucket_offsets_ub = self.tik_inst.Tensor(self.bucket_offsets_dtype, (self.MAX_NPROBE,),
                                                 name="bucket_offsets_ub", scope=tik.scope_ubuf)

        self.tik_inst.data_move(bucket_list_ub, self.bucket_list_gm[self.bucket_idx_offset + bucket_idx_start],
                                0, 1, util_common.ceil(bucket_num, self.BLOCK_INT32), 0, 0)
        self.tik_inst.data_move(bucket_limits_ub,
                                self.bucket_limits_gm[self.bucket_idx_offset + bucket_idx_start],
                                0, 1, util_common.ceil(bucket_num, self.BLOCK_INT32), 0, 0)
        self.tik_inst.data_move(bucket_offsets_ub,
                                self.bucket_offsets_gm[self.bucket_idx_offset + bucket_idx_start],
                                0, 1, util_common.ceil(bucket_num, self.BLOCK_INT64), 0, 0)

        blocks = self.d // self.BLOCK_FP16
        repeat = self.MASK_FP16 // self.d - 1
        self.tik_inst.data_move(query_ub, self.query_gm, 0, 1, blocks, 0, 0)
        self.tik_inst.data_move(vmin_ub, self.vmin_gm, 0, 1, blocks, 0, 0)
        self.tik_inst.data_move(vdiff_ub, self.vdiff_gm, 0, 1, blocks, 0, 0)
        self.tik_inst.vmuls(self.d, query_ub[self.d], query_ub, 1.0, repeat, 1, 1, blocks, 0)
        self.tik_inst.vmuls(self.d, vmin_ub[self.d], vmin_ub, 1.0, repeat, 1, 1, blocks, 0)
        self.tik_inst.vmuls(self.d, vdiff_ub[self.d], vdiff_ub, 1.0, repeat, 1, 1, blocks, 0)

        return [query_ub, bucket_list_ub, bucket_limits_ub, bucket_offsets_ub, vmin_ub, vdiff_ub]

    def _init_assist_ub(self):
        """
        assist_idx_ub, shape is [1024], value is 0,1,2,...,1023
        """
        assist_idx_ub = self.tik_inst.Tensor(self.bucket_list_dtype, (self.SLICE_SIZE,),
                                             name="assist_idx_ub", scope=tik.scope_ubuf)
        for i in range(8):
            assist_idx_ub[i].set_as(i)
        self.tik_inst.vadds(8, assist_idx_ub[8], assist_idx_ub, 8, 1, 1, 1, 8, 8)
        self.tik_inst.vadds(16, assist_idx_ub[16], assist_idx_ub, 16, 1, 1, 1, 8, 8)
        self.tik_inst.vadds(32, assist_idx_ub[32], assist_idx_ub, 32, 1, 1, 1, 8, 8)
        self.tik_inst.vadds(64, assist_idx_ub[64], assist_idx_ub, 64, 1, 1, 1, 8, 8)

        with self.tik_inst.new_stmt_scope(disable_sync=True):
            for i in range(1, 8):
                self.tik_inst.vadds(64, assist_idx_ub[128 * i], assist_idx_ub, 128 * i, 2, 1, 1, 8, 8)

        return assist_idx_ub

    def _get_tiling_args(self):
        """
        get runtime params from tiling data
        """
        self.bucket_num_total = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="bucket_num_total")
        self.bucket_idx_offset = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="bucket_idx_offset")
        self.need_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.low_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="low_core_num")
        self.bucket_num_low = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="bucket_num_low")
        self.core_num_var = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="core_num_var")

        with self.tik_inst.new_stmt_scope():
            tiling_ub = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
            self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // self.BLOCK_INT64, 0, 0)

            self.bucket_num_total.set_as(tiling_ub[0])
            self.bucket_idx_offset.set_as(tiling_ub[1])
            self.need_core_num.set_as(tiling_ub[2])
            self.low_core_num.set_as(tiling_ub[3])
            self.bucket_num_low.set_as(tiling_ub[4])
            self.core_num_var.set_as(tiling_ub[5])

    def _output_offset_compute(self, core_id):
        """
        calculate output offset for every bucket, and output actual_count by core 0 only
        """
        output_offset_ub = self.tik_inst.Tensor(self.bucket_limits_dtype, (self.MAX_NPROBE + self.BLOCK_INT32,),
                                                name="output_offset_ub", scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            actual_count_s = self.tik_inst.Scalar(dtype=self.bucket_limits_dtype, name="actual_count_s", init_value=0)
            bucket_limit_s = self.tik_inst.Scalar(dtype=self.bucket_limits_dtype, name="bucket_limit_s")
            bucket_limits_ub = self.tik_inst.Tensor(self.bucket_limits_dtype, (self.MAX_NPROBE,),
                                                    name="bucket_limits_ub_int32", scope=tik.scope_ubuf)
            self.tik_inst.data_move(bucket_limits_ub, self.bucket_limits_gm[self.bucket_idx_offset],
                                    0, 1, util_common.ceil(self.bucket_num_total, self.BLOCK_INT32), 0, 0)
            with self.tik_inst.for_range(0, self.bucket_num_total) as idx:
                output_offset_ub[idx].set_as(actual_count_s)
                bucket_limit_s.set_as(bucket_limits_ub[idx])
                actual_count_s.set_as(actual_count_s + util_common.align(bucket_limit_s, self.SLICE_SIZE))

            # output 0
            with self.tik_inst.if_scope(core_id == 0):
                output_offset_ub[self.MAX_NPROBE].set_as(actual_count_s)
                self.tik_inst.data_move(self.actual_count_gm, output_offset_ub[self.MAX_NPROBE], 0, 1, 1, 0, 0)

        return output_offset_ub

    def _cal_loop_params(self, elem_cnt):
        """
        calculate loop params
        """
        repeats = util_common.ceil(elem_cnt, self.MASK_FP16)
        max_repeat = 255
        loop = repeats // max_repeat
        tail_repeat = repeats % max_repeat
        one_loop_elems = max_repeat * self.MASK_FP16

        return [max_repeat, loop, tail_repeat, one_loop_elems]

    def _ivf_u8_vconv(self, data_ub, data_shape, elem_cnt, bucket_offset):
        """
        conv uint8 ivf to float16
        """
        with self.tik_inst.new_stmt_scope():
            ivf_u8_ub = self.tik_inst.Tensor(self.ivf_dtype, data_shape, name="ivf_u8_ub", scope=tik.scope_ubuf)
            self.tik_inst.data_move(ivf_u8_ub, self.ivf_gm[bucket_offset * self.d], 0, 1,
                                    elem_cnt // self.BLOCK_U8, 0, 0)

            max_repeat, loop, tail_repeat, one_loop_elems = self._cal_loop_params(elem_cnt)
            with self.tik_inst.for_range(0, loop) as idx:
                self.tik_inst.vconv(self.MASK_FP16, "", data_ub[one_loop_elems * idx],
                                    ivf_u8_ub[one_loop_elems * idx], max_repeat, 1, 1, 8, 4)
            with self.tik_inst.if_scope(tail_repeat > 0):
                self.tik_inst.vconv(self.MASK_FP16, "", data_ub[one_loop_elems * loop],
                                    ivf_u8_ub[one_loop_elems * loop], tail_repeat, 1, 1, 8, 4)

    def _data_decode_inner(self, ub_tensors, offset, repeat, rec_255_s):
        """
        decode data inner.
        ub_tensors is (vcgadd_res_ub, data_ub, query_ub, vmin_ub, vdiff_ub)
        """
        (vcgadd_res_ub, data_ub, query_ub, vmin_ub, vdiff_ub) = ub_tensors
        self.tik_inst.vmuls(self.MASK_FP16, data_ub[offset], data_ub[offset], rec_255_s,
                            repeat, 1, 1, 8, 8)
        self.tik_inst.vmadd(self.MASK_FP16, data_ub[offset], vdiff_ub, vmin_ub,
                            repeat, 1, 1, 1, 8, 0, 0)
        self.tik_inst.vsub(self.MASK_FP16, data_ub[offset], query_ub, data_ub[offset],
                           repeat, 1, 1, 1, 8, 0, 8)
        self.tik_inst.vmul(self.MASK_FP16, data_ub[offset], data_ub[offset], data_ub[offset],
                           repeat, 1, 1, 1, 8, 8, 8)

        self.tik_inst.vcgadd(self.MASK_FP16, vcgadd_res_ub[offset // self.BLOCK_FP16], data_ub[offset],
                             repeat, 1, 1, 8)

    def _data_decode(self, ub_tensors, elem_cnt, rec_255_s):
        """
        decode data for SQ8.
        ub_tensors is (vcgadd_res_ub, data_ub, query_ub, vmin_ub, vdiff_ub)
        """
        max_repeat, loop, tail_repeat, one_loop_elems = self._cal_loop_params(elem_cnt)

        with self.tik_inst.for_range(0, loop) as idx:
            self._data_decode_inner(ub_tensors, one_loop_elems * idx, max_repeat, rec_255_s)
        with self.tik_inst.if_scope(tail_repeat > 0):
            self._data_decode_inner(ub_tensors, one_loop_elems * loop, tail_repeat, rec_255_s)

    def _sq_distance_tail_dup(self, sq_distance_ub, tail, extreme_value_s):
        """
        fill the end of the sq_distance_ub with extreme value
        """
        block_pad = self.BLOCK_FP16 - tail % self.BLOCK_FP16
        with self.tik_inst.for_range(0, block_pad) as idx:
            sq_distance_ub[tail + idx].set_as(extreme_value_s)

        elems_after_block_pad = tail + block_pad
        elems_after_vector_pad = util_common.align(tail, self.MASK_FP16)
        vector_pad = elems_after_vector_pad - elems_after_block_pad
        with self.tik_inst.if_scope(vector_pad > 0):
            self.tik_inst.vector_dup(vector_pad, sq_distance_ub[elems_after_block_pad], extreme_value_s, 1, 1, 8)

        rest_of_elems = self.SLICE_SIZE - elems_after_vector_pad
        with self.tik_inst.if_scope(rest_of_elems > 0):
            self.tik_inst.vector_dup(self.MASK_FP16, sq_distance_ub[elems_after_vector_pad], extreme_value_s,
                                     rest_of_elems // self.MASK_FP16, 1, 8)

    def _bucket_inner_process(self, ub_tensors, sq_ivf_ub, rec_255_s, params):
        """
        process part of one bucket.
        ub_tensors is (query_ub, vmin_ub, vdiff_ub, assist_idx_ub)
        params is (extreme_value_s, bucket_offset_s, output_offset_base_s, thread_idx, tail, is_tail)
        """
        (query_ub, vmin_ub, vdiff_ub, assist_idx_ub) = ub_tensors
        extreme_value_s, bucket_offset_s, output_offset_base_s, thread_idx, tail, is_tail = params
        output_distance_offset = output_offset_base_s + self.SLICE_SIZE * thread_idx
        if not is_tail:
            elem_cnt = self.SLICE_SIZE * self.d
        else:
            elem_cnt = tail * self.d
        data_shape = (self.SLICE_SIZE * self.d,)
        data_ub = self.tik_inst.Tensor(self.query_dtype, data_shape, name="data_ub", scope=tik.scope_ubuf)
        self._ivf_u8_vconv(data_ub, data_shape, elem_cnt, bucket_offset_s + self.SLICE_SIZE * thread_idx)

        sq_distance_ub = self.tik_inst.Tensor(self.query_dtype, (self.SLICE_SIZE,), name="sq_distance_ub",
                                              scope=tik.scope_ubuf)
        vcgadd_res_shape = (self.SLICE_SIZE * self.d // self.BLOCK_FP16,)
        vcgadd_res_ub = self.tik_inst.Tensor(self.query_dtype, vcgadd_res_shape, name="vcgadd_res_ub",
                                             scope=tik.scope_ubuf)
        self._data_decode((vcgadd_res_ub, data_ub, query_ub, vmin_ub, vdiff_ub), elem_cnt, rec_255_s)

        repeat = self.SLICE_SIZE * 2 // self.MASK_FP16
        if self.d == 32:
            self.tik_inst.vcpadd(self.MASK_FP16, sq_distance_ub, vcgadd_res_ub, repeat, 1, 1, 8)
        else:
            with self.tik_inst.new_stmt_scope():
                vcpadd_res_ub = self.tik_inst.Tensor(self.query_dtype, (self.SLICE_SIZE * 2,),
                                                     name="vcpadd_res_ub", scope=tik.scope_ubuf)
                self.tik_inst.vcpadd(self.MASK_FP16, vcpadd_res_ub, vcgadd_res_ub,
                                     vcgadd_res_shape[0] // self.MASK_FP16, 1, 1, 8)
                self.tik_inst.vcpadd(self.MASK_FP16, sq_distance_ub, vcpadd_res_ub, repeat, 1, 1, 8)

        if is_tail:
            self._sq_distance_tail_dup(sq_distance_ub, tail, extreme_value_s)
        # output 1: sq_distance
        self.tik_inst.data_move(self.sq_distance_gm[output_distance_offset], sq_distance_ub,
                                0, 1, self.SLICE_SIZE // self.BLOCK_FP16, 0, 0)

        groups = self.SLICE_SIZE // self.group_size
        extreme_ub = self.tik_inst.Tensor(self.query_dtype, (groups * 2,), name="extreme_ub", scope=tik.scope_ubuf)
        grouped_extreme_dist_ub = self.tik_inst.Tensor(self.query_dtype, (groups,), name="grouped_extreme_dist_ub",
                                                       scope=tik.scope_ubuf)
        if self.extreme_mode == 1:
            self.tik_inst.vcmax(self.group_size, extreme_ub, sq_distance_ub, groups,
                                1, 1, self.group_size // self.BLOCK_FP16)
        else:
            self.tik_inst.vcmin(self.group_size, extreme_ub, sq_distance_ub, groups,
                                1, 1, self.group_size // self.BLOCK_FP16)
        self.tik_inst.vreduce(groups * 2, grouped_extreme_dist_ub, extreme_ub, 1, 1, 1, 1, 0, 0, None, "counter")
        # output 2: grouped_extreme_distance
        grouped_offset = output_offset_base_s // self.group_size + groups * thread_idx
        self.tik_inst.data_move(self.grouped_extreme_distance_gm[grouped_offset], grouped_extreme_dist_ub,
                                0, 1, util_common.ceil(groups, self.BLOCK_FP16), 0, 0)

        # output 3: sq_ivf
        self.tik_inst.data_move(self.sq_ivf_gm[output_distance_offset], sq_ivf_ub,
                                0, 1, self.SLICE_SIZE // self.BLOCK_INT32, 0, 0)

        sq_index_ub = self.tik_inst.Tensor(self.bucket_list_dtype, (self.SLICE_SIZE,), name="sq_index_ub",
                                           scope=tik.scope_ubuf)
        self.tik_inst.vadds(self.MASK_INT32, sq_index_ub, assist_idx_ub, self.SLICE_SIZE * thread_idx,
                            self.SLICE_SIZE // self.MASK_INT32, 1, 1, 8, 8)
        # output 4: sq_index
        self.tik_inst.data_move(self.sq_index_gm[output_distance_offset], sq_index_ub,
                                0, 1, self.SLICE_SIZE // self.BLOCK_INT32, 0, 0)

    def _one_bucket_process(self, ub_tensors, rec_255_s, extreme_value_s, params):
        """
        process one bucket.
        ub_tensors is (query_ub, vmin_ub, vdiff_ub, assist_idx_ub)
        params is (bucket_id_s, bucket_limit_s, bucket_offset_s, output_offset_base_s)
        """
        bucket_id_s, bucket_limit_s, bucket_offset_s, output_offset_base_s = params
        loop = bucket_limit_s // self.SLICE_SIZE
        tail = bucket_limit_s % self.SLICE_SIZE

        sq_ivf_ub = self.tik_inst.Tensor(self.bucket_list_dtype, (self.SLICE_SIZE,), name="sq_ivf_ub",
                                         scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(self.MASK_INT32, sq_ivf_ub, bucket_id_s, self.SLICE_SIZE // self.MASK_INT32, 1, 8)

        with self.tik_inst.for_range(0, loop) as thread_idx:
            self._bucket_inner_process(
                ub_tensors, sq_ivf_ub, rec_255_s,
                (extreme_value_s, bucket_offset_s, output_offset_base_s, thread_idx, tail, False))

        with self.tik_inst.if_scope(tail > 0):
            self._bucket_inner_process(ub_tensors, sq_ivf_ub, rec_255_s,
                                       (extreme_value_s, bucket_offset_s, output_offset_base_s, loop, tail, True))

    def _one_core_compute(self, core_id, bucket_idx_start, bucket_num):
        """
        compute for one core
        """
        rec_255_s = self.tik_inst.Scalar(dtype=self.query_dtype, name="rec_255_s", init_value=1.0 / 255)
        extreme_value_s = self.tik_inst.Scalar(dtype=self.query_dtype, name="extreme_value_s")
        if self.extreme_mode == 1:
            extreme_value_s.set_as(self.MIN_FP16)
        else:
            extreme_value_s.set_as(self.MAX_FP16)

        query_ub, bucket_list_ub, bucket_limits_ub, bucket_offsets_ub, vmin_ub, vdiff_ub = \
            self._init_ub_tensor(bucket_idx_start, bucket_num)
        assist_idx_ub = self._init_assist_ub()
        output_offset_ub = self._output_offset_compute(core_id)
        ub_tensors = (query_ub, vmin_ub, vdiff_ub, assist_idx_ub)

        bucket_id_s = self.tik_inst.Scalar(dtype=self.bucket_list_dtype, name="bucket_id_s")
        bucket_limit_s = self.tik_inst.Scalar(dtype=self.bucket_limits_dtype, name="bucket_limit_s")
        bucket_offset_s = self.tik_inst.Scalar(dtype=self.bucket_offsets_dtype, name="bucket_offset_s")
        output_offset_base_s = self.tik_inst.Scalar(dtype=self.bucket_limits_dtype, name="output_offset_base_s")
        with self.tik_inst.for_range(0, bucket_num) as idx:
            bucket_id_s.set_as(bucket_list_ub[idx])
            bucket_limit_s.set_as(bucket_limits_ub[idx])
            bucket_offset_s.set_as(bucket_offsets_ub[idx])
            output_offset_base_s.set_as(output_offset_ub[bucket_idx_start + idx])

            self._one_bucket_process(ub_tensors, rec_255_s, extreme_value_s,
                                     (bucket_id_s, bucket_limit_s, bucket_offset_s, output_offset_base_s))


def _check_input_params(args_list, kernel_name):
    """
    check input parameters.
    args_list is (ivf, query, bucket_list, bucket_limits, bucket_offsets, vmin, vdiff,
                  actual_count, sq_distance, grouped_extreme_distance, sq_ivf, sq_index, group_size)
    """
    (ivf, query, bucket_list, bucket_limits, bucket_offsets, vmin, vdiff,
     actual_count, sq_distance, grouped_extreme_distance, sq_ivf, sq_index, group_size) = args_list

    ivf_dtype = ivf.get("dtype").lower()
    query_dtype = query.get("dtype").lower()
    bucket_list_dtype = bucket_list.get("dtype").lower()
    bucket_limits_dtype = bucket_limits.get("dtype").lower()
    bucket_offsets_dtype = bucket_offsets.get("dtype").lower()
    vmin_dtype = vmin.get("dtype").lower()
    vdiff_dtype = vdiff.get("dtype").lower()
    para_check.check_dtype(ivf_dtype, ("uint8",), param_name="ivf")
    para_check.check_dtype(query_dtype, ("float16",), param_name="query")
    para_check.check_dtype(bucket_list_dtype, ("int32",), param_name="bucket_list")
    para_check.check_dtype(bucket_limits_dtype, ("int32",), param_name="bucket_limits")
    para_check.check_dtype(bucket_offsets_dtype, ("int64",), param_name="bucket_offsets")
    para_check.check_dtype(vmin_dtype, ("float16",), param_name="vmin")
    para_check.check_dtype(vdiff_dtype, ("float16",), param_name="vdiff")

    ivf_shape = ivf.get("shape")
    query_shape = query.get("shape")
    bucket_list_shape = bucket_list.get("shape")
    bucket_limits_shape = bucket_limits.get("shape")
    bucket_offsets_shape = bucket_offsets.get("shape")
    vmin_shape = vmin.get("shape")
    vdiff_shape = vdiff.get("shape")
    para_check.check_shape(ivf_shape, min_rank=2, max_rank=2, param_name="ivf")
    para_check.check_shape(query_shape, min_rank=1, max_rank=1, param_name="query")
    para_check.check_shape(bucket_list_shape, min_rank=1, max_rank=1, param_name="bucket_list")
    para_check.check_shape(bucket_limits_shape, min_rank=1, max_rank=1, param_name="bucket_limits")
    para_check.check_shape(bucket_offsets_shape, min_rank=1, max_rank=1, param_name="bucket_offsets")
    para_check.check_shape(vmin_shape, min_rank=1, max_rank=1, param_name="vmin")
    para_check.check_shape(vdiff_shape, min_rank=1, max_rank=1, param_name="vdiff")

    # SQ8 code, d only support 32 and 64
    if (query_shape[0] != 32 and query_shape[0] != 64) or query_shape[0] != vmin_shape[0] or \
            query_shape[0] != vdiff_shape[0]:
        rule = "the dim of query shape should be equal to 32 or 64 for SQ8 code"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)

    if tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // 1024 < 256 and query_shape[0] == 64:
        rule = "the dim of query shape should not be equal to 64 for Ascend910B"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)

    if group_size != 64:
        rule = "the group_size only support 64 now"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)

    actual_count_dtype = actual_count.get("dtype").lower()
    sq_distance_dtype = sq_distance.get("dtype").lower()
    grouped_extreme_distance_dtype = grouped_extreme_distance.get("dtype").lower()
    sq_ivf_dtype = sq_ivf.get("dtype").lower()
    sq_index_dtype = sq_index.get("dtype").lower()
    para_check.check_dtype(actual_count_dtype, ("int32",), param_name="actual_count")
    para_check.check_dtype(sq_distance_dtype, ("float16",), param_name="sq_distance")
    para_check.check_dtype(grouped_extreme_distance_dtype, ("float16",), param_name="grouped_extreme_distance")
    para_check.check_dtype(sq_ivf_dtype, ("int32",), param_name="sq_ivf")
    para_check.check_dtype(sq_index_dtype, ("int32",), param_name="sq_index")


@register_operator("ScanSQCodes")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def scan_sq_codes(ivf, query, bucket_list, bucket_limits, bucket_offsets, vmin, vdiff,
                  actual_count, sq_distance, grouped_extreme_distance, sq_ivf, sq_index,
                  total_limit, group_size=64, extreme_mode=0, split_count=1, split_index=0,
                  kernel_name="scan_sq_codes"):
    """
    ScanSQCodes op

    Parameters
    ----------
    ivf: dict
        the dict of input ivf, shape is [dim0, d], d=32 or 64.
    query: dict
        the dict of input query, shape is [d], d=32 or 64.
    bucket_list: dict
        the dict of input bucket_list, shape is [nprobe], e.g. nprobe=64.
    bucket_limits: dict
        the dict of input bucket_limits, shape is [nprobe].
    bucket_offsets: dict
        the dict of input bucket_offsets, shape is [nprobe].
    vmin: dict
        the dict of input vmin, shape is [d], d=32 or 64.
    vdiff: dict
        the dict of input vdiff, shape is [d], d=32 or 64.
    actual_count: dict
        the dict of output actual_count, shape is [1].
    sq_distance: dict
        the dict of output sq_distance, shape is [sum(shape_i)], shape_i=align_value(bucket_limits[i], 1024).
    grouped_extreme_distance: dict
        the dict of output grouped_extreme_distance, shape is [sum(shape_i // group_size)].
    sq_ivf: dict
        the dict of output sq_ivf, shape is same as sq_distance.
    sq_index: dict
        the dict of output sq_index, shape is same as sq_distance.
    total_limit: int
        an integer indicating the max dim of output sq_distance.
    group_size: int
        an optional attr, default value is 64.
    extreme_mode: int
        an optional attr, default value is 0. 0 means minimum, 1 means maximum.
    split_count: int
        an optional attr, default value is 1.
        1: aicore, 2: aicore + vector core.
    split_index: int
        an optional attr, default value is 0.
        0: aicore, 1: vector core.
    kernel_name: str
        cce kernel name, default value is "scan_sq_codes".

    Returns
    -------
    tik instance
    """
    args_list = (ivf, query, bucket_list, bucket_limits, bucket_offsets, vmin, vdiff,
                 actual_count, sq_distance, grouped_extreme_distance, sq_ivf, sq_index, group_size)
    _check_input_params(args_list, kernel_name)

    obj = ScanSQCodes((ivf, query, bucket_list, bucket_limits, bucket_offsets, vmin, vdiff),
                      total_limit, group_size, extreme_mode)
    obj.scan_sq_codes_compute()

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {
        "core_num": obj.core_num,
        "split_count": split_count,
        "split_index": split_index
    })

    tik_inst = obj.tik_inst
    inputs_gm, outputs_gm = obj.get_inputs_outputs_gm()
    opt_config = {"enable_const_fold": True}
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=inputs_gm,
                      outputs=outputs_gm,
                      flowtable=(obj.tiling_gm,),
                      config=opt_config)

    return tik_inst
