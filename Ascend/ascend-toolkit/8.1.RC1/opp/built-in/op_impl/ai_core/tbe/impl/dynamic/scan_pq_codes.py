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
scan_pq_codes.py
"""
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.dynamic.scan_pq_codes_expand import ScanPQCodesExpand


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max int64
    MAX_INT64 = 2**63 - 1
    MIN_FP16 = -65504
    MAX_FP16 = 65504
    # tiling param num
    TILING_ARG_NUM = 8
    # reserved ub size
    MASK_FLOAT16 = 128
    MASK_FLOAT32 = 64
    BLOCK_INT64 = 4
    BLOCK_INT32 = 8
    BLOCK_FLOAT16 = 16
    BLOCK_UINT8 = 32
    IVF_UNIT_LEN = 16
    SLICE_SIZE = 1024
    SLICE_INNER_SIZE = 256
    IVF_SLICE_SIZE = SLICE_SIZE * IVF_UNIT_LEN
    IVF_SLICE_INNER_SIZE = SLICE_INNER_SIZE * IVF_UNIT_LEN
    INNER_LOOP_TIME = SLICE_SIZE // SLICE_INNER_SIZE
    MAX_BUCKET_LEN = 64
    ADC_DIM = 1


def _ceil_div(dividend, divisor):
    result = (dividend + divisor - 1) // divisor
    return result


def _ceil_fill(dividend, divisor):
    result = ((dividend + divisor - 1) // divisor) * divisor
    return result


def _floor_fill(dividend, divisor):
    result = (dividend // divisor) * divisor
    return result


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,too-many-statements,unused-argument,invalid-name
# 'pylint: disable=too-few-public-methods
class ScanPQCodes():
    """
    Function: use to store ScanPQCodes base parameters
    """

    def __init__(self, attrs, dtypes):
        self.tik_instance = tik.Tik()
        self.opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        # tiling params
        self.bucket_num_total = self.tik_instance.Scalar("int64", name="bucket_num_total")
        self.bucket_start_base = self.tik_instance.Scalar("int64", name="bucket_start_base")
        self.bucket_num_low = self.tik_instance.Scalar("int64", name="bucket_num_low")
        self.bucket_num_high = self.tik_instance.Scalar("int64", name="bucket_num_high")
        self.high_core_num = self.tik_instance.Scalar("int64", name="high_core_num")
        self.core_num_var = self.tik_instance.Scalar("int64", name="core_num_var")
        # attrs
        (total_limit, group_size, extreme_mode, split_count, split_index) = attrs
        self.group_size = group_size
        self.total_limit = total_limit
        self.extreme_mode = extreme_mode
        self.split_count = split_count
        self.split_index = split_index
        # dtype
        (ivf_dtype, bucket_list_dtype, bucket_base_distance_dtype, bucket_limits_dtype, bucket_offsets_dtype,
         adc_tables_dtype) = dtypes
        self.ivf_dtype = ivf_dtype
        self.bucket_list_dtype = bucket_list_dtype
        self.bucket_base_distance_dtype = bucket_base_distance_dtype
        self.bucket_limits_dtype = bucket_limits_dtype
        self.bucket_offsets_dtype = bucket_offsets_dtype
        self.adc_tables_dtype = adc_tables_dtype
        # input gm
        self.ivf_gm = None
        self.bucket_list_gm = None
        self.bucket_base_distance_gm = None
        self.bucket_limits_gm = None
        self.bucket_offsets_gm = None
        self.adc_tables_gm = None
        # output gm
        self.actual_count_gm = None
        self.pq_distance_gm = None
        self.grouped_extrim_distance_gm = None
        self.pq_ivf_gm = None
        self.pq_index_gm = None
        # ub
        self.adc_tables_ub_fp16 = None
        self.adc_trans_ub_fp16 = None
        self.assist_add_init_ub_fp32 = None
        self.assist_pq_index_init_ub_fp32 = None
        self.ivf_cur_process_ub_uint8 = None
        self.ivf_cur_process_ub_fp16 = None
        self.pq_distance_ub_fp16 = None
        self.block_extrim_ub_fp16 = None
        self.grouped_extrim_distance_ub_fp16 = None
        self.bucket_list_ub_int32 = None
        self.bucket_base_distance_ub_fp16 = None
        self.bucket_limits_ub_int32 = None
        self.bucket_offsets_ub_int64 = None
        self.ivf_cur_process_ub_fp32 = None

    def scan_pq_codes_operator(self, kernel_name):
        """
        scan_pq_codes_operator
        """
        self._tiling_args()
        self._init_gm_tensor()
        self._run_multi_core()
        # Build CCE
        # this "global_variable_link" flag suggest ccec.py do link without "-r" option
        # which will result in global variable in cce file with wrong address
        tbe_context.get_context().add_compile_info("vars", {
            "core_nums": self.core_nums,
            "split_count": self.split_count,
            "split_index": self.split_index
        })
        input_list = [
            self.ivf_gm, self.bucket_list_gm, self.bucket_base_distance_gm, self.bucket_limits_gm,
            self.bucket_offsets_gm, self.adc_tables_gm
        ]
        output_list = [
            self.actual_count_gm, self.pq_distance_gm, self.grouped_extrim_distance_gm, self.pq_ivf_gm, self.pq_index_gm
        ]
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=input_list,
                                   outputs=output_list,
                                   flowtable=(self.tiling_gm,),
                                   config=self.opt_config)

        return self.tik_instance

    def _init_gm_tensor(self):
        # input gm
        self.ivf_gm = self.tik_instance.Tensor(self.ivf_dtype, (Constant.MAX_INT64,), name="ivf", scope=tik.scope_gm)
        self.bucket_list_gm = self.tik_instance.Tensor(self.bucket_list_dtype, (Constant.MAX_INT64,),
                                                       name="bucket_list",
                                                       scope=tik.scope_gm)
        self.bucket_base_distance_gm = self.tik_instance.Tensor(self.bucket_base_distance_dtype, (Constant.MAX_INT64,),
                                                                name="bucket_base_distance",
                                                                scope=tik.scope_gm)
        self.bucket_limits_gm = self.tik_instance.Tensor(self.bucket_limits_dtype, (Constant.MAX_INT64,),
                                                         name="bucket_limits",
                                                         scope=tik.scope_gm)
        self.bucket_offsets_gm = self.tik_instance.Tensor(self.bucket_offsets_dtype, (Constant.MAX_INT64,),
                                                          name="bucket_offsets",
                                                          scope=tik.scope_gm)
        self.adc_tables_gm = self.tik_instance.Tensor(self.adc_tables_dtype, (Constant.MAX_INT64,),
                                                      name="adc_tables",
                                                      scope=tik.scope_gm)
        # output gm
        self.actual_count_gm = self.tik_instance.Tensor(self.bucket_list_dtype, (1,),
                                                        name="actual_count",
                                                        scope=tik.scope_gm)
        self.pq_distance_gm = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                       (_ceil_fill(self.total_limit, Constant.SLICE_SIZE),),
                                                       name="pq_distance",
                                                       scope=tik.scope_gm)
        self.grouped_extrim_distance_gm = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                                   (_ceil_div(self.total_limit, self.group_size),),
                                                                   name="grouped_extrim_distance",
                                                                   scope=tik.scope_gm)
        self.pq_ivf_gm = self.tik_instance.Tensor(self.bucket_list_dtype, (self.total_limit,),
                                                  name="pq_ivf",
                                                  scope=tik.scope_gm)
        self.pq_index_gm = self.tik_instance.Tensor(self.bucket_list_dtype, (self.total_limit,),
                                                    name="pq_index",
                                                    scope=tik.scope_gm)

    def _init_ub_tensor(self):
        #adc ub
        self.adc_tables_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (272, 16, 16),
                                                           name="adc_tables_ub_fp16",
                                                           scope=tik.scope_ubuf)
        self.adc_trans_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (256 * 17,),
                                                          name="adc_trans_ub_fp16",
                                                          scope=tik.scope_ubuf)
        #assist ub
        self.assist_add_init_ub_fp32 = self.tik_instance.Tensor("float32", (Constant.MASK_FLOAT32 * 2 + 24,),
                                                                name="assist_add_init_ub_fp32",
                                                                scope=tik.scope_ubuf)
        self.assist_pq_index_init_ub_fp32 = self.tik_instance.Tensor("float32", (Constant.SLICE_SIZE,),
                                                                     name="assist_pq_index_init_ub_fp32",
                                                                     scope=tik.scope_ubuf)
        #input data
        self.bucket_list_ub_int32 = self.tik_instance.Tensor(self.bucket_list_dtype, (Constant.MAX_BUCKET_LEN,),
                                                             name="bucket_list_ub_int32",
                                                             scope=tik.scope_ubuf)
        self.bucket_base_distance_ub_fp16 = self.tik_instance.Tensor(self.bucket_base_distance_dtype,
                                                                     (Constant.MAX_BUCKET_LEN,),
                                                                     name="bucket_base_distance_ub_fp16",
                                                                     scope=tik.scope_ubuf)
        self.bucket_limits_ub_int32 = self.tik_instance.Tensor(self.bucket_limits_dtype, (Constant.MAX_BUCKET_LEN,),
                                                               name="bucket_limits_ub_int32",
                                                               scope=tik.scope_ubuf)
        self.bucket_offsets_ub_int64 = self.tik_instance.Tensor(self.bucket_offsets_dtype, (Constant.MAX_BUCKET_LEN,),
                                                                name="bucket_offsets_ub_int64",
                                                                scope=tik.scope_ubuf)

    def _tiling_args(self):
        """
        tiling_args
        """
        tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                             name="tiling_ub",
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.TILING_ARG_NUM // Constant.BLOCK_INT64, 0,
                                    0)
        tiling_para_index = 0
        self.bucket_num_total.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.bucket_start_base.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.bucket_num_low.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.bucket_num_high.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.high_core_num.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.core_num_var.set_as(tiling_ub[tiling_para_index])

    def _calc_output_count(self, output_count, bucket_idx):
        output_count.set_as(0)
        with self.tik_instance.if_scope(bucket_idx // Constant.MAX_BUCKET_LEN > 0):
            with self.tik_instance.for_range(0, bucket_idx // Constant.MAX_BUCKET_LEN) as loop_idx:
                self.tik_instance.data_move(
                    self.bucket_limits_ub_int32,
                    self.bucket_limits_gm[self.bucket_start_base + loop_idx * Constant.MAX_BUCKET_LEN], 0, 1,
                    Constant.MAX_BUCKET_LEN // Constant.BLOCK_INT32, 0, 0)
                bucket_counts = self.tik_instance.Scalar("int32", name="bucket_counts")
                with self.tik_instance.for_range(0, Constant.MAX_BUCKET_LEN) as idx:
                    bucket_counts.set_as(self.bucket_limits_ub_int32[idx])
                    output_count.set_as(output_count + _ceil_fill(bucket_counts, Constant.SLICE_SIZE))
        with self.tik_instance.if_scope(bucket_idx % Constant.MAX_BUCKET_LEN > 0):
            self.tik_instance.data_move(
                self.bucket_limits_ub_int32,
                self.bucket_limits_gm[self.bucket_start_base + _floor_fill(bucket_idx, Constant.MAX_BUCKET_LEN)], 0, 1,
                _ceil_div(bucket_idx % Constant.MAX_BUCKET_LEN, Constant.BLOCK_INT32), 0, 0)
            bucket_counts = self.tik_instance.Scalar("int32", name="bucket_counts")
            with self.tik_instance.for_range(0, bucket_idx % Constant.MAX_BUCKET_LEN) as idx:
                bucket_counts.set_as(self.bucket_limits_ub_int32[idx])
                output_count.set_as(output_count + _ceil_fill(bucket_counts, Constant.SLICE_SIZE))

    def _create_adc_table(self, bucket_idx):
        # conver adc_tables shape from (256,16) to (256,16,16) to prevent bank conflict
        self.tik_instance.data_move(self.adc_trans_ub_fp16,
                                    self.adc_tables_gm[(bucket_idx + self.bucket_start_base) * 256 * 16], 0, 1, 256, 0,
                                    0)
        for j in range(16):
            dst_list = [self.adc_tables_ub_fp16[j * 16 + i * 256 + i * 16] for i in range(16)]
            src_list = [self.adc_trans_ub_fp16[j * 256 + i * 16] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list, 16, 256 + 16, 1)
        if self.split_index == 0:
            self.tik_instance.data_move(self.adc_tables_ub_fp16[256], self.adc_tables_ub_fp16[256 + 16], 0, 256, 16, 1,
                                        0)
        else:
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                adc_tables_assis_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (128, 16, 16),
                                                                    name="adc_tables_assis_ub_fp16",
                                                                    scope=tik.scope_ubuf)
                self.tik_instance.vadds(Constant.MASK_FLOAT16, adc_tables_assis_ub_fp16,
                                        self.adc_tables_ub_fp16[256 + 16 + 128], 0, 255, 1, 1, 8, 17)
                self.tik_instance.vadds(Constant.MASK_FLOAT16, self.adc_tables_ub_fp16[256],
                                        self.adc_tables_ub_fp16[256 + 16], 0, 255, 1, 1, 16, 17)
                self.tik_instance.vadds(Constant.MASK_FLOAT16, self.adc_tables_ub_fp16[256 + 128],
                                        adc_tables_assis_ub_fp16, 0, 255, 1, 1, 16, 8)

    def _init_assist_ub(self):
        assist_init_ub = self.tik_instance.Tensor("float32", (Constant.IVF_UNIT_LEN,),
                                                  name="assist_init_ub",
                                                  scope=tik.scope_ubuf)
        assist_value = self.tik_instance.Scalar("float32", name="assist_value")
        assist_value.set_as(0)
        assist_init_ub[0].set_as(assist_value)
        assist_value.set_as(1)
        assist_init_ub[1].set_as(assist_value)
        assist_value.set_as(2)
        assist_init_ub[2].set_as(assist_value)
        assist_value.set_as(3)
        assist_init_ub[3].set_as(assist_value)
        assist_value.set_as(4)
        assist_init_ub[4].set_as(assist_value)
        assist_value.set_as(5)
        assist_init_ub[5].set_as(assist_value)
        assist_value.set_as(6)
        assist_init_ub[6].set_as(assist_value)
        assist_value.set_as(7)
        assist_init_ub[7].set_as(assist_value)
        self.tik_instance.vadds(8, assist_init_ub[8], assist_init_ub, 8, 1, 1, 1, 8, 8)
        self.tik_instance.vadds(16, self.assist_add_init_ub_fp32, assist_init_ub, 0, 1, 1, 1, 2, 2)
        self.tik_instance.vmuls(16, self.assist_add_init_ub_fp32, self.assist_add_init_ub_fp32, 32, 1, 1, 1, 8, 8)
        self.tik_instance.vadds(16, self.assist_add_init_ub_fp32[16], self.assist_add_init_ub_fp32, 0, 1, 1, 1, 2, 2)
        self.tik_instance.vadds(32, self.assist_add_init_ub_fp32[32], self.assist_add_init_ub_fp32, 0, 1, 1, 1, 4, 4)
        self.tik_instance.vadds(64, self.assist_add_init_ub_fp32[64], self.assist_add_init_ub_fp32, 0, 1, 1, 1, 8, 8)
        self.tik_instance.vadds(16, self.assist_pq_index_init_ub_fp32, assist_init_ub, 0, 1, 1, 1, 2, 2)
        self.tik_instance.vadds(16, self.assist_pq_index_init_ub_fp32[16], self.assist_pq_index_init_ub_fp32, 16, 1, 1,
                                1, 2, 2)
        self.tik_instance.vadds(32, self.assist_pq_index_init_ub_fp32[32], self.assist_pq_index_init_ub_fp32, 32, 1, 1,
                                1, 4, 4)
        self.tik_instance.vadds(64, self.assist_pq_index_init_ub_fp32[64], self.assist_pq_index_init_ub_fp32, 64, 1, 1,
                                1, 8, 8)
        self.tik_instance.vadds(64, self.assist_pq_index_init_ub_fp32[128], self.assist_pq_index_init_ub_fp32, 128, 2,
                                1, 1, 8, 8)
        self.tik_instance.vadds(64, self.assist_pq_index_init_ub_fp32[256], self.assist_pq_index_init_ub_fp32, 256, 4,
                                1, 1, 8, 8)
        self.tik_instance.vadds(64, self.assist_pq_index_init_ub_fp32[512], self.assist_pq_index_init_ub_fp32, 512, 8,
                                1, 1, 8, 8)

    def _get_input_data(self, bucket_idx):
        self.tik_instance.data_move(
            self.bucket_list_ub_int32,
            self.bucket_list_gm[_floor_fill(bucket_idx, Constant.MAX_BUCKET_LEN) + self.bucket_start_base], 0, 1,
            _ceil_div((bucket_idx % Constant.MAX_BUCKET_LEN) + 1, Constant.BLOCK_INT32), 0, 0)
        self.tik_instance.data_move(
            self.bucket_base_distance_ub_fp16,
            self.bucket_base_distance_gm[_floor_fill(bucket_idx, Constant.MAX_BUCKET_LEN) + self.bucket_start_base], 0,
            1, _ceil_div((bucket_idx % Constant.MAX_BUCKET_LEN) + 1, Constant.BLOCK_FLOAT16), 0, 0)
        self.tik_instance.data_move(
            self.bucket_offsets_ub_int64,
            self.bucket_offsets_gm[_floor_fill(bucket_idx, Constant.MAX_BUCKET_LEN) + self.bucket_start_base], 0, 1,
            _ceil_div((bucket_idx % Constant.MAX_BUCKET_LEN) + 1, Constant.BLOCK_INT64), 0, 0)
        self.tik_instance.data_move(
            self.bucket_limits_ub_int32,
            self.bucket_limits_gm[_floor_fill(bucket_idx, Constant.MAX_BUCKET_LEN) + self.bucket_start_base], 0, 1,
            _ceil_div((bucket_idx % Constant.MAX_BUCKET_LEN) + 1, Constant.BLOCK_INT32), 0, 0)

    def _set_single_bucket_param(self, args):
        (bucket_idx, bucket_id, bucket_base_dis, bucket_limit, bucket_offset_input, bucket_offset_output,
         bucket_max_offset) = args
        bucket_id.set_as(self.bucket_list_ub_int32[bucket_idx % Constant.MAX_BUCKET_LEN])
        bucket_base_dis.set_as(self.bucket_base_distance_ub_fp16[bucket_idx % Constant.MAX_BUCKET_LEN])
        bucket_limit.set_as(self.bucket_limits_ub_int32[bucket_idx % Constant.MAX_BUCKET_LEN])
        bucket_offset_input.set_as(self.bucket_offsets_ub_int64[bucket_idx % Constant.MAX_BUCKET_LEN])
        bucket_offset_input.set_as(bucket_offset_input * Constant.IVF_UNIT_LEN)
        self._calc_output_count(bucket_offset_output, bucket_idx)
        bucket_max_offset.set_as(bucket_offset_output // self.group_size)

    def _run_multi_core(self):
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_idx:
            self._init_ub_tensor()
            self._init_assist_ub()
            bucket_id = self.tik_instance.Scalar(self.bucket_list_dtype, name="bucket_id")
            bucket_base_dis = self.tik_instance.Scalar(self.bucket_base_distance_dtype, name="bucket_base_dis")
            bucket_limit = self.tik_instance.Scalar(self.bucket_limits_dtype, name="bucket_limit")
            bucket_offset_input = self.tik_instance.Scalar(self.bucket_offsets_dtype, name="bucket_offset_input")
            bucket_offset_output = self.tik_instance.Scalar(self.bucket_limits_dtype, name="bucket_offset_output")
            bucket_offset_max = self.tik_instance.Scalar(self.bucket_limits_dtype, name="bucket_offset_max")
            actual_count_ub_int32 = self.tik_instance.Tensor("int32", (Constant.BLOCK_INT32,),
                                                             name="actual_count_ub_int32",
                                                             scope=tik.scope_ubuf)

            def _inner_handle(bucket_start, bucket_end):
                with self.tik_instance.for_range(bucket_start, bucket_end) as bucket_idx:
                    args = (bucket_idx, bucket_id, bucket_base_dis, bucket_limit, bucket_offset_input,
                            bucket_offset_output, bucket_offset_max)
                    self._get_input_data(bucket_idx)
                    self._set_single_bucket_param(args)
                    self._run_one_core_loop(args)

            # calculate and output actual_total_num by core 0 for multi core
            with self.tik_instance.if_scope(core_idx == 0):
                actual_total_num = self.tik_instance.Scalar("int32", name="actual_total_num")
                self._calc_output_count(actual_total_num, self.bucket_num_total)
                actual_count_ub_int32[0].set_as(actual_total_num)
                self.tik_instance.data_move(self.actual_count_gm, actual_count_ub_int32, 0, 1, 1, 0, 0)
            with self.tik_instance.if_scope(core_idx < self.high_core_num):
                _inner_handle(self.bucket_num_high * core_idx, self.bucket_num_high * (core_idx + 1))
            with self.tik_instance.else_scope():
                high_base = self.tik_instance.Scalar("int32", name="high_base")
                low_core_idx = self.tik_instance.Scalar("int32", name="low_core_idx")
                high_base.set_as(self.bucket_num_high * self.high_core_num)
                low_core_idx.set_as(core_idx - self.high_core_num)
                _inner_handle(high_base + self.bucket_num_low * low_core_idx,
                              high_base + self.bucket_num_low * (low_core_idx + 1))

    def _handle_input_data(self, args):
        (bucket_offset_input, ivf_slice_size, inner_loop_time, thread_idx) = args
        # input data
        with self.tik_instance.if_scope(ivf_slice_size == Constant.IVF_SLICE_SIZE):
            self.tik_instance.data_move(self.ivf_cur_process_ub_uint8,
                                        self.ivf_gm[bucket_offset_input + thread_idx * Constant.IVF_SLICE_SIZE], 0, 1,
                                        Constant.IVF_SLICE_SIZE // Constant.BLOCK_UINT8, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.ivf_cur_process_ub_uint8,
                                        self.ivf_gm[bucket_offset_input + thread_idx * Constant.IVF_SLICE_SIZE], 0, 1,
                                        _ceil_div(ivf_slice_size, Constant.BLOCK_UINT8), 0, 0)

        with self.tik_instance.for_range(0, inner_loop_time) as count_idx:
            # ivf reprocess for vgather, coordination = (ivf * 256 + offset) * 2
            self.tik_instance.vconv(Constant.MASK_FLOAT16, "", self.ivf_cur_process_ub_fp16,
                                    self.ivf_cur_process_ub_uint8[count_idx * Constant.IVF_SLICE_INNER_SIZE],
                                    Constant.IVF_SLICE_INNER_SIZE // Constant.MASK_FLOAT16, 1, 1, 8, 4)
            self.tik_instance.vconv(Constant.MASK_FLOAT32, "", self.ivf_cur_process_ub_fp32,
                                    self.ivf_cur_process_ub_fp16,
                                    Constant.IVF_SLICE_INNER_SIZE // Constant.MASK_FLOAT32, 1, 1, 8, 4)
            vmuls_result_ub_fp32 = self.ivf_cur_process_ub_fp16.reinterpret_cast_to("float32")
            self.tik_instance.vmuls(Constant.MASK_FLOAT32, vmuls_result_ub_fp32, self.ivf_cur_process_ub_fp32, 512,
                                    Constant.IVF_SLICE_INNER_SIZE // Constant.MASK_FLOAT32, 1, 1, 8, 8)
            self.tik_instance.vadd(Constant.MASK_FLOAT32, self.ivf_cur_process_ub_fp32, vmuls_result_ub_fp32,
                                   self.assist_add_init_ub_fp32,
                                   Constant.IVF_SLICE_INNER_SIZE // 2 // Constant.MASK_FLOAT32, 1, 1, 1, 16, 16, 0)
            self.tik_instance.vadd(Constant.MASK_FLOAT32, self.ivf_cur_process_ub_fp32[Constant.MASK_FLOAT32],
                                   vmuls_result_ub_fp32[Constant.MASK_FLOAT32],
                                   self.assist_add_init_ub_fp32[Constant.MASK_FLOAT32],
                                   Constant.IVF_SLICE_INNER_SIZE // 2 // Constant.MASK_FLOAT32, 1, 1, 1, 16, 16, 0)
            ivf_cur_process_ub_int32 = self.ivf_cur_process_ub_fp32.reinterpret_cast_to("int32")
            self.tik_instance.vconv(Constant.MASK_FLOAT32, "floor", ivf_cur_process_ub_int32,
                                    self.ivf_cur_process_ub_fp32,
                                    Constant.IVF_SLICE_INNER_SIZE // Constant.MASK_FLOAT32, 1, 1, 8, 8)
            # distance
            vgather_result_ub_fp16 = self.ivf_cur_process_ub_fp16
            self.tik_instance.vgather(Constant.MASK_FLOAT16, vgather_result_ub_fp16, self.adc_tables_ub_fp16,
                                      ivf_cur_process_ub_int32, Constant.IVF_SLICE_INNER_SIZE // Constant.MASK_FLOAT16,
                                      8, 0, 0, "normal")
            self.tik_instance.vcgadd(Constant.MASK_FLOAT16,
                                     self.pq_distance_ub_fp16[count_idx * Constant.SLICE_INNER_SIZE],
                                     vgather_result_ub_fp16, Constant.IVF_SLICE_INNER_SIZE // Constant.MASK_FLOAT16, 1,
                                     1, 8)

    def _handle_pq_distance(self, args):
        (_, _, slice_size) = args
        # extrim
        if self.extreme_mode == 1:
            self.tik_instance.vcmax(self.group_size, self.block_extrim_ub_fp16, self.pq_distance_ub_fp16,
                                    slice_size // self.group_size, 1, 1, self.group_size // Constant.BLOCK_FLOAT16)
        else:
            self.tik_instance.vcmin(self.group_size, self.block_extrim_ub_fp16, self.pq_distance_ub_fp16,
                                    slice_size // self.group_size, 1, 1, self.group_size // Constant.BLOCK_FLOAT16)
        self.tik_instance.vreduce((slice_size // self.group_size) * 2, self.grouped_extrim_distance_ub_fp16,
                                  self.block_extrim_ub_fp16, 1, 1, 1, 1, 0, 0, None, "counter")

    def _run_one_core_loop(self, args):
        (bucket_idx, bucket_id, bucket_base_dis, bucket_limit, bucket_offset_input, bucket_offset_output,
         bucket_offset_max) = args
        self._create_adc_table(bucket_idx)
        thread_loop = self.tik_instance.Scalar("int32", name="thread_loop")
        thread_tail = self.tik_instance.Scalar("int32", name="thread_tail")
        index_offset = self.tik_instance.Scalar("float32", name="index_offset")
        thread_loop.set_as(bucket_limit // Constant.SLICE_SIZE)
        thread_tail.set_as(bucket_limit % Constant.SLICE_SIZE)
        with self.tik_instance.for_range(0, thread_loop, thread_num=2) as thread_idx:
            self.ivf_cur_process_ub_uint8 = self.tik_instance.Tensor(self.ivf_dtype, (Constant.IVF_SLICE_SIZE,),
                                                                     name="ivf_cur_process_ub_uint8",
                                                                     scope=tik.scope_ubuf)
            self.ivf_cur_process_ub_fp16 = self.tik_instance.Tensor(
                self.adc_tables_dtype, (Constant.IVF_SLICE_INNER_SIZE * 2 + Constant.MASK_FLOAT16,),
                name="ivf_cur_process_ub_fp16",
                scope=tik.scope_ubuf)
            self.ivf_cur_process_ub_fp32 = self.tik_instance.Tensor("float32", (Constant.IVF_SLICE_INNER_SIZE,),
                                                                    name="ivf_cur_process_ub_fp32",
                                                                    scope=tik.scope_ubuf)
            self.pq_distance_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (Constant.SLICE_SIZE,),
                                                                name="pq_distance_ub_fp16",
                                                                scope=tik.scope_ubuf)
            self.block_extrim_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (Constant.BLOCK_FLOAT16 * 2,),
                                                                 name="block_extrim_ub_fp16",
                                                                 scope=tik.scope_ubuf)
            self.grouped_extrim_distance_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                                            (Constant.BLOCK_FLOAT16,),
                                                                            name="grouped_extrim_distance_ub_fp16",
                                                                            scope=tik.scope_ubuf)
            index_offset.set_as(thread_idx * Constant.SLICE_SIZE)
            args = (bucket_offset_input, Constant.IVF_SLICE_SIZE, Constant.INNER_LOOP_TIME, thread_idx)
            self._handle_input_data(args)
            self.tik_instance.vadds(Constant.MASK_FLOAT16, self.pq_distance_ub_fp16, self.pq_distance_ub_fp16,
                                    bucket_base_dis, Constant.SLICE_SIZE // Constant.MASK_FLOAT16, 1, 1, 8, 8)
            args_dis = (bucket_offset_output, bucket_offset_max, Constant.SLICE_SIZE)
            self._handle_pq_distance(args_dis)
            pq_ivf_ub_int32 = self.ivf_cur_process_ub_uint8.reinterpret_cast_to("int32")
            self.tik_instance.vector_dup(Constant.MASK_FLOAT32, pq_ivf_ub_int32, bucket_id,
                                         Constant.SLICE_SIZE // Constant.MASK_FLOAT32, 1, 8)
            # index set by assistant cube for performance
            pq_index_ub_fp32 = self.ivf_cur_process_ub_fp32
            self.tik_instance.vadds(Constant.MASK_FLOAT32, pq_index_ub_fp32, self.assist_pq_index_init_ub_fp32,
                                    index_offset, Constant.SLICE_SIZE // Constant.MASK_FLOAT32, 1, 1, 8, 8)
            pq_index_ub_int32 = pq_index_ub_fp32.reinterpret_cast_to("int32")
            self.tik_instance.vconv(Constant.MASK_FLOAT32, "round", pq_index_ub_int32, pq_index_ub_fp32,
                                    Constant.SLICE_SIZE // Constant.MASK_FLOAT32, 1, 1, 8, 8)
            self.tik_instance.data_move(self.pq_index_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_idx],
                                        pq_index_ub_int32, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_INT32, 0, 0)
            self.tik_instance.data_move(
                self.grouped_extrim_distance_gm[bucket_offset_max +
                                                (Constant.SLICE_SIZE // self.group_size) * thread_idx],
                self.grouped_extrim_distance_ub_fp16, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.pq_distance_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_idx],
                                        self.pq_distance_ub_fp16, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_FLOAT16,
                                        0, 0)
            self.tik_instance.data_move(self.pq_ivf_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_idx],
                                        pq_ivf_ub_int32, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_INT32, 0, 0)
        with self.tik_instance.if_scope(thread_tail > 0):
            self.ivf_cur_process_ub_uint8 = self.tik_instance.Tensor(self.ivf_dtype, (Constant.IVF_SLICE_SIZE,),
                                                                     name="ivf_cur_process_ub_uint8",
                                                                     scope=tik.scope_ubuf)
            self.ivf_cur_process_ub_fp16 = self.tik_instance.Tensor(
                self.adc_tables_dtype, (Constant.IVF_SLICE_INNER_SIZE * 2 + Constant.MASK_FLOAT16,),
                name="ivf_cur_process_ub_fp16",
                scope=tik.scope_ubuf)
            self.ivf_cur_process_ub_fp32 = self.tik_instance.Tensor("float32", (Constant.IVF_SLICE_INNER_SIZE,),
                                                                    name="ivf_cur_process_ub_fp32",
                                                                    scope=tik.scope_ubuf)
            self.pq_distance_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (Constant.SLICE_SIZE,),
                                                                name="pq_distance_ub_fp16",
                                                                scope=tik.scope_ubuf)
            self.block_extrim_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (Constant.BLOCK_FLOAT16 * 2,),
                                                                 name="block_extrim_ub_fp16",
                                                                 scope=tik.scope_ubuf)
            self.grouped_extrim_distance_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                                            (Constant.BLOCK_FLOAT16,),
                                                                            name="grouped_extrim_distance_ub_fp16",
                                                                            scope=tik.scope_ubuf)
            index_offset.set_as(thread_loop * Constant.SLICE_SIZE)
            self.tik_instance.vector_dup(Constant.MASK_FLOAT16,
                                         self.ivf_cur_process_ub_uint8.reinterpret_cast_to("float16"), 0,
                                         Constant.IVF_SLICE_SIZE // 2 // Constant.MASK_FLOAT16, 1, 1)
            args = (bucket_offset_input, thread_tail * Constant.IVF_UNIT_LEN, Constant.INNER_LOOP_TIME, thread_loop)
            self._handle_input_data(args)
            self.tik_instance.vadds(Constant.MASK_FLOAT16, self.pq_distance_ub_fp16, self.pq_distance_ub_fp16,
                                    bucket_base_dis, Constant.SLICE_SIZE // Constant.MASK_FLOAT16, 1, 1, 8, 8)
            extreme_value = self.tik_instance.Scalar("float16", name="extreme_value")
            if self.extreme_mode == 1:
                extreme_value.set_as(Constant.MIN_FP16)
            else:
                extreme_value.set_as(Constant.MAX_FP16)
            with self.tik_instance.for_range(0, Constant.BLOCK_FLOAT16 - thread_tail % Constant.BLOCK_FLOAT16) as idx:
                self.pq_distance_ub_fp16[thread_tail + idx].set_as(extreme_value)
            with self.tik_instance.if_scope(
                (Constant.SLICE_SIZE - _ceil_fill(thread_tail, Constant.BLOCK_FLOAT16)) // Constant.BLOCK_FLOAT16 > 0):
                self.tik_instance.vector_dup(
                    Constant.BLOCK_FLOAT16, self.pq_distance_ub_fp16[_ceil_fill(thread_tail,
                                                                                Constant.BLOCK_FLOAT16)], extreme_value,
                    (Constant.SLICE_SIZE - _ceil_fill(thread_tail, Constant.BLOCK_FLOAT16)) // Constant.BLOCK_FLOAT16,
                    1, 1)
            args_dis = (bucket_offset_output, bucket_offset_max, Constant.SLICE_SIZE)
            self._handle_pq_distance(args_dis)
            pq_ivf_ub_int32 = self.ivf_cur_process_ub_uint8.reinterpret_cast_to("int32")
            self.tik_instance.vector_dup(Constant.MASK_FLOAT32, pq_ivf_ub_int32, bucket_id,
                                         Constant.SLICE_SIZE // Constant.MASK_FLOAT32, 1, 8)
            # index set by assistant cube for performance
            pq_index_ub_fp32 = self.ivf_cur_process_ub_fp32
            self.tik_instance.vadds(Constant.MASK_FLOAT32, pq_index_ub_fp32, self.assist_pq_index_init_ub_fp32,
                                    index_offset, Constant.SLICE_SIZE // Constant.MASK_FLOAT32, 1, 1, 8, 8)
            pq_index_ub_int32 = pq_index_ub_fp32.reinterpret_cast_to("int32")
            self.tik_instance.vconv(Constant.MASK_FLOAT32, "round", pq_index_ub_int32, pq_index_ub_fp32,
                                    Constant.SLICE_SIZE // Constant.MASK_FLOAT32, 1, 1, 8, 8)
            self.tik_instance.data_move(self.pq_index_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_loop],
                                        pq_index_ub_int32, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_INT32, 0, 0)
            self.tik_instance.data_move(
                self.grouped_extrim_distance_gm[bucket_offset_max +
                                                (Constant.SLICE_SIZE // self.group_size) * thread_loop],
                self.grouped_extrim_distance_ub_fp16, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.pq_distance_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_loop],
                                        self.pq_distance_ub_fp16, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_FLOAT16,
                                        0, 0)
            self.tik_instance.data_move(self.pq_ivf_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_loop],
                                        pq_ivf_ub_int32, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_INT32, 0, 0)


def _para_dtype_check(args_list):
    (ivf, bucket_list, bucket_base_distance, bucket_limits, bucket_offsets, adc_tables, actual_count, pq_distance,
     grouped_extrim_distance, pq_ivf, pq_index) = args_list
    # input
    ivf_dtype = ivf.get("dtype").lower()
    bucket_list_dtype = bucket_list.get("dtype").lower()
    bucket_base_distance_dtype = bucket_base_distance.get("dtype").lower()
    bucket_limits_dtype = bucket_limits.get("dtype").lower()
    bucket_offsets_dtype = bucket_offsets.get("dtype").lower()
    adc_tables_dtype = adc_tables.get("dtype").lower()
    para_check.check_dtype(ivf_dtype, ("uint8"), param_name="ivf")
    para_check.check_dtype(bucket_list_dtype, ("int32"), param_name="bucket_list")
    para_check.check_dtype(bucket_base_distance_dtype, ("float16"), param_name="bucket_base_distance")
    para_check.check_dtype(bucket_limits_dtype, ("int32"), param_name="bucket_limits")
    para_check.check_dtype(bucket_offsets_dtype, ("int64"), param_name="bucket_offsets")
    para_check.check_dtype(adc_tables_dtype, ("float16"), param_name="adc_tables")

    para_check.check_shape(ivf.get("shape"), min_rank=2, max_rank=2, param_name="ivf")
    para_check.check_shape(bucket_list.get("shape"), min_rank=1, max_rank=1, param_name="bucket_list")
    para_check.check_shape(bucket_base_distance.get("shape"), min_rank=1, max_rank=1,
                           param_name="bucket_base_distance")
    para_check.check_shape(bucket_limits.get("shape"), min_rank=1, max_rank=1, param_name="bucket_limits")
    para_check.check_shape(bucket_offsets.get("shape"), min_rank=1, max_rank=1, param_name="bucket_offsets")
    para_check.check_shape(adc_tables.get("shape"), min_rank=3, max_rank=3, param_name="adc_tables")

    # output
    actual_count_dtype = actual_count.get("dtype").lower()
    pq_distance_dtype = pq_distance.get("dtype").lower()
    grouped_extrim_distance_dtype = grouped_extrim_distance.get("dtype").lower()
    pq_ivf_dtype = pq_ivf.get("dtype").lower()
    pq_index_dtype = pq_index.get("dtype").lower()
    para_check.check_dtype(actual_count_dtype, ("int32"), param_name="actual_count")
    para_check.check_dtype(pq_distance_dtype, ("float16"), param_name="pq_distance")
    para_check.check_dtype(grouped_extrim_distance_dtype, ("float16"), param_name="grouped_extrim_distance")
    para_check.check_dtype(pq_ivf_dtype, ("int32"), param_name="pq_ivf")
    para_check.check_dtype(pq_index_dtype, ("int32"), param_name="pq_index")


@register_operator("ScanPQCodes")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def scan_pq_codes(ivf,
                  bucket_list,
                  bucket_base_distance,
                  bucket_limits,
                  bucket_offsets,
                  adc_tables,
                  actual_count,
                  pq_distance,
                  grouped_extrim_distance,
                  pq_ivf,
                  pq_index,
                  total_limit,
                  group_size,
                  extreme_mode,
                  split_count,
                  split_index,
                  kernel_name="scan_pq_codes"):
    """
    scan_pq_codes
    """
    args_list = (ivf, bucket_list, bucket_base_distance, bucket_limits, bucket_offsets, adc_tables, actual_count,
                 pq_distance, grouped_extrim_distance, pq_ivf, pq_index)

    _para_dtype_check(args_list)
    if any((group_size is None, group_size == 0)):
        error_manager_vector.raise_err_specific_reson(kernel_name, "group size should be nonzero value")

    ivf_dtype = ivf.get("dtype").lower()
    bucket_list_dtype = bucket_list.get("dtype").lower()
    bucket_base_distance_dtype = bucket_base_distance.get("dtype").lower()
    bucket_limits_dtype = bucket_limits.get("dtype").lower()
    bucket_offsets_dtype = bucket_offsets.get("dtype").lower()
    adc_tables_dtype = adc_tables.get("dtype").lower()
    adc_tables_shape = adc_tables.get("shape")
    if len(adc_tables_shape) < 2:
        error_manager_vector.raise_err_specific_reson(kernel_name, "adc_tables shape is invalid")
    ivf_dim = adc_tables_shape[Constant.ADC_DIM]
    if ivf_dim == 0:
        error_manager_vector.raise_err_specific_reson(kernel_name, "ivf_dim should be nonzero value")
    dtypes = (ivf_dtype, bucket_list_dtype, bucket_base_distance_dtype, bucket_limits_dtype, bucket_offsets_dtype,
              adc_tables_dtype)
    if ivf_dim == Constant.IVF_UNIT_LEN:
        attrs = (total_limit, group_size, extreme_mode, split_count, split_index)
        obj = ScanPQCodes(attrs, dtypes)
    else:
        attrs = (total_limit, group_size, extreme_mode, split_count, split_index, ivf_dim)
        obj = ScanPQCodesExpand(attrs, dtypes)
    return obj.scan_pq_codes_operator(kernel_name)
