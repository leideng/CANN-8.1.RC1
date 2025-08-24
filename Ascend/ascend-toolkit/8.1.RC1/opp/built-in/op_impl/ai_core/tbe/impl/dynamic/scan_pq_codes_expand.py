# Copyright 2020 Huawei Technologies Co., Ltd
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
scan_pq_codes
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context


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
    MASK_INT32 = 64
    BLOCK_INT64 = 4
    BLOCK_INT32 = 8
    BLOCK_FLOAT16 = 16
    BLOCK_UINT8 = 32
    MAX_BUCKET_LEN = 64
    IVF_INNER_LOOP_LEN = 4096
    SLICE_SIZE = 1024
    ADC_DIM_0 = 256
    ADC_SLICE = 64
    DOUBLE = 2
    TAIL = 24
    IDX_0 = 0
    IDX_1 = 1
    IDX_2 = 2
    IDX_3 = 3
    IDX_4 = 4
    IDX_5 = 5
    IDX_6 = 6
    IDX_7 = 7


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
class ScanPQCodesExpand():
    """
    ScanPQCodesExpand
    """
    def __init__(self, attrs, dtypes):
        """
        Function: use to store ScanPQCodes base parameters
        """
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
        # attrs
        (total_limit, group_size, extreme_mode, split_count, split_index, ivf_dim) = attrs
        self.group_size = group_size
        self.total_limit = total_limit
        self.extreme_mode = extreme_mode
        self.split_count = split_count
        self.split_index = split_index
        self.ivf_dim = ivf_dim
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
        self.assist_add_init_ub_fp32 = None
        self.assist_pq_index_init_ub_int32 = None
        self.pq_distance_ub_fp16 = None
        self.grouped_extrim_distance_ub_fp16 = None
        self.bucket_list_ub_int32 = None
        self.bucket_base_distance_ub_fp16 = None
        self.bucket_limits_ub_int32 = None
        self.bucket_offsets_ub_int64 = None
        # loop param
        self.slice_inner_size = Constant.IVF_INNER_LOOP_LEN // self.ivf_dim
        self.ivf_slice_size = Constant.SLICE_SIZE * self.ivf_dim
        self.inner_dim_size = self.slice_inner_size * Constant.BLOCK_FLOAT16

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
        input_list_ex = [
            self.ivf_gm, self.bucket_list_gm, self.bucket_base_distance_gm, self.bucket_limits_gm,
            self.bucket_offsets_gm, self.adc_tables_gm
        ]
        output_list_ex = [
            self.actual_count_gm, self.pq_distance_gm,
            self.grouped_extrim_distance_gm, self.pq_ivf_gm, self.pq_index_gm
        ]
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=input_list_ex,
                                   outputs=output_list_ex,
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
        self.adc_tables_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                           (Constant.ADC_DIM_0, Constant.ADC_DIM_0),
                                                           name="adc_tables_ub_fp16",
                                                           scope=tik.scope_ubuf)
        #assist ub
        self.assist_add_init_ub_fp32 = self.tik_instance.Tensor("float32",
            (Constant.MASK_INT32 * Constant.DOUBLE + Constant.TAIL,),
            name="assist_add_init_ub_fp32", scope=tik.scope_ubuf)
        self.assist_pq_index_init_ub_int32 = self.tik_instance.Tensor("int32", (Constant.SLICE_SIZE,),
                                                                      name="assist_pq_index_init_ub_int32",
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
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                        Constant.TILING_ARG_NUM // Constant.BLOCK_INT64, 0, 0)
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
        # conver adc_tables shape from (256,x) to (256,16,16) to prevent bank conflict, x is 32 or 64
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            adc_input_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                         (1, self.ivf_dim, 1, Constant.ADC_DIM_0),
                                                         name="adc_input_ub_fp16",
                                                         scope=tik.scope_ubuf)
            adc_trans_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                         (1, 1, Constant.ADC_DIM_0, self.ivf_dim),
                                                         name="adc_trans_ub_fp16",
                                                         scope=tik.scope_ubuf)
            adc_exp_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                       (Constant.ADC_SLICE, Constant.ADC_DIM_0),
                                                       name="adc_trans_ub_fp16",
                                                       scope=tik.scope_ubuf)
            adc_offset = (bucket_idx + self.bucket_start_base) * Constant.ADC_DIM_0 * self.ivf_dim
            self.tik_instance.data_move(adc_input_ub_fp16,
                                        self.adc_tables_gm[adc_offset],
                                        0, 1, Constant.ADC_DIM_0 * self.ivf_dim // Constant.BLOCK_FLOAT16, 0, 0)
            self.tik_instance.v4dtrans(True, adc_trans_ub_fp16[0, 0, 0, 0],
                                       adc_input_ub_fp16[0, 0, 0, 0], Constant.ADC_DIM_0, self.ivf_dim)
            with self.tik_instance.for_range(0, Constant.ADC_DIM_0 // Constant.ADC_SLICE) as idx:
                ivf_dim_floor = _floor_fill(self.ivf_dim, Constant.MASK_FLOAT16)
                with self.tik_instance.if_scope(self.ivf_dim // Constant.MASK_FLOAT16 > 0):
                    self.tik_instance.vadds(Constant.MASK_FLOAT16, adc_exp_ub_fp16,
                                            adc_trans_ub_fp16[Constant.ADC_SLICE * self.ivf_dim * idx],
                                            0, Constant.ADC_SLICE, 1, 1, Constant.BLOCK_FLOAT16,
                                            self.ivf_dim // Constant.BLOCK_FLOAT16)
                with self.tik_instance.if_scope(self.ivf_dim % Constant.MASK_FLOAT16 > 0):
                    self.tik_instance.vadds(self.ivf_dim, adc_exp_ub_fp16[ivf_dim_floor],
                                            adc_trans_ub_fp16[Constant.ADC_SLICE * self.ivf_dim * idx + ivf_dim_floor],
                                            0, Constant.ADC_SLICE, 1, 1, Constant.BLOCK_FLOAT16,
                                            self.ivf_dim // Constant.BLOCK_FLOAT16)
                self.tik_instance.vec_trans(self.adc_tables_ub_fp16[Constant.ADC_SLICE * Constant.ADC_DIM_0 * idx],
                                            adc_exp_ub_fp16, Constant.ADC_SLICE, 1, 1)

    def _init_assist_ub(self):
        assist_init_ub = self.tik_instance.Tensor("float32", (self.ivf_dim,),
                                                  name="assist_init_ub",
                                                  scope=tik.scope_ubuf)
        assist_init_ub_int32 = self.tik_instance.Tensor("int32", (self.ivf_dim,),
                                                        name="assist_init_ub_int32",
                                                        scope=tik.scope_ubuf)
        assist_value = 0
        assist_init_ub[Constant.IDX_0].set_as(assist_value + Constant.IDX_0)
        assist_init_ub[Constant.IDX_1].set_as(assist_value + Constant.IDX_1)
        assist_init_ub[Constant.IDX_2].set_as(assist_value + Constant.IDX_2)
        assist_init_ub[Constant.IDX_3].set_as(assist_value + Constant.IDX_3)
        assist_init_ub[Constant.IDX_4].set_as(assist_value + Constant.IDX_4)
        assist_init_ub[Constant.IDX_5].set_as(assist_value + Constant.IDX_5)
        assist_init_ub[Constant.IDX_6].set_as(assist_value + Constant.IDX_6)
        assist_init_ub[Constant.IDX_7].set_as(assist_value + Constant.IDX_7)
        self.tik_instance.vadds(Constant.BLOCK_INT32, assist_init_ub[Constant.BLOCK_INT32], assist_init_ub,
                                Constant.BLOCK_INT32, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.BLOCK_FLOAT16, self.assist_add_init_ub_fp32, assist_init_ub,
                                0, 1, 1, 1, Constant.DOUBLE, Constant.DOUBLE)
        self.tik_instance.vmuls(Constant.BLOCK_FLOAT16, self.assist_add_init_ub_fp32, self.assist_add_init_ub_fp32,
                                Constant.BLOCK_UINT8, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.BLOCK_FLOAT16, self.assist_add_init_ub_fp32[Constant.BLOCK_FLOAT16],
                                self.assist_add_init_ub_fp32, 0, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.BLOCK_UINT8, self.assist_add_init_ub_fp32[Constant.BLOCK_UINT8],
                                self.assist_add_init_ub_fp32, 0, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.MASK_INT32, self.assist_add_init_ub_fp32[Constant.MASK_INT32],
                                self.assist_add_init_ub_fp32, 0, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vconv(Constant.BLOCK_FLOAT16, "floor", assist_init_ub_int32,
                                assist_init_ub, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.BLOCK_FLOAT16, self.assist_pq_index_init_ub_int32, assist_init_ub_int32,
                                0, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.BLOCK_FLOAT16, self.assist_pq_index_init_ub_int32[Constant.BLOCK_FLOAT16],
                                self.assist_pq_index_init_ub_int32,
                                Constant.BLOCK_FLOAT16, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.BLOCK_UINT8, self.assist_pq_index_init_ub_int32[Constant.BLOCK_UINT8],
                                self.assist_pq_index_init_ub_int32,
                                Constant.BLOCK_UINT8, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.MASK_INT32, self.assist_pq_index_init_ub_int32[Constant.MASK_INT32],
                                self.assist_pq_index_init_ub_int32,
                                Constant.MASK_INT32, 1, 1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.MASK_INT32, self.assist_pq_index_init_ub_int32[Constant.MASK_FLOAT16],
                                self.assist_pq_index_init_ub_int32,
                                Constant.MASK_FLOAT16, Constant.DOUBLE, 1, 1,
                                Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.MASK_INT32, self.assist_pq_index_init_ub_int32[Constant.ADC_DIM_0],
                                self.assist_pq_index_init_ub_int32,
                                Constant.ADC_DIM_0, Constant.BLOCK_INT64, 1, 1, Constant.BLOCK_INT32,
                                Constant.BLOCK_INT32)
        self.tik_instance.vadds(Constant.MASK_INT32,
                                self.assist_pq_index_init_ub_int32[Constant.ADC_DIM_0 * Constant.DOUBLE],
                                self.assist_pq_index_init_ub_int32,
                                Constant.ADC_DIM_0 * Constant.DOUBLE, Constant.BLOCK_INT32, 1, 1, Constant.BLOCK_INT32,
                                Constant.BLOCK_INT32)

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
        bucket_offset_input.set_as(bucket_offset_input * self.ivf_dim)
        self._calc_output_count(bucket_offset_output, bucket_idx)
        bucket_max_offset.set_as(bucket_offset_output // self.group_size)

    def _run_multi_core(self):
        with self.tik_instance.for_range(0, self.core_nums, block_num=self.core_nums) as core_idx:
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
        (bucket_offset_input, ivf_slice_size, thread_idx, count_idx) = args
        # input data
        vadds_result_ub_int32 = self.tik_instance.Tensor("int32", (Constant.IVF_INNER_LOOP_LEN,),
                                                         name="vadds_result_ub_int32",
                                                         scope=tik.scope_ubuf)
        vadds_result_ub_fp32 = vadds_result_ub_int32.reinterpret_cast_to("float32")
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            ivf_cur_process_ub_fp32 = self.tik_instance.Tensor("float32", (Constant.IVF_INNER_LOOP_LEN,),
                                                                name="ivf_cur_process_ub_fp32",
                                                                scope=tik.scope_ubuf)
            ivf_cur_process_ub_uint8 = vadds_result_ub_int32.reinterpret_cast_to(self.ivf_dtype)
            ivf_cur_process_ub_fp16 = ivf_cur_process_ub_fp32.reinterpret_cast_to(self.adc_tables_dtype)
            with self.tik_instance.if_scope(ivf_slice_size < Constant.IVF_INNER_LOOP_LEN):
                self.tik_instance.vector_dup(Constant.MASK_FLOAT16,
                                             ivf_cur_process_ub_uint8.reinterpret_cast_to("float16"),
                                             0, Constant.IVF_INNER_LOOP_LEN // Constant.MASK_FLOAT16 // 2, 1, 1)
            ivf_offset = bucket_offset_input + thread_idx * self.ivf_slice_size + \
                         Constant.IVF_INNER_LOOP_LEN * count_idx
            self.tik_instance.data_move(ivf_cur_process_ub_uint8,
                                        self.ivf_gm[ivf_offset], 0, 1,
                                        _ceil_div(ivf_slice_size, Constant.BLOCK_UINT8), 0, 0)
            # ivf reprocess for vgather, coordination = (ivf * 256 + offset) * 2
            self.tik_instance.vconv(Constant.MASK_FLOAT16, "", ivf_cur_process_ub_fp16,
                                    ivf_cur_process_ub_uint8,
                                    Constant.IVF_INNER_LOOP_LEN // Constant.MASK_FLOAT16, 1, 1, 8, 4)
            self.tik_instance.vconv(Constant.MASK_INT32, "", vadds_result_ub_fp32,
                                    ivf_cur_process_ub_fp16,
                                    Constant.IVF_INNER_LOOP_LEN // Constant.MASK_INT32, 1, 1, 8, 4)
            self.tik_instance.vmuls(Constant.MASK_INT32, ivf_cur_process_ub_fp32, vadds_result_ub_fp32, 512,
                                    Constant.IVF_INNER_LOOP_LEN // Constant.MASK_INT32, 1, 1, 8, 8)
            self.tik_instance.vadd(Constant.MASK_INT32, vadds_result_ub_fp32, ivf_cur_process_ub_fp32,
                                    self.assist_add_init_ub_fp32,
                                    Constant.IVF_INNER_LOOP_LEN // 2 // Constant.MASK_INT32, 1, 1, 1, 16, 16, 0)
            self.tik_instance.vadd(Constant.MASK_INT32, vadds_result_ub_fp32[Constant.MASK_INT32],
                                    ivf_cur_process_ub_fp32[Constant.MASK_INT32],
                                    self.assist_add_init_ub_fp32[Constant.MASK_INT32],
                                    Constant.IVF_INNER_LOOP_LEN // 2 // Constant.MASK_INT32, 1, 1, 1, 16, 16, 0)
            with self.tik_instance.for_range(0, self.ivf_dim // Constant.BLOCK_FLOAT16) as dim_idx:
                self.tik_instance.vadds(Constant.BLOCK_FLOAT16, ivf_cur_process_ub_fp32[dim_idx * self.inner_dim_size],
                                        vadds_result_ub_fp32[dim_idx * Constant.BLOCK_FLOAT16],
                                        dim_idx * 2, Constant.IVF_INNER_LOOP_LEN // self.ivf_dim,
                                        1, 1, Constant.BLOCK_FLOAT16 // Constant.BLOCK_INT32,
                                        self.ivf_dim // Constant.BLOCK_INT32)
            self.tik_instance.vconv(Constant.MASK_INT32, "floor", vadds_result_ub_int32,
                                    ivf_cur_process_ub_fp32,
                                    Constant.IVF_INNER_LOOP_LEN // Constant.MASK_INT32,
                                    1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
        vgather_result_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                          (self.inner_dim_size,),
                                                          name="vgather_result_ub_fp16", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # distance
            dim_buffer_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                          (Constant.IVF_INNER_LOOP_LEN,),
                                                          name="dim_buffer_ub_fp16",
                                                          scope=tik.scope_ubuf)
            self.tik_instance.vgather(Constant.MASK_FLOAT16, dim_buffer_ub_fp16, self.adc_tables_ub_fp16,
                                      vadds_result_ub_int32, Constant.IVF_INNER_LOOP_LEN // Constant.MASK_FLOAT16,
                                      Constant.BLOCK_INT32, 0, 0, "normal")
            self.tik_instance.vadds(Constant.MASK_FLOAT16, vgather_result_ub_fp16, dim_buffer_ub_fp16,
                                    0, self.inner_dim_size // Constant.MASK_FLOAT16,
                                    1, 1, Constant.BLOCK_INT32, Constant.BLOCK_INT32)
            with self.tik_instance.for_range(1, self.ivf_dim // Constant.BLOCK_FLOAT16) as dim_idx:
                self.tik_instance.vadd(Constant.MASK_FLOAT16, vgather_result_ub_fp16, vgather_result_ub_fp16,
                                       dim_buffer_ub_fp16[dim_idx * self.inner_dim_size],
                                       self.inner_dim_size // Constant.MASK_FLOAT16, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vcgadd(Constant.MASK_FLOAT16,
                                 self.pq_distance_ub_fp16[count_idx * self.slice_inner_size],
                                 vgather_result_ub_fp16, self.inner_dim_size // Constant.MASK_FLOAT16, 1,
                                 1, 8)

    def _handle_pq_distance(self, args):
        (_, _, slice_size) = args
        # extrim
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            block_extrim_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (Constant.BLOCK_FLOAT16 * 2,),
                                                            name="block_extrim_ub_fp16",
                                                            scope=tik.scope_ubuf)
            if self.extreme_mode == 1:
                self.tik_instance.vcmax(self.group_size, block_extrim_ub_fp16, self.pq_distance_ub_fp16,
                                        slice_size // self.group_size, 1, 1, self.group_size // Constant.BLOCK_FLOAT16)
            else:
                self.tik_instance.vcmin(self.group_size, block_extrim_ub_fp16, self.pq_distance_ub_fp16,
                                        slice_size // self.group_size, 1, 1, self.group_size // Constant.BLOCK_FLOAT16)
            self.tik_instance.vreduce((slice_size // self.group_size) * 2, self.grouped_extrim_distance_ub_fp16,
                                    block_extrim_ub_fp16, 1, 1, 1, 1, 0, 0, None, "counter")

    def _run_one_core_loop(self, args_ex):
        (bucket_idx, bucket_id, bucket_base_dis, bucket_limit, bucket_offset_input, bucket_offset_output,
         bucket_offset_max) = args_ex
        self._create_adc_table(bucket_idx)
        thread_loop = self.tik_instance.Scalar("int32", name="thread_loop")
        thread_tail = self.tik_instance.Scalar("int32", name="thread_tail")
        index_offset = self.tik_instance.Scalar("int32", name="index_offset")
        thread_loop.set_as(bucket_limit // Constant.SLICE_SIZE)
        thread_tail.set_as(bucket_limit % Constant.SLICE_SIZE)
        with self.tik_instance.for_range(0, thread_loop, thread_num=2) as thread_idx:
            self.pq_distance_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (Constant.SLICE_SIZE,),
                                                                name="pq_distance_ub_fp16",
                                                                scope=tik.scope_ubuf)
            self.grouped_extrim_distance_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                                            (Constant.BLOCK_FLOAT16,),
                                                                            name="grouped_extrim_distance_ub_fp16",
                                                                            scope=tik.scope_ubuf)
            index_offset.set_as(thread_idx * Constant.SLICE_SIZE)
            inner_loop_time = Constant.SLICE_SIZE // self.slice_inner_size
            with self.tik_instance.for_range(0, inner_loop_time) as count_idx:
                args = (bucket_offset_input, Constant.IVF_INNER_LOOP_LEN, thread_idx, count_idx)
                self._handle_input_data(args)
            self.tik_instance.vadds(Constant.MASK_FLOAT16, self.pq_distance_ub_fp16, self.pq_distance_ub_fp16,
                                    bucket_base_dis, Constant.SLICE_SIZE // Constant.MASK_FLOAT16, 1, 1, 8, 8)
            args_dis = (bucket_offset_output, bucket_offset_max, Constant.SLICE_SIZE)
            self._handle_pq_distance(args_dis)
            # index set by assistant cube for performance
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                pq_index_ub_int32 = self.tik_instance.Tensor("int32", (Constant.SLICE_SIZE,),
                                                             name="pq_index_ub_int32", scope=tik.scope_ubuf)
                self.tik_instance.vadds(Constant.MASK_INT32, pq_index_ub_int32, self.assist_pq_index_init_ub_int32,
                                        index_offset, Constant.SLICE_SIZE // Constant.MASK_INT32, 1, 1, 8, 8)
                self.tik_instance.data_move(self.pq_index_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_idx],
                                            pq_index_ub_int32, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_INT32, 0, 0)
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                pq_ivf_ub_int32 = self.tik_instance.Tensor("int32", (Constant.SLICE_SIZE,),
                                                           name="pq_ivf_ub_int32", scope=tik.scope_ubuf)
                self.tik_instance.vector_dup(Constant.MASK_INT32, pq_ivf_ub_int32, bucket_id,
                                             Constant.SLICE_SIZE // Constant.MASK_INT32, 1, 8)
                self.tik_instance.data_move(self.pq_ivf_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_idx],
                                            pq_ivf_ub_int32, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_INT32, 0, 0)
            max_offset = bucket_offset_max + (Constant.SLICE_SIZE // self.group_size) * thread_idx
            self.tik_instance.data_move(self.grouped_extrim_distance_gm[max_offset],
                                        self.grouped_extrim_distance_ub_fp16, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.pq_distance_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_idx],
                                        self.pq_distance_ub_fp16, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_FLOAT16,
                                        0, 0)
        with self.tik_instance.if_scope(thread_tail > 0):
            self.pq_distance_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype, (Constant.SLICE_SIZE,),
                                                                name="pq_distance_ub_fp16",
                                                                scope=tik.scope_ubuf)
            self.grouped_extrim_distance_ub_fp16 = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                                            (Constant.BLOCK_FLOAT16,),
                                                                            name="grouped_extrim_distance_ub_fp16",
                                                                            scope=tik.scope_ubuf)
            index_offset.set_as(thread_loop * Constant.SLICE_SIZE)
            inner_loop_time = _ceil_div(thread_tail, self.slice_inner_size)
            with self.tik_instance.for_range(0, inner_loop_time) as count_idx:
                with self.tik_instance.if_scope(count_idx < inner_loop_time - 1):
                    args = (bucket_offset_input, Constant.IVF_INNER_LOOP_LEN, thread_loop, count_idx)
                    self._handle_input_data(args)
                with self.tik_instance.if_scope(count_idx == inner_loop_time - 1):
                    count_left = thread_tail % self.slice_inner_size
                    with self.tik_instance.if_scope(count_left > 0):
                        args = (bucket_offset_input, count_left * self.ivf_dim, thread_loop, count_idx)
                        self._handle_input_data(args)
                    with self.tik_instance.else_scope():
                        args = (bucket_offset_input, Constant.IVF_INNER_LOOP_LEN, thread_loop, count_idx)
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
            pq_left = (Constant.SLICE_SIZE - _ceil_fill(thread_tail, Constant.BLOCK_FLOAT16)) // Constant.BLOCK_FLOAT16
            with self.tik_instance.if_scope(pq_left > 0):
                self.tik_instance.vector_dup(Constant.BLOCK_FLOAT16,
                                             self.pq_distance_ub_fp16[_ceil_fill(thread_tail, Constant.BLOCK_FLOAT16)],
                                             extreme_value, pq_left, 1, 1)
            args_dis = (bucket_offset_output, bucket_offset_max, Constant.SLICE_SIZE)
            self._handle_pq_distance(args_dis)
            # index set by assistant cube for performance
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                pq_index_ub_int32 = self.tik_instance.Tensor("int32", (Constant.SLICE_SIZE,),
                                                             name="pq_index_ub_int32", scope=tik.scope_ubuf)
                self.tik_instance.vadds(Constant.MASK_INT32, pq_index_ub_int32, self.assist_pq_index_init_ub_int32,
                                        index_offset, Constant.SLICE_SIZE // Constant.MASK_INT32, 1, 1, 8, 8)
                self.tik_instance.data_move(self.pq_index_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_loop],
                                            pq_index_ub_int32, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_INT32, 0, 0)
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                pq_ivf_ub_int32 = self.tik_instance.Tensor("int32", (Constant.SLICE_SIZE,),
                                                           name="pq_ivf_ub_int32", scope=tik.scope_ubuf)
                self.tik_instance.vector_dup(Constant.MASK_INT32, pq_ivf_ub_int32, bucket_id,
                                            Constant.SLICE_SIZE // Constant.MASK_INT32, 1, 8)
                self.tik_instance.data_move(self.pq_ivf_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_loop],
                                            pq_ivf_ub_int32, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_INT32, 0, 0)
            max_offset = bucket_offset_max + (Constant.SLICE_SIZE // self.group_size) * thread_loop
            self.tik_instance.data_move(self.grouped_extrim_distance_gm[max_offset],
                                        self.grouped_extrim_distance_ub_fp16, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.pq_distance_gm[bucket_offset_output + Constant.SLICE_SIZE * thread_loop],
                                        self.pq_distance_ub_fp16, 0, 1, Constant.SLICE_SIZE // Constant.BLOCK_FLOAT16,
                                        0, 0)
