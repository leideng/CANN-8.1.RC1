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
lru
"""
# 'pylint: disable=unused-argument,too-many-arguments,invalid-name,no-self-use,too-many-branches
# 'pylint: disable=too-many-instance-attributes,unnecessary-comprehension,inconsistent-return-statements
import math
from impl.util.util_tik_comm_func import OpBase
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from tbe.common.platform import get_bit_len


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    FP32_VECTOR_MASK_MAX = 64
    BLOCK_BYTES = 32
    SMMU_ID = 0
    DATA_MOV_STRIDE = 0
    DATA_MOV_NBURST = 1
    # vector instr stride 8 block
    VEC_STRIDE = 8
    # ub init all 0 value
    INIT_ZERO = 0
    # int32 in 32B
    ONE_BLK_INT32_NUMS = 8
    # int64 in 32B
    ONE_BLK_INT64_NUMS = 4
    # `1 byte = 8 bit`
    BYTE_BITS = 8
    TILING_NUMS = 8
    MODE0 = 0
    FP32_MIN_VALUE = 0.00001
    THREAD_NUM = 4
    INT32_BYTES = 4

# 'pylint: disable=no-member,attribute-defined-outside-init,dangerous-default-value,consider-using-enumerate


class Lru(OpBase):
    """
       Function: use to store  base parameters
       Modify : 2021-07-09
    """

    # 'pylint: disable=too-many-statements,too-many-locals
    def __init__(self, index_list, data, cache, tag, is_last_call, out_data, out_cache, out_tag, index_offset_list,
                 not_in_cache_index_list, not_in_cache_number, pre_route_count, kernel_name):
        OpBase.__init__(self)
        self.index_list_dtype = index_list.get("dtype").lower()
        self.data_shape = data.get("shape")
        self.cache_shape = cache.get("shape")
        self.cache_size = self.cache_shape[-1]
        self.cache_dtype = cache.get("dtype").lower()
        self.data_dtype = data.get("dtype").lower()
        self.tag_shape = tag.get("shape")
        self.tag_dtype = tag.get("dtype").lower()
        self.is_last_call_dtype = is_last_call.get("dtype").lower()
        self.time_stamp_dtype = "float32"
        self.kernel_name = kernel_name
        # way_number can choose 32,64,96,128,256
        self.way_number = 64
        self.embedding_size = self.data_shape[-1]
        self.set_number = self.cache_size // self.embedding_size // self.way_number
        self.set_number_ceil_64 = (self.set_number + Constant.FP32_VECTOR_MASK_MAX -
                                   1) // Constant.FP32_VECTOR_MASK_MAX * Constant.FP32_VECTOR_MASK_MAX
        self.pre_route_number = pre_route_count
        # input param check
        self.check_param()
        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.aicore_num = 8  # only use 8 cores
        self.embedding_bytes = self.embedding_size * \
            get_bit_len(self.data_dtype) // Constant.BYTE_BITS
        self.tiling_dtype = "int64"
        self.tiling_shape = (Constant.TILING_NUMS,)
        # cal every core process sets
        self.index_list_ele_per_block = self._get_ele_block(self.index_list_dtype)
        self.tag_ele_per_block = self._get_ele_block(self.tag_dtype)
        self.tag_move_blocks = (self.way_number + self.tag_ele_per_block - 1) // self.tag_ele_per_block
        self.sorted_index_shape = self.way_number
        self.sorted_index_blocks = int(self.sorted_index_shape / Constant.ONE_BLK_INT32_NUMS)
        # The number of times that each core data cycle is moved in
        self.index_num_per_loop = 1024
        self.index_list_loop_times = None
        self.index_num_tail_loop = None
        # The blocks that need to be calculated in each cycle
        self.one_set_size = self.way_number * \
            get_bit_len(self.tag_dtype) // Constant.BYTE_BITS
        # The blocks to be calculated in the last loop
        self.index_list_size = self.index_num_per_loop * \
            get_bit_len(self.index_list_dtype) // Constant.BYTE_BITS
        self.remain_ub_size = self.total_ub - self.index_list_size
        self.tag_set_nums_per_loop = 0
        self.tag_rate_int32 = get_bit_len(self.tag_dtype) // 32
        self.index_list_bytes = get_bit_len(self.index_list_dtype) // Constant.BYTE_BITS
        self.time_stamp_bytes = get_bit_len(self.time_stamp_dtype) // Constant.BYTE_BITS
        self.tag_bytes = get_bit_len(self.tag_dtype) // Constant.BYTE_BITS
        # The number of elements to be calculated in each loop
        self.vector_mask_max = 0
        self.blk_stride = 0
        self.dst_rep_stride = 0
        self.src_rep_stride = 0
        self.index_vand_ub = None
        self.vcmax_scalar_max = None
        self.vcmax_scalar_cnt = None
        self.vcmax_scalar_index = None
        self.not_in_cache_count_ub = None
        self.index_offset_ub = None
        self.miss_index_wsp = self.tik_instance.Tensor(self.tag_dtype, self.unknown_max_shape, name="miss_index_wsp",
                                                       is_workspace=True, scope=tik.scope_gm)
        # the last is iterate timestamp
        self.time_stamp_wsp = self.tik_instance.Tensor(self.time_stamp_dtype, [self.set_number * self.way_number + 1],
                                                       name="time_stamp_wsp", is_global_tensor=True, is_atomic_add=True,
                                                       scope=tik.scope_gm)
        # vsort32 use
        sorted_index_init_list = [i for i in range(0, self.sorted_index_shape)]
        self.sorted_index_gm = self.tik_instance.Tensor("uint32", [self.sorted_index_shape],
                                                        name="sorted_index_gm", scope=tik.scope_gm,
                                                        init_value=sorted_index_init_list)
        self.sorted_index_ub = self.tik_instance.Tensor("uint32", [self.sorted_index_shape],
                                                        name="sorted_index_ub", scope=tik.scope_ubuf)
        # position_index help
        position_index_list = [
                               i
                               for i in range(self.way_number)
                               for j in range(self.tag_rate_int32)
        ]
        self.position_index_gm = self.tik_instance.Tensor("uint16", [self.way_number * self.tag_rate_int32],
                                                          name="position_index_gm",
                                                          scope=tik.scope_gm,
                                                          init_value=position_index_list)
        self.position_index_ub = self.tik_instance.Tensor("uint16", [self.way_number * self.tag_rate_int32],
                                                          name="position_index_ub",
                                                          scope=tik.scope_ubuf)

    @staticmethod
    def is_power(k):
        """
        input is or not 2**n
        """
        if k < 1:
            return False
        m = k & (k - 1)
        return m == 0

    @staticmethod
    def _get_ele_block(dtype):
        """
        get this dtype block num
        """
        return Constant.BLOCK_BYTES // (get_bit_len(dtype) // Constant.BYTE_BITS)

    def tiling_args(self):
        """
        tiling info:
            tiling_key
            index_list_len
        """
        # tiling scaler init
        self.tiling_key = self.tik_instance.Scalar(self.tiling_dtype, "tiling_key", init_value=0)
        self.index_list_len = self.tik_instance.Scalar(self.tiling_dtype, "index_list_len", init_value=0)
        self.tag_lens = self.tik_instance.Scalar(self.tiling_dtype, "tag_lens", init_value=0)
        self.index_loops_times = self.tik_instance.Scalar(self.tiling_dtype, "index_loops_times", init_value=0)
        self.index_num_tail_loop = self.tik_instance.Scalar(self.tiling_dtype, "index_num_tail_loop", init_value=0)
        tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_NUMS,), name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                    Constant.TILING_NUMS // Constant.ONE_BLK_INT64_NUMS, 0, 0)
        self.tiling_key.set_as(tiling_ub[0])
        self.index_list_len.set_as(tiling_ub[1])
        self.tag_lens.set_as(tiling_ub[2])
        # cal index_list loop
        self.index_loops_times.set_as((self.index_list_len + self.index_num_per_loop - 1) // self.index_num_per_loop)
        self.index_num_tail_loop.set_as(self.index_list_len - (self.index_loops_times - 1) * self.index_num_per_loop)

    def gm_scalar_init(self):
        """
        Define input and output gm
        """
        self.tag_one_set_ub = self.tik_instance.Tensor(self.tag_dtype, [self.way_number],
                                                       name="tag_one_set_ub",
                                                       scope=tik.scope_ubuf)
        self.index_list_ub = self.tik_instance.Tensor(self.index_list_dtype, [self.index_num_per_loop],
                                                      name="index_list_ub",
                                                      scope=tik.scope_ubuf)
        self.mask_addr_lower = self.tik_instance.Tensor("uint16", [self.way_number // 16 * self.tag_rate_int32],
                                                        name="mask_addr_lower",
                                                        scope=tik.scope_ubuf)
        self.mask_addr_upper = self.tik_instance.Tensor("uint16", [self.way_number // 16 * self.tag_rate_int32],
                                                        name="mask_addr_upper",
                                                        scope=tik.scope_ubuf)
        self.mask_addr_tmp = self.tik_instance.Tensor("uint16", [self.way_number // 16 * self.tag_rate_int32],
                                                      name="mask_addr_tmp",
                                                      scope=tik.scope_ubuf)
        self.vreduce_addr = self.tik_instance.Tensor("uint16", [self.way_number * self.tag_rate_int32],
                                                     name="position_index_ub",
                                                     scope=tik.scope_ubuf)
        self.move_out_tmp_ub = self.tik_instance.Tensor(self.index_list_dtype,
                                                        [Constant.BLOCK_BYTES // Constant.INT32_BYTES],
                                                        name="move_out_tmp_ub",
                                                        scope=tik.scope_ubuf)
        self.time_stamp_ub = self.tik_instance.Tensor("float32", [self.sorted_index_shape],
                                                      name="time_stamp_ub",
                                                      scope=tik.scope_ubuf)
        self.vsort_ub_a = self.tik_instance.Tensor("float32", [self.sorted_index_shape, 2],
                                                   name="vsort_ub_a",
                                                   scope=tik.scope_ubuf)
        self.vsort_ub_b = self.tik_instance.Tensor("float32", [self.sorted_index_shape, 2],
                                                   name="vsort_ub_b",
                                                   scope=tik.scope_ubuf)

        self.tag_index = self.tik_instance.Scalar(dtype="int32", name="tag_index", init_value=0)
        self.min_timestamp_index = self.tik_instance.Scalar(dtype="int32", name="min_timestamp_index", init_value=0)
        self.tik_instance.data_move(self.position_index_ub, self.position_index_gm, 0, 1,
                                    self.way_number * self.tag_rate_int32 // 16, 0, 0)
        self.tik_instance.data_move(self.sorted_index_ub, self.sorted_index_gm, 0, 1, self.sorted_index_blocks, 0, 0)
        # scalar
        self.iterate_timestamp_scalar = self.tik_instance.Scalar(self.time_stamp_dtype,
                                                                 "iterate_timestamp_scalar",
                                                                 init_value=0)
        self.index_list_offset = self.tik_instance.Scalar(self.tiling_dtype, "index_list_offset", init_value=0)
        self.rsvd_cnt = self.tik_instance.Scalar(dtype="uint32", name="rsvd_cnt", init_value=0)
        self.index_list_blocks = self.tik_instance.Scalar(dtype="int64", name="index_list_blocks", init_value=0)
        self.index_upper_tmp = self.tik_instance.Scalar(dtype="int64", name="index_upper_tmp", init_value=0)
        self.index_lower_tmp = self.tik_instance.Scalar(dtype="int64", name="index_lower_tmp", init_value=0)
        self.index_upper = self.tik_instance.Scalar(dtype="int32", name="index_upper", init_value=0)
        self.index_lower = self.tik_instance.Scalar(dtype="int32", name="index_lower", init_value=0)
        self.miss_cnt_ub = self.tik_instance.Tensor("int32", [self.set_number_ceil_64],
                                                    name="miss_cnt_ub",
                                                    scope=tik.scope_ubuf)
        self.cache_cnt_ub = self.tik_instance.Tensor("int32", [self.set_number_ceil_64],
                                                     name="cache_cnt_ub",
                                                     scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(Constant.FP32_VECTOR_MASK_MAX, self.miss_cnt_ub, Constant.INIT_ZERO,
                                  self.set_number_ceil_64 // Constant.FP32_VECTOR_MASK_MAX, Constant.VEC_STRIDE)
        self.tik_instance.vec_dup(Constant.FP32_VECTOR_MASK_MAX, self.cache_cnt_ub, Constant.INIT_ZERO,
                                  self.set_number_ceil_64 // Constant.FP32_VECTOR_MASK_MAX, Constant.VEC_STRIDE)
        self.miss_cnt_scalar = self.tik_instance.Scalar(dtype="int32", name="miss_cnt_scalar", init_value=0)
        self.cache_cnt_scalar = self.tik_instance.Scalar(dtype="int32", name="cache_cnt_scalar", init_value=0)
        self.index_scalar = self.tik_instance.Scalar(dtype=self.index_list_dtype, name="index_scalar", init_value=0)
        self.index_core = self.tik_instance.Scalar(dtype=self.index_list_dtype, name="index_core", init_value=0)
        self.index_set = self.tik_instance.Scalar(dtype=self.index_list_dtype, name="index_set", init_value=0)
        self.is_last_call = self.tik_instance.Scalar(dtype=self.is_last_call_dtype, name="is_last_call", init_value=0)
        self.exchane_out_index = self.tik_instance.Scalar(dtype=self.index_list_dtype,
                                                          name="exchane_out_index",
                                                          init_value=0)
        self.exchane_in_index = self.tik_instance.Scalar(dtype=self.index_list_dtype,
                                                         name="exchane_in_index",
                                                         init_value=0)
        self.atomic_add_index = self.tik_instance.Scalar(dtype=self.time_stamp_dtype,
                                                         name="atomic_add_index",
                                                         init_value=Constant.FP32_MIN_VALUE)
        self.offset_index = self.tik_instance.Scalar(dtype=self.index_list_dtype, name="offset_index", init_value=-1)
        self.way_index = self.tik_instance.Scalar(dtype="uint16", name="way_index", init_value=0)

    def init_src_dst_gm(self, input_dict_list, output_dict_list):
        """
        init gm tensor set tiling, input, output tensor(gm)
        """
        tiling_dict = {"dtype": self.tiling_dtype, "shape": self.tiling_shape}
        output_dict_list[5]["is_atomic_add"] = True
        self.op_init_gm(input_dict_list, output_dict_list, tiling_info=tiling_dict)
        self.index_list_gm = self.input_gm_list[0]
        self.data_gm = self.input_gm_list[1]
        self.cache_gm = self.input_gm_list[2]
        self.tag_gm = self.input_gm_list[3]
        self.is_last_call_gm = self.input_gm_list[4]
        self.out_data_gm = self.output_gm_list[0]
        self.out_cache_gm = self.output_gm_list[1]
        self.out_tag_gm = self.output_gm_list[2]
        self.index_offset_list_gm = self.output_gm_list[3]
        self.not_in_cache_index_list_gm = self.output_gm_list[4]
        self.not_in_cache_number_gm = self.output_gm_list[5]

    def iterate_timestamp_refresh(self):
        """
        iterate timestamp refresh
        """
        self.time_stamp_ub[0].set_as(self.atomic_add_index)
        self.tik_instance.set_atomic_add(1)
        self.tik_instance.data_move_pad(self.time_stamp_wsp[self.set_number * self.way_number], self.time_stamp_ub, 1,
                                        self.time_stamp_bytes, 0, 0)
        self.tik_instance.set_atomic_add(0)
        self.tik_instance.data_move_pad(self.time_stamp_ub, self.time_stamp_wsp[self.set_number * self.way_number], 1,
                                        self.time_stamp_bytes, 0, 0)
        self.iterate_timestamp_scalar.set_as(self.time_stamp_ub[0])

    def sort_index(self, index, score, result_ub_a, result_ub_b):
        """
        sort index
        """
        index_len = index.shape[0]
        index_process_list = [1 for i in range(index_len // 32)]
        self.tik_instance.vsort32(result_ub_a, score, index, len(index_process_list))
        ele_count_list = [index_process_list[0] * 32, index_process_list[1] * 32]
        self.tik_instance.vmrgsort(result_ub_b,
                                   (result_ub_a[0:ele_count_list[0] * 2],
                                    result_ub_a[ele_count_list[0] * 2:ele_count_list[0] * 2 + ele_count_list[1] * 2]),
                                   ele_count_list, False, 1)

    def lru_compute(self):
        """
        lru_cache_v2_compute
        """
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as core_id:
            self.tiling_args()
            self.gm_scalar_init()
            self.iterate_timestamp_refresh()
            self.is_last_call.set_as(self.is_last_call_gm[0])
            with self.tik_instance.if_scope(self.tiling_key == Constant.MODE0):
                with self.tik_instance.if_scope(self.is_last_call_gm == 1):
                    self.move_data_back(core_id)
                with self.tik_instance.else_scope():
                    with self.tik_instance.new_stmt_scope():
                        with self.tik_instance.for_range(0, self.index_loops_times) as index_loops:
                            self.index_list_offset.set_as(index_loops * self.index_num_per_loop)
                            with self.tik_instance.if_scope(index_loops < self.index_loops_times - 1):
                                index_list_len = self.index_num_per_loop
                            with self.tik_instance.else_scope():
                                index_list_len = self.index_num_tail_loop
                            self.index_list_process_each_loop(self.index_list_offset, index_list_len, core_id)
                        self.after_process(core_id)
        wr_compile_info = {}
        wr_compile_info["set_num"] = self.set_number
        wr_compile_info["time_stamp_wsp_size"] = self.time_stamp_bytes * (self.set_number * self.way_number + 1)
        wr_compile_info["miss_index_bytes"] = self.tag_bytes
        tbe_context.get_context().add_compile_info("global_variable_link", True)
        tbe_context.get_context().add_compile_info("vars", wr_compile_info)
        self.op_build_cce()

        return self.tik_instance

    def index_list_process_each_loop(self, index_list_offset, index_list_len, core_id):
        """
        tag_set_nums : actual tag sets
        """
        self.index_list_blocks.set_as(
            (index_list_len + self.index_list_ele_per_block - 1) / self.index_list_ele_per_block)
        # do vcmpvs in int32
        vcmpvs_repeats = self.way_number * self.tag_rate_int32 // 64
        # index_list move in
        self.tik_instance.data_move(self.index_list_ub, self.index_list_gm[index_list_offset], Constant.SMMU_ID,
                                    Constant.DATA_MOV_NBURST, self.index_list_blocks, Constant.DATA_MOV_STRIDE,
                                    Constant.DATA_MOV_STRIDE)
        # index for loop
        with self.tik_instance.for_range(0, index_list_len) as index_id:
            self.index_scalar.set_as(self.index_list_ub[index_id])
            self.index_set.set_as((self.index_scalar >> int(math.log(self.pre_route_number, 2))) % self.set_number)
            self.index_core.set_as(self.index_set % self.aicore_num)
            with self.tik_instance.if_scope(self.index_core == core_id):
                self.tik_instance.data_move(self.tag_one_set_ub, self.tag_gm[self.index_set * self.way_number],
                                            Constant.SMMU_ID, Constant.DATA_MOV_NBURST, self.tag_move_blocks,
                                            Constant.DATA_MOV_STRIDE, Constant.DATA_MOV_STRIDE)
                if self.index_list_dtype == "int64":
                    self.index_upper_tmp.set_as(self.index_scalar >> 32)
                    self.index_lower_tmp.set_as(self.index_scalar & 0xffffffff)
                    self.index_upper.set_as(self.index_upper_tmp)
                    self.index_lower.set_as(self.index_lower_tmp)
                    self.tik_instance.vcmpvs_eq(self.mask_addr_lower, self.tag_one_set_ub.reinterpret_cast_to("int32"),
                                                self.index_lower, vcmpvs_repeats, 1, 8)
                    self.tik_instance.vcmpvs_eq(self.mask_addr_upper, self.tag_one_set_ub.reinterpret_cast_to("int32"),
                                                self.index_upper, vcmpvs_repeats, 1, 8)
                    self.tik_instance.vshr(self.way_number // 16 * self.tag_rate_int32, self.mask_addr_tmp,
                                           self.mask_addr_upper, 1, 1, vcmpvs_repeats, 8, 8, 0)
                    self.tik_instance.vand(self.way_number // 16 * self.tag_rate_int32, self.mask_addr_tmp,
                                           self.mask_addr_tmp, self.mask_addr_lower, 1, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vreduce(128,
                                              self.vreduce_addr,
                                              self.position_index_ub,
                                              self.mask_addr_tmp,
                                              1,
                                              1,
                                              8,
                                              8,
                                              rsvd_scalar=self.rsvd_cnt,
                                              mask_mode="counter")
                else:
                    self.tik_instance.vcmpvs_eq(self.mask_addr_tmp, self.tag_one_set_ub, self.index_scalar,
                                                vcmpvs_repeats, 1, 8)
                    self.tik_instance.vreduce(64,
                                              self.vreduce_addr,
                                              self.position_index_ub,
                                              self.mask_addr_tmp,
                                              1,
                                              1,
                                              8,
                                              8,
                                              rsvd_scalar=self.rsvd_cnt,
                                              mask_mode="counter")
                with self.tik_instance.if_scope(self.rsvd_cnt == 1):
                    self.way_index.set_as(self.vreduce_addr[0])
                    self.cache_cnt_scalar.set_as(self.cache_cnt_ub[self.index_set])
                    self.cache_cnt_ub[self.index_set].set_as(self.cache_cnt_scalar + 1)
                    self.time_stamp_ub[0].set_as(self.iterate_timestamp_scalar)
                    self.tik_instance.data_move_pad(
                        self.time_stamp_wsp[self.index_set * self.way_number + self.way_index], self.time_stamp_ub, 1,
                        self.time_stamp_bytes, 0, 0)
                    self.move_out_tmp_ub[0].set_as(
                        (self.index_set * self.way_number + self.way_index) * self.embedding_size)
                with self.tik_instance.else_scope():
                    self.move_out_tmp_ub[0].set_as(self.index_scalar)
                    self.miss_cnt_scalar.set_as(self.miss_cnt_ub[self.index_set])
                    self.miss_cnt_scalar.set_as(self.miss_cnt_scalar + 1)
                    self.miss_cnt_ub[self.index_set].set_as(self.miss_cnt_scalar)
                    self.tik_instance.data_move_pad(
                        self.miss_index_wsp[self.index_set * self.index_list_len + self.miss_cnt_scalar -
                                            1].reinterpret_cast_to("int32"),
                        self.move_out_tmp_ub.reinterpret_cast_to("int32"), 1, self.index_list_bytes, 0, 0)
                    self.move_out_tmp_ub[0].set_as(self.offset_index)
                self.tik_instance.data_move_pad(
                    self.index_offset_list_gm[index_list_offset + index_id].reinterpret_cast_to("int32"),
                    self.move_out_tmp_ub.reinterpret_cast_to("int32"), 1, self.index_list_bytes, 0, 0)

    def after_process(self, core_id):
        """
        after index loop
        """
        with self.tik_instance.for_range(0, self.set_number) as set_id:
            with self.tik_instance.if_scope(set_id % self.aicore_num == core_id):
                self.tik_instance.data_move(self.time_stamp_ub, self.time_stamp_wsp[set_id * self.way_number], 0, 1,
                                            self.sorted_index_blocks, 0, 0)
                self.sort_index(self.sorted_index_ub, self.time_stamp_ub, self.vsort_ub_a, self.vsort_ub_b)
                self.vsort_ub_b.reinterpret_cast_to("int32")
                self.miss_cnt_scalar.set_as(self.miss_cnt_ub[set_id])
                self.cache_cnt_scalar.set_as(self.cache_cnt_ub[set_id])
                # set thread_num as 4 to accelerate the data_move process
                with self.tik_instance.for_range(0, self.miss_cnt_scalar, thread_num=Constant.THREAD_NUM) as miss_id:
                    data_exchange_ub = self.tik_instance.Tensor(self.data_dtype,
                                                                [self.embedding_size * Constant.THREAD_NUM],
                                                                name="data_exchange_ub",
                                                                scope=tik.scope_ubuf)
                    move_out_ub = self.tik_instance.Tensor(self.index_list_dtype,
                                                           [Constant.BLOCK_BYTES // Constant.INT32_BYTES *
                                                            Constant.THREAD_NUM],
                                                           name="move_out_ub",
                                                           scope=tik.scope_ubuf)
                    self.tik_instance.data_move_pad(
                        move_out_ub.reinterpret_cast_to("int32"),
                        self.miss_index_wsp[set_id * self.index_list_len + miss_id].reinterpret_cast_to("int32"), 1,
                        self.index_list_bytes, 0, 0)
                    with self.tik_instance.if_scope(miss_id < self.way_number - self.cache_cnt_scalar):
                        self.exchane_in_index.set_as(move_out_ub[0])
                        self.min_timestamp_index.set_as(self.vsort_ub_b[self.sorted_index_shape * 2 - 1 - miss_id * 2])
                        self.tag_index.set_as(self.min_timestamp_index + set_id * self.way_number)
                        self.tik_instance.data_move_pad(move_out_ub.reinterpret_cast_to("int32"),
                                                        self.tag_gm[self.tag_index].reinterpret_cast_to("int32"), 1,
                                                        self.index_list_bytes, 0, 0)
                        self.exchane_out_index.set_as(move_out_ub[0])
                        self.cache_exchange(self.exchane_out_index, self.exchane_in_index, self.tag_index,
                                            move_out_ub, data_exchange_ub)

    def move_data_back(self, core_id):
        """
        move data from cache back to gm when last call
        """
        with self.tik_instance.for_range(0, self.tag_lens) as tag_idx:
            data_exchange_ub = self.tik_instance.Tensor(self.data_dtype,
                                                        [self.embedding_size * Constant.THREAD_NUM],
                                                        name="data_exchange_ub",
                                                        scope=tik.scope_ubuf)
            move_out_ub = self.tik_instance.Tensor(self.index_list_dtype,
                                                   [Constant.BLOCK_BYTES // Constant.INT32_BYTES *
                                                    Constant.THREAD_NUM],
                                                   name="move_out_ub",
                                                   scope=tik.scope_ubuf)
            self.tik_instance.data_move_pad(move_out_ub.reinterpret_cast_to("int32"),
                                            self.tag_gm[tag_idx].reinterpret_cast_to("int32"), 1,
                                            self.index_list_bytes, 0, 0)
            self.exchane_out_index.set_as(move_out_ub[0])
            with self.tik_instance.if_scope(tag_idx % self.aicore_num == core_id):
                self.tik_instance.data_move_pad(data_exchange_ub, self.cache_gm[tag_idx * self.embedding_size], 1,
                                                self.embedding_bytes, 0, 0)
                self.tik_instance.data_move_pad(self.out_data_gm[self.exchane_out_index * self.embedding_size],
                                                data_exchange_ub, 1,
                                                self.embedding_bytes, 0, 0)

    def cache_exchange(self, out_index, in_index, tag_index, move_out_ub, data_exchange_ub):
        """
        exchange cache and data
        """
        # disable the sync to accelerate the data_move process
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            # last time stamp cache move to data
            self.tik_instance.data_move_pad(data_exchange_ub, self.cache_gm[tag_index * self.embedding_size], 1,
                                            self.embedding_bytes, 0, 0)
            self.tik_instance.data_move_pad(self.out_data_gm[out_index * self.embedding_size], data_exchange_ub, 1,
                                            self.embedding_bytes, 0, 0)
            # not in cache list move in cahce
            self.tik_instance.data_move_pad(data_exchange_ub, self.data_gm[in_index * self.embedding_size], 1,
                                            self.embedding_bytes, 0, 0)
            self.tik_instance.data_move_pad(self.out_cache_gm[tag_index * self.embedding_size], data_exchange_ub, 1,
                                            self.embedding_bytes, 0, 0)
            # tag exchange
            move_out_ub[0].set_as(in_index)
            self.tik_instance.data_move_pad(self.out_tag_gm[tag_index].reinterpret_cast_to("int32"),
                                            move_out_ub.reinterpret_cast_to("int32"), 1, self.index_list_bytes, 0,
                                            0)
            # timestamp exchange
            self.time_stamp_ub[0].set_as(self.iterate_timestamp_scalar)
            self.tik_instance.data_move_pad(self.time_stamp_wsp[tag_index], self.time_stamp_ub, 1,
                                            self.time_stamp_bytes, 0, 0)

    def check_param(self):
        """
        check_param
        """
        para_check.check_shape(self.data_shape, min_rank=2, max_rank=2, param_name="data")
        para_check.check_shape(self.tag_shape, min_rank=1, max_rank=1, param_name="tag")
        para_check.check_shape(self.cache_shape, min_rank=1, max_rank=1, param_name="cache")

        if self.tag_shape[0] != self.set_number * self.way_number:
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'lru'
            error_info['param_name'] = "lru's tag size"
            error_info['expected_value'] = self.set_number * self.way_number
            error_info['real_value'] = self.tag_shape[0]
            raise RuntimeError(
                error_info, "In op[{op_name}], the parameter[{param_name}] "
                "should be [{expected_value}], "
                "but actually is [{real_value}].".format(**error_info))
        if self.cache_shape[0] != self.tag_shape[0] * self.embedding_size:
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'lru'
            error_info['param_name'] = "lru's cahce size"
            error_info['expected_value'] = self.tag_shape[0] * \
                self.embedding_size
            error_info['real_value'] = self.cache_shape[0]
            raise RuntimeError(
                error_info, "In op[{op_name}], the parameter[{param_name}] "
                "should be [{expected_value}], "
                "but actually is [{real_value}].".format(**error_info))
        check_list_input_list = ["int32", "int64"]
        check_list_data = ["float16", "float32"]
        para_check.check_dtype(self.index_list_dtype, check_list_input_list, param_name="input_list")
        para_check.check_dtype(self.tag_dtype, check_list_input_list, param_name="tag")
        para_check.check_dtype(self.cache_dtype, check_list_data, param_name="cache")
        para_check.check_dtype(self.data_dtype, check_list_data, param_name="data")
        if not self.is_power(self.set_number):
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'lru'
            error_info['param_name'] = "set_number"
            error_info['expected_value'] = "2**n"
            error_info['real_value'] = self.set_number
            raise RuntimeError(
                error_info, "In op[{op_name}], the parameter[{param_name}] "
                "should be [{expected_value}], but actually "
                "is [{real_value}].".format(**error_info))
        if not self.is_power(self.pre_route_number):
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'lru'
            error_info['param_name'] = "pre_route_number"
            error_info['expected_value'] = "2**n"
            error_info['real_value'] = self.pre_route_number
            raise RuntimeError(
                error_info, "In op[{op_name}], the parameter[{param_name}] "
                "should be [{expected_value}], but actually "
                "is [{real_value}].".format(**error_info))


@register_operator("LRUCacheV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def lru_cache_v2(index_list,
                 data,
                 cache,
                 tag,
                 is_last_call,
                 out_data,
                 out_cache,
                 out_tag,
                 index_offset_list,
                 not_in_cache_index_list,
                 not_in_cache_number,
                 pre_route_count,
                 kernel_name="lru_cache_v2"):
    """
    index_list:exchange index list
    data:host data
    cache:gm cache
    tag:cache's tag
    is_last_call: if is last call write all cache to data
    out_data:output data
    out_cache:output gm cache
    out_tag:output cache's tag
    index_offset_list,
    not_in_cache_index_list,
    not_in_cache_number,
    pre_route_count,
    """
    obj = Lru(index_list, data, cache, tag, is_last_call, out_data, out_cache, out_tag, index_offset_list,
              not_in_cache_index_list, not_in_cache_number, pre_route_count, kernel_name)
    obj.init_src_dst_gm([index_list, data, cache, tag, is_last_call],
                        [out_data, out_cache, out_tag, index_offset_list, not_in_cache_index_list, not_in_cache_number])
    return obj.lru_compute()
