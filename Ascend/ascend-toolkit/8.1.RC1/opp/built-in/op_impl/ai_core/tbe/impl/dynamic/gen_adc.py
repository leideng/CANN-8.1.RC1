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
gen_adc
"""
import math

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class GenADC():
    """
    Class for Dynamic shape operator GenADC
    """
    TYPE_LEN_DICT = {"float16": 2, "float32": 4,
                     "int8": 1, "uint8": 1,
                    "int16": 2, "uint16": 2, "int32": 4, "uint32": 4,
                    "int64": 8, "uint64": 8}
    BLOCK_SIZE = 32
    DISTANCE_TYPE_L2SQR = "l2sqr"
    DISTANCE_TYPE_INNER_PRODUCT = "inner_product"

    # 'pylint: disable=too-many-arguments,too-many-statements
    def __init__(self, query, code_book, centroids, bucket_list, adc_tables, distance_type,
                 kernel_name):
        self.kernel_name = kernel_name

        self.tik_inst = tik.Tik(tik.Dprofile)

        self.query_dtype = query.get("dtype").lower()
        self.query_shape = query.get("shape")

        self.code_book_dtype = code_book.get("dtype").lower()
        self.code_book_shape = code_book.get("shape")

        self.centroids_dtype = centroids.get("dtype").lower()
        self.centroids_shape = centroids.get("shape")

        self.bucket_list_dtype = bucket_list.get("dtype").lower()
        self.bucket_list_dsize = GenADC.TYPE_LEN_DICT.get(self.bucket_list_dtype)
        self.bucket_list_shape = bucket_list.get("shape")

        self.adc_tables_dtype = adc_tables.get("dtype").lower()

        para_check.check_dtype(self.query_dtype, ("float16", "float32"),
                               param_name="query")
        para_check.check_dtype(self.code_book_dtype, ("float16"),
                               param_name="code_book")
        para_check.check_dtype(self.centroids_dtype, ("float16"),
                               param_name="centroids")
        para_check.check_dtype(self.bucket_list_dtype, ("int32", "int64"),
                               param_name="bucket_list")
        para_check.check_dtype(self.adc_tables_dtype, ("float16"),
                               param_name="adc_tables")
        para_check.check_shape(self.code_book_shape, min_rank=3, param_name="code_book")

        self.distance_type = distance_type
        if self.distance_type not in (GenADC.DISTANCE_TYPE_L2SQR,
                                      GenADC.DISTANCE_TYPE_INNER_PRODUCT):
            rule = "Distance type should be l2sqr or inner_product"
            error_manager_vector.raise_err_specific_reson(self.distance_type, rule)

        self.dim_d = self.query_shape[0]
        self.dim_m = self.code_book_shape[0]
        self.dim_ksub = self.code_book_shape[1]
        self.dim_dsub = self.code_book_shape[2]

        if self.dim_d != self.dim_m * self.dim_dsub:
            rule = "Failed to check the division of subspaces."
            error_manager_vector.raise_err_specific_reson(self.kernel_name, rule)

        if self.dim_d < 128:
            if self.dim_m % 8 != 0:
                rule = "The number of subspaces (M) should be a multiple of 8."
                error_manager_vector.raise_err_specific_reson(self.kernel_name, rule)
        elif self.dim_d % 128 != 0:
            rule = "If d is greater than 128, it should be a multiple of 128."
            error_manager_vector.raise_err_specific_reson(self.kernel_name, rule)

        self.dim_m_stride = 8
        self.dim_ksub_stride = 1

        self.op_query_data_type = query.get("dtype").lower()
        self.op_data_type = code_book.get("dtype").lower()
        self.op_data_size = GenADC.TYPE_LEN_DICT.get(self.op_data_type)

        self.tiling_dtype = "int64"
        self.tiling_para_num = 8
        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_para_num,),
                                              name="tiling_gm",
                                              scope=tik.scope_gm)

        self.max_int32 = 2 ** 31 - 1
        self.centroids_gm = self.tik_inst.Tensor(self.op_data_type, (self.max_int32,),
                                                 name="centroids_gm",
                                                 scope=tik.scope_gm)
        self.bucket_list_gm = self.tik_inst.Tensor(self.bucket_list_dtype, (self.max_int32,),
                                                   name="bucket_list_gm",
                                                   scope=tik.scope_gm)
        self.query_gm = self.tik_inst.Tensor(self.op_query_data_type, (self.max_int32,),
                                             name="query_gm",
                                             scope=tik.scope_gm)
        self.code_book_gm = self.tik_inst.Tensor(self.op_data_type, (self.max_int32,),
                                                 name="code_book_gm",
                                                 scope=tik.scope_gm)

        self.adc_tables_gm = self.tik_inst.Tensor(self.op_data_type, (self.max_int32,),
                                                  name="adc_tables_gm",
                                                  scope=tik.scope_gm)

        self.row_num_each_core = self.tik_inst.Scalar("int64",
                                                      name="row_num_each_core")
        self.remaining_row = self.tik_inst.Scalar("int64",
                                                  name="remaining_row")
        self.bucket_list_burst_len = self.tik_inst.Scalar("int64",
                                                          name="bucket_list_burst_len")
        self.remain_bucket_list_burst_len = self.tik_inst.Scalar("int64",
                                                                 name="remain_bucket_list_burst_len")
        self.core_used_num = self.tik_inst.Scalar("int64",
                                                  name="core_used_num")
        self.dim_ns = self.tik_inst.Scalar("int64",
                                           name="dim_ns")
        self.core_num_var = self.tik_inst.Scalar(self.tiling_dtype,
                                                 name="core_num_var")

        if self.bucket_list_shape[0] != -1:
            self.dim_ns = self.bucket_list_shape[0]

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        if self.ai_core_num <= 0:
            error_manager_vector.raise_err_specific_reson(self.kernel_name, "The value of the CORE_NUM is illegal!")

        self.tiling_ub = None
        self.query_ub = None
        self.query_fp16_ub = None
        self.code_book_ub = None
        self.bucket_list_ub = None
        self.centroids_ub = None

        self.zero_ub = None

        self.tmp_buf = None
        self.code_book_local_ub = None

        self.reduce_0_local_ub = None

    @staticmethod
    def _compute_burst_len(dtype, shape):
        block_bite_size = 32
        dtype_bytes = get_bit_len(dtype) // 8
        data_each_block = block_bite_size // dtype_bytes

        burst_len = math.ceil(para_check.check_tensor_shape_size(shape) / data_each_block)

        return burst_len

    def gen_adc_compute(self):
        """
        Generate adc tables.
        """
        factor_m, factor_k = self._get_m_ksub_factors()
        self.dim_m_stride = self.dim_m // factor_m
        self.dim_ksub_stride = self.dim_ksub // factor_k

        self.tiling_ub = self.tik_inst.Tensor("int64", (self.tiling_para_num,),
                                              name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        burst = math.ceil(self.tiling_para_num * GenADC.TYPE_LEN_DICT.get(self.tiling_dtype) / GenADC.BLOCK_SIZE)
        self.tik_inst.data_move(self.tiling_ub, self.tiling_gm, 0, 1, burst, 0, 0)
        self._tiling_args()
        with self.tik_inst.for_range(0, self.core_num_var,
                                     block_num=self.core_num_var) as block_i:
            self._init_ub_data()

            self.tik_inst.data_move(self.query_ub, self.query_gm, 0, 1,
                                    self._compute_burst_len(self.op_query_data_type, self.query_shape),
                                    0, 0)
            if self.op_data_type != self.op_query_data_type:
                conv_repeat = max(self.dim_d // 64, 1)
                conv_mask = min(self.dim_d, 64)
                self.tik_inst.vconv(conv_mask, "", self.query_fp16_ub,
                                    self.query_ub, conv_repeat, 1, 1, 4, 8)

            with self.tik_inst.if_scope(block_i == (self.core_used_num - 1)):
                self._adc_compute(block_i, self.remaining_row,
                                  self.remain_bucket_list_burst_len,
                                  self.row_num_each_core)
            with self.tik_inst.if_scope(block_i < (self.core_used_num - 1)):
                self._adc_compute(block_i, self.row_num_each_core,
                                  self.bucket_list_burst_len,
                                  self.row_num_each_core)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.ai_core_num})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.query_gm, self.code_book_gm,
                                       self.centroids_gm, self.bucket_list_gm],
                               outputs=[self.adc_tables_gm],
                               flowtable=(self.tiling_gm,),
                               config=opt_config)

    # 'pylint: disable=invalid-name
    def _get_m_ksub_factors(self):
        """
        compute k factor and m factor

        return: factor_m, factor_k
        """
        m_std = 8
        ksub_std = 256
        d_std = 2

        m = self.dim_m
        ksub = self.dim_ksub
        d = self.dim_dsub

        sur_multiple_d = d // d_std
        sur_multiple_ksub = ksub // ksub_std
        sur_multiple_m = m // m_std

        factor_d = sur_multiple_d if sur_multiple_d else 1
        factor_k = sur_multiple_ksub if sur_multiple_ksub else 1
        factor_m = sur_multiple_m if sur_multiple_m else 1

        factor_total = factor_d * factor_k * factor_m

        if factor_total == 1:
            factor_k = 1
            factor_m = 1
        else:
            factor_k = factor_total // factor_m

        return factor_m, factor_k

    def _tiling_args(self):
        """
        Get runtime tiling parameters from tiling.
        """
        self.row_num_each_core.set_as(self.tiling_ub[0])
        self.remaining_row.set_as(self.tiling_ub[1])
        self.bucket_list_burst_len.set_as(self.tiling_ub[2])
        self.remain_bucket_list_burst_len.set_as(self.tiling_ub[3])
        self.core_used_num.set_as(self.tiling_ub[4])
        if not isinstance(self.dim_ns, int):
            self.dim_ns.set_as(self.tiling_ub[5])
        self.core_num_var.set_as(self.tiling_ub[6])

    def _init_ub_data(self):
        """
        Apply unified buffer for variables.
        """

        self.query_ub = self.tik_inst.Tensor(self.op_query_data_type, self.query_shape,
                                             name="query_ub",
                                             scope=tik.scope_ubuf)

        if self.op_data_type != self.op_query_data_type:
            self.query_fp16_ub = self.tik_inst.Tensor(self.op_data_type, self.query_shape,
                                                      name="query_fp16_ub",
                                                      scope=tik.scope_ubuf)

        bucket_list_dtype_bytes = get_bit_len(self.bucket_list_dtype) // 8
        bucket_list_num_each_block = 32 // bucket_list_dtype_bytes

        if not isinstance(self.dim_ns, int):
            bucket_list_ub_len = (self.row_num_each_core + bucket_list_num_each_block - 1) // \
                                    bucket_list_num_each_block * bucket_list_num_each_block
        else:
            bucket_list_ub_len = (self.dim_ns + bucket_list_num_each_block - 1) // \
                                   bucket_list_num_each_block * bucket_list_num_each_block

        self.bucket_list_ub = self.tik_inst.Tensor(self.bucket_list_dtype, (bucket_list_ub_len,),
                                                   name="bucket_list_ub",
                                                   scope=tik.scope_ubuf)

        self.centroids_ub = self.tik_inst.Tensor(self.op_data_type, (self.dim_ns, self.dim_d),
                                                 name="centroids_ub",
                                                 scope=tik.scope_ubuf)

        self.zero_ub = self.tik_inst.Tensor("uint16", (128,), name="zero_ub",
                                            scope=tik.scope_ubuf)

        self.tmp_buf = self.tik_inst.Tensor(self.op_data_type, (128,), name="tmp_buf",
                                            scope=tik.scope_ubuf)
        self.code_book_local_ub = self.tik_inst.Tensor(self.op_data_type,
                                                       (self.dim_m_stride, self.dim_ksub_stride, self.dim_dsub),
                                                       name="code_book_local_ub",
                                                       scope=tik.scope_ubuf)

        self.reduce_0_local_ub = self.tik_inst.Tensor(self.op_data_type, (self.dim_m_stride, self.dim_ksub_stride,
                                                                          self.dim_dsub),
                                                      name="reduce_0_local_ub",
                                                      scope=tik.scope_ubuf)

    def _adc_compute(self, block_i, process_row, bucket_list_burst_len, core_row_offset):
        """
        Compute adc table.
        """
        self.tik_inst.data_move(self.bucket_list_ub, self.bucket_list_gm[block_i * core_row_offset],
                                0, 1, bucket_list_burst_len, 0, 0)
        with self.tik_inst.for_range(0, process_row, name="row") as i:
            tmp_data = self.tik_inst.Scalar(dtype=self.bucket_list_dtype,
                                            init_value=self.bucket_list_ub[i])
            gather_len = self._compute_burst_len(self.op_data_type, (self.dim_d,))
            self.tik_inst.data_move(self.centroids_ub[i * self.dim_d],
                                    self.centroids_gm[tmp_data * self.dim_d],
                                    0, 1, gather_len, 0, 0)

        rep_stride = self.dim_d * GenADC.TYPE_LEN_DICT.get(self.op_data_type) // GenADC.BLOCK_SIZE
        src_add = self.query_ub
        if self.op_data_type != self.op_query_data_type:
            src_add = self.query_fp16_ub

        vec_sub_mask = min(128, self.dim_d)
        vec_sub_repeat = process_row * max(self.dim_d // 128, 1)
        sub_repeat = max(self.dim_d // 128, 1)
        for insn_nums in range(0, sub_repeat):
            self.tik_inst.vsub(vec_sub_mask, self.centroids_ub[vec_sub_mask * insn_nums],
                               src_add[vec_sub_mask * insn_nums], self.centroids_ub[vec_sub_mask * insn_nums],
                               vec_sub_repeat,
                               1, 1, 1,
                               rep_stride, 0, rep_stride)

        self._compute_distance(block_i, core_row_offset, process_row)

    # 'pylint: disable=too-many-arguments,too-many-statements
    def _compute_distance(self, block_i, core_row_offset, process_row):
        """
        Compute adc distance.
        """
        self.tik_inst.vector_dup(128, self.zero_ub, 0, 1, 1, 8)

        residual_vect_ub_int32 = self.centroids_ub.reinterpret_cast_to("int32")
        with self.tik_inst.for_range(0, process_row, name="i0_inner") as i:
            with self.tik_inst.for_range(0, self.dim_m // self.dim_m_stride, name="dim_m_i") as dim_m_i:
                with self.tik_inst.for_range(0, self.dim_ksub // self.dim_ksub_stride,
                                             name="dim_ksub_i") as dim_ksub_i:
                    self.tik_inst.data_move(self.code_book_local_ub,
                                            self.code_book_gm[dim_m_i * self.dim_m_stride *
                                                              self.dim_ksub * self.dim_dsub +
                                                              dim_ksub_i * self.dim_m_stride *
                                                              self.dim_ksub_stride * self.dim_dsub],
                                            0, 1,
                                            self.dim_m_stride * self.dim_ksub_stride * self.dim_dsub * 2 // 32,
                                            0, 0)
                    tmp_buf_int32 = self.tmp_buf.reinterpret_cast_to("int32")

                    def _compute_residual_for_less_block():
                        copy_times = 16 // self.dim_dsub
                        bro_repeat_times = self.dim_m_stride
                        bro_times = self.dim_ksub_stride * self.dim_dsub // 16 // 8
                        group_block = self.dim_ksub_stride * self.dim_dsub // 16
                        k_repeat = self.dim_ksub // self.dim_ksub_stride
                        for insn_nums in range(0, self.dim_m_stride // k_repeat):
                            for k_stride in range(0, k_repeat):
                                for copy_nums in range(0, copy_times):
                                    for ele_nums in range(0, self.dim_dsub // 2):
                                        tmp_buf_int32[insn_nums * 8 * k_repeat + k_stride * 8 + copy_nums *
                                                      self.dim_dsub // 2 + ele_nums].set_as(
                                            residual_vect_ub_int32[(i * self.dim_m * self.dim_dsub +
                                                                    dim_m_i * self.dim_m_stride * self.dim_dsub +
                                                                    dim_ksub_i * self.dim_m_stride //
                                                                    k_repeat * self.dim_dsub) // 2
                                                                   + insn_nums * self.dim_dsub // 2 + ele_nums])
                        for insn_nums in range(0, bro_times):
                            if self.distance_type == GenADC.DISTANCE_TYPE_L2SQR:
                                self.tik_inst.vsub(128, self.code_book_local_ub[insn_nums * 128],
                                                   self.code_book_local_ub[insn_nums * 128],
                                                   self.tmp_buf,
                                                   bro_repeat_times,
                                                   1, 1, 0,
                                                   group_block,
                                                   group_block, 1)
                            elif self.distance_type == GenADC.DISTANCE_TYPE_INNER_PRODUCT:
                                self.tik_inst.vmul(128, self.code_book_local_ub[insn_nums * 128],
                                                   self.code_book_local_ub[insn_nums * 128],
                                                   self.tmp_buf,
                                                   bro_repeat_times, 1, 1, 0,
                                                   group_block,
                                                   group_block, 1)

                    def _compute_residual_for_greater_block():
                        if self.dim_dsub == 16:
                            bro_times = self.dim_m_stride
                            bro_repeat_times = self.dim_ksub_stride // 8
                            for insn_nums in range(0, bro_times):
                                if self.distance_type == GenADC.DISTANCE_TYPE_L2SQR:
                                    self.tik_inst.vsub(128,
                                                       self.code_book_local_ub[insn_nums * self.dim_ksub_stride * 16],
                                                       self.code_book_local_ub[insn_nums * self.dim_ksub_stride * 16],
                                                       self.centroids_ub[i * self.dim_m + dim_m_i * self.dim_m_stride +
                                                                         insn_nums * 16],
                                                       bro_repeat_times, 1, 1, 0, 8, 8, 0)
                                elif self.distance_type == GenADC.DISTANCE_TYPE_INNER_PRODUCT:
                                    self.tik_inst.vmul(128,
                                                       self.code_book_local_ub[insn_nums * self.dim_ksub_stride * 16],
                                                       self.code_book_local_ub[insn_nums * self.dim_ksub_stride * 16],
                                                       self.centroids_ub[i * self.dim_m + dim_m_i * self.dim_m_stride +
                                                                         insn_nums * 16],
                                                       bro_repeat_times, 1, 1, 0, 8, 8, 0)
                        else:
                            vector_nums = self.dim_dsub // 16
                            sub_mask = self.dim_m_stride * 16
                            bro_repeat_times = self.dim_ksub_stride
                            for insn_nums in range(0, vector_nums):
                                if self.distance_type == GenADC.DISTANCE_TYPE_L2SQR:
                                    self.tik_inst.vsub(sub_mask, self.code_book_local_ub[insn_nums * 16],
                                                       self.code_book_local_ub[insn_nums * 16],
                                                       self.centroids_ub[i * self.dim_m + dim_m_i * self.dim_m_stride +
                                                                         insn_nums * 16],
                                                       bro_repeat_times,
                                                       vector_nums * self.dim_ksub_stride,
                                                       vector_nums * self.dim_ksub_stride, vector_nums,
                                                       vector_nums, vector_nums, 2)
                                elif self.distance_type == GenADC.DISTANCE_TYPE_INNER_PRODUCT:
                                    self.tik_inst.vmul(sub_mask, self.code_book_local_ub[insn_nums * 16],
                                                       self.code_book_local_ub[insn_nums * 16],
                                                       self.centroids_ub[i * self.dim_m + dim_m_i * self.dim_m_stride +
                                                                         insn_nums * 16],
                                                       bro_repeat_times,
                                                       vector_nums * self.dim_ksub_stride,
                                                       vector_nums * self.dim_ksub_stride, vector_nums,
                                                       vector_nums, vector_nums, 2)

                    if self.dim_dsub < 16:
                        _compute_residual_for_less_block()
                    else:
                        _compute_residual_for_greater_block()

                    if self.distance_type == GenADC.DISTANCE_TYPE_L2SQR:
                        self.tik_inst.vmul(128, self.code_book_local_ub, self.code_book_local_ub,
                                           self.code_book_local_ub,
                                           self.dim_m_stride * self.dim_ksub_stride * self.dim_dsub // 128,
                                           1, 1, 1, 8, 8, 8)

                    def _compute_reduce_for_two_eles():
                        vcpadd_repeat_times = self.dim_m_stride * self.dim_ksub_stride * self.dim_dsub // 128
                        vcpadd_cycle_times = (vcpadd_repeat_times + 128 - 1) // 128
                        with self.tik_inst.for_range(0, vcpadd_cycle_times, name="vcpadd_i") as vcpadd_i:
                            with self.tik_inst.if_scope(vcpadd_i < (vcpadd_repeat_times - 1)):
                                self.tik_inst.vcpadd(128, self.reduce_0_local_ub[vcpadd_i * 128 * 128 // 2],
                                                     self.code_book_local_ub[vcpadd_i * 128 * 128],
                                                     128, 1, 1, 8)
                            with self.tik_inst.if_scope(vcpadd_i == (vcpadd_repeat_times - 1)):
                                vcpadd_tile_repeat_times = vcpadd_repeat_times - vcpadd_i * 128
                                self.tik_inst.vcpadd(128, self.reduce_0_local_ub[vcpadd_i * 128 * 128 // 2],
                                                     self.code_book_local_ub[vcpadd_i * 128 * 128],
                                                     vcpadd_tile_repeat_times,
                                                     1, 1, 8)
                        self.tik_inst.data_move(self.adc_tables_gm[
                                                    block_i * core_row_offset * self.dim_m * self.dim_ksub +
                                                    i * self.dim_m * self.dim_ksub +
                                                    dim_m_i * self.dim_m_stride * self.dim_ksub +
                                                    dim_ksub_i * self.dim_m_stride * self.dim_ksub_stride],
                                                self.reduce_0_local_ub, 0, 1,
                                                self.dim_m_stride * self.dim_ksub_stride * 2 // 32,
                                                0, 0)

                    def _compute_reduce_for_multiple_eles():
                        vcpadd_repeat_times = self.dim_m_stride * self.dim_ksub_stride * self.dim_dsub // 128
                        addr_list = [self.code_book_local_ub, self.reduce_0_local_ub]
                        insn_times = int(math.log(self.dim_dsub, 2))
                        for insn_nums in range(0, insn_times):
                            dst_addr = (insn_nums + 1) % 2
                            src_addr = (insn_nums) % 2
                            if vcpadd_repeat_times > 255:
                                vcp_nums = vcpadd_repeat_times // 255
                                last_nums = vcpadd_repeat_times % 255
                                for times in range(0, vcp_nums):
                                    self.tik_inst.vcpadd(128, addr_list[dst_addr][times * 255 * 128 // 2],
                                                         addr_list[src_addr][times * 255 * 128],
                                                         255, 1, 1, 8)
                                if last_nums != 0:
                                    self.tik_inst.vcpadd(128, addr_list[dst_addr][vcp_nums * 255 * 128 // 2],
                                                         addr_list[src_addr][vcp_nums * 255 * 128],
                                                         last_nums, 1, 1, 8)
                            else:
                                self.tik_inst.vcpadd(128, addr_list[dst_addr],
                                                     addr_list[src_addr],
                                                     vcpadd_repeat_times, 1, 1, 8)
                            vcpadd_repeat_times = vcpadd_repeat_times // 2
                        src_index = insn_times % 2
                        self.tik_inst.data_move(self.adc_tables_gm[
                                                     block_i * core_row_offset * self.dim_m * self.dim_ksub +
                                                     i * self.dim_m * self.dim_ksub +
                                                     dim_m_i * self.dim_m_stride * self.dim_ksub +
                                                     dim_ksub_i * self.dim_m_stride * self.dim_ksub_stride],
                                                addr_list[src_index], 0, 1,
                                                self.dim_m_stride * self.dim_ksub_stride * 2 // 32, 0, 0)

                    if self.dim_dsub == 2:
                        _compute_reduce_for_two_eles()
                    else:
                        _compute_reduce_for_multiple_eles()


# 'pylint: disable=too-many-arguments,too-many-statements,huawei-too-many-arguments
@register_operator("GenADC")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def gen_adc(query, code_book, centroids, bucket_list, adc_tables,
            distance_type=GenADC.DISTANCE_TYPE_L2SQR,
            kernel_name="gen_adc"):
    """
    Compute ADC tables for query vector.

    Parameters
    ----------
    query : dict
        shape and dtype of input data query
        The shape is like (d,).
        The support value of d in range(start=16, end=1024, step=16).
        If d is greater than 128, it should be a multiple of 128.
    code_book : dict
        shape and dtype of input data code_book
        The shape is like (M, ksub, dsub). Support values are as follows:
            M = d / dsub, and M is a multiple of 8
            ksub in {256, 512}
            dsub in {2, 4, 8}
    centroids : dict
        shape and dtype of input data centroids
        The shape is like (nc, d).
        The value of nc in range(start=1, end=1e7, step=1).
        The value of d is the same as query.
    bucket_list : dict
        shape and dtype of input data bucket_list
        The shape is like (ns,). The value of ns in range(start=1, end=nc, step=1).
    adc_tables : dict
        shape and dtype of output data adc_tables
        The shape is like (ns, M, ksub). Values of dimensions are the same as input tensors' dictionary.
    distance_type: string
        The distance type to compute. Value is "l2sqr" or "inner_product".
    kernel_name : str
        cce kernel name, default value is "gen_adc"

    Algorithm
    -------
    query = Tensor(size=(d,))
    code_book = Tensor(size=(M, ksub, dsub))
    centroids = Tensor(size=(nc, d))
    bucket_list = Tensor(size=(ns,))
    bucket_centroids = gather(centroids, bucket_list)
    residual_vect = query - bucket_centroids
    residual_vect_list = residual_vect.reshape(size=(ns, M, 1, dsub))
    if distance_type == "l2sqr":
        distance = code_book - residual_vect_list
        square_distance = square(distance)
        adc_tables = sum(square_distance, axis=-1)
    elif distance_type == "inner_product":
        distance = code_book * residual_vect_list
        adc_tables = sum(distance, axis=-1)

    Returns
    -------
    None
    """
    obj = GenADC(query, code_book, centroids, bucket_list, adc_tables, distance_type,
                 kernel_name)
    obj.gen_adc_compute()
