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
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import check_support_block_size_16
from impl.util.platform_adapter import tbe_context
from impl.util import util_tik_comm_func
from tbe.common.platform import get_bit_len
from impl import common_util
import te.platform as te_platform
import numpy as np


class Constant:
    N16 = 16
    N32 = 32
    N128 = 128
    N256 = 256
    B32_BYTE = 4
    B16_BYTE = 2
    SCALAR_MIN_FP16 = -65504.0
    SCALAR_MIN_FP32 = float(np.finfo(np.float32).min)
    SCALAR_MIN_INT64 = -2**63
    SCALAR_MIN_INT32 = -2**31
    SCALAR_MAX_FP16 = (2**16 - 1)
    SCALAR_MAX_FP32 = float(np.finfo(np.float32).max)
    SCALAR_MAX_INT64 = 2**63 - 1
    SCALAR_MAX_INT32 = 2 ** 31 - 1
    MAX_SEGMENT_LEN = 2048 * 4
    MAX_SEGMENT_LEN_NANO = 2048
    MAX_SEGMENT_LEN_INT64 = 2048 * 2
    MAX_FIRST_DIM_LEN = 8192
    MAX_INT32 = 2**31 - 1
    MAX_MASK_INT64 = 2**64 - 1
    # int32 num in 8*block
    OUT_MASK_NUM = 64
    OUT_MASK_NUM_INT64 = 32
    MAX_REPEAT_NUM = 255
    # 0101 mask value
    MASK_0_1_BIT64 = 6148914691236517205
    MASK_0_1_BIT32 = 1431655765
    VEC_BLOCK_NUM = 8
    # reserved ub size
    RESERVER_UB_SIZE = 8 * 1024
    TILING_ARG_NUM = 24
    TILING_ARG_BLOCK = 6

    TILING_MODE_LAST_LESS_SEGMENT_LEN = 0
    TILING_MODE_LAST_OVER_DATA_VECTOR = 1
    TILING_MODE_LAST_LESS_DATA_VECTOR = 2
    TILING_MODE_LAST_OVER_SEGMENT_LEN = 3
    TILING_MODE_LAST_AXIS_VCMP = 4
    TILING_MODE_LAST_LESS_BLOCK = 5
    TILING_MODE_NLAST_CUT_FIRST_DIM = 6
    TILING_MODE_NLAST_CUT_FIRST_DIM_AXIS_LESS = 7
    TILING_MODE_NLAST_FP_ALIGN = 8
    TILING_MODE_NLAST_CUT_LAST_DIM = 9
    TILING_MODE_NLAST_CUT_LAST_DIM_AXIS_LESS = 10
    TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM = 11
    TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_LONG_AXIS = 12
    TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_SHORT_AXIS = 13
    TILING_MODE_NO_COMPUTE = 14


class ArgCommon:
    # ori_dtype map to type
    type_mapping = {
        "float16" : "float16",
        "float32" : "float32",
        "bfloat16" : "float32",
        "int32": "int32",
        "int64" : "int64"
    }

    max_segment_len_mapping = {
        "float16" : Constant.MAX_SEGMENT_LEN,
        "float32" : Constant.MAX_SEGMENT_LEN,
        "bfloat16" : Constant.MAX_SEGMENT_LEN,
        "int32" : Constant.MAX_SEGMENT_LEN,
        "int64" : Constant.MAX_SEGMENT_LEN_INT64,
    }

    def __init__(self, is_min, dtype_x, dtype_y, is_dynamic, kernel_name, is_with_value = False) -> None:
        self.is_min = is_min
        self.block_size = int(tbe_platform.get_soc_spec("ubblock_size"))
        self.tik = tik.Tik(block_size=self.block_size)
        self.ori_dtype_x = dtype_x
        self.dtype_y = dtype_y
        self.dtype_x = self.type_mapping.get(self.ori_dtype_x, "")
        self.kernel_name = kernel_name
        self.is_with_value = is_with_value
        self.is_dynamic = is_dynamic
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.is_vccmp_support = te_platform.cce_conf.api_check_support("tik.vcmax", self.dtype_x)
        if self.dtype_x == "int64":
            # int64 vcmax/vcmin not return index
            self.is_vccmp_support = False
        self.is_data_move_pad_support = te_platform.cce_conf.api_check_support("tik.data_move_pad")
        self.is_vsel_support = te_platform.cce_conf.api_check_support("tik.vsel", self.dtype_x)
        self._check()
        self._init_data_block()
        self._init_vec_method()
        self._init_default_value()
        self._init_gm_buffer()
        self._init_ub_buffer()
        self._malloc_tiling_args()
        self._malloc_running_args()

    @staticmethod
    def _is_support_inf_nan():
        """ only return train soc, not handle infterence soc
        """
        cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        inf_nan_soc_list = ("Ascend910B", "Ascend910_93")
        return cur_cce_product in inf_nan_soc_list

    def get_tik_instance(self):
        self._do_compute()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars", {
            "ub_ele": self.ub_ele,
            "max_segment_len": self.max_segment_len,
            "core_num": self.core_num,
            "is_vccmp_support" : self.is_vccmp_support,
            "is_data_move_pad_support" : self.is_data_move_pad_support,
            "is_vsel_support" : self.is_vsel_support,
            "block_size": self.block_size,
            "segment_len": self.segment_len,
            "first_dim_segment": self.first_dim_segment
        })
        fatbin = None
        if self.is_dynamic:
            fatbin = {"tiling_key": [self.tiling_mode],
                      "tiling_key_value": [[Constant.TILING_MODE_LAST_LESS_SEGMENT_LEN],
                                           [Constant.TILING_MODE_LAST_OVER_DATA_VECTOR],
                                           [Constant.TILING_MODE_LAST_LESS_DATA_VECTOR],
                                           [Constant.TILING_MODE_LAST_OVER_SEGMENT_LEN],
                                           [Constant.TILING_MODE_LAST_AXIS_VCMP],
                                           [Constant.TILING_MODE_LAST_LESS_BLOCK],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_DIM],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_DIM_AXIS_LESS],
                                           [Constant.TILING_MODE_NLAST_FP_ALIGN],
                                           [Constant.TILING_MODE_NLAST_CUT_LAST_DIM],
                                           [Constant.TILING_MODE_NLAST_CUT_LAST_DIM_AXIS_LESS],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_LONG_AXIS],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_SHORT_AXIS],
                                           [Constant.TILING_MODE_NO_COMPUTE]]}
        self.tik.BuildCCE(kernel_name=self.kernel_name, inputs=[self.data_gm, self.axis_gm],
                          outputs=[self.result_gm_index], flowtable=[self.tiling_gm],
                          config=opt_config, extend_params={"build_multi_kernels": fatbin} if fatbin else None)
        return self.tik

    def _malloc_scalar(self, dtype, name, value=None):
        return self.tik.Scalar(dtype, name, init_value=value) if value is not None else self.tik.Scalar(dtype, name)

    def _malloc_tensor(self, dtype, shape, tname, tscope=tik.scope_ubuf):
        return self.tik.Tensor(dtype, shape, name=tname, scope=tscope)

    def _get_ceil_int(self, int1, int2):
        return (int1 + int2 - 1) // int2

    def _get_floor_int(self, int1, int2):
        return (int1 // int2 * int2)

    def _check(self):
        if self.block_size != Constant.N32 and self.block_size != Constant.N16:
            raise RuntimeError("only support ubblock_size is 32 and 16 but is %d", self.block_size)
        if self.dtype_x == "":
            raise RuntimeError("origin dtype %s not support", self.ori_dtype_x)

    def _scalar_cmp(self, a, b):
        return tik.all(a < b, ) if self.is_min else tik.all(a > b, )

    def _init_ub_buffer(self):
        self.tiling_ub = self._malloc_tensor("int64", (Constant.TILING_ARG_NUM,), "tiling_ub")

    def _init_gm_buffer(self):
        self.tiling_gm = self._malloc_tensor("int64", (Constant.TILING_ARG_NUM,), "tiling_gm", tik.scope_gm)
        self.data_gm = self._malloc_tensor(self.ori_dtype_x, (Constant.MAX_INT32,), "data_gm", tik.scope_gm)
        self.axis_gm = self._malloc_tensor("int32", (1,), "dimension", tik.scope_gm)
        self.result_gm_index = self._malloc_tensor(self.dtype_y, (Constant.MAX_INT32,), "result_gm_index", tik.scope_gm)

    def _init_data_block(self):
        self.dtype_x_size = get_bit_len(self.dtype_x) // 8
        self.dtype_y_size = get_bit_len(self.dtype_y) // 8
        self.ori_dtype_x_size = get_bit_len(self.ori_dtype_x) // 8
        if check_support_block_size_16():
            self.max_segment_len = Constant.MAX_SEGMENT_LEN_NANO
        else:
            self.max_segment_len = self.max_segment_len_mapping.get(self.ori_dtype_x, "")
        self.segment = self.max_segment_len
        self.mask_out_num = Constant.OUT_MASK_NUM
        if self.dtype_y == "int64" or self.dtype_x == "int64":
            self.mask_out_num = Constant.OUT_MASK_NUM_INT64
        self.mask_out_dtype = "uint64"
        if self.dtype_y == "int64" or self.dtype_x == "int64":
            self.mask_out_dtype = "uint32"
        self.index_out_each_block = self.block_size // self.dtype_y_size
        self.index_out_each_vector = self.index_out_each_block * Constant.VEC_BLOCK_NUM
        self.data_calc_each_block = self.block_size // self.dtype_x_size
        self.data_calc_each_vector = self.data_calc_each_block * Constant.VEC_BLOCK_NUM
        self.data_move_each_block = self.block_size // self.ori_dtype_x_size
        self.data_move_each_vector = self.data_move_each_block * Constant.VEC_BLOCK_NUM
        self.do_data_move_each_block = max(self.index_out_each_block, self.data_move_each_block)
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVER_UB_SIZE) // self.dtype_x_size
        segment_factor = 2 if self.is_with_value else 3
        if self.dtype_x == "float16" and not check_support_block_size_16():
            self.segment = self.max_segment_len * segment_factor
        # segment must within vector repeat times restriction
        self.segment_len = min(self.segment, self.data_calc_each_vector * Constant.MAX_REPEAT_NUM)
        # segment must not exceed ub size (input_tensor of size t, vcmax_tensor of size t / data_calc_each_vector * 2)
        max_segment_len = self.ub_ele // self.data_calc_each_vector * (self.data_calc_each_vector + 2); 
        max_segment_len = max_segment_len // self.block_size * self.block_size
        self.segment_len = min(self.segment_len, max_segment_len)
        # segment align to vector
        self.segment_len = (self.segment_len + self.data_calc_each_vector - 1) // \
            self.data_calc_each_vector * self.data_calc_each_vector
        # segment must not exceed ub size
        self.first_dim_segment = min(Constant.MAX_FIRST_DIM_LEN, self.ub_ele)
        self.first_dim_segment = (self.first_dim_segment + self.block_size - 1) // self.block_size * self.block_size

    def _init_vec_method(self):
        self.vcmp_func = self.tik.vmin if self.is_min else self.tik.vmax
        self.vccmp_func = self.tik.vcmin if self.is_min else self.tik.vcmax
        self.vcmpv_func = self.tik.vcmpv_gt if self.is_min else self.tik.vcmpv_lt

    def _init_default_value(self):
        self.fp16_default = Constant.SCALAR_MAX_FP16 if self.is_min else Constant.SCALAR_MIN_FP16
        self.fp32_default = Constant.SCALAR_MAX_FP32 if self.is_min else Constant.SCALAR_MIN_FP32
        if ArgCommon._is_support_inf_nan():
            self.fp32_default = np.inf if self.is_min else -np.inf
            self.fp16_default = np.inf if self.is_min else -np.inf
        self.int64_default = Constant.SCALAR_MAX_INT64 if self.is_min else Constant.SCALAR_MIN_INT64
        self.int32_default = Constant.SCALAR_MAX_INT32 if self.is_min else Constant.SCALAR_MIN_INT32
        if self.dtype_x_size == Constant.B16_BYTE:
            self.default_value = self.fp16_default
        elif self.dtype_x_size == Constant.B32_BYTE:
            self.default_value = self.fp32_default
            if self.dtype_x == "int32":
                self.default_value = self.int32_default
        else:
            self.default_value = self.int64_default

    def _malloc_tiling_args(self):
        self.tiling_mode = self._malloc_scalar("int64", "tiling_mode")
        self.first_dim_size = self._malloc_scalar("int64", "first_dim_size")
        self.axis_size = self._malloc_scalar("int64", "axis_size")
        self.last_dim_size = self._malloc_scalar("int64", "last_dim_size")
        self.act_core_num = self._malloc_scalar("int64", "act_core_num")
        self.one_core_ele = self._malloc_scalar("int64", "one_core_ele")
        self.last_core_ele = self._malloc_scalar("int64", "last_core_ele")
        self.align_num = self._malloc_scalar("int64", "align_num")
        self.axis_size_one_time = self._malloc_scalar("int64", "axis_size_one_time")
        self.loop_times = self._malloc_scalar("int64", "loop_times")
        self.tail_size = self._malloc_scalar("int64", "tail_size")
        self.one_core_segment_loop = self._malloc_scalar("int64", "one_core_segment_loop")
        self.one_core_segment_tail = self._malloc_scalar("int64", "one_core_segment_tail")
        self.one_core_segment_tail_data = self._malloc_scalar("int64", "one_core_segment_tail_data")
        self.one_core_offset = self._malloc_scalar("int64", "one_core_offset")
        self.last_core_segment_loop = self._malloc_scalar("int64", "last_core_segment_loop")
        self.last_core_segment_tail = self._malloc_scalar("int64", "last_core_segment_tail")
        self.last_core_segment_tail_data = self._malloc_scalar("int64", "last_core_segment_tail_data")
        self.last_core_offset = self._malloc_scalar("int64", "last_core_offset")
        self.running_core_num = self._malloc_scalar("int64", "running_core_num")
        self.n_inner = self._malloc_scalar("int64", "n_inner")

    def _malloc_running_args(self):
        """set init_value, Static shape scenes simplify if branches
        """
        self.core_ele = self._malloc_scalar("int64", "core_ele", 0)
        self.segment_loop = self._malloc_scalar("int64", "segment_loop", 0)
        self.segment_tail = self._malloc_scalar("int64", "segment_tail", 0)
        self.segment_tail_data = self._malloc_scalar("int64", "segment_tail_data", 0)
        self.offset_data = self._malloc_scalar("int64", "offset_data", 0)


    def _init_tiling(self):
        self.tik.data_move(self.tiling_ub, self.tiling_gm, 0, 1, Constant.TILING_ARG_BLOCK, 0, 0)
        params = [self.tiling_mode, self.first_dim_size, self.axis_size, self.last_dim_size, self.act_core_num,
                  self.one_core_ele, self.last_core_ele, self.align_num, self.axis_size_one_time, self.loop_times,
                  self.tail_size, self.one_core_segment_loop, self.one_core_segment_tail,
                  self.one_core_segment_tail_data, self.one_core_offset, self.last_core_segment_loop,
                  self.last_core_segment_tail, self.last_core_segment_tail_data, self.last_core_offset,
                  self.running_core_num, self.n_inner]
        for key, value in enumerate(params):
            value.set_as(self.tiling_ub[key])

    def _get_tail_mask(self, tail_len, mask_h, mask_l):
        """get_tail_mask
        """
        mask = self._malloc_scalar("int64", "mask", 1)
        length = self._malloc_scalar("int64", "length", tail_len)
        with self.tik.if_scope(length <= Constant.OUT_MASK_NUM):
            with self.tik.for_range(0, length):
                mask.set_as(2 * mask)
            mask.set_as(mask - 1)
            mask_h.set_as(Constant.MAX_MASK_INT64)
            mask_l.set_as(Constant.MAX_MASK_INT64 - mask)
        with self.tik.else_scope():
            mask_l.set_as(0)
            with self.tik.for_range(Constant.OUT_MASK_NUM, length):
                mask.set_as(2 * mask)
            mask.set_as(mask - 1)
            mask_h.set_as(Constant.MAX_MASK_INT64 - mask)

    def _calu_mask_by_one_zero(self, len, mask_h, mask_l):
        """calu mask with first step vcmax/vcmin result,
        get 0101010101010101010101010101010101010101010101010101010101010101 bit to vcmax/vcmin twice
        """
        mask_h.set_as(0)
        mask_l.set_as(0)
        mask = self._malloc_scalar("int64", "mask", 1)
        with self.tik.if_scope(len > 32):
            mask_l.set_as(Constant.MASK_0_1_BIT64)
            with self.tik.for_range(0, len - 32) as i:
                mask.set_as(mask * 2 * 2 + 1)
            with self.tik.if_scope(len == 64):
                mask_h.set_as(mask)
            with self.tik.else_scope():
                mask_h.set_as(mask // 4)
        with self.tik.else_scope():
            mask_h.set_as(0)
            with self.tik.for_range(0, len) as i:
                mask.set_as(mask * 2 * 2 + 1)
            with self.tik.if_scope(len == 32):
                mask_l.set_as(mask)
            with self.tik.else_scope():
                mask_l.set_as(mask // 4)

    def _calu_mask_by_repeat_times(self, len, mask_h, mask_l):
        """float16 data_each_block = 16, so base value: 0000 0000 0000 0001
        float32 data_each_block = 8, so base value 0000 0001
        """
        def get_init_mask_and_unit(unit_len):
            mask_init_value = 0
            unit_value = 2 ** unit_len
            for _ in range(Constant.VEC_BLOCK_NUM // 2):
                mask_init_value *= unit_value
                mask_init_value += 1
            return mask_init_value, unit_value

        mask_init_value, unit_value = get_init_mask_and_unit(self.data_calc_each_block)
        with self.tik.if_scope(len >= 8):
            mask_h.set_as(mask_init_value)
            mask_l.set_as(mask_init_value)
        with self.tik.else_scope():
            with self.tik.if_scope(len > 4):
                mask_h.set_as(mask_init_value)
                mask_l.set_as(mask_init_value)
                with self.tik.for_range(len, 8) as i:
                    mask_h.set_as((mask_h - 1) // unit_value)
            with self.tik.else_scope():
                mask_h.set_as(0)
                mask_l.set_as(mask_init_value)
                with self.tik.for_range(len, 4) as i:
                    mask_l.set_as((mask_l - 1) // unit_value)

    def _transpose_row_2_col(self, src_buf, dst_buf, row, col):
        if self.dtype_x_size == Constant.B16_BYTE:
            loop_times = row * col // Constant.N16 // Constant.N16
        elif self.dtype_x_size == Constant.B32_BYTE:
            loop_times = row * col * 2 // Constant.N16 // Constant.N16
        else:
            loop_times = row * col * 4 // Constant.N16 // Constant.N16
        src_buf = src_buf.reinterpret_cast_to("float16")
        dst_buf = dst_buf.reinterpret_cast_to("float16")
        with self.tik.for_range(0, loop_times) as loop_idx:
            src_list = [src_buf[16 * (i * loop_times + loop_idx)] for i in range(16)]
            dst_list = [dst_buf[16 * (i + loop_idx * Constant.N16)] for i in range(16)]
            self.tik.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

    def _transpose_col_2_row(self, src_buf, dst_buf, row, col):
        if self.dtype_x_size == Constant.B16_BYTE:
            loop_times = row * col // Constant.N16 // Constant.N16
        elif self.dtype_x_size == Constant.B32_BYTE:
            loop_times = row * col * 2 // Constant.N16 // Constant.N16
        else:
            loop_times = row * col * 4 // Constant.N16 // Constant.N16
        src_buf = src_buf.reinterpret_cast_to("float16")
        dst_buf = dst_buf.reinterpret_cast_to("float16")
        with self.tik.for_range(0, loop_times) as loop_idx:
            src_list = [src_buf[16 * (i + loop_idx * Constant.N16)] for i in range(16)]
            dst_list = [dst_buf[16 * (i * loop_times + loop_idx)] for i in range(16)]
            self.tik.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

    def _init_vec(self, ub_index, ele_num, total_len, init_value):
        dtype = ub_index.dtype
        dtype_len = get_bit_len(dtype) // 8
        repeat_num = self._get_ceil_int(total_len, ele_num)
        repeat_index = repeat_num // Constant.MAX_REPEAT_NUM
        repeat_stride = ele_num * dtype_len // self.block_size
        with self.tik.for_range(0, repeat_index) as i:
            self.tik.vector_dup(ele_num, ub_index[Constant.MAX_REPEAT_NUM * i * ele_num], init_value,
                                Constant.MAX_REPEAT_NUM, 1, repeat_stride)

        with self.tik.if_scope(total_len < ele_num):
            self.tik.vector_dup(total_len, ub_index[Constant.MAX_REPEAT_NUM * repeat_index * ele_num], init_value,
                                repeat_num - Constant.MAX_REPEAT_NUM * repeat_index, 1, repeat_stride)
        with self.tik.else_scope():
            self.tik.vector_dup(ele_num, ub_index[Constant.MAX_REPEAT_NUM * repeat_index * ele_num], init_value,
                                repeat_num - Constant.MAX_REPEAT_NUM * repeat_index, 1, repeat_stride)

    def _do_move_index_out(self, dest, src, segment):
        if self.is_data_move_pad_support:
            if get_bit_len(self.dtype_y) == 64:
                #int64 need cast
                dest = dest.reinterpret_cast_to("int8")
                src = src.reinterpret_cast_to("int8")
            self.tik.data_move_pad(dest, src, 1, segment * self.dtype_y_size, 0, 0)
        else:
            out_nbust_index = self._get_ceil_int(segment, self.index_out_each_block)
            self.tik.data_move(dest, src, 0, 1, out_nbust_index, 0, 0)

    def _do_data_move(self, dest, src, segment):
        with self.tik.new_stmt_scope():
            if self.ori_dtype_x == self.dtype_x:
                if self.is_data_move_pad_support:
                    if get_bit_len(self.dtype_x) == 64:
                        #int64 need cast
                        dest = dest.reinterpret_cast_to("int8")
                        src = src.reinterpret_cast_to("int8")
                    self.tik.data_move_pad(dest, src, 1, segment * self.dtype_x_size, 0, 0)
                else:
                    nburst_len = self._get_ceil_int(segment, self.data_move_each_block)
                    self.tik.data_move(dest, src, 0, 1, nburst_len, 0, 0)
            else:
                src_dtype = src.dtype
                dest_dtype = dest.dtype
                ub_data_tmp = self._malloc_tensor(self.ori_dtype_x, (self.segment, ), "ub_data_tmp")
                if src_dtype == self.ori_dtype_x:
                    if tbe_platform.api_check_support("tik.data_move_pad", self.ori_dtype_x):
                        self.tik.data_move_pad(ub_data_tmp, src, 1, segment * self.ori_dtype_x_size, 0, 0)
                    else:
                        nburst_len = self._get_ceil_int(segment, self.data_move_each_block)
                        self.tik.data_move(ub_data_tmp, src, 0, 1, nburst_len, 0, 0)
                    common_util.conv_s4_to_s8(self.tik, dest, ub_data_tmp, segment)
                elif dest_dtype == self.ori_dtype_x:
                    common_util.conv_s8_to_s4(self.tik, ub_data_tmp, src, segment, 'round')
                    if tbe_platform.api_check_support("tik.data_move_pad", self.ori_dtype_x):
                        self.tik.data_move_pad(dest, ub_data_tmp, 1, segment * self.ori_dtype_x_size, 0, 0)
                    else:
                        nburst_len = self._get_ceil_int(segment, self.data_move_each_block)
                        self.tik.data_move(dest, ub_data_tmp, 0, 1, nburst_len, 0, 0)

    def _do_data_move_pad_to_ub(self, dest, src, nburst, segment, nblock):
        with self.tik.new_stmt_scope():
            block_num = self._get_ceil_int(self.last_dim_size * self.ori_dtype_x_size, 32)
            if self.ori_dtype_x == self.dtype_x:
                vec_block_num = 8
                self.tik.data_move_pad(dest, src, nburst, segment * self.dtype_x_size,
                                       vec_block_num * nblock - block_num, 0)
            else:
                half_vec_block_num = 4
                src_dtype = src.dtype
                dest_dtype = dest.dtype
                ub_data_tmp = self._malloc_tensor(self.ori_dtype_x, ((nburst + 1) * self.data_calc_each_vector, ),
                                                  "ub_data_tmp")
                if src_dtype == self.ori_dtype_x:
                    self.tik.data_move_pad(ub_data_tmp, src, nburst, segment * self.ori_dtype_x_size,
                                           half_vec_block_num * nblock - block_num, 0)
                    common_util.conv_s4_to_s8(self.tik, dest, ub_data_tmp, nburst * self.data_calc_each_vector)
                elif dest_dtype == self.ori_dtype_x:
                    common_util.conv_s8_to_s4(self.tik, ub_data_tmp, src, nburst * self.data_calc_each_vector, 'round')
                    self.tik.data_move_pad(dest, ub_data_tmp, nburst, segment * self.ori_dtype_x_size,
                                           half_vec_block_num * nblock - block_num, 0)
                    
    def _do_data_move_pad_to_ub_with_loop(self, dest, src, nburst, segment, nblock):
        with self.tik.new_stmt_scope():
            block_num = self._get_ceil_int(self.last_dim_size * self.ori_dtype_x_size, 32)
            if self.ori_dtype_x == self.dtype_x:
                vec_block_num = 8
                with self.tik.for_range(0, nblock) as in_idx:
                    self.tik.data_move_pad(dest[in_idx * self.data_calc_each_vector], src[in_idx * nburst * segment],
                                           nburst, segment * self.dtype_x_size, vec_block_num * nblock - block_num, 0)
            else:
                half_vec_block_num = 4
                ub_data_tmp = self._malloc_tensor(self.ori_dtype_x, (nburst * nblock * self.data_calc_each_vector, ),
                                                  "ub_data_tmp")
                with self.tik.for_range(0, nblock) as in_idx:
                    self.tik.data_move_pad(ub_data_tmp[in_idx * self.data_calc_each_vector],
                                           src[in_idx * nburst * segment], nburst, segment * self.ori_dtype_x_size,
                                           half_vec_block_num * nblock - block_num, 0)
                common_util.conv_s4_to_s8(self.tik, dest, ub_data_tmp, nblock * nburst * self.data_calc_each_vector)

    def _do_data_move_pad_to_gm(self, dest, src, segment):
        with self.tik.new_stmt_scope():
            if self.ori_dtype_x == self.dtype_x:
                self.tik.data_move_pad(dest, src, 1, segment * self.dtype_x_size, 0, 0)
            else:
                src_dtype = src.dtype
                dest_dtype = dest.dtype
                ub_data_tmp = self._malloc_tensor(self.ori_dtype_x, (self.segment, ), "ub_data_tmp")
                if src_dtype == self.ori_dtype_x:
                    self.tik.data_move_pad(ub_data_tmp, src, 1, segment * self.ori_dtype_x_size, 0, 0)
                    common_util.conv_s4_to_s8(self.tik, dest, ub_data_tmp, segment)
                elif dest_dtype == self.ori_dtype_x:
                    common_util.conv_s8_to_s4(self.tik, ub_data_tmp, src, segment, 'round')
                    self.tik.data_move_pad(dest, ub_data_tmp, 1, segment * self.ori_dtype_x_size, 0, 0)

    # 'pylint: disable=too-many-arguments
    def _do_stride_data_move(self, dest, src, segment, nburst, align_num, is_dest_continues = False):
        if self.ori_dtype_x == self.dtype_x:
            vector_size_repeat = self._get_ceil_int(segment, self.data_move_each_vector)
            nburst_len = self._get_ceil_int(segment, self.data_move_each_block)
            src_nburst_stride = self._get_ceil_int(segment * align_num, self.data_move_each_block) - nburst_len
            dst_nburst_stride = vector_size_repeat * Constant.VEC_BLOCK_NUM - nburst_len if not is_dest_continues else 0
            dst_nburst_offset = vector_size_repeat * Constant.VEC_BLOCK_NUM if not is_dest_continues else nburst_len
            src_gap = (align_num - 1) * segment * self.dtype_x_size
            if self.is_data_move_pad_support:
                with self.tik.if_scope(src_nburst_stride != 0):
                    self.tik.data_move_pad(dest, src, nburst, segment * self.dtype_x_size, dst_nburst_stride, src_gap)
                with self.tik.else_scope():
                    with self.tik.if_scope(nburst > 1):
                        self.tik.data_move(dest, src, 0, nburst - 1, nburst_len, src_nburst_stride, dst_nburst_stride)
                    dest_offset = (nburst - 1) * dst_nburst_offset * self.data_move_each_block
                    src_offset = (nburst - 1) * align_num * segment
                    self.tik.data_move_pad(dest[dest_offset], src[src_offset], 1, segment * self.dtype_x_size, 0, 0)
            else:
                self.tik.data_move(dest, src, 0, nburst, nburst_len, src_nburst_stride, dst_nburst_stride)
        else:
            src_dtype = src.dtype
            dest_dtype = dest.dtype
            ub_data_tmp = self._malloc_tensor(self.ori_dtype_x, (self.segment, ), "ub_data_tmp")
            if src_dtype == "bfloat16" and dest_dtype == "float32":
                src_dtype_size = get_bit_len(src_dtype) // 8
                src_each_block = self.block_size // src_dtype_size
                nburst_len = self._get_ceil_int(segment, src_each_block)
                vector_size_repeat = self._get_ceil_int(segment, src_each_block * Constant.VEC_BLOCK_NUM)
                src_nburst_stride = self._get_ceil_int(segment * align_num, src_each_block) - nburst_len
                dst_nburst_stride = \
                    vector_size_repeat * Constant.VEC_BLOCK_NUM - nburst_len if not is_dest_continues else 0
                dst_nburst_offset = vector_size_repeat * Constant.VEC_BLOCK_NUM if not is_dest_continues else nburst_len
                src_gap = (align_num - 1) * segment * src_dtype_size
                if self.is_data_move_pad_support:
                    with self.tik.if_scope(src_nburst_stride != 0):
                        self.tik.data_move_pad(ub_data_tmp, src, nburst, segment * src_dtype_size,
                                               dst_nburst_stride, src_gap)
                    with self.tik.if_scope(nburst > 1):
                        self.tik.data_move(ub_data_tmp, src, 0, nburst - 1, nburst_len,
                                           src_nburst_stride, dst_nburst_stride)
                    dest_offset = (nburst - 1) * dst_nburst_offset * self.data_move_each_block
                    src_offset = (nburst - 1) * align_num * segment
                    self.tik.data_move_pad(ub_data_tmp[dest_offset], src[src_offset], 1, segment * src_dtype_size, 0, 0)
                else:
                    self.tik.data_move(ub_data_tmp, src, 0, nburst, nburst_len, src_nburst_stride, dst_nburst_stride)
                common_util.conv_s4_to_s8(self.tik, dest, ub_data_tmp, self.segment)

    def _get_cmp_mask(self, ub_first_line, ub_second_line, length, ub_mask):
        repeat = self._get_ceil_int(length, self.data_calc_each_vector)
        if ArgCommon._is_support_inf_nan() and self.dtype_x in ("float16", "float32"):
            # only support 910B and dtype in float
            mask_out_repeat = self._get_ceil_int(length, self.mask_out_num)
            ub_mask_first_no_nan = self._malloc_tensor(self.mask_out_dtype, (mask_out_repeat,), "ub_mask_first_no_nan")
            self.tik.vcmpv_eq(ub_mask_first_no_nan, ub_first_line, ub_first_line, repeat, 1, 1, 8, 8)
            ub_mask_first_no_nan_cast = ub_mask_first_no_nan.reinterpret_cast_to("uint16")

            ub_mask_second_no_nan = self._malloc_tensor(self.mask_out_dtype, (mask_out_repeat,), "ub_mask_sec_no_nan")
            self.tik.vcmpv_eq(ub_mask_second_no_nan, ub_second_line, ub_second_line, repeat, 1, 1, 8, 8)
            ub_mask_second_no_nan_cast = ub_mask_second_no_nan.reinterpret_cast_to("uint16")

            self.vcmpv_func(ub_mask, ub_first_line, ub_second_line, repeat, 1, 1, 8, 8)
            ub_mask_cast = ub_mask.reinterpret_cast_to("uint16")
            #vec_not/vec_and only support uint16/int16
            vec_calc_num = self._get_ceil_int(self.mask_out_num, Constant.N16) * mask_out_repeat
            vec_calc_repeat = vec_calc_num // Constant.N128
            with self.tik.if_scope(vec_calc_repeat > 0):
                self.tik.vec_not(Constant.N128, ub_mask_second_no_nan_cast, ub_mask_second_no_nan_cast, vec_calc_repeat,
                                 8, 8)
                self.tik.vec_and(Constant.N128, ub_mask_first_no_nan_cast, ub_mask_second_no_nan_cast,
                                 ub_mask_first_no_nan_cast, vec_calc_repeat, 8, 8, 8)
                self.tik.vec_or(Constant.N128, ub_mask_cast, ub_mask_first_no_nan_cast, ub_mask_cast, vec_calc_repeat,
                                8, 8, 8)
            offset = vec_calc_repeat * Constant.N128
            extra_num = vec_calc_num - offset
            with self.tik.if_scope(extra_num > 0):
                self.tik.vec_not(extra_num, ub_mask_second_no_nan_cast[offset], ub_mask_second_no_nan_cast[offset], 1,
                                 8, 8)
                self.tik.vec_and(extra_num, ub_mask_first_no_nan_cast[offset], ub_mask_second_no_nan_cast[offset],
                                 ub_mask_first_no_nan_cast[offset], 1, 8, 8, 8)
                self.tik.vec_or(extra_num, ub_mask_cast[offset], ub_mask_first_no_nan_cast[offset],
                                ub_mask_cast[offset], 1, 8, 8, 8)
        elif self.dtype_x in ("int32",):
            self.vcmp_func(self.data_calc_each_vector, ub_second_line, ub_first_line, ub_second_line, repeat, 1, 1, 1,
                           8, 8, 8)

            mask_out_repeat = self._get_ceil_int(length, self.mask_out_num)
            self.tik.vcmpv_eq(ub_mask, ub_second_line, ub_first_line, repeat, 1, 1, 8, 8)

            ub_mask_cast = ub_mask.reinterpret_cast_to("uint16")
            vec_calc_num = self._get_ceil_int(self.mask_out_num, Constant.N16) * mask_out_repeat
            vec_calc_repeat = vec_calc_num // Constant.N128
            with self.tik.if_scope(vec_calc_repeat > 0):
                self.tik.vec_not(Constant.N128, ub_mask_cast, ub_mask_cast, vec_calc_repeat, 8, 8)
            offset = vec_calc_repeat * Constant.N128
            extra_num = vec_calc_num - offset
            with self.tik.if_scope(extra_num > 0):
                self.tik.vec_not(extra_num, ub_mask_cast[offset], ub_mask_cast[offset], 1, 8, 8)
        else:
            self.vcmpv_func(ub_mask, ub_first_line, ub_second_line, repeat, 1, 1, 8, 8)

    def _fill_src_offset(self, idx_ub, idx_num, vector_num=64):
        """
        fill 0,1,2  .... (idx_num -1) in idx_ub
        """
        vector_num_ub = self._malloc_tensor(idx_ub.dtype, (vector_num,), "vector_num_ub")
        factor_ub = self._malloc_tensor(idx_ub.dtype, (idx_num,), "factor_ub")
        for _idx in range(vector_num // 8):
            idx_ub[_idx].set_as(_idx)
        self.tik.vector_dup(vector_num, vector_num_ub, vector_num // 8, 1, 1, 8)
        with self.tik.for_range(1, 8) as add_idx:
            add_offset = add_idx * vector_num // 8
            self.tik.vadd(vector_num // 8, idx_ub[add_offset:], vector_num_ub, idx_ub[add_offset - (vector_num // 8):],
                          1, 1, 1, 1, 8, 0, 8)

        self.tik.vector_dup(vector_num, vector_num_ub, vector_num, 1, 1, 8)
        idx_vector_num = (idx_num + vector_num - 1) // vector_num
        with self.tik.for_range(1, idx_vector_num) as add_idx:
            add_offset = add_idx * vector_num
            self.tik.vadd(vector_num, idx_ub[add_offset:], vector_num_ub, idx_ub[add_offset - vector_num:],
                          1, 1, 1, 1, 8, 0, 8)

        repeat_times = self._get_ceil_int(idx_num, vector_num)
        self.tik.vector_dup(vector_num, factor_ub, self.dtype_x_size, repeat_times, 1, 8)
        self.tik.vmul(vector_num, idx_ub, idx_ub, factor_ub, repeat_times, 1, 1, 1, 8, 8, 8)

    def _set_running_params(self, core_idx):
        with self.tik.if_scope(core_idx <= self.act_core_num - 1):
            with self.tik.if_scope(core_idx < self.act_core_num - 1):
                self.core_ele.set_as(self.one_core_ele)
                self.segment_loop.set_as(self.one_core_segment_loop)
                self.segment_tail.set_as(self.one_core_segment_tail)
                self.segment_tail_data.set_as(self.one_core_segment_tail_data)
                self.offset_data.set_as(self.one_core_offset)
                self.n_inner.set_as(self.n_inner)
            with self.tik.else_scope():
                self.core_ele.set_as(self.last_core_ele)
                self.segment_loop.set_as(self.last_core_segment_loop)
                self.segment_tail.set_as(self.last_core_segment_tail)
                self.segment_tail_data.set_as(self.last_core_segment_tail_data)
                self.offset_data.set_as(self.last_core_offset)

    def do_compute_nlast_axis_cut_by_first_and_last_dim_move_direct(self, core_idx, max_first_dim_len, loop_func):
        segment = max_first_dim_len // self.data_calc_each_vector

        def _run(segment_len, segment_index):
            core_in_offset = self.one_core_ele * core_idx
            seg_in_offset = (core_in_offset + segment_index * segment) * self.axis_size * self.last_dim_size
            seg_out_offset = (core_in_offset + segment_index * segment) * self.last_dim_size
            loop_func(segment_len, seg_in_offset, seg_out_offset)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(segment, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    def do_loop_compute_for_last_axis_copy_one_time(self, seg_offset, result_index, result_value = None):
        """reduce last dim axis size over data_each_block * 2 and below 8192
        need to call vcmax/vcmin twice
        """
        max_data_len = self.data_calc_each_vector * self.data_calc_each_vector // 2
        ub_data = self._malloc_tensor(self.dtype_x, (max_data_len,), "ub_data")
        gm_in_offset = seg_offset * self.axis_size
        self._do_data_move(ub_data, self.data_gm[gm_in_offset], self.axis_size)

        mask_h = self._malloc_scalar("int64", "mask_h")
        mask_l = self._malloc_scalar("int64", "mask_l")

        tail = self.axis_size % self.data_calc_each_vector
        with self.tik.if_scope(tail != 0):
            self._get_tail_mask(tail, mask_h, mask_l)
            offset = self.axis_size // self.data_calc_each_vector
            if check_support_block_size_16():
                mask_h.set_as(0)
            self.tik.vector_dup([mask_h, mask_l], ub_data[offset * self.data_calc_each_vector], self.default_value,
                                1, 1, 8)

        ub_result = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_result")
        repeat_times = self._get_ceil_int(self.axis_size, self.data_calc_each_vector)
        self.vccmp_func(self.data_calc_each_vector, ub_result, ub_data, repeat_times, 1, 1, 8)

        self._calu_mask_by_one_zero(repeat_times, mask_h, mask_l)
        if check_support_block_size_16():
            mask_h.set_as(0)
        ub_second_result = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_second_result")
        self.vccmp_func([mask_h, mask_l], ub_second_result, ub_result, 1, 1, 1, 8)
        second_result_index = self._malloc_scalar("uint16", "second_result_index", ub_second_result[1])
        first_result_index = self._malloc_scalar("uint16", "first_result_index", ub_result[second_result_index + 1])
        result_index.set_as(second_result_index * self.data_calc_each_vector // 2 + first_result_index)
        if self.is_with_value:
            result_value.set_as(ub_result[second_result_index])

    def do_compute_for_last_axis_copy_one_time(self, core_idx):
        ub_result_index = self._malloc_tensor(self.dtype_y, (self.first_dim_segment,), "ub_result_index")

        def _run(segment_len, segment_index):
            with self.tik.for_range(0, segment_len) as idx:
                seg_offset = core_idx * self.one_core_ele + idx + self.first_dim_segment * segment_index
                result_index = self._malloc_scalar(self.dtype_y, "result_index")
                self.do_loop_compute_for_last_axis_copy_one_time(seg_offset, result_index)
                ub_result_index[idx] = result_index

            gm_out_offset = core_idx * self.one_core_ele + self.first_dim_segment * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(self.first_dim_segment, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)


    def do_loop_compute_last_axis_more_vector(self, seg_offset, segment_len, ub_result_index, ub_result_value = None):
        """axis size between one data_each_vector and two data_each_vector
        max vector_size_repeat twice, and vcmax/vcmin twice
        Move in data in align_num alignment, each time segment_align_num rows are moved,
        looping align_num times, such as processing 0, align_num, align_num*2....row data for the first time
        """
        vector_size_repeat = self._get_ceil_int(self.axis_size, self.data_calc_each_vector)
        vector_size = vector_size_repeat * self.data_calc_each_vector
        result_index = self._malloc_scalar(self.dtype_y, "result_index")

        segment_align_num = segment_len // self.align_num
        segment_align_tail = segment_len % self.align_num

        mask_h = self._malloc_scalar("int64", "mask_h")
        mask_l = self._malloc_scalar("int64", "mask_l")

        with self.tik.for_range(0, self.align_num) as align_idx:
            ub_data = self._malloc_tensor(self.dtype_x, (self.segment,), "ub_data")
            ub_second_result_size = self._get_ceil_int(
                self._get_ceil_int(self.segment, self.data_calc_each_vector) * 2, self.data_calc_each_vector) * \
                    self.data_calc_each_vector
            ub_second_result = self._malloc_tensor(self.dtype_x, (ub_second_result_size,), "ub_second_result")
            nburst = self._malloc_scalar("uint32", "nburst", segment_align_num)
            with self.tik.if_scope(align_idx < segment_align_tail):
                nburst.set_as(segment_align_num + 1)

            with self.tik.if_scope(nburst >= 1):
                gm_in_offset = (seg_offset + align_idx) * self.axis_size
                self._do_stride_data_move(ub_data, self.data_gm[gm_in_offset], self.axis_size, nburst, self.align_num)

                tail = self.axis_size % self.data_calc_each_vector
                with self.tik.if_scope(tail != 0):
                    self._get_tail_mask(tail, mask_h, mask_l)
                    if check_support_block_size_16():
                        mask_h.set_as(0)
                    offset = self.axis_size // self.data_calc_each_vector
                    self.tik.vector_dup([mask_h, mask_l], ub_data[offset * self.data_calc_each_vector],
                                        self.default_value, nburst, 1, vector_size_repeat * Constant.VEC_BLOCK_NUM)
                repeat_times = self._get_ceil_int(self.axis_size, self.data_calc_each_vector)
                # 8 * 8 means block_size(32B) * 8 / (vcmax/vcmin des_rep_stride)4B
                dst_rep_stride = self.data_calc_each_vector // 2
                self.vccmp_func(self.data_calc_each_vector, ub_data, ub_data, vector_size_repeat * nburst,
                        dst_rep_stride, 1, 8)
                self._calu_mask_by_repeat_times(repeat_times, mask_h, mask_l)
                self.vccmp_func([mask_h, mask_l], ub_second_result, ub_data, nburst, 1, 8, repeat_times * 8)
                with self.tik.for_range(0, nburst) as out_idx:
                    second_cmp_index = self._malloc_scalar("uint16", "second_cmp_index",
                                                           ub_second_result[out_idx * 2 + 1])
                    first_cmp_index = self._malloc_scalar("uint16", "first_cmp_index")
                    first_cmp_index.set_as(ub_data[vector_size * out_idx + second_cmp_index * 8 + 1])
                    result_index.set_as(second_cmp_index * 8 + first_cmp_index)
                    ub_result_index[out_idx * self.align_num + align_idx].set_as(result_index)
                    if self.is_with_value:
                        first_cmp_value = self._malloc_scalar(self.dtype_x, "first_cmp_value")
                        first_cmp_value.set_as(ub_data[vector_size * out_idx + second_cmp_index * 8])
                        ub_result_value[out_idx * self.align_num + align_idx].set_as(first_cmp_value)

    def do_compute_last_axis_more_vector(self, core_idx):
        ub_result_index = self._malloc_tensor(self.dtype_y, (self.max_segment_len,), "ub_result_index")

        def _run(segment_len, segment_index):
            seg_offset = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            self.do_loop_compute_last_axis_more_vector(seg_offset, segment_len, ub_result_index)
            gm_out_offset = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(self.axis_size_one_time, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)


    def do_loop_compute_last_axis_less_vector(self, seg_offset, segment_len, ub_result_index, ub_result_value = None):
        """axis size less one data_each_vector, call vcmax/vcmin one time 
        Move in data in align_num alignment, each time segment_align_num rows are moved,
        looping align_num times, such as processing 0, align_num, align_num*2....row data for the first time
        """
        segment_align_num = segment_len // self.align_num
        segment_align_tail = segment_len % self.align_num

        with self.tik.for_range(0, self.align_num) as align_idx:
            ub_data = self._malloc_tensor(self.dtype_x, (self.segment,), "ub_data")
            nburst = self._malloc_scalar("uint16", "nburst", segment_align_num)
            repeat_block = self._malloc_scalar("uint16", "repeat_block")
            repeat_times = self._malloc_scalar("uint16", "repeat_times")
            with self.tik.if_scope(align_idx < segment_align_tail):
                nburst.set_as(segment_align_num + 1)
            real_segment = self._get_ceil_int(self.axis_size, self.data_move_each_block) * self.data_move_each_block
            repeat_block.set_as(self._get_ceil_int(real_segment, self.data_calc_each_block))
            repeat_times.set_as(nburst)
            with self.tik.if_scope(self.align_num == 1):
                repeat_times.set_as(segment_len)

            gm_in_offset = (seg_offset + align_idx) * self.axis_size
            with self.tik.if_scope(nburst >= 1):
                with self.tik.if_scope(self.align_num == 1):
                    self._do_data_move(ub_data, self.data_gm[gm_in_offset], self.axis_size * nburst)
                with self.tik.else_scope():
                    self._do_stride_data_move(ub_data, self.data_gm[gm_in_offset], self.axis_size, nburst,
                                            self.align_num, True)
                self.vccmp_func(self.axis_size, ub_data, ub_data, repeat_times, 1, 1, repeat_block)
                with self.tik.for_range(0, repeat_times) as idx:
                    cmp_index = self._malloc_scalar("uint16", "cmp_index")
                    cmp_index.set_as(ub_data[idx * 2 + 1])
                    ub_result_index[idx * self.align_num + align_idx].set_as(cmp_index)
                    if self.is_with_value:
                        cmp_value = self._malloc_scalar(self.dtype_x, "cmp_value")
                        cmp_value.set_as(ub_data[idx * 2])
                        ub_result_value[idx * self.align_num + align_idx].set_as(cmp_value)

    def do_compute_last_axis_less_vector(self, core_idx):
        ub_result_index = self._malloc_tensor(self.dtype_y, (self.max_segment_len,), "ub_result_int32")

        def _run(segment_len, segment_index):
            seg_offset = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            self.do_loop_compute_last_axis_less_vector(seg_offset, segment_len, ub_result_index)
            gm_out_offset = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(self.axis_size_one_time, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)


    def do_loop_compute_last_axis_over_segment(self, segment, loop_idx, seg_offset, tmp_result_value, tmp_result_index):
        """axis size over segment, call vcmax/vcmin two or thread times according to the repeat times
        """
        segment_size = self._malloc_scalar("int32", "segment_size", segment)
        # ub_result is used to store the results of the first VCMAX/VCMIN
        ub_result_size = self._get_ceil_int(
            self._get_ceil_int(self.segment_len, self.data_calc_each_vector) * 2, self.data_calc_each_vector) * \
                self.data_calc_each_vector
        ub_result = self._malloc_tensor(self.dtype_x, (ub_result_size,), "ub_result")
        ub_data = self._malloc_tensor(self.dtype_x, (self.segment_len,), "ub_data")

        gm_in_offset = loop_idx * self.segment_len + seg_offset * self.axis_size
        self._do_data_move(ub_data, self.data_gm[gm_in_offset], segment_size)

        mask_h = self._malloc_scalar("int64", "mask_h")
        mask_l = self._malloc_scalar("int64", "mask_l")
        tail = segment_size % self.data_calc_each_vector
        with self.tik.if_scope(tail != 0):
            self._get_tail_mask(tail, mask_h, mask_l)
            if check_support_block_size_16():
                mask_h.set_as(0)
            offset = segment_size // self.data_calc_each_vector
            self.tik.vector_dup([mask_h, mask_l], ub_data[offset * self.data_calc_each_vector], self.default_value,
                                1, 1, 8)

        repeat_times = self._get_ceil_int(segment_size, self.data_calc_each_vector)
        self.vccmp_func(self.data_calc_each_vector, ub_result, ub_data, repeat_times, 1, 1, 8)

        cmp_index = self._malloc_scalar(self.dtype_y, "cmp_index")
        vccmp_value_size = self.data_calc_each_vector // 2
        with self.tik.if_scope(repeat_times >= vccmp_value_size):
            repeat_times_twice = self._get_ceil_int(repeat_times, vccmp_value_size)
            repeat_tail_twice = (repeat_times * 2) % self.data_calc_each_vector
            with self.tik.if_scope(repeat_tail_twice != 0):
                self._get_tail_mask(repeat_tail_twice, mask_h, mask_l)
                if check_support_block_size_16():
                    mask_h.set_as(0)
                offset = repeat_times * 2 // self.data_calc_each_vector
                self.tik.vector_dup([mask_h, mask_l], ub_result[offset * self.data_calc_each_vector],
                                    self.default_value, 1, 1, 8)

            ub_second_result = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_second_result")
            mask_l.set_as(Constant.MASK_0_1_BIT64)
            self.vccmp_func([mask_l, mask_l], ub_second_result, ub_result, repeat_times_twice, 1, 1, 8)
            ub_third_result = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_third_result")
            self._calu_mask_by_one_zero(repeat_times_twice % vccmp_value_size, mask_h, mask_l)
            if check_support_block_size_16():
                mask_h.set_as(0)
            self.vccmp_func([mask_h, mask_l], ub_third_result, ub_second_result, 1, 1, 1, 8)

            third_cmp_index = self._malloc_scalar("uint16", "third_cmp_index", ub_third_result[1])
            second_cmp_index = self._malloc_scalar("uint16", "second_cmp_index")
            second_cmp_index.set_as(ub_second_result[third_cmp_index + 1])
            first_cmp_index = self._malloc_scalar("uint16", "first_cmp_index")
            first_cmp_index.set_as(ub_result[third_cmp_index * vccmp_value_size + second_cmp_index + 1])
            cmp_index.set_as(third_cmp_index * vccmp_value_size * vccmp_value_size +
                             second_cmp_index * vccmp_value_size + first_cmp_index)
        with self.tik.else_scope():
            self._calu_mask_by_one_zero(repeat_times % vccmp_value_size, mask_h, mask_l)
            if check_support_block_size_16():
                mask_h.set_as(0)
            ub_second_result = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_second_result")
            self.vccmp_func([mask_h, mask_l], ub_second_result, ub_result, 1, 1, 1, 8)
            second_cmp_index = self._malloc_scalar("uint16", "second_cmp_index", ub_second_result[1])
            first_cmp_index = self._malloc_scalar("uint16", "first_cmp_index")
            first_cmp_index.set_as(ub_result[second_cmp_index + 1])
            cmp_index.set_as(second_cmp_index * vccmp_value_size + first_cmp_index)

        ub_result_cmp = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_result_cmp")
        ub_index = self._malloc_tensor(self.dtype_y, (16,), "ub_index")
        ub_result_cmp[0].set_as(tmp_result_value)
        ub_result_cmp[1].set_as(ub_data[cmp_index])
        ub_index[0].set_as(tmp_result_index)
        ub_index[1].set_as(cmp_index + loop_idx * self.segment_len)
        self.vccmp_func(2, ub_result_cmp, ub_result_cmp, 1, 1, 1, 8)
        tmp_cmp_index = self._malloc_scalar("uint16", "tmp_cmp_index", ub_result_cmp[1])
        tmp_result_index.set_as(ub_index[tmp_cmp_index])
        tmp_result_value.set_as(ub_result_cmp[0])

    def do_compute_last_axis_over_segment(self, core_idx):
        """Two layers of circulation,
        the outer loop is divided by first_dim,
        and the inner loop is divided by axis_size by segment
        """
        ub_result_index = self._malloc_tensor(self.dtype_y, (Constant.MAX_FIRST_DIM_LEN,), "ub_result_index")
        default_result_value = self._malloc_tensor(self.dtype_x, (16,), "default_result_value")
        self.tik.vector_dup(16, default_result_value, self.default_value, 1, 1, 8)
        def _run(segment_len, segment_index):
            with self.tik.for_range(0, segment_len) as idx:
                seg_offset = core_idx * self.one_core_ele + idx + Constant.MAX_FIRST_DIM_LEN * segment_index
                tmp_result_index = self._malloc_scalar(self.dtype_y, "tmp_result_index", 0)
                tmp_result_value = self._malloc_scalar(self.dtype_x, "tmp_result_value")
                tmp_result_value.set_as(default_result_value[0])
                with self.tik.for_range(0, self.loop_times) as loop_idx:
                    self.do_loop_compute_last_axis_over_segment(
                        self.segment_len, loop_idx, seg_offset,
                        tmp_result_value, tmp_result_index
                        )
                with self.tik.if_scope(self.tail_size != 0):
                    self.do_loop_compute_last_axis_over_segment(self.tail_size, self.loop_times, seg_offset,
                                                                tmp_result_value, tmp_result_index)
                ub_result_index[idx] = tmp_result_index
            gm_out_offset = core_idx * self.one_core_ele + Constant.MAX_FIRST_DIM_LEN * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(Constant.MAX_FIRST_DIM_LEN, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    # 'pylint: disable=too-many-arguments
    def do_loop_compute_last_axis_vcmp(self, segment_size, loop_idx, seg_offset, tmp_result_value, tmp_result_index):
        """In this scenario, because the vcmax/vcmin instruction does not support float32,
        use the vmax/vmin +scalar comparison to obtain the maximum and minimum values
        """
        ub_data = self._malloc_tensor(self.dtype_x, (self.segment,), "ub_data")
        gm_in_offset = loop_idx * self.segment + seg_offset * self.axis_size
        self._do_data_move(ub_data, self.data_gm[gm_in_offset], segment_size)

        mask_h = self._malloc_scalar("int64", "mask_h")
        mask_l = self._malloc_scalar("int64", "mask_l")
        tail = segment_size % self.data_calc_each_vector
        with self.tik.if_scope(tail != 0):
            self._get_tail_mask(tail, mask_h, mask_l)
            offset = segment_size // self.data_calc_each_vector
            self.tik.vector_dup([mask_h, mask_l], ub_data[offset * self.data_calc_each_vector], self.default_value,
                                1, 1, 8)
        ub_mask = self._malloc_tensor(self.mask_out_dtype, (self.segment // self.mask_out_num,), "ub_mask")
        ub_value = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_value")
        self.tik.vector_dup(self.data_calc_each_vector, ub_value, self.default_value, 1, 1, 8)
        ub_index = self._malloc_tensor(self.dtype_y, (self.data_calc_each_vector,), "ub_idx")
        self._init_vec(ub_index, self.mask_out_num, self.data_calc_each_vector, 0)
        repeat = self._get_ceil_int(segment_size, self.data_calc_each_vector)

        if self.dtype_x == "int64":
            ub_value_dst = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_value_dst")
            with self.tik.for_range(0, repeat) as i:
                index = repeat - 1 - i
                self.vcmp_func(self.data_calc_each_vector, ub_value_dst, ub_data[index * self.data_calc_each_vector],
                               ub_value, 1, 1, 1, 1, 0, 8, 0)
                self.tik.vcmpv_eq(ub_mask, ub_value_dst, ub_data[index * self.data_calc_each_vector], 1, 1, 1, 0, 8)
                self.tik.data_move(ub_value, ub_value_dst, 0, 1, 8, 0, 0)
                mask = self._malloc_scalar(self.mask_out_dtype, "mask", ub_mask[0])
                with self.tik.if_scope(mask != 0):
                    self.tik.vector_dup([mask, mask], ub_index, index * self.data_calc_each_vector, 1, 1, 8)
        else:
            self.vcmp_func(self.data_calc_each_vector, ub_value, ub_data, ub_value, repeat, 1, 1, 1, 0, 8, 0)
            self.tik.vcmpv_eq(ub_mask, ub_value, ub_data, repeat, 1, 1, 0, 8)

            if self.dtype_y == "int64":
                ub_mask_cast = ub_mask.reinterpret_cast_to("uint32")
                with self.tik.for_range(0, repeat) as i:
                    offset = repeat - 1 - i
                    index = offset * 2
                    mask = self._malloc_scalar("uint32", "mask", ub_mask_cast[index])
                    with self.tik.if_scope(mask != 0):
                        self.tik.vector_dup([mask, mask], ub_index, offset * self.data_calc_each_vector, 1, 1, 8)
                    mask.set_as(ub_mask_cast[index + 1])
                    with self.tik.if_scope(mask != 0):
                        self.tik.vector_dup([mask, mask], ub_index[32], offset * self.data_calc_each_vector, 1, 1, 8)
            else:
                with self.tik.for_range(0, repeat) as i:
                    index = repeat - 1 - i
                    mask = self._malloc_scalar(self.mask_out_dtype, "mask", ub_mask[index])
                    with self.tik.if_scope(mask != 0):
                        self.tik.vector_dup([mask, mask], ub_index, index * self.data_calc_each_vector, 1, 1, 8)

        cmp_value = self._malloc_scalar(self.dtype_x, "cmp_value", ub_value[0])
        cmp_index = self._malloc_scalar(self.dtype_y, "cmp_index", ub_index[0])
        scalar_valid = self._malloc_scalar("int32", "scalar_valid")
        with self.tik.if_scope(segment_size > self.data_calc_each_vector):
            scalar_valid.set_as(self.data_calc_each_vector)
        with self.tik.else_scope():
            scalar_valid.set_as(segment_size)

        with self.tik.for_range(1, scalar_valid) as i:
            tmp_cmp_value = self._malloc_scalar(self.dtype_x, "tmp_cmp_value", ub_value[i])
            tmp_cmp_index = self._malloc_scalar(self.dtype_y, "tmp_cmp_index", ub_index[i])
            with self.tik.if_scope(self._scalar_cmp(tmp_cmp_value, cmp_value)):
                cmp_value.set_as(ub_value[i])
                cmp_index.set_as(tmp_cmp_index + i)
            with self.tik.if_scope(tik.all(tmp_cmp_value == cmp_value, tmp_cmp_index + i < cmp_index)):
                cmp_value.set_as(ub_value[i])
                cmp_index.set_as(tmp_cmp_index + i)
        with self.tik.if_scope(self._scalar_cmp(cmp_value, tmp_result_value)):
            tmp_result_value.set_as(cmp_value)
            tmp_result_index.set_as(cmp_index + loop_idx * self.segment)

    def do_compute_last_axis_vcmp(self, core_idx):
        """Two layers of circulation,
        the outer loop is divided by first_dim,
        and the inner loop is divided by axis_size by segment
        """
        ub_result_index = self._malloc_tensor(self.dtype_y, (Constant.MAX_FIRST_DIM_LEN,), "ub_result_index")
        default_result_value = self._malloc_tensor(self.dtype_x, (16,), "default_result_value")
        self.tik.vector_dup(16, default_result_value, self.default_value, 1, 1, 8)

        def _run(segment_len, segment_index):
            with self.tik.new_stmt_scope():
                with self.tik.for_range(0, segment_len) as idx:
                    result_index = self._malloc_scalar(self.dtype_y, "result_index", 0)
                    result_value = self._malloc_scalar(self.dtype_x, "result_value")
                    result_value.set_as(default_result_value[0])
                    seg_offset = core_idx * self.one_core_ele + idx + Constant.MAX_FIRST_DIM_LEN * segment_index
                    with self.tik.for_range(0, self.loop_times) as loop:
                        self.do_loop_compute_last_axis_vcmp(self.segment, loop, seg_offset, result_value, result_index)
                    with self.tik.if_scope(self.tail_size != 0):
                        self.do_loop_compute_last_axis_vcmp(self.tail_size, self.loop_times, seg_offset, result_value,
                                                            result_index)
                    ub_result_index[idx] = result_index

                gm_out_offset = core_idx * self.one_core_ele + Constant.MAX_FIRST_DIM_LEN * segment_index
                self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(Constant.MAX_FIRST_DIM_LEN, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    def do_loop_compute_nlast_axis(self, segment, gm_in_offset, gm_out_offset):
        """Using the vcmpv_func and vmax/vmin instructions,
        cycle through axis_size to get the maximum and minimum values
        """
        ub_data = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_data")
        self._do_data_move(ub_data, self.data_gm[gm_in_offset], segment)
        with self.tik.new_stmt_scope():
            index_out_repeat = self._get_ceil_int(segment, self.mask_out_num)
            repeat = self._get_ceil_int(segment, self.data_calc_each_vector)
            ub_index = self._malloc_tensor(self.dtype_y, (self.max_segment_len,), "ub_index")
            self._init_vec(ub_index, self.mask_out_num, self.max_segment_len, 0)
            with self.tik.for_range(1, self.axis_size) as axis_i:
                ub_tmp = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_tmp")
                ub_mask = self._malloc_tensor(self.mask_out_dtype, (self.max_segment_len // self.mask_out_num,),
                                              "ub_mask")
                self._do_data_move(ub_tmp, self.data_gm[gm_in_offset + axis_i * self.last_dim_size], segment)

                self._get_cmp_mask(ub_data, ub_tmp, segment, ub_mask)
                with self.tik.for_range(0, index_out_repeat) as i:
                    mask_l = self._malloc_scalar(self.mask_out_dtype, "mask_l")
                    mask_l.set_as(ub_mask[i])
                    with self.tik.if_scope(mask_l != 0):
                        self.tik.vector_dup([mask_l, mask_l], ub_index[i * self.mask_out_num], axis_i, 1, 1, 8)
                self.vcmp_func(self.data_calc_each_vector, ub_data, ub_data, ub_tmp, repeat, 1, 1, 1, 8, 8, 8)
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_index, segment)
        return ub_data

    def do_loop_compute_nlast_axis_less(self, segment, gm_in_offset, gm_out_offset):
        """Using the vcmpv_func and vmax/vmin instructions,
        cycle through axis_size to get the maximum and minimum values
        """
        ub_data = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_data")
        self._do_data_move(ub_data, self.data_gm[gm_in_offset], segment)
        repeat = self._get_ceil_int(segment, self.data_calc_each_vector)
        ub_vec_index = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_vec_index")
        ub_index_float = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_index_float")
        self.tik.vector_dup(self.data_calc_each_vector, ub_index_float, 0, repeat, 1, 8)
        self.tik.vector_dup(self.data_calc_each_vector, ub_vec_index, 0, 1, 1, 8)
        with self.tik.for_range(1, self.axis_size) as axis_i:
            ub_tmp = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_tmp")
            ub_mask = self._malloc_tensor(self.mask_out_dtype, (self.max_segment_len // self.mask_out_num,), "ub_mask")
            self._do_data_move(ub_tmp, self.data_gm[gm_in_offset + axis_i * self.last_dim_size], segment)
            self.tik.vadds(self.data_calc_each_vector, ub_vec_index, ub_vec_index, 1, 1, 1, 1, 8, 8)
            with self.tik.for_range(0, repeat) as i:
                offset = i * self.data_calc_each_vector
                self._get_cmp_mask(ub_data[offset], ub_tmp[offset], self.data_calc_each_vector, ub_mask)
                self.tik.vec_sel(self.data_calc_each_vector, 0, ub_index_float[offset], ub_mask, ub_vec_index,
                                 ub_index_float[offset], 1, 8, 0, 0)
            self.vcmp_func(self.data_calc_each_vector, ub_data, ub_data, ub_tmp, repeat, 1, 1, 1, 8, 8, 8)
        ub_index = self._malloc_tensor(self.dtype_y, (self.max_segment_len,), "ub_index")
        self._init_vec(ub_index, self.mask_out_num, self.max_segment_len, 0)
        util_tik_comm_func.tik_func_vconv(self.tik, ub_index, ub_index_float, self.max_segment_len, mode="round")
        self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_index, segment)
        return ub_data

    def do_compute_nlast_axis_cut_by_first_dim(self, core_idx, loop_func):
        """The outer loop is processed by first_dim dividing the nucleus,
        and the inner loop is divided by last_dim segment,
        which needs to be considered when the tail block is not aligned, and the data needs to be rolled back
        """
        def _run(core_offset, segment_loop, segment_tail, segment_tail_data, offset_data):
            with self.tik.for_range(0, segment_loop) as segm_i:
                gm_in_offset = core_offset * self.axis_size * self.last_dim_size + segm_i * self.max_segment_len
                gm_out_offset = core_offset * self.last_dim_size + segm_i * self.max_segment_len
                loop_func(self.max_segment_len, gm_in_offset, gm_out_offset)

            with self.tik.if_scope(segment_tail != 0):
                if tbe_platform.api_check_support("tik.data_move_pad", self.ori_dtype_x) and \
                    tbe_platform.api_check_support("tik.data_move_pad", self.dtype_y):
                    gm_in_offset = core_offset * self.axis_size * self.last_dim_size + \
                        segment_loop * self.max_segment_len
                    gm_out_offset = core_offset * self.last_dim_size + segment_loop * self.max_segment_len
                    loop_func(segment_tail, gm_in_offset, gm_out_offset)
                else:
                    with self.tik.if_scope(
                        tik.all(segment_tail_data % self.do_data_move_each_block != 0,
                                segment_tail_data > self.do_data_move_each_block)):
                        pro_len = self._get_ceil_int(segment_tail_data, 2)
                        pro_len = (self._get_ceil_int(pro_len, self.do_data_move_each_block) *
                                   self.do_data_move_each_block)
                        offset = segment_tail_data - pro_len
                        gm_in_offset = (core_offset * self.axis_size * self.last_dim_size +
                                        segment_loop * self.max_segment_len)
                        gm_out_offset = core_offset * self.last_dim_size + segment_loop * self.max_segment_len
                        with self.tik.new_stmt_scope():
                            loop_func(pro_len, gm_in_offset, gm_out_offset)
                        gm_in_offset = (core_offset * self.axis_size * self.last_dim_size + segment_loop *
                            self.max_segment_len + offset + offset_data)
                        gm_out_offset = (core_offset * self.last_dim_size + segment_loop * self.max_segment_len +
                                         offset + offset_data)
                        with self.tik.new_stmt_scope():
                            loop_func(pro_len, gm_in_offset, gm_out_offset)
                    with self.tik.else_scope():
                        with self.tik.if_scope(segment_tail_data % self.do_data_move_each_block == 0):
                            gm_in_offset = (core_offset * self.axis_size * self.last_dim_size + segment_loop *
                                            self.max_segment_len + offset_data)
                            gm_out_offset = (core_offset * self.last_dim_size + segment_loop * self.max_segment_len +
                                             offset_data)
                            loop_func(segment_tail_data, gm_in_offset, gm_out_offset)
                        with self.tik.else_scope():
                            gm_in_offset = (core_offset * self.axis_size * self.last_dim_size + segment_loop *
                                self.max_segment_len + offset_data)
                            gm_out_offset = (core_offset * self.last_dim_size + segment_loop * self.max_segment_len +
                                             offset_data)
                            loop_func(segment_tail_data, gm_in_offset, gm_out_offset)

        with self.tik.for_range(0, self.core_ele) as ele_idx:
            core_offset = core_idx * self.one_core_ele + ele_idx
            _run(core_offset, self.segment_loop, self.segment_tail, self.segment_tail_data, self.offset_data)

    def do_loop_compute_nlast_fp_align(self, segment, gm_in_offset, gm_out_offset):
        """process for a segment when arg not last dim for fp16 align
        """
        ub_data = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_data")
        ub_index_out = self._malloc_tensor(self.dtype_y, (self.max_segment_len,), "ub_index_out")
        ub_index_data = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_index_data")
        ub_out = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_out")

        repeat = self._get_ceil_int(segment, self.data_calc_each_vector)
        self._init_vec(ub_data, self.data_calc_each_vector, segment, self.default_value)
        self._init_vec(ub_index_data, self.data_calc_each_vector, self.data_calc_each_vector, -1)
        self._init_vec(ub_out, self.data_calc_each_vector, segment, 0)
        self._init_vec(ub_index_out, self.mask_out_num, self.max_segment_len, 0)

        last_align = self._get_ceil_int(segment, self.data_move_each_vector) * self.data_move_each_vector
        max_axis_len = self.max_segment_len // last_align
        axis_loop = self.axis_size // max_axis_len
        axis_tail = self.axis_size % max_axis_len

        def _run_one_sigment(axis_idx, axis_len):
            ub_tmp = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_tmp")
            ub_mask = self._malloc_tensor(self.mask_out_dtype, (self.max_segment_len // self.mask_out_num,), "ub_mask")
            self._do_stride_data_move(ub_tmp, self.data_gm[gm_in_offset + axis_idx * segment], segment, axis_len, 1)
            with self.tik.for_range(0, axis_len) as axis_i:
                self.tik.vadds(self.data_calc_each_vector, ub_index_data, ub_index_data, 1, 1, 1, 1, 8, 8)
                _axis_offset = last_align * axis_i
                with self.tik.for_range(0, repeat) as i:
                    offset = i * self.data_calc_each_vector
                    self._get_cmp_mask(ub_data[offset], ub_tmp[offset + _axis_offset], self.data_calc_each_vector,
                                       ub_mask)
                    self.tik.vec_sel(self.data_calc_each_vector, 0, ub_out[offset], ub_mask, ub_index_data,
                                     ub_out[offset], 1, 8, 0, 0)
                self.vcmp_func(self.data_calc_each_vector, ub_data, ub_data, ub_tmp[_axis_offset], repeat,
                               1, 1, 1, 8, 8, 8)

        with self.tik.for_range(0, axis_loop) as _axis_loop:
            input_axis_offset = _axis_loop * max_axis_len
            _run_one_sigment(input_axis_offset, max_axis_len)
        with self.tik.if_scope(axis_tail != 0):
            _run_one_sigment(axis_loop * max_axis_len, axis_tail)

        util_tik_comm_func.tik_func_vconv(self.tik, ub_index_out, ub_out, self.max_segment_len, mode="round")
        self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_index_out, segment)
        return ub_data

    def do_compute_nlast_axis_fp_align(self, core_idx):

        def _run(segment, gm_in_offset, gm_out_offset):
            self.do_loop_compute_nlast_fp_align(segment, gm_in_offset, gm_out_offset)

        with self.tik.for_range(0, self.core_ele) as ele_idx:
            core_offset = core_idx * self.one_core_ele + ele_idx
            gm_in_offset = core_offset * self.axis_size * self.last_dim_size
            gm_out_offset = core_offset * self.last_dim_size
            _run(self.last_dim_size, gm_in_offset, gm_out_offset)

    def do_compute_nlast_axis_cut_by_last_dim(self, core_idx, loop_func):
        
        def _run(in_offset, out_offset):
            with self.tik.for_range(0, self.segment_loop) as segm_i:
                gm_in_offset = in_offset + self.max_segment_len * segm_i
                gm_out_offset = out_offset + self.max_segment_len * segm_i
                loop_func(self.max_segment_len, gm_in_offset, gm_out_offset)
            with self.tik.if_scope(self.segment_tail != 0):
                gm_in_offset = in_offset + self.max_segment_len * self.segment_loop - self.offset_data
                gm_out_offset = out_offset + self.max_segment_len * self.segment_loop - self.offset_data
                loop_func(self.segment_tail_data, gm_in_offset, gm_out_offset)

        with self.tik.for_range(0, self.first_dim_size) as ele_idx:
            offset_in = ele_idx * self.axis_size * self.last_dim_size + core_idx * self.one_core_ele
            offset_out = ele_idx * self.last_dim_size + core_idx * self.one_core_ele
            _run(offset_in, offset_out)

    # 'pylint: disable=too-many-arguments
    def do_loop_compute_last_axis_less_block(self, start_idx, segment_num, line_num, tail_line, result_value,
                                             result_index):
        """In this scenario, when VCMAX/VCMIN does not support float32 and lastdim is less than one data_vector,
        use the vnchwconv and transpose+vmax/vmin calculation
        """
        src_ub = self._malloc_tensor(self.dtype_x, (Constant.N256, self.data_calc_each_vector), "src_ub")
        dst_ub = self._malloc_tensor(self.dtype_x, (Constant.N256, self.data_calc_each_vector), "dst_ub")

        # move 16 line one time
        with self.tik.if_scope(self.last_dim_size % self.data_move_each_block == 0):
            src_offset = start_idx * self.last_dim_size
            self._do_data_move(dst_ub, self.data_gm[src_offset], segment_num * self.last_dim_size)
        with self.tik.elif_scope(segment_num < 128):
            src_offset = start_idx * self.last_dim_size
            with self.tik.for_range(0, segment_num) as idx:
                dst_offset = idx * self.data_calc_each_vector
                self._do_data_move(dst_ub[dst_offset], self.data_gm[src_offset + idx * self.last_dim_size],
                                   self.last_dim_size)
        with self.tik.else_scope():
            with self.tik.for_range(0, line_num) as line_idx:
                src_offset = (start_idx + Constant.N16 * line_idx) * self.last_dim_size
                dst_offset = Constant.N16 * self.data_calc_each_vector * line_idx
                self._do_data_move(src_ub[dst_offset], self.data_gm[src_offset], Constant.N16 * self.last_dim_size)

            with self.tik.if_scope(tail_line != 0):
                src_offset = (start_idx + Constant.N16 * line_num) * self.last_dim_size
                dst_offset = Constant.N16 * self.data_calc_each_vector * line_num
                self._do_data_move(src_ub[dst_offset], self.data_gm[src_offset], tail_line * self.last_dim_size)

            self._transpose_row_2_col(src_ub, dst_ub, Constant.N256, self.data_calc_each_vector)
            with self.tik.for_range(0, Constant.N16) as line_idx:
                src_offset = line_idx * Constant.N16 * self.last_dim_size
                dst_offset = line_idx * Constant.N16 * self.data_calc_each_vector
                block_num = self._get_ceil_int(self.last_dim_size * Constant.N16, self.data_calc_each_block)
                self.tik.data_move(src_ub[dst_offset], dst_ub[src_offset], 0, 1, block_num, 0, 0)
            self._transpose_col_2_row(src_ub, dst_ub, Constant.N256, self.data_calc_each_vector)

        def do_compre(ub_line, result_value_s, result_index_s):
            cmp_value = self._malloc_scalar(self.dtype_x, "cmp_value", ub_line[0])
            cmp_index = self._malloc_scalar(self.dtype_y, "cmp_index", 0)
            with self.tik.for_range(1, self.last_dim_size) as i:
                tmp_cmp_value = self._malloc_scalar(self.dtype_x, "tmp_cmp_value", ub_line[i])
                with self.tik.if_scope(self._scalar_cmp(tmp_cmp_value, cmp_value)):
                    cmp_value.set_as(ub_line[i])
                    cmp_index.set_as(i)
            result_value_s.set_as(cmp_value)
            result_index_s.set_as(cmp_index)

        result_value_s = self._malloc_scalar(self.dtype_x, "result_value_s")
        result_index_s = self._malloc_scalar(self.dtype_y, "result_index_s")
        with self.tik.for_range(0, segment_num) as idx:
            with self.tik.if_scope(self.last_dim_size % self.data_move_each_block == 0):
                do_compre(dst_ub[idx * self.last_dim_size], result_value_s, result_index_s)
            with self.tik.else_scope():
                do_compre(dst_ub[idx * self.data_calc_each_vector], result_value_s, result_index_s)
            result_value[idx] = result_value_s
            result_index[idx] = result_index_s

    def do_compute_last_axis_less_block(self, core_idx):
        ub_result_index = self._malloc_tensor(self.dtype_y, (Constant.N256,), "ub_result_index")
        ub_result_value = self._malloc_tensor(self.dtype_x, (Constant.N256,), "ub_result_value")

        def _run(segment, segment_index, line_num, tail_line):
            start_idx = core_idx * self.one_core_ele + Constant.N256 * segment_index
            self.do_loop_compute_last_axis_less_block(start_idx, segment, line_num, tail_line, ub_result_value,
                                                      ub_result_index)

            gm_out_offset = core_idx * self.one_core_ele + Constant.N256 * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment)

        segment_loop = self._malloc_scalar("int64", "segment_loop", self.core_ele // Constant.N256)
        segment_tail = self._malloc_scalar("int64", "segment_tail", self.core_ele % Constant.N256)
        with self.tik.for_range(0, segment_loop) as loop_idx:
            _run(Constant.N256, loop_idx, Constant.N16, 0)
        with self.tik.if_scope(segment_tail != 0):
            _run(segment_tail, segment_loop, segment_tail // (Constant.N16), segment_tail % (Constant.N16))

    def do_loop_compute_nlast_axis_cut_by_first_and_last_dim(self,
                                                             segment,
                                                             seg_in_offset,
                                                             result_index,
                                                             result_value = None):
        ub_value = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector, ), "ub_value")
        ub_index = self._malloc_tensor(self.dtype_y, (self.data_calc_each_vector, ), "ub_index")
        index_out_repeat = self._get_ceil_int(self.last_dim_size, self.mask_out_num)

        with self.tik.for_range(0, segment) as idx:
            base_offset = seg_in_offset + idx * self.axis_size * self.last_dim_size
            self._do_data_move(ub_value, self.data_gm[base_offset], self.last_dim_size)
            self._init_vec(ub_index, self.mask_out_num, self.data_calc_each_vector, 0)
            with self.tik.for_range(1, self.axis_size) as axis_idx:
                ub_tmp = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector, ), "ub_tmp")
                ub_mask = self._malloc_tensor(self.mask_out_dtype, (index_out_repeat,), "ub_mask")
                self._do_data_move(ub_tmp, self.data_gm[base_offset + axis_idx * self.last_dim_size],
                                   self.last_dim_size)

                self._get_cmp_mask(ub_value, ub_tmp, self.last_dim_size, ub_mask)
                with self.tik.for_range(0, index_out_repeat) as i:
                    mask_l = self._malloc_scalar(self.mask_out_dtype, "mask_l", ub_mask[i])
                    with self.tik.if_scope(mask_l != 0):
                        self.tik.vector_dup([mask_l, mask_l], ub_index[i * self.mask_out_num], axis_idx, 1, 1, 8)
                self.vcmp_func(self.data_calc_each_vector, ub_value, ub_value, ub_tmp, 1, 1, 1, 1, 8, 8, 8)

            with self.tik.for_range(0, self.last_dim_size) as dim_i:
                result_index[idx * self.last_dim_size + dim_i].set_as(ub_index[dim_i])
                if self.is_with_value:
                    result_value[idx * self.last_dim_size + dim_i].set_as(ub_value[dim_i])

    def do_loop_compute_nlast_axis_cut_by_first_and_last_dim_short_axis(self,
                                                                       segment,
                                                                       seg_in_offset,
                                                                       seg_out_offset):

        def _move_result_to_gm(dest, src, dtype, dtype_size, offset, is_index):
            factor = 1
            dest_cast = dest
            src_cast = src
            if get_bit_len(dtype) == 64:
                dest_cast = dest.reinterpret_cast_to("int32")
                src_cast = src.reinterpret_cast_to("int32")
                factor = 2
            if is_index:
                self.tik.data_move_pad(dest_cast[offset * factor], src_cast, 1, self.last_dim_size * dtype_size, 0, 0)
            else:
                self._do_data_move_pad_to_gm(dest_cast[offset * factor], src_cast, self.last_dim_size)
        
        def _add_sync():
            with self.tik.new_stmt_scope(disable_sync=True):
                self.tik.set_flag("PIPE_MTE3", "PIPE_MTE2", 0)
                self.tik.wait_flag("PIPE_MTE3", "PIPE_MTE2", 0)

        def _run_i64(out_idx, inner_count):
            ub_value_inner = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector 
                                                                * self.n_inner * self.axis_size,), "ub_value_inner")
            ub_index = self._malloc_tensor(self.dtype_y, (self.data_calc_each_vector,), "ub_index")
            index_out_repeat = self._get_ceil_int(self.last_dim_size, self.mask_out_num)
            offset = seg_in_offset + out_idx * self.n_inner * self.axis_size * self.last_dim_size
            ub_value_dst_cast = ub_value_inner
            factor = 1
            data_gm_cast = self.data_gm
            if get_bit_len(self.dtype_x) == 64:
                ub_value_dst_cast = ub_value_inner.reinterpret_cast_to("int32")
                data_gm_cast = self.data_gm.reinterpret_cast_to("int32")
                factor = 2
            self._do_data_move_pad_to_ub(ub_value_dst_cast, data_gm_cast[offset * factor], inner_count * self.axis_size,
                                         self.last_dim_size, 1)
            
            with self.tik.for_range(0, inner_count) as in_idx:
                ub_value = ub_value_inner[in_idx * self.axis_size * self.data_calc_each_vector]
                self._init_vec(ub_index, self.mask_out_num, self.data_calc_each_vector, 0)
                with self.tik.for_range(1, self.axis_size) as axis_idx:
                    ub_mask = self._malloc_tensor(self.mask_out_dtype, (index_out_repeat,), "ub_mask")
                    ub_tmp = ub_value_inner[(in_idx * self.axis_size + axis_idx) * self.data_calc_each_vector]
                    self._get_cmp_mask(ub_value, ub_tmp, self.last_dim_size, ub_mask)
                    with self.tik.for_range(0, index_out_repeat) as i:
                        mask_l = self._malloc_scalar(self.mask_out_dtype, "mask_l", ub_mask[i])
                        with self.tik.if_scope(mask_l != 0):
                            self.tik.vector_dup([mask_l, mask_l], ub_index[i * self.mask_out_num], axis_idx, 1, 1, 8)
                    self.vcmp_func(self.data_calc_each_vector, ub_value, ub_value, ub_tmp, 1, 1, 1, 1, 8, 8, 8)
                
                idx = out_idx * self.n_inner + in_idx
                idx_out_offset = seg_out_offset + idx * self.last_dim_size
                if self.is_with_value:
                    _move_result_to_gm(self.result_gm_value, ub_value,
                                       self.dtype_x, self.dtype_x_size, idx_out_offset, False)
                _move_result_to_gm(self.result_gm_index, ub_index,
                                   self.dtype_y, self.dtype_y_size, idx_out_offset, True)
                
        def _run(out_idx, inner_count):
            with self.tik.new_stmt_scope():
                max_inner_num = 128
                ub_value_inner = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector 
                                                                    * self.n_inner * self.axis_size,),
                                                    "ub_value_inner")
                ub_index_out = self._malloc_tensor(self.dtype_y, (max_inner_num * self.data_calc_each_vector,),
                                                   "ub_index_out")
                index_out_repeat = self._get_ceil_int(inner_count * self.data_calc_each_vector, self.mask_out_num)
                index_dtype = "float32" if get_bit_len(self.dtype_x) == 32 else "float16"
                ub_index_vec = self._malloc_tensor(index_dtype, (max_inner_num * self.data_calc_each_vector,),
                                                   "ub_index_vec")
                ub_index_step = self._malloc_tensor(index_dtype, (self.data_calc_each_vector,),
                                                    "ub_index_step")
                ub_mask = self._malloc_tensor(self.mask_out_dtype, (index_out_repeat,), "ub_mask")

                offset = seg_in_offset + out_idx * self.n_inner * self.axis_size * self.last_dim_size
                self._do_data_move_pad_to_ub_with_loop(ub_value_inner,
                                                       self.data_gm[offset],
                                                       self.axis_size,
                                                       self.last_dim_size,
                                                       inner_count)
                
                self._init_vec(ub_index_vec, self.mask_out_num, max_inner_num * self.data_calc_each_vector, 0)
                self._init_vec(ub_index_step, self.mask_out_num, self.data_calc_each_vector, 0)
                ub_value = ub_value_inner
                with self.tik.for_range(1, self.axis_size) as axis_idx:
                    self.tik.vadds(self.data_calc_each_vector, ub_index_step, ub_index_step, 1, 1, 1, 1, 8, 8)
                    ub_tmp = ub_value_inner[axis_idx * inner_count * self.data_calc_each_vector]
                    self._get_cmp_mask(ub_value, ub_tmp, inner_count * self.data_calc_each_vector, ub_mask)
                    self.tik.vec_sel(self.data_calc_each_vector, 2, ub_index_vec, ub_mask, ub_index_step, ub_index_vec,
                                     inner_count, 8, 0, 8)
                    self.vcmp_func(self.data_calc_each_vector, ub_value, ub_value, ub_tmp, inner_count, 1, 1, 1,
                                   8, 8, 8)
                    
                util_tik_comm_func.tik_func_vconv(self.tik, ub_index_out, ub_index_vec,
                                                  max_inner_num * self.data_calc_each_vector, mode="round")
                with self.tik.for_range(0, inner_count) as in_idx:
                    idx = out_idx * self.n_inner + in_idx
                    idx_out_offset = seg_out_offset + idx * self.last_dim_size
                    if self.is_with_value:
                        _move_result_to_gm(self.result_gm_value, ub_value[in_idx * self.data_calc_each_vector],
                                           self.dtype_x, self.dtype_x_size, idx_out_offset, False)
                    _move_result_to_gm(self.result_gm_index, ub_index_out[in_idx * self.data_calc_each_vector],
                                       self.dtype_y, self.dtype_y_size, idx_out_offset, True)

        n_outer = segment // self.n_inner
        with self.tik.for_range(0, n_outer) as out_idx:
            if get_bit_len(self.dtype_x) == 64 or get_bit_len(self.dtype_y) == 64:
                _run_i64(out_idx, self.n_inner)
            else:
                _run(out_idx, self.n_inner)
                _add_sync()
        
        tail_num = segment % self.n_inner
        with self.tik.if_scope(tail_num != 0):
            if get_bit_len(self.dtype_x) == 64 or get_bit_len(self.dtype_y) == 64:
                _run_i64(n_outer, tail_num)
            else:
                _run(n_outer, tail_num)
                _add_sync()

    def do_loop_compute_nlast_axis_cut_by_first_and_last_dim_long_axis(self,
                                                                       segment,
                                                                       seg_in_offset,
                                                                       result_index,
                                                                       result_value = None):
        src_ub = self._malloc_tensor(self.dtype_x, (Constant.N256, self.data_calc_each_vector), "src_ub")
        dst_ub = self._malloc_tensor(self.dtype_x, (Constant.N256, self.data_calc_each_vector), "dst_ub")
        ub_value = self._malloc_tensor(self.dtype_x, (self.data_calc_each_vector,), "ub_value")
        ub_index = self._malloc_tensor(self.dtype_y, (self.data_calc_each_vector,), "ub_index")
        index_out_repeat = self._get_ceil_int(self.last_dim_size, self.mask_out_num)

        def _run(start_idx, line_num, tail_line, axis_start):
            # move 16 line one time
            with self.tik.for_range(0, line_num) as line_idx:
                src_offset = start_idx + Constant.N16 * line_idx * self.last_dim_size
                dst_offset = Constant.N16 * self.data_calc_each_vector * line_idx
                self._do_data_move(src_ub[dst_offset], self.data_gm[src_offset], Constant.N16 * self.last_dim_size)

            with self.tik.if_scope(tail_line != 0):
                src_offset = start_idx + Constant.N16 * line_num * self.last_dim_size
                dst_offset = Constant.N16 * self.data_calc_each_vector * line_num
                self._do_data_move(src_ub[dst_offset], self.data_gm[src_offset], tail_line * self.last_dim_size)

            self._transpose_row_2_col(src_ub, dst_ub, Constant.N256, self.data_calc_each_vector)
            with self.tik.for_range(0, Constant.N16) as line_idx:
                src_offset = line_idx * Constant.N16 * self.last_dim_size
                dst_offset = line_idx * Constant.N16 * self.data_calc_each_vector
                block_num = self._get_ceil_int(self.last_dim_size * Constant.N16, self.data_calc_each_block)
                self.tik.data_move(src_ub[dst_offset], dst_ub[src_offset], 0, 1, block_num, 0, 0)
            self._transpose_col_2_row(src_ub, dst_ub, Constant.N256, self.data_calc_each_vector)

            total_line = line_num * Constant.N16 + tail_line
            with self.tik.for_range(0, total_line) as axis_idx:
                ub_mask = self._malloc_tensor(self.mask_out_dtype, (index_out_repeat,), "ub_mask")
                self._get_cmp_mask(ub_value, dst_ub[axis_idx * self.data_calc_each_vector], self.last_dim_size, ub_mask)
                with self.tik.for_range(0, index_out_repeat) as i:
                    mask_l = self._malloc_scalar(self.mask_out_dtype, "mask_l", ub_mask[i])
                    with self.tik.if_scope(mask_l != 0):
                        self.tik.vector_dup([mask_l, mask_l], ub_index[i * self.mask_out_num],
                                            axis_idx + axis_start, 1, 1, 8)
                self.vcmp_func(self.data_calc_each_vector, ub_value, ub_value,
                               dst_ub[axis_idx * self.data_calc_each_vector], 1, 1, 1, 1, 8, 8, 8)

        with self.tik.for_range(0, segment) as idx:
            base_offset = seg_in_offset + idx * self.axis_size * self.last_dim_size
            self.tik.vector_dup(self.data_calc_each_vector, ub_value, self.default_value, 1, 1, 8)
            self._init_vec(ub_index, self.mask_out_num, self.last_dim_size, 0)
            axis_loop = self.axis_size // Constant.N256
            axis_tail = self.axis_size % Constant.N256
            with self.tik.for_range(0, axis_loop) as loop_idx:
                start_idx = base_offset + Constant.N256 * loop_idx * self.last_dim_size
                _run(start_idx, Constant.N16, 0, loop_idx * Constant.N256)
            with self.tik.if_scope(axis_tail != 0):
                start_idx = base_offset + Constant.N256 * axis_loop * self.last_dim_size
                _run(start_idx, axis_tail // Constant.N16, axis_tail % Constant.N16, axis_loop * Constant.N256)

            with self.tik.for_range(0, self.last_dim_size) as dim_i:
                result_index[idx * self.last_dim_size + dim_i].set_as(ub_index[dim_i])
                if self.is_with_value:
                    result_value[idx * self.last_dim_size + dim_i].set_as(ub_value[dim_i])

    def do_compute_nlast_axis_cut_by_first_and_last_dim(self, core_idx, max_first_dim_len, loop_func):
        """In this scenario, the core is aligned according to the last_dim
        to ensure that the data is aligned when the partition is copied,
        and the vnchwconv+vmax/vim command compute is used after the partition
        """
        segment = max_first_dim_len // self.data_calc_each_vector
        ub_result_index = self._malloc_tensor(self.dtype_y, (max_first_dim_len,), "ub_result_index")

        def _run(segment_len, segment_index):
            core_in_offset = self.one_core_ele * core_idx
            seg_in_offset = (core_in_offset + segment_index * segment) * self.axis_size * self.last_dim_size
            seg_out_offset = (core_in_offset + segment_index * segment) * self.last_dim_size
            loop_func(segment_len, seg_in_offset, ub_result_index)
            self._do_move_index_out(self.result_gm_index[seg_out_offset], ub_result_index,
                                    segment_len * self.last_dim_size)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(segment, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    def do_compute_no_need_compre(self, core_idx):
        ub_result_index = self._malloc_tensor(self.dtype_y, (self.max_segment_len,), "ub_result_index")
        self._init_vec(ub_result_index, self.mask_out_num, self.max_segment_len, 0)

        total_data_num = self.first_dim_size * self.last_dim_size
        segment_loop = total_data_num // self.max_segment_len
        segment_tail = total_data_num % self.max_segment_len
        with self.tik.for_range(0, segment_loop) as segm_i:
            gm_out_offset = core_idx * self.one_core_ele + self.max_segment_len * segm_i
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, self.max_segment_len)

        with self.tik.if_scope(segment_tail != 0):
            gm_out_offset = core_idx * self.one_core_ele + self.max_segment_len * segment_loop
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_tail)

    def _do_compute(self):
        self._init_tiling()
        with self.tik.for_range(0, self.running_core_num, block_num=self.running_core_num) as core_idx:
            self._set_running_params(core_idx)
            if self.is_vccmp_support:
                with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_LAST_LESS_SEGMENT_LEN):
                    with self.tik.new_stmt_scope():
                        self.do_compute_for_last_axis_copy_one_time(core_idx)
                with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_LAST_OVER_DATA_VECTOR):
                    with self.tik.new_stmt_scope():
                        self.do_compute_last_axis_more_vector(core_idx)
                with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_LAST_LESS_DATA_VECTOR):
                    with self.tik.new_stmt_scope():
                        self.do_compute_last_axis_less_vector(core_idx)
                with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_LAST_OVER_SEGMENT_LEN):
                    with self.tik.new_stmt_scope():
                        self.do_compute_last_axis_over_segment(core_idx)

            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_LAST_AXIS_VCMP):
                with self.tik.new_stmt_scope():
                    if self.dtype_x in ("float32", "int64", "int32",):
                        self.do_compute_last_axis_vcmp(core_idx)
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_LAST_LESS_BLOCK):
                with self.tik.new_stmt_scope():
                    if self.dtype_x in ("float32", "int64", "int32",):
                        self.do_compute_last_axis_less_block(core_idx)
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_NLAST_CUT_FIRST_DIM):
                with self.tik.new_stmt_scope():
                    self.do_compute_nlast_axis_cut_by_first_dim(core_idx, self.do_loop_compute_nlast_axis)
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_NLAST_CUT_FIRST_DIM_AXIS_LESS):
                with self.tik.new_stmt_scope():
                    if check_support_block_size_16():
                        self.do_compute_nlast_axis_cut_by_first_dim(core_idx, self.do_loop_compute_nlast_axis)
                    elif self.dtype_x in ("float16", "float32"):
                        if ArgCommon._is_support_inf_nan():
                            self.do_compute_nlast_axis_cut_by_first_dim(core_idx, self.do_loop_compute_nlast_axis)
                        else:
                            self.do_compute_nlast_axis_cut_by_first_dim(core_idx, 
                                                                        self.do_loop_compute_nlast_axis_less)
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_NLAST_FP_ALIGN):
                with self.tik.new_stmt_scope():
                    if check_support_block_size_16():
                        self.do_compute_nlast_axis_cut_by_first_dim(core_idx, self.do_loop_compute_nlast_axis)
                    elif self.dtype_x in ("float16", "float32"):
                        self.do_compute_nlast_axis_fp_align(core_idx)
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_NLAST_CUT_LAST_DIM):
                with self.tik.new_stmt_scope():
                    self.do_compute_nlast_axis_cut_by_last_dim(core_idx, self.do_loop_compute_nlast_axis)
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_NLAST_CUT_LAST_DIM_AXIS_LESS):
                with self.tik.new_stmt_scope():
                    if self.dtype_x in ("float16", "float32"):
                        if ArgCommon._is_support_inf_nan():
                            self.do_compute_nlast_axis_cut_by_last_dim(core_idx, self.do_loop_compute_nlast_axis)
                        # nano not support set tensor as cmpmask, may cause compilation error
                        elif not check_support_block_size_16():
                            self.do_compute_nlast_axis_cut_by_last_dim(core_idx, 
                                                                       self.do_loop_compute_nlast_axis_less)
            if self.core_num == 1:
                return
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM):
                with self.tik.new_stmt_scope():
                    self.do_compute_nlast_axis_cut_by_first_and_last_dim(core_idx, Constant.MAX_FIRST_DIM_LEN,
                        self.do_loop_compute_nlast_axis_cut_by_first_and_last_dim)
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_SHORT_AXIS):
                with self.tik.new_stmt_scope():
                    if self.is_data_move_pad_support:
                        self.do_compute_nlast_axis_cut_by_first_and_last_dim_move_direct(core_idx, 
                            Constant.MAX_FIRST_DIM_LEN,
                            self.do_loop_compute_nlast_axis_cut_by_first_and_last_dim_short_axis)
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_LONG_AXIS):
                with self.tik.new_stmt_scope():
                    max_first_dim_len = Constant.MAX_FIRST_DIM_LEN
                    if self.is_with_value:
                        max_first_dim_len = max_first_dim_len // 2
                    if self.dtype_x == "int64" or self.dtype_y == "int64":
                        max_first_dim_len = max_first_dim_len // 2
                    self.do_compute_nlast_axis_cut_by_first_and_last_dim(core_idx, max_first_dim_len,
                        self.do_loop_compute_nlast_axis_cut_by_first_and_last_dim_long_axis)
            with self.tik.if_scope(self.tiling_mode == Constant.TILING_MODE_NO_COMPUTE):
                with self.tik.new_stmt_scope():
                    self.do_compute_no_need_compre(core_idx)


class ArgCommonWithValue(ArgCommon):
    def __init__(self, is_min, dtype_x, dtype_y, is_dynamic, kernel_name) -> None:
        super().__init__(is_min, dtype_x, dtype_y, is_dynamic, kernel_name, True)

    def _init_gm_buffer(self):
        self.tiling_gm = self._malloc_tensor("int64", (Constant.TILING_ARG_NUM,), "tiling_gm", tik.scope_gm)
        self.data_gm = self._malloc_tensor(self.ori_dtype_x, (Constant.MAX_INT32,), "data_gm", tik.scope_gm)
        self.result_gm_index = self._malloc_tensor(self.dtype_y, (Constant.MAX_INT32,), "result_gm_index", tik.scope_gm)
        self.result_gm_value = self._malloc_tensor(self.ori_dtype_x, (Constant.MAX_INT32,), "result_gm_value",
                                                   tik.scope_gm)

    def do_compute_for_last_axis_copy_one_time(self, core_idx):
        """reduce last dim axis size over data_each_block * 2 and below 8192
        need to call vcmax/vcmin twice
        """
        ub_result_index = self._malloc_tensor(self.dtype_y, (Constant.MAX_FIRST_DIM_LEN,), "ub_result_index")
        ub_result_value = self._malloc_tensor(self.dtype_x, (Constant.MAX_FIRST_DIM_LEN,), "ub_result_value")

        def _run(segment_len, segment_index):
            with self.tik.for_range(0, segment_len) as idx:
                seg_offset = core_idx * self.one_core_ele + idx + Constant.MAX_FIRST_DIM_LEN * segment_index
                result_index = self._malloc_scalar(self.dtype_y, "result_index")
                result_value = self._malloc_scalar(self.dtype_x, "result_value")
                self.do_loop_compute_for_last_axis_copy_one_time(seg_offset, result_index, result_value)
                ub_result_index[idx] = result_index
                ub_result_value[idx] = result_value

            gm_out_offset = core_idx * self.one_core_ele + Constant.MAX_FIRST_DIM_LEN * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)
            self._do_data_move(self.result_gm_value[gm_out_offset], ub_result_value, segment_len)

        with self.tik.for_range(0, self.segment_loop) as _loop:
            _run(Constant.MAX_FIRST_DIM_LEN, _loop)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    def do_compute_last_axis_more_vector(self, core_idx):
        ub_result_index = self._malloc_tensor(self.dtype_y, (self.max_segment_len,), "ub_result_index")
        ub_result_value = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_result_value")

        def _run(segment_len, segment_index):
            seg_offset = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            self.do_loop_compute_last_axis_more_vector(seg_offset, segment_len, ub_result_index, ub_result_value)

            gm_out_offset = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)
            self._do_data_move(self.result_gm_value[gm_out_offset], ub_result_value, segment_len)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(self.axis_size_one_time, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    def do_compute_last_axis_less_vector(self, core_idx):
        """compute_argmin_last_axis_fp16_less_vector
        """
        ub_result_index = self._malloc_tensor(self.dtype_y, (self.max_segment_len,), "ub_result_index")
        ub_result_value = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_result_value")

        def _run(segment_len, segment_index):
            start_idx = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            self.do_loop_compute_last_axis_less_vector(start_idx, segment_len, ub_result_index, ub_result_value)
            gm_out_offset = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)
            self._do_data_move(self.result_gm_value[gm_out_offset], ub_result_value, segment_len)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(self.axis_size_one_time, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    def do_compute_last_axis_over_segment(self, core_idx):
        ub_result_index = self._malloc_tensor(self.dtype_y, (Constant.MAX_FIRST_DIM_LEN,), "ub_result_index")
        ub_result_value = self._malloc_tensor(self.dtype_x, (Constant.MAX_FIRST_DIM_LEN,), "ub_result_value")
        default_result_value = self._malloc_tensor(self.dtype_x, (16,), "default_result_value")
        self.tik.vector_dup(16, default_result_value, self.default_value, 1, 1, 8)

        def _run(segment_len, segment_index):
            with self.tik.for_range(0, segment_len) as idx:
                seg_offset = core_idx * self.one_core_ele + idx + Constant.MAX_FIRST_DIM_LEN * segment_index
                tmp_result_index = self._malloc_scalar(self.dtype_y, "tmp_result_index", 0)
                tmp_result_value = self._malloc_scalar(self.dtype_x, "tmp_result_value")
                tmp_result_value.set_as(default_result_value[0])
                with self.tik.for_range(0, self.loop_times) as loop_idx:
                    self.do_loop_compute_last_axis_over_segment(self.segment, loop_idx, seg_offset, tmp_result_value,
                                                                tmp_result_index)
                with self.tik.if_scope(self.tail_size != 0):
                    self.do_loop_compute_last_axis_over_segment(self.tail_size, self.loop_times, seg_offset,
                                                                tmp_result_value, tmp_result_index)
                ub_result_index[idx] = tmp_result_index
                ub_result_value[idx] = tmp_result_value
            gm_out_offset = core_idx * self.one_core_ele + Constant.MAX_FIRST_DIM_LEN * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)
            self._do_data_move(self.result_gm_value[gm_out_offset], ub_result_value, segment_len)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(Constant.MAX_FIRST_DIM_LEN, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    def do_compute_last_axis_vcmp(self, core_idx):
        """Two layers of circulation,
        the outer loop is divided by first_dim,
        and the inner loop is divided by axis_size by segment
        """
        ub_result_index = self._malloc_tensor(self.dtype_y, (Constant.MAX_FIRST_DIM_LEN,), "ub_result_index")
        ub_result_value = self._malloc_tensor(self.dtype_x, (Constant.MAX_FIRST_DIM_LEN,), "ub_result_value")
        default_result_value = self._malloc_tensor(self.dtype_x, (16,), "default_result_value")
        self.tik.vector_dup(16, default_result_value, self.default_value, 1, 1, 8)

        def _run(segment_len, segment_index):
            with self.tik.new_stmt_scope():
                with self.tik.for_range(0, segment_len) as idx:
                    result_index = self._malloc_scalar(self.dtype_y, "result_index", 0)
                    result_value = self._malloc_scalar(self.dtype_x, "result_value")
                    result_value.set_as(default_result_value[0])
                    seg_offset = core_idx * self.one_core_ele + idx + Constant.MAX_FIRST_DIM_LEN * segment_index
                    with self.tik.for_range(0, self.loop_times) as loop:
                        self.do_loop_compute_last_axis_vcmp(self.segment, loop, seg_offset, result_value, result_index)
                    with self.tik.if_scope(self.tail_size != 0):
                        self.do_loop_compute_last_axis_vcmp(self.tail_size, self.loop_times, seg_offset, result_value,
                                                            result_index)
                    ub_result_index[idx] = result_index
                    ub_result_value[idx] = result_value

                gm_out_offset = core_idx * self.one_core_ele + Constant.MAX_FIRST_DIM_LEN * segment_index
                self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment_len)
                self._do_data_move(self.result_gm_value[gm_out_offset], ub_result_value, segment_len)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(Constant.MAX_FIRST_DIM_LEN, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    def do_loop_compute_nlast_axis(self, segment, gm_in_offset, gm_out_offset):
        ub_data = super().do_loop_compute_nlast_axis(segment, gm_in_offset, gm_out_offset)
        self._do_data_move(self.result_gm_value[gm_out_offset], ub_data, segment)

    def do_loop_compute_nlast_axis_less(self, segment, gm_in_offset, gm_out_offset):
        ub_data = super().do_loop_compute_nlast_axis_less(segment, gm_in_offset, gm_out_offset)
        self._do_data_move(self.result_gm_value[gm_out_offset], ub_data, segment)

    def do_loop_compute_nlast_fp_align(self, segment, gm_in_offset, gm_out_offset):
        ub_data = super().do_loop_compute_nlast_fp_align(segment, gm_in_offset, gm_out_offset)
        self._do_data_move(self.result_gm_value[gm_out_offset], ub_data, segment)

    def do_compute_last_axis_less_block(self, core_idx):
        ub_result_index = self._malloc_tensor(self.dtype_y, (Constant.N256,), "ub_result_index")
        ub_result_value = self._malloc_tensor(self.dtype_x, (Constant.N256,), "ub_result_value")

        def _run(segment, segment_index, line_num, last_col_num):
            start_idx = core_idx * self.one_core_ele + Constant.N256 * segment_index
            self.do_loop_compute_last_axis_less_block(start_idx, segment, line_num, last_col_num, ub_result_value,
                                                      ub_result_index)

            gm_out_offset = core_idx * self.one_core_ele + Constant.N256 * segment_index
            self._do_move_index_out(self.result_gm_index[gm_out_offset], ub_result_index, segment)
            self._do_data_move(self.result_gm_value[gm_out_offset], ub_result_value, segment)

        segment_loop = self._malloc_scalar("int64", "segment_loop", self.core_ele // Constant.N256)
        segment_tail = self._malloc_scalar("int64", "segment_tail", self.core_ele % Constant.N256)
        with self.tik.for_range(0, segment_loop) as loop_idx:
            _run(Constant.N256, loop_idx, Constant.N16, 0)
        with self.tik.if_scope(segment_tail != 0):
            _run(segment_tail, segment_loop, segment_tail // (Constant.N16), segment_tail % (Constant.N16))

    def do_compute_nlast_axis_cut_by_first_and_last_dim(self, core_idx, max_first_dim_len, loop_func):
        segment = max_first_dim_len // self.data_calc_each_vector
        ub_result_value = self._malloc_tensor(self.dtype_x, (max_first_dim_len,), "ub_result_value")
        ub_result_index = self._malloc_tensor(self.dtype_y, (max_first_dim_len,), "ub_result_index")

        def _run(segment_len, segment_index):
            core_in_offset = self.one_core_ele * core_idx
            seg_in_offset = (core_in_offset + segment_index * segment) * self.axis_size * self.last_dim_size
            seg_out_offset = (core_in_offset + segment_index * segment) * self.last_dim_size
            loop_func(segment_len, seg_in_offset, ub_result_index, ub_result_value)
            self._do_move_index_out(self.result_gm_index[seg_out_offset], ub_result_index,
                                    segment_len * self.last_dim_size)
            self._do_data_move(self.result_gm_value[seg_out_offset], ub_result_value, segment_len * self.last_dim_size)

        with self.tik.for_range(0, self.segment_loop) as loop_idx:
            _run(segment, loop_idx)
        with self.tik.if_scope(self.segment_tail != 0):
            _run(self.segment_tail, self.segment_loop)

    def do_compute_no_need_compre(self, core_idx):
        ub_result_index = self._malloc_tensor(self.dtype_y, (self.max_segment_len,), "ub_result_index")
        ub_result_value = self._malloc_tensor(self.dtype_x, (self.max_segment_len,), "ub_result_value")
        self._init_vec(ub_result_index, self.mask_out_num, self.max_segment_len, 0)

        total_data_num = self.first_dim_size * self.last_dim_size
        segment_loop = total_data_num // self.max_segment_len
        segment_tail = total_data_num % self.max_segment_len
        with self.tik.for_range(0, segment_loop) as segm_i:
            offset = core_idx * self.one_core_ele + self.max_segment_len * segm_i
            self._do_data_move(ub_result_value, self.data_gm[offset], self.max_segment_len)

            self._do_data_move(self.result_gm_value[offset], ub_result_value, self.max_segment_len)
            self._do_move_index_out(self.result_gm_index[offset], ub_result_index, self.max_segment_len)

        with self.tik.if_scope(segment_tail != 0):
            offset = core_idx * self.one_core_ele + self.max_segment_len * segment_loop
            self._do_data_move(ub_result_value, self.data_gm[offset], segment_tail)

            self._do_data_move(self.result_gm_value[offset], ub_result_value, segment_tail)
            self._do_move_index_out(self.result_gm_index[offset], ub_result_index, segment_tail)

    def get_tik_instance(self):
        super()._do_compute()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars", {
            "ub_ele": self.ub_ele,
            "max_segment_len": self.max_segment_len,
            "core_num": self.core_num,
            "is_vccmp_support" : self.is_vccmp_support,
            "is_data_move_pad_support" : self.is_data_move_pad_support,
            "is_vsel_support" : self.is_vsel_support,
            "block_size": self.block_size,
            "segment_len": self.segment_len,
            "first_dim_segment": self.first_dim_segment
        })
        fatbin = None
        if self.is_dynamic:
            fatbin = {"tiling_key": [self.tiling_mode],
                      "tiling_key_value": [[Constant.TILING_MODE_LAST_LESS_SEGMENT_LEN],
                                           [Constant.TILING_MODE_LAST_OVER_DATA_VECTOR],
                                           [Constant.TILING_MODE_LAST_LESS_DATA_VECTOR],
                                           [Constant.TILING_MODE_LAST_OVER_SEGMENT_LEN],
                                           [Constant.TILING_MODE_LAST_AXIS_VCMP],
                                           [Constant.TILING_MODE_LAST_LESS_BLOCK],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_DIM],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_DIM_AXIS_LESS],
                                           [Constant.TILING_MODE_NLAST_FP_ALIGN],
                                           [Constant.TILING_MODE_NLAST_CUT_LAST_DIM],
                                           [Constant.TILING_MODE_NLAST_CUT_LAST_DIM_AXIS_LESS],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_LONG_AXIS],
                                           [Constant.TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_SHORT_AXIS],
                                           [Constant.TILING_MODE_NO_COMPUTE]]}
        self.tik.BuildCCE(kernel_name=self.kernel_name, inputs=[self.data_gm],
                          outputs=[self.result_gm_index, self.result_gm_value],
                          flowtable=[self.tiling_gm], config=opt_config,
                          extend_params={"build_multi_kernels": fatbin} if fatbin else None)
        return self.tik
