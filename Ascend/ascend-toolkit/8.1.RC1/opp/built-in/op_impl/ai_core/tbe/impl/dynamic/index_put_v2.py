
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
dynamic index_put_v2
"""
from functools import reduce as functools_reduce
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
    RESERVED_UB = 20480
    INT8_BLOCK = 32
    BYTE_SIZE = 8
    BLOCK_NUM = 16
    # the number of blocks skipped per repeat
    STRIDE_EIGHT = 8
    # the number of blocks skipped per repeat
    STRIDE_FOUR = 4
    # the number of blocks skipped per repeat
    STRIDE_ONE = 1
    # the number of blocks per transposition
    LIST_NUMBER = 8
    # the number of transposes per repeat
    NUMBER_TWO = 2
    # max int32
    MAX_INT32 = 2 ** 31 - 1
    # ting param num
    TILING_ARG_NUM = 32
    # MAX loop number
    MAX_LOOP = 2000
    # support multi atmoic_add
    SUPPORT_MULTI_ATMOIC_ADD = 1
    # only support fp32 atmoic_add
    SUPPORT_FP32_ATMOIC_ADD = 0
    # not support bf16 atmoic_add
    UNSUPPORT_BF16_ATMOIC_ADD = 2
    # mask number of fp32
    MASK_NUM_32 = 64
    MAX_RESERVE = 100
    MAX_INDEICES = 20000
    MAX_AICORE_TIAL = 70
    MULTI_DTYPE_SUPPORT_NUM = 200
    MULTI_DTYPE_SUPPORT_TAIL = 128
    MAX_SUPPORTTYPE_INDICES_NUM = 60000000
    MAX_SHPAE_NUM = 5400000
    TAIL_SIZE = 1024
    support_dtype_A2_t = ("float32", "int32", "float16", "int8", "bool", "bfloaf16")
    support_dtype_A2_i = ("float32", "int32", "float16", "bool", "bfloaf16")


# 'pylint: disable=too-many-locals, too-many-arguments
def IsAiCPUSupportCheckIndices(indices_size, value, indices):
    if indices_size == 0 or len(value.get("shape")) >= 8 :
        return False
    indices_dtype = indices[0].get("dtype")
    if indices_dtype == "bool" :
        return False
    for i in range(1, indices_size) :
        if indices[i].get("dtype") == "bool" :
            return False
        if len(indices[i].get("shape")) != len(indices[0].get("shape")) :
            return False
        for j in range(len(indices[0].get("shape"))) :
            if indices[i].get("shape")[j] != indices[0].get("shape")[j]:
                return False
    return True


# 'pylint: disable=too-many-locals, too-many-arguments
def check_supported(x, value, indexed_sizes, indexed_strides, indices, result, accumulate,
                    kernel_name="index_put_v2"):
    """
        check the op support situation.
    """
    x_shape = x.get("shape")
    x_num = functools_reduce(lambda x, y: x * y, x_shape)
    x_dtype = x.get("dtype").lower()
    indices_size = len(indices)
    x_dim = len(x_shape)
    indices_shape = indices[0].get("shape")
    indices_num = functools_reduce(lambda x, y: x * y, indices_shape)
    tail_size = 1
    is_support_atomic = tbe_platform.api_check_support("tik.vec_reduce_max", "float32")
    
    if int(-1) in x_shape or int(-2) in x_shape:
        return "Unknown"

    api_support_flag = tbe_platform.api_check_support("tik.vgather")
    api_support_flag = api_support_flag and tbe_platform.api_check_support("tik.vconv", "s642s32")
    api_support_flag = api_support_flag and tbe_platform.api_check_support("tik.vshr", "int32")

    start = 0
    for i in range(indices_size):
        size = functools_reduce(lambda x, y: x * y, indices[i].get("shape"))
        if size != 0:
            break
        start += 1
    is_zero_in_masks = False
    for i in range(start, indices_size):
        size = functools_reduce(lambda x, y: x * y, indices[i].get("shape"))
        if is_zero_in_masks and size != 0:
            return False
        if size == 0:
            is_zero_in_masks = True
    if start != 0 and is_support_atomic == False :
        return False

    if x_dtype == "float64" or x_dtype == "int16" or x_dtype == "uint8" or x_dtype == "int64":
        reason = "IndexPutV2 not support float64, int16, uint8, int64"
        return False, reason

    if not IsAiCPUSupportCheckIndices(indices_size, value, indices) :
        return False

    if indices_num > Constant.MAX_SUPPORTTYPE_INDICES_NUM :
        reason = "IndexPutV2 not support indices num greater than 60000000."
        return False, reason

    tail_size = 1
    for i in range(indices_size, x_dim):
        tail_size = tail_size * x_shape[i]
    if (tail_size <= Constant.MAX_RESERVE and tail_size > Constant.MAX_AICORE_TIAL) and\
       (indices_num <= Constant.MAX_INDEICES):
        return False
    
    if not is_support_atomic or (api_support_flag and x_dtype not in Constant.support_dtype_A2_t)\
        or (not api_support_flag and x_dtype not in Constant.support_dtype_A2_i) :
        if (tail_size <= Constant.MAX_RESERVE and indices_num <= Constant.MAX_INDEICES) :
            if tail_size > Constant.MAX_AICORE_TIAL or x_num >= Constant.MAX_SHPAE_NUM:
                return False
        if indices_num > Constant.MAX_INDEICES and tail_size <= Constant.TAIL_SIZE :
            return False

    if (x_dtype != "float16" and x_dtype != "float32" and x_dtype != "bfloat16") or\
       (indices_num <= Constant.MAX_RESERVE and tail_size <= Constant.MAX_RESERVE) and\
       x_num >= Constant.MAX_INDEICES:
        return False

    if x_dtype != "float16" and x_dtype != "float32" and x_dtype != "bfloat16" and\
       (indices_num > Constant.MULTI_DTYPE_SUPPORT_NUM or tail_size > Constant.MULTI_DTYPE_SUPPORT_TAIL):
        return False

    tailSizeTranspose = 1
    if start != 0 and tail_size < Constant.TAIL_SIZE and is_support_atomic == True :
        for i in range(start) :
            tailSizeTranspose = tailSizeTranspose * x_shape[i]
        tailSizeTranspose = tailSizeTranspose * tail_size
    if indices_num > Constant.MAX_RESERVE or tail_size > Constant.MULTI_DTYPE_SUPPORT_TAIL or\
        (start != 0 and tail_size < Constant.TAIL_SIZE and tailSizeTranspose > Constant.MULTI_DTYPE_SUPPORT_TAIL) :
        if x_dtype != "float16" and x_dtype != "float32" and is_support_atomic == False :
            return False
        if (x_dtype == "float16" or x_dtype == "float32") and accumulate == False and is_support_atomic == False :
            return False

    return True


# 'pylint: disable=useless-object-inheritance,too-many-instance-attributes
class IndexPut(object):

    # 'pylint: disable=too-many-arguments,invalid-name
    def __init__(self, x, value, indexed_sizes, indexed_strides,
                 indices, result, accumulate, kernel_name):
        """
        Init IndexPutV2 base parameters

        Parameters
        ----------
        x : dict
            shape and dtype of input x
        value : dict
            shape and dtype of update value
        indexed_sizes : dict
        indexed_strides : dict
        indices : list
            dynamic input
        accumulate : bool
            false means replace the x, true means accumulate value
        kernel_name : str
            kernel name, default value is "index_put_v2"

        Returns
        -------
        None
        """
        byte_size = 8
        block_number_fp16 = 32
        self.accumulate = accumulate
        self.tik_instance = tik.Tik()
        self.x_dtype = x.get("dtype").lower()
        self.x_dtype_before = x.get("dtype").lower()
        if self.x_dtype == "bool":
            self.x_dtype = "int8"
            self.x_dtype_before = "int8"
        if self.x_dtype == "bfloat16":
            self.x_dtype = "float32"
            self.x_dtype_before = "bfloat16"
        self.value_dtype = value.get("dtype").lower()
        self.value_dtype_before = value.get("dtype").lower()
        if self.value_dtype == "bool":
            self.value_dtype = "int8"
            self.value_dtype_before = "int8"
        if self.value_dtype == "bfloat16":
            self.value_dtype = "float32"
            self.value_dtype_before = "bfloat16"
        self.kernel_name = kernel_name
        self.x_dtype_bytes_size = tbe_platform.get_bit_len(self.x_dtype) // byte_size
        self.x_dtype_bytes_size_before = tbe_platform.get_bit_len(self.x_dtype_before) // byte_size
        self.x_data_each_block = constant.BLOCK_SIZE // self.x_dtype_bytes_size
        self.x_data_each_block_before = constant.BLOCK_SIZE // self.x_dtype_bytes_size_before
        self.value_dtype_bytes_size = tbe_platform.get_bit_len(self.value_dtype) // byte_size
        self.value_data_each_block = constant.BLOCK_SIZE // self.value_dtype_bytes_size
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.each_repeat_block_number = block_number_fp16
        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.ub_max_size = (self.total_ub - Constant.RESERVED_UB) // Constant.LIST_NUMBER
        self.available_ub_size = self.tik_instance.Scalar("int64", name="available_ub_size", init_value=2048)
        self.support_multi_atmoic_add = tbe_platform.api_check_support("tik.data_move_pad") # 910B and 310B
        self.support_int8_atmoic_add = tbe_platform.api_check_support("tik.set_atomic_add", "int8")
        self.dtype_support_pad = tbe_platform.api_check_support("tik.data_move_pad", self.value_dtype)
        self.support_dtype = ["float32", "float16", "bfloat16", "int32", "int8"]
        self.support_dtype_310b = ["float32", "float16", "int32"]
        if not self.support_multi_atmoic_add:
            self.soc_version = Constant.SUPPORT_FP32_ATMOIC_ADD
        elif self.support_int8_atmoic_add:
            self.soc_version = Constant.SUPPORT_MULTI_ATMOIC_ADD
        else:
            self.soc_version = Constant.UNSUPPORT_BF16_ATMOIC_ADD
        
        self.offset = self.tik_instance.Scalar("int64", name="offset", init_value=0)
        self.indices_type = indices[0].get("dtype").lower()
        if self.indices_type == "bool":
            self.indices_type = "int8"
        self.indices_bytes_size = tbe_platform.get_bit_len(self.indices_type) // byte_size
        self.indices_each_block = constant.BLOCK_SIZE // self.indices_bytes_size

        # init gm data
        self.x_gm = self.tik_instance.Tensor(self.x_dtype_before, [Constant.MAX_INT32],
                                             name="x_gm", scope=tik.scope_gm)
        self.value_gm = self.tik_instance.Tensor(self.value_dtype_before, [Constant.MAX_INT32],
                                                 name="value_gm", scope=tik.scope_gm)
        self.mask_gm = self.tik_instance.Tensor("int64", [Constant.MAX_INT32],
                                                name="mask_gm", scope=tik.scope_gm)
        self.mask_gm2 = self.tik_instance.Tensor("int64", [Constant.MAX_INT32],
                                                 name="mask_gm2", scope=tik.scope_gm)
        self.input_tensors = [self.x_gm, self.value_gm, self.mask_gm, self.mask_gm2]
        self.indices_count = len(indices)
        for index in range(self.indices_count):
            tensor_name = "_".join(["gm_input", str(index)])
            dtype = indices[index].get("dtype").lower()
            if dtype == "bool":
                dtype = "int8"
            gm_tensor = self.tik_instance.Tensor(dtype, [Constant.MAX_INT32], name=tensor_name, scope=tik.scope_gm)
            self.input_tensors.append(gm_tensor)
        self.input_count = len(self.input_tensors)
        self.result_gm = self.tik_instance.Tensor(self.x_dtype_before, [Constant.MAX_INT32],
                                                  name="result_gm", scope=tik.scope_gm)

        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.init_tiling()

    def init_tiling(self):
        # init tiling data
        self.box_num = self.tik_instance.Scalar("int64", name="box_num")
        self.core_data = self.tik_instance.Scalar("int64", name="core_data")
        self.core_used = self.tik_instance.Scalar("int64", name="core_used")
        self.copy_loop = self.tik_instance.Scalar("int64", name="copy_loop")
        self.copy_tail = self.tik_instance.Scalar("int64", name="copy_tail")
        self.last_copy_loop = self.tik_instance.Scalar("int64", name="last_copy_loop")
        self.last_copy_tail = self.tik_instance.Scalar("int64", name="last_copy_tail")
        self.reserver_dim_number = self.tik_instance.Scalar("int64", name="reserver_dim_number")
        self.value_number = self.tik_instance.Scalar("int64", name="value_number")
        self.indices_number = self.tik_instance.Scalar("int64", name="indices_number")
        self.indices_list_number = self.tik_instance.Scalar("int64", name="indices_list_number")
        self.mask_dim = self.tik_instance.Scalar("int64", name="mask_dim")
        self.dim_0 = self.tik_instance.Scalar("int64", name="dim_0")
        self.dim_1 = self.tik_instance.Scalar("int64", name="dim_1")
        self.dim_2 = self.tik_instance.Scalar("int64", name="dim_2")
        self.dim_3 = self.tik_instance.Scalar("int64", name="dim_3")
        self.dim_4 = self.tik_instance.Scalar("int64", name="dim_4")
        self.dim_5 = self.tik_instance.Scalar("int64", name="dim_5")
        self.dim_6 = self.tik_instance.Scalar("int64", name="dim_6")
        self.dim_7 = self.tik_instance.Scalar("int64", name="dim_7")
        self.reserver_dim_0 = self.tik_instance.Scalar("int64", name="reserver_dim_0")
        self.reserver_dim_1 = self.tik_instance.Scalar("int64", name="reserver_dim_1")
        self.reserver_dim_2 = self.tik_instance.Scalar("int64", name="reserver_dim_2")
        self.reserver_dim_3 = self.tik_instance.Scalar("int64", name="reserver_dim_3")
        self.reserver_dim_4 = self.tik_instance.Scalar("int64", name="reserver_dim_4")
        self.reserver_dim_5 = self.tik_instance.Scalar("int64", name="reserver_dim_5")
        self.reserver_dim_6 = self.tik_instance.Scalar("int64", name="reserver_dim_6")
        self.reserver_dim_7 = self.tik_instance.Scalar("int64", name="reserver_dim_7")
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode", init_value=0)
        self.x_shape_per_dim = self.tik_instance.ScalarArray(dtype="int64", length=8)
        self.x_shape_reserver_dim = self.tik_instance.ScalarArray(dtype="int64", length=8)
        self.tiling_core_num = self.tik_instance.Scalar(dtype="int64", name="tiling_core_num")

    def data_move_pad(self, dst, src, nburst, burst, dst_gap, src_gap, right_padding=0, left_padding=0, padding_value=None):
        if self.dtype_support_pad:
            self.tik_instance.data_move_pad(dst, src, nburst, burst, dst_gap, src_gap, right_padding,
                                            left_padding, padding_value)
        else:
            dst_int8 = dst.reinterpret_cast_to("int8")
            src_int8 = src.reinterpret_cast_to("int8")
            self.tik_instance.data_move_pad(dst_int8, src_int8, nburst, burst, dst_gap, src_gap, right_padding,
                                            left_padding, padding_value)

    def index_put_compute_move_pad_each(self, reserver_num, addr_value, addr_x_loop, temp_scalar, is_tail=False):

        x_burst = (reserver_num + self.x_data_each_block - 1) // self.x_data_each_block
        x_burst_pad = reserver_num * self.x_dtype_bytes_size
        if self.x_dtype_before == "bfloat16":
            x_burst_before = (reserver_num + self.x_data_each_block_before - 1) // self.x_data_each_block_before
            x_burst_pad_before = reserver_num * self.x_dtype_bytes_size_before
            repeattime = (reserver_num + Constant.MASK_NUM_32 - 1) // Constant.MASK_NUM_32

        value_ub = self.tik_instance.Tensor(self.value_dtype, (self.available_ub_size,),
                                            name="value_ub", scope=tik.scope_ubuf)

        if self.x_dtype_before == "bfloat16":
            value_ub_before = self.tik_instance.Tensor(self.value_dtype_before,
                                                        (self.available_ub_size,),
                                                        name="value_ub_before", scope=tik.scope_ubuf)
            if not is_tail:
                self.tik_instance.data_move(value_ub_before,
                                            self.value_gm[addr_value + temp_scalar],
                                            constant.SID, constant.DEFAULT_NBURST, x_burst_before,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            else:
                self.tik_instance.data_move_pad(value_ub_before,
                                                self.value_gm[addr_value + temp_scalar],
                                                constant.DEFAULT_NBURST, x_burst_pad_before,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            self.tik_instance.vec_conv(Constant.MASK_NUM_32, '', value_ub,
                                        value_ub_before, repeattime, 8, 4)
        else:
            if self.support_multi_atmoic_add:
                self.tik_instance.data_move_pad(value_ub, self.value_gm[addr_value + temp_scalar],
                                                constant.DEFAULT_NBURST, x_burst_pad,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            else:
                self.tik_instance.data_move(value_ub, self.value_gm[addr_value + temp_scalar],
                                            constant.SID, constant.DEFAULT_NBURST, x_burst,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        if self.support_multi_atmoic_add:
            if self.x_dtype_before == "bfloat16":
                self.tik_instance.vec_conv(Constant.MASK_NUM_32, 'floor', value_ub_before,
                                            value_ub, repeattime, 4, 8)
                self.tik_instance.data_move_pad(self.result_gm[addr_x_loop + temp_scalar], value_ub_before,
                                                constant.DEFAULT_NBURST, x_burst_pad_before,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            else:
                self.tik_instance.data_move_pad(self.result_gm[addr_x_loop + temp_scalar], value_ub,
                                                constant.DEFAULT_NBURST, x_burst_pad,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
    
    def index_put_compute_unfirst(self, loop_input, accumulate, cal_num):
        """
        describe the non head axis of int64 indices calculation process
        Parameters
        ----------
        loop_input: the start address of x_gm
        accumulate: attr
        cal_num: the number of calculation in this loop
        """
        if self.support_multi_atmoic_add is False or self.x_dtype_before not in self.support_dtype:
            return
        if self.support_int8_atmoic_add is False and self.x_dtype_before not in self.support_dtype_310b:
            return
        if accumulate == True:
            self.tik_instance.set_atomic_add(self.x_dtype_before)
        # move indices to ub
        indices_ub_list = []
        for indices_list_idx in range(self.indices_count):
            tensor_name = "_".join(["indices_data_ub", str(indices_list_idx)])
            # apply to load indices_data
            indices_ub = self.tik_instance.Tensor(self.indices_type, [cal_num], name=tensor_name, scope=tik.scope_ubuf)
            self.tik_instance.data_move(indices_ub, self.input_tensors[4 + indices_list_idx][loop_input],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        (cal_num + self.indices_each_block - 1) // self.indices_each_block,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            indices_ub_list.append(indices_ub)
        value_ub = self.tik_instance.Tensor(self.value_dtype, (self.available_ub_size,),
                                            name="value_ub", scope=tik.scope_ubuf)
        value_back_num = self.tik_instance.Scalar("int64", name="value_back_num")
        value_back_num.set_as(self.indices_number * self.reserver_dim_number)
        value_first_num = self.tik_instance.Scalar("int64", name="value_first_num")
        value_first_num.set_as(self.x_shape_reserver_dim[0] * self.x_shape_per_dim[0] //
                               self.x_shape_reserver_dim[self.mask_dim - 1])
        # each time loop cal_num number of indices
        with self.tik_instance.for_range(0, cal_num) as indices_idx:
            addr_x = self.tik_instance.Scalar("int64", name="addr_x", init_value=0)
            addr_value = self.tik_instance.Scalar("int64", name="addr_value", init_value=0)
            # each addr of indices in input_x
            for indices_list_idx in range(self.indices_count):
                scalar_t = self.tik_instance.Scalar("int64", name="scalar_t")
                scalar_t.set_as(indices_ub_list[indices_list_idx][indices_idx])
                scalar_t.set_as(scalar_t + self.x_shape_per_dim[indices_list_idx + self.mask_dim])
                scalar_t.set_as(scalar_t % self.x_shape_per_dim[indices_list_idx + self.mask_dim])
                reserver_dim_temp = self.tik_instance.Scalar("int64", name="reserver_dim_temp")
                reserver_dim_temp.set_as(scalar_t * self.x_shape_reserver_dim[indices_list_idx + self.mask_dim])
                addr_x.set_as(addr_x + reserver_dim_temp)

            with self.tik_instance.for_range(0, value_first_num) as value_form_idx:
                addr_x_loop = self.tik_instance.Scalar("int64", name="addr_x_loop", init_value=0)
                addr_value.set_as(value_form_idx * value_back_num + \
                                  (loop_input + indices_idx) * self.reserver_dim_number)
                addr_x_loop.set_as(value_form_idx * self.x_shape_reserver_dim[self.mask_dim - 1] + addr_x)
                # update input_x corresponding of indices
                reserver_dim_loop = self.tik_instance.Scalar("int64", name="reserver_dim_loop", init_value=0)
                reserver_dim_loop.set_as(self.reserver_dim_number // self.available_ub_size)
                temp_scalar = self.tik_instance.Scalar("int64", name="temp_scalar", init_value=0)
                # update input_x corresponding of indices
                with self.tik_instance.for_range(0, reserver_dim_loop) as reserver_idx:
                    temp_scalar.set_as(reserver_idx * self.available_ub_size)
                    # move value to ub value is broadcast to the same shape as indices
                    self.index_put_compute_move_pad_each(self.available_ub_size, addr_value, addr_x_loop, temp_scalar)
            
                reserver_num = self.tik_instance.Scalar("int64", name="reserver_num", init_value=0)
                reserver_num.set_as(self.reserver_dim_number - reserver_dim_loop * self.available_ub_size)
                with self.tik_instance.if_scope(reserver_num > 0):
                    temp_scalar.set_as(self.reserver_dim_number // self.available_ub_size * self.available_ub_size)
                    self.index_put_compute_move_pad_each(reserver_num, addr_value,
                                                         addr_x_loop, temp_scalar, is_tail=True)
        self.tik_instance.set_atomic_add(0)
        
    # 'pylint: disable=too-many-locals
    def index_put_compute_move_pad(self, loop_input, cal_num):
        """
        describe the large number of int64 indices calculation process

        Parameters
        ----------
        loop_input: the start address of x_gm
        cal_num: the number of calculation in this loop

        """
        if self.support_multi_atmoic_add is False or self.x_dtype_before not in self.support_dtype:
            return
        if self.support_int8_atmoic_add is False and self.x_dtype_before not in self.support_dtype_310b:
            return
        if self.accumulate:
            if self.support_multi_atmoic_add and self.x_dtype_before in self.support_dtype:
                self.tik_instance.set_atomic_add(self.x_dtype_before)
            else:
                self.tik_instance.set_atomic_add("float32")
        else:
            self.tik_instance.set_atomic_add("float32")
            self.tik_instance.set_atomic_add(0)
        # move indices to ub
        indices_ub_list = []
        for indices_list_idx in range(self.indices_count):
            tensor_name = "_".join(["indices_data_ub", str(indices_list_idx)])
            # apply to load indices_data
            indices_ub = self.tik_instance.Tensor(self.indices_type, [cal_num], name=tensor_name,
                                                  scope=tik.scope_ubuf)
            self.tik_instance.data_move(indices_ub, self.input_tensors[4 + indices_list_idx][loop_input],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        (cal_num + self.indices_each_block - 1) // self.indices_each_block,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            indices_ub_list.append(indices_ub)
        value_ub = self.tik_instance.Tensor(self.value_dtype, (self.available_ub_size,),
                                            name="value_ub", scope=tik.scope_ubuf)
        # each time loop cal_num number of indices
        with self.tik_instance.for_range(0, cal_num) as indices_idx:
            addr_x = self.tik_instance.Scalar("int64", name="addr_x", init_value=0)
            addr_value = self.tik_instance.Scalar("int64", name="addr_value", init_value=0)
            # each addr of indices in input_x
            for indices_list_idx in range(self.indices_count):
                scalar_t = self.tik_instance.Scalar("int64", name="scalar_t")
                scalar_t.set_as(indices_ub_list[indices_list_idx][indices_idx])
                scalar_t.set_as(scalar_t + self.x_shape_per_dim[indices_list_idx])
                scalar_t.set_as(scalar_t % self.x_shape_per_dim[indices_list_idx])
                reserver_dim_temp = self.tik_instance.Scalar("int64", name="reserver_dim_temp")
                reserver_dim_temp.set_as(scalar_t * self.x_shape_reserver_dim[indices_list_idx])
                addr_x.set_as(addr_x + reserver_dim_temp)
            addr_value.set_as((loop_input + indices_idx) * self.reserver_dim_number)
            # update input_x corresponding of indices
            reserver_dim_loop = self.tik_instance.Scalar("int64", name="reserver_dim_loop", init_value=0)
            reserver_dim_loop.set_as(self.reserver_dim_number//self.available_ub_size)
            temp_scalar = self.tik_instance.Scalar("int64", name="temp_scalar", init_value=0)

            # update input_x corresponding of indices
            with self.tik_instance.for_range(0, reserver_dim_loop) as reserver_idx:
                temp_scalar.set_as(reserver_idx * self.available_ub_size)
                # move value to ub value is broadcast to the same shape as indices
                self.index_put_compute_move_pad_each(self.available_ub_size, addr_value, addr_x, temp_scalar)
        
            reserver_num = self.tik_instance.Scalar("int64", name="reserver_num", init_value=0)
            reserver_num.set_as(self.reserver_dim_number - reserver_dim_loop * self.available_ub_size)
            with self.tik_instance.if_scope(reserver_num > 0):
                temp_scalar.set_as(self.reserver_dim_number // self.available_ub_size * self.available_ub_size)
                self.index_put_compute_move_pad_each(reserver_num, addr_value, addr_x, temp_scalar, is_tail=True)
        self.tik_instance.set_atomic_add(0)

    def move_reserver_dim(self, value_ub, addr_value, addr_x):
        reserver_dim_loop = self.tik_instance.Scalar("int64", name="reserver_dim_loop", init_value=0)
        reserver_dim_loop.set_as(self.reserver_dim_number//self.available_ub_size)
        temp_scalar = self.tik_instance.Scalar("int64", name="temp_scalar", init_value=0)
        x_burst = (self.available_ub_size + self.x_data_each_block - 1) // self.x_data_each_block
        x_burst_before = (self.available_ub_size + self.x_data_each_block_before - 1) // self.x_data_each_block_before
        repeattime = (self.available_ub_size + Constant.MASK_NUM_32 - 1)//Constant.MASK_NUM_32
        # update input_x corresponding of indices
        with self.tik_instance.for_range(0, reserver_dim_loop) as reserver_idx:
            temp_scalar.set_as(reserver_idx * self.available_ub_size)
            # move value to ub value is broadcast to the same shape as indices
            if self.x_dtype_before == "bfloat16":
                value_ub_before = self.tik_instance.Tensor(self.value_dtype_before,
                                                           (self.available_ub_size,),
                                                           name="value_ub_before", scope=tik.scope_ubuf)
                self.tik_instance.data_move(value_ub_before,
                                            self.value_gm[addr_value + temp_scalar],
                                            constant.SID, constant.DEFAULT_NBURST, x_burst_before,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                self.tik_instance.vec_conv(Constant.MASK_NUM_32, '', value_ub, value_ub_before, repeattime, 8, 4)
            else:
                self.tik_instance.data_move(value_ub,
                                            self.value_gm[addr_value + temp_scalar],
                                            constant.SID, constant.DEFAULT_NBURST, x_burst,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            if self.x_dtype_before == "bfloat16":
                self.tik_instance.vec_conv(Constant.MASK_NUM_32, 'floor', value_ub_before, value_ub, repeattime, 4, 8)
                self.tik_instance.data_move(self.result_gm[addr_x + temp_scalar], value_ub_before, constant.SID,
                                            constant.DEFAULT_NBURST, x_burst_before,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            else:
                self.tik_instance.data_move(self.result_gm[addr_x + temp_scalar], value_ub, constant.SID,
                                            constant.DEFAULT_NBURST, x_burst,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
             
        reserver_num = self.tik_instance.Scalar("int64", name="reserver_num", init_value=0)
        reserver_num.set_as(self.reserver_dim_number - reserver_dim_loop * self.available_ub_size)
        with self.tik_instance.if_scope(reserver_num > 0):
            temp_scalar.set_as(self.reserver_dim_number // self.available_ub_size * self.available_ub_size)
            reserver_block = self.tik_instance.Scalar("int64", name="temp_scalar2", init_value=0)
            temp_full_block = self.tik_instance.Scalar("int64", name="temp_full_block", init_value=0)
            temp_full_block.set_as((reserver_num + self.x_data_each_block_before - 1) //
                                   self.x_data_each_block_before * self.x_data_each_block_before)
            reserver_block.set_as(temp_full_block - reserver_num)
            x_burst = (reserver_num + self.x_data_each_block - 1) // self.x_data_each_block
            x_burst_before = (reserver_num + self.x_data_each_block_before - 1) // self.x_data_each_block_before
            repeattime = (reserver_num + Constant.MASK_NUM_32 - 1) // Constant.MASK_NUM_32
            # move value to ub value is broadcast to the same shape as indices
            if self.x_dtype_before == "bfloat16":
                value_ub_before = self.tik_instance.Tensor(self.value_dtype_before,
                                                           (self.available_ub_size,),
                                                           name="value_ub_before", scope=tik.scope_ubuf)
                self.tik_instance.data_move(value_ub_before,
                                            self.value_gm[addr_value + temp_scalar],
                                            constant.SID, constant.DEFAULT_NBURST, x_burst_before,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                self.tik_instance.vec_conv(Constant.MASK_NUM_32, '', value_ub, value_ub_before, repeattime, 8, 4)
            else:
                self.tik_instance.data_move(value_ub,
                                            self.value_gm[addr_value + temp_scalar],
                                            constant.SID, constant.DEFAULT_NBURST, x_burst,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            with self.tik_instance.for_range(0, reserver_block) as reserver_over_idx:
                value_ub[reserver_num + reserver_over_idx].set_as(0)
            if self.x_dtype_before == "bfloat16":
                self.tik_instance.vec_conv(Constant.MASK_NUM_32, 'floor', value_ub_before, value_ub, repeattime, 4, 8)
                self.tik_instance.data_move(self.result_gm[addr_x + temp_scalar], value_ub_before, constant.SID,
                                            constant.DEFAULT_NBURST, x_burst_before,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            else:
                self.tik_instance.data_move(self.result_gm[addr_x + temp_scalar], value_ub, constant.SID,
                                            constant.DEFAULT_NBURST, x_burst,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                                                
    # 'pylint: disable=too-many-locals
    def index_put_compute_indices_large(self, loop_input, cal_num):
        """
        describe the large number of int64 indices calculation process

        Parameters
        ----------
        loop_input: the start address of x_gm
        cal_num: the number of calculation in this loop

        """
        if self.support_multi_atmoic_add is False and self.x_dtype_before != "float32":
            return
        if self.support_multi_atmoic_add and self.x_dtype_before not in self.support_dtype:
            return
        if self.support_int8_atmoic_add is False and self.x_dtype_before not in self.support_dtype_310b:
            return
        if self.support_multi_atmoic_add and self.x_dtype_before in self.support_dtype:
            self.tik_instance.set_atomic_add(self.x_dtype_before)
        else:
            self.tik_instance.set_atomic_add("float32")

        value_ub = self.tik_instance.Tensor(self.value_dtype,
                                            (self.available_ub_size,),
                                            name="value_ub", scope=tik.scope_ubuf)

        # move indices to ub
        indices_ub_list = []
        for indices_list_idx in range(self.indices_count):
            tensor_name = "_".join(["indices_data_ub", str(indices_list_idx)])
            # apply to load indices_data
            indices_ub = self.tik_instance.Tensor(self.indices_type, [cal_num], name=tensor_name,
                                                  scope=tik.scope_ubuf)
            self.tik_instance.data_move(indices_ub, self.input_tensors[4 + indices_list_idx][loop_input],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        (cal_num + self.indices_each_block - 1) // self.indices_each_block,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            indices_ub_list.append(indices_ub)

        # each time loop cal_num number of indices
        with self.tik_instance.for_range(0, cal_num) as indices_idx:
            addr_x = self.tik_instance.Scalar("int64", name="addr_x", init_value=0)
            addr_value = self.tik_instance.Scalar("int64", name="addr_value", init_value=0)
            # each addr of indices in input_x
            for indices_list_idx in range(self.indices_count):
                scalar_t = self.tik_instance.Scalar("int64", name="scalar_t")
                scalar_t.set_as(indices_ub_list[indices_list_idx][indices_idx])
                scalar_t.set_as(scalar_t + self.x_shape_per_dim[indices_list_idx])
                scalar_t.set_as(scalar_t % self.x_shape_per_dim[indices_list_idx])
                reserver_dim_temp = self.tik_instance.Scalar("int64", name="reserver_dim_temp")
                reserver_dim_temp.set_as(scalar_t * self.x_shape_reserver_dim[indices_list_idx])
                addr_x.set_as(addr_x + reserver_dim_temp)
            addr_value.set_as((loop_input + indices_idx) * self.reserver_dim_number)
            # update input_x corresponding of indices
            self.move_reserver_dim(value_ub, addr_value, addr_x)
        self.tik_instance.set_atomic_add(0)

    def data_move_mte3_function(self, loop_input, burst, result_ub):
        """
        move output data of updated value from gm to ub with each pinpang of each core

        Parameters
        ----------
        loop_input: the starting address of the gm data
        burst: the burst of each repeat
        result_ub: the ub tensor of output data

        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.result_gm[loop_input],
                                    result_ub, constant.SID,
                                    constant.DEFAULT_NBURST, burst,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                                 
    def move_reserver_dim_one(self, value_ub, addr_value, addr_x):
        temp_ub = self.tik_instance.Tensor(self.x_dtype, (self.x_data_each_block_before,),
                                           name="x_ub", scope=tik.scope_ubuf)
        if self.x_dtype_before == "bfloat16":
            value_ub_before = self.tik_instance.Tensor(self.value_dtype_before,
                                                        (self.x_data_each_block_before,),
                                                        name="value_ub_before", scope=tik.scope_ubuf)
            self.tik_instance.data_move(value_ub_before,
                                        self.value_gm[addr_value],
                                        constant.SID, constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            self.tik_instance.vec_conv(self.x_data_each_block_before, '', temp_ub, value_ub_before, 1, 8, 4)
        else:
            self.tik_instance.data_move(temp_ub,
                                        self.value_gm[addr_value],
                                        constant.SID, constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        if self.x_dtype == "float32" or self.x_dtype == "int32":
            self.tik_instance.vec_dup(self.x_data_each_block_before, value_ub, 0, 1, 8)
        else:
            for value_addr_id in range(self.x_data_each_block_before):
                value_ub[value_addr_id].set_as(0)
        value_ub[0].set_as(temp_ub[0])
        if self.x_dtype_before == "bfloat16":
            self.tik_instance.vec_conv(self.x_data_each_block_before, 'round', value_ub_before, value_ub, 1, 4, 8)
            self.tik_instance.data_move(self.result_gm[addr_x], value_ub_before, constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        else:
            self.tik_instance.data_move(self.result_gm[addr_x], value_ub, constant.SID,
                                        constant.DEFAULT_NBURST, 1,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)   

    # 'pylint: disable=too-many-locals
    def index_put_compute_indices_large_reserver_one(self, loop_input, cal_num):
        """
        describe the large number of int64 indices calculation process

        Parameters
        ----------
        loop_input: the start address of x_gm
        cal_num: the number of calculation in this loop

        """
        if self.support_multi_atmoic_add is False and self.x_dtype_before != "float32":
            return
        if self.support_multi_atmoic_add and self.x_dtype_before not in self.support_dtype:
            return
        if self.support_int8_atmoic_add is False and self.x_dtype_before not in self.support_dtype_310b:
            return
        if self.support_multi_atmoic_add and self.x_dtype_before in self.support_dtype:
            self.tik_instance.set_atomic_add(self.x_dtype_before)
        else:
            self.tik_instance.set_atomic_add("float32")

        value_ub = self.tik_instance.Tensor(self.value_dtype, (self.x_data_each_block_before,),
                                            name="value_ub", scope=tik.scope_ubuf)
        # move indices to ub
        indices_ub_list = []
        for indices_list_idx in range(self.indices_count):
            tensor_name = "_".join(["indices_data_ub", str(indices_list_idx)])
            # apply to load indices_data
            indices_ub = self.tik_instance.Tensor(self.indices_type, [cal_num], name=tensor_name,
                                                  scope=tik.scope_ubuf)
            self.tik_instance.data_move(indices_ub, self.input_tensors[4 + indices_list_idx][loop_input],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        (cal_num + self.indices_each_block - 1) // self.indices_each_block,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            indices_ub_list.append(indices_ub)
        # each time loop cal_num number of indices
        with self.tik_instance.for_range(0, cal_num) as indices_idx:
            addr_x = self.tik_instance.Scalar("int64", name="addr_x", init_value=0)
            addr_value = self.tik_instance.Scalar("int64", name="addr_value", init_value=0)
            # each addr of indices in input_x
            for indices_list_idx in range(self.indices_count):
                scalar_t = self.tik_instance.Scalar("int64", name="scalar_t")
                scalar_t.set_as(indices_ub_list[indices_list_idx][indices_idx])
                scalar_t.set_as(scalar_t + self.x_shape_per_dim[indices_list_idx])
                scalar_t.set_as(scalar_t % self.x_shape_per_dim[indices_list_idx])
                reserver_dim_temp = self.tik_instance.Scalar("int64", name="reserver_dim_temp")
                reserver_dim_temp.set_as(scalar_t * self.x_shape_reserver_dim[indices_list_idx])
                addr_x.set_as(addr_x + reserver_dim_temp)
            addr_value.set_as((loop_input + indices_idx) * self.reserver_dim_number)
            # move value to ub value is broadcast to the same shape as indices
            self.move_reserver_dim_one(value_ub, addr_value, addr_x)
        self.tik_instance.set_atomic_add(0)

    # 'pylint: disable=too-many-locals, too-many-statements
    def index_put_compute_int64(self, loop_input, burst, accumulate, cal_num, burst_before):
        """
        describe the int64 indices calculation process

        Parameters
        ----------
        loop_input: the start address of x_gm
        burst: the burst of data move
        accumulate: attr
        cal_num: the number of calculation in this loop

        """
        self.tik_instance.set_atomic_add("float32")
        self.tik_instance.set_atomic_add(0)
        value_burst = self.tik_instance.Scalar("int64", name="value_burst")
        value_burst.set_as(self.reserver_dim_number + self.value_data_each_block - 1)
        value_burst.set_as(value_burst // self.value_data_each_block)
        value_burst_pad = self.tik_instance.Scalar("int64", name="value_burst_pad")
        value_burst_pad.set_as(self.reserver_dim_number * self.value_dtype_bytes_size)
        value_burst_before = self.tik_instance.Scalar("int64", name="value_burst_before")
        value_burst_before.set_as(self.reserver_dim_number + self.x_data_each_block_before - 1)
        value_burst_before.set_as(value_burst_before // self.x_data_each_block_before)
        x_ub = self.tik_instance.Tensor(self.x_dtype, (cal_num,), name="x_ub", scope=tik.scope_ubuf)
        repeattime = (cal_num + Constant.MASK_NUM_32 - 1) // Constant.MASK_NUM_32
        if self.x_dtype_before == "bfloat16":
            x_ub_before = self.tik_instance.Tensor(self.x_dtype_before, (cal_num,),
                                                   name="x_ub_before", scope=tik.scope_ubuf)
            self.tik_instance.data_move(x_ub_before, self.x_gm[loop_input],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        burst_before, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            self.tik_instance.vec_conv(Constant.MASK_NUM_32, '', x_ub, x_ub_before, repeattime, 8, 4)
        else:
            self.tik_instance.data_move(x_ub, self.x_gm[loop_input],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        burst, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        indices_loop = self.tik_instance.Scalar("int64", name="indices_loop")
        indices_loop.set_as((self.indices_number + self.available_ub_size - 1) // self.available_ub_size)     
        cal_num1 = self.tik_instance.Scalar("int64", name="cal_num1")
        with self.tik_instance.for_range(0, indices_loop) as indices_loop_idx:
            indices_addr = indices_loop_idx * self.available_ub_size
            with self.tik_instance.if_scope(indices_loop_idx < (indices_loop - 1)):
                cal_num1.set_as(self.available_ub_size)
            with self.tik_instance.elif_scope(indices_loop_idx == (indices_loop - 1)):
                cal_num1.set_as(self.indices_number - indices_addr)
           
            # move indices to ub
            indices_ub_list = []
            for indices_list_idx in range(self.indices_count):
                tensor_name = "_".join(["indices_data_ub", str(indices_list_idx)])
                # apply to load indices_data
                indices_ub = self.tik_instance.Tensor(self.indices_type, [cal_num1], name=tensor_name,
                                                      scope=tik.scope_ubuf)
                self.tik_instance.data_move(indices_ub, self.input_tensors[4 + indices_list_idx][indices_addr],
                                            constant.SID, constant.DEFAULT_NBURST,
                                            (cal_num1 + self.indices_each_block - 1) // self.indices_each_block,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                indices_ub_list.append(indices_ub)
           
            value_ub = self.tik_instance.Tensor(self.value_dtype,
                                                (self.reserver_dim_number,),
                                                name="value_ub", scope=tik.scope_ubuf)
            # each time loop cal_num number of indices
            with self.tik_instance.for_range(0, cal_num1) as indices_idx:
                # move value to ub value is broadcast to the same shape as indices
                if self.x_dtype_before == "bfloat16":
                    value_ub_before = self.tik_instance.Tensor(self.x_dtype_before, (self.reserver_dim_number,),
                                                               name="value_ub_before", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(value_ub_before,
                                                self.value_gm[(indices_addr + indices_idx) * self.reserver_dim_number],
                                                constant.SID, constant.DEFAULT_NBURST, value_burst_before,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                    self.tik_instance.vec_conv(Constant.MASK_NUM_32, '', value_ub, value_ub_before, repeattime, 8, 4)
                else:
                    if self.support_multi_atmoic_add:
                        self.data_move_pad(value_ub,
                                           self.value_gm[(indices_addr + indices_idx) * self.reserver_dim_number],
                                           constant.DEFAULT_NBURST, value_burst_pad,
                                           constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                    else:
                        self.tik_instance.data_move(value_ub,
                                                    self.value_gm[(indices_addr + indices_idx) * self.reserver_dim_number],
                                                    constant.SID, constant.DEFAULT_NBURST, value_burst,
                                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                addr_x = self.tik_instance.Scalar("int64", name="addr_x", init_value=0)
                # each addr of indices in input_x
                for indices_list_idx in range(self.indices_count):
                    scalar_t = self.tik_instance.Scalar("int64", name="scalar_t")
                    scalar_t.set_as(indices_ub_list[indices_list_idx][indices_idx])
                    scalar_t.set_as(scalar_t + self.x_shape_per_dim[indices_list_idx])
                    scalar_t.set_as(scalar_t % self.x_shape_per_dim[indices_list_idx])
                    addr_x.set_as(addr_x + scalar_t * self.x_shape_reserver_dim[indices_list_idx])
                
                # update input_x corresponding of indices
                with self.tik_instance.if_scope(tik.all(addr_x >= loop_input,
                                                        (addr_x + self.reserver_dim_number) <= (loop_input + cal_num))):
                    addr_x.set_as(addr_x - loop_input)
                    with self.tik_instance.if_scope(self.reserver_dim_number == 1):
                        scalar_value = self.tik_instance.Scalar(self.value_dtype, name="scalar_value")
                        scalar_value.set_as(value_ub[0])
                        scalar_x = self.tik_instance.Scalar(self.x_dtype, name="scalar_x")
                        scalar_x.set_as(x_ub[addr_x])
                        with self.tik_instance.if_scope(accumulate):
                            x_ub[addr_x].set_as(scalar_x + scalar_value)
                        with self.tik_instance.else_scope():
                            x_ub[addr_x].set_as(scalar_value)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(0, self.reserver_dim_number) as cal_idx:
                            scalar_value = self.tik_instance.Scalar(self.value_dtype, name="scalar_value")
                            scalar_value.set_as(value_ub[cal_idx])
                            scalar_x = self.tik_instance.Scalar(self.x_dtype, name="scalar_x")
                            scalar_x.set_as(x_ub[addr_x + cal_idx])
                            with self.tik_instance.if_scope(accumulate):
                                x_ub[addr_x + cal_idx].set_as(scalar_x + scalar_value)
                            with self.tik_instance.else_scope():
                                x_ub[addr_x + cal_idx].set_as(scalar_value)
        if self.x_dtype_before == "bfloat16":
            self.tik_instance.vec_conv(Constant.MASK_NUM_32, 'round', x_ub_before, x_ub, repeattime, 4, 8)
            self.data_move_mte3_function(loop_input, burst_before, x_ub_before)
        else:
            self.data_move_mte3_function(loop_input, burst, x_ub)

    def caculation_process(self, core_index, loop_idx, burst, cal_num, burst_before):
        loop_input = core_index * self.core_data + loop_idx * self.available_ub_size
        with self.tik_instance.if_scope(self.tiling_mode == 0):
            self.index_put_compute_int64(loop_input, burst, self.accumulate, cal_num, burst_before)
        with self.tik_instance.elif_scope(self.tiling_mode == 1):
            if self.support_multi_atmoic_add:
                self.index_put_compute_move_pad(loop_input, cal_num)
            else:
                with self.tik_instance.if_scope(self.reserver_dim_number == 1):
                    self.index_put_compute_indices_large_reserver_one(loop_input, cal_num)
                with self.tik_instance.else_scope():
                    self.index_put_compute_indices_large(loop_input, cal_num)
        with self.tik_instance.elif_scope(self.tiling_mode == 2):
            self.index_put_compute_move_pad(loop_input, cal_num)
        with self.tik_instance.elif_scope(self.tiling_mode == 3):
            self.index_put_compute_unfirst(loop_input, self.accumulate, cal_num)
            
    def copy_only(self, core_index, loop_num, tail_num):
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            burst = (self.available_ub_size + self.x_data_each_block - 1) // self.x_data_each_block
            burst_before = (self.available_ub_size + self.x_data_each_block_before - 1) // self.x_data_each_block_before
            self.caculation_process(core_index, loop_idx, burst, self.available_ub_size, burst_before)
                
        with self.tik_instance.if_scope(tail_num > 0):
            burst = (tail_num + self.x_data_each_block - 1) // self.x_data_each_block
            burst_before = (tail_num + self.x_data_each_block_before - 1) // self.x_data_each_block_before
            self.caculation_process(core_index, loop_num, burst, tail_num, burst_before)
    
    def get_tiling_args(self, tiling_ub):
        """
        get runtime tiling params from tiling

        Parameters
        ----------

        Returns
        -------
        None
        """
        # read tiling int64 scalar
        self.box_num.set_as(tiling_ub[0])
        self.core_data.set_as(tiling_ub[1])
        self.core_used.set_as(tiling_ub[2])
        self.copy_loop.set_as(tiling_ub[3])
        self.copy_tail.set_as(tiling_ub[4])
        self.last_copy_loop.set_as(tiling_ub[5])
        self.last_copy_tail.set_as(tiling_ub[6])
        self.reserver_dim_number.set_as(tiling_ub[7])
        self.value_number.set_as(tiling_ub[8])
        self.indices_number.set_as(tiling_ub[9])
        self.indices_list_number.set_as(tiling_ub[10])
        self.dim_0.set_as(tiling_ub[11])
        self.dim_1.set_as(tiling_ub[12])
        self.dim_2.set_as(tiling_ub[13])
        self.dim_3.set_as(tiling_ub[14])
        self.dim_4.set_as(tiling_ub[15])
        self.dim_5.set_as(tiling_ub[16])
        self.dim_6.set_as(tiling_ub[17])
        self.dim_7.set_as(tiling_ub[18])
        self.x_shape_per_dim.set_as([self.dim_0, self.dim_1, self.dim_2, self.dim_3,
                                     self.dim_4, self.dim_5, self.dim_6, self.dim_7])
        self.reserver_dim_0.set_as(tiling_ub[19])
        self.reserver_dim_1.set_as(tiling_ub[20])
        self.reserver_dim_2.set_as(tiling_ub[21])
        self.reserver_dim_3.set_as(tiling_ub[22])
        self.reserver_dim_4.set_as(tiling_ub[23])
        self.reserver_dim_5.set_as(tiling_ub[24])
        self.reserver_dim_6.set_as(tiling_ub[25])
        self.reserver_dim_7.set_as(tiling_ub[26])
        self.x_shape_reserver_dim.set_as([self.reserver_dim_0, self.reserver_dim_1, self.reserver_dim_2,
                                          self.reserver_dim_3, self.reserver_dim_4, self.reserver_dim_5,
                                          self.reserver_dim_6, self.reserver_dim_7])
        self.tiling_mode.set_as(tiling_ub[27])
        self.available_ub_size.set_as((tiling_ub[28]))
        self.tiling_core_num.set_as((tiling_ub[29]))
        self.mask_dim.set_as(tiling_ub[30])
            
    def tik_instance_function(self):
        """
        the entry of index_put_v2 calculation

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 8, 0, 0)
            self.get_tiling_args(tiling_ub)
        # core process
        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_index:
            with self.tik_instance.if_scope(core_index < (self.core_used - 1)):
                self.copy_only(core_index, self.copy_loop, self.copy_tail)
            with self.tik_instance.elif_scope(core_index == (self.core_used - 1)):
                self.copy_only(core_index, self.last_copy_loop, self.last_copy_tail)

        opt_config = {}
        tbe_context.get_context().add_compile_info("vars",
                                                   {"core_num": self.core_num,
                                                    "x_data_each_block": self.x_data_each_block,
                                                    "each_repeat_block_number": self.each_repeat_block_number,
                                                    "ub_max_size": self.ub_max_size,
                                                    "indices_count": self.indices_count,
                                                    "soc_version": self.soc_version})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_tensors,
                                   outputs=[self.result_gm],
                                   flowtable=[self.tiling_gm], config=opt_config)
        return self.tik_instance


# 'pylint: disable=unused-argument, too-many-locals, too-many-lines, too-many-arguments
@register_operator("IndexPutV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.DYNAMIC_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def index_put_v2(x, value, indexed_sizes, indexed_strides,
                 indices, result, accumulate=False,
                 kernel_name="index_put_v2"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input x
    value : dict
        shape and dtype of update value
    indexed_sizes : dict
    indexed_strides : dict
    indices : list
        dynamic input
    accumulate : bool
        false means replace the x, true means accumulate value
    kernel_name : str
        kernel name, default value is "index_put_v2"

    Returns
    -------
    Nonevim i
    """
    x_dtype = x.get("dtype").lower()
    value_dtype = value.get("dtype").lower()
    if x_dtype != value_dtype:
        error_detail = "dtype of x and value should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", "value", error_detail)

    result_instance = IndexPut(x, value, indexed_sizes, indexed_strides,
                               indices, result, accumulate, kernel_name)
    instance = result_instance.tik_instance_function()
    return instance