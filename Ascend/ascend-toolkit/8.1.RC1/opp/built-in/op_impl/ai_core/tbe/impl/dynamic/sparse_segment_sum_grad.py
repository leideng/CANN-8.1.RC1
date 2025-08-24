# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
sparse_segment_sum_grad
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


class Constant:
    """
    The class for constant
    """
    BLOCK_BYTE_SIZE = 32
    MAX_REPEAT_NUM_EACH_CORE = 64
    SEGMENT_NUM_ONE_REPEAT = 64
    MAX_DATA_MOVE_BLOCK = 1024 * 2  # fp16,2KB data
    DATA_MEMSIZE_INVALID = 64
    MAX_INT64_VALUE = 2 ** 63 - 1
    RESERVED_UB = 1024 * 8
    GRAD_NUM_ONE_BLOCK_FP16 = 16
    DTYPE_FP32 = "float32"
    DTYPE_FP16 = "float16"
    DTYPE_INT32 = "int32"
    # tiling params dtype
    TILING_PARAM_DTYPE = "int32"
    # tiling params num
    TILING_PARAMS_NUM = 8

    # fp32 byte
    BYTE_FP32 = 4
    # int32 byte
    BYTE_INT32 = 4
    # byte of one block
    BYTE_BLOCK = 32
    # byte of one repeat block
    BYTE_REPEAT_BLOCK = 256
    # max repeat
    MAX_REPEAT = 255
    # full mask for fp32
    MASK_FP32 = 64
    # full mask for int32
    MASK_INT32 = 64
    # VECTOR FP32 SIZE
    VECTOR_FP32_SIZE = 64

    SELECT_KEY_MODE_FP32_INPUT_INVALID = 0
    SELECT_KEY_MODE_FP32_INPUT_VALID = 1


def _ceil_div(val, block):
    """
    compute ceil div
    """
    return (val + block - 1) // block


class SparseSegmentSumGradCompute:
    """
    SPARSE_SEGMENT_SUM_GRAD
    """

    def __init__(self, grad, indices, segment_ids, output_dim0, output, kernel_name):
        self.tik_instance = tik.Tik()
        self.kernel_name = kernel_name
        self.dtype_grad = grad.get("dtype").lower()
        self.dtype_indices = indices.get("dtype").lower()
        self.dtype_segment_ids = segment_ids.get("dtype").lower()
        self.dtype_output_dim0 = output_dim0.get("dtype").lower()
        self.data_grad = self.tik_instance.Tensor(self.dtype_grad, (Constant.MAX_INT64_VALUE,),
                                                  name="grad", scope=tik.scope_gm)
        self.data_indices = self.tik_instance.Tensor(self.dtype_indices, (Constant.MAX_INT64_VALUE,),
                                                     name="indices", scope=tik.scope_gm)
        self.data_segment_ids = self.tik_instance.Tensor(self.dtype_segment_ids, (Constant.MAX_INT64_VALUE,),
                                                         name="segment_ids", scope=tik.scope_gm)
        self.data_output_dim0 = self.tik_instance.Tensor(self.dtype_output_dim0, (Constant.MAX_INT64_VALUE,),
                                                         name="output_dim0", scope=tik.scope_gm)
        self.grad_num_one_block = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="grad_num_one_block",
                                                           init_value=8)
        self.atomic_support_fp16 = tbe_platform.api_check_support("tik.set_atomic_add", "float16")
        if not self.atomic_support_fp16:
            self.data_output = self.tik_instance.Tensor(Constant.DTYPE_FP32, (Constant.MAX_INT64_VALUE,),
                                                        name="output", scope=tik.scope_gm, is_atomic_add=True)
        else:
            self.data_output = self.tik_instance.Tensor(self.dtype_grad, (Constant.MAX_INT64_VALUE,),
                                                        name="output", scope=tik.scope_gm, is_atomic_add=True)
        if self.dtype_grad == Constant.DTYPE_FP16:
            self.grad_num_one_block.set_as(Constant.GRAD_NUM_ONE_BLOCK_FP16)
        self.one_repeat_num_indices = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="one_repeat_num_indices",
                                                               init_value=64)
        self.one_block_size_indices = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="one_block_size_indices",
                                                               init_value=8)
        self.repeat_indices = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat_indices", init_value=1)
        self.one_repeat_num_segment_ids = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                                   name="one_repeat_num_segment_ids",
                                                                   init_value=64)
        self.one_block_size_segment_ids = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                                   name="one_block_size_segment_ids",
                                                                   init_value=8)
        self.repeat_segment_ids = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat_segment_ids",
                                                           init_value=1)
        self.tiling_ub = None
        self.input_data_ub = None
        self.input_data_ub_fp32 = None
        self.select_key = self.tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="select_key")
        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB
        self.need_core_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="need_core_num")
        self.segment_num_each_core = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="segment_num_front_core")
        self.segment_num_rest = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="segment_num_last_core")
        self.grad_second_dim_size = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="grad_second_dim_size")
        self.new_core_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="new_core_num")

    def sparse_segment_sum_grad_compute(self):
        """
        The tik implementation of operator Assign
        """
        self._init_tiling_param()
        self._enable_atomic_add()
        with self.tik_instance.for_range(0, self.new_core_num, block_num=self.new_core_num) as core_index:
            with self.tik_instance.if_scope(core_index < self.need_core_num):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.select_key == Constant.SELECT_KEY_MODE_FP32_INPUT_INVALID):
                        self.input_data_ub = self.tik_instance.Tensor(Constant.DTYPE_FP32,
                                                                      (Constant.DATA_MEMSIZE_INVALID,),
                                                                      name="input_data_ub", scope=tik.scope_ubuf)
                        self.tik_instance.vec_dup(Constant.MASK_FP32, self.input_data_ub[0], 0, 1, 8)
                        self.tik_instance.data_move(self.data_output[0], self.input_data_ub[0], 0, 1, 1, 0, 0)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(self.select_key == Constant.SELECT_KEY_MODE_FP32_INPUT_VALID):
                        self.input_data_ub = self.tik_instance.Tensor(
                            self.dtype_grad, (Constant.MAX_DATA_MOVE_BLOCK * self.grad_num_one_block,),
                            name="input_data_ub", scope=tik.scope_ubuf)
                        self.input_data_ub_fp32 = self.tik_instance.Tensor(
                            Constant.DTYPE_FP32, (Constant.MAX_DATA_MOVE_BLOCK * self.grad_num_one_block,),
                            name="input_data_ub_fp32", scope=tik.scope_ubuf)
                        self._run_one_core(core_index)
        self._disable_atomic_add()
        tbe_context.get_context().add_compile_info("vars", {"ub_size": self.ub_size, "core_num": self.ai_core_num})
        opt_config = {"enable_const_fold": True, "out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.data_grad, self.data_indices, self.data_segment_ids,
                                           self.data_output_dim0),
                                   outputs=(self.data_output,), flowtable=(self.tiling_gm,), config=opt_config)

    def _get_tiling_params(self):
        """
        get runtime tiling parameters from tiling
        """
        # read tiling int64 scalar
        self.select_key.set_as(self.tiling_ub[0])
        self.need_core_num.set_as(self.tiling_ub[1])
        self.segment_num_each_core.set_as(self.tiling_ub[2])
        self.segment_num_rest.set_as(self.tiling_ub[3])
        self.grad_second_dim_size.set_as(self.tiling_ub[4])
        self.new_core_num.set_as(self.tiling_ub[5])

    def _enable_atomic_add(self):
        """
        enable atomic add
        """
        if (self.dtype_grad == Constant.DTYPE_FP16) and self.atomic_support_fp16:
            self.tik_instance.set_atomic_add(2)
        elif tbe_platform.api_check_support("tik.set_atomic_add"):
            self.tik_instance.set_atomic_add(1)

    def _disable_atomic_add(self):
        """
        disable atomic add
        """
        if tbe_platform.api_check_support("tik.set_atomic_add"):
            self.tik_instance.set_atomic_add(0)

    def _init_tiling_param(self):
        """
        _init_tiling_param
        """
        with self.tik_instance.new_stmt_scope():
            self.tiling_ub = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                      name="tiling_ub", scope=tik.scope_ubuf)
            # mov tiling params from gm to ub
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1,
                                        Constant.TILING_PARAMS_NUM * Constant.BYTE_INT32 // Constant.BYTE_BLOCK, 0, 0)
            self._get_tiling_params()

    def _run_one_loop(self, segment_num_front_part, repeat_num_each, segment_increment, segment_num_front_core,
                      is_last):
        """
        _run_one_loop
        """
        segment_offset_gm = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="segment_offset_gm")
        segment_offset_gm.set_as(segment_num_front_core + segment_num_front_part)
        indices_nburst = repeat_num_each * self.repeat_indices
        segment_ids_nburst = repeat_num_each * self.repeat_segment_ids

        indice_calc_mem_size = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="indice_calc_mem_size")
        indice_calc_mem_size.set_as(
            _ceil_div(segment_increment, self.one_repeat_num_indices) * self.one_repeat_num_indices)
        segment_ids_calc_mem_size = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="indice_calc_mem_size")
        segment_ids_calc_mem_size.set_as(
            _ceil_div(segment_increment, self.one_repeat_num_segment_ids) * self.one_repeat_num_segment_ids)
        ub_indices = self.tik_instance.Tensor(self.dtype_indices, (indice_calc_mem_size,), name="ub_indices",
                                              scope=tik.scope_ubuf)
        ub_segment_ids = self.tik_instance.Tensor(self.dtype_segment_ids, (segment_ids_calc_mem_size,),
                                                  name="ub_segment_ids", scope=tik.scope_ubuf)
        indices_offset_stride = self.tik_instance.Tensor(self.dtype_indices, (indice_calc_mem_size,),
                                                         name="indices_offset_stride", scope=tik.scope_ubuf)
        segment_ids_offset_stride = self.tik_instance.Tensor(self.dtype_segment_ids, (segment_ids_calc_mem_size,),
                                                             name="segment_ids_offset_stride", scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(not is_last):
            self.tik_instance.data_move(ub_indices, self.data_indices[segment_offset_gm], 0, indices_nburst,
                                        self.one_block_size_indices, 0, 0)
            self.tik_instance.data_move(ub_segment_ids, self.data_segment_ids[segment_offset_gm], 0,
                                        segment_ids_nburst, self.one_block_size_segment_ids, 0, 0)
        with self.tik_instance.else_scope():
            indices_burst_len_front_part = _ceil_div(segment_increment, self.one_block_size_indices)
            segment_burst_len_front_part = _ceil_div(segment_increment, self.one_block_size_segment_ids)
            self.tik_instance.data_move(ub_indices, self.data_indices[segment_offset_gm], 0, indices_nburst,
                                        indices_burst_len_front_part, 0, 0)
            self.tik_instance.data_move(ub_segment_ids, self.data_segment_ids[segment_offset_gm], 0,
                                        segment_ids_nburst, segment_burst_len_front_part, 0, 0)
        self.tik_instance.vec_dup(self.one_repeat_num_indices, indices_offset_stride, self.grad_second_dim_size,
                                  indices_nburst, self.one_block_size_indices)
        self.tik_instance.vec_dup(self.one_repeat_num_segment_ids, segment_ids_offset_stride, self.grad_second_dim_size,
                                  segment_ids_nburst, self.one_block_size_segment_ids)
        self.tik_instance.vec_mul(self.one_repeat_num_indices, ub_indices, ub_indices,
                                  indices_offset_stride, indices_nburst,
                                  self.one_block_size_indices, self.one_block_size_indices, self.one_block_size_indices)
        self.tik_instance.vec_mul(self.one_repeat_num_segment_ids, ub_segment_ids, ub_segment_ids,
                                  segment_ids_offset_stride, segment_ids_nburst,
                                  self.one_block_size_segment_ids, self.one_block_size_segment_ids,
                                  self.one_block_size_segment_ids)
        with self.tik_instance.for_range(0, segment_increment) as index:
            self._handle_grad_data_row(ub_indices[index], ub_segment_ids[index])

    def _run_one_core(self, core_index):
        """
        _run_one_core
        """
        actual_segment_num_each_core = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                                name="actual_segment_num_each_core",
                                                                init_value=0)
        actual_segment_num_each_core.set_as(self.segment_num_each_core)
        with self.tik_instance.if_scope(tik.all(core_index < self.segment_num_rest, self.segment_num_rest != 0)):
            actual_segment_num_each_core.set_as(self.segment_num_each_core + 1)

        segment_num_front_core = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="segment_num_front_core",
                                                          init_value=0)
        with self.tik_instance.if_scope(actual_segment_num_each_core > self.segment_num_each_core):
            segment_num_front_core.set_as(actual_segment_num_each_core * core_index)
        with self.tik_instance.elif_scope(actual_segment_num_each_core == self.segment_num_each_core):
            segment_num_front_core.set_as(self.segment_num_rest + core_index * self.segment_num_each_core)

        segment_num_front_part = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="segment_num_front_part",
                                                          init_value=0)
        segment_increment = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="segment_increment", init_value=0)
        repeat_num_one_part = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="repeat_num_one_part", init_value=0)

        segment_mov_times_each_core = actual_segment_num_each_core // Constant.MAX_REPEAT_NUM_EACH_CORE // \
                                      Constant.SEGMENT_NUM_ONE_REPEAT
        repeat_num_one_part.set_as(Constant.MAX_REPEAT_NUM_EACH_CORE)
        with self.tik_instance.for_range(0, segment_mov_times_each_core) as index:
            segment_num_front_part.set_as(
                index * Constant.MAX_REPEAT_NUM_EACH_CORE * Constant.SEGMENT_NUM_ONE_REPEAT)
            segment_increment.set_as(Constant.MAX_REPEAT_NUM_EACH_CORE * Constant.SEGMENT_NUM_ONE_REPEAT)
            self._run_one_loop(segment_num_front_part, repeat_num_one_part, segment_increment,
                               segment_num_front_core, False)

        # rest repeat
        repeat_num_one_part.set_as(
            actual_segment_num_each_core % (Constant.MAX_REPEAT_NUM_EACH_CORE * Constant.SEGMENT_NUM_ONE_REPEAT)
            // Constant.SEGMENT_NUM_ONE_REPEAT)
        with self.tik_instance.if_scope(repeat_num_one_part > 0):
            segment_num_front_part.set_as(
                segment_mov_times_each_core * Constant.MAX_REPEAT_NUM_EACH_CORE * Constant.SEGMENT_NUM_ONE_REPEAT)
            segment_increment.set_as(repeat_num_one_part * Constant.SEGMENT_NUM_ONE_REPEAT)
            self._run_one_loop(segment_num_front_part, repeat_num_one_part, segment_increment,
                               segment_num_front_core, False)

        # rest less than 64
        last_num = actual_segment_num_each_core % Constant.SEGMENT_NUM_ONE_REPEAT % Constant.SEGMENT_NUM_ONE_REPEAT
        with self.tik_instance.if_scope(last_num > 0):
            segment_num_front_part.set_as(
                segment_mov_times_each_core * Constant.MAX_REPEAT_NUM_EACH_CORE * Constant.SEGMENT_NUM_ONE_REPEAT \
                + repeat_num_one_part * Constant.SEGMENT_NUM_ONE_REPEAT)
            segment_increment.set_as(last_num)
            repeat_num_one_part.set_as(1)
            self._run_one_loop(segment_num_front_part, repeat_num_one_part, segment_increment,
                               segment_num_front_core, True)

    def _vconv_fp162fp32(self, src, src_start, dst, dst_start, ele_num):
        total_repeat_time = ele_num // Constant.VECTOR_FP32_SIZE
        remain_ele = ele_num % Constant.VECTOR_FP32_SIZE
        mask_value = Constant.VECTOR_FP32_SIZE

        repeat_max_time = total_repeat_time // Constant.MAX_REPEAT
        remain_repeat_time = total_repeat_time % Constant.MAX_REPEAT

        src_stride, dst_stride = 4, 8
        with self.tik_instance.if_scope(repeat_max_time > 0):
            with self.tik_instance.for_range(0, repeat_max_time) as loop1:
                self.tik_instance.vconv(Constant.MASK_FP32, "",
                                        dst[dst_start + loop1 * Constant.MAX_REPEAT * mask_value],
                                        src[src_start + loop1 * Constant.MAX_REPEAT * mask_value], Constant.MAX_REPEAT,
                                        1, 1, dst_stride, src_stride)
        with self.tik_instance.if_scope(remain_repeat_time > 0):
            self.tik_instance.vconv(Constant.MASK_FP32, "",
                                    dst[dst_start + repeat_max_time * Constant.MAX_REPEAT * mask_value],
                                    src[src_start + repeat_max_time * Constant.MAX_REPEAT * mask_value],
                                    remain_repeat_time, 1, 1, dst_stride, src_stride)
        with self.tik_instance.if_scope(remain_ele > 0):
            self.tik_instance.vconv(
                remain_ele, "", dst[dst_start + repeat_max_time * Constant.MAX_REPEAT *
                                    mask_value + remain_repeat_time * mask_value],
                src[src_start + repeat_max_time * Constant.MAX_REPEAT *
                    mask_value + remain_repeat_time * mask_value], 1, 1, 1, dst_stride, src_stride)

    def _mov_grad(self, offset_in, offset_out, burst_len, grad_num_last=0):
        self.tik_instance.data_move(self.input_data_ub, self.data_grad[offset_in], 0, 1, burst_len, 0, 0)
        with self.tik_instance.if_scope(grad_num_last != 0):
            erase_data_num = self.grad_num_one_block - grad_num_last
            with self.tik_instance.for_range(0, erase_data_num) as index_tail:
                self.input_data_ub[grad_num_last + index_tail] = 0
        with self.tik_instance.if_scope((self.dtype_grad == Constant.DTYPE_FP16) and (not self.atomic_support_fp16)):
            self._vconv_fp162fp32(self.input_data_ub, 0, self.input_data_ub_fp32, 0,
                                  burst_len * self.grad_num_one_block)
            self.tik_instance.data_move(self.data_output[offset_out], self.input_data_ub_fp32, 0, 1, burst_len * 2, 0,
                                        0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.data_output[offset_out], self.input_data_ub, 0, 1, burst_len, 0, 0)

    def _handle_grad_data_row(self, para_dst_offset, para_src_offset):
        """
        _handle_grad_data_row
        """
        ub_indices = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="ub_indices", init_value=para_dst_offset)
        ub_segment_ids = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="ub_segment_ids",
                                                  init_value=para_src_offset)
        input_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="input_offset", init_value=0)
        output_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="output_offset", init_value=0)

        grad_block_num_front_part = self.grad_second_dim_size // self.grad_num_one_block
        grad_mov_times_one_row = grad_block_num_front_part // Constant.MAX_DATA_MOVE_BLOCK
        with self.tik_instance.if_scope(grad_mov_times_one_row >= 2):
            with self.tik_instance.for_range(0, grad_mov_times_one_row, thread_num=2) as index:
                input_offset.set_as(ub_segment_ids + index * Constant.MAX_DATA_MOVE_BLOCK * self.grad_num_one_block)
                output_offset.set_as(ub_indices + index * Constant.MAX_DATA_MOVE_BLOCK * self.grad_num_one_block)
                self._mov_grad(input_offset, output_offset, Constant.MAX_DATA_MOVE_BLOCK)
        with self.tik_instance.elif_scope(grad_mov_times_one_row == 1):
            input_offset.set_as(ub_segment_ids)
            output_offset.set_as(ub_indices)
            self._mov_grad(input_offset, output_offset, Constant.MAX_DATA_MOVE_BLOCK)

        last_block_num = grad_block_num_front_part % Constant.MAX_DATA_MOVE_BLOCK
        with self.tik_instance.if_scope(last_block_num > 0):
            input_offset.set_as(
                ub_segment_ids + grad_mov_times_one_row * Constant.MAX_DATA_MOVE_BLOCK * self.grad_num_one_block)
            output_offset.set_as(
                ub_indices + grad_mov_times_one_row * Constant.MAX_DATA_MOVE_BLOCK * self.grad_num_one_block)
            self._mov_grad(input_offset, output_offset, last_block_num)

        # rest nums, return address
        grad_num_last_part = self.grad_second_dim_size % self.grad_num_one_block
        with self.tik_instance.if_scope(grad_num_last_part > 0):
            input_offset.set_as(ub_segment_ids + grad_block_num_front_part * self.grad_num_one_block)
            output_offset.set_as(ub_indices + grad_block_num_front_part * self.grad_num_one_block)
            self._mov_grad(input_offset, output_offset, 1, grad_num_last_part)


@register_operator("SparseSegmentSumGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sparse_segment_sum_grad(grad, indices, segment_ids, output_dim0, output, kernel_name="sparse_segment_sum_grad"):
    """
    calculating data

    Parameters
    ----------
    grad : dict
        shape and dtype of input, dtype only support float16, float32.
    indices : dict
        shape and dtype of input, the length of shape must be one, dtype only support int32, int64.
    segment_ids : dict
        shape and dtype of input, the length of shape must be one, dtype only support int32, int64.
    output_dim0 : dict
        shape and dtype of input, it is a scalar.
    output : dict
        shape and dtype of output, Must have the same data type as grad.
    kernel_name : str
        kernel name, default value is "sparse_segment_sum_grad"

    Returns
    -------
    None
    """
    sparse_segment_sum_grad_compute = SparseSegmentSumGradCompute(grad, indices, segment_ids, output_dim0, output,
                                                                  kernel_name)
    sparse_segment_sum_grad_compute.sparse_segment_sum_grad_compute()
