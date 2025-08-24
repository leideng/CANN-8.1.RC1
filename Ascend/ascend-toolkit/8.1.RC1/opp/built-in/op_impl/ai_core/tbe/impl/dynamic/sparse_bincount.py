#  Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ============================================================================
"""
SparseBincount
"""
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator


class Constant:
    """
    The class for constant
    """
    # 8k UB buffer is a reserved space
    RESERVE_SIZE = 8 * 1024
    MAX_INT32 = 2 ** 31 - 1
    MAX_REPEAT_TIME = 255
    DUM_NUM = 2
    TILING_ARG_NUM = 17
    INDICES_TENSOR_NUM = 8
    DST_REP_STRIDE = 8
    DTYPE_FP32 = "float32"
    DTYPE_INT32 = "int32"
    DTYPE_INT64 = "int64"
    BYTE_FP32 = 4
    BYTE_INT32 = 4
    BYTE_INT64 = 8
    TILING_PARA_DTYPE = DTYPE_INT64
    BLOCK_BYTE_SIZE = 32
    MASK_INDICES = 8
    MASK_32BIT = 64
    MASK_64BIT = 32
    ONE_DIMENSION_SMALL_OUTPUT = 1
    ONE_DIMENSION_LARGE_OUTPUT = 2
    MULTI_DIMENSIONS_SMALL_OUTPUT = 3
    MULTI_DIMENSIONS_LARGE_OUTPUT = 4
    BINARY_OUTPUT_IS_TRUE = True
    BINARY_OUTPUT_IS_FALSE = False


class SparseBinCount:
    """
    The class of SparseBincount op
    """

    def __init__(
            self,
            indices,
            values,
            dense_shape,
            size,
            weights,
            output,
            binary_output,
            kernel_name="SparseBincount"):
        """
        algorithm:SparseBincount
        Counts the number of occurrences of each value in an integer array.

        Parameters
        ----------
        indices : dict
            dict with keys(shape and dtype) of 2D tensor
        values: dict
            dict with keys(shape and dtype) of 1D tensor
        dense_shape : dict
            dict with keys(shape and dtype) of 1D tensor
        size: dict
            dict with the scalar
        weights: dict
            dict with keys(shape and dtype) of 1D tensor
        output: dict
            dict with keys(shape and dtype) of 1D tensor
        binary_output : bool
            Whether the kernel should count the appearance or number of occurrences
        kernel_name : str
            kernel name, default value is "SparseBincount"

        Returns
        -------
        None
        """

        self.dtype_indices = indices.get('dtype').lower()
        self.dtype_values = values.get('dtype').lower()
        self.dtype_dense_shape = dense_shape.get('dtype').lower()
        self.dtype_size = size.get("dtype").lower()
        self.dtype_weights = weights.get('dtype').lower()
        self.dtype_output = output.get('dtype').lower()
        self.binary_output = binary_output
        self.kernel_name = kernel_name
        self.tiling_dtype = Constant.TILING_PARA_DTYPE

        self.tik_instance = tik.Tik()
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE) - Constant.RESERVE_SIZE
        self.total_core_number = cce.get_soc_spec(cce.CORE_NUM)
        self.tiling_each_block = Constant.BLOCK_BYTE_SIZE // Constant.BYTE_INT64
        self.value_num_each_block = Constant.BLOCK_BYTE_SIZE // Constant.BYTE_INT32

        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                                  scope=tik.scope_gm, name='tiling_gm')
        self.indices_gm = self.tik_instance.Tensor(self.dtype_indices, (Constant.MAX_INT32,),
                                                   scope=tik.scope_gm, name="indices_gm")
        self.values_gm = self.tik_instance.Tensor(self.dtype_values, (Constant.MAX_INT32,),
                                                  scope=tik.scope_gm, name="values_gm")
        self.dense_shape_gm = self.tik_instance.Tensor(self.dtype_dense_shape, (Constant.MAX_INT32,),
                                                       scope=tik.scope_gm, name="dense_shape_gm")
        self.size_gm = self.tik_instance.Tensor(self.dtype_size, (Constant.TILING_ARG_NUM,),
                                                scope=tik.scope_gm, name="size_gm")
        self.weights_gm = self.tik_instance.Tensor(self.dtype_weights, (Constant.MAX_INT32,),
                                                   scope=tik.scope_gm, name="weights_gm")
        self.output_gm = self.tik_instance.Tensor(self.dtype_output, (Constant.MAX_INT32,),
                                                  scope=tik.scope_gm, name="output_gm", is_atomic_add=True)
        self.tiling_ub = self.tik_instance.Tensor(
            self.tiling_dtype, (Constant.TILING_ARG_NUM,), name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1,
                                    (Constant.TILING_ARG_NUM + self.tiling_each_block - 1) // self.tiling_each_block,
                                    0, 0)

        self.tiling_mode = self.tik_instance.Scalar(self.tiling_dtype, name='tiling_mode')
        self.indices_line = self.tik_instance.Scalar(self.tiling_dtype, name='indices_line')
        self.indices_row = self.tik_instance.Scalar(self.tiling_dtype, name='indices_row')
        self.dense_shape_data = self.tik_instance.Scalar(self.tiling_dtype, name='dense_shape_data')
        self.size_data = self.tik_instance.Scalar(self.tiling_dtype, name='size_data')
        self.used_core_num = self.tik_instance.Scalar(self.tiling_dtype, name='used_core_num')
        self.total_values_num = self.tik_instance.Scalar(self.tiling_dtype, name='total_values_num')
        self.total_weights_num = self.tik_instance.Scalar(self.tiling_dtype, name='total_weights_num')
        self.per_core_calc_max_values_num = self.tik_instance.Scalar(self.tiling_dtype,
                                                                     name='per_core_calc_max_values_num')
        self.per_core_calc_max_output_num = self.tik_instance.Scalar(self.tiling_dtype,
                                                                     name='per_core_calc_max_output_num')
        self.per_core_calc_num = self.tik_instance.Scalar(self.tiling_dtype, name='per_core_calc_num')
        self.last_core_calc_num = self.tik_instance.Scalar(self.tiling_dtype, name='last_core_calc_num')

        self.output_move_num = self.tik_instance.Scalar(self.tiling_dtype, name='output_move_num')
        self.output_tail_burst_len = self.tik_instance.Scalar(self.tiling_dtype, name='output_tail_burst_len')
        # core info
        self.ub_max_tensor_size = self.tik_instance.Scalar(self.tiling_dtype, name='ub_max_tensor_size')
        self.ub_max_block_len = self.tik_instance.Scalar(self.tiling_dtype, name='ub_max_block_len')
        self.new_core_num = self.tik_instance.Scalar(self.tiling_dtype, name='new_core_num')

        self.tiling_mode.set_as(self.tiling_ub[0])
        self.indices_line.set_as(self.tiling_ub[1])
        self.indices_row.set_as(self.tiling_ub[2])
        self.dense_shape_data.set_as(self.tiling_ub[3])
        self.size_data.set_as(self.tiling_ub[4])
        self.used_core_num.set_as(self.tiling_ub[5])
        self.total_values_num.set_as(self.tiling_ub[6])
        self.total_weights_num.set_as(self.tiling_ub[7])
        self.per_core_calc_max_values_num.set_as(self.tiling_ub[8])
        self.per_core_calc_max_output_num.set_as(self.tiling_ub[9])
        self.per_core_calc_num.set_as(self.tiling_ub[10])
        self.last_core_calc_num.set_as(self.tiling_ub[11])
        self.output_move_num.set_as(self.tiling_ub[12])
        self.output_tail_burst_len.set_as(self.tiling_ub[13])
        self.ub_max_tensor_size.set_as(self.tiling_ub[14])
        self.ub_max_block_len.set_as(self.tiling_ub[15])
        self.new_core_num.set_as(self.tiling_ub[16])

        self.indices_ub = self.tik_instance.Tensor(Constant.DTYPE_INT64, (Constant.INDICES_TENSOR_NUM,),
                                                   name='indices_ub', scope=tik.scope_ubuf)
        self.values_ub = self.tik_instance.Tensor(self.dtype_values, (self.ub_max_tensor_size,),
                                                  name='values_ub', scope=tik.scope_ubuf)
        self.weights_ub = self.tik_instance.Tensor(self.dtype_weights, (self.ub_max_tensor_size,),
                                                   name='weights_ub', scope=tik.scope_ubuf)
        self.output_ub = self.tik_instance.Tensor(Constant.DTYPE_FP32, (self.ub_max_tensor_size,),
                                                  name='output_ub', scope=tik.scope_ubuf)

    def data_move_gm2ub(self, input_offset, values_burst_len):
        """
        ub data_move gm
        Parameters
        ----------
        input_offset : input offset
        loop_count : value and weights index
        values_burst_len : values data_move burst

        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.values_ub, self.values_gm[input_offset], 0, 1, values_burst_len, 0, 0)
        self.tik_instance.data_move(self.weights_ub, self.weights_gm[input_offset], 0, 1, values_burst_len, 0, 0)

    def enable_atomic_add(self):
        """
        enable atomic add

        Parameters
        ----------
        tik_instance : self.tik_instance

        Returns
        -------
        None
        """
        if cce.api_check_support("tik.set_atomic_add"):
            self.tik_instance.set_atomic_add(1)

    def disable_atomic_add(self):
        """
        disable atomic add

        Parameters
        ----------
        tik_instance : self.tik_instance

        Returns
        -------
        None
        """
        if cce.api_check_support("tik.set_atomic_add"):
            self.tik_instance.set_atomic_add(0)

    def compute_one_dimension_small_output(self, loop_value_num, input_offset, values_burst_len, output_burst_len):
        """
        compute one dimension when output size is small

        Parameters
        ----------
        loop_value_num : value need loop num
        input_offset : input offset
        values_burst_len : values data_move burst
        output_burst_len : output data_move burst

        Returns
        -------
        None
        """
        self.init_values_ub()
        self.init_weights_ub()
        self.data_move_gm2ub(input_offset, values_burst_len)
        with self.tik_instance.for_range(0, loop_value_num) as ub_index:
            value = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='value', init_value=0)
            value.set_as(self.values_ub[ub_index])
            weights_data = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='weights_data', init_value=0)
            with self.tik_instance.if_scope(tik.any(self.total_weights_num == 0,
                                                    self.binary_output == Constant.BINARY_OUTPUT_IS_TRUE)):
                weights_data.set_as(1.0)
            with self.tik_instance.if_scope(tik.all(self.total_weights_num != 0)):
                weights_data.set_as(self.weights_ub[ub_index])
            with self.tik_instance.if_scope(value < self.size_data):
                output_data = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='output_data', init_value=0)
                output_data.set_as(self.output_ub[value])
                result = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='result', init_value=0)
                result.set_as(output_data + weights_data)
                self.output_ub[value].set_as(result)
        self.tik_instance.data_move(self.output_gm, self.output_ub, 0, 1, output_burst_len, 0, 0)

    def init_values_ub(self):
        """
        initilize values ub to zero

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        mask = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='mask', init_value=0)
        mask.set_as(Constant.MASK_32BIT)
        dup_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='dup_num', init_value=0)
        dup_num.set_as(Constant.DUM_NUM)
        repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat', init_value=0)
        repeat.set_as(self.per_core_calc_max_values_num // Constant.MASK_32BIT // Constant.DUM_NUM)
        with self.tik_instance.for_range(0, dup_num) as dup_loop:
            self.tik_instance.vec_dup(mask, self.output_ub[dup_loop * self.per_core_calc_max_values_num // 2], 0.0,
                                      repeat, 8)

    def init_weights_ub(self):
        """
        initilize weights ub to zero

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        mask = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='mask', init_value=0)
        mask.set_as(Constant.MASK_32BIT)
        dup_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='dup_num', init_value=0)
        dup_num.set_as(Constant.DUM_NUM)
        repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat', init_value=0)
        repeat.set_as(self.per_core_calc_max_values_num // Constant.MASK_32BIT // Constant.DUM_NUM)
        with self.tik_instance.for_range(0, dup_num) as dup_loop:
            self.tik_instance.vec_dup(mask, self.output_ub[dup_loop * self.per_core_calc_max_values_num // 2], 0.0,
                                      repeat, 8)

    def init_output_ub(self):
        """
        initilize output ub to zero

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        mask = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='mask', init_value=0)
        mask.set_as(Constant.MASK_32BIT)
        dup_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='dup_num', init_value=0)
        dup_num.set_as(Constant.DUM_NUM)
        repeat = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='repeat', init_value=0)
        repeat.set_as(self.per_core_calc_max_output_num // Constant.MASK_32BIT // Constant.DUM_NUM)
        with self.tik_instance.for_range(0, dup_num) as dup_loop:
            self.tik_instance.vec_dup(mask, self.output_ub[dup_loop * self.per_core_calc_max_output_num // 2], 0.0,
                                      repeat, 8)

    def one_dimension_small_output(self, core_idx):
        """
        small output size when one dimension

        Parameters
        ----------
        core_idx : core number

        Returns
        -------
        None
        """
        self.init_output_ub()
        per_core_loop = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='per_core_loop', init_value=0)
        tail_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_num', init_value=0)
        input_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='input_offset', init_value=0)
        tail_burst_len = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_burst_len', init_value=0)

        with self.tik_instance.if_scope(core_idx < self.used_core_num - 1):
            per_core_loop.set_as(self.per_core_calc_num // self.per_core_calc_max_values_num)
            tail_num.set_as(self.per_core_calc_num % self.per_core_calc_max_values_num)
            tail_burst_len.set_as((tail_num + self.value_num_each_block - 1) // self.value_num_each_block)
        with self.tik_instance.else_scope():
            per_core_loop.set_as(self.last_core_calc_num // self.per_core_calc_max_values_num)
            tail_num.set_as(self.last_core_calc_num % self.per_core_calc_max_values_num)
            tail_burst_len.set_as((tail_num + self.value_num_each_block - 1) // self.value_num_each_block)

        with self.tik_instance.for_range(0, per_core_loop) as index:
            input_offset.set_as(core_idx * self.per_core_calc_num + index * self.per_core_calc_max_values_num)

            self.compute_one_dimension_small_output(self.per_core_calc_max_values_num, input_offset,
                                                    self.ub_max_block_len, self.output_tail_burst_len)
        with self.tik_instance.if_scope(tail_num != 0):
            input_offset.set_as(core_idx * self.per_core_calc_num + per_core_loop * self.per_core_calc_max_values_num)
            self.compute_one_dimension_small_output(tail_num, input_offset, tail_burst_len, self.output_tail_burst_len)

    def compute_one_dimension_large_output(self, loop_value_num, input_offset, values_burst_len, output_burst_len):
        """
        compute one dimension when output size is large

        Parameters
        ----------
        loop_value_num : value need loop num
        input_offset : input offset
        values_burst_len : values data_move burst
        output_burst_len : output data_move burst

        Returns
        -------
        None
        """
        self.init_values_ub()
        self.init_weights_ub()
        self.data_move_gm2ub(input_offset, values_burst_len)
        with self.tik_instance.for_range(0, self.output_move_num) as loop_id:
            min_size_value = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='min_size_value', init_value=0)
            max_size_value = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='max_size_value', init_value=0)
            min_size_value.set_as(loop_id * self.per_core_calc_max_output_num)
            max_size_value.set_as((loop_id + 1) * self.per_core_calc_max_output_num)
            with self.tik_instance.if_scope(tik.all(min_size_value < self.size_data, max_size_value > self.size_data)):
                max_size_value.set_as(self.size_data)
            with self.tik_instance.for_range(0, loop_value_num) as ub_index:
                value = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='value', init_value=0)
                value.set_as(self.values_ub[ub_index])
                weights_data = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='weights_data', init_value=0)
                with self.tik_instance.if_scope(tik.any(self.total_weights_num == 0,
                                                        self.binary_output == Constant.BINARY_OUTPUT_IS_TRUE)):
                    weights_data.set_as(1.0)
                with self.tik_instance.if_scope(tik.all(self.total_weights_num != 0)):
                    weights_data.set_as(self.weights_ub[ub_index])
                with self.tik_instance.if_scope(tik.all(value < max_size_value, value >= min_size_value)):
                    value_ub_index = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='value_ub_index')
                    value_ub_index.set_as(value - min_size_value)
                    output_data = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='output_data')
                    output_data.set_as(self.output_ub[value_ub_index])
                    result = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='result', init_value=0)
                    result.set_as(weights_data + output_data)
                    self.output_ub[value_ub_index].set_as(result)
            with self.tik_instance.if_scope(loop_id < self.output_move_num - 1):
                self.tik_instance.data_move(self.output_gm[loop_id * self.per_core_calc_max_output_num],
                                            self.output_ub, 0, 1, self.ub_max_block_len, 0, 0)
            with self.tik_instance.if_scope(loop_id == self.output_move_num - 1):
                self.tik_instance.data_move(self.output_gm[loop_id * self.per_core_calc_max_output_num],
                                            self.output_ub, 0, 1, output_burst_len, 0, 0)
            self.init_output_ub()

    def one_dimension_large_output(self, core_idx):
        """
        large output size when one dimension

        Parameters
        ----------
        core_idx : core number

        Returns
        -------
        None
        """
        self.init_output_ub()
        per_core_loop = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='per_core_loop', init_value=0)
        tail_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_num', init_value=0)
        input_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='input_offset', init_value=0)
        tail_burst_len = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_burst_len', init_value=0)

        with self.tik_instance.if_scope(core_idx < self.used_core_num - 1):
            per_core_loop.set_as(self.per_core_calc_num // self.per_core_calc_max_values_num)
            tail_num.set_as(self.per_core_calc_num % self.per_core_calc_max_values_num)
            tail_burst_len.set_as((tail_num + self.value_num_each_block - 1) // self.value_num_each_block)
        with self.tik_instance.else_scope():
            per_core_loop.set_as(self.last_core_calc_num // self.per_core_calc_max_values_num)
            tail_num.set_as(self.last_core_calc_num % self.per_core_calc_max_values_num)
            tail_burst_len.set_as((tail_num + self.value_num_each_block - 1) // self.value_num_each_block)

        with self.tik_instance.for_range(0, per_core_loop) as index:
            input_offset.set_as(core_idx * self.per_core_calc_num + index * self.per_core_calc_max_values_num)
            self.compute_one_dimension_large_output(self.per_core_calc_max_values_num, input_offset,
                                                    self.ub_max_block_len, self.output_tail_burst_len)
        with self.tik_instance.if_scope(tail_num != 0):
            input_offset.set_as(core_idx * self.per_core_calc_num + per_core_loop * self.per_core_calc_max_values_num)
            self.compute_one_dimension_large_output(tail_num, input_offset, tail_burst_len, self.output_tail_burst_len)

    def one_dimension(self, core_idx):
        """
        output is one dimension

        Parameters
        ----------
        core_idx : core number

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.tiling_mode == Constant.ONE_DIMENSION_SMALL_OUTPUT):
            self.one_dimension_small_output(core_idx)
        with self.tik_instance.if_scope(self.tiling_mode == Constant.ONE_DIMENSION_LARGE_OUTPUT):
            self.one_dimension_large_output(core_idx)

    def calc_indices_offset(self, input_offset, ub_index):
        """
        calc indices offset

        Parameters
        ----------
        input_offset : input offset
        ub_index : value ub index

        Returns
        -------
        offset : indice offset
        """
        indices_burst_len = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='indices_burst_len', init_value=0)
        indices_burst_len.set_as(Constant.INDICES_TENSOR_NUM * Constant.BYTE_INT32 // Constant.BLOCK_BYTE_SIZE)
        self.indices_ub[0].set_as(0)
        offset = self.tik_instance.Scalar(self.dtype_indices, name='offset', init_value=0)
        offset.set_as(self.indices_row * (ub_index + input_offset))
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[offset], 0, 1, indices_burst_len, 0, 0)

    def compute_multi_dimension_small_output(self, loop_value_num, input_offset, values_burst_len, output_burst_len):
        """
        compute multi dimension when output size is small

        Parameters
        ----------
        loop_value_num : value need loop num
        input_offset : input offset
        values_burst_len : values data_move burst
        output_burst_len : output data_move burst

        Returns
        -------
        None
        """
        self.init_values_ub()
        self.init_weights_ub()
        self.data_move_gm2ub(input_offset, values_burst_len)
        max_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='max_offset')
        max_offset.set_as(self.dense_shape_data * self.size_data)
        with self.tik_instance.for_range(0, loop_value_num) as ub_index:
            with self.tik_instance.if_scope(input_offset + ub_index < self.indices_line):
                self.calc_indices_offset(input_offset, ub_index)
                small_batch = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='small_batch', init_value=0)
                small_batch.set_as(self.indices_ub[0])
                small_bin = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='small_bin')
                small_bin.set_as(self.values_ub[ub_index])
                gm_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='gm_offset')
                gm_offset.set_as(small_batch * self.size_data + small_bin)
                with self.tik_instance.if_scope(tik.all(gm_offset >= 0, gm_offset < max_offset)):
                    with self.tik_instance.if_scope(small_bin < self.size_data):
                        weights_data = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='weights_data', init_value=0)
                        with self.tik_instance.if_scope(tik.any(self.total_weights_num == 0,
                                                                self.binary_output == Constant.BINARY_OUTPUT_IS_TRUE)):
                            weights_data.set_as(1.0)
                        with self.tik_instance.if_scope(tik.all(self.total_weights_num != 0)):
                            weights_data.set_as(self.weights_ub[ub_index])
                        output_data = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='output_data', init_value=0)
                        output_data.set_as(self.output_ub[gm_offset])
                        result = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='result', init_value=0)
                        result.set_as(output_data + weights_data)
                        self.output_ub[gm_offset].set_as(result)
        self.tik_instance.data_move(self.output_gm, self.output_ub, 0, 1, output_burst_len, 0, 0)
        self.init_output_ub()

    def compute_multi_dimension_large_output(self, loop_value_num, input_offset, values_burst_len, output_burst_len):
        """
        compute multi dimension when output size is large

        Parameters
        ----------
        loop_value_num : value need loop num
        input_offset : input offset
        values_burst_len : values data_move burst
        output_burst_len : output data_move burst

        Returns
        -------
        None
        """
        self.init_values_ub()
        self.init_weights_ub()
        self.data_move_gm2ub(input_offset, values_burst_len)
        with self.tik_instance.for_range(0, self.output_move_num) as loop_id:
            min_gm_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='min_gm_offset')
            min_gm_offset.set_as(loop_id * self.per_core_calc_max_output_num)
            max_gm_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='max_gm_offset')
            max_gm_offset.set_as((loop_id + 1) * self.per_core_calc_max_output_num)
            with self.tik_instance.for_range(0, loop_value_num) as ub_index:
                with self.tik_instance.if_scope(input_offset + ub_index < self.indices_line):
                    self.calc_indices_offset(input_offset, ub_index)
                    large_batch = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='large_batch', init_value=0)
                    large_batch.set_as(self.indices_ub[0])
                    large_bin = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='large_bin')
                    large_bin.set_as(self.values_ub[ub_index])
                    with self.tik_instance.if_scope(large_bin < self.size_data):
                        weights_data = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='weights_data', init_value=0)
                        with self.tik_instance.if_scope(tik.any(self.total_weights_num == 0,
                                                                self.binary_output == Constant.BINARY_OUTPUT_IS_TRUE)):
                            weights_data.set_as(1.0)
                        with self.tik_instance.if_scope(tik.all(self.total_weights_num != 0)):
                            weights_data.set_as(self.weights_ub[ub_index])
                        gm_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='gm_offset', init_value=0)
                        gm_offset.set_as(large_batch * self.size_data + large_bin)
                        with self.tik_instance.if_scope(tik.all(gm_offset < max_gm_offset, gm_offset >= min_gm_offset)):
                            gm_offset.set_as(gm_offset - loop_id * self.per_core_calc_max_output_num)
                            output_data = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='output_data',
                                                                   init_value=0)
                            output_data.set_as(self.output_ub[gm_offset])
                            result = self.tik_instance.Scalar(Constant.DTYPE_FP32, name='result', init_value=0)
                            result.set_as(output_data + weights_data)
                            self.output_ub[gm_offset].set_as(result)
            with self.tik_instance.if_scope(loop_id < self.output_move_num - 1):
                self.tik_instance.data_move(self.output_gm[loop_id * self.per_core_calc_max_output_num],
                                            self.output_ub, 0, 1, self.ub_max_block_len, 0, 0)
            with self.tik_instance.if_scope(loop_id == self.output_move_num - 1):
                self.tik_instance.data_move(self.output_gm[loop_id * self.per_core_calc_max_output_num],
                                            self.output_ub, 0, 1, output_burst_len, 0, 0)
            self.init_output_ub()

    def multi_dimension_small_output(self, core_idx):
        """
        small output size when mluti dimension

        Parameters
        ----------
        core_idx : core number

        Returns
        -------
        None
        """
        self.init_output_ub()
        per_core_loop = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='per_core_loop', init_value=0)
        tail_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_num', init_value=0)
        input_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='input_offset', init_value=0)
        tail_burst_len = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_burst_len', init_value=0)

        with self.tik_instance.if_scope(core_idx < self.used_core_num - 1):
            per_core_loop.set_as(self.per_core_calc_num // self.per_core_calc_max_values_num)
            tail_num.set_as(self.per_core_calc_num % self.per_core_calc_max_values_num)
            tail_burst_len.set_as((tail_num + self.value_num_each_block - 1) // self.value_num_each_block)
        with self.tik_instance.else_scope():
            per_core_loop.set_as(self.last_core_calc_num // self.per_core_calc_max_values_num)
            tail_num.set_as(self.last_core_calc_num % self.per_core_calc_max_values_num)
            tail_burst_len.set_as((tail_num + self.value_num_each_block - 1) // self.value_num_each_block)

        with self.tik_instance.for_range(0, per_core_loop) as index:
            input_offset.set_as(core_idx * self.per_core_calc_num + index * self.per_core_calc_max_values_num)
            self.compute_multi_dimension_small_output(self.per_core_calc_max_values_num, input_offset,
                                                      self.ub_max_block_len,
                                                      self.output_tail_burst_len)
        with self.tik_instance.if_scope(tail_num != 0):
            input_offset.set_as(core_idx * self.per_core_calc_num + per_core_loop * self.per_core_calc_max_values_num)
            self.compute_multi_dimension_small_output(tail_num, input_offset, tail_burst_len,
                                                      self.output_tail_burst_len)

    def multi_dimension_large_output(self, core_idx):
        """
        small output size when mluti dimension

        Parameters
        ----------
        core_idx : core number

        Returns
        -------
        None
        """
        self.init_output_ub()
        per_core_loop = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='per_core_loop', init_value=0)
        tail_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_num', init_value=0)
        input_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='input_offset', init_value=0)
        tail_burst_len = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='tail_burst_len', init_value=0)

        with self.tik_instance.if_scope(core_idx < self.used_core_num - 1):
            per_core_loop.set_as(self.per_core_calc_num // self.per_core_calc_max_values_num)
            tail_num.set_as(self.per_core_calc_num % self.per_core_calc_max_values_num)
            tail_burst_len.set_as((tail_num + self.value_num_each_block - 1) // self.value_num_each_block)
        with self.tik_instance.else_scope():
            per_core_loop.set_as(self.last_core_calc_num // self.per_core_calc_max_values_num)
            tail_num.set_as(self.last_core_calc_num % self.per_core_calc_max_values_num)
            tail_burst_len.set_as((tail_num + self.value_num_each_block - 1) // self.value_num_each_block)

        with self.tik_instance.for_range(0, per_core_loop) as index:
            input_offset.set_as(core_idx * self.per_core_calc_num + index * self.per_core_calc_max_values_num)
            self.compute_multi_dimension_large_output(self.per_core_calc_max_values_num, input_offset,
                                                      self.ub_max_block_len,
                                                      self.ub_max_block_len)
        with self.tik_instance.if_scope(tail_num != 0):
            input_offset.set_as(core_idx * self.per_core_calc_num + per_core_loop * self.per_core_calc_max_values_num)
            self.compute_multi_dimension_large_output(tail_num, input_offset, tail_burst_len,
                                                      self.output_tail_burst_len)

    def multi_dimension(self, core_idx):
        """
        output is multi dimension

        Parameters
        ----------
        core_idx : core number

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.tiling_mode == Constant.MULTI_DIMENSIONS_SMALL_OUTPUT):
            self.multi_dimension_small_output(core_idx)
        with self.tik_instance.if_scope(self.tiling_mode == Constant.MULTI_DIMENSIONS_LARGE_OUTPUT):
            self.multi_dimension_large_output(core_idx)

    def sparse_bincount_compute(self):
        """
        Main process of SparseBincount

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.enable_atomic_add()
        with self.tik_instance.for_range(0, self.new_core_num, block_num=self.new_core_num) as core_id:
            with self.tik_instance.if_scope(core_id < self.used_core_num):
                with self.tik_instance.if_scope(self.dense_shape_data == 0):
                    self.one_dimension(core_id)
                with self.tik_instance.if_scope(self.dense_shape_data != 0):
                    self.multi_dimension(core_id)
        self.disable_atomic_add()
        tbe_context.get_context().add_compile_info('vars',
                                                   {'core_num': self.total_core_number,
                                                    'ub_size': self.ub_size_bytes
                                                    })
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.indices_gm, self.values_gm, self.dense_shape_gm,
                    self.size_gm, self.weights_gm],
            outputs=[self.output_gm],
            flowtable=[self.tiling_gm])
        return self.tik_instance


def _check_param(indices, values, dense_shape, size, weights, output, binary_output, kernel_name):
    """
    algorithm:SparseBincount
    Counts the number of occurrences of each value in an integer array.

    Parameters
    ----------
    indices : dict
        dict with keys(shape and dtype) of 2D tensor
    values: dict
        dict with keys(shape and dtype) of 1D tensor
    dense_shape : dict
        dict with keys(shape and dtype) of 1D tensor
    size: dict
        dict with the scalar
    weights: dict
        dict with keys(shape and dtype) of 1D tensor
    output: dict
        dict with keys(shape and dtype) of 1D tensor
    binary_output : bool
        Whether the kernel should count the appearance or number of occurrences
    kernel_name : str
        kernel name, default value is "SparseBincount"

    Returns
    -------
    None
    """

    indices_dtype = indices.get("dtype").lower()
    para_check.check_dtype(indices_dtype, ("int64"), param_name="indices")

    values_dtype = values.get("dtype").lower()
    para_check.check_dtype(values_dtype, ("int32", "int64"), param_name="values")

    dense_shape_dtype = dense_shape.get("dtype").lower()
    para_check.check_dtype(dense_shape_dtype, ("int64"), param_name="dense_shape")

    size_dtype = size.get("dtype").lower()
    para_check.check_dtype(size_dtype, ("int32", "int64"), param_name="size")

    if weights is None:
        weights_dtype = values_dtype
    else:
        # current surpport float32
        weights_dtype = weights.get("dtype").lower()
        para_check.check_dtype(weights_dtype, ("float32"), param_name="weights")

    output_dtype = output.get("dtype").lower()
    para_check.check_dtype(output_dtype, ("float32"), param_name="output")

    if output_dtype != weights_dtype:
        rule = "output type not match weights type"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)

    if binary_output not in (True, False):
        rule = "binary_output should be True or False,default to False"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)


@register_operator("SparseBincount")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sparse_bincount(indices, values, dense_shape,
                    size, weights, output,
                    binary_output=False,
                    kernel_name="SparseBincount"):
    """
    algorithm:SparseBincount
    Counts the number of occurrences of each value in an integer array.

    Parameters
    ----------
    indices : dict
        dict with keys(shape and dtype) of 2D tensor
    values: dict
        dict with keys(shape and dtype) of 1D tensor
    dense_shape : dict
        dict with keys(shape and dtype) of 1D tensor
    size: dict
        dict with the scalar
    weights: dict
        dict with keys(shape and dtype) of 1D tensor
    output: dict
        dict with keys(shape and dtype) of 1D tensor
    binary_output : bool
        Whether the kernel should count the appearance or number of occurrences
    kernel_name : str
        kernel name, default value is "SparseBincount"

    Returns
    -------
    None
    """

    _check_param(indices, values, dense_shape, size, weights, output, binary_output, kernel_name)
    sparse_bin_count_instance = SparseBinCount(indices, values, dense_shape, size,
                                               weights, output, binary_output, kernel_name)

    tik_instance = sparse_bin_count_instance.sparse_bincount_compute()
    return tik_instance
