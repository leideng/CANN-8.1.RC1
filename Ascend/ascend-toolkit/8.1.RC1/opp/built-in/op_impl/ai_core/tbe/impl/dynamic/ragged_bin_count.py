"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ragged_bin_count
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_tik_comm_func import ceil_div


class Constant:
    """
    The class for constant
    """
    MAX_INT64_VALUE = 2 ** 63 - 1
    BLOCK_SIZE = 32
    DTYPE_INT32 = "int32"
    DTYPE_INT64 = "int64"
    DTYPE_FP32 = "float32"
    BTYE_32BIT = 4
    BTYE_64BIT = 8
    REPEAT_MASK_32BIT = 64
    REPEAT_MASK_64BIT = 32
    ONE_BLOCK_NUM_32BIT = 8
    ONE_BLOCK_NUM_64BIT = 4
    EACH_REPEAT_TIMES_32BIT = 1
    EACH_REPEAT_TIMES_64BIT = 2

    RESERVED_UB = 1024 * 8
    TILING_PARAMS_NUM = 32
    TILING_PARAMS_DTYPE = DTYPE_INT32

    BATCH_IDX_CHANGED = 999
    BATCH_IDX_NOT_CHANGE = -999

    WEIGHT_VALUE_IN_COMPUTE_IS_WEIGHT = False
    WEIGHT_VALUE_IN_COMPUTE_IS_1 = True


class RaggedBinCount:
    """
    The class for ragged bin count
    """

    def __init__(self, splits, values, size, weights, output, binary_output=False, kernel_name="ragged_bin_count"):
        """
        constructor of class RaggedBinCount

        Parameters
        ----------
        splits: splits input
        values: values input
        size: size input
        weights: weights input
        output: output
        binary_output: alternative parameter to control weight_value, default to "False"
        kernel_name: default to "ragged_bin_count"

        Returns
        -------
        None
        """
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(
            tbe_platform.UB_SIZE) - Constant.RESERVED_UB

        class CalculationScalar:
            """
            The class for calculation scalar
            """

            def __init__(self, tik_inst):
                """
                constructor of class CalculationScalar

                Parameters
                ----------
                tik_inst: self.tik_instance

                Returns
                -------
                None
                """
                self.batch_idx = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                 name="batch_idx", init_value=0)
                self.ub_calc_output_offset = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                             name="ub_calc_output_offset", init_value=0)

        class GmTensor:
            """
            The class for gm tensor
            """

            def __init__(self, tik_inst, splits_dtype, values_dtype, weights_dtype, output_dtype):
                """
                constructor of class GmTensor

                Parameters
                ----------
                tik_inst: self.tik_instance
                splits_dtype: splits dtype
                values_dtype: values dtype
                weights_dtype: weights dtype
                output_dtype: output dtype

                Returns
                -------
                None
                """
                self.tiling_gm = tik_inst.Tensor(Constant.TILING_PARAMS_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                 name="tiling_gm", scope=tik.scope_gm)
                self.splits_gm = tik_inst.Tensor(splits_dtype, (Constant.MAX_INT64_VALUE,),
                                                 name="splits_gm", scope=tik.scope_gm)
                self.values_gm = tik_inst.Tensor(values_dtype, (Constant.MAX_INT64_VALUE,),
                                                 name="values_gm", scope=tik.scope_gm)
                self.size_gm = tik_inst.Tensor(values_dtype, (Constant.MAX_INT64_VALUE,),
                                               name="size_gm", scope=tik.scope_gm)
                self.weights_gm = tik_inst.Tensor(weights_dtype, (Constant.MAX_INT64_VALUE,),
                                                  name="weights_gm", scope=tik.scope_gm)
                self.output_gm = tik_inst.Tensor(output_dtype, (Constant.MAX_INT64_VALUE,),
                                                 name="output_gm", scope=tik.scope_gm, is_atomic_add=True)

        class UbTensor:
            """
            The class for ub tensor
            """

            def __init__(self):
                """
                constructor of class UbTensor

                Parameters
                ----------
                None

                Returns
                -------
                None
                """
                self.tiling_ub = None
                self.ub_calc_output_ub = None
                self.ub_calc_values_ub = None
                self.ub_calc_weights_ub = None

        class TilingScalar:
            """
            The class for tiling scalar
            """

            def __init__(self, tik_inst):
                """
                constructor of class TilingScalar

                Parameters
                ----------
                tik_inst: self.tik_instance

                Returns
                -------
                None
                """
                self.need_core_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                     name="need_core_num")
                self.size_data = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                 name="size_data")
                self.splits_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                  name="splits_num")
                self.weights_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                   name="weights_num")
                self.output_total_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                        name="output_total_num")
                self.values_num_each_core = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                            name="values_num_each_core")
                self.values_num_tail_core = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                            name="values_num_tail_core")
                self.each_ub_block_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                         name="each_ub_block_num")
                self.max_ub_calc_values_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                              name="max_ub_calc_values_num")
                self.new_core_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                              name="new_core_num")

        class DataSizeScalar:
            """
            The class for data size scalar
            """

            def __init__(self, tik_inst):
                """
                constructor of class CalculationScalar

                Parameters
                ----------
                tik_inst: self.tik_instance

                Returns
                -------
                None
                """
                self.splits_each_size = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                        name="splits_each_size", init_value=0)
                self.values_each_size = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                        name="values_each_size", init_value=0)
                self.weights_each_size = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                         name="weights_each_size", init_value=0)
                self.output_each_size = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                        name="output_each_size", init_value=0)

        self.splits_dtype = splits.get("dtype").lower()
        self.values_dtype = values.get("dtype").lower()
        self.weights_dtype = weights.get("dtype").lower()
        self.output_dtype = output.get("dtype").lower()

        self.obj_calc_scalar = CalculationScalar(self.tik_instance)
        self.obj_gm_tensor = GmTensor(self.tik_instance, self.splits_dtype, self.values_dtype,
                                      self.weights_dtype, self.output_dtype)
        self.obj_ub_tensor = UbTensor()
        self.obj_tiling_scalar = TilingScalar(self.tik_instance)
        self.obj_data_size_scalar = DataSizeScalar(self.tik_instance)
        self.binary_output = binary_output

        if self.splits_dtype in (Constant.DTYPE_INT64,):
            self.obj_data_size_scalar.splits_each_size.set_as(
                Constant.BTYE_64BIT)

        if self.values_dtype in (Constant.DTYPE_INT32,):
            self.obj_data_size_scalar.values_each_size.set_as(
                Constant.BTYE_32BIT)
        elif self.values_dtype in (Constant.DTYPE_INT64,):
            self.obj_data_size_scalar.values_each_size.set_as(
                Constant.BTYE_64BIT)

        if self.weights_dtype in (Constant.DTYPE_FP32,):
            self.obj_data_size_scalar.weights_each_size.set_as(
                Constant.BTYE_32BIT)

        if self.output_dtype in (Constant.DTYPE_FP32,):
            self.obj_data_size_scalar.output_each_size.set_as(
                Constant.BTYE_32BIT)

        # used to store actually calculated numbers of values in each core
        self.actual_values_num_each_core = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                                    name="actual_values_num_each_core")

    def each_core_compute(self):
        """
        each core compute

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.obj_tiling_scalar.new_core_num,
                                         block_num=self.obj_tiling_scalar.new_core_num) as _core_idx:
            values_begin = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                    name="values_begin", init_value=0)
            with self.tik_instance.if_scope(_core_idx < self.obj_tiling_scalar.need_core_num - 1):
                self.actual_values_num_each_core.set_as(
                    self.obj_tiling_scalar.values_num_each_core)
                values_begin.set_as(
                    self.actual_values_num_each_core * _core_idx)
            with self.tik_instance.if_scope(_core_idx == self.obj_tiling_scalar.need_core_num - 1):
                self.actual_values_num_each_core.set_as(
                    self.obj_tiling_scalar.values_num_tail_core)
                values_begin.set_as(self.obj_tiling_scalar.values_num_each_core *
                                    (self.obj_tiling_scalar.need_core_num - 1))

            each_loop_calc_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                             name="each_core_calc_offset", init_value=0)
            ub_calc_values_num = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                          name="ub_calc_values_num", init_value=0)

            # compute times according to the ub can handle values numbers
            max_ub_calc_loop = self.actual_values_num_each_core // self.obj_tiling_scalar.max_ub_calc_values_num
            with self.tik_instance.for_range(0, max_ub_calc_loop) as _loop_idx:
                each_loop_calc_offset.set_as(
                    _loop_idx * self.obj_tiling_scalar.max_ub_calc_values_num)
                ub_calc_values_num.set_as(
                    self.obj_tiling_scalar.max_ub_calc_values_num)
                self._run_compute_wth_diff_case(
                    ub_calc_values_num, each_loop_calc_offset, values_begin)

            # calculate the last numbers of values
            last_values_num = self.actual_values_num_each_core % self.obj_tiling_scalar.max_ub_calc_values_num
            with self.tik_instance.if_scope(last_values_num > 0):
                each_loop_calc_offset.set_as(
                    max_ub_calc_loop * self.obj_tiling_scalar.max_ub_calc_values_num)
                ub_calc_values_num.set_as(last_values_num)
                self._run_compute_wth_diff_case(
                    ub_calc_values_num, each_loop_calc_offset, values_begin)

    def ragged_bin_count_compute(self):
        """
        The tik implementation of operator RaggedBinCount
        """
        self._init_tiling_params()
        _enable_atomic_add(self.tik_instance)
        self.each_core_compute()
        _disable_atomic_add(self.tik_instance)

        tbe_context.get_context().add_compile_info("vars",
                                                   {"ub_size": self.ub_size,
                                                    "core_num": self.ai_core_num})
        opt_config = {"enable_const_fold": True,
                      "out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.obj_gm_tensor.splits_gm,
                                           self.obj_gm_tensor.values_gm,
                                           self.obj_gm_tensor.size_gm,
                                           self.obj_gm_tensor.weights_gm,),
                                   outputs=(self.obj_gm_tensor.output_gm,),
                                   flowtable=(self.obj_gm_tensor.tiling_gm,),
                                   config=opt_config)

    def _get_tiling_params(self):
        """
        get tiling parameters from tiling ub

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.obj_tiling_scalar.need_core_num.set_as(
            self.obj_ub_tensor.tiling_ub[0])
        self.obj_tiling_scalar.size_data.set_as(
            self.obj_ub_tensor.tiling_ub[1])
        self.obj_tiling_scalar.splits_num.set_as(
            self.obj_ub_tensor.tiling_ub[2])
        self.obj_tiling_scalar.weights_num.set_as(
            self.obj_ub_tensor.tiling_ub[3])
        self.obj_tiling_scalar.output_total_num.set_as(
            self.obj_ub_tensor.tiling_ub[4])
        self.obj_tiling_scalar.values_num_each_core.set_as(
            self.obj_ub_tensor.tiling_ub[5])
        self.obj_tiling_scalar.values_num_tail_core.set_as(
            self.obj_ub_tensor.tiling_ub[6])
        self.obj_tiling_scalar.each_ub_block_num.set_as(
            self.obj_ub_tensor.tiling_ub[7])
        self.obj_tiling_scalar.max_ub_calc_values_num.set_as(
            self.obj_ub_tensor.tiling_ub[8])
        self.obj_tiling_scalar.new_core_num.set_as(
            self.obj_ub_tensor.tiling_ub[9])

    def _init_tiling_params(self):
        """
        initialize tiling ub for this scope only

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.new_stmt_scope():
            self.obj_ub_tensor.tiling_ub = self.tik_instance.Tensor(Constant.TILING_PARAMS_DTYPE,
                                                                    (Constant.TILING_PARAMS_NUM,),
                                                                    name="tiling_ub",
                                                                    scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.obj_ub_tensor.tiling_ub,
                                        self.obj_gm_tensor.tiling_gm,
                                        0, 1,
                                        Constant.TILING_PARAMS_NUM * Constant.BTYE_32BIT // Constant.BLOCK_SIZE,
                                        0, 0)
            self._get_tiling_params()

    def _run_process_output_le_max_ub_wth_1(self, ub_calc_values_begin, ub_calc_values_end,
                                            ub_calc_values_offset, output_ub_block_num,
                                            get_output_result_func):
        """
        compute output result when output total number less than or
        equal to max ub size can calculate, and with weight_value = 1

        Parameters
        ----------
        ub_calc_values_begin: 0
        ub_calc_values_end: ub_calc_values_num
        ub_calc_values_offset: values_begin + each_loop_calc_offset
        output_ub_block_num: output_total_ub_block_num
        get_output_result_func: _get_output_result
                                    or
                                _get_output_result_binout

        Returns
        -------
        None
        """
        values_data = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                               name="values_data", init_value=0)
        weights_value = self.tik_instance.Scalar(Constant.DTYPE_FP32,
                                                 name="weights_value", init_value=1)
        result_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                 name="result_offset", init_value=0)

        with self.tik_instance.for_range(ub_calc_values_begin, ub_calc_values_end) as _values_idx:
            with self.tik_instance.if_scope(self.obj_calc_scalar.batch_idx < self.obj_tiling_scalar.splits_num - 1):
                batch_result = _get_batch_idx(self.tik_instance, self.obj_calc_scalar, self.obj_data_size_scalar,
                                              self.obj_tiling_scalar, _values_idx + ub_calc_values_offset,
                                              self.obj_gm_tensor.splits_gm)
                self.obj_calc_scalar.batch_idx.set_as(batch_result)

            values_data.set_as(
                self.obj_ub_tensor.ub_calc_values_ub[_values_idx])
            with self.tik_instance.if_scope(values_data < self.obj_tiling_scalar.size_data):
                result_offset.set_as((self.obj_calc_scalar.batch_idx - 1) *
                                     self.obj_tiling_scalar.size_data + values_data)
                get_output_result_func(
                    self.tik_instance, self.obj_ub_tensor, result_offset, weights_value)

        _handle_output_result(self.tik_instance, self.obj_ub_tensor.ub_calc_output_ub,
                              self.obj_gm_tensor.output_gm, self.obj_calc_scalar.ub_calc_output_offset,
                              output_ub_block_num)

    def _run_process_output_gt_max_ub_wth_1(self, ub_calc_values_begin, ub_calc_values_end, ub_calc_values_offset,
                                            output_ub_block_num, output_extended, output_init_mask,
                                            output_init_repeat_times, get_output_result_func):
        """
        compute output result when output total number greater than
        max ub size can calculate, and with weight_value = 1

        Parameters
        ----------
        ub_calc_values_begin: 0
        ub_calc_values_end: ub_calc_values_num
        ub_calc_values_offset: values_begin + each_loop_calc_offset
        output_ub_block_num: self.obj_tiling_scalar.each_ub_block_num
        output_extended: output_ub_block_num * Constant.BLOCK_SIZE // self.obj_data_size_scalar.output_each_size
        output_init_mask: Constant.REPEAT_MASK_32BIT
                            or
                          output_extended
        output_init_repeat_tims: output_extended // Constant.REPEAT_MASK_32BIT or 1
        get_output_result_func: _get_output_result
                                    or
                                _get_output_result_binout

        Returns
        -------
        None
        """
        values_data = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                               name="values_data", init_value=0)
        weights_value = self.tik_instance.Scalar(Constant.DTYPE_FP32,
                                                 name="weights_value", init_value=1)
        result_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                 name="result_offset", init_value=0)
        output_ub_origin_idx = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                        name="output_ub_origin_idx", init_value=0)
        output_ub_current_idx = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                         name="output_ub_current_idx", init_value=0)

        with self.tik_instance.for_range(ub_calc_values_begin, ub_calc_values_end) as _values_idx:
            with self.tik_instance.if_scope(self.obj_calc_scalar.batch_idx < self.obj_tiling_scalar.splits_num - 1):
                batch_result = _get_batch_idx(self.tik_instance, self.obj_calc_scalar, self.obj_data_size_scalar,
                                              self.obj_tiling_scalar, _values_idx + ub_calc_values_offset,
                                              self.obj_gm_tensor.splits_gm)
                self.obj_calc_scalar.batch_idx.set_as(batch_result)

            values_data.set_as(
                self.obj_ub_tensor.ub_calc_values_ub[_values_idx])

            self.obj_calc_scalar.ub_calc_output_offset.set_as(
                output_ub_origin_idx * output_extended)
            output_ub_current_idx.set_as(((self.obj_calc_scalar.batch_idx - 1) *
                                          self.obj_tiling_scalar.size_data + values_data) // output_extended)
            with self.tik_instance.if_scope(output_ub_origin_idx != output_ub_current_idx):
                _handle_output_result(self.tik_instance, self.obj_ub_tensor.ub_calc_output_ub,
                                      self.obj_gm_tensor.output_gm, self.obj_calc_scalar.ub_calc_output_offset,
                                      output_ub_block_num)
                _init_output_ub(self.tik_instance, output_init_mask, self.obj_ub_tensor.ub_calc_output_ub,
                                output_init_repeat_times)
                output_ub_origin_idx.set_as(output_ub_current_idx)

            self.obj_calc_scalar.ub_calc_output_offset.set_as(
                output_ub_origin_idx * output_extended)
            with self.tik_instance.if_scope(values_data < self.obj_tiling_scalar.size_data):
                result_offset.set_as(((self.obj_calc_scalar.batch_idx - 1) *
                                      self.obj_tiling_scalar.size_data + values_data) % output_extended)
                get_output_result_func(
                    self.tik_instance, self.obj_ub_tensor, result_offset, weights_value)

        _handle_output_result(self.tik_instance, self.obj_ub_tensor.ub_calc_output_ub,
                              self.obj_gm_tensor.output_gm, self.obj_calc_scalar.ub_calc_output_offset,
                              output_ub_block_num)

    def _run_process_output_le_max_ub_wth_wgt(self, ub_calc_values_begin, ub_calc_values_end,
                                              ub_calc_values_offset, output_ub_block_num):
        """
        compute output result when output total number less than or
        equal to max ub size can calculate, and with weight_value = weights[_values_idx]

        Parameters
        ----------
        ub_calc_values_begin: 0
        ub_calc_values_end: ub_calc_values_num
        ub_calc_values_offset: values_begin + each_loop_calc_offset
        output_ub_block_num: output_total_ub_block_num

        Returns
        -------
        None
        """
        values_data = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                               name="values_data", init_value=0)
        weights_value = self.tik_instance.Scalar(Constant.DTYPE_FP32,
                                                 name="weights_value", init_value=0)
        result_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                 name="result_offset", init_value=0)

        with self.tik_instance.for_range(ub_calc_values_begin, ub_calc_values_end) as _values_idx:
            with self.tik_instance.if_scope(self.obj_calc_scalar.batch_idx < self.obj_tiling_scalar.splits_num - 1):
                batch_result = _get_batch_idx(self.tik_instance, self.obj_calc_scalar, self.obj_data_size_scalar,
                                              self.obj_tiling_scalar, _values_idx + ub_calc_values_offset,
                                              self.obj_gm_tensor.splits_gm)
                self.obj_calc_scalar.batch_idx.set_as(batch_result)

            values_data.set_as(
                self.obj_ub_tensor.ub_calc_values_ub[_values_idx])
            with self.tik_instance.if_scope(values_data < self.obj_tiling_scalar.size_data):
                weights_value.set_as(
                    self.obj_ub_tensor.ub_calc_weights_ub[_values_idx])
                result_offset.set_as((self.obj_calc_scalar.batch_idx - 1) *
                                     self.obj_tiling_scalar.size_data + values_data)
                _get_output_result(
                    self.tik_instance, self.obj_ub_tensor, result_offset, weights_value)

        _handle_output_result(self.tik_instance, self.obj_ub_tensor.ub_calc_output_ub,
                              self.obj_gm_tensor.output_gm, self.obj_calc_scalar.ub_calc_output_offset,
                              output_ub_block_num)

    def _run_process_output_gt_max_ub_wth_wgt(self, ub_calc_values_begin, ub_calc_values_end, ub_calc_values_offset,
                                              output_ub_block_num, output_extended, output_init_mask,
                                              output_init_repeat_times):
        """
        compute output result when output total number greater than
        max ub size can calculate, and with weight_value = weights[_values_idx]

        Parameters
        ----------
        ub_calc_values_begin: 0
        ub_calc_values_end: ub_calc_values_num
        ub_calc_values_offset: values_begin + each_loop_calc_offset
        output_ub_block_num: self.obj_tiling_scalar.each_ub_block_num
        output_extended: output_ub_block_num * Constant.BLOCK_SIZE // self.obj_data_size_scalar.output_each_size
        output_init_mask: Constant.REPEAT_MASK_32BIT
                            or
                          output_extended
        output_init_repeat_tims: output_extended // Constant.REPEAT_MASK_32BIT
                            or
                            1

        Returns
        -------
        None
        """
        values_data = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                               name="values_data", init_value=0)
        weights_value = self.tik_instance.Scalar(Constant.DTYPE_FP32,
                                                 name="weights_value", init_value=0)
        result_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                 name="result_offset", init_value=0)
        output_ub_origin_idx = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                        name="output_ub_origin_idx", init_value=0)
        output_ub_current_idx = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                         name="output_ub_current_idx", init_value=0)

        with self.tik_instance.for_range(ub_calc_values_begin, ub_calc_values_end) as _values_idx:
            with self.tik_instance.if_scope(self.obj_calc_scalar.batch_idx < self.obj_tiling_scalar.splits_num - 1):
                batch_result = _get_batch_idx(self.tik_instance, self.obj_calc_scalar, self.obj_data_size_scalar,
                                              self.obj_tiling_scalar, _values_idx + ub_calc_values_offset,
                                              self.obj_gm_tensor.splits_gm)
                self.obj_calc_scalar.batch_idx.set_as(batch_result)

            values_data.set_as(
                self.obj_ub_tensor.ub_calc_values_ub[_values_idx])

            self.obj_calc_scalar.ub_calc_output_offset.set_as(
                output_ub_origin_idx * output_extended)
            output_ub_current_idx.set_as(((self.obj_calc_scalar.batch_idx - 1) *
                                          self.obj_tiling_scalar.size_data + values_data) // output_extended)
            with self.tik_instance.if_scope(output_ub_origin_idx != output_ub_current_idx):
                _handle_output_result(self.tik_instance, self.obj_ub_tensor.ub_calc_output_ub,
                                      self.obj_gm_tensor.output_gm, self.obj_calc_scalar.ub_calc_output_offset,
                                      output_ub_block_num)
                _init_output_ub(self.tik_instance, output_init_mask, self.obj_ub_tensor.ub_calc_output_ub,
                                output_init_repeat_times)
                output_ub_origin_idx.set_as(output_ub_current_idx)

            self.obj_calc_scalar.ub_calc_output_offset.set_as(
                output_ub_origin_idx * output_extended)
            with self.tik_instance.if_scope(values_data < self.obj_tiling_scalar.size_data):
                weights_value.set_as(
                    self.obj_ub_tensor.ub_calc_weights_ub[_values_idx])
                result_offset.set_as(((self.obj_calc_scalar.batch_idx - 1) *
                                      self.obj_tiling_scalar.size_data + values_data) % output_extended)
                _get_output_result(
                    self.tik_instance, self.obj_ub_tensor, result_offset, weights_value)

        _handle_output_result(self.tik_instance, self.obj_ub_tensor.ub_calc_output_ub,
                              self.obj_gm_tensor.output_gm, self.obj_calc_scalar.ub_calc_output_offset,
                              output_ub_block_num)

    def _run_compute_wth_diff_case(self, ub_calc_values_num, each_loop_calc_offset, values_begin):
        """
        prepare some data ub initialization and select run which calculation case

        Parameters
        ----------
        ub_calc_values_num: self.obj_tiling_scalar.max_ub_calc_values_num or last_values_num
        each_loop_calc_offset: _loop_idx * self.obj_tiling_scalar.max_ub_calc_values_num
                                    or
                               max_ub_calc_loop * self.obj_tiling_scalar.max_ub_calc_values_num
        values_begin: self.actual_values_num_each_core * _core_idx
                            or
                      self.obj_tiling_scalar.values_num_tail_core + _core_idx *
                      self.obj_tiling_scalar.values_num_each_core

        Returns
        -------
        None
        """
        ub_calc_values_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                         name="ub_calc_values_offset")
        ub_calc_values_offset.set_as(values_begin + each_loop_calc_offset)

        ub_calc_values_ub_block_num = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                               name="ub_calc_values_ub_block_num", init_value=0)
        ub_calc_values_extended = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                           name="ub_calc_values_extended", init_value=0)

        ub_calc_values_ub_block_num.set_as(ceil_div(ub_calc_values_num *
                                                     self.obj_data_size_scalar.values_each_size, Constant.BLOCK_SIZE))
        ub_calc_values_extended.set_as(ub_calc_values_ub_block_num * Constant.BLOCK_SIZE //
                                       self.obj_data_size_scalar.values_each_size)

        self.obj_ub_tensor.ub_calc_values_ub = self.tik_instance.Tensor(self.values_dtype,
                                                                        (ub_calc_values_extended,),
                                                                        name="ub_calc_values_ub",
                                                                        scope=tik.scope_ubuf)
        _param_data_move_gm2ub(self.tik_instance, self.obj_ub_tensor.ub_calc_values_ub,
                               self.obj_gm_tensor.values_gm, ub_calc_values_offset, ub_calc_values_ub_block_num)

        ub_calc_weights_ub_block_num = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                                name="ub_calc_weights_ub_block_num", init_value=0)
        ub_calc_weights_extended = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                            name="ub_calc_weights_extended", init_value=0)

        # default to the numbers of weights has the same as values has
        ub_calc_weights_ub_block_num.set_as(ub_calc_values_ub_block_num)
        ub_calc_weights_extended.set_as(ub_calc_values_extended)

        self.obj_ub_tensor.ub_calc_weights_ub = self.tik_instance.Tensor(self.weights_dtype,
                                                                         (ub_calc_weights_extended,),
                                                                         name="ub_calc_weights_ub",
                                                                         scope=tik.scope_ubuf)
        _param_data_move_gm2ub(self.tik_instance, self.obj_ub_tensor.ub_calc_weights_ub,
                               self.obj_gm_tensor.weights_gm, ub_calc_values_offset, ub_calc_weights_ub_block_num)

        # output_ub must be initialized out of the compute process,
        # it can be reused many times with _init_output_ub function
        output_init_repeat_times = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                            name="output_init_repeat_times", init_value=0)
        output_init_repeat_tail = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                           name="output_init_repeat_tail", init_value=0)
        output_init_mask = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                    name="output_init_mask", init_value=0)
        output_init_mask.set_as(Constant.REPEAT_MASK_32BIT)

        output_total_ub_block_num = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                             name="output_ub_block_num")
        output_total_ub_block_num.set_as(ceil_div(self.obj_tiling_scalar.output_total_num *
                                                   self.obj_data_size_scalar.output_each_size, Constant.BLOCK_SIZE))
        output_ub_block_num = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                       name="output_ub_block_num")
        with self.tik_instance.if_scope(output_total_ub_block_num > self.obj_tiling_scalar.each_ub_block_num):
            output_ub_block_num.set_as(
                self.obj_tiling_scalar.each_ub_block_num)
        with self.tik_instance.else_scope():
            output_ub_block_num.set_as(output_total_ub_block_num)
        output_extended = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                   name="output_extended")
        output_extended.set_as(output_ub_block_num * Constant.BLOCK_SIZE //
                               self.obj_data_size_scalar.output_each_size)
        output_init_repeat_times.set_as(
            ceil_div(output_extended, Constant.REPEAT_MASK_32BIT))
        output_init_repeat_tail.set_as(
            output_extended % Constant.REPEAT_MASK_32BIT)

        with self.tik_instance.if_scope(tik.all(output_init_repeat_times == 1,
                                        output_init_repeat_tail == output_extended)):
            output_init_mask.set_as(output_extended)
            output_init_repeat_times.set_as(1)

        self.obj_ub_tensor.ub_calc_output_ub = self.tik_instance.Tensor(self.output_dtype,
                                                                        (output_extended,),
                                                                        name="ub_calc_output_ub",
                                                                        scope=tik.scope_ubuf)
        _init_output_ub(self.tik_instance, output_init_mask, self.obj_ub_tensor.ub_calc_output_ub,
                        output_init_repeat_times)

        ub_calc_values_begin = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                        name="ub_calc_values_begin", init_value=0)
        ub_calc_values_end = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                      name="ub_calc_values_end", init_value=ub_calc_values_num)

        with self.tik_instance.if_scope(tik.any(self.binary_output ==
                                        Constant.WEIGHT_VALUE_IN_COMPUTE_IS_1,
                                        self.obj_tiling_scalar.weights_num == 0)):
            get_output_result_func = None
            if self.binary_output == Constant.WEIGHT_VALUE_IN_COMPUTE_IS_1:
                get_output_result_func = _get_output_result_binout
            else:
                get_output_result_func = _get_output_result

            with self.tik_instance.if_scope(output_total_ub_block_num < self.obj_tiling_scalar.each_ub_block_num):
                self._run_process_output_le_max_ub_wth_1(ub_calc_values_begin, ub_calc_values_end,
                                                         ub_calc_values_offset, output_ub_block_num,
                                                         get_output_result_func)
            with self.tik_instance.else_scope():
                self._run_process_output_gt_max_ub_wth_1(ub_calc_values_begin, ub_calc_values_end,
                                                         ub_calc_values_offset, output_ub_block_num,
                                                         output_extended, output_init_mask,
                                                         output_init_repeat_times,
                                                         get_output_result_func)
        with self.tik_instance.elif_scope(self.binary_output ==
                                          Constant.WEIGHT_VALUE_IN_COMPUTE_IS_WEIGHT):
            with self.tik_instance.if_scope(output_total_ub_block_num < self.obj_tiling_scalar.each_ub_block_num):
                self._run_process_output_le_max_ub_wth_wgt(ub_calc_values_begin, ub_calc_values_end,
                                                           ub_calc_values_offset, output_ub_block_num)
            with self.tik_instance.else_scope():
                self._run_process_output_gt_max_ub_wth_wgt(ub_calc_values_begin, ub_calc_values_end,
                                                           ub_calc_values_offset, output_ub_block_num,
                                                           output_extended, output_init_mask,
                                                           output_init_repeat_times)


def _enable_atomic_add(tik_inst):
    """
    enable atomic add

    Parameters
    ----------
    tik_inst: self.tik_instance

    Returns
    -------
    None
    """
    if tbe_platform.api_check_support("tik.set_atomic_add"):
        tik_inst.set_atomic_add(1)


def _disable_atomic_add(tik_inst):
    """
    disable atomic add

    Parameters
    ----------
    tik_inst: self.tik_instance

    Returns
    -------
    None
    """
    if tbe_platform.api_check_support("tik.set_atomic_add"):
        tik_inst.set_atomic_add(0)


def _init_output_ub(tik_inst, mask, output_ub, repeat):
    """
    initialize output ub with 0.0 to make the output ub can be used many times with once ub declaration

    Parameters
    ----------
    tik_inst: self.tik_instance
    mask: output_init_mask
    output_ub: self.obj_ub_tensor.ub_calc_output_ub
    repeat: output_init_repeat_times

    Returns
    -------
    None
    """
    tik_inst.vec_dup(mask, output_ub, 0.0, repeat, 8)


def _param_data_move_gm2ub(tik_inst, param_ub, param_gm, param_offset, param_ub_block_num):
    """
    data move from gm to ub

    Parameters
    ----------
    tik_inst: self.tik_instance
    param_ub: param_ub
    param_gm: param_gm
    param_offset: param_offset
    param_ub_block_num: param_ub_block_num

    Returns
    -------
    None
    """
    tik_inst.data_move(
        param_ub, param_gm[param_offset], 0, 1, param_ub_block_num, 0, 0)


def _run_each_splits_stride_loop(tik_inst, _values_idx, obj_calc_scalar,
                                 splits_ub, splits_calc_stride, splits_calc_max_data):
    """
    run each splits stride comparation loop

    Parameters
    ----------
    tik_inst: self.tik_instance
    _values_idx: _values_idx
    obj_calc_scalar: obj_calc_scalar
    splits_ub: splits_ub
    splits_calc_stride: splits_calc_stride or splits_calc_tail_num
    splits_calc_max_data: splits_calc_max_data

    Returns
    -------
    None
    """
    tmp_batch_idx = obj_calc_scalar.batch_idx
    tmp_splits_max_data = splits_calc_max_data
    splits_calc_data = tik_inst.Scalar(Constant.DTYPE_INT32,
                                       name="splits_calc_data")
    splits_calc_batch_idx = tik_inst.Scalar(Constant.DTYPE_INT32,
                                            name="splits_calc_batch_idx", init_value=0)

    with tik_inst.if_scope(tmp_batch_idx < splits_calc_stride):
        splits_calc_batch_idx.set_as(tmp_batch_idx)

    with tik_inst.for_range(0, splits_calc_stride):
        with tik_inst.if_scope(splits_calc_batch_idx < splits_calc_stride):
            splits_calc_data.set_as(splits_ub[splits_calc_batch_idx])

            with tik_inst.if_scope(tmp_splits_max_data < splits_calc_data):
                tmp_splits_max_data.set_as(splits_calc_data)

            with tik_inst.if_scope(_values_idx >= tmp_splits_max_data):
                tmp_batch_idx.set_as(tmp_batch_idx + 1)
                splits_calc_batch_idx.set_as(splits_calc_batch_idx + 1)

    return tmp_batch_idx


def _get_batch_idx(tik_inst, obj_calc_scalar, obj_data_size_scalar,  # 标记一下，可以优化
                   obj_tiling_scalar, _values_idx, splits_gm):
    """
    get batch idx

    Parameters
    ----------
    tik_inst: self.tik_instance
    obj_calc_scalar: self.obj_calc_scalar
    obj_data_size_scalar: self.obj_data_size_scalar
    obj_tiling_scalar: self.obj_tiling_scalar
    _values_idx: _values_idx + ub_calc_values_offset
    splits_gm: self.obj_gm_tensor.splits_gm

    Returns
    -------
    None
    """
    splits_calc_loop = tik_inst.Scalar(Constant.DTYPE_INT32,
                                       name="splits_calc_loop", init_value=0)
    splits_calc_loop.set_as(obj_tiling_scalar.splits_num //
                            ((obj_tiling_scalar.each_ub_block_num * Constant.BLOCK_SIZE) //
                             obj_data_size_scalar.splits_each_size))
    splits_calc_offset = tik_inst.Scalar(Constant.DTYPE_INT32,
                                         name="splits_calc_offset", init_value=0)
    splits_calc_stride = tik_inst.Scalar(Constant.DTYPE_INT32,
                                         name="splits_calc_stride", init_value=0)
    splits_calc_ub_block_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                               name="splits_calc_ub_block_num", init_value=0)
    # store current max splits data
    splits_calc_max_data = tik_inst.Scalar(Constant.DTYPE_INT32,
                                           name="splits_calc_max_data", init_value=0)
    splits_calc_result = None

    with tik_inst.for_range(0, splits_calc_loop) as _splits_idx:
        splits_calc_offset.set_as(_splits_idx * splits_calc_stride)
        splits_calc_ub_block_num.set_as(obj_tiling_scalar.each_ub_block_num)
        splits_calc_stride.set_as(splits_calc_ub_block_num * Constant.BLOCK_SIZE //
                                  obj_data_size_scalar.splits_each_size)
        splits_ub = tik_inst.Tensor(Constant.DTYPE_INT64, (splits_calc_stride,),
                                    name="splits_ub", scope=tik.scope_ubuf)
        _param_data_move_gm2ub(
            tik_inst, splits_ub, splits_gm, splits_calc_offset, splits_calc_ub_block_num)
        splits_calc_result = _run_each_splits_stride_loop(tik_inst, _values_idx, obj_calc_scalar,
                                                          splits_ub, splits_calc_stride, splits_calc_max_data)

    splits_calc_tail_num = tik_inst.Scalar(
        Constant.DTYPE_INT32, name="splits_calc_tail_num", init_value=0)
    splits_calc_tail_num.set_as(obj_tiling_scalar.splits_num %
                                ((obj_tiling_scalar.each_ub_block_num * Constant.BLOCK_SIZE) //
                                 obj_data_size_scalar.splits_each_size))
    with tik_inst.if_scope(splits_calc_tail_num > 0):
        splits_calc_offset.set_as(
            obj_tiling_scalar.splits_num - splits_calc_tail_num)
        splits_calc_ub_block_num.set_as(ceil_div(splits_calc_tail_num * obj_data_size_scalar.splits_each_size,
                                        Constant.BLOCK_SIZE))
        splits_calc_stride.set_as(splits_calc_ub_block_num * Constant.BLOCK_SIZE //
                                  obj_data_size_scalar.splits_each_size)
        splits_ub = tik_inst.Tensor(Constant.DTYPE_INT64, (splits_calc_stride,),
                                    name="splits_ub", scope=tik.scope_ubuf)
        _param_data_move_gm2ub(
            tik_inst, splits_ub, splits_gm, splits_calc_offset, splits_calc_ub_block_num)
        splits_calc_result = _run_each_splits_stride_loop(tik_inst, _values_idx, obj_calc_scalar,
                                                          splits_ub, splits_calc_tail_num, splits_calc_max_data)

    return splits_calc_result


def _get_output_result_binout(tik_inst, obj_ub_tensor, result_offset, weights_value):
    """
    calculate output value

    Parameters
    ----------
    tik_inst: None
    obj_ub_tensor: self.obj_ub_tensor
    result_offset: ((self.obj_calc_scalar.batch_idx - 1) *
                    self.obj_tiling_scalar.size_data + values_data) % output_extended
    weights_value: None

    Returns
    -------
    None
    """
    obj_ub_tensor.ub_calc_output_ub[result_offset].set_as(1)


def _get_output_result(tik_inst, obj_ub_tensor, result_offset, weights_value):
    """
    calculate output value

    Parameters
    ----------
    tik_inst: self.tik_instance
    obj_ub_tensor: self.obj_ub_tensor
    result_offset: ((self.obj_calc_scalar.batch_idx - 1) *
                    self.obj_tiling_scalar.size_data + values_data) % output_extended
    weights_value: self.obj_ub_tensor.ub_calc_weights_ub[_values_idx] or 1

    Returns
    -------
    None
    """
    output_data = tik_inst.Scalar(Constant.DTYPE_FP32,
                                  name="output_data")
    output_data.set_as(obj_ub_tensor.ub_calc_output_ub[result_offset])
    obj_ub_tensor.ub_calc_output_ub[result_offset].set_as(
        output_data + weights_value)


def _handle_output_result(tik_inst, output_ub, output_gm, output_offset, output_ub_block_num):
    """
    calculate output value

    Parameters
    ----------
    tik_inst: self.tik_instance
    output_ub: self.obj_ub_tensor.ub_calc_output_ub
    output_gm: self.obj_gm_tensor.output_gm
    output_offset: self.obj_calc_scalar.ub_calc_output_offset
    output_ub_block_num: output_ub_block_num

    Returns
    -------
    None
    """
    tik_inst.data_move(output_gm[output_offset],
                       output_ub, 0, 1, output_ub_block_num, 0, 0)


def check_supported(splits, values, size, weights, output, binary_output=False, kernel_name="ragged_bin_count"):
    """
    output has the same type as weights, all only can use float32, because of the set_atomic_add() only supports float32
    """
    splits_dtype = splits.get("dtype").lower()
    values_dtype = values.get("dtype").lower()
    size_dtype = size.get("dtype").lower()
    weights_dtype = weights.get("dtype").lower()
    output_dtype = output.get("dtype").lower()

    if splits_dtype not in ("int64"):
        reason = "the splits's dtype must be int64, splits_dtype=%s" % splits_dtype
        return False, reason
    if values_dtype not in ("int32", "int64"):
        reason = "the values's dtype must be int32 or int64, values_dtype=%s" % values_dtype
        return False, reason
    if size_dtype not in ("int32", "int64"):
        reason = "the size's dtype must be int32 or int64, size_dtype=%s" % size_dtype
        return False, reason
    if size_dtype != values_dtype:
        reason = "the size's dtype must be the same as values's dtype, size_dtype=%s, values_dtype=%s" % \
            (size_dtype, values_dtype)
        return False, reason
    if weights_dtype not in ("float32"):
        reason = "the weights's dtype must be float32, weights_dtype=%s" % weights_dtype
        return False, reason
    if output_dtype not in ("float32"):
        reason = "the output's dtype must be float32, output_dtype=%s" % output_dtype
        return False, reason

    return True, ""


@register_operator("RaggedBinCount")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def ragged_bin_count(splits, values, size, weights, output, binary_output=False, kernel_name="ragged_bin_count"):
    """
    algorithm: ragged_bin_count
    calculating: Counts the number of occurrences of each value in an integer array

    Parameters
    ----------
    splits: dict
        dict of input_splits, include shape with 1D and dtype with int64,

    values: dict
        dict of input_values, include shape with 2D and dtype with int32 or int64,

    size: dict
        dict of input_size, include shape and dtype,
        Must have the same dtype as input_values, and must be a non-negative int scalar Tensor,

    weights: dict
        dict of input_weights, include shape and dtype,
        which has two cases:
            case 1: Must have the same shapes as input_values with dtype int32, int64, float32, or float64,
            case 2: Can be a length-0 Tensor, in which case it acts as all weights equal to 1,

    output: dict
        dict of output,
        Must be a vector with length size and the same dtype as weights,

    binary_output: bool
        An optional bool. Defaults to False,
        Whether the kernel should count the appearance or number of occurrences,

    kernel_name : str
        cce kernel name, default value is "ragged_bin_count"

    Returns
    -------
    None
    """

    obj = RaggedBinCount(splits, values, size, weights,
                         output, binary_output, kernel_name)
    obj.ragged_bin_count_compute()
