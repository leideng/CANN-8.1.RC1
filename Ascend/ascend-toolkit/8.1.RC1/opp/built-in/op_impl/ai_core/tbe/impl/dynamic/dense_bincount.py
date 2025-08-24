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

dense_bincount
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
    TILING_PARAMS_NUM = 13
    TILING_PARAMS_DTYPE = DTYPE_INT32

    DEFAULT_SPLIT_NUM = 2

    MAX_INPUT_DIM_NUM = 2
    EACH_GE_1_SPLIT_1_TAIL_0 = 1
    EACH_GE_1_SPLIT_0_TAIL_N_0 = 2
    EACH_1_SPLIT_GT_1_CS_1 = 3
    EACH_1_SPLIT_GT_1_CS_2 = 4
    EACH_1_SPLIT_EFFECT_N_0 = 5

    WEIGHT_VALUE_IN_COMPUTE_IS_WEIGHT = False
    WEIGHT_VALUE_IN_COMPUTE_IS_1 = True


class DenseBincount:
    """
    The class for dense bincount
    """

    def __init__(self, input_, size, weights, output, binary_output=False, kernel_name="dense_bincount"):
        """
        constructor of class DenseBincount

        Parameters
        ----------
        input_: input_ input
        size: size input
        weights: weights input
        output: output
        binary_output: alternative parameter to control weight_value, default to "False"
        kernel_name: default to "dense_bincount"

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

            def __init__(self, tik_inst, input_dtype, weights_dtype):
                """
                constructor of class CalculationScalar

                Parameters
                ----------
                tik_inst: self.tik_instance

                Returns
                -------
                None
                """
                self.input_ub_block_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                          name="input_ub_block_num", init_value=0)
                self.ub_calc_input_max_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                             name="ub_calc_input_max_num", init_value=0)
                self.ub_calc_output_offset = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                             name="ub_calc_output_offset", init_value=0)

                # ub needed scalars
                self.ub_calc_input_offset = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                            name="ub_calc_input_offset", init_value=0)
                self.ub_calc_input_ub_block_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                                  name="ub_calc_input_ub_block_num", init_value=0)
                self.ub_calc_input_extended = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                              name="ub_calc_input_extended", init_value=0)
                self.output_init_repeat_times = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                                name="output_init_repeat_times", init_value=0)
                self.output_init_mask = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                        name="output_init_mask", init_value=0)
                self.output_total_ub_block_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                                 name="output_total_ub_block_num", init_value=0)
                self.actual_output_ub_block_num = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                                  name="actual_output_ub_block_num", init_value=0)
                self.output_extended = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                       name="output_extended", init_value=0)

                # calculation needed scalars
                self.ub_calc_input_begin = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                           name="ub_calc_input_begin", init_value=0)
                self.ub_calc_input_end = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                         name="ub_calc_input_end", init_value=0)
                self.input_data = tik_inst.Scalar(input_dtype,
                                                  name="input_data", init_value=0)
                self.weights_value = tik_inst.Scalar(weights_dtype,
                                                     name="weights_value", init_value=0)
                self.result_offset = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                     name="result_offset", init_value=0)
                self.output_ub_origin_idx = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                            name="output_ub_origin_idx", init_value=0)
                self.output_ub_current_idx = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                             name="output_ub_current_idx", init_value=0)
                self.negative_offset = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                       name="negative_offset", init_value=0)
                self.output_dim0_offset = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                          name="output_dim0_offset", init_value=0)
                self.output_dim0_origin_idx = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                              name="output_dim0_origin_idx", init_value=0)
                self.output_dim0_current_idx = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                               name="output_dim0_current_idx", init_value=0)

        class GmTensor:
            """
            The class for gm tensor
            """

            def __init__(self, tik_inst, input_dtype, size_dtype, weights_dtype, output_dtype):
                """
                constructor of class GmTensor

                Parameters
                ----------
                tik_inst: self.tik_instance
                input_dtype: input_ dtype
                size_dtype: size dtype
                weights_dtype: weights dtype
                output_dtype: output dtype

                Returns
                -------
                None
                """
                self.tiling_gm = tik_inst.Tensor(Constant.TILING_PARAMS_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                 name="tiling_gm", scope=tik.scope_gm)
                self.input_gm = tik_inst.Tensor(input_dtype, (Constant.MAX_INT64_VALUE,),
                                                name="input_gm", scope=tik.scope_gm)
                self.size_gm = tik_inst.Tensor(size_dtype, (Constant.MAX_INT64_VALUE,),
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
                self.ub_calc_input_ub = None
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
                self.need_core_num = tik_inst.Scalar(
                    Constant.DTYPE_INT32, name="need_core_num")
                self.input_col_num = tik_inst.Scalar(
                    Constant.DTYPE_INT32, name="input_dim_num")
                self.size_data = tik_inst.Scalar(
                    Constant.DTYPE_INT32, name="size_data")
                self.weights_num = tik_inst.Scalar(
                    Constant.DTYPE_INT32, name="weights_num")
                self.output_total_num = tik_inst.Scalar(
                    Constant.DTYPE_INT32, name="output_total_num")
                self.input_num_each_core = tik_inst.Scalar(
                    Constant.DTYPE_INT32, name="input_num_each_core")
                self.input_num_tail_core = tik_inst.Scalar(
                    Constant.DTYPE_INT32, name="input_num_tail_core")
                self.input_ub_block_num_wth_1 = tik_inst.Scalar(
                    Constant.DTYPE_INT32, name="input_ub_block_num_wth_1")
                self.input_ub_block_num_wth_wgt = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                                  name="input_ub_block_num_wth_wgt")
                self.ub_calc_input_max_num_wth_1 = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                                   name="ub_calc_input_max_num_wth_1")
                self.ub_calc_input_max_num_wth_wgt = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                                     name="ub_calc_input_max_num_wth_wgt")
                self.output_ub_block_num = tik_inst.Scalar(
                    Constant.DTYPE_INT32, name="output_ub_block_num")
                self.core_num_var = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                    name="core_num_var")

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
                self.input_each_size = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                       name="input_each_size", init_value=0)
                self.weights_each_size = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                         name="weights_each_size", init_value=0)
                self.output_each_size = tik_inst.Scalar(Constant.DTYPE_INT32,
                                                        name="output_each_size", init_value=0)

        self.input_dtype = input_.get("dtype").lower()
        self.size_dtype = size.get("dtype").lower()
        self.weights_dtype = weights.get("dtype").lower()
        self.output_dtype = output.get("dtype").lower()

        self.obj_calc_scalar = CalculationScalar(
            self.tik_instance, self.input_dtype, self.weights_dtype)
        self.obj_gm_tensor = GmTensor(self.tik_instance, self.input_dtype, self.size_dtype,
                                      self.weights_dtype, self.output_dtype)
        self.obj_ub_tensor = UbTensor()
        self.obj_tiling_scalar = TilingScalar(self.tik_instance)
        self.obj_data_size_scalar = DataSizeScalar(self.tik_instance)
        self.binary_output = binary_output

        if self.input_dtype in (Constant.DTYPE_INT32,):
            self.obj_data_size_scalar.input_each_size.set_as(
                Constant.BTYE_32BIT)
        elif self.input_dtype in (Constant.DTYPE_INT64,):
            self.obj_data_size_scalar.input_each_size.set_as(
                Constant.BTYE_64BIT)

        if self.weights_dtype in (Constant.DTYPE_FP32,):
            self.obj_data_size_scalar.weights_each_size.set_as(
                Constant.BTYE_32BIT)

        if self.output_dtype in (Constant.DTYPE_FP32,):
            self.obj_data_size_scalar.output_each_size.set_as(
                Constant.BTYE_32BIT)

    def dense_bincount_compute(self):
        """
        The tik implementation of operator DenseBincount
        """
        self._init_tiling_params()
        _enable_atomic_add(self.tik_instance)
        self._run_diff_calc_case()
        _disable_atomic_add(self.tik_instance)

        tbe_context.get_context().add_compile_info("vars",
                                                   {"ub_size": self.ub_size,
                                                    "core_num": self.ai_core_num})
        opt_config = {"enable_const_fold": True,
                      "out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.obj_gm_tensor.input_gm,
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
        self.obj_tiling_scalar.input_col_num.set_as(
            self.obj_ub_tensor.tiling_ub[1])
        self.obj_tiling_scalar.size_data.set_as(
            self.obj_ub_tensor.tiling_ub[2])
        self.obj_tiling_scalar.weights_num.set_as(
            self.obj_ub_tensor.tiling_ub[3])
        self.obj_tiling_scalar.output_total_num.set_as(
            self.obj_ub_tensor.tiling_ub[4])
        self.obj_tiling_scalar.input_num_each_core.set_as(
            self.obj_ub_tensor.tiling_ub[5])
        self.obj_tiling_scalar.input_num_tail_core.set_as(
            self.obj_ub_tensor.tiling_ub[6])
        self.obj_tiling_scalar.input_ub_block_num_wth_1.set_as(
            self.obj_ub_tensor.tiling_ub[7])
        self.obj_tiling_scalar.input_ub_block_num_wth_wgt.set_as(
            self.obj_ub_tensor.tiling_ub[8])
        self.obj_tiling_scalar.ub_calc_input_max_num_wth_1.set_as(
            self.obj_ub_tensor.tiling_ub[9])
        self.obj_tiling_scalar.ub_calc_input_max_num_wth_wgt.set_as(
            self.obj_ub_tensor.tiling_ub[10])
        self.obj_tiling_scalar.output_ub_block_num.set_as(
            self.obj_ub_tensor.tiling_ub[11])
        self.obj_tiling_scalar.core_num_var.set_as(
            self.obj_ub_tensor.tiling_ub[12])

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
            tiling_each_block = Constant.BLOCK_SIZE // Constant.BTYE_32BIT
            self.tik_instance.data_move(self.obj_ub_tensor.tiling_ub,
                                        self.obj_gm_tensor.tiling_gm,
                                        0, 1,
                                        (Constant.TILING_PARAMS_NUM + tiling_each_block - 1) // tiling_each_block,
                                        0, 0)
            self._get_tiling_params()

    def _run_compute_wth_diff_case(self, ub_calc_input_num, each_loop_calc_offset, input_begin):
        """
        prepare some data ub initialization and select run which calculation case

        Parameters
        ----------
        ub_calc_input_num: self.obj_calc_scalar.input_ub_block_num or last_input_num
        each_loop_calc_offset: _loop_idx * self.obj_calc_scalar.input_ub_block_num
                                    or
                               max_ub_calc_loop * self.obj_calc_scalar.input_ub_block_num
        input_begin: self.actual_input_num_each_core * _core_idx
                            or
                      self.obj_tiling_scalar.input_num_tail_core + _core_idx *
                      self.obj_tiling_scalar.input_num_each_core

        Returns
        -------
        None
        """
        self.obj_calc_scalar.ub_calc_input_offset.set_as(
            input_begin + each_loop_calc_offset)

        self.obj_calc_scalar.ub_calc_input_ub_block_num.set_as(ceil_div(ub_calc_input_num *
                                                               self.obj_data_size_scalar.input_each_size,
                                                               Constant.BLOCK_SIZE))
        self.obj_calc_scalar.ub_calc_input_extended.set_as(self.obj_calc_scalar.ub_calc_input_ub_block_num *
                                                           Constant.BLOCK_SIZE //
                                                           self.obj_data_size_scalar.input_each_size)
        _create_input_ub(self.tik_instance, self.obj_calc_scalar,
                         self.obj_ub_tensor, self.input_dtype, self.obj_gm_tensor)

        # output_ub must be initialized out of the compute process,
        # it can be reused many times with _init_output_ub function
        _init_output_need_scalar(self.tik_instance, self.obj_calc_scalar, self.obj_tiling_scalar,
                                 self.obj_data_size_scalar)

        self.obj_calc_scalar.ub_calc_input_end.set_as(ub_calc_input_num)

        with self.tik_instance.if_scope(tik.any(self.binary_output ==
                                        Constant.WEIGHT_VALUE_IN_COMPUTE_IS_1,
                                        self.obj_tiling_scalar.weights_num == 0)):

            self.obj_calc_scalar.weights_value.set_as(1)

            _run_process_output_calculation_wth_1(
                self.tik_instance, self.obj_calc_scalar, self.obj_tiling_scalar,
                self.obj_ub_tensor, self.obj_gm_tensor, self.output_dtype)

        with self.tik_instance.elif_scope(self.binary_output ==
                                          Constant.WEIGHT_VALUE_IN_COMPUTE_IS_WEIGHT):
            # default to the numbers of weights has the same as input_ has
            _create_weights_ub(self.tik_instance, self.obj_ub_tensor,
                               self.weights_dtype, self.obj_calc_scalar, self.obj_gm_tensor)

            _run_process_output_calculation_wth_wgt(
                self.tik_instance, self.obj_calc_scalar, self.obj_tiling_scalar,
                self.obj_ub_tensor, self.obj_gm_tensor, self.output_dtype)

    def _each_core_compute(self, input_begin, actual_input_num_each_core):
        """
        each core compute

        Parameters
        ----------
        input_begin: input_begin
        actual_input_num_each_core: actual_input_num_each_core

        Returns
        -------
        None
        """
        each_loop_calc_offset = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                         name="each_core_calc_offset", init_value=0)
        ub_calc_input_num = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                     name="ub_calc_input_num", init_value=0)

        # compute times according to the ub can handle input_ numbers
        max_ub_calc_loop = actual_input_num_each_core // self.obj_calc_scalar.ub_calc_input_max_num
        with self.tik_instance.for_range(0, max_ub_calc_loop) as _loop_idx:
            each_loop_calc_offset.set_as(
                _loop_idx * self.obj_calc_scalar.ub_calc_input_max_num)
            ub_calc_input_num.set_as(
                self.obj_calc_scalar.ub_calc_input_max_num)
            self._run_compute_wth_diff_case(
                ub_calc_input_num, each_loop_calc_offset, input_begin)

        # calculate the last numbers of input_
        last_input_num = actual_input_num_each_core % self.obj_calc_scalar.ub_calc_input_max_num
        with self.tik_instance.if_scope(last_input_num > 0):
            each_loop_calc_offset.set_as(
                max_ub_calc_loop * self.obj_calc_scalar.ub_calc_input_max_num)
            ub_calc_input_num.set_as(last_input_num)
            self._run_compute_wth_diff_case(
                ub_calc_input_num, each_loop_calc_offset, input_begin)

    def _run_diff_calc_case(self):
        with self.tik_instance.for_range(0, self.obj_tiling_scalar.core_num_var,
                                         block_num=self.obj_tiling_scalar.core_num_var) as _core_idx:
            with self.tik_instance.if_scope(_core_idx < self.obj_tiling_scalar.need_core_num):
                # used to store actually calculated numbers of input_ in each core
                actual_input_num_each_core = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                                      name="actual_input_num_each_core")
                input_begin = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                       name="input_begin", init_value=0)
                actual_input_num_each_core.set_as(
                    self.obj_tiling_scalar.input_num_each_core)
                with self.tik_instance.if_scope(tik.all(_core_idx < self.obj_tiling_scalar.input_num_tail_core,
                                                self.obj_tiling_scalar.input_num_tail_core != 0)):
                    actual_input_num_each_core.set_as(
                        self.obj_tiling_scalar.input_num_each_core + 1)

                with self.tik_instance.if_scope(actual_input_num_each_core >
                                                self.obj_tiling_scalar.input_num_each_core):
                    input_begin.set_as(actual_input_num_each_core * _core_idx)
                with self.tik_instance.elif_scope(actual_input_num_each_core ==
                                                  self.obj_tiling_scalar.input_num_each_core):
                    input_begin.set_as(self.obj_tiling_scalar.input_num_tail_core + _core_idx *
                                       self.obj_tiling_scalar.input_num_each_core)

                with self.tik_instance.if_scope(tik.any(self.binary_output ==
                                                        Constant.WEIGHT_VALUE_IN_COMPUTE_IS_1,
                                                        self.obj_tiling_scalar.weights_num == 0)):
                    self.obj_calc_scalar.input_ub_block_num.set_as(
                        self.obj_tiling_scalar.input_ub_block_num_wth_1)
                    self.obj_calc_scalar.ub_calc_input_max_num.set_as(
                        self.obj_tiling_scalar.ub_calc_input_max_num_wth_1)
                with self.tik_instance.elif_scope(self.binary_output ==
                                                  Constant.WEIGHT_VALUE_IN_COMPUTE_IS_WEIGHT):
                    self.obj_calc_scalar.input_ub_block_num.set_as(
                        self.obj_tiling_scalar.input_ub_block_num_wth_wgt)
                    self.obj_calc_scalar.ub_calc_input_max_num.set_as(
                        self.obj_tiling_scalar.ub_calc_input_max_num_wth_wgt)

                self._each_core_compute(
                    input_begin, actual_input_num_each_core)


def _run_process_output_calculation_wth_1(tik_inst, o_calc, o_tiling, o_ub, o_gm, output_dtype):
    """
    compute output result with weight_value = weights[_input_idx]

    Parameters
    ----------
    tik_inst: self.tik_instance
    o_calc: self.obj_calc_scalar
    o_tiling: self.obj_tiling_scalar
    o_ub: self.obj_ub_tensor
    o_gm: self.obj_gm_tensor
    output_dtype: self.output_dtype

    Returns
    -------
    None
    """
    ub_calc_output_ub = tik_inst.Tensor(output_dtype,
                                        (o_calc.output_extended,),
                                        name="ub_calc_output_ub",
                                        scope=tik.scope_ubuf)
    _init_output_ub(tik_inst, o_calc.output_init_mask, ub_calc_output_ub,
                    o_calc.output_init_repeat_times)

    o_calc.output_dim0_origin_idx.set_as(
        o_calc.ub_calc_input_offset // o_tiling.input_col_num)
    with tik_inst.for_range(o_calc.ub_calc_input_begin, o_calc.ub_calc_input_end) as _input_idx:
        o_calc.input_data.set_as(
            o_ub.ub_calc_input_ub[_input_idx])

        with tik_inst.if_scope(o_calc.input_data < o_tiling.size_data):
            o_calc.output_dim0_offset.set_as(
                (o_calc.ub_calc_input_offset + _input_idx) // o_tiling.input_col_num)

            with tik_inst.if_scope(o_calc.input_data < 0):
                o_calc.negative_offset.set_as(
                    ceil_div(-o_calc.input_data, o_tiling.size_data))
                o_calc.input_data.set_as(
                    o_calc.negative_offset * o_tiling.size_data + o_calc.input_data)

            with tik_inst.if_scope(o_calc.negative_offset <= o_calc.output_dim0_offset):
                o_calc.ub_calc_output_offset.set_as(
                    o_calc.output_dim0_origin_idx * o_tiling.size_data +
                    o_calc.output_ub_origin_idx * o_calc.output_extended)

                o_calc.output_ub_current_idx.set_as(
                    o_calc.input_data // o_calc.output_extended)
                o_calc.output_dim0_current_idx.set_as(
                    o_calc.output_dim0_offset - o_calc.negative_offset)
                with tik_inst.if_scope(tik.any(o_calc.output_ub_origin_idx != o_calc.output_ub_current_idx,
                                               o_calc.output_dim0_origin_idx != o_calc.output_dim0_current_idx)):
                    _handle_output_result(tik_inst, ub_calc_output_ub,
                                          o_gm.output_gm, o_calc.ub_calc_output_offset,
                                          o_calc.actual_output_ub_block_num)
                    _init_output_ub(tik_inst, o_calc.output_init_mask, ub_calc_output_ub,
                                    o_calc.output_init_repeat_times)

                    o_calc.output_ub_origin_idx.set_as(
                        o_calc.output_ub_current_idx)
                    o_calc.output_dim0_origin_idx.set_as(
                        o_calc.output_dim0_current_idx)

                    o_calc.ub_calc_output_offset.set_as(
                        o_calc.output_dim0_origin_idx * o_tiling.size_data +
                        o_calc.output_ub_origin_idx * o_calc.output_extended)

                o_calc.result_offset.set_as(
                    o_calc.input_data % o_calc.output_extended)

                _get_output_result(
                    tik_inst, ub_calc_output_ub, o_calc.result_offset, o_calc.weights_value)

            o_calc.negative_offset.set_as(0)

    _handle_output_result(tik_inst, ub_calc_output_ub,
                          o_gm.output_gm, o_calc.ub_calc_output_offset,
                          o_calc.actual_output_ub_block_num)


def _run_process_output_calculation_wth_wgt(tik_inst, o_calc, o_tiling, o_ub, o_gm, output_dtype):
    """
    compute output result with weight_value = weights[_input_idx]

    Parameters
    ----------
    tik_inst: self.tik_instance
    o_calc: self.obj_calc_scalar
    o_tiling: self.obj_tiling_scalar
    o_ub: self.obj_ub_tensor
    o_gm: self.obj_gm_tensor
    output_dtype: self.output_dtype

    Returns
    -------
    None
    """
    ub_calc_output_ub = tik_inst.Tensor(output_dtype,
                                        (o_calc.output_extended,),
                                        name="ub_calc_output_ub",
                                        scope=tik.scope_ubuf)
    _init_output_ub(tik_inst, o_calc.output_init_mask, ub_calc_output_ub,
                    o_calc.output_init_repeat_times)

    o_calc.output_dim0_origin_idx.set_as(
        o_calc.ub_calc_input_offset // o_tiling.input_col_num)
    with tik_inst.for_range(o_calc.ub_calc_input_begin, o_calc.ub_calc_input_end) as _input_idx:
        o_calc.input_data.set_as(
            o_ub.ub_calc_input_ub[_input_idx])

        with tik_inst.if_scope(o_calc.input_data < o_tiling.size_data):
            o_calc.output_dim0_offset.set_as(
                (o_calc.ub_calc_input_offset + _input_idx) // o_tiling.input_col_num)

            with tik_inst.if_scope(o_calc.input_data < 0):
                o_calc.negative_offset.set_as(
                    ceil_div(-o_calc.input_data, o_tiling.size_data))
                o_calc.input_data.set_as(
                    o_calc.negative_offset * o_tiling.size_data + o_calc.input_data)

            with tik_inst.if_scope(o_calc.negative_offset <= o_calc.output_dim0_offset):
                o_calc.ub_calc_output_offset.set_as(
                    o_calc.output_dim0_origin_idx * o_tiling.size_data +
                    o_calc.output_ub_origin_idx * o_calc.output_extended)

                o_calc.output_ub_current_idx.set_as(
                    o_calc.input_data // o_calc.output_extended)
                o_calc.output_dim0_current_idx.set_as(
                    o_calc.output_dim0_offset - o_calc.negative_offset)
                with tik_inst.if_scope(tik.any(o_calc.output_ub_origin_idx != o_calc.output_ub_current_idx,
                                               o_calc.output_dim0_origin_idx != o_calc.output_dim0_current_idx)):
                    _handle_output_result(tik_inst, ub_calc_output_ub,
                                          o_gm.output_gm, o_calc.ub_calc_output_offset,
                                          o_calc.actual_output_ub_block_num)
                    _init_output_ub(tik_inst, o_calc.output_init_mask, ub_calc_output_ub,
                                    o_calc.output_init_repeat_times)

                    o_calc.output_ub_origin_idx.set_as(
                        o_calc.output_ub_current_idx)
                    o_calc.output_dim0_origin_idx.set_as(
                        o_calc.output_dim0_current_idx)

                    o_calc.ub_calc_output_offset.set_as(
                        o_calc.output_dim0_origin_idx * o_tiling.size_data +
                        o_calc.output_ub_origin_idx * o_calc.output_extended)

                o_calc.result_offset.set_as(
                    o_calc.input_data % o_calc.output_extended)

                o_calc.weights_value.set_as(
                    o_ub.ub_calc_weights_ub[_input_idx])
                _get_output_result(
                    tik_inst, ub_calc_output_ub, o_calc.result_offset, o_calc.weights_value)

            o_calc.negative_offset.set_as(0)

    _handle_output_result(tik_inst, ub_calc_output_ub,
                          o_gm.output_gm, o_calc.ub_calc_output_offset,
                          o_calc.actual_output_ub_block_num)


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


def _create_input_ub(tik_inst, o_calc, o_ub, input_dtype, o_gm):
    """
    create input needed ub tensor

    Parameters
    ----------
    tik_inst: self.tik_instance
    o_calc: self.obj_calc_scalar
    o_ub: self.obj_ub_tensor
    input_dtype: self.input_dtype
    o_gm: self.obj_gm_tensor

    Returns
    -------
    None
    """
    o_ub.ub_calc_input_ub = tik_inst.Tensor(input_dtype,
                                            (o_calc.ub_calc_input_extended,),
                                            name="ub_calc_input_ub",
                                            scope=tik.scope_ubuf)
    _param_data_move_gm2ub(tik_inst, o_ub.ub_calc_input_ub,
                           o_gm.input_gm, o_calc.ub_calc_input_offset, o_calc.ub_calc_input_ub_block_num)


def _init_output_need_scalar(tik_inst, o_calc, o_tiling, o_dsize):
    """
    create output needed ub tensor

    Parameters
    ----------
    tik_inst: self.tik_instance
    o_calc: self.obj_calc_scalar
    o_tiling: self.obj_tiling_scalar
    o_dsize: self.obj_data_size_scalar

    Returns
    -------
    None
    """
    o_calc.output_init_mask.set_as(Constant.REPEAT_MASK_32BIT)

    o_calc.output_total_ub_block_num.set_as(ceil_div(o_tiling.output_total_num *
                                                     o_dsize.output_each_size, Constant.BLOCK_SIZE))
    with tik_inst.if_scope(o_calc.output_total_ub_block_num > o_tiling.output_ub_block_num):
        o_calc.actual_output_ub_block_num.set_as(
            o_tiling.output_ub_block_num)
    with tik_inst.else_scope():
        o_calc.actual_output_ub_block_num.set_as(
            o_calc.output_total_ub_block_num)
    o_calc.output_extended.set_as(o_calc.actual_output_ub_block_num * Constant.BLOCK_SIZE //
                                  o_dsize.output_each_size)
    o_calc.output_extended.set_as(ceil_div(o_calc.output_extended, Constant.REPEAT_MASK_32BIT) *
                                  Constant.REPEAT_MASK_32BIT)
    o_calc.output_init_repeat_times.set_as(
        ceil_div(o_calc.output_extended, Constant.REPEAT_MASK_32BIT))


def _create_weights_ub(tik_inst, o_ub, weights_dtype, o_calc, o_gm):
    """
    create weights needed ub tensor

    Parameters
    ----------
    tik_inst: self.tik_instance
    o_ub: self.obj_ub_tensor
    weights_dtype: self.weights_dtype
    o_calc: self.obj_calc_scalar
    o_gm: self.obj_gm_tensor

    Returns
    -------
    None
    """
    o_ub.ub_calc_weights_ub = tik_inst.Tensor(weights_dtype,
                                              (o_calc.ub_calc_input_extended,),
                                              name="ub_calc_weights_ub",
                                              scope=tik.scope_ubuf)
    _param_data_move_gm2ub(tik_inst, o_ub.ub_calc_weights_ub,
                           o_gm.weights_gm, o_calc.ub_calc_input_offset, o_calc.ub_calc_input_ub_block_num)


def _init_output_ub(tik_inst, mask, output_ub, repeat):
    """
    initialize output ub with 0.0 to make the output ub can be used many times with once ub declaration

    Parameters
    ----------
    tik_inst: self.tik_instance
    mask: output_init_mask
    output_ub: ub_calc_output_ub
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


def _get_output_result(tik_inst, ub_calc_output_ub, result_offset, weights_value):
    """
    calculate output value

    Parameters
    ----------
    tik_inst: self.tik_instance
    ub_calc_output_ub: ub_calc_output_ub
    result_offset: (self.obj_tiling_scalar.size_data + input_data) % output_extended
    weights_value: self.obj_ub_tensor.ub_calc_weights_ub[_input_idx] or 1

    Returns
    -------
    None
    """
    output_data = tik_inst.Scalar(Constant.DTYPE_FP32,
                                  name="output_data")
    output_data.set_as(ub_calc_output_ub[result_offset])
    ub_calc_output_ub[result_offset].set_as(
        output_data + weights_value)


def _handle_output_result(tik_inst, output_ub, output_gm, output_offset, output_ub_block_num):
    """
    calculate output value

    Parameters
    ----------
    tik_inst: self.tik_instance
    output_ub: ub_calc_output_ub
    output_gm: self.obj_gm_tensor.output_gm
    output_offset: self.obj_calc_scalar.ub_calc_output_offset
    output_ub_block_num: output_ub_block_num

    Returns
    -------
    None
    """
    tik_inst.data_move(output_gm[output_offset],
                       output_ub, 0, 1, output_ub_block_num, 0, 0)


def check_supported(input_, size, weights, output, binary_output=False, kernel_name="dense_bincount"):
    """
    output has the same type as weights, all only can use float32,
    because of the set_atomic_add() only supports float32
    """
    input_dtype = input_.get("dtype").lower()
    size_dtype = size.get("dtype").lower()
    weights_dtype = weights.get("dtype").lower()
    output_dtype = output.get("dtype").lower()

    if input_dtype not in ("int32", "int64"):
        reason = "the input_'s dtype must be int32 or int64, input_dtype=%s" % input_dtype
        return False, reason
    if size_dtype not in ("int32", "int64"):
        reason = "the size's dtype must be int32 or int64, size_dtype=%s" % size_dtype
        return False, reason
    if size_dtype != input_dtype:
        reason = "the size's dtype must be the same as input_'s dtype, size_dtype=%s, input_dtype=%s" % \
            (size_dtype, input_dtype)
        return False, reason
    if weights_dtype not in ("float32"):
        reason = "the weights's dtype must be float32, weights_dtype=%s" % weights_dtype
        return False, reason
    if output_dtype not in ("float32"):
        reason = "the output's dtype must be float32, output_dtype=%s" % output_dtype
        return False, reason

    return True, ""


@register_operator("DenseBincount")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def dense_bincount(input_, size, weights, output, binary_output=False, kernel_name="dense_bincount"):
    """
    algorithm: dense_bincount
    calculating: Counts the number of occurrences of each value in an integer array

    Parameters
    ----------
    input_: dict
        dict of input_input, include shape with 2D and dtype with int32 or int64,

    size: dict
        dict of input_size, include shape and dtype,
        Must have the same dtype as input_input, and must be a non-negative int scalar Tensor,

    weights: dict
        dict of input_weights, include shape and dtype,
        which has two cases:
            case 1: Must have the same shapes as input_input with dtype int32, int64, float32, or float64,
            case 2: Can be a length-0 Tensor, in which case it acts as all weights equal to 1,

    output: dict
        dict of output,
        Must be a vector with length size and the same dtype as weights,

    binary_output: bool
        An optional bool. Defaults to False,
        Whether the kernel should count the appearance or number of occurrences,

    kernel_name : str
        cce kernel name, default value is "dense_bincount"

    Returns
    -------
    None
    """

    obj = DenseBincount(input_, size, weights, output,
                        binary_output, kernel_name)
    obj.dense_bincount_compute()
