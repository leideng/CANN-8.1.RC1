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
unsorted_segment_sum
"""
# 'pylint: disable=too-many-lines
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    # int32 select key
    SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID = 9
    SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID = 10
    SELECT_KEY_MODE_NO_ATOMIC_BIG_E_SMALL_ID = 11
    SELECT_KEY_MODE_NO_ATOMIC_BIG_E_BIG_ID = 12
    SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID_BLOCK = 13
    SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID_BLOCK = 14
    SELECT_KEY_MODE_NO_ATOMIC_NUM_SEGMENT_ONE = 15
    SELECT_KEY_MODE_NO_ATOMIC_ALL_IN_ALIGN = 16

    DTYPE_FP32 = "float32"
    DTYPE_FP16 = "float16"
    DTYPE_INT32 = "int32"
    TILING_PARAM_DTYPE = DTYPE_INT32

    # max_int32
    MAX_INT32 = 2**31 - 1

    # fp32 byte
    BYTE_FP32 = 4

    # int32 byte
    BYTE_INT32 = 4
    BYTE_FLOAT16 = 2

    # full mask for fp32
    MASK_FP32 = 64

    # full mask for int32
    MASK_INT32 = 64
    # full mask for float16
    MASK_FP16 = 128

    # byte of one block
    BYTE_BLOCK = 32

    # min_tensor_ele_num
    MIN_TENSOR_ELE_NUM = 32

    # tiling params num
    TILING_PARAMS_NUM = 64

    # fp32 ele num one ub block
    ELE_NUM_ONE_BLOCK_FP32 = BYTE_BLOCK // BYTE_FP32
    # fp16 ele num one ub block
    ELE_NUM_ONE_BLOCK_FP16 = BYTE_BLOCK // BYTE_FLOAT16
    # int32 ele num one ub block
    ELE_NUM_ONE_BLOCK_INT32 = BYTE_BLOCK // BYTE_INT32


# 'pylint: disable=invalid-name,too-many-instance-attributes,too-many-arguments,too-many-statements
# 'pylint: disable=too-many-locals,too-few-public-methods,unused-argument
def _ceil_div(val, block):
    """
    compute ceil div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    ceil value
    """
    return (val + block - 1) // block


def _floor(val, block):
    """
    compute floor div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    floor value
    """
    return val // block * block


def _div(val, block):
    """
    compute front part and last part in ceil div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    front_part_num: front part in ceil div
    last_part: last part in ceil div
    """
    front_part_num = val // block
    last_part = val - front_part_num * block
    return front_part_num, last_part


class UnsortedSegmentSumNoAtomoic():
    """
        Function: use to store concat base parameters
        Modify : 2020-12-9
    """

    def __init__(self, x_dict, segment_ids_dict, num_segments_dict, y_dict, kernel_name, opname="unsorted_segment_sum"):
        """
        constructor of class UnsortedSegmentSumNoAtomoic

        Parameters
        ----------
        x_dict: dict
            shape and dtype of x
        segment_ids_dict: dict
            shape and dtype of segment_ids
        num_segments_dict: dict
            shape and dtype of num_segments
        y_dict: dict
            shape and dtype of y
        kernel_name: str
            kernel_name, default value is "UnsortedSegmentSumNoAtomoic"

        Returns
        -------
        None
        """
        # get dtype
        self.input_dtype = x_dict.get("dtype", None)
        self.input_dtype = self.input_dtype.lower()
        self.ids_dtype = segment_ids_dict.get("dtype", None)
        self.ids_dtype = self.ids_dtype.lower()
        self.num_segments_dtype = num_segments_dict.get("dtype", None)
        self.num_segments_dtype = self.num_segments_dtype.lower()
        self.output_dtype = self.input_dtype
        self.fp32_ele_num_one_block = Constant.ELE_NUM_ONE_BLOCK_FP32
        self.is_double_buffer = False
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.ub_size = _tik_get_ub_size(self.is_double_buffer)
        self.core_num = _tik_get_core_num()
        self.ub_tensor_num = 3
        self.opname = opname

        class GmTensor():
            """
            Function: use to store concat base parameters
            Modify : 2020-12-9
            """

            def __init__(self, tik_instance, input_dtype, ids_dtype, num_segments_dtype, opname):
                """
                constructor of class GmTensor

                Parameters
                ----------
                tik_instance: tik_instance
                input_dtype: x dtype
                ids_dtype: ids dtype
                num_segments_dtype: num_segments dtype

                Returns
                -------
                None
                """
                self.input_gm = tik_instance.Tensor(input_dtype, (Constant.MAX_INT32,), name="input_gm",
                                                    scope=tik.scope_gm)
                self.ids_gm = tik_instance.Tensor(ids_dtype, (Constant.MAX_INT32,), name="ids_gm", scope=tik.scope_gm)
                self.num_segments_gm = tik_instance.Tensor(num_segments_dtype, (Constant.MIN_TENSOR_ELE_NUM,),
                                                           name="num_segments_gm",
                                                           scope=tik.scope_gm)
                if opname == "unsorted_segment_sum":
                    self.num_segments_gm = tik_instance.Tensor(num_segments_dtype, (Constant.MIN_TENSOR_ELE_NUM,),
                                                               name="num_segments_gm",
                                                               scope=tik.scope_gm)

                self.output_gm = tik_instance.Tensor(input_dtype, (Constant.MAX_INT32,),
                                                     name="output_gm",
                                                     scope=tik.scope_gm,
                                                     is_atomic_add=True)
                self.tiling_gm = tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                     name="tiling_gm",
                                                     scope=tik.scope_gm)

        class UbTensor():
            """
            Function: use to store concat base parameters
            Modify : 2020-12-9
            """

            def __init__(self, opname):
                """
                constructor of class UbTensor

                Parameters
                ----------
                None

                Returns
                -------
                None
                """
                self.input_ub = None
                self.ids_ub = None
                self.output_ub = None
                if opname == "unsorted_segment_sum":
                    self.num_segments_ub = None

        # scalar of tiling params
        class CommonScalar():
            """
            Function: use to store concat base parameters
            Modify : 2020-12-9
            """

            def __init__(self, tik_instance, num_segments_dtype, ids_dtype, core_num):
                """
                constructor of class CommonScalar

                Parameters
                ----------
                tik_instance: tik_instance
                num_segments_dtype: num_segments dtype
                ids_dtype: ids dtype
                core_num: core_num

                Returns
                -------
                None
                """
                self.num_segments_scalar = tik_instance.Scalar(dtype=num_segments_dtype, name="num_segments_scalar")
                self.id_val_scalar = tik_instance.Scalar(dtype=ids_dtype, name="id_val_scalar")
                self.output_id_scalar = tik_instance.Scalar(dtype=ids_dtype, name="output_id_scalar", init_value=0)
                self.select_key = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="select_key")
                self.need_core_num = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="need_core_num")
                self.num_segments_front_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="num_segments_front_core")
                self.num_segments_last_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="num_segments_last_core")
                self.num_segments = tik_instance.Scalar(dtype=num_segments_dtype,
                                                        name="num_segments")
                self.core_num_var = tik_instance.Scalar(name="core_num_var", init_value=core_num)

            def set_running_core_num(self, tiling_core_num):
                self.core_num_var.set_as(tiling_core_num)

        class Int32IdsScalar():
            """
            Function: use to store concat base parameters
            Modify : 2020-12-9
            """

            def __init__(self, tik_instance):
                """
                constructor of class Int32IdsScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.size = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="size")
                self.mov_times_gm2ub = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="mov_times_gm2ub")
                self.ele_num_ub_front_part = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="ele_num_ub_front_part")
                self.front_burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="front_burst_len")
                self.ele_num_ub_last_part = \
                    tik_instance.Scalar(
                        dtype=Constant.TILING_PARAM_DTYPE,
                        name="ele_num_ub_last_part")
                self.last_burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="last_burst_len")

        class Int32ENumScalar():
            """
            Function: use to store concat base parameters
            Modify : 2020-12-9
            """

            def __init__(self, tik_instance):
                """
                constructor of class Int32ENumScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.e_num = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="e_num")
                self.e_mov_times_gm2ub = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                             name="e_mov_times_gm2ub")
                self.e_ub2gm_front_burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="e_ub2gm_front_burst_len")
                self.e_num_front_part = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="e_num_front_part")
                self.repeat_time_front_part = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="repeat_time_front_part")
                self.e_ub2gm_last_burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="e_ub2gm_last_burst_len")
                self.e_num_last_part = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="e_num_last_part")
                self.repeat_time_last_part = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                 name="repeat_time_last_part")
                self.align_scalar = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="align_scalar")
                self.align_scalar_last_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="align_scalar_last_core")
                self.e_gm2ub_front_burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                   name="e_gm2ub_front_burst_len")
                self.e_gm2ub_last_burst_len = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="e_gm2ub_last_burst_len")
                self.num_segment_max = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="num_segment_max")
                self.e_max_num_time = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="e_max_num_time")
                self.e_max_num_time_last_core = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                    name="e_max_num_time_last_core")
                self.front_num_segment = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                             name="front_num_segment")
                self.front_num_segment_last = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="front_num_segment_last")
                self.front_num_segment_lastcore = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                      name="front_num_segment_lastcore")
                self.front_num_segment_last_lastcore = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                           name="front_num_segment_last_lastcore")
                self.e_ub2gm_front_bust_len_small_e_last_core = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="e_ub2gm_front_bust_len_small_e_last_core")
                self.e_ub2gm_last_burst_len_input_scalar_lastcore = tik_instance.Scalar(
                    dtype=Constant.TILING_PARAM_DTYPE, name="e_ub2gm_last_burst_len_input_scalar_lastcore")
                self.repeat_times = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE, name="repeat_times")
                self.repeat_times_last_part = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                  name="repeat_times_last_part")
                self.repeat_times_last_part_lastcore = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                           name="repeat_times_last_part_lastcore")
                self.e_mov_times_gm2ub_lastcore = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                      name="e_mov_times_gm2ub_lastcore")
                self.repeat_time_last_part_lastcore = tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                                                          name="repeat_time_last_part_lastcore")

        self.obj_gm_tensor = GmTensor(self.tik_instance, self.input_dtype, self.ids_dtype, self.num_segments_dtype,
                                      self.opname)
        self.obj_ub_tensor = UbTensor(self.opname)
        self.obj_common_scalar = CommonScalar(self.tik_instance, self.num_segments_dtype, 
                                              self.ids_dtype, self.core_num)

        self.obj_int32_ids_input_scalar = Int32IdsScalar(self.tik_instance)
        self.obj_int32_e_num_input_scalar = Int32ENumScalar(self.tik_instance)

        with self.tik_instance.new_stmt_scope():
            if self.opname == "unsorted_segment_sum":
                # num_segments
                self.obj_ub_tensor.num_segments_ub = self.tik_instance.Tensor(self.num_segments_dtype,
                                                                              (Constant.MIN_TENSOR_ELE_NUM,),
                                                                              name="num_segments_ub",
                                                                              scope=tik.scope_ubuf)
                self.tik_instance.data_move(self.obj_ub_tensor.num_segments_ub,
                                            self.obj_gm_tensor.num_segments_gm, 0, 1, 1, 0, 0)
                self.obj_common_scalar.num_segments_scalar.set_as(self.obj_ub_tensor.num_segments_ub[0])

        with self.tik_instance.new_stmt_scope():
            self.obj_ub_tensor.tiling_ub = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE,
                                                                    (Constant.TILING_PARAMS_NUM,),
                                                                    name="tiling_ub",
                                                                    scope=tik.scope_ubuf)
            # mov tiling params from gm to ub
            self.tik_instance.data_move(self.obj_ub_tensor.tiling_ub,
                                        self.obj_gm_tensor.tiling_gm, 0, 1,
                                        Constant.TILING_PARAMS_NUM * Constant.BYTE_INT32 // \
                                        Constant.BYTE_BLOCK,
                                        0, 0)
            # input scalar in flowtable
            input_scalar_index = 0

            # common params
            self.obj_common_scalar.select_key.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_common_scalar.need_core_num.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_common_scalar.num_segments_front_core.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_common_scalar.num_segments_last_core.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_ids_input_scalar.size.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_ids_input_scalar.mov_times_gm2ub.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_ids_input_scalar.ele_num_ub_front_part.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_ids_input_scalar.front_burst_len.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_ids_input_scalar.ele_num_ub_last_part.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_ids_input_scalar.last_burst_len.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_num.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_mov_times_gm2ub.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_num_front_part.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.repeat_time_front_part.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_num_last_part.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.repeat_time_last_part.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.align_scalar.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.align_scalar_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_gm2ub_last_burst_len.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.num_segment_max.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_max_num_time.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_max_num_time_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.front_num_segment.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1

            self.obj_int32_e_num_input_scalar.front_num_segment_last.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.front_num_segment_lastcore.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.front_num_segment_last_lastcore.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_ub2gm_front_bust_len_small_e_last_core.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len_input_scalar_lastcore.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.repeat_times.set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.repeat_times_last_part.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.repeat_times_last_part_lastcore.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.e_mov_times_gm2ub_lastcore.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_int32_e_num_input_scalar.repeat_time_last_part_lastcore.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            if self.opname == "segment_sum":
                input_scalar_index = input_scalar_index + 1
                self.obj_common_scalar.num_segments.set_as(input_scalar_index)
            
            input_scalar_index = input_scalar_index + 1
            self.obj_common_scalar.set_running_core_num( \
                self.obj_ub_tensor.tiling_ub[input_scalar_index])

    def unsorted_segment_sum(self):
        """
        main process of unsorted_segment_sum

        Parameters
        ----------
        None

        Returns:
        -------
        None
        """
        byte = Constant.BYTE_INT32 if self.input_dtype == Constant.DTYPE_INT32 else Constant.BYTE_FLOAT16
        mask = Constant.MASK_FP16 if self.input_dtype == Constant.DTYPE_FP16 else Constant.MASK_INT32
        ub_size = self.ub_size - 2 * 16000 * byte
        with self.tik_instance.for_range(0, self.obj_common_scalar.core_num_var, 
                                         block_num=self.obj_common_scalar.core_num_var) as block_index:
            with self.tik_instance.if_scope(block_index < self.obj_common_scalar.need_core_num):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == Constant.SELECT_KEY_MODE_NO_ATOMIC_NUM_SEGMENT_ONE):
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype,
                                                                               (self.ub_size // 2 // byte,),
                                                                               name="input_ub",
                                                                               scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype,
                                                                                (self.ub_size // 2 // byte,),
                                                                                name="output_ub",
                                                                                scope=tik.scope_ubuf)
                        _tik_no_atomic_num_segment_one(block_index, self.tik_instance, self.obj_gm_tensor,
                                                       self.obj_ub_tensor, self.obj_common_scalar,
                                                       self.obj_int32_ids_input_scalar,
                                                       self.obj_int32_e_num_input_scalar, self.input_dtype)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == Constant.SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID):
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype, (16000,),
                                                                               name="input_ub",
                                                                               scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype, (16000,),
                                                                                name="output_ub",
                                                                                scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype, (ub_size // \
                                                                                              Constant.BYTE_INT32,),
                                                                             name="ids_ub",
                                                                             scope=tik.scope_ubuf)
                        _tik_no_atomic_small_e_small_id(block_index, self.tik_instance, self.obj_gm_tensor,
                                                        self.obj_ub_tensor, self.obj_common_scalar,
                                                        self.obj_int32_ids_input_scalar,
                                                        self.obj_int32_e_num_input_scalar, self.input_dtype)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == Constant.SELECT_KEY_MODE_NO_ATOMIC_ALL_IN_ALIGN):
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype, (16000,),
                                                                               name="input_ub",
                                                                               scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype, (16000,),
                                                                                name="output_ub",
                                                                                scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype, (ub_size // \
                                                                                              Constant.BYTE_INT32,),
                                                                             name="ids_ub",
                                                                             scope=tik.scope_ubuf)
                        _tik_no_atomic_all_in_align(block_index, self.tik_instance, self.obj_gm_tensor,
                                                    self.obj_ub_tensor, self.obj_common_scalar,
                                                    self.obj_int32_ids_input_scalar, self.obj_int32_e_num_input_scalar,
                                                    self.input_dtype)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == Constant.SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID):
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype, (16000,),
                                                                               name="input_ub",
                                                                               scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype, (ub_size // \
                                                                                              Constant.BYTE_INT32,),
                                                                             name="ids_ub",
                                                                             scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype, (16000,),
                                                                                name="output_ub",
                                                                                scope=tik.scope_ubuf)
                        _tik_no_atomic_small_e_big_id(block_index, self.tik_instance, self.obj_gm_tensor,
                                                      self.obj_ub_tensor, self.obj_common_scalar,
                                                      self.obj_int32_ids_input_scalar,
                                                      self.obj_int32_e_num_input_scalar, self.input_dtype)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == Constant.SELECT_KEY_MODE_NO_ATOMIC_BIG_E_SMALL_ID):
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype, (16000,),
                                                                               name="input_ub",
                                                                               scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype, (ub_size // \
                                                                                              Constant.BYTE_INT32,),
                                                                             name="ids_ub",
                                                                             scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype, (16000,),
                                                                                name="output_ub",
                                                                                scope=tik.scope_ubuf)
                        _tik_no_atomic_big_e_small_id(block_index, self.tik_instance, self.obj_gm_tensor,
                                                      self.obj_ub_tensor, self.obj_common_scalar,
                                                      self.obj_int32_ids_input_scalar,
                                                      self.obj_int32_e_num_input_scalar, self.input_dtype)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == Constant.SELECT_KEY_MODE_NO_ATOMIC_BIG_E_BIG_ID):
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype, (16000 // byte,),
                                                                               name="input_ub",
                                                                               scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype, (16000 // \
                                                                                              Constant.BYTE_INT32,),
                                                                             name="ids_ub",
                                                                             scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype, (ub_size // byte,),
                                                                                name="output_ub",
                                                                                scope=tik.scope_ubuf)
                        _tik_no_atomic_big_e_big_id(block_index, self.tik_instance, self.obj_gm_tensor,
                                                    self.obj_ub_tensor, self.obj_common_scalar,
                                                    self.obj_int32_ids_input_scalar, self.obj_int32_e_num_input_scalar,
                                                    self.input_dtype)
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == \
                                Constant.SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID_BLOCK):
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype, (mask,),
                                                                               name="input_ub",
                                                                               scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype,
                                                                             ((self.ub_size - 256) // 2 // \
                                                                                 Constant.BYTE_INT32,),
                                                                             name="ids_ub",
                                                                             scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype,
                                                                                ((self.ub_size - 256) // 2 // byte,),
                                                                                name="output_ub",
                                                                                scope=tik.scope_ubuf)
                        _tik_no_atomic_small_e_block_small_id(block_index, self.tik_instance, self.obj_gm_tensor,
                                                              self.obj_ub_tensor, self.obj_common_scalar,
                                                              self.obj_int32_ids_input_scalar,
                                                              self.obj_int32_e_num_input_scalar, self.input_dtype)

                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key == \
                                Constant.SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID_BLOCK):
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(self.input_dtype, (mask,),
                                                                               name="input_ub",
                                                                               scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(self.ids_dtype,
                                                                             ((self.ub_size - 256) // 2 // \
                                                                                 Constant.BYTE_INT32,),
                                                                             name="ids_ub",
                                                                             scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(self.output_dtype,
                                                                                ((self.ub_size - 256) // 2 // byte,),
                                                                                name="output_ub",
                                                                                scope=tik.scope_ubuf)
                        _tik_no_atomic_small_e_block_big_id(block_index, self.tik_instance, self.obj_gm_tensor,
                                                            self.obj_ub_tensor, self.obj_common_scalar,
                                                            self.obj_int32_ids_input_scalar,
                                                            self.obj_int32_e_num_input_scalar, self.input_dtype)

        # add compile info
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size,
                "core_num": self.core_num,
                "dtype": self.obj_gm_tensor.input_gm.dtype,
                "ub_tensor_num": self.ub_tensor_num,
                "impl_mode": 0
            })
        opt_config = {
            "enable_const_fold": True
        }

        if self.opname == "segment_sum":
            self.tik_instance.BuildCCE(
                kernel_name=self.kernel_name,
                inputs=[self.obj_gm_tensor.input_gm, self.obj_gm_tensor.ids_gm],
                outputs=[self.obj_gm_tensor.output_gm],
                flowtable=[self.obj_gm_tensor.tiling_gm], config=opt_config)
        else:
            self.tik_instance.BuildCCE(
                kernel_name=self.kernel_name,
                inputs=[self.obj_gm_tensor.input_gm, self.obj_gm_tensor.ids_gm, self.obj_gm_tensor.num_segments_gm],
                outputs=[self.obj_gm_tensor.output_gm],
                flowtable=[self.obj_gm_tensor.tiling_gm], config=opt_config)


def _enable_atomic_add(tik_inst, dtype):
    """
    enable atomic add

    Parameters
    ----------
    tik_inst: tik instance

    Returns
    -------
    None
    """
    if tbe_platform.api_check_support("tik.set_atomic_add") and dtype == Constant.DTYPE_FP32:
        tik_inst.set_atomic_add(1)


def _disable_atomic_add(tik_inst, dtype):
    """
    disable atomic add

    Parameters
    ----------
    tik_inst: tik instance

    Returns
    -------
    None
    """
    if tbe_platform.api_check_support("tik.set_atomic_add") and dtype == Constant.DTYPE_FP32:
        tik_inst.set_atomic_add(0)


def _tik_get_ub_size(is_double_buffer=True):
    """
    get ub size

    Parameters
    ----------
    is_double_buffer: is_double_buffer flag

    Returns
    -------
    ub_size
    """
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 512
    if is_double_buffer:
        return ub_size // 2
    return ub_size


def _tik_get_core_num():
    """
    get core num

    Parameters
    ----------
    None

    Returns
    -------
    core num
    """
    return tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)


def _tik_init_ub_tensor_once(tik_inst, ub_tensor, repeat_time, mask, uboffset=0):
    """
    init ub tensor once

    Parameters
    ----------
    tik_inst: tik instance
    ub_tensor: ub_tensor
    repeat_time: repeat time
    mask: mask

    Returns
    -------
    None
    """
    tik_inst.vector_dup(mask, ub_tensor[uboffset], 0, repeat_time, 1, 8)


def _tik_vadd(tik_inst, input_ub, output_ub, repeat_time, mask, uboffset=0):
    """
    tik_vadd

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input ub tensor
    output_ub: output ub tensor
    repeat_time: repeat time
    mask: mask

    Returns
    -------
    None
    """
    tik_inst.vadd(mask, output_ub, output_ub, input_ub[uboffset], repeat_time, 1, 1, 1, 8, 8, 8)


def _tik_vadd_num_one(tik_inst, input_ub, output_ub, repeat_time, mask, uboffset=0):
    """
    _tik_vadd_num_one

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input ub tensor
    output_ub: output ub tensor
    repeat_time: repeat time
    mask: mask

    Returns
    -------
    None
    """
    tik_inst.vadd(mask, output_ub[uboffset], output_ub[uboffset], input_ub[uboffset], repeat_time, 1, 1, 1, 8, 8, 8)


def _tik_mov_output_ub2gm_continue(tik_inst, output_gm, output_ub, output_offset_gm, output_offset_ub, output_n_burst,
                                   output_burst_len):
    """
    tik_mov_output_ub2gm_continue

    Parameters
    ----------
    tik_inst: tik instance
    output_gm: output gm tensor
    output_ub: output ub tensor
    output_offset_gm: output offset gm
    output_offset_ub: output offset ub
    output_n_burst: n_burst
    output_burst_len: burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(output_gm[output_offset_gm], output_ub[output_offset_ub], 0, output_n_burst, output_burst_len, 0,
                       0)


def _tik_mov_input_gm2ub_continue(tik_inst, input_gm, input_ub, input_offset_gm, input_offset_ub, input_n_burst,
                                  input_burst_len):
    """
    tik_mov_input_gm2ub_continue

    Parameters
    ----------
    tik_inst: tik instance
    input_gm: input gm tensor
    input_ub: input ub tensor
    input_offset_gm: input offset gm
    input_offset_ub: input offset ub
    input_n_burst: n_burst
    input_burst_len: burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(input_ub[input_offset_ub], input_gm[input_offset_gm], 0, input_n_burst, input_burst_len, 0, 0)


def _tik_mov_ids_gm2ub(tik_inst, ids_gm, ids_ub, ids_offset_gm, ids_offset_ub, ids_n_burst, ids_burst_len):
    """
    tik_mov_ids_gm2ub

    Parameters
    ----------
    tik_inst: tik instance
    ids_gm: ids_gm tensor
    ids_ub: ids_ub tensor
    ids_offset_gm: ids_offset_gm
    ids_offset_ub: ids_offset_ub
    ids_n_burst: ids_n_burst
    ids_burst_len: ids_burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(ids_ub[ids_offset_ub], ids_gm[ids_offset_gm], 0, ids_n_burst, ids_burst_len, 0, 0)


def _tik_no_atomic_num_segment_one(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                   obj_int32_ids_input_scalar, obj_int32_e_num_input_scalar, dtype):
    """
    _tik_int32_add_big_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar
    dtype:input dtype

    Returns
    -------
    None
    """
    mask = Constant.MASK_FP16 if dtype == Constant.DTYPE_FP16 else Constant.MASK_INT32
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_mov_times_gm2ub) as e_mov_index_gm2ub:
            # e divide by ub
            with tik_inst.if_scope(e_mov_index_gm2ub < obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # front part e
                with tik_inst.for_range(0, obj_int32_e_num_input_scalar.repeat_times) as i:
                    ub_offset = i * 255 * mask
                    with tik_inst.if_scope(i < obj_int32_e_num_input_scalar.repeat_times - 1):
                        _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub, 255, mask, ub_offset)
                    with tik_inst.if_scope(i == obj_int32_e_num_input_scalar.repeat_times - 1):
                        _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                                 obj_int32_e_num_input_scalar.repeat_time_front_part, mask, ub_offset)
                with tik_inst.for_range(0, obj_int32_ids_input_scalar.size) as segment_index_front_core:
                    # num_segments divide by core
                    input_offset_gm = segment_index_front_core * obj_int32_e_num_input_scalar.e_num + \
                                      block_index * obj_common_scalar.num_segments_front_core + \
                                      e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.repeat_times) as i:
                        ub_offset = i * 255 * mask
                        with tik_inst.if_scope(i < obj_int32_e_num_input_scalar.repeat_times - 1):
                            _tik_vadd_num_one(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, 255, mask,
                                              ub_offset)
                        with tik_inst.if_scope(i == obj_int32_e_num_input_scalar.repeat_times - 1):
                            _tik_vadd_num_one(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask, ub_offset)
                    # mov output data ub2gm
                output_offset_gm = block_index * obj_common_scalar.num_segments_front_core + \
                                   e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                output_offset_ub = 0
                output_n_burst = 1
                output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                               output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)

            with tik_inst.if_scope(e_mov_index_gm2ub == obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                with tik_inst.for_range(0, obj_int32_e_num_input_scalar.repeat_times_last_part) as i:
                    ub_offset = i * 255 * mask
                    with tik_inst.if_scope(i < obj_int32_e_num_input_scalar.repeat_times_last_part - 1):
                        _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub, 255, mask, ub_offset)
                    with tik_inst.if_scope(i == obj_int32_e_num_input_scalar.repeat_times_last_part - 1):
                        _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                                 obj_int32_e_num_input_scalar.repeat_time_last_part, mask, ub_offset)

                # last part e
                with tik_inst.for_range(0, obj_int32_ids_input_scalar.size) as segment_index_front_core:
                    # num_segments divide by core
                    # init output_ub
                    input_offset_gm = segment_index_front_core * obj_int32_e_num_input_scalar.e_num + \
                                      block_index * obj_common_scalar.num_segments_front_core + \
                                      e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.repeat_times_last_part) as i:
                        ub_offset = i * 255 * mask
                        with tik_inst.if_scope(i < obj_int32_e_num_input_scalar.repeat_times_last_part - 1):
                            _tik_vadd_num_one(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, 255, mask,
                                              ub_offset)
                        with tik_inst.if_scope(i == obj_int32_e_num_input_scalar.repeat_times_last_part - 1):
                            _tik_vadd_num_one(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_last_part, mask, ub_offset)
                    # mov output data ub2gm
                output_offset_gm = block_index * obj_common_scalar.num_segments_front_core + \
                                   e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                output_offset_ub = 0
                output_n_burst = 1
                output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                               output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_mov_times_gm2ub_lastcore) as e_mov_index_gm2ub:
            # e divide by ub
            with tik_inst.if_scope(e_mov_index_gm2ub < obj_int32_e_num_input_scalar.e_mov_times_gm2ub_lastcore - 1):
                with tik_inst.for_range(0, obj_int32_e_num_input_scalar.repeat_times) as i:
                    with tik_inst.if_scope(i < obj_int32_e_num_input_scalar.repeat_times - 1):
                        ub_offset = i * 255 * mask
                        _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub, 255, mask, ub_offset)
                    with tik_inst.if_scope(i == obj_int32_e_num_input_scalar.repeat_times - 1):
                        _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                                 obj_int32_e_num_input_scalar.repeat_time_front_part, mask, ub_offset)
                # front part e
                with tik_inst.for_range(0, obj_int32_ids_input_scalar.size) as segment_index_last_core:
                    # num_segments divide by core
                    # init output_ub
                    input_offset_gm = segment_index_last_core * obj_int32_e_num_input_scalar.e_num + \
                                      block_index * obj_common_scalar.num_segments_front_core + \
                                      e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.repeat_times) as i:
                        ub_offset = i * 255 * mask
                        with tik_inst.if_scope(i < obj_int32_e_num_input_scalar.repeat_times - 1):
                            _tik_vadd_num_one(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, 255, mask,
                                              ub_offset)
                        with tik_inst.if_scope(i == obj_int32_e_num_input_scalar.repeat_times - 1):
                            _tik_vadd_num_one(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask, ub_offset)
                    # mov output data ub2gm
                output_offset_gm = block_index * obj_common_scalar.num_segments_front_core + \
                                   e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                output_offset_ub = 0
                output_n_burst = 1
                output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                               output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)

            with tik_inst.if_scope(e_mov_index_gm2ub == obj_int32_e_num_input_scalar.e_mov_times_gm2ub_lastcore - 1):
                with tik_inst.for_range(0, obj_int32_e_num_input_scalar.repeat_times_last_part_lastcore) as i:
                    ub_offset = i * 255 * mask
                    with tik_inst.if_scope(i < obj_int32_e_num_input_scalar.repeat_times_last_part_lastcore - 1):
                        _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub, 255, mask, ub_offset)
                    with tik_inst.if_scope(i == obj_int32_e_num_input_scalar.repeat_times_last_part_lastcore - 1):
                        _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                                 obj_int32_e_num_input_scalar.repeat_time_last_part_lastcore, mask,
                                                 ub_offset)
                # last part e
                with tik_inst.for_range(0, obj_int32_ids_input_scalar.size) as segment_index_last_core:
                    # num_segments divide by core
                    # init output_ub
                    # id in segment
                    input_offset_gm = segment_index_last_core * obj_int32_e_num_input_scalar.e_num + \
                                      block_index * obj_common_scalar.num_segments_front_core + \
                                      e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len_input_scalar_lastcore
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.repeat_times_last_part_lastcore) as i:
                        ub_offset = i * 255 * mask
                        with tik_inst.if_scope(i < obj_int32_e_num_input_scalar.repeat_times_last_part_lastcore - 1):
                            _tik_vadd_num_one(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub, 255, mask,
                                              ub_offset)
                        with tik_inst.if_scope(i == obj_int32_e_num_input_scalar.repeat_times_last_part_lastcore - 1):
                            _tik_vadd_num_one(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_last_part_lastcore, mask,
                                              ub_offset)
                    # mov output data ub2gm
                output_offset_gm = block_index * obj_common_scalar.num_segments_front_core + \
                                   e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                output_offset_ub = 0
                output_n_burst = 1
                output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len_input_scalar_lastcore
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                               output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)


def _tik_no_atomic_all_in_align(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                obj_int32_ids_input_scalar, obj_int32_e_num_input_scalar, dtype):
    """
    _tik_no_atomic_all_in_align

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar
    dtype: input dtype

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    output_id_scalar = obj_common_scalar.output_id_scalar
    mask = Constant.MASK_FP16 if dtype == Constant.DTYPE_FP16 else Constant.MASK_INT32
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        ids_offset_gm = 0
        ids_offset_ub = 0
        ids_n_burst = 1
        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                           ids_n_burst, ids_burst_len)
        input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, 0, 0, 1,
                                      input_burst_len)
        # front core
        with tik_inst.for_range(0, obj_common_scalar.num_segments_front_core) as \
                segment_index_front_core:
            # num_segments divide by core
            # init output_ub
            _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                     obj_int32_e_num_input_scalar.repeat_time_front_part, mask)

            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                # visit ids
                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                with tik_inst.if_scope(segment_index_front_core +
                                       block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                    output_id_scalar.set_as(1)
                    # id in segment
                    input_offset_ub = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask, input_offset_ub)
            with tik_inst.if_scope(output_id_scalar == 1):
                output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                    segment_index_front_core) * obj_int32_e_num_input_scalar.e_num
                output_offset_ub = 0
                output_n_burst = 1
                output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                               output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)

    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        ids_offset_gm = 0
        ids_offset_ub = 0
        ids_n_burst = 1
        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                           ids_n_burst, ids_burst_len)
        input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub, 0, 0, 1,
                                      input_burst_len)
        # last core
        with tik_inst.for_range(0, obj_common_scalar.num_segments_last_core) as segment_index_last_core:
            # num_segments divide by core
            # init output_ub
            _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                     obj_int32_e_num_input_scalar.repeat_time_front_part, mask)

            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                # visit ids
                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                with tik_inst.if_scope(segment_index_last_core +
                                       block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                    output_id_scalar.set_as(1)
                    # id in segment
                    input_offset_ub = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask, input_offset_ub)
            with tik_inst.if_scope(output_id_scalar == 1):
                output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                    segment_index_last_core) * obj_int32_e_num_input_scalar.e_num
                output_offset_ub = 0
                output_n_burst = 1
                output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                               output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)


def _tik_no_atomic_small_e_small_id(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                    obj_int32_ids_input_scalar, obj_int32_e_num_input_scalar, dtype):
    """
    _tik_no_atomic_small_e_small_id

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar
    dtype: input dtype

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    align_scalar = obj_int32_e_num_input_scalar.align_scalar
    mask = Constant.MASK_FP16 if dtype == Constant.DTYPE_FP16 else Constant.MASK_INT32
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        ids_offset_gm = 0
        ids_offset_ub = 0
        ids_n_burst = 1
        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                           ids_n_burst, ids_burst_len)
        # front core
        with tik_inst.for_range(0, obj_common_scalar.num_segments_front_core) as \
                segment_index_front_core:
            # num_segments divide by core
            # init output_ub
            _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                     obj_int32_e_num_input_scalar.repeat_time_front_part, mask)

            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                # visit ids
                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                with tik_inst.if_scope(segment_index_front_core +
                                       block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                    # id in segment
                    input_offset_gm = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
            output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                segment_index_front_core) * obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
            _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub, output_offset_gm,
                                           output_offset_ub, output_n_burst, output_burst_len)
            with tik_inst.if_scope(tik.all(obj_common_scalar.need_core_num > 1, align_scalar > 0)):
                align_ub = tik_inst.Tensor(dtype, (mask,), name="align_ub", scope=tik.scope_ubuf)
                output_offset_gm_one = output_offset_gm + output_burst_len * (mask // 8) - align_scalar
                with tik_inst.for_range(0, mask // 8) as num_i:
                    align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * (mask // 8) - align_scalar +
                                                                   num_i])
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub, output_offset_gm_one,
                                               output_offset_ub, output_n_burst, 1)

    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        ids_offset_gm = 0
        ids_offset_ub = 0
        ids_n_burst = 1
        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                           ids_n_burst, ids_burst_len)
        # last core
        with tik_inst.for_range(0, obj_common_scalar.num_segments_last_core) as segment_index_last_core:
            # num_segments divide by core
            # init output_ub
            _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                     obj_int32_e_num_input_scalar.repeat_time_front_part, mask)

            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                # visit ids
                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                with tik_inst.if_scope(segment_index_last_core +
                                       block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                    # id in segment
                    input_offset_gm = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                    input_offset_ub = 0
                    input_n_burst = 1
                    input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                  input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
            output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                segment_index_last_core) * obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
            _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub, output_offset_gm,
                                           output_offset_ub, output_n_burst, output_burst_len)
            with tik_inst.if_scope(tik.all(obj_common_scalar.need_core_num > 1, align_scalar > 0)):
                align_ub = tik_inst.Tensor(dtype, (mask,), name="align_ub", scope=tik.scope_ubuf)
                output_offset_gm_one = output_offset_gm + output_burst_len * (mask // 8) - align_scalar
                with tik_inst.for_range(0, mask // 8) as num_i:
                    align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * (mask // 8) - align_scalar +
                                                                   num_i])
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub, output_offset_gm_one,
                                               output_offset_ub, output_n_burst, 1)


def _tik_no_atomic_small_e_big_id(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                  obj_int32_ids_input_scalar, obj_int32_e_num_input_scalar, dtype):
    """
    _tik_no_atomic_small_e_big_id

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar
    dtype:input dtype

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    align_scalar = obj_int32_e_num_input_scalar.align_scalar
    mask = Constant.MASK_FP16 if dtype == Constant.DTYPE_FP16 else Constant.MASK_INT32
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0, obj_common_scalar.num_segments_front_core) as \
                segment_index_front_core:
            # num_segments divide by core
            # init output_ub
            _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                     obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
            with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as \
                    ids_mov_index_gm2ub:
                # ids divide by ub
                with tik_inst.if_scope(ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                    # ids front part
                    ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                    ids_offset_ub = 0
                    ids_n_burst = 1
                    ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                    _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                       ids_offset_ub, ids_n_burst, ids_burst_len)
                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                            ids_index_front_part:
                        # visit ids
                        id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                        with tik_inst.if_scope(
                                segment_index_front_core +
                                block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                            # id in segment
                            input_offset_gm = (ids_offset_gm + ids_index_front_part) * \
                                              obj_int32_e_num_input_scalar.e_num
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                    # ids last part
                    ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                    ids_offset_ub = 0
                    ids_n_burst = 1
                    ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                    _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                       ids_offset_ub, ids_n_burst, ids_burst_len)
                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                        # visit ids
                        id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                        with tik_inst.if_scope(
                                segment_index_front_core +
                                block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                            # id in segment
                            input_offset_gm = (ids_offset_gm + ids_index_last_part) * obj_int32_e_num_input_scalar.e_num
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.repeat_time_front_part, mask)

            output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                segment_index_front_core) * obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
            _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub, output_offset_gm,
                                           output_offset_ub, output_n_burst, output_burst_len)
            with tik_inst.if_scope(tik.all(obj_common_scalar.need_core_num > 1, align_scalar > 0)):
                align_ub = tik_inst.Tensor(dtype, (mask,), name="align_ub", scope=tik.scope_ubuf)
                output_offset_gm_one = output_offset_gm + output_burst_len * (mask // 8) - align_scalar
                with tik_inst.for_range(0, mask // 8) as num_i:
                    align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * (mask // 8) - align_scalar +
                                                                   num_i])
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub, output_offset_gm_one,
                                               output_offset_ub, output_n_burst, 1)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0, obj_common_scalar.num_segments_last_core) as segment_index_last_core:
            # num_segments divide by core
            # init output_ub
            _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                     obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
            with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as ids_mov_index_gm2ub:
                # ids divide by ub
                with tik_inst.if_scope(ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                    # ids front part
                    ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                    ids_offset_ub = 0
                    ids_n_burst = 1
                    ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                    _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                       ids_offset_ub, ids_n_burst, ids_burst_len)
                    with tik_inst.for_range(0,
                                            obj_int32_ids_input_scalar.ele_num_ub_front_part) as ids_index_front_part:
                        # visit ids
                        id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                        with tik_inst.if_scope(
                                segment_index_last_core +
                                block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                            # id in segment
                            input_offset_gm = (ids_offset_gm +
                                               ids_index_front_part) * obj_int32_e_num_input_scalar.e_num
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                    # ids last part
                    ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                    ids_offset_ub = 0
                    ids_n_burst = 1
                    ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                    _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                       ids_offset_ub, ids_n_burst, ids_burst_len)
                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                        # visit ids
                        id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                        with tik_inst.if_scope(
                                segment_index_last_core +
                                block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                            # id in segment
                            input_offset_gm = (ids_offset_gm + ids_index_last_part) * obj_int32_e_num_input_scalar.e_num
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
            output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                segment_index_last_core) * obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
            _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub, output_offset_gm,
                                           output_offset_ub, output_n_burst, output_burst_len)
            with tik_inst.if_scope(tik.all(obj_common_scalar.need_core_num > 1, align_scalar > 0)):
                align_ub = tik_inst.Tensor(dtype, (mask,), name="align_ub", scope=tik.scope_ubuf)
                output_offset_gm_one = output_offset_gm + output_burst_len * (mask // 8) - align_scalar
                with tik_inst.for_range(0, mask // 8) as num_i:
                    align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * (mask // 8) - align_scalar +
                                                                   num_i])
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub, output_offset_gm_one,
                                               output_offset_ub, output_n_burst, 1)


def _tik_no_atomic_big_e_small_id(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                  obj_int32_ids_input_scalar, obj_int32_e_num_input_scalar, dtype):
    """
    _tik_int32_add_big_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar
    dtype:input dtype

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    align_scalar = obj_int32_e_num_input_scalar.align_scalar
    mask = Constant.MASK_FP16 if dtype == Constant.DTYPE_FP16 else Constant.MASK_INT32
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        ids_offset_gm = 0
        ids_offset_ub = 0
        ids_n_burst = 1
        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                           ids_n_burst, ids_burst_len)
        # front core
        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_mov_times_gm2ub) as e_mov_index_gm2ub:
            # e divide by ub
            with tik_inst.if_scope(e_mov_index_gm2ub < obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # front part e
                with tik_inst.for_range(0, obj_common_scalar.num_segments_front_core) as segment_index_front_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.repeat_time_front_part, mask)

                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                        # visit ids
                        id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                        with tik_inst.if_scope(
                                segment_index_front_core +
                                block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                            # id in segment
                            input_offset_gm = ids_index_last_part * obj_int32_e_num_input_scalar.e_num + \
                                              e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                    # mov output data ub2gm
                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        segment_index_front_core) * obj_int32_e_num_input_scalar.e_num + \
                                       e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)

            with tik_inst.if_scope(e_mov_index_gm2ub == obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # last part e
                with tik_inst.for_range(0, obj_common_scalar.num_segments_front_core) as \
                        segment_index_front_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.repeat_time_last_part, mask)

                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                        # visit ids
                        id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                        with tik_inst.if_scope(
                                segment_index_front_core +
                                block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                            # id in segment
                            input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                              obj_int32_e_num_input_scalar.e_num + \
                                              e_mov_index_gm2ub * \
                                              obj_int32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.repeat_time_last_part, mask)
                    # mov output data ub2gm
                    output_offset_gm = (block_index *
                                        obj_common_scalar.
                                        num_segments_front_core +
                                        segment_index_front_core) * \
                                       obj_int32_e_num_input_scalar.e_num + \
                                       e_mov_index_gm2ub * \
                                       obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                    with tik_inst.if_scope(tik.all(obj_common_scalar.need_core_num > 1, align_scalar > 0)):
                        align_ub = tik_inst.Tensor(dtype, (mask,), name="align_ub", scope=tik.scope_ubuf)
                        output_offset_gm_one = output_offset_gm + output_burst_len * (mask // 8) - align_scalar
                        with tik_inst.for_range(0, mask // 8) as num_i:
                            align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * (mask // 8) -
                                                                           align_scalar + num_i])
                        _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub,
                                                       output_offset_gm_one, output_offset_ub, output_n_burst, 1)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        ids_offset_gm = 0
        ids_offset_ub = 0
        ids_n_burst = 1
        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                           ids_n_burst, ids_burst_len)
        # last core
        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_mov_times_gm2ub) as e_mov_index_gm2ub:
            # e divide by ub
            with tik_inst.if_scope(e_mov_index_gm2ub < obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # front part e
                with tik_inst.for_range(0, obj_common_scalar.num_segments_last_core) as segment_index_last_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.repeat_time_front_part, mask)

                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                        # visit ids
                        id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                        with tik_inst.if_scope(
                                segment_index_last_core +
                                block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                            # id in segment
                            input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                              obj_int32_e_num_input_scalar.e_num + \
                                              e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                    # mov output data ub2gm
                    output_offset_gm = (block_index *
                                        obj_common_scalar.
                                        num_segments_front_core +
                                        segment_index_last_core) * \
                                       obj_int32_e_num_input_scalar.e_num + \
                                       e_mov_index_gm2ub * \
                                       obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)

            with tik_inst.if_scope(e_mov_index_gm2ub == obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # last part e
                with tik_inst.for_range(0, obj_common_scalar.num_segments_last_core) as \
                        segment_index_last_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.repeat_time_last_part, mask)

                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                        # visit ids
                        id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                        with tik_inst.if_scope(
                                segment_index_last_core +
                                block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                            # id in segment
                            input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                              obj_int32_e_num_input_scalar.e_num + \
                                              e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                          input_offset_gm, input_offset_ub, input_n_burst,
                                                          input_burst_len)
                            _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.repeat_time_last_part, mask)
                    # mov output data ub2gm
                    output_offset_gm = (block_index *
                                        obj_common_scalar.
                                        num_segments_front_core +
                                        segment_index_last_core) * \
                                       obj_int32_e_num_input_scalar.e_num + \
                                       e_mov_index_gm2ub * \
                                       obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                    with tik_inst.if_scope(tik.all(obj_common_scalar.need_core_num > 1, align_scalar > 0)):
                        align_ub = tik_inst.Tensor(dtype, (mask,), name="align_ub", scope=tik.scope_ubuf)
                        output_offset_gm_one = output_offset_gm + output_burst_len * (mask // 8) - align_scalar
                        with tik_inst.for_range(0, mask // 8) as num_i:
                            align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * (mask // 8) -
                                                                           align_scalar + num_i])
                        _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub,
                                                       output_offset_gm_one, output_offset_ub, output_n_burst, 1)


def _tik_no_atomic_big_e_big_id(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                obj_int32_ids_input_scalar, obj_int32_e_num_input_scalar, dtype):
    """
    _tik_no_atomic_big_e_big_id

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar
    dtype:input dtype

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    align_scalar = obj_int32_e_num_input_scalar.align_scalar
    mask = Constant.MASK_FP16 if dtype == Constant.DTYPE_FP16 else Constant.MASK_INT32
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_mov_times_gm2ub) as e_mov_index_gm2ub:
            # e divide by ub
            with tik_inst.if_scope(e_mov_index_gm2ub < obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # front part e
                with tik_inst.for_range(0,
                                        obj_common_scalar.num_segments_front_core) as \
                        segment_index_front_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                    with tik_inst.for_range(0,
                                            obj_int32_ids_input_scalar.mov_times_gm2ub) as \
                            ids_mov_index_gm2ub:
                        # ids divide by ub
                        with tik_inst.if_scope(ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                            # ids front part
                            ids_offset_gm = ids_mov_index_gm2ub * \
                                            obj_int32_ids_input_scalar.ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                            _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                               ids_offset_ub, ids_n_burst, ids_burst_len)
                            with tik_inst.for_range(0,
                                                    obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                    ids_index_front_part:
                                # visit ids
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                                with tik_inst.if_scope(
                                        segment_index_front_core +
                                        block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm + ids_index_front_part) * \
                                                      obj_int32_e_num_input_scalar.e_num + \
                                                      e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                  obj_ub_tensor.input_ub, input_offset_gm,
                                                                  input_offset_ub, input_n_burst, input_burst_len)
                                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                        with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                            # ids last part
                            ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                            _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                               ids_offset_ub, ids_n_burst, ids_burst_len)
                            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                    ids_index_last_part:
                                # visit ids
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                                with tik_inst.if_scope(
                                        segment_index_front_core +
                                        block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                                      obj_int32_e_num_input_scalar.e_num + \
                                                      e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                  obj_ub_tensor.input_ub, input_offset_gm,
                                                                  input_offset_ub, input_n_burst, input_burst_len)
                                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                    # mov output data ub2gm
                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        segment_index_front_core) * obj_int32_e_num_input_scalar.e_num + \
                                       e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)

            with tik_inst.if_scope(e_mov_index_gm2ub == obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # last part e
                with tik_inst.for_range(0, obj_common_scalar.num_segments_front_core) as segment_index_front_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.repeat_time_last_part, mask)
                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as ids_mov_index_gm2ub:
                        # ids divide by ub
                        with tik_inst.if_scope(ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                            # ids front part
                            ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                            _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                               ids_offset_ub, ids_n_burst, ids_burst_len)
                            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                    ids_index_front_part:
                                # visit ids
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                                with tik_inst.if_scope(
                                        segment_index_front_core +
                                        block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm + ids_index_front_part) * \
                                                      obj_int32_e_num_input_scalar.e_num + \
                                                      e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_last_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                  obj_ub_tensor.input_ub, input_offset_gm,
                                                                  input_offset_ub, input_n_burst, input_burst_len)
                                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_last_part, mask)
                        with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                            # ids last part
                            ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                            _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                               ids_offset_ub, ids_n_burst, ids_burst_len)
                            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                    ids_index_last_part:
                                # visit ids
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                                with tik_inst.if_scope(
                                        segment_index_front_core +
                                        block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm +
                                                       ids_index_last_part) * \
                                                      obj_int32_e_num_input_scalar.e_num + \
                                                      e_mov_index_gm2ub * \
                                                      obj_int32_e_num_input_scalar.e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                        obj_int32_e_num_input_scalar.e_gm2ub_last_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                  obj_ub_tensor.input_ub, input_offset_gm,
                                                                  input_offset_ub, input_n_burst, input_burst_len)
                                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_last_part, mask)
                    # mov output data ub2gm
                    output_offset_gm = (block_index *
                                        obj_common_scalar.
                                        num_segments_front_core +
                                        segment_index_front_core) * \
                                       obj_int32_e_num_input_scalar.e_num + \
                                       e_mov_index_gm2ub * \
                                       obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                    with tik_inst.if_scope(tik.all(obj_common_scalar.need_core_num > 1, align_scalar > 0)):
                        align_ub = tik_inst.Tensor(dtype, (mask,), name="align_ub", scope=tik.scope_ubuf)
                        output_offset_gm_one = output_offset_gm + output_burst_len * (mask // 8) - align_scalar
                        with tik_inst.for_range(0, mask // 8) as num_i:
                            align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * (mask // 8) -
                                                                           align_scalar + num_i])
                        _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub,
                                                       output_offset_gm_one, output_offset_ub, output_n_burst, 1)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_mov_times_gm2ub) as e_mov_index_gm2ub:
            # e divide by ub
            with tik_inst.if_scope(e_mov_index_gm2ub < obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # front part e
                with tik_inst.for_range(0, obj_common_scalar.num_segments_last_core) as segment_index_last_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as ids_mov_index_gm2ub:
                        # ids divide by ub
                        with tik_inst.if_scope(ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                            # ids front part
                            ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                            _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                               ids_offset_ub, ids_n_burst, ids_burst_len)
                            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                    ids_index_front_part:
                                # visit ids
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                                with tik_inst.if_scope(
                                        segment_index_last_core +
                                        block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm + ids_index_front_part) * \
                                                      obj_int32_e_num_input_scalar.e_num + e_mov_index_gm2ub * \
                                                      obj_int32_e_num_input_scalar.e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                  obj_ub_tensor.input_ub, input_offset_gm,
                                                                  input_offset_ub, input_n_burst, input_burst_len)
                                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                        with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                            # ids last part
                            ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                            _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                               ids_offset_ub, ids_n_burst, ids_burst_len)
                            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                    ids_index_last_part:
                                # visit ids
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                                with tik_inst.if_scope(
                                        segment_index_last_core +
                                        block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                                      obj_int32_e_num_input_scalar.e_num + \
                                                      e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                  obj_ub_tensor.input_ub, input_offset_gm,
                                                                  input_offset_ub, input_n_burst, input_burst_len)
                                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_front_part, mask)
                    # mov output data ub2gm
                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        segment_index_last_core) * obj_int32_e_num_input_scalar.e_num + \
                                       e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)

            with tik_inst.if_scope(e_mov_index_gm2ub == obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # last part e
                with tik_inst.for_range(0, obj_common_scalar.num_segments_last_core) as segment_index_last_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst, obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.repeat_time_last_part, mask)
                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as ids_mov_index_gm2ub:
                        # ids divide by ub
                        with tik_inst.if_scope(ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                            # ids front part
                            ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                            _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                               ids_offset_ub, ids_n_burst, ids_burst_len)
                            with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                    ids_index_front_part:
                                # visit ids
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                                with tik_inst.if_scope(
                                        segment_index_last_core +
                                        block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm +
                                                       ids_index_front_part) * \
                                                      obj_int32_e_num_input_scalar.e_num + \
                                                      e_mov_index_gm2ub * \
                                                      obj_int32_e_num_input_scalar.e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                        obj_int32_e_num_input_scalar.e_gm2ub_last_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                  obj_ub_tensor.input_ub, input_offset_gm,
                                                                  input_offset_ub, input_n_burst, input_burst_len)
                                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_last_part, mask)
                        with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                            # ids last part
                            ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                            _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                               ids_offset_ub, ids_n_burst, ids_burst_len)
                            with tik_inst.for_range(0,
                                                    obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                    ids_index_last_part:
                                # visit ids
                                id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                                with tik_inst.if_scope(
                                        segment_index_last_core +
                                        block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                                      obj_int32_e_num_input_scalar.e_num + \
                                                      e_mov_index_gm2ub * \
                                                      obj_int32_e_num_input_scalar.e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = obj_int32_e_num_input_scalar.e_gm2ub_last_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                  obj_ub_tensor.input_ub, input_offset_gm,
                                                                  input_offset_ub, input_n_burst, input_burst_len)
                                    _tik_vadd(tik_inst, obj_ub_tensor.input_ub, obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.repeat_time_last_part, mask)
                    # mov output data ub2gm
                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        segment_index_last_core) * obj_int32_e_num_input_scalar.e_num + \
                                       e_mov_index_gm2ub * obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                    with tik_inst.if_scope(tik.all(obj_common_scalar.need_core_num > 1, align_scalar > 0)):
                        align_ub = tik_inst.Tensor(dtype, (mask,), name="align_ub", scope=tik.scope_ubuf)
                        output_offset_gm_one = output_offset_gm + output_burst_len * (mask // 8) - align_scalar
                        with tik_inst.for_range(0, mask // 8) as num_i:
                            align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * (mask // 8) -
                                                                           align_scalar + num_i])
                        _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub,
                                                       output_offset_gm_one, output_offset_ub, output_n_burst, 1)


def _tik_no_atomic_small_e_block_small_id(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                          obj_int32_ids_input_scalar, obj_int32_e_num_input_scalar, dtype):
    """
    _tik_no_atomic_small_e_block_small_id

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar
    dtype:input dtype

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    align_scalar = obj_int32_e_num_input_scalar.align_scalar
    align_scalar_last_core = obj_int32_e_num_input_scalar.align_scalar_last_core
    mask = Constant.MASK_FP16 if dtype == Constant.DTYPE_FP16 else Constant.MASK_INT32
    if dtype == Constant.DTYPE_FP16:
        ele_num_one_block = Constant.ELE_NUM_ONE_BLOCK_FP16
    else:
        ele_num_one_block = Constant.ELE_NUM_ONE_BLOCK_INT32
    mask_ub = tik_inst.Tensor(dtype, (mask,), name="mask_ub", scope=tik.scope_ubuf)
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        ids_offset_gm = 0
        ids_offset_ub = 0
        ids_n_burst = 1
        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                           ids_n_burst, ids_burst_len)
        # front core
        with tik_inst.if_scope(obj_int32_e_num_input_scalar.e_max_num_time > 1):
            with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_max_num_time) as index_e_max_num_time:
                with tik_inst.if_scope(index_e_max_num_time < obj_int32_e_num_input_scalar.e_max_num_time - 1):
                    with tik_inst.for_range(0,
                                            obj_int32_e_num_input_scalar.front_num_segment) as segment_index_last_core:
                        # num_segments divide by core
                        # init output_ub
                        _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)

                        with tik_inst.for_range(0,
                                                obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                            # visit ids
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                            with tik_inst.if_scope(
                                    index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment +
                                    segment_index_last_core +
                                    block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                # id in segment
                                input_offset_gm = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                                input_offset_ub = 0
                                input_n_burst = 1
                                input_burst_len = 1
                                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                              input_offset_gm, input_offset_ub, input_n_burst,
                                                              input_burst_len)
                                _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                            obj_ub_tensor.output_ub[segment_index_last_core * obj_int32_e_num_input_scalar.e_num +
                                                    j].set_as(mask_ub[j])

                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment) * \
                                       obj_int32_e_num_input_scalar.e_num
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                with tik_inst.if_scope(index_e_max_num_time == obj_int32_e_num_input_scalar.e_max_num_time - 1):
                    with tik_inst.for_range(0,
                                            obj_int32_e_num_input_scalar.front_num_segment_last) as \
                            segment_index_last_core:
                        # num_segments divide by core
                        # init output_ub
                        _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)

                        with tik_inst.for_range(0,
                                                obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                            # visit ids
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                            with tik_inst.if_scope(
                                    index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment_last +
                                    segment_index_last_core +
                                    block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                # id in segment
                                input_offset_gm = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                                input_offset_ub = 0
                                input_n_burst = 1
                                input_burst_len = 1
                                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                              input_offset_gm, input_offset_ub, input_n_burst,
                                                              input_burst_len)
                                _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                            obj_ub_tensor.output_ub[segment_index_last_core * obj_int32_e_num_input_scalar.e_num +
                                                    j].set_as(mask_ub[j])

                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment_last) * \
                                       obj_int32_e_num_input_scalar.e_num
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                    with tik_inst.if_scope(align_scalar > 0):
                        align_ub = tik_inst.Tensor(dtype, (ele_num_one_block,), name="align_ub", scope=tik.scope_ubuf)
                        output_offset_gm_one = output_offset_gm + output_burst_len * ele_num_one_block - align_scalar
                        with tik_inst.for_range(0, ele_num_one_block) as num_i:
                            align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * ele_num_one_block -
                                                                           align_scalar + num_i])
                        _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub,
                                                       output_offset_gm_one, output_offset_ub, output_n_burst, 1)

        with tik_inst.else_scope():
            with tik_inst.for_range(0, obj_common_scalar.num_segments_front_core) as segment_index_front_core:
                # num_segments divide by core
                # init output_ub
                _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)

                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                    # visit ids
                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                    with tik_inst.if_scope(segment_index_front_core +
                                           block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                        # id in segment
                        input_offset_gm = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = 1
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                    obj_ub_tensor.output_ub[segment_index_front_core * obj_int32_e_num_input_scalar.e_num + j].set_as(
                        mask_ub[j])

            output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core) * \
                               obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
            _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub, output_offset_gm,
                                           output_offset_ub, output_n_burst, output_burst_len)

    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        ids_offset_gm = 0
        ids_offset_ub = 0
        ids_n_burst = 1
        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm, ids_offset_ub,
                           ids_n_burst, ids_burst_len)
        with tik_inst.if_scope(obj_int32_e_num_input_scalar.e_max_num_time_last_core > 1):
            with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_max_num_time_last_core) as index_e_max_num_time:
                with tik_inst.if_scope(index_e_max_num_time < obj_int32_e_num_input_scalar.e_max_num_time - 1):
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.front_num_segment_lastcore) as \
                            segment_index_last_core:
                        # num_segments divide by core
                        # init output_ub
                        _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)

                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                ids_index_last_part:
                            # visit ids
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                            with tik_inst.if_scope(
                                    index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment_lastcore +
                                    segment_index_last_core +
                                    block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                # id in segment
                                input_offset_gm = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                                input_offset_ub = 0
                                input_n_burst = 1
                                input_burst_len = 1
                                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                              input_offset_gm, input_offset_ub, input_n_burst,
                                                              input_burst_len)
                                _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                            obj_ub_tensor.output_ub[segment_index_last_core * obj_int32_e_num_input_scalar.e_num +
                                                    j].set_as(mask_ub[j])

                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        index_e_max_num_time *
                                        obj_int32_e_num_input_scalar.front_num_segment_lastcore) * \
                                       obj_int32_e_num_input_scalar.e_num
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_bust_len_small_e_last_core
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)

                with tik_inst.if_scope(index_e_max_num_time == obj_int32_e_num_input_scalar.e_max_num_time_last_core -
                                       1):
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.front_num_segment_last_lastcore) as \
                            segment_index_last_core:
                        # num_segments divide by core
                        # init output_ub
                        _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)

                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                ids_index_last_part:
                            # visit ids
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                            with tik_inst.if_scope(
                                    index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment_lastcore +
                                    segment_index_last_core +
                                    block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                # id in segment
                                input_offset_gm = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                                input_offset_ub = 0
                                input_n_burst = 1
                                input_burst_len = 1
                                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                              input_offset_gm, input_offset_ub, input_n_burst,
                                                              input_burst_len)
                                _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                            obj_ub_tensor.output_ub[segment_index_last_core * obj_int32_e_num_input_scalar.e_num +
                                                    j].set_as(mask_ub[j])

                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        index_e_max_num_time *
                                        obj_int32_e_num_input_scalar.front_num_segment_lastcore) * \
                                       obj_int32_e_num_input_scalar.e_num
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len_input_scalar_lastcore
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                    with tik_inst.if_scope(align_scalar_last_core > 0):
                        align_ub = tik_inst.Tensor(dtype, (ele_num_one_block,), name="align_ub", scope=tik.scope_ubuf)
                        output_offset_gm_one = output_offset_gm + output_burst_len * ele_num_one_block - \
                                               align_scalar_last_core
                        with tik_inst.for_range(0, ele_num_one_block) as num_i:
                            align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * ele_num_one_block -
                                                                           align_scalar_last_core + num_i])
                        _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub,
                                                       output_offset_gm_one, output_offset_ub, output_n_burst, 1)
        with tik_inst.else_scope():
            # last core
            with tik_inst.for_range(0, obj_common_scalar.num_segments_last_core) as segment_index_last_core:
                # num_segments divide by core
                # init output_ub
                _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)

                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as ids_index_last_part:
                    # visit ids
                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                    with tik_inst.if_scope(segment_index_last_core +
                                           block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                        # id in segment
                        input_offset_gm = ids_index_last_part * obj_int32_e_num_input_scalar.e_num
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = 1
                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                      input_offset_gm, input_offset_ub, input_n_burst, input_burst_len)
                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                        obj_ub_tensor.output_ub[segment_index_last_core * obj_int32_e_num_input_scalar.e_num +
                                                j].set_as(mask_ub[j])

            output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core) * \
                               obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len_input_scalar_lastcore
            _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub, output_offset_gm,
                                           output_offset_ub, output_n_burst, output_burst_len)
            with tik_inst.if_scope(align_scalar_last_core > 0):
                align_ub = tik_inst.Tensor(dtype, (ele_num_one_block,), name="align_ub", scope=tik.scope_ubuf)
                output_offset_gm_one = output_offset_gm + output_burst_len * ele_num_one_block - align_scalar_last_core
                with tik_inst.for_range(0, ele_num_one_block) as num_i:
                    align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * ele_num_one_block -
                                                                   align_scalar_last_core + num_i])
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub, output_offset_gm_one,
                                               output_offset_ub, output_n_burst, 1)


def _tik_no_atomic_small_e_block_big_id(block_index, tik_inst, obj_gm_tensor, obj_ub_tensor, obj_common_scalar,
                                        obj_int32_ids_input_scalar, obj_int32_e_num_input_scalar, dtype):
    """
    _tik_no_atomic_small_e_block_big_id

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar
    dtype:input dtype

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    align_scalar = obj_int32_e_num_input_scalar.align_scalar
    align_scalar_last_core = obj_int32_e_num_input_scalar.align_scalar_last_core
    mask = Constant.MASK_FP16 if dtype == Constant.DTYPE_FP16 else Constant.MASK_INT32
    if dtype == Constant.DTYPE_FP16:
        ele_num_one_block = Constant.ELE_NUM_ONE_BLOCK_FP16
    else:
        ele_num_one_block = Constant.ELE_NUM_ONE_BLOCK_INT32
    mask_ub = tik_inst.Tensor(dtype, (mask,), name="mask_ub", scope=tik.scope_ubuf)
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.if_scope(obj_int32_e_num_input_scalar.e_max_num_time > 1):
            with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_max_num_time) as index_e_max_num_time:
                with tik_inst.if_scope(index_e_max_num_time < obj_int32_e_num_input_scalar.e_max_num_time - 1):
                    with tik_inst.for_range(0,
                                            obj_int32_e_num_input_scalar.front_num_segment) as segment_index_front_core:
                        # num_segments divide by core
                        # init output_ub
                        _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as \
                                ids_mov_index_gm2ub:
                            # ids divide by ub
                            with tik_inst.if_scope(
                                    ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                                # ids front part
                                ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                                ids_offset_ub = 0
                                ids_n_burst = 1
                                ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                                   ids_offset_ub, ids_n_burst, ids_burst_len)
                                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                        ids_index_front_part:
                                    # visit ids
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                                    with tik_inst.if_scope(
                                            index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment +
                                            segment_index_front_core +
                                            block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                        # id in segment
                                        input_offset_gm = (ids_offset_gm + ids_index_front_part) * \
                                                          obj_int32_e_num_input_scalar.e_num
                                        input_offset_ub = 0
                                        input_n_burst = 1
                                        input_burst_len = 1
                                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                      obj_ub_tensor.input_ub, input_offset_gm,
                                                                      input_offset_ub, input_n_burst, input_burst_len)
                                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)

                            with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub -
                                                   1):
                                # ids last part
                                ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                                ids_offset_ub = 0
                                ids_n_burst = 1
                                ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                                   ids_offset_ub, ids_n_burst, ids_burst_len)
                                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                        ids_index_last_part:
                                    # visit ids
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                                    with tik_inst.if_scope(
                                            index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment +
                                            segment_index_front_core +
                                            block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                        # id in segment
                                        input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                                          obj_int32_e_num_input_scalar.e_num
                                        input_offset_ub = 0
                                        input_n_burst = 1
                                        input_burst_len = 1
                                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                      obj_ub_tensor.input_ub, input_offset_gm,
                                                                      input_offset_ub, input_n_burst, input_burst_len)
                                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)

                        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                            obj_ub_tensor.output_ub[segment_index_front_core * obj_int32_e_num_input_scalar.e_num +
                                                    j].set_as(mask_ub[j])

                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment) * \
                                       obj_int32_e_num_input_scalar.e_num
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                with tik_inst.if_scope(index_e_max_num_time == obj_int32_e_num_input_scalar.e_max_num_time - 1):
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.front_num_segment_last) as \
                            segment_index_front_core:
                        # num_segments divide by core
                        # init output_ub
                        _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as ids_mov_index_gm2ub:
                            # ids divide by ub
                            with tik_inst.if_scope(
                                    ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                                # ids front part
                                ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                                ids_offset_ub = 0
                                ids_n_burst = 1
                                ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                                   ids_offset_ub, ids_n_burst, ids_burst_len)
                                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                        ids_index_front_part:
                                    # visit ids
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                                    with tik_inst.if_scope(
                                            index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment +
                                            segment_index_front_core +
                                            block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                        # id in segment
                                        input_offset_gm = (ids_offset_gm + ids_index_front_part) * \
                                                          obj_int32_e_num_input_scalar.e_num
                                        input_offset_ub = 0
                                        input_n_burst = 1
                                        input_burst_len = 1
                                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                      obj_ub_tensor.input_ub, input_offset_gm,
                                                                      input_offset_ub, input_n_burst, input_burst_len)
                                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                            with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub -
                                                   1):
                                # ids last part
                                ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                                ids_offset_ub = 0
                                ids_n_burst = 1
                                ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                                   ids_offset_ub, ids_n_burst, ids_burst_len)
                                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                        ids_index_last_part:
                                    # visit ids
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                                    with tik_inst.if_scope(
                                            index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment +
                                            segment_index_front_core +
                                            block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                        # id in segment
                                        input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                                          obj_int32_e_num_input_scalar.e_num
                                        input_offset_ub = 0
                                        input_n_burst = 1
                                        input_burst_len = 1
                                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                      obj_ub_tensor.input_ub, input_offset_gm,
                                                                      input_offset_ub, input_n_burst, input_burst_len)
                                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                            obj_ub_tensor.output_ub[segment_index_front_core * obj_int32_e_num_input_scalar.e_num +
                                                    j].set_as(mask_ub[j])

                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        index_e_max_num_time * obj_int32_e_num_input_scalar.front_num_segment) * \
                                       obj_int32_e_num_input_scalar.e_num
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                    with tik_inst.if_scope(align_scalar > 0):
                        align_ub = tik_inst.Tensor(dtype, (ele_num_one_block,), name="align_ub", scope=tik.scope_ubuf)
                        output_offset_gm_one = output_offset_gm + output_burst_len * ele_num_one_block - align_scalar
                        with tik_inst.for_range(0, ele_num_one_block) as num_i:
                            align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * ele_num_one_block -
                                                                           align_scalar + num_i])
                        _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub,
                                                       output_offset_gm_one, output_offset_ub, output_n_burst, 1)

        with tik_inst.else_scope():
            with tik_inst.for_range(0, obj_common_scalar.num_segments_front_core) as segment_index_front_core:
                # num_segments divide by core
                # init output_ub
                _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)
                with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as ids_mov_index_gm2ub:
                    # ids divide by ub
                    with tik_inst.if_scope(ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                        # ids front part
                        ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                        ids_offset_ub = 0
                        ids_n_burst = 1
                        ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                           ids_offset_ub, ids_n_burst, ids_burst_len)
                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                ids_index_front_part:
                            # visit ids
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                            with tik_inst.if_scope(
                                    segment_index_front_core +
                                    block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                # id in segment
                                input_offset_gm = (ids_offset_gm + ids_index_front_part) * \
                                                  obj_int32_e_num_input_scalar.e_num
                                input_offset_ub = 0
                                input_n_burst = 1
                                input_burst_len = 1
                                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                              input_offset_gm, input_offset_ub, input_n_burst,
                                                              input_burst_len)
                                _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                    with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                        # ids last part
                        ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                        ids_offset_ub = 0
                        ids_n_burst = 1
                        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                           ids_offset_ub, ids_n_burst, ids_burst_len)
                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                ids_index_last_part:
                            # visit ids
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                            with tik_inst.if_scope(
                                    segment_index_front_core +
                                    block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                # id in segment
                                input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                                  obj_int32_e_num_input_scalar.e_num
                                input_offset_ub = 0
                                input_n_burst = 1
                                input_burst_len = 1
                                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                              input_offset_gm, input_offset_ub, input_n_burst,
                                                              input_burst_len)
                                _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                    obj_ub_tensor.output_ub[segment_index_front_core * obj_int32_e_num_input_scalar.e_num + j].set_as(
                        mask_ub[j])
            output_offset_gm = (block_index *
                                obj_common_scalar.num_segments_front_core) * obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_burst_len
            _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub, output_offset_gm,
                                           output_offset_ub, output_n_burst, output_burst_len)
            with tik_inst.if_scope(align_scalar > 0):
                align_ub = tik_inst.Tensor(dtype, (ele_num_one_block,), name="align_ub", scope=tik.scope_ubuf)
                output_offset_gm_one = output_offset_gm + output_burst_len * ele_num_one_block - align_scalar
                with tik_inst.for_range(0, ele_num_one_block) as num_i:
                    align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * ele_num_one_block - align_scalar +
                                                                   num_i])
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub, output_offset_gm_one,
                                               output_offset_ub, output_n_burst, 1)

    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.if_scope(obj_int32_e_num_input_scalar.e_max_num_time_last_core > 1):
            with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_max_num_time_last_core) as index_e_max_num_time:
                with tik_inst.if_scope(
                        index_e_max_num_time < obj_int32_e_num_input_scalar.e_max_num_time_last_core - 1):
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.front_num_segment_lastcore) as \
                            segment_index_last_core:
                        # num_segments divide by core
                        # init output_ub
                        _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as ids_mov_index_gm2ub:
                            # ids divide by ub
                            with tik_inst.if_scope(
                                    ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                                # ids front part
                                ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                                ids_offset_ub = 0
                                ids_n_burst = 1
                                ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                                   ids_offset_ub, ids_n_burst, ids_burst_len)
                                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                        ids_index_front_part:
                                    # visit ids
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                                    with tik_inst.if_scope(
                                            obj_int32_e_num_input_scalar.front_num_segment_lastcore *
                                            index_e_max_num_time + segment_index_last_core +
                                            block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                        # id in segment
                                        input_offset_gm = (ids_offset_gm +
                                                           ids_index_front_part) * obj_int32_e_num_input_scalar.e_num
                                        input_offset_ub = 0
                                        input_n_burst = 1
                                        input_burst_len = 1
                                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                      obj_ub_tensor.input_ub, input_offset_gm,
                                                                      input_offset_ub, input_n_burst, input_burst_len)
                                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                            with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub -
                                                   1):
                                # ids last part
                                ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                                ids_offset_ub = 0
                                ids_n_burst = 1
                                ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                                   ids_offset_ub, ids_n_burst, ids_burst_len)
                                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                        ids_index_last_part:
                                    # visit ids
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                                    with tik_inst.if_scope(
                                            index_e_max_num_time *
                                            obj_int32_e_num_input_scalar.front_num_segment_lastcore +
                                            segment_index_last_core +
                                            block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                        # id in segment
                                        input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                                          obj_int32_e_num_input_scalar.e_num
                                        input_offset_ub = 0
                                        input_n_burst = 1
                                        input_burst_len = 1
                                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                      obj_ub_tensor.input_ub, input_offset_gm,
                                                                      input_offset_ub, input_n_burst, input_burst_len)
                                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                            obj_ub_tensor.output_ub[segment_index_last_core * obj_int32_e_num_input_scalar.e_num +
                                                    j].set_as(mask_ub[j])

                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        index_e_max_num_time *
                                        obj_int32_e_num_input_scalar.front_num_segment_lastcore) * \
                                       obj_int32_e_num_input_scalar.e_num
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_bust_len_small_e_last_core
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)

                with tik_inst.if_scope(index_e_max_num_time == obj_int32_e_num_input_scalar.e_max_num_time - 1):
                    with tik_inst.for_range(0, obj_int32_e_num_input_scalar.front_num_segment_last_lastcore) as \
                            segment_index_last_core:
                        # num_segments divide by core
                        # init output_ub
                        _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as ids_mov_index_gm2ub:
                            # ids divide by ub
                            with tik_inst.if_scope(
                                    ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                                # ids front part
                                ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                                ids_offset_ub = 0
                                ids_n_burst = 1
                                ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                                   ids_offset_ub, ids_n_burst, ids_burst_len)
                                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                        ids_index_front_part:
                                    # visit ids
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                                    with tik_inst.if_scope(
                                            obj_int32_e_num_input_scalar.front_num_segment_lastcore *
                                            index_e_max_num_time + segment_index_last_core +
                                            block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                        # id in segment
                                        input_offset_gm = (ids_offset_gm +
                                                           ids_index_front_part) * obj_int32_e_num_input_scalar.e_num
                                        input_offset_ub = 0
                                        input_n_burst = 1
                                        input_burst_len = 1
                                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                      obj_ub_tensor.input_ub, input_offset_gm,
                                                                      input_offset_ub, input_n_burst, input_burst_len)
                                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                            with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub -
                                                   1):
                                # ids last part
                                ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                                ids_offset_ub = 0
                                ids_n_burst = 1
                                ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                                _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                                   ids_offset_ub, ids_n_burst, ids_burst_len)
                                with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                        ids_index_last_part:
                                    # visit ids
                                    id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                                    with tik_inst.if_scope(
                                            index_e_max_num_time *
                                            obj_int32_e_num_input_scalar.front_num_segment_lastcore +
                                            segment_index_last_core +
                                            block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                        # id in segment
                                        input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                                          obj_int32_e_num_input_scalar.e_num
                                        input_offset_ub = 0
                                        input_n_burst = 1
                                        input_burst_len = 1
                                        _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm,
                                                                      obj_ub_tensor.input_ub, input_offset_gm,
                                                                      input_offset_ub, input_n_burst, input_burst_len)
                                        _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                            obj_ub_tensor.output_ub[segment_index_last_core * obj_int32_e_num_input_scalar.e_num +
                                                    j].set_as(mask_ub[j])

                    output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core +
                                        index_e_max_num_time *
                                        obj_int32_e_num_input_scalar.front_num_segment_lastcore) * \
                                       obj_int32_e_num_input_scalar.e_num
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_last_burst_len_input_scalar_lastcore
                    _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub,
                                                   output_offset_gm, output_offset_ub, output_n_burst, output_burst_len)
                    with tik_inst.if_scope(align_scalar_last_core > 0):
                        align_ub = tik_inst.Tensor(dtype, (mask,), name="align_ub", scope=tik.scope_ubuf)
                        output_offset_gm_one = output_offset_gm + output_burst_len * ele_num_one_block - align_scalar
                        with tik_inst.for_range(0, ele_num_one_block) as num_i:
                            align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * ele_num_one_block -
                                                                           align_scalar_last_core + num_i])
                        _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub,
                                                       output_offset_gm_one, output_offset_ub, output_n_burst, 1)

        with tik_inst.else_scope():
            with tik_inst.for_range(0, obj_common_scalar.num_segments_last_core) as segment_index_last_core:
                # num_segments divide by core
                # init output_ub
                _tik_init_ub_tensor_once(tik_inst, mask_ub, 1, mask)
                with tik_inst.for_range(0, obj_int32_ids_input_scalar.mov_times_gm2ub) as ids_mov_index_gm2ub:
                    # ids divide by ub
                    with tik_inst.if_scope(ids_mov_index_gm2ub < obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                        # ids front part
                        ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                        ids_offset_ub = 0
                        ids_n_burst = 1
                        ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                           ids_offset_ub, ids_n_burst, ids_burst_len)
                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                                ids_index_front_part:
                            # visit ids
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_front_part])
                            with tik_inst.if_scope(
                                    segment_index_last_core +
                                    block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                # id in segment
                                input_offset_gm = (ids_offset_gm +
                                                   ids_index_front_part) * obj_int32_e_num_input_scalar.e_num
                                input_offset_ub = 0
                                input_n_burst = 1
                                input_burst_len = 1
                                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                              input_offset_gm, input_offset_ub, input_n_burst,
                                                              input_burst_len)
                                _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                    with tik_inst.if_scope(ids_mov_index_gm2ub == obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                        # ids last part
                        ids_offset_gm = ids_mov_index_gm2ub * obj_int32_ids_input_scalar.ele_num_ub_front_part
                        ids_offset_ub = 0
                        ids_n_burst = 1
                        ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                        _tik_mov_ids_gm2ub(tik_inst, obj_gm_tensor.ids_gm, obj_ub_tensor.ids_ub, ids_offset_gm,
                                           ids_offset_ub, ids_n_burst, ids_burst_len)
                        with tik_inst.for_range(0, obj_int32_ids_input_scalar.ele_num_ub_last_part) as \
                                ids_index_last_part:
                            # visit ids
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[ids_index_last_part])
                            with tik_inst.if_scope(
                                    segment_index_last_core +
                                    block_index * obj_common_scalar.num_segments_front_core == id_val_scalar):
                                # id in segment
                                input_offset_gm = (ids_offset_gm + ids_index_last_part) * \
                                                  obj_int32_e_num_input_scalar.e_num
                                input_offset_ub = 0
                                input_n_burst = 1
                                input_burst_len = 1
                                _tik_mov_input_gm2ub_continue(tik_inst, obj_gm_tensor.input_gm, obj_ub_tensor.input_ub,
                                                              input_offset_gm, input_offset_ub, input_n_burst,
                                                              input_burst_len)
                                _tik_vadd(tik_inst, obj_ub_tensor.input_ub, mask_ub, 1, mask)
                with tik_inst.for_range(0, obj_int32_e_num_input_scalar.e_num) as j:
                    obj_ub_tensor.output_ub[segment_index_last_core * obj_int32_e_num_input_scalar.e_num + j].set_as(
                        mask_ub[j])
            output_offset_gm = (block_index * obj_common_scalar.num_segments_front_core) * \
                               obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.e_ub2gm_front_bust_len_small_e_last_core
            _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, obj_ub_tensor.output_ub, output_offset_gm,
                                           output_offset_ub, output_n_burst, output_burst_len)
            with tik_inst.if_scope(align_scalar_last_core):
                align_ub = tik_inst.Tensor(dtype, (ele_num_one_block,), name="align_ub", scope=tik.scope_ubuf)
                output_offset_gm_one = output_offset_gm + output_burst_len * ele_num_one_block - align_scalar
                with tik_inst.for_range(0, ele_num_one_block) as num_i:
                    align_ub[num_i].set_as(obj_ub_tensor.output_ub[output_burst_len * ele_num_one_block - align_scalar +
                                                                   num_i])
                _tik_mov_output_ub2gm_continue(tik_inst, obj_gm_tensor.output_gm, align_ub, output_offset_gm_one,
                                               output_offset_ub, output_n_burst, 1)


def unsorted_segment_sum_no_atomic(x_dict,
                                   segment_ids_dict,
                                   num_segments_dict,
                                   y_dict,
                                   kernel_name="UnsortedSegmentSumNoAtomoic",
                                   opname="unsorted_segment_sum"):
    """
    unsorted_segment_sum_no_atomic entry interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    segment_ids_dict: segment_ids shape, dtype and range
    num_segments_dict: num_segments shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of UnsortedSegmentSumNoAtomoic op

    Returns
    -------
    compile info
    """

    obj = UnsortedSegmentSumNoAtomoic(x_dict, segment_ids_dict, num_segments_dict, y_dict, kernel_name,
                                      opname="unsorted_segment_sum")
    obj.unsorted_segment_sum()
