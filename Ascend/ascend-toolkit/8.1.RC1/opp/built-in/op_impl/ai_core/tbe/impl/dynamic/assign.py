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
assign.py
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_tik_comm_func import ceil_div
from impl.util.util_common import is_unknown
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max int32
    MAX_INT32 = 2 ** 31 - 1
    # tiling param num
    TILING_ARG_NUM = 16
    BYTE_LEN = 8


# 'pylint: disable=too-many-instance-attributes,invalid-name
class Assign:
    """
    Class for Dynamic shape operator Assign
    """

    def __init__(self, ref, value, output, kernel_name):
        # reserved ub size
        reserved_ub_size = 8 * 1024
        self.tik_instance = tik.Tik(tik.Dprofile)
        self.ref_dtype = ref.get("dtype").lower()
        self.ref_dtype = "int8" if self.ref_dtype == "bool" else self.ref_dtype
        self.value_dtype = value.get("dtype").lower()
        self.value_dtype = "int8" if self.value_dtype  == "bool" else self.value_dtype

        # check dtype
        para_check.check_dtype(self.ref_dtype,
                               ("float16", "float32", "int8", "int32", "int64", "uint8",
                                "int16", "uint16", "uint32", "uint64", "bfloat16"), param_name="ref")
        para_check.check_dtype(self.value_dtype,
                               ("float16", "float32", "int8", "int32", "int64", "uint8",
                                "int16", "uint16", "uint32", "uint64", "bfloat16"), param_name="value")
        if self.ref_dtype != self.value_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal("Assign", "ref", "value",
                                                                  self.ref_dtype, self.value_dtype)
        self.kernel_name = kernel_name

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - reserved_ub_size)
        self.max_burst_len = self.ub_size_bytes // (2 * 32)  # 2 means double buffer, 32 means one burst of UB is 32B
        self.running_core_num = self.tik_instance.Scalar(
            dtype="int64", name="running_core_num", init_value=self.ai_core_num)

        if self.ref_dtype in ("int8", "uint8"):
            self.ele_per_block = 32
        elif self.ref_dtype in ("float16", "int16", "uint16", "bfloat16"):
            self.ele_per_block = 16
        elif self.ref_dtype in ("float32", "int32", "uint32"):
            self.ele_per_block = 8
        else:
            self.ele_per_block = 4

        self.max_tensor_size = self.max_burst_len * self.ele_per_block

        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,), name="tiling_gm", \
        scope=tik.scope_gm)
        self.ref_gm = self.tik_instance.Tensor(self.ref_dtype, (Constant.MAX_INT32,), name="ref_gm", \
        scope=tik.scope_gm)
        self.value_gm = self.tik_instance.Tensor(self.value_dtype, (Constant.MAX_INT32,), name="value_gm", \
        scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.ref_dtype, (Constant.MAX_INT32,), name="out_gm", \
        scope=tik.scope_gm)

        self.tiling_ub = None
        self.value_ub = None
        self.out_ub = None

        self.core_used_num = self.tik_instance.Scalar("int64", name="core_used_num")
        self.block_per_core = self.tik_instance.Scalar("int64", name="block_per_core")
        self.tiling_key = self.tik_instance.Scalar("int64", name="tiling_key")
        self.ele_tail_core = self.tik_instance.Scalar("int64", name="ele_tail_core")

        self.dmp_flag = tbe_platform.api_check_support("tik.data_move_pad")
        self.b8_times = get_bit_len(self.ref_dtype) // Constant.BYTE_LEN

    def _tiling_args(self):
        """
        get runtime tiling parameters from tiling
        """
        # read tiling int64 scalar
        self.tiling_key.set_as(self.tiling_ub[0])
        self.core_used_num.set_as(self.tiling_ub[1])
        self.block_per_core.set_as(self.tiling_ub[2])
        self.ele_tail_core.set_as(self.tiling_ub[3])
        self.running_core_num.set_as(self.tiling_ub[4])

    def _broadcast_value(self, value_ub, dup_ele):
        self.tik_instance.data_move(value_ub, self.value_gm, 0, 1, 1, 0, 0)
        for i in range(1, self.ele_per_block):
            value_ub[i].set_as(value_ub[0])
        dup_blocks = ceil_div(dup_ele, self.ele_per_block)
        with self.tik_instance.if_scope(dup_blocks > 1):
            b16_per_block = 16
            dup_len = self.tik_instance.Scalar(init_value=(dup_blocks - 1) * b16_per_block)
            value_ub_b16 = value_ub.reinterpret_cast_to("float16")
            self.tik_instance.vcopy(dup_len, value_ub_b16[b16_per_block], value_ub_b16, 1, 1, 0, 8, 0, "counter")

    def _run_one_loop(self, gm_offset, burst_len, value_ub):
        if self.dmp_flag is False:
            burst = ceil_div(burst_len, self.ele_per_block)
            self.tik_instance.data_move(value_ub, self.value_gm[gm_offset], 0, 1, burst, 0, 0)
            self.tik_instance.data_move(self.out_gm[gm_offset], value_ub, 0, 1, burst, 0, 0)
        else:
            value_ub_b8 = value_ub.reinterpret_cast_to("int8")
            value_gm_b8 = self.value_gm[gm_offset].reinterpret_cast_to("int8")
            out_gm_b8 = self.out_gm[gm_offset].reinterpret_cast_to("int8")
            with self.tik_instance.if_scope(self.tiling_key == 0):
                self.tik_instance.data_move_pad(value_ub_b8, value_gm_b8, 1, burst_len * self.b8_times, 0, 0)
            self.tik_instance.data_move_pad(out_gm_b8, value_ub_b8, 1, burst_len * self.b8_times, 0, 0)

    def run_one_core(self, _core_idx, ele_num):
        """
        run assign in one core
        """
        max_burst_ele = self.max_burst_len * self.ele_per_block
        copy_loop = ele_num // max_burst_ele
        copy_tail = ele_num % max_burst_ele

        with self.tik_instance.for_range(0, copy_loop, thread_num=2) as _copy_idx:
            value_ub = self.tik_instance.Tensor(self.value_dtype, (self.max_tensor_size,),
                                                name="value_ub", scope=tik.scope_ubuf)
            copy_gm_offset = _core_idx * self.block_per_core * self.ele_per_block + _copy_idx * max_burst_ele
            self._run_one_loop(copy_gm_offset, max_burst_ele, value_ub)
        with self.tik_instance.if_scope(copy_tail > 0):
            value_ub = self.tik_instance.Tensor(self.value_dtype, (self.max_tensor_size,),
                                                name="value_ub", scope=tik.scope_ubuf)
            copy_gm_offset = _core_idx * self.block_per_core * self.ele_per_block + copy_loop * max_burst_ele
            self._run_one_loop(copy_gm_offset, copy_tail, value_ub)
    
    def run_one_core_scalar(self, _core_idx, ele_num):
        """
        run assign in one core for value is scalar
        """
        max_burst_ele = self.max_burst_len * self.ele_per_block
        copy_loop = ele_num // max_burst_ele
        copy_tail = ele_num % max_burst_ele
        value_ub = self.tik_instance.Tensor(self.value_dtype, (self.max_tensor_size,),
                                                name="value_ub", scope=tik.scope_ubuf)
        with self.tik_instance.if_scope(copy_loop > 0):
            self._broadcast_value(value_ub, max_burst_ele)
        with self.tik_instance.else_scope():
            self._broadcast_value(value_ub, copy_tail)

        with self.tik_instance.for_range(0, copy_loop) as _copy_idx:
            copy_gm_offset = _core_idx * self.block_per_core * self.ele_per_block + _copy_idx * max_burst_ele
            self._run_one_loop(copy_gm_offset, max_burst_ele, value_ub)
        with self.tik_instance.if_scope(copy_tail > 0):
            copy_gm_offset = _core_idx * self.block_per_core * self.ele_per_block + copy_loop * max_burst_ele
            self._run_one_loop(copy_gm_offset, copy_tail, value_ub)
    
    def run_one_core_entrance(self, _core_idx, ele_num):
        """
        The entrance of run function
        """
        if self.dmp_flag is True:
            with self.tik_instance.if_scope(self.tiling_key == 0):
                self.run_one_core(_core_idx, ele_num)
            with self.tik_instance.else_scope():
                self.run_one_core_scalar(_core_idx, ele_num)
        else:
            self.run_one_core(_core_idx, ele_num)

    def assign_compute(self, is_dynamic=True):
        """
        The tik implementation of operator Assign
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
        self._tiling_args()

        with self.tik_instance.for_range(0, self.running_core_num, block_num=self.running_core_num) as _core_idx:
            with self.tik_instance.if_scope(_core_idx < (self.core_used_num - 1)):
                self.run_one_core_entrance(_core_idx, self.block_per_core * self.ele_per_block)
            with self.tik_instance.if_scope(_core_idx == (self.core_used_num - 1)):
                self.run_one_core_entrance(_core_idx, self.ele_tail_core)

        tbe_context.get_context().add_compile_info("vars",
                                                   {"ub_size": self.ub_size_bytes, "core_num": self.ai_core_num,
                                                    "dmp_flag": self.dmp_flag})
        opt_config = {"out_of_bound_sync_check": True}
        if is_dynamic is True and self.dmp_flag is True:
            tiling_key_list = [[0], [1]]
            tiling_params = {"build_multi_kernels" : {"tiling_key" : [self.tiling_key],
                                                      "tiling_key_value": tiling_key_list}}
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=(self.ref_gm, self.value_gm),
                                       outputs=(self.out_gm,),
                                       flowtable=(self.tiling_gm,),
                                       extend_params=tiling_params, config=opt_config)
        else:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=(self.ref_gm, self.value_gm),
                                       outputs=(self.out_gm,),
                                       flowtable=(self.tiling_gm,), config=opt_config)


@register_operator("Assign")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def assign(ref, value, output, kernel_name="assign"):
    """
    algorithm: assign
    calculating: update 'ref' by assigning 'value' to it

    Parameters
    ----------
    ref: dict
        dict of input_ref, include shape and dtype,
    value: dict
        dict of input_value, include shape and dtype,
        Must have the same shape and dtype as input_ref
    output: dict
        dict of output
    kernel_name : str
        cce kernel name, default value is assign

    Returns
    -------
    None
    """
    is_dynamic = is_unknown([ref, value, output])
    obj = Assign(ref, value, output, kernel_name)
    obj.assign_compute(is_dynamic)
