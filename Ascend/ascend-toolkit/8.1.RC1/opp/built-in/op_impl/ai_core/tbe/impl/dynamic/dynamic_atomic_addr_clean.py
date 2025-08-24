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
atomic_addr_clean
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_register


@tbe_register.register_param_generalization("DynamicAtomicAddrClean")
def dynamic_atomic_addr_clean_generalization(size_list, generalize_config=None):
    """dynamic_atomic_addr_clean_generalization
    """
    if generalize_config["mode"] == "all_shape":
        size_list = [-1 for _ in size_list]
        generalization_res = [size_list]

        return [generalization_res]

    return None


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max_int32
    MAX_INT32 = 2 ** 31 - 1
    # full mask for fp32
    MASK_FP32 = 64
    # max repeat time of vector instruction
    MAX_REPEAT_TIME = 255
    # max tiling params num
    MAX_TILING_PARAMS_NUM = 64
    # int32 byte
    INT32_BYTE = 4
    # block byte
    BLOCK_BYTE = 32
    ZERO_FP32 = 0.0


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
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    if is_double_buffer:
        return ub_size // 2
    return ub_size


# 'pylint: disable=too-many-instance-attributes,too-few-public-methods
class DynamicAtomicAddrClean():
    """
    DynamicAtomicAddrClean
    """

    # 'pylint: disable=too-few-public-methods,too-many-statements
    def __init__(self, size_list):
        """
        constructor of class DynamicAtomicAddrClean

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.is_double_buffer = True
        self.workspace_num = len(size_list)
        self.workspace_addrs = []
        self.ub_size = _tik_get_ub_size(self.is_double_buffer)
        self.gm_tensor = self.tik_instance.Tensor("float32", (Constant.MAX_INT32,),
                                                  tik.scope_gm, "gm_tensor")
        self.tiling_gm = self.tik_instance.Tensor("int32",
                                                  (Constant.MAX_TILING_PARAMS_NUM,),
                                                  tik.scope_gm, "tiling_gm")

        # 'pylint: disable=too-few-public-methods
        class CommonInputScalar():
            """
            CommonInputScalar
            modify date 2020-12-10
            """

            def __init__(self, tik_instance):
                """
                constructor of class CommonInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.select_key = tik_instance.Scalar(
                    dtype="int32", name="select_key")
                self.need_core_num = tik_instance.Scalar(
                    dtype="int32", name="need_core_num")
                self.ele_num_full_mask_repeat_time = \
                    tik_instance.Scalar(
                        dtype="int32",
                        name="ele_num_full_mask_repeat_time")
                self.burst_len_full_mask_repeat_time = \
                    tik_instance.Scalar(
                        dtype="int32",
                        name="burst_len_full_mask_repeat_time")

        # 'pylint: disable=too-few-public-methods
        class InitInputScalar():
            """
            InitInputScalar
            """

            def __init__(self, tik_instance):
                """
                constructor of class InitInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                # front core
                self.ele_num_front_core = tik_instance.Scalar(
                    dtype="int32", name="ele_num_front_core")
                # front part full mask full repeat time front core
                self.init_times_full_mask_repeat_time_front_core = \
                    tik_instance.Scalar(
                        dtype="int32",
                        name="init_times_full_mask_repeat_time_front_core")
                self.ele_num_front_part_front_core = tik_instance.Scalar(
                    dtype="int32",
                    name="ele_num_front_part_front_core")
                # last part front
                self.burst_len_last_part_front_core = tik_instance.Scalar(
                    dtype="int32",
                    name="burst_len_last_part_front_core")
                self.repeat_time_last_part_front_core = tik_instance.Scalar(
                    dtype="int32",
                    name="repeat_time_last_part_front_core")

                # last core
                self.ele_num_last_core = tik_instance.Scalar(
                    dtype="int32", name="ele_num_last_core")
                # front part full mask full repeat time last core
                self.init_times_full_mask_repeat_time_last_core = \
                    tik_instance.Scalar(
                        dtype="int32",
                        name="init_times_full_mask_repeat_time_last_core")
                self.ele_num_front_part_last_core = tik_instance.Scalar(
                    dtype="int32",
                    name="ele_num_front_part_last_core")
                # last part last core
                self.burst_len_last_part_last_core = tik_instance.Scalar(
                    dtype="int32",
                    name="burst_len_last_part_last_core")
                self.repeat_time_last_part_last_core = tik_instance.Scalar(
                    dtype="int32",
                    name="repeat_time_last_part_last_core")
                self.core_num = tik_instance.Scalar(dtype="int32", name="core_num")

        self.obj_common_input_scalar = CommonInputScalar(self.tik_instance)
        self.obj_init_input_scalar = InitInputScalar(self.tik_instance)
        self.ub_tensor = self.tik_instance.Tensor("float32", (
            Constant.MASK_FP32 * Constant.MAX_REPEAT_TIME,), tik.scope_ubuf, "ub_tensor")

    # 'pylint: disable=unused-argument
    def addr_clean(self, core_index, workspace_addr):
        """
        addr_clean
        :param core_index:
        :param workspace_addr:
        :return:
        """
        with self.tik_instance.if_scope(core_index < self.obj_common_input_scalar.need_core_num - 1):
            # front core
            with self.tik_instance.for_range(
                    0, self.obj_init_input_scalar.init_times_full_mask_repeat_time_front_core) as init_index:
                # front part front core full mask full repeat time
                self.tik_instance.vector_dup(Constant.MASK_FP32, self.ub_tensor[0],
                                             Constant.ZERO_FP32, Constant.MAX_REPEAT_TIME, 1, 8)
                gm_offset = core_index * self.obj_init_input_scalar.ele_num_front_core + \
                            init_index * self.obj_common_input_scalar.ele_num_full_mask_repeat_time
                ub_offset = 0
                self.tik_instance.data_move(workspace_addr[gm_offset],
                                            self.ub_tensor[ub_offset], 0, 1,
                                            self.obj_common_input_scalar.
                                            burst_len_full_mask_repeat_time,
                                            0, 0)
            # last part front core
            with self.tik_instance.if_scope(
                    self.obj_init_input_scalar.init_times_full_mask_repeat_time_front_core == 0):
                self.tik_instance.vector_dup(Constant.MASK_FP32, self.ub_tensor[0],
                                             Constant.ZERO_FP32,
                                             self.obj_init_input_scalar.
                                             repeat_time_last_part_front_core,
                                             1, 8)
            gm_offset = core_index * \
                        self.obj_init_input_scalar.ele_num_front_core + \
                        self.obj_init_input_scalar.ele_num_front_part_front_core
            with self.tik_instance.if_scope(self.obj_init_input_scalar.burst_len_last_part_front_core > 0):
                self.tik_instance.data_move(workspace_addr[gm_offset],
                                            self.ub_tensor[0], 0, 1,
                                            self.obj_init_input_scalar.
                                            burst_len_last_part_front_core,
                                            0, 0)
        with self.tik_instance.if_scope(core_index == self.obj_common_input_scalar.need_core_num - 1):
            # last core
            with self.tik_instance.for_range(
                    0, self.obj_init_input_scalar.init_times_full_mask_repeat_time_last_core) as init_index:
                # front part last core full mask full repeat time
                self.tik_instance.vector_dup(Constant.MASK_FP32, self.ub_tensor[0],
                                             Constant.ZERO_FP32,
                                             Constant.MAX_REPEAT_TIME, 1, 8)
                gm_offset = \
                    core_index * self.obj_init_input_scalar.ele_num_front_core \
                    + init_index * self.obj_common_input_scalar.ele_num_full_mask_repeat_time
                ub_offset = 0
                self.tik_instance.data_move(workspace_addr[gm_offset],
                                            self.ub_tensor[ub_offset], 0, 1,
                                            self.obj_common_input_scalar.
                                            burst_len_full_mask_repeat_time,
                                            0, 0)
            # last part last core
            with self.tik_instance.if_scope(
                    self.obj_init_input_scalar.init_times_full_mask_repeat_time_last_core == 0):
                self.tik_instance.vector_dup(Constant.MASK_FP32, self.ub_tensor[0],
                                             Constant.ZERO_FP32,
                                             self.obj_init_input_scalar.
                                             repeat_time_last_part_last_core,
                                             1, 8)
            gm_offset = core_index * \
                        self.obj_init_input_scalar.ele_num_front_core + \
                        self.obj_init_input_scalar.ele_num_front_part_last_core
            with self.tik_instance.if_scope(self.obj_init_input_scalar.burst_len_last_part_last_core > 0):
                self.tik_instance.data_move(workspace_addr[gm_offset],
                                            self.ub_tensor[0], 0, 1,
                                            self.obj_init_input_scalar.
                                            burst_len_last_part_last_core,
                                            0, 0)

    def init_tiling_ub(self):
        tiling_ub = self.tik_instance.Tensor("int32", (Constant.MAX_TILING_PARAMS_NUM, ), tik.scope_ubuf, "tiling_ub")
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                    Constant.MAX_TILING_PARAMS_NUM * Constant.INT32_BYTE // Constant.BLOCK_BYTE,
                                    0, 0)
        return tiling_ub

    def malloc_gm_tensors(self):
        for idx in range(self.workspace_num):
            gm = self.tik_instance.Tensor("float32", (Constant.MAX_INT32, ), tik.scope_gm, "".join(["gm", str(idx)]))
            self.workspace_addrs.append(gm)

    def tik_instance_fun(self, kernel_name):
        """
        tik_instance_fun
        """
        self.malloc_gm_tensors()
        tiling_ub = self.init_tiling_ub()

        # The only core_num and tiling_key
        self.obj_common_input_scalar.select_key.set_as(tiling_ub[0])
        self.obj_init_input_scalar.core_num.set_as(tiling_ub[14])
        core_num = self.obj_init_input_scalar.core_num

        with self.tik_instance.for_range(0, core_num, block_num=core_num) as core_index:
            for idx in range(self.workspace_num):
                input_scalar_index = 0 + idx * 15
                # common part input scalar
                input_scalar_index = input_scalar_index + 1
                self.obj_common_input_scalar.need_core_num.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_common_input_scalar.ele_num_full_mask_repeat_time.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_common_input_scalar.burst_len_full_mask_repeat_time.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                # init part input scalar
                self.obj_init_input_scalar.ele_num_front_core.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_init_input_scalar. \
                    init_times_full_mask_repeat_time_front_core.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_init_input_scalar.ele_num_front_part_front_core.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_init_input_scalar.burst_len_last_part_front_core.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_init_input_scalar.repeat_time_last_part_front_core.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_init_input_scalar.ele_num_last_core.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_init_input_scalar. \
                    init_times_full_mask_repeat_time_last_core.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_init_input_scalar.ele_num_front_part_last_core.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_init_input_scalar.burst_len_last_part_last_core.set_as(
                    tiling_ub[input_scalar_index])
                input_scalar_index = input_scalar_index + 1
                self.obj_init_input_scalar.repeat_time_last_part_last_core.set_as(
                    tiling_ub[input_scalar_index])

                self.addr_clean(core_index, self.workspace_addrs[idx])
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=self.workspace_addrs,
                                   outputs=[],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance


# 'pylint: disable=unused-argument
@register_operator("DynamicAtomicAddrClean")
@para_check.check_op_params(para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def dynamic_atomic_addr_clean(size_list, kernel_name="DynamicAtomicAddrClean"):
    """
    clean memory of workspace list
    Parameters
    ----------
    size_list :  list
        sizes of workspaces
    kernel_name : str
        kernel name, default value is "DynamicAtomicAddrClean"

    Returns
    -------
    compile info
    """
    obj_dynamic_atomic_addr_clean = DynamicAtomicAddrClean(size_list)
    obj_dynamic_atomic_addr_clean.tik_instance_fun(kernel_name)
    # add compile info
    tbe_context.get_context().add_compile_info("vars",
                                               {"ub_size": obj_dynamic_atomic_addr_clean.ub_size,
                                                "core_num": obj_dynamic_atomic_addr_clean.core_num,
                                                "workspace_num": obj_dynamic_atomic_addr_clean.workspace_num})
