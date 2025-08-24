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
concat_offset.py
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # Tiling Arg size for int64
    TILING_ARG_NUM = 4
    # the input num for one input
    MAX_INPUT_RANK = 64


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
def get_op_support_info(concat_dim, x, y, kernel_name="concat_offset_d"):
    """
    get_op_support_info
    """
    axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
def check_supported(concat_dim, x, y, kernel_name="concat_offset"):
    """
    when input is more than 95, changed to aicpu
    """
    x_len = len(x)
    if x_len > 95:
        return False, "aicore only support cases that all num of x is <= 95"

    return True, ""


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,too-many-instance-attributes
# 'pylint: disable=useless-object-inheritance
class ConcatOffsetCompute(object):
    """
    Function: use to store ConcatOffset base fuction
    Modify: 2020-12-05
    """
    def __init__(self, concat_dim, x, kernel_name):
        """
        init the input param

        Parameters
        ----------
        concat_dim: dict
            the input concat dim
        x: list
            list of dict, the input x
        kernel_name: str
            kernel name

        """
        self.concat_dim_dtype = concat_dim.get("dtype").lower()
        self.input_dtype = x[0].get("dtype").lower()
        self.concat_num = len(x)
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.concat_dim_gm = self.tik_instance.Tensor(self.concat_dim_dtype, (1,),
                                                      name="concat_dim_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,), name="ting_gm", \
        scope=tik.scope_gm)
        self.gm_input_list = []
        self.gm_output_list = []
        self.init_gm_mem(x)

    def init_gm_mem(self, input_list):
        """
        init_gm_mem , gm_input_list and gm_output_list
        """
        for i, input_dict in enumerate(input_list):
            _input_shape = [Constant.MAX_INPUT_RANK]
            _input_dtype = input_dict.get("dtype").lower()
            _input_name = "x_input_list{}".format(i)
            _input_gm = self.tik_instance.Tensor(_input_dtype, _input_shape, name=_input_name, scope=tik.scope_gm)
            self.gm_input_list.append(_input_gm)
            _output_name = "x_output_list{}".format(i)
            _output_gm = self.tik_instance.Tensor(_input_dtype, _input_shape, name=_output_name, scope=tik.scope_gm)
            self.gm_output_list.append(_output_gm)

    def run_compute(self):
        """
        run_compute: calcu the concat offset
        """
        # read tiling data
        input_x_num = self.tik_instance.Scalar("int64", name="input_x_num")
        vcetor_mask = self.tik_instance.Scalar("int64", name="vcetor_mask")
        tiling_ub = \
            self.tik_instance.Tensor("int64", [Constant.TILING_ARG_NUM], name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        input_x_num.set_as(tiling_ub[0])
        # read concat dim
        axis_num = self.tik_instance.Scalar(self.concat_dim_dtype, name="axis_num")
        axis_ub = \
            self.tik_instance.Tensor(self.concat_dim_dtype, [8], name="axis_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(axis_ub, self.concat_dim_gm, 0, 1, 1, 0, 0)
        axis_num.set_as(axis_ub[0])
        with self.tik_instance.if_scope(axis_num < 0):
            axis_num.set_as(axis_num + input_x_num)
        input_x_num.set_as((input_x_num + 7) // 8)
        vcetor_mask.set_as(1)
        with self.tik_instance.for_range(0, axis_num):
            vcetor_mask.set_as(vcetor_mask * 2)

        data_row1_ub = self.tik_instance.Tensor(self.input_dtype, [Constant.MAX_INPUT_RANK], \
        name="data_row1_ub", scope=tik.scope_ubuf)
        data_row2_ping = self.tik_instance.Tensor(self.input_dtype, [Constant.MAX_INPUT_RANK], \
        name="data_row2_ping", scope=tik.scope_ubuf)
        data_row2_pang = self.tik_instance.Tensor(self.input_dtype, [Constant.MAX_INPUT_RANK], \
        name="data_row2_pang", scope=tik.scope_ubuf)

        self.tik_instance.vector_dup(Constant.MAX_INPUT_RANK, data_row1_ub, 0, 1, 1, 1)

        # copy out the first output
        self.tik_instance.data_move(self.gm_output_list[0], data_row1_ub, 0, 1, input_x_num, 0, 0)
        self.concat_num = self.concat_num - 1
        for i in range(self.concat_num // 2):
            input_idx = i * 2 + 1
            self.tik_instance.data_move(data_row2_ping, self.gm_input_list[input_idx - 1], 0, 1, input_x_num, 0, 0)
            self.tik_instance.vadd([vcetor_mask, vcetor_mask], data_row1_ub,
                                   data_row1_ub, data_row2_ping, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.data_move(self.gm_output_list[input_idx], data_row1_ub, 0, 1, input_x_num, 0, 0)
            input_idx = i * 2 + 1 + 1
            self.tik_instance.data_move(data_row2_pang, self.gm_input_list[input_idx - 1], 0, 1, input_x_num, 0, 0)
            self.tik_instance.vadd([vcetor_mask, vcetor_mask], data_row1_ub,
                                   data_row1_ub, data_row2_pang, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.data_move(self.gm_output_list[input_idx], data_row1_ub, 0, 1, input_x_num, 0, 0)

        if self.concat_num % 2 == 1:
            input_idx = self.concat_num - 1 + 1
            self.tik_instance.data_move(data_row2_ping, self.gm_input_list[input_idx - 1], 0, 1, input_x_num, 0, 0)
            self.tik_instance.vadd([vcetor_mask, vcetor_mask], data_row1_ub,
                                   data_row1_ub, data_row2_ping, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.data_move(self.gm_output_list[input_idx], data_row1_ub, 0, 1, input_x_num, 0, 0)

    def run_build_cce(self):
        """
        run_build_cce
        """
        self.run_compute()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars", {"core_num": 1})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.concat_dim_gm] + self.gm_input_list,
                                   outputs=self.gm_output_list,
                                   flowtable=(self.tiling_gm,), config=opt_config)


@register_operator("ConcatOffset")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.DYNAMIC_INPUT,
                            para_check.DYNAMIC_OUTPUT, para_check.KERNEL_NAME)
def concat_offset(concat_dim, x, y, kernel_name="concat_offset"):
    """
    Compute the concat offset of the input tensor along `concat_dim`.

    Parameters
    ----------
    concat_dim: dict
                a number of int32, The dimension along which to concatenate,
                must be in the range [-rank(shape), rank(shape))
    x: list of dict, dict include shape and dtype, dtype must be in ('int32')
    y: list of dict, dict include shape and dtype, dtype must be in ('int32')
    kernel_name: kernel name

    Returns
    -------
    None
    """
    concat_object = ConcatOffsetCompute(concat_dim, x, kernel_name)

    concat_object.run_build_cce()
