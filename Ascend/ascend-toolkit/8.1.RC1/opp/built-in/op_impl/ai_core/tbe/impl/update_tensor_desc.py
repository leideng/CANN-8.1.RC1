#!/usr/bin/python
# -*- coding: utf-8 -*-
# copyright 2021 huawei technologies co., ltd
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
# http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
# ============================================================================
"""
update_tensor_desc
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    Constant in this class
    """

    DESC_SIZE = 128
    DIM_BASE_IDX = 3
    INT64_BYTE_SIZE = 8
    BLOCK_BYTE_SIZE = 32


# 'pylint: disable=too-few-public-methods
class UpdateTensorDesc:
    """
    UpdateTensorDesc
    """

    def __init__(self, input_x, shape):
        """
        constructor of class UpdateTensorDesc

        parameters
        ----------
        none

        returns
        -------
        none
        """

        self.tik_instance = tik.Tik()
        self.shape = shape
        self.dim_num = len(self.shape)
        self.dtype_x = input_x.get("dtype")

    def tik_instance_fun(self, kernel_name):
        """
        tik_instance_fun
        """

        para_check.check_shape(self.shape, param_name="shape")
        x_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.DESC_SIZE,), tbe_platform.scope_gm, "x_gm")
        y_gm = self.tik_instance.Tensor("int64", (Constant.DESC_SIZE,), tbe_platform.scope_gm, "y_gm")
        y_ub = self.tik_instance.Tensor("int64", (Constant.DESC_SIZE,), tbe_platform.scope_ubuf, "y_ub")

        burst_len = Constant.DESC_SIZE * Constant.INT64_BYTE_SIZE // Constant.BLOCK_BYTE_SIZE
        self.tik_instance.data_move(y_ub, y_gm, 0, 1, burst_len, 0, 0)

        dim_num_scalar = self.tik_instance.Scalar(dtype="int64", init_value=self.dim_num)
        y_ub[Constant.DIM_BASE_IDX].set_as(dim_num_scalar)

        for idx, value in enumerate(self.shape):
            dim_num_scalar.set_as(value)
            desc_idx = Constant.DIM_BASE_IDX + idx + 1
            y_ub[desc_idx].set_as(dim_num_scalar)
        self.tik_instance.data_move(y_gm, y_ub, 0, 1, burst_len, 0, 0)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[x_gm],
                                   outputs=[y_gm])
        return self.tik_instance


# 'pylint: disable=locally-disabled,invalid-name,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def update_tensor_desc(x, y, shape, kernel_name="UpdateTensorDesc"):
    """
    update the y tensor_desc
    parameters
    ----------
    shape :  list
        list of input_desc
    kernel_name : str
        kernel name, default value is "UpdateTensorDesc"

    returns
    -------
    compile info
    """
    obj_update_tensor_desc = UpdateTensorDesc(x, shape)
    obj_update_tensor_desc.tik_instance_fun(kernel_name)
