# Copyright 2021 Huawei Technologies Co., Ltd
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
get_shape
"""

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tik


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    common constants
    """
    X_SHAPE_SIZE = 128
    Y_SHAPE_SIZE = 128
    UB_BLOCK_SIZE = 32


def _ceil(x_1, x_2):
    return (x_1 + x_2 - 1) // x_2


# 'pylint: disable=too-many-instance-attributes
class GetShape():
    """
    class for operator GetShape
    """

    def __init__(self, x, y, kernel_name):
        self.kernel_name = kernel_name

        self.y_dtype = y.get("dtype").lower()
        self.y_shape = y.get("shape")
        self.y_shape_value = self.y_shape[0]
        self.input_tensor_num = len(x)
        para_check.check_dtype(self.y_dtype, ("int32"), param_name="y_dtype")
        if self.y_shape_value > Constant.Y_SHAPE_SIZE:
            rule = "GetShape output size should not be greater than 128"
            error_manager_vector.raise_err_specific_reson(self.kernel_name, rule)
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.x_gm = []
        for i in range(self.input_tensor_num):
            self.x_gm.append(
                self.tik_instance.Tensor("int64", (Constant.X_SHAPE_SIZE,),
                                         scope=tik.scope_gm,
                                         name="x_gm_{}".format(i)))
        self.y_gm = self.tik_instance.Tensor(self.y_dtype, self.y_shape, scope=tik.scope_gm, name="y_gm")

    def get_shape_compute(self):
        """
        get_shape compute
        """
        x_ub = self.tik_instance.Tensor("int64", (Constant.X_SHAPE_SIZE,), scope=tik.scope_ubuf, name="x_ub")
        y_ub = self.tik_instance.Tensor("int32", (Constant.Y_SHAPE_SIZE,), scope=tik.scope_ubuf, name="y_ub")
        dim_num = self.tik_instance.Scalar(dtype="int32", init_value=0)
        y_base_idx = self.tik_instance.Scalar(dtype="int32", init_value=0)
        for i in range(self.input_tensor_num):
            self.tik_instance.data_move(x_ub, self.x_gm[i], 0, 1, 32, 0, 0)
            dim_num.set_as(x_ub[3])
            with self.tik_instance.for_range(0, dim_num) as idx:
                y_ub[y_base_idx + idx].set_as(x_ub[idx + 4])
            if self.input_tensor_num > 1:
                y_base_idx.set_as(y_base_idx + dim_num)

        burst_num = _ceil(self.y_shape_value * 4, Constant.UB_BLOCK_SIZE)
        self.tik_instance.data_move(self.y_gm, y_ub, 0, 1, burst_num, 0, 0)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=self.x_gm, outputs=[self.y_gm])
        return self.tik_instance


# 'pylint: disable=invalid-name
@register_operator("GetShape")
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def get_shape(x, y, kernel_name="get_shape"):
    """
    Parameters
    ----------
    x: dict
        shape and dtype of input
    y: dict
        shape and dtype of output
    kernel_name : str
        kernel name, default value is "get_shape"
    """

    get_shape_value = GetShape(x, y, kernel_name)
    return get_shape_value.get_shape_compute()
