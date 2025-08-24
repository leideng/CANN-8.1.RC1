# Copyright 2019 Huawei Technologies Co., Ltd
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
sparse_apply_ftrl_d
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl import sparse_apply_common


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class SparseApplyFtrl(sparse_apply_common.SparseApply):
    """
    Sub class inherited form SparseApply for sparse_apply_ftrl op
    """

    # 'pylint: disable=invalid-name,too-many-arguments
    def __init__(self, var, accum, linear, grad, indices, lr, lr_power, l1, l2, l2_shrinkage, kernel_name):
        """
        init sparse_apply_ftrl  base parameters

        Parameters
        ----------
        lr: float
            scalar
        lr_power: float
            scalar
        l1: float
            scalar
        l2: float
            scalar
        l2_shrinkage: float
            scalar

        Returns
        -------
        None
        """
        super().__init__(var, grad, indices, kernel_name)
        self.lr = lr
        self.lr_power = lr_power
        self.l1 = l1
        self.l2 = l2
        self.l2_shrinkage = l2_shrinkage
        self.lr_vrec = 1.0 / self.lr
        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()
        self.accum_shape = accum.get("shape")
        self.accum_dtype = accum.get("dtype").lower()
        self.linear_shape = linear.get("shape")
        self.linear_dtype = linear.get("dtype").lower()
        self._check_param()

    def _check_param(self):
        """
        Check parameter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        para_check.check_shape(self.var_shape, param_name="var")
        para_check.check_shape(self.accum_shape, param_name="accum")
        para_check.check_shape(self.linear_shape, param_name="linear")
        para_check.check_dtype(self.var_dtype, ("float32", ), param_name="input_x")
        para_check.check_dtype(self.accum_dtype, ("float32", ), param_name="input_x")
        para_check.check_dtype(self.linear_dtype, ("float32", ), param_name="input_x")

        if self.accum_shape != self.var_shape:
            error_manager_vector.raise_err_inputs_shape_not_equal("sparse_apply_ftrl_d", "accum", "var",
                                                                  self.accum_shape, self.var_shape, self.var_shape)

        if self.linear_shape != self.var_shape:
            error_manager_vector.raise_err_inputs_shape_not_equal("sparse_apply_ftrl_d", "linear", "var",
                                                                  self.linear_shape, self.var_shape, self.var_shape)

    def _calculate(self, repeat_times, mask, offset):
        """
        calculate data according to the Ftrl-proximal scheme
        will automated called by basic class function

        Parameters
        ----------
        repeat_times: repeat times of insn
        mask: mask for vector insn
        offset: offset of ub addr

        Returns
        -------
        None
        """
        tmp_ub = self._get_ub("tmp_ub")[offset]
        tmp2_ub = self._get_ub("tmp2_ub")[offset]
        if self.each_row_data_num <= self.cache_threshold_col:
            var_ub = self._get_ub("var_align_ub")[offset]
            accum_ub = self._get_ub("accum_align_ub")[offset]
            linear_ub = self._get_ub("linear_align_ub")[offset]
            grad_ub = self.grad_align_ub[offset]
        else:
            var_ub = self._get_ub("var_ub")[offset]
            accum_ub = self._get_ub("accum_ub")[offset]
            linear_ub = self._get_ub("linear_ub")[offset]
            grad_ub = self.grad_ub[offset]

        if self.l2_shrinkage != 0:
            self.tik_instance.vmuls(mask, tmp_ub, var_ub, 2 * self.l2_shrinkage, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vadd(mask, tmp2_ub, grad_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmul(mask, tmp_ub, grad_ub, grad_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        # linear += grad, grad will not used after this operation
        if self.l2_shrinkage != 0:
            self.tik_instance.vadd(mask, linear_ub, tmp2_ub, linear_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        else:
            self.tik_instance.vadd(mask, linear_ub, grad_ub, linear_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        # `a^b = e^(b*lna)`
        # `grad = ln(accum)`
        self.tik_instance.vln(mask, grad_ub, accum_ub, repeat_times, 1, 1, 8, 8)
        # `grad = -lr_power*ln(accum)`
        self.tik_instance.vmuls(mask, grad_ub, grad_ub, -self.lr_power, repeat_times, 1, 1, 8, 8)
        # `grad = e^(-lr_power*ln(accum)) = accum ^ (-lr_power)`
        self.tik_instance.vexp(mask, grad_ub, grad_ub, repeat_times, 1, 1, 8, 8)

        # `accum += grad*grad`
        self.tik_instance.vadd(mask, accum_ub, accum_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        # `tmp = ln(accum_new)`
        self.tik_instance.vln(mask, tmp_ub, accum_ub, repeat_times, 1, 1, 8, 8)

        # `accum_new^(-lr_power)`
        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, -self.lr_power, repeat_times, 1, 1, 8, 8)

        # `tmp = accum_new ^ (-lr_power)`
        self.tik_instance.vexp(mask, tmp_ub, tmp_ub, repeat_times, 1, 1, 8, 8)

        # `tmp2 =accum_new^(-lr_power)/lr, used by y_res`
        self.tik_instance.vmuls(mask, tmp2_ub, tmp_ub, self.lr_vrec, repeat_times, 1, 1, 8, 8)

        # `tmp = accum^(-lr_power)- accum_new^(-lr_power)`
        self.tik_instance.vsub(mask, tmp_ub, grad_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        # `tmp = tmp / lr`
        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, self.lr_vrec, repeat_times, 1, 1, 8, 8)

        # `tmp = tmp / lr * var`
        self.tik_instance.vmul(mask, tmp_ub, tmp_ub, var_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        # `linear += grad +(accum^(-lr_power)-accum_new^(-lr_power))/lr*var`
        self.tik_instance.vadd(mask, linear_ub, tmp_ub, linear_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        # `x_res=linear.min(l1).max(-l1)-linear`
        self.tik_instance.vector_dup(mask, tmp_ub, self.l1, repeat_times, 1, 8)
        self.tik_instance.vmin(mask, grad_ub, linear_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vector_dup(mask, tmp_ub, -self.l1, repeat_times, 1, 8)
        self.tik_instance.vmax(mask, tmp_ub, grad_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, tmp_ub, tmp_ub, linear_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        # `y_res = accum_new^(-lr_power)/lr + 2*l2`
        self.tik_instance.vadds(mask, tmp2_ub, tmp2_ub, 2 * self.l2, repeat_times, 1, 1, 8, 8)

        # `var = x_res/y_res`
        self.tik_instance.vdiv(mask, var_ub, tmp_ub, tmp2_ub, repeat_times, 1, 1, 1, 8, 8, 8)


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT),
                            (para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT),
                            (para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT),
                            (para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT), para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def sparse_apply_ftrl_d(var,
                        accum,
                        linear,
                        grad,
                        indices,
                        var_out,
                        accum_out,
                        linear_out,
                        lr,
                        l1,
                        l2,
                        lr_power,
                        use_locking=False,
                        kernel_name="sparse_apply_ftrl"):
    """
    Update the variable referenced by resource.

    Parameters
    ----------
    var: dict
        data of input var
        datatype suports float32,float16
    accum: dict
        data of input accum
        datatype suports float32,float16
    linear: dict
        data of input linear
        datatype suports float32,float16
    grad: dict
        data of grad
        datatype supports float32,float16
    indices: dict
        data of indices
        datatype supports int32
    lr: const
        data of lr
        datatype supports float32,float16,int32
    l1: const
        data of l1
        datatype supports float32,float16,int32
    l2: const
        data of l2
        datatype supports float32,float16,int32
    lr_power: const
        data of lr_power
        datatype supports float32,float16,int32
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_ftrl"

    Returns:
    None
    """
    apply_ftrl = SparseApplyFtrl(var, accum, linear, grad, indices, lr, lr_power, l1, l2, 0, kernel_name)
    var_shape = var.get('shape')
    apply_ftrl.add_input("var_in_gm", "float32", var_shape)
    apply_ftrl.add_input("accum_in_gm", "float32", var_shape)
    apply_ftrl.add_input("linear_in_gm", "float32", var_shape)
    apply_ftrl.add_output("var_out_gm", "float32", var_shape)
    apply_ftrl.add_output("accum_out_gm", "float32", var_shape)
    apply_ftrl.add_output("linear_out_gm", "float32", var_shape)
    apply_ftrl.reserve_ub("var_ub", "float32", "var_align_ub")
    apply_ftrl.reserve_ub("accum_ub", "float32", "accum_align_ub")
    apply_ftrl.reserve_ub("linear_ub", "float32", "linear_align_ub")
    apply_ftrl.reserve_ub("tmp_ub", "float32")
    apply_ftrl.reserve_ub("tmp2_ub", "float32")
    apply_ftrl.set_var_shape(var_shape)
    apply_ftrl.sparse_apply_operator()
