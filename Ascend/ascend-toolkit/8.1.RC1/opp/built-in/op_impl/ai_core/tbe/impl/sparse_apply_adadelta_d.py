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
sparse_apply_adadelta_d
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.sparse_apply_common import SparseApply


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals,too-few-public-methods
class SparseApplyAdadelta(SparseApply):
    """
        Function: use to store sparse_apply_adadelta base parameters
    """

    # 'pylint: disable=too-many-statements
    def __init__(self, var, accum, accum_update, learning_rate, rho, grad, indices, epsilon, kernel_name):
        """
        Init sparse_apply_adadelta base parameters

        Parameters
        ----------
        var: dict
            dict of tensor var, include shape and dtype.
        accum: dict
            dict of tensor accum, include shape and dtype.
            Must have the same dtype and shape as var.
        accum_update: dict
            dict of tensor accum_update, include shape and dtype.
            Must have the same dtype and shape as var.
        learning_rate: dict
            dict of scalar learning_rate,
            Must have the same dtype as var.
        grad: dict
            dict of tensor grad,
            Must have the same dtype  as var.
        indices: dict
           dict of tensor indices, include shape and dtype, only support int32.
        rho: float
            scalar
        accum_updateentum: float
            scalar
        epsilon: float
            scalar
        kernel_name: str
            default value is "sparse_apply_adadelta_d"

        Returns:
        None
        """
        super().__init__(var, grad, indices, kernel_name)
        self.epsilon = epsilon

        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()

        self.accum_shape = accum.get("shape")
        self.accum_dtype = accum.get("dtype").lower()

        self.accum_update_shape = accum_update.get("shape")
        self.accum_update_dtype = accum_update.get("dtype").lower()

        self.lr_shape = learning_rate.get("shape")
        self.lr_dtype = learning_rate.get("dtype").lower()

        self.rho_shape = rho.get("shape")
        self.rho_dtype = rho.get("dtype").lower()

        self.vdiv_support = False

        self.lr_scalar = self.tik_instance.Scalar(self.lr_dtype)
        self.rho_scalar = self.tik_instance.Scalar(self.rho_dtype)

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
        add_support = tbe_platform.api_check_support("tik.vadd", "float32")

        self.vdiv_support = tbe_platform.api_check_support("tik.vdiv", "float32")

        if self.var_dtype == "float32" and not add_support:
            error_manager_vector.raise_err_input_dtype_not_supported("sparse_apply_adadelta_d", "var", [],
                                                                     self.var_dtype)

        para_check.check_shape(self.var_shape, param_name="var")
        para_check.check_shape(self.accum_shape, param_name="accum")
        para_check.check_shape(self.accum_update_shape, param_name="accum_update")
        para_check.check_shape(self.lr_shape, param_name="lr")
        para_check.check_shape(self.rho_shape, param_name="rho")

        para_check.check_dtype(self.var_dtype, ("float32", ), param_name="var")
        para_check.check_dtype(self.accum_dtype, ("float32", ), param_name="accum")
        para_check.check_dtype(self.accum_update_dtype, ("float32", ), param_name="accum_update")
        para_check.check_dtype(self.lr_dtype, ("float32", ), param_name="lr")
        para_check.check_dtype(self.rho_dtype, ("float32", ), param_name="rho")

        if self.accum_shape != self.var_shape:
            error_manager_vector.raise_err_inputs_shape_not_equal("sparse_apply_adadelta_d", "accum", "var",
                                                                  self.accum_shape, self.var_shape, self.var_shape)

        if self.accum_update_shape != self.var_shape:
            error_manager_vector.raise_err_inputs_shape_not_equal("sparse_apply_adadelta_d", "accum_update", "var",
                                                                  self.accum_update_shape, self.var_shape,
                                                                  self.var_shape)

    def _calculate(self, repeat_times, mask, offset):
        tmp1_ub = self._get_ub("tmp1_ub")[offset]
        tmp2_ub = self._get_ub("tmp2_ub")[offset]

        lr_ub = self._get_ub("lr_ub")
        rho_ub = self._get_ub("rho_ub")

        lr_gm = self._get_scalar_gm("lr_gm")
        rho_gm = self._get_scalar_gm("rho_gm")

        self.tik_instance.tensor_mov(lr_ub, lr_gm, '', 1, 1, 0, 0)
        self.lr_scalar.set_as(lr_ub[0])

        self.tik_instance.tensor_mov(rho_ub, rho_gm, '', 1, 1, 0, 0)
        self.rho_scalar.set_as(rho_ub[0])

        if self.each_row_data_num <= self.cache_threshold_col:
            var_ub = self._get_ub("var_align_ub")[offset]
            accum_ub = self._get_ub("accum_align_ub")[offset]
            accum_update_ub = self._get_ub("accum_update_align_ub")[offset]
            grad_ub = self.grad_align_ub[offset]
        else:
            var_ub = self._get_ub("var_ub")[offset]
            accum_ub = self._get_ub("accum_ub")[offset]
            accum_update_ub = self._get_ub("accum_update_ub")[offset]
            grad_ub = self.grad_ub[offset]
        self.tik_instance.vmuls(mask, accum_ub, accum_ub, self.rho_scalar, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, tmp1_ub, grad_ub, grad_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, tmp1_ub, tmp1_ub, (1 - self.rho_scalar), repeat_times, 1, 1, 8, 8)
        self.tik_instance.vadd(mask, accum_ub, accum_ub, tmp1_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadds(mask, tmp1_ub, accum_update_ub, self.epsilon, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vsqrt(mask, tmp1_ub, tmp1_ub, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, tmp2_ub, accum_ub, self.epsilon, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vsqrt(mask, tmp2_ub, tmp2_ub, repeat_times, 1, 1, 8, 8)
        if self.vdiv_support:
            self.tik_instance.vdiv(mask, tmp2_ub, grad_ub, tmp2_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        else:
            self.tik_instance.vrec(mask, tmp2_ub, tmp2_ub, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmul(mask, tmp2_ub, grad_ub, tmp2_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmul(mask, tmp1_ub, tmp1_ub, tmp2_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmuls(mask, tmp2_ub, tmp1_ub, self.lr_scalar, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, var_ub, var_ub, tmp2_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmuls(mask, accum_update_ub, accum_update_ub, self.rho_scalar, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, tmp2_ub, tmp1_ub, tmp1_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, tmp2_ub, tmp2_ub, (1 - self.rho_scalar), repeat_times, 1, 1, 8, 8)
        self.tik_instance.vadd(mask, accum_update_ub, accum_update_ub, tmp2_ub, repeat_times, 1, 1, 1, 8, 8, 8)


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def sparse_apply_adadelta_d(var,
                            accum,
                            accum_update,
                            lr,
                            rho,
                            grad,
                            indices,
                            out_var,
                            out_accum,
                            out_accum_update,
                            epsilon,
                            use_locking=False,
                            kernel_name="sparse_apply_adadelta_d"):
    """
    Updates "var" in specified index according to the Adadelta algorithm.

    accum{t} <- rho * accum{t - 1} + (1 - rho) * grad.square()
    update <- (accum_update{t - 1} + epsilon).sqrt() *
              (accum{t} + epsilon()).rsqrt() * grad
    var{t} <- var{t - 1} - update * lr
    accum_update{t} <- rho() * accum_update{t - 1} +
                      (1 - rho()) * update.square()

    Parameters
    ----------
    var: dict
        dict of tensor var, include shape and dtype,
        dtype only support float32.
    accum: dict
        dict of tensor accum, include shape and dtype.
        Must have the same dtype and shape as var.
    accum_update: dict
        dict of tensor accum_update, include shape and dtype.
        Must have the same dtype and shape as var.
    lr: dict
        dict of scalar lr,
        Must have the same dtype as var.
    grad: dict
        dict of tensor grad,
        Must have the same dtype  as var.
    indices: dict
       dict of tensor indices, include shape and dtype, only support int32.
    out_var: dict
        dict of out_var, include shape and dtype.
    out_accum: dict
        dict of out_accum, include shape and dtype.
    out_accum_update: dict
        dict of out_accum_update, include shape and dtype.
    rho: float
        scalar
    accum_updateentum: float
        scalar
    epsilon: float
        scalar
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_adadelta_d"

    Returns:
    None
    """
    sparse_apply_adadelta = SparseApplyAdadelta(var, accum, accum_update, lr, rho, grad, indices, epsilon, kernel_name)
    var_shape = var.get("shape")
    var_dtype = var.get("dtype").lower()

    sparse_apply_adadelta.add_input("var_in_gm", var_dtype, var_shape)
    sparse_apply_adadelta.add_input("accum_in_gm", var_dtype, var_shape)
    sparse_apply_adadelta.add_input("accum_update_in_gm", var_dtype, var_shape)
    sparse_apply_adadelta.allocate_scalar_gm("lr_gm", var_dtype)
    sparse_apply_adadelta.allocate_scalar_gm("rho_gm", var_dtype)

    sparse_apply_adadelta.add_output("var_out_gm", var_dtype, var_shape)
    sparse_apply_adadelta.add_output("accum_out_gm", var_dtype, var_shape)
    sparse_apply_adadelta.add_output("accum_update_out_gm", var_dtype, var_shape)
    sparse_apply_adadelta.reserve_ub("var_ub", var_dtype, "var_align_ub")
    sparse_apply_adadelta.reserve_ub("accum_ub", var_dtype, "accum_align_ub")
    sparse_apply_adadelta.reserve_ub("accum_update_ub", var_dtype, "accum_update_align_ub")
    sparse_apply_adadelta.reserve_ub("lr_ub", var_dtype, is_scalar=True)
    sparse_apply_adadelta.reserve_ub("rho_ub", var_dtype, is_scalar=True)
    sparse_apply_adadelta.reserve_ub("tmp1_ub", var_dtype)
    sparse_apply_adadelta.reserve_ub("tmp2_ub", var_dtype)
    sparse_apply_adadelta.set_var_shape(var_shape)
    sparse_apply_adadelta.sparse_apply_operator()
