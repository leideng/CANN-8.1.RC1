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
sparse_apply_rms_prop_d
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.sparse_apply_common import SparseApply


# 'pylint: disable=too-many-instance-attributes,too-few-public-methods
class SparseApplyRMSProp(SparseApply):
    """
        Function: use to store sparse_apply_rms_prop base parameters
    """

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, var, mean_square, mom, learning_rate, grad, indices, rho, momentum, epsilon, kernel_name):
        """
        Init sparse_apply_rms_prop base parameters

        Parameters
        ----------
        var: dict
            dict of tensor var, include shape and dtype.
        mean_square: dict
            dict of tensor mean_square, include shape and dtype.
            Must have the same dtype and shape as var.
        mom: dict
            dict of tensor mom, include shape and dtype.
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
        momentum: float
            scalar
        epsilon: float
            scalar
        kernel_name: str
            default value is "sparse_apply_rms_prop_d"

        Returns:
        None
        """
        super().__init__(var, grad, indices, kernel_name)
        self.epsilon = epsilon
        self.rho = rho
        self.momentum = momentum

        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()

        self.ms_shape = mean_square.get("shape")
        self.ms_dtype = mean_square.get("dtype").lower()

        self.mom_shape = mom.get("shape")
        self.mom_dtype = mom.get("dtype").lower()

        self.lr_shape = learning_rate.get("shape")
        self.lr_dtype = learning_rate.get("dtype").lower()

        self.vdiv_support = False

        self.lr_scalar = self.tik_instance.Scalar(self.lr_dtype)

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
            error_manager_vector.raise_err_input_dtype_not_supported("sparse_apply_rms_prop_d", "var", [],
                                                                     self.var_dtype)

        para_check.check_shape(self.var_shape, param_name="var")
        para_check.check_shape(self.ms_shape, param_name="ms")
        para_check.check_shape(self.mom_shape, param_name="mom")
        para_check.check_shape(self.lr_shape, param_name="lr")
        para_check.check_dtype(self.var_dtype, ("float32", ), param_name="var")
        para_check.check_dtype(self.ms_dtype, ("float32", ), param_name="ms")
        para_check.check_dtype(self.mom_dtype, ("float32", ), param_name="mom")
        para_check.check_dtype(self.lr_dtype, ("float32", ), param_name="lr")

        if self.ms_shape != self.var_shape:
            error_manager_vector.raise_err_inputs_shape_not_equal("sparse_apply_rms_prop_d", "ms", "var", self.ms_shape,
                                                                  self.var_shape, self.var_shape)

        if self.mom_shape != self.var_shape:
            error_manager_vector.raise_err_inputs_shape_not_equal("sparse_apply_rms_prop_d", "mom", "var",
                                                                  self.mom_shape, self.var_shape, self.var_shape)

    def _calculate(self, repeat_times, mask, offset):
        tmp_ub = self._get_ub("tmp_ub")[offset]

        lr_ub = self._get_ub("lr_ub")
        lr_gm = self._get_scalar_gm("lr_gm")

        self.tik_instance.tensor_mov(lr_ub, lr_gm, '', 1, 1, 0, 0)
        self.lr_scalar.set_as(lr_ub[0])

        if self.each_row_data_num <= self.cache_threshold_col:
            var_ub = self._get_ub("var_align_ub")[offset]
            ms_ub = self._get_ub("ms_align_ub")[offset]
            mom_ub = self._get_ub("mom_align_ub")[offset]
            grad_ub = self.grad_align_ub[offset]
        else:
            var_ub = self._get_ub("var_ub")[offset]
            ms_ub = self._get_ub("ms_ub")[offset]
            mom_ub = self._get_ub("mom_ub")[offset]
            grad_ub = self.grad_ub[offset]

        # ms_: `ms * rho + grad * grad * (1 - rho)`
        self.tik_instance.vmuls(mask, ms_ub, ms_ub, self.rho, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, tmp_ub, grad_ub, grad_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, (1 - self.rho), repeat_times, 1, 1, 8, 8)
        self.tik_instance.vadd(mask, ms_ub, ms_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        # mom_: `mom * momentum + (ms_ + epsilon).rsqrt() * lr * grad`
        self.tik_instance.vmuls(mask, mom_ub, mom_ub, self.momentum, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, tmp_ub, ms_ub, self.epsilon, repeat_times, 1, 1, 8, 8)

        self.tik_instance.vsqrt(mask, tmp_ub, tmp_ub, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vmuls(mask, grad_ub, grad_ub, self.lr_scalar, repeat_times, 1, 1, 8, 8)

        if self.vdiv_support:
            self.tik_instance.vdiv(mask, tmp_ub, grad_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        else:
            self.tik_instance.vrec(mask, tmp_ub, tmp_ub, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmul(mask, tmp_ub, grad_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, mom_ub, mom_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)

        # var_: `var - mom_`
        self.tik_instance.vsub(mask, var_ub, var_ub, mom_ub, repeat_times, 1, 1, 1, 8, 8, 8)


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sparse_apply_rms_prop_d(var,
                            ms,
                            mom,
                            lr,
                            grad,
                            indices,
                            out_var,
                            out_ms,
                            out_mom,
                            rho,
                            momentum,
                            epsilon,
                            use_locking=False,
                            kernel_name="sparse_apply_rms_prop_d"):
    """
    Adds sparse updates to the variable referenced by resource.

    Parameters
    ----------
    var: dict
        dict of tensor var, include shape and dtype,
        dtype only support float32.
    ms: dict
        dict of tensor ms, include shape and dtype.
        Must have the same dtype and shape as var.
    mom: dict
        dict of tensor mom, include shape and dtype.
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
    out_ms: dict
        dict of out_ms, include shape and dtype.
    out_mom: dict
        dict of out_mom, include shape and dtype.
    rho: float
        scalar
    momentum: float
        scalar
    epsilon: float
        scalar
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_rms_prop_d"

    Returns:
    None
    """
    sparse_apply_rms_prop = SparseApplyRMSProp(var, ms, mom, lr, grad, indices, rho, momentum, epsilon, kernel_name)

    var_shape = var.get("shape")
    var_dtype = var.get("dtype").lower()

    sparse_apply_rms_prop.add_input("var_in_gm", var_dtype, var_shape)
    sparse_apply_rms_prop.add_input("ms_in_gm", var_dtype, var_shape)
    sparse_apply_rms_prop.add_input("mom_in_gm", var_dtype, var_shape)
    sparse_apply_rms_prop.allocate_scalar_gm("lr_gm", var_dtype)

    sparse_apply_rms_prop.add_output("var_out_gm", var_dtype, var_shape)
    sparse_apply_rms_prop.add_output("ms_out_gm", var_dtype, var_shape)
    sparse_apply_rms_prop.add_output("mom_out_gm", var_dtype, var_shape)
    sparse_apply_rms_prop.reserve_ub("var_ub", var_dtype, "var_align_ub")
    sparse_apply_rms_prop.reserve_ub("ms_ub", var_dtype, "ms_align_ub")
    sparse_apply_rms_prop.reserve_ub("mom_ub", var_dtype, "mom_align_ub")
    sparse_apply_rms_prop.reserve_ub("lr_ub", var_dtype, is_scalar=True)
    sparse_apply_rms_prop.reserve_ub("tmp_ub", var_dtype)
    sparse_apply_rms_prop.set_var_shape(var_shape)
    sparse_apply_rms_prop.sparse_apply_operator()
