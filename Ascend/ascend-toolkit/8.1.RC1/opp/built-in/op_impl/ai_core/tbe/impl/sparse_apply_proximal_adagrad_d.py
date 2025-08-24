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
sparse_apply_proximal_adagrad_d.py
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.sparse_apply_common import SparseApply


# 'pylint: disable=too-many-instance-attributes,too-few-public-methods
class SparseApplyProximalAdagrad(SparseApply):
    """
    Function: use to store sparse_apply_proximal_adagrad base parameters
    """

    # 'pylint: disable=invalid-name,too-many-arguments
    def __init__(self, var, accum, lr, l1, l2, grad, indices, kernel_name):
        """
        Init sparse_apply_proximal_adagrad base parameters
        """
        super().__init__(var, grad, indices, kernel_name)

        self.accum_shape = accum.get("shape")
        self.accum_dtype = accum.get("dtype").lower()
        self.lr_shape = lr.get("shape")
        self.lr_dtype = lr.get("dtype").lower()
        self.l1_shape = l1.get("shape")
        self.l1_dtype = l1.get("dtype").lower()
        self.l2_shape = l2.get("shape")
        self.l2_dtype = l2.get("dtype").lower()
        self.vdiv_support = False

        self.lr_scalar = self.tik_instance.Scalar(self.lr_dtype)
        self.l1_scalar = self.tik_instance.Scalar(self.l1_dtype)
        self.l2_scalar = self.tik_instance.Scalar(self.l2_dtype)

        self._check_param()

    def _check_param(self):
        """
        Check parameter
        """
        add_support = tbe_platform.api_check_support("tik.vadd", "float32")
        self.vdiv_support = tbe_platform.api_check_support("tik.vdiv", "float32")
        if self.var_dtype == "float32" and not add_support:
            error_manager_vector.raise_err_input_dtype_not_supported("sparse_apply_proximal_adagrad_d", "var", [],
                                                                     self.var_dtype)

        para_check.check_shape(self.var_shape, param_name="var")
        para_check.check_shape(self.accum_shape, param_name="accum")
        para_check.check_shape(self.lr_shape, param_name="lr")
        para_check.check_shape(self.l1_shape, param_name="l1")
        para_check.check_shape(self.l2_shape, param_name="l2")
        para_check.check_shape(self.grad_shape, param_name="grad")
        para_check.check_shape(self.indices_shape, param_name="indice")

        para_check.check_dtype(self.var_dtype, ("float32", ), param_name="var")
        para_check.check_dtype(self.accum_dtype, ("float32", ), param_name="accum")
        para_check.check_dtype(self.lr_dtype, ("float32", ), param_name="lr")
        para_check.check_dtype(self.l1_dtype, ("float32", ), param_name="l1")
        para_check.check_dtype(self.l2_dtype, ("float32", ), param_name="l2")
        para_check.check_dtype(self.grad_dtype, ("float32", ), param_name="grad")
        para_check.check_dtype(self.indices_dtype, ("int32", "int64"), param_name="indice")

        if self.accum_shape != self.var_shape:
            error_manager_vector.raise_err_inputs_shape_not_equal("sparse_apply_proximal_adagrad_d", "accum", "var",
                                                                  self.accum_shape, self.var_shape, self.var_shape)

    # 'pylint: disable=too-many-locals,too-many-statements
    def _calculate(self, repeat_times, mask, offset):
        """
        The logical
        """
        tmp_ub = self._get_ub("tmp_ub")[offset]
        lr1_ub = self._get_ub("lr1_ub")[offset]
        lr2_ub = self._get_ub("lr2_ub")[offset]
        prox_var_ub = self._get_ub("prox_var_ub")[offset]
        oct_float_ub = self._get_ub("oct_float_ub")  # eight float32

        lr_ub = self._get_ub("lr_ub")
        lr_gm = self._get_scalar_gm("lr_gm")
        self.tik_instance.tensor_mov(lr_ub, lr_gm, '', 1, 1, 0, 0)
        self.lr_scalar.set_as(lr_ub[0])
        l1_ub = self._get_ub("l1_ub")
        l1_gm = self._get_scalar_gm("l1_gm")
        self.tik_instance.tensor_mov(l1_ub, l1_gm, '', 1, 1, 0, 0)
        self.l1_scalar.set_as(l1_ub[0])
        l2_ub = self._get_ub("l2_ub")
        l2_gm = self._get_scalar_gm("l2_gm")
        self.tik_instance.tensor_mov(l2_ub, l2_gm, '', 1, 1, 0, 0)
        self.l2_scalar.set_as(l2_ub[0])

        if self.each_row_data_num <= self.cache_threshold_col:
            var_ub = self._get_ub("var_align_ub")[offset]
            accum_ub = self._get_ub("accum_align_ub")[offset]
            grad_ub = self.grad_align_ub[offset]
        else:
            var_ub = self._get_ub("var_ub")[offset]
            accum_ub = self._get_ub("accum_ub")[offset]
            grad_ub = self.grad_ub[offset]

        # `Procedure1: accum = accum + grad * grad`
        self.tik_instance.vmla(mask, accum_ub, grad_ub, grad_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        # End Procedure1

        # `Procedure2: lr1 = lr / sqrt(accum)`
        # Step1: `tmp = sqrt(accum)`
        self.tik_instance.vsqrt(mask, tmp_ub, accum_ub, repeat_times, 1, 1, 8, 8)
        if self.vdiv_support:
            # Step2: `oct_float_ub = lr_scalar`
            self.tik_instance.vector_dup(8, oct_float_ub, self.lr_scalar, 1, 1, 8)
            # Step3: `lr1 = oct_float_ub / tmp`
            self.tik_instance.vdiv(mask, lr1_ub, oct_float_ub, tmp_ub, repeat_times, 1, 0, 1, 8, 0, 8)
        else:
            # Step2: `tmp = tmp ^ -1`
            self.tik_instance.vrec(mask, tmp_ub, tmp_ub, repeat_times, 1, 1, 8, 8)
            # Step3: `lr1 = tmp * lr_scalar`
            self.tik_instance.vmuls(mask, lr1_ub, tmp_ub, self.lr_scalar, repeat_times, 1, 1, 8, 8)
        # End Procedure2

        # Procedure3: `lr2 = 1.0 / (1.0 + l2 * lr1)`
        # Step1: `lr2 = lr1 * l2`
        self.tik_instance.vmuls(mask, lr2_ub, lr1_ub, self.l2_scalar, repeat_times, 1, 1, 8, 8)
        # Step2: `lr2 = lr2 + 1.0`
        self.tik_instance.vadds(mask, lr2_ub, lr2_ub, 1.0, repeat_times, 1, 1, 8, 8)
        # Step3: `lr2 = 1.0 / lr2`
        if self.vdiv_support:
            self.tik_instance.vector_dup(8, oct_float_ub, 1, 1, 1, 8)
            self.tik_instance.vdiv(mask, lr2_ub, oct_float_ub, lr2_ub, repeat_times, 1, 0, 1, 8, 0, 8)
        else:
            self.tik_instance.vrec(mask, lr2_ub, lr2_ub, repeat_times, 1, 1, 8, 8)
        # End Procedure3

        # Procedure4: `prox_var = var - grad * lr1`
        # Step1: `prox_var = grad * lr1`
        self.tik_instance.vmul(mask, prox_var_ub, grad_ub, lr1_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        # Step2: `prox_var = var - prox_var`
        self.tik_instance.vsub(mask, prox_var_ub, var_ub, prox_var_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        # End Procedure4

        with self.tik_instance.if_scope(self.l1_scalar > 0):
            # Procedure5: `var = np.sign(prox_var) * np.maximum(np.abs(prox_var) - lr1 * l1, 0.0) * lr2`
            # Step1: `grad_as_temp = abs(prox_var)`
            self.tik_instance.vabs(mask, grad_ub, prox_var_ub, repeat_times, 1, 1, 8, 8)
            # Step2: `var = lr1 * l1`
            self.tik_instance.vmuls(mask, var_ub, lr1_ub, self.l1_scalar, repeat_times, 1, 1, 8, 8)
            # Step3: `var = grad_as_temp - var`
            self.tik_instance.vsub(mask, var_ub, grad_ub, var_ub, repeat_times, 1, 1, 1, 8, 8, 8)
            # Step4: `tmp = 0.0`
            self.tik_instance.vmuls(mask, tmp_ub, var_ub, 0.0, repeat_times, 1, 1, 8, 8)
            # Step4: `var = maximum(var, tmp)`
            self.tik_instance.vmax(mask, var_ub, var_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
            # Step5: `tmp = sign(prox_var)`
            #            `= abs(prox_var) / (prox_var + 1.18e-38)`
            #            `= grad_as_temp / (prox_var + 1.18e-38)`
            # Step5-1: `tmp = prox_var + 1.18e-38`
            self.tik_instance.vadds(mask, tmp_ub, prox_var_ub, 1.18e-38, repeat_times, 1, 1, 8, 8)
            # Step5-2: `tmp = grad_as_temp / tmp`
            if self.vdiv_support:
                self.tik_instance.vdiv(mask, tmp_ub, grad_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
            else:
                self.tik_instance.vrec(mask, tmp_ub, tmp_ub, repeat_times, 1, 1, 8, 8)
                self.tik_instance.vmul(mask, tmp_ub, grad_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
            # Step6: `var = tmp * var`
            self.tik_instance.vmul(mask, var_ub, var_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
            # Step7: `var = var * lr2`
            self.tik_instance.vmul(mask, var_ub, var_ub, lr2_ub, repeat_times, 1, 1, 1, 8, 8, 8)
            # End Procedure5
        with self.tik_instance.else_scope():
            # Procedure6: `var = prox_var * lr2`
            self.tik_instance.vmul(mask, var_ub, prox_var_ub, lr2_ub, repeat_times, 1, 1, 1, 8, 8, 8)
            # End Procedure6


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sparse_apply_proximal_adagrad_d(var,
                                    accum,
                                    lr,
                                    l1,
                                    l2,
                                    grad,
                                    indices,
                                    var_out,
                                    accum_out,
                                    use_locking=False,
                                    kernel_name="sparse_apply_proximal_adagrad_d"):
    """
    The operator's compute logical:
    for i in range(index_len):
        gid = i
        vid = index[i]

        accum[vid] = accum[vid] + grad[gid] * grad[gid]

        lr1 = lr / sqrt(accum[vid])
        prox_var = var[vid] - grad[gid] * lr1
        lr2 = 1.0 / (1.0 + l2 * lr1)

        if l1 > 0:
            var_t1 = abs(prox_var) - lr1 * l1
            var[vid] = sign(prox_var) * maximum(var_t1, 0.0) * lr2
        else:
            var[vid] = prox_var * lr2

    Parameters
    ----------
    var: dict
        Input tensor contains shape and dtype attributes.
        Dtype only support float32.
    accum: dict
        Input tensor contains shape and dtype attributes.
        Dtype only support float32.
    lr: dict
        Input tensor contains shape and dtype attributes.
        Only lr[0] is used.
        Dtype only support float32.
    l1: dict
        Input tensor contains shape and dtype attributes.
        Only l1[0] is used.
        Dtype only support float32.
    l2: dict
        Input tensor contains shape and dtype attributes.
        Only l2[0] is used.
        Dtype only support float32.
    grad: dict
        Input tensor contains shape and dtype attributes.
        Dtype only support float32.
    indices: dict
        Input tensor contains shape and dtype attributes.
        Dtype only support "int64", "int32".
    var_out: dict
        Output tensor contains shape and dtype attributes.
        Dtype only support float32.
    accum_out: dict
        Output tensor contains shape and dtype attributes.
        Dtype only support float32.
    use_locking: bool
        Default value is "False"
    kernel_name: str
        Kernel name, default value is "sparse_apply_proximal_adagrad_d"

    Returns:
    None
    """
    sparse_apply_proximal_adagrad = SparseApplyProximalAdagrad(var, accum, lr, l1, l2, grad, indices, kernel_name)

    var_shape = var.get("shape")
    var_dtype = var.get("dtype").lower()

    sparse_apply_proximal_adagrad.add_input("var_in_gm", var_dtype, var_shape)
    sparse_apply_proximal_adagrad.add_input("accum_in_gm", var_dtype, var_shape)
    sparse_apply_proximal_adagrad.allocate_scalar_gm("lr_gm", var_dtype)
    sparse_apply_proximal_adagrad.allocate_scalar_gm("l1_gm", var_dtype)
    sparse_apply_proximal_adagrad.allocate_scalar_gm("l2_gm", var_dtype)

    sparse_apply_proximal_adagrad.add_output("var_out_gm", var_dtype, var_shape)
    sparse_apply_proximal_adagrad.add_output("accum_out_gm", var_dtype, var_shape)

    sparse_apply_proximal_adagrad.reserve_ub("var_ub", var_dtype, "var_align_ub")
    sparse_apply_proximal_adagrad.reserve_ub("accum_ub", var_dtype, "accum_align_ub")

    sparse_apply_proximal_adagrad.reserve_ub("lr_ub", var_dtype, is_scalar=True)
    sparse_apply_proximal_adagrad.reserve_ub("l1_ub", var_dtype, is_scalar=True)
    sparse_apply_proximal_adagrad.reserve_ub("l2_ub", var_dtype, is_scalar=True)

    sparse_apply_proximal_adagrad.reserve_ub("tmp_ub", var_dtype)
    sparse_apply_proximal_adagrad.reserve_ub("lr1_ub", var_dtype)
    sparse_apply_proximal_adagrad.reserve_ub("lr2_ub", var_dtype)
    sparse_apply_proximal_adagrad.reserve_ub("prox_var_ub", var_dtype)
    sparse_apply_proximal_adagrad.reserve_ub("oct_float_ub", var_dtype, align_name="oct_float_ub")
    sparse_apply_proximal_adagrad.set_var_shape(var_shape)
    sparse_apply_proximal_adagrad.sparse_apply_operator()
