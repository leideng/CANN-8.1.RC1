#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# licensed under the Apache License, Version 2.0 (the "License");
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
max_pool_grad_grad_with_argmax op, if no cbuf in env.
computes second-order gradients of the maxpooling function.
"""
import te
from tbe import tvm
import tbe.common.platform
from impl.util.platform_adapter import build_config
from te.lang import cce
from te.utils import para_check
from te.utils.error_manager import error_manager_vector


def _check_5hd(param, shape_length):
    if shape_length != 5:
        error_manager_vector.raise_err_input_param_range_invalid("max_pool_grad_grad_with_argmax",
                                                                 param, 5, 5, shape_length)


def _check_last_dim(param, shape):
    if shape[4] != 16:
        error_manager_vector.raise_err_check_params_rules("max_pool_grad_grad_with_argmax",
                                                          "the last dimension must be equal to 16", param, shape)


# 'pylint: disable=too-many-lines
def _check_shape_and_format(x, grad, argmax, y, ksize, strides, padding):
    x_dtype = x.get("dtype").lower()
    grad_dtype = grad.get("dtype").lower()
    argmax_dtype = argmax.get("dtype").lower()
    y_dtype = y.get("dtype").lower()

    para_check.check_dtype(x_dtype, ("float", "float32", "float16",), param_name="x")
    para_check.check_dtype(grad_dtype, ("float", "float32", "float16",), param_name="grad")
    para_check.check_dtype(argmax_dtype, ("uint16",), param_name="argmax")
    para_check.check_dtype(y_dtype, ("float", "float32", "float16",), param_name="y")

    x_shape = x.get("shape")
    grad_shape = grad.get("shape")
    argmax_shape = argmax.get("shape")
    y_shape = y.get("shape")

    para_check.check_shape(x_shape, param_name="x")
    para_check.check_shape(grad_shape, param_name="grad")
    para_check.check_shape(argmax_shape, param_name="argmax")
    para_check.check_shape(y_shape, param_name="y")

    _check_5hd("x", len(x_shape))
    _check_5hd("grad", len(grad_shape))
    _check_5hd("argmax", len(argmax_shape))
    _check_5hd("y", len(y_shape))

    _check_last_dim("x", x_shape)
    _check_last_dim("grad", grad_shape)
    _check_last_dim("y", y_shape)
    if argmax_shape[4] != 16 and argmax_shape[4] != 1:
        error_manager_vector.raise_err_check_params_rules("max_pool_grad_grad_with_argmax",
                                                          "the last dimension must be equal to 16 or 1",
                                                          "argmax", argmax_shape)

    x_ori_format = x.get("ori_format")
    if x_ori_format not in ("NHWC", "NCHW"):
        error_manager_vector.raise_err_input_format_invalid("max_pool_grad_grad_with_argmax",
                                                            "x", ("NHWC", "NCHW"), x_ori_format)

    if x_ori_format == "NCHW":
        ksize = [ksize[0], ksize[2], ksize[3], ksize[1]]
        strides = [strides[0], strides[2], strides[3], strides[1]]

    kn, kh, kw, kc = ksize
    stride_n, stride_h, stride_w, stride_c = strides
    if kn != 1 or kc != 1 or kh <= 0 or kw <= 0:
        error_manager_vector.raise_err_check_params_rules("max_pool_grad_grad_with_argmax",
                                                          "the dimensions of ksize are invalid",
                                                          ["ksize"], [ksize])
    if stride_n != 1 or stride_c != 1 or stride_h <= 0 or stride_w <= 0:
        error_manager_vector.raise_err_check_params_rules("max_pool_grad_grad_with_argmax",
                                                          "the dimensions of strides are invalid",
                                                          ["strides"], [strides])

    if padding not in ("SAME", "VALID"):
        error_manager_vector.raise_err_input_value_invalid("max_pool_grad_grad_with_argmax",
                                                           "padding", ("SAME", "VALID"), padding)

    n, c1, hi, wi, c0 = x_shape
    ho, pad_top, pad_bottom = cce.te_compute.common.tf_get_windowed_output_size_verbose(
        hi, kh, stride_h, padding)
    wo, pad_left, pad_right = cce.te_compute.common.tf_get_windowed_output_size_verbose(
        wi, kw, stride_w, padding)
    if wo <= 1:
        error_manager_vector.raise_err_specific_reson(
            "max_pool_grad_grad_with_argmax",
            "wo must be bigger than 1")

    expect_argmax_shape_1 = (n, c1, kh * kw, (ho * wo + 31) // 16, 16)
    expect_argmax_shape_2 = (n, c1, kh * kw, (ho * wo + 31) // 16 * 16, 1)

    if list(x_shape) != list(grad_shape):
        error_manager_vector.raise_err_inputs_shape_not_equal("max_pool_grad_grad_with_argmax", "x", "grad",
                                                              x_shape, grad_shape, x_shape)
    if list(argmax_shape) != list(expect_argmax_shape_1) and list(argmax_shape) != list(expect_argmax_shape_2):
        error_manager_vector.raise_err_specific_reson("max_pool_grad_grad_with_argmax", "shape_argmax not support.")
    if list(y_shape) != [n, c1, ho, wo, c0]:
        error_manager_vector.raise_err_input_value_invalid("max_pool_grad_grad_with_argmax",
                                                           "y.shape", [n, c1, ho, wo, c0], y_shape)

    shape_args = (n, c1, hi, wi, c0, kh, kw, stride_h, stride_w, ho, wo, pad_top, pad_bottom, pad_left, pad_right)
    return grad_dtype, shape_args


class MaxPoolGradGradWithArgmaxWithoutCbuf:
    def __init__(self, x, grad, argmax, y, ksize, strides, padding, kernel_name):
        self.dtype, shape_args = _check_shape_and_format(x, grad, argmax, y, ksize, strides, padding)
        self.n, self.c1, self.hi, self.wi, self.c0, \
            self.kh, self.kw, self.stride_h, self.stride_w, self.ho, self.wo, \
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = shape_args
        self.blocks_per_c0 = {"float": 2, "float32": 2, "float16": 1}.get(self.dtype)

        self.nc1 = self.n * self.c1

        self.core_num = te.platform.get_soc_spec(te.platform.CORE_NUM)
        # `byte_per_block = 32`
        self.block_num = te.platform.get_soc_spec(te.platform.UB_SIZE) // 32

        self.double_buffer_on = True
        self.cut_nc1_factor = 1
        self.cut_ho_factor = self.ho
        self.cut_wo_factor = self.wo
        self.cut_kh_factor = self.kh

        self.kernel_name = kernel_name

    def build(self):
        self._calculate_tiling_args()

        compute_list, data_list = self._compute()
        sch = self._schedule(compute_list)
        with build_config():
            tvm.build(sch, data_list, name=self.kernel_name)

    def _optimize_tiling(self, max_unit_per_loop, cut_which_axis, outer_axis):
        """
        `for example:`
        `cut_ho, max_ho_per_loop = 7, ho = 35, nc1 = 13, core_num = 32`
        `case_A: ho_per_loop = 7`
        `        loop_per_nc1 = (ho + ho_per_loop - 1) // ho_per_loop  # loop_per_nc1 = 5`
        `        total_loop = loop_per_nc1 * nc1  # total_loop = 65`
        `        loop_per_core = (total_loop + core_num - 1) // core_num  # loop_per_core = 3`
        `        ho_per_core = ho_per_loop * loop_per_core  # ho_per_core = 21`
        `case_B: ho_per_loop = 5`
        `        loop_per_nc1 = (ho + ho_per_loop - 1) // ho_per_loop  # loop_per_nc1 = 7`
        `        total_loop = loop_per_nc1 * nc1  # total_loop = 91`
        `        loop_per_core = (total_loop + core_num - 1) // core_num  # loop_per_core = 3`
        `        ho_per_core = ho_per_loop * loop_per_core  # ho_per_core = 15`
        case_B be better than case_A
        """
        loop_per_inner = (cut_which_axis + max_unit_per_loop - 1) // max_unit_per_loop
        total_loop = loop_per_inner * outer_axis
        loop_per_core = (total_loop + self.core_num - 1) // self.core_num
        total_loop = loop_per_core * self.core_num
        loop_per_inner = total_loop // outer_axis
        unit_per_loop = (cut_which_axis + loop_per_inner - 1) // loop_per_inner
        return unit_per_loop

    def _only_one_loop(self):
        nc1_per_core = (self.nc1 + self.core_num - 1) // self.core_num
        core_per_nc1 = self.core_num // self.nc1
        if core_per_nc1 <= 1:
            # `if nc1 > core_num / 2: cut nc1, cut_nc1_factor = nc1_per_core, cut_ho_factor = ho_per_core = ho`
            ho_per_core = self.ho
        else:
            # `if nc1 <= core_num / 2: cut ho, cut_nc1_factor = nc1_per_core = 1, cut_ho_foctor = ho_per_core`
            ho_per_core = (self.ho + core_per_nc1 - 1) // core_per_nc1

        block_for_zero_tensor = self.blocks_per_c0
        block_for_argmax_ub = nc1_per_core * self.kh * self.kw * ((ho_per_core * self.wo + 15) // 16)
        block_for_grad_ub = nc1_per_core * ((ho_per_core - 1) * self.stride_h + self.kh) * \
                            (self.wi + self.pad_left + self.pad_right) * self.blocks_per_c0
        block_for_grad_load = nc1_per_core * ho_per_core * self.wo * self.blocks_per_c0
        block_for_res_ub = nc1_per_core * ho_per_core * self.wo * self.blocks_per_c0
        if self.block_num >= block_for_zero_tensor + block_for_argmax_ub + block_for_grad_ub + \
                             block_for_grad_load + block_for_res_ub:
            self.double_buffer_on = False
            self.cut_nc1_factor = nc1_per_core
            self.cut_ho_factor = ho_per_core
            return True
        return False

    def _cut_nc1(self):
        """
        `block_for_zero_tensor = blocks_per_c0`
        `block_for_argmax_ub = nc1_per_loop * kh * kw * ((ho * wo + 15) // 16) * 2`
        `block_for_grad_ub = nc1_per_loop * ((ho - 1) * stride_h + kh) *
                             (wi + pad_left + pad_right) * 2 * blocks_per_c0`
        `block_for_grad_load = nc1_per_loop * ho * wo * blocks_per_c0`
        `block_for_res_ub = nc1_per_loop * ho * wo * 2 * blocks_per_c0`
        """
        if self.nc1 <= self.core_num:
            return False
        max_nc1_per_loop = (self.block_num - self.blocks_per_c0) // \
                           (self.kh * self.kw * ((self.ho * self.wo + 15) // 16) * 2 + \
                            ((self.ho - 1) * self.stride_h + self.kh) * \
                            (self.wi + self.pad_left + self.pad_right) * 2 * self.blocks_per_c0 + \
                            self.ho * self.wo * 3 * self.blocks_per_c0)
        if max_nc1_per_loop >= 1:
            nc1_per_core = (self.nc1 + self.core_num - 1) // self.core_num
            loop_per_core = (nc1_per_core + max_nc1_per_loop - 1) // max_nc1_per_loop
            self.cut_nc1_factor = (nc1_per_core + loop_per_core - 1) // loop_per_core
            return True
        return False

    def _cut_ho(self):
        """
        `block_for_zero_tensor = blocks_per_c0`
        `block_for_argmax_ub = kh * kw * (ho_per_loop * wo / 16 + 1) * 2`
        `block_for_grad_ub = ((ho_per_loop - 1) * stride_h + kh) * ((wo - 1) * stride_w + kw) * 2 * blocks_per_c0`
        `block_for_grad_load = ho_per_loop * wo * blocks_per_c0`
        `block_for_res_ub = ho_per_loop * wo * 2 * blocks_per_c0`
        """
        max_ho_per_loop = int((self.block_num - self.blocks_per_c0 - self.kh * self.kw * 2 - \
                               (self.kh - self.stride_h) * ((self.wo - 1) * self.stride_w + self.kw) * \
                               2 * self.blocks_per_c0) // \
                              (self.kh * self.kw * self.wo / 8 + self.wo * 3 * self.blocks_per_c0 + \
                               self.stride_h * ((self.wo - 1) * self.stride_w + self.kw) * 2 * self.blocks_per_c0))
        if max_ho_per_loop >= 1:
            self.cut_ho_factor = self._optimize_tiling(max_ho_per_loop, self.ho, self.nc1)
            return True
        return False

    def _cut_wo(self):
        """
        `block_for_zero_tensor = blocks_per_c0`
        `block_for_argmax_ub = kh * kw * (wo_per_loop / 16 + 1) * 2`
        `block_for_grad_ub = kh * ((wo_per_loop - 1) * stride_w + kw) * 2 * blocks_per_c0`
        `block_for_grad_load = wo_per_loop * blocks_per_c0`
        `block_for_res_ub = wo_per_loop * 2 * blocks_per_c0`
        """
        max_wo_per_loop = int((self.block_num - self.blocks_per_c0 - \
                               2 * (self.blocks_per_c0 + 1) * self.kh * self.kw + \
                               2 * self.blocks_per_c0 * self.kh * self.stride_w) // \
                              (self.kh * self.kw / 8 + 2 * self.blocks_per_c0 * self.kh * self.stride_w + \
                               3 * self.blocks_per_c0))
        if max_wo_per_loop >= 2:
            # if wo % wo_per_loop == 1, tvm would raise an error
            for wo_per_loop in range(max_wo_per_loop, 1, -1):
                if self.wo % wo_per_loop != 1:
                    self.cut_ho_factor = 1
                    self.cut_wo_factor = wo_per_loop
                    return True
        return False

    def _cut_kh(self):
        """
        `block_for_zero_tensor = blocks_per_c0`
        `block_for_argmax_ub = kh_per_loop * kw * ((wo_per_loop + 15) // 16) * 2`
        `block_for_grad_ub = kh_per_loop * ((wo_per_loop - 1) * stride_w + kw) * 2 * blocks_per_c0`
        `block_for_grad_load = wo_per_loop * blocks_per_c0`
        `block_for_res_ub = wo_per_loop * 2 * blocks_per_c0`
        """
        for wo_per_loop in range(2, self.wo + 1):
            if self.wo % wo_per_loop != 1:
                break
        else:
            return False
        
        max_kh_per_loop = (self.block_num - self.blocks_per_c0 - wo_per_loop * 3 * self.blocks_per_c0) // \
                          (self.kw * ((wo_per_loop + 15) // 16) * 2 + \
                           ((wo_per_loop - 1) * self.stride_w + self.kw) * 2 * self.blocks_per_c0)
        if max_kh_per_loop >= 1:
            loop_per_kh = (self.kh + max_kh_per_loop - 1) // max_kh_per_loop
            kh_per_loop = (self.kh + loop_per_kh - 1) // loop_per_kh

            self.cut_ho_factor = 1
            self.cut_wo_factor = wo_per_loop
            self.cut_kh_factor = kh_per_loop
            return True
        return False

    def _preload(self, grad):
        preload_shape = (
            self.nc1,
            self.hi + self.pad_top + self.pad_bottom,
            self.wi + self.pad_left + self.pad_right,
            self.c0
        )

        grad_data = tvm.compute(
            preload_shape,
            lambda i0, i1, i2, i3:
                tvm.select(i1 >= self.pad_top,
                    tvm.select(i1 < self.hi + self.pad_top,
                        tvm.select(i2 >= self.pad_left,
                            tvm.select(i2 < self.wi + self.pad_left,
                                grad[i0, i1 - self.pad_top, i2 - self.pad_left, i3])))),
            name="grad_data")

        padding_top = tvm.compute(preload_shape,
                                  lambda i0, i1, i2, i3:
                                      tvm.select(i1 < self.pad_top, tvm.const(0.0, self.dtype)),
                                  name="padding_top")

        padding_bottom = tvm.compute(preload_shape,
                                     lambda i0, i1, i2, i3:
                                         tvm.select(i1 >= self.pad_top + self.hi, tvm.const(0.0, self.dtype)),
                                     name="padding_bottom")

        padding_left = tvm.compute(preload_shape,
                                   lambda i0, i1, i2, i3:
                                       tvm.select(i2 < self.pad_left, tvm.const(0.0, self.dtype)),
                                   name="padding_left")

        padding_right = tvm.compute(preload_shape,
                                    lambda i0, i1, i2, i3:
                                        tvm.select(i2 >= self.pad_left + self.wi, tvm.const(0.0, self.dtype)),
                                    name="padding_right")

        padding_compute = (grad_data, padding_top, padding_bottom, padding_left, padding_right)

        grad_ub = tvm.compute(preload_shape,
                              lambda i0, i1, i2, i3:
                                  tvm.select(i1 < self.pad_top,
                                             padding_top[i0, i1, i2, i3],
                                             tvm.select(i1 >= self.pad_top + self.hi,
                                                        padding_bottom[i0, i1, i2, i3],
                                                        tvm.select(i2 < self.pad_left,
                                                                   padding_left[i0, i1, i2, i3],
                                                                   tvm.select(i2 >= self.pad_left + self.wi,
                                                                              padding_right[i0, i1, i2, i3],
                                                                              grad_data[i0, i1, i2, i3])))),
                              name="grad_ub")

        return padding_compute, grad_ub

    def _calculate_tiling_args(self):
        if self._only_one_loop():
            return

        if self._cut_nc1():
            return

        if self._cut_ho():
            return

        if self._cut_wo():
            return

        if self._cut_kh():
            return

        error_manager_vector.raise_err_specific_reson(
            "max_pool_grad_grad_with_argmax",
            "Not supported fmap shape = (%d, %d, %d, %d, %d), kernel = (1, %u, %u, 1), strides = (1, %u, %u, 1)" %
            (self.n, self.c1, self.hi, self.wi, self.c0, self.kh, self.kw, self.stride_h, self.stride_w))

    def _compute(self):
        grad_shape = (self.nc1, self.hi, self.wi, self.c0)
        x = tvm.placeholder(grad_shape, dtype=self.dtype, name="x")
        grad = tvm.placeholder(grad_shape, dtype=self.dtype, name="grad")

        # argmax's axis howo is aligned to 16
        howo_align = (self.ho * self.wo + 15) // 16 * 16
        # argmax in gm is 16 bytes bigger per kw for storing some args
        argmax_shape_gm = (self.nc1, self.kh, self.kw, howo_align + 16, self.c0)
        argmax = tvm.placeholder(argmax_shape_gm, dtype="bool", name="argmax")

        zero_tensor = tvm.compute((self.c0, ), lambda i: tvm.const(0.0, self.dtype), name="zero_tensor")

        argmax_shape_ub = (self.nc1, self.kh, self.kw, howo_align, self.c0)
        argmax_ub = tvm.compute(argmax_shape_ub, lambda *i: argmax(*i), name="argmax_ub")

        padding_compute, grad_ub = self._preload(grad)

        load_shape = (self.nc1, self.kh, self.kw, self.ho, self.wo, self.c0)
        grad_load = tvm.compute(
            load_shape,
            lambda i0, i1, i2, i3, i4, i5:
                grad_ub[i0, i3 * self.stride_h + i1, i4 * self.stride_w + i2, i5] + tvm.const(0.0, self.dtype),
            name="grad_load")

        grad_select = tvm.compute(load_shape,
                                  lambda i0, i1, i2, i3, i4, i5:
                                      tvm.select(argmax_ub[i0, i1, i2, i3 * self.wo + i4, i5],
                                                 grad_load[i0, i1, i2, i3, i4, i5],
                                                 zero_tensor[i5]),
                                  name="grad_select")

        reduce_kh = tvm.reduce_axis((0, self.kh), "reduce_kh")
        reduce_kw = tvm.reduce_axis((0, self.kw), "reduce_kw")
        res_shape = (self.nc1, self.ho, self.wo, self.c0)
        res_ub = tvm.compute(res_shape,
                             lambda i0, i1, i2, i3:
                                 tvm.sum(grad_select[i0, reduce_kh, reduce_kw, i1, i2, i3],
                                         axis=[reduce_kh, reduce_kw]),
                             name="res_ub")

        res = tvm.compute(res_shape, lambda *i: res_ub(*i), name="res")

        compute_list = [zero_tensor, argmax_ub, padding_compute, grad_ub, grad_load, grad_select, res_ub, res]
        data_list = [x, grad, argmax, res]

        return compute_list, data_list

    # 'pylint: disable=too-many-lines
    def _schedule(self, compute_list):
        zero_tensor, argmax_ub, padding_compute, grad_ub, grad_load, grad_select, res_ub, res = compute_list

        sch = tvm.create_schedule(res.op)

        # `axis[0]: nc1, axis[1]: ho, axis[2]: wo, axis[3]: c0`
        res_nc1_outer, res_nc1_inner = sch[res].split(res.op.axis[0], factor=self.cut_nc1_factor)
        res_ho_outer, res_ho_inner = sch[res].split(res.op.axis[1], factor=self.cut_ho_factor)
        res_wo_outer, res_wo_inner = sch[res].split(res.op.axis[2], factor=self.cut_wo_factor)
        sch[res].reorder(res_nc1_outer, res_ho_outer, res_wo_outer,
                         res_nc1_inner, res_ho_inner, res_wo_inner, res.op.axis[3])
        res_fuse_nc1ho_outer = sch[res].fuse(res_nc1_outer, res_ho_outer)

        # `axis[0]: nc1, axis[1]: ho, axis[2]: wo, axis[3]: c0`
        res_ub_nc1_outer, res_ub_nc1_inner = sch[res_ub].split(res_ub.op.axis[0], factor=self.cut_nc1_factor)
        res_ub_ho_outer, res_ub_ho_inner = sch[res_ub].split(res_ub.op.axis[1], factor=self.cut_ho_factor)
        res_ub_wo_outer, res_ub_wo_inner = sch[res_ub].split(res_ub.op.axis[2], factor=self.cut_wo_factor)
        res_ub_kh_outer, res_ub_kh_inner = sch[res_ub].split(res_ub.op.reduce_axis[0], factor=self.cut_kh_factor)
        sch[res_ub].reorder(res_ub_nc1_outer, res_ub_ho_outer, res_ub_wo_outer, res_ub_kh_outer,
                            res_ub_kh_inner, res_ub.op.reduce_axis[1],
                            res_ub_nc1_inner, res_ub_ho_inner, res_ub_wo_inner, res_ub.op.axis[3])

        sch[zero_tensor].set_scope(te.platform.scope_ubuf)
        sch[argmax_ub].set_scope(te.platform.scope_ubuf)
        for i in padding_compute:
            sch[i].set_scope(te.platform.scope_ubuf)
        sch[grad_ub].set_scope(te.platform.scope_ubuf)
        sch[grad_load].set_scope(te.platform.scope_ubuf)
        sch[grad_select].set_scope(te.platform.scope_ubuf)
        sch[res_ub].set_scope(te.platform.scope_ubuf)

        for i in padding_compute:
            sch[i].reused_by(grad_ub)
        sch[grad_load].reused_by(grad_select)

        # `axis[0]: nc1, axis[1]: kh, axis[2]: kw, axis[3]: howo, axis[4]: c0`
        sch[argmax_ub].storage_align(argmax_ub.op.axis[2], 256, 0)

        sch[argmax_ub].compute_at(sch[res_ub], res_ub_kh_outer)
        for i in padding_compute:
            sch[i].compute_at(sch[res_ub], res_ub_kh_outer)
        sch[grad_ub].compute_at(sch[res_ub], res_ub_kh_outer)
        sch[grad_load].compute_at(sch[res_ub], res_ub.op.reduce_axis[1])
        sch[grad_select].compute_at(sch[res_ub], res_ub.op.reduce_axis[1])
        sch[res_ub].compute_at(sch[res], res_wo_outer)

        sch[zero_tensor].emit_insn(zero_tensor.op.axis[0], te.platform.DUP)
        sch[argmax_ub].emit_insn(argmax_ub.op.axis[0], te.platform.DMA_COPY)
        for i in padding_compute:
            sch[i].emit_insn(i.op.axis[0], te.platform.DMA_COPY, {"split_select": 1})
        sch[grad_ub].emit_insn(grad_ub.op.axis[0], te.platform.DMA_COPY)
        sch[grad_load].emit_insn(grad_load.op.axis[0], "vector_adds")
        sch[grad_select].emit_insn(grad_select.op.axis[0], te.platform.SELECT)
        sch[res_ub].emit_insn(res_ub_nc1_inner, te.platform.REDUCE_SUM)
        sch[res].emit_insn(res_nc1_inner, te.platform.DMA_COPY)

        if self.double_buffer_on:
            sch[argmax_ub].double_buffer()
            for i in padding_compute:
                sch[i].double_buffer()
            sch[grad_ub].double_buffer()
            sch[res_ub].double_buffer()

        sch[res].bind(res_fuse_nc1ho_outer, tvm.thread_axis("blockIdx.x"))

        return sch
