#!/usr/bin/python
# -*- coding: utf-8 -*-
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
centralization.py
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util import util_common
from impl.util import util_select_op_base


# 'pylint: disable=unused-argument,invalid-name,too-many-locals,too-many-statements
def op_select_format(x, y, axes, kernel_name="centralization"):
    """
    algorithm: op_select_format charge the dtype and format for centralization

    Parameters
    ----------
    x: dict
        contains shape and dtype information of input tensor
    y: dict
        contains shape and dtype information of output tensor
    kernel_name: str
        cce kernel name, default value is "slice_d".

    Returns
    -------
    None
    """
    centralization_object = Centralization(x, y, axes, kernel_name)
    # charge whether support FRACTAL_Z
    is_support_fz = centralization_object.check_tik_fz_supported()
    # charge whether support FRACTAL_NZ
    is_support_nz = centralization_object.check_tik_nz_supported()

    base_data_type = ["float", "float16"]
    dtype_base_out = base_data_type[:]
    format_base_out = ["ND"] * len(base_data_type)

    if is_support_fz:
        other_format = "FRACTAL_Z"
        dtype_base_out = dtype_base_out + base_data_type
        format_base_out = format_base_out + [other_format] * len(base_data_type)
    if is_support_nz:
        other_format = "FRACTAL_NZ"
        dtype_base_out = dtype_base_out + base_data_type
        format_base_out = format_base_out + [other_format] * len(base_data_type)

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    input0 = util_select_op_base.gen_param(
        classify="input0", name="x", datatype=dtype_str, format=format_str)
    output0 = util_select_op_base.gen_param(
        classify="output0", name="y", datatype=dtype_str, format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def get_all_num(shape):
    """
    get_all_num
    """
    total_num = 1
    for i in shape:
        total_num = i * total_num
    return total_num


def reduce_ub_half(tik_instance, vadd_ub, vector_num, vcetor_mask, result_ub=None):
    """
    reduce_ub_half
    """
    if result_ub is None:
        result_ub = vadd_ub
    if vector_num == 1:
        return 1
    output_num = vector_num // 2
    ub2_offset = output_num * vcetor_mask

    tik_instance.vadd(vcetor_mask, result_ub, vadd_ub, vadd_ub[ub2_offset],
                      output_num, 1, 1, 1, 8, 8, 8)

    if vector_num % 2 == 1:
        tik_instance.vadd(vcetor_mask, result_ub, result_ub, vadd_ub[ub2_offset * 2],
                          1, 1, 1, 1, 8, 8, 8)
    return output_num


def reduce_to_64(tik_instance, vadd_ub, repeat_num, vcetor_mask, result_ub=None):
    """
    reduce_to_64
    """
    if result_ub is None:
        while repeat_num > 1:
            repeat_num = reduce_ub_half(tik_instance, vadd_ub, repeat_num, vcetor_mask)
    else:
        if repeat_num > 1:
            repeat_num = reduce_ub_half(tik_instance, vadd_ub, repeat_num, vcetor_mask, result_ub)
        while repeat_num > 1:
            repeat_num = reduce_ub_half(tik_instance, result_ub, repeat_num, vcetor_mask)


def reduce_vector_to_c0(tik_instance, vector_ub, reduce_result, vector_num):
    """
    reduce_vector_to_c0
    """
    shape_c0 = 16
    loop_num = vector_num // shape_c0
    for i in range(loop_num):
        tik_instance.vadd(shape_c0, reduce_result, reduce_result, vector_ub[i * shape_c0],
                          1, 1, 1, 1, 8, 8, 8)


def redifine_reduce_case(input_shape, input_axis):
    """
    redifine_reduce_case
    """
    shape_len = len(input_shape)
    shape_one = [False] * shape_len
    for _axis in input_axis:
        shape_one[_axis % shape_len] = True

    new_shape = []
    new_axis = []
    shape_dim = 1
    for i, reduce_flag in enumerate(shape_one):
        if i == 0:
            shape_dim = input_shape[i]
            continue

        before_reduce_flag = shape_one[i - 1]
        if before_reduce_flag != reduce_flag:
            if not reduce_flag:
                new_axis.append(len(new_shape))
            # save before data
            new_shape.append(shape_dim)
            shape_dim = 1

        shape_dim = shape_dim * input_shape[i]

        if i == shape_len - 1:
            if reduce_flag:
                new_axis.append(len(new_shape))
            new_shape.append(shape_dim)

    return new_shape, new_axis


# 'pylint: disable=too-many-instance-attributes
class Centralization:
    """
    Class for Dynamic shape operator Assign
    """
    def __init__(self, x, y, axes, kernel_name):
        self.tik_instance = tik.Tik()
        self.x_shape = x.get("shape")
        self.x_ori_shape = x.get("ori_shape")
        self.input_ori_foramt = x.get("ori_format")
        self.input_foramt = x.get("format")
        self.x_dtype = x.get("dtype")
        dim_size = 1
        for axis in axes:
            dim_size = dim_size * self.x_ori_shape[axis]
        self.reduce_scalar = 1 / float(dim_size)
        self.x_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape, name="x_gm", scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape, name="y_gm", scope=tik.scope_gm)

        self.axes = list(axes)
        self.kernel_name = kernel_name

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        self.vector_num = 64 if self.x_dtype == "float32" else 128
        self.block_num = self.vector_num // 8
        self.input_ori_n = 0
        self.input_ori_c = 0
        self.input_ori_h = 0
        self.input_ori_w = 0

    def check_tik_nz_supported(self):
        """
        check_tik_nz_supported
        """
        # nz case supported
        if len(self.x_ori_shape) == 2 and len(self.axes) == 1 and self.axes[0] in (0,):
            return True
        if len(self.x_ori_shape) == 2 and len(self.axes) == 1 and self.axes[0] in (1,):
            return True

        return False

    def check_tik_fz_supported(self):
        """
        check_tik_fz_supported
        """
        hd_support_format = util_common.get_fused_format_str(["N", "H", "W", "C"])
        # fz case supported
        if self.input_ori_foramt in hd_support_format and len(self.x_ori_shape) == 4:
            reduce_format = []
            for axis in self.axes:
                reduce_format.append(self.input_ori_foramt[axis])
            reduce_format = list(set(reduce_format))
            reduce_format.sort()
            if reduce_format == ['C', 'H', 'W']:
                return True
        return False

    def _compute_nz_one_core(self, copy_offset_for_one_core):
        """
        _compute_nz_one_core
        """
        process_data = 20480 // 2
        reduce_result_c0 = self.tik_instance.Tensor(self.x_dtype, (self.vector_num,),
                                                    name="reduce_result", scope=tik.scope_ubuf)
        reduce_result_vector = self.tik_instance.Tensor(self.x_dtype, (self.vector_num,),
                                                        name="reduce_result", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.vector_num, reduce_result_c0, 0.0, 1, 1, 8)
        self.tik_instance.vector_dup(self.vector_num, reduce_result_vector, 0.0, 1, 1, 8)
        ping_ub = self.tik_instance.Tensor(self.x_dtype, (process_data,),
                                           name="ping_ub", scope=tik.scope_ubuf)
        pang_ub = self.tik_instance.Tensor(self.x_dtype, (process_data,),
                                           name="pang_ub", scope=tik.scope_ubuf)
        total_num = self.x_ori_shape[0] * 16
        sigment_loop = total_num // process_data
        sigment_tail = total_num % process_data

        def _run_sum(_ub, _copy_offset, vector_repeat, copy_block):
            if copy_block % 8 != 0:
                self.tik_instance.vector_dup(self.vector_num, _ub[(vector_repeat - 1) * self.vector_num], 0.0, 1, 1, 8)

            self.tik_instance.data_move(_ub, self.x_gm[_copy_offset],
                                        0, 1, copy_block, 0, 0)
            reduce_to_64(self.tik_instance, _ub, vector_repeat, self.vector_num)
            # add to reduce_result_vector
            self.tik_instance.vadd(self.vector_num, reduce_result_vector, reduce_result_vector,
                                   _ub, 1, 1, 1, 1, 8, 8, 8)

        repeat = (process_data + self.vector_num - 1) // self.vector_num
        block = (process_data + (self.vector_num // 8) - 1) // (self.vector_num // 8)
        with self.tik_instance.for_range(0, sigment_loop // 2) as _loop_idx:
            _idx = _loop_idx * 2
            new_offset = copy_offset_for_one_core + _idx * process_data
            _run_sum(ping_ub, new_offset, repeat, block)
            _idx = _loop_idx * 2 + 1
            new_offset = copy_offset_for_one_core + _idx * process_data
            _run_sum(pang_ub, new_offset, repeat, block)

        if sigment_loop % 2 != 0:
            _idx = sigment_loop - 1
            new_offset = copy_offset_for_one_core + _idx * process_data
            _run_sum(ping_ub, new_offset, repeat, block)

        if sigment_tail != 0:
            _idx = sigment_loop
            repeat = (sigment_tail + self.vector_num - 1) // self.vector_num
            block = (sigment_tail + (self.vector_num // 8) - 1) // (self.vector_num // 8)
            new_offset = copy_offset_for_one_core + _idx * process_data
            _run_sum(pang_ub, new_offset, repeat, block)

        # calcu c0 sum
        reduce_vector_to_c0(self.tik_instance, reduce_result_vector, reduce_result_c0, self.vector_num)
        for i in range(self.vector_num // 16 - 1):
            self.tik_instance.vmuls(16, reduce_result_c0[16 + i * 16], reduce_result_c0, 1.0, 1, 1, 1, 8, 8)
        self.tik_instance.vmuls(self.vector_num, reduce_result_c0, reduce_result_c0,
                                self.reduce_scalar, 1, 1, 1, 8, 8)

        # do vsub
        def _run_sub(_ub, _copy_offset, vector_repeat, copy_block):
            self.tik_instance.data_move(_ub, self.x_gm[_copy_offset],
                                        0, 1, copy_block, 0, 0)
            self.tik_instance.vsub(self.vector_num, _ub, _ub,
                                   reduce_result_c0, vector_repeat, 1, 1, 1, 8, 8, 0)
            self.tik_instance.data_move(self.y_gm[_copy_offset], _ub,
                                        0, 1, copy_block, 0, 0)
        repeat = (process_data + self.vector_num - 1) // self.vector_num
        block = (process_data + (self.vector_num // 8) - 1) // (self.vector_num // 8)
        with self.tik_instance.for_range(0, sigment_loop // 2) as _loop_idx:
            _idx = _loop_idx * 2
            new_offset = copy_offset_for_one_core + _idx * process_data
            _run_sub(ping_ub, new_offset, repeat, block)
            _idx = _loop_idx * 2 + 1
            new_offset = copy_offset_for_one_core + _idx * process_data
            _run_sub(pang_ub, new_offset, repeat, block)

        if sigment_loop % 2 != 0:
            _idx = sigment_loop - 1
            new_offset = copy_offset_for_one_core + _idx * process_data
            _run_sub(ping_ub, new_offset, repeat, block)

        if sigment_tail != 0:
            _idx = sigment_loop
            repeat = (sigment_tail + self.vector_num - 1) // self.vector_num
            block = (sigment_tail + (self.vector_num // 8) - 1) // (self.vector_num // 8)
            new_offset = copy_offset_for_one_core + _idx * process_data
            _run_sub(pang_ub, new_offset, repeat, block)

    def compute_nz(self):
        """
        compute_nz
        """
        # calcu core
        self.x_shape, _ = redifine_reduce_case(self.x_shape, [1, 2])
        if get_all_num(self.x_shape[1:]) != self.x_ori_shape[0] * 16:
            self.y_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape,
                                                 name="y_gm", scope=tik.scope_gm,
                                                 is_atomic_add=True)
        first_dim = self.x_shape[0]
        reduce_dim = self.x_shape[1]
        last_dim = self.x_shape[2]
        one_core_dim = (first_dim + self.ai_core_num - 1) // self.ai_core_num
        used_core_num = (first_dim + one_core_dim - 1) // one_core_dim
        tail_core = first_dim - (used_core_num - 1) * one_core_dim
        with self.tik_instance.for_range(0, used_core_num) as _core_idx:
            with self.tik_instance.if_scope(_core_idx < used_core_num - 1):
                with self.tik_instance.for_range(0, one_core_dim) as _dim_idx:
                    copy_offset = (_core_idx * one_core_dim + _dim_idx) * reduce_dim * last_dim
                    self._compute_nz_one_core(copy_offset)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, tail_core) as _dim_idx:
                    copy_offset = (_core_idx * one_core_dim + _dim_idx) * reduce_dim * last_dim
                    self._compute_nz_one_core(copy_offset)

    def _do_mask_for_fz(self, mask_ub, mask_ub_offset, pad_first_dim_num):
        """
        _do_mask_for_fz
        """
        shape_c0_value = 16
        if pad_first_dim_num != 0:
            pad_c0_num = self.input_ori_h * self.input_ori_w
            pad_mask_c0 = "1" * (shape_c0_value - pad_first_dim_num) + "0" * pad_first_dim_num
            pad_mask_vector = pad_mask_c0 * (self.vector_num // shape_c0_value)
            pad_block_num = pad_c0_num * (shape_c0_value // self.block_num)
            if pad_block_num // 8 > 0:
                pad_mask_dec_0 = int(pad_mask_vector, 2)
                pad_mask_dec_1 = int(pad_mask_vector, 2)
                if mask_ub.dtype in ("float32",):
                    pad_mask_dec_0 = 0
                self.tik_instance.vector_dup([pad_mask_dec_0, pad_mask_dec_1], mask_ub[mask_ub_offset],
                                             0.0, pad_block_num // 8, 1, 8)
            if pad_block_num % 8 > 0:
                mask_ub_offset = mask_ub_offset + (pad_block_num // 8) * self.vector_num
                tail_pad_c0_num = (pad_block_num % 8) // (shape_c0_value // self.block_num)
                pad_mask_vector = pad_mask_c0 * tail_pad_c0_num
                pad_mask_0 = "0"
                pad_mask_1 = pad_mask_vector
                if len(pad_mask_vector) > 64:
                    pad_mask_0 = pad_mask_vector[0:len(pad_mask_vector) - 64]
                    pad_mask_1 = pad_mask_vector[0:64]
                pad_mask_dec_0 = int(pad_mask_0, 2)
                pad_mask_dec_1 = int(pad_mask_1, 2)
                self.tik_instance.vector_dup([pad_mask_dec_0, pad_mask_dec_1], mask_ub[mask_ub_offset],
                                             0.0, 1, 1, 8)

    def _compute_fz_one_by_one(self, core_pro_dims, copy_offset):
        """
        _compute_fz_one_by_one
        """
        first_dim, second_dim, last_dim = self.x_shape
        process_n_num = 1
        ori_c_dim = self.input_ori_c
        copy_burst_len = last_dim * process_n_num
        copy_burst_num = first_dim
        copy_burst_offset = second_dim * last_dim - copy_burst_len
        copy_burst_len_block = copy_burst_len // self.block_num
        copy_burst_offset_block = copy_burst_offset // self.block_num

        def _run_idx(run_dim_idx):
            shape_c0_value = 16
            copy_new_offset = (copy_offset + run_dim_idx) * last_dim
            ub_size = \
                ((copy_burst_len * copy_burst_num + self.vector_num - 1) // self.vector_num) * self.vector_num
            ping_ub = self.tik_instance.Tensor(self.x_dtype, (ub_size,),
                                               name="ping_ub", scope=tik.scope_ubuf)
            ping_ub_1 = self.tik_instance.Tensor(self.x_dtype, (ub_size,),
                                                 name="ping_ub_1", scope=tik.scope_ubuf)
            vector_repeat = (copy_burst_len * copy_burst_num + self.vector_num - 1) // self.vector_num
            if (copy_burst_len * copy_burst_num) % self.vector_num != 0:
                self.tik_instance.vector_dup(self.vector_num, ping_ub_1[(vector_repeat - 1) * self.vector_num],
                                             0.0, 1, 1, 8)
            self.tik_instance.data_move(ping_ub_1, self.x_gm[copy_new_offset],
                                        0, copy_burst_num, copy_burst_len_block, copy_burst_offset_block, 0)
            # do pad
            mask_offset = (first_dim - self.input_ori_h * self.input_ori_w) * shape_c0_value
            self._do_mask_for_fz(ping_ub_1, mask_offset, ori_c_dim % shape_c0_value)
            # add to one vector
            if vector_repeat > 1:
                reduce_to_64(self.tik_instance, ping_ub_1, vector_repeat, self.vector_num, ping_ub)
                src_ub = ping_ub
            else:
                src_ub = ping_ub_1
            if self.x_dtype == "float32":
                # add to one block
                self.tik_instance.vadd(self.block_num * 4, ping_ub, src_ub[self.block_num * 4:],
                                       src_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vadd(self.block_num * 2, ping_ub, ping_ub[self.block_num * 2:],
                                       ping_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vadd(self.block_num * 1, ping_ub, ping_ub[self.block_num * 1:],
                                       ping_ub, 1, 1, 1, 1, 8, 8, 8)
                # add to one num
                for i in range(self.block_num - 1):
                    scalar_tmp = self.tik_instance.Scalar(self.x_dtype, name="scalar_tmp")
                    scalar_tmp.set_as(ping_ub[i + 1])
                    ping_ub[(i + 1) * self.block_num].set_as(scalar_tmp)
                for i in range(self.block_num - 1):
                    self.tik_instance.vadd(1, ping_ub, ping_ub[self.block_num * (i + 1):],
                                           ping_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vmuls(1, ping_ub, ping_ub, -1.0 * self.reduce_scalar, 1, 1, 1, 8, 8)
            else:
                with self.tik_instance.new_stmt_scope():
                    des_ub = self.tik_instance.Tensor(self.x_dtype, (self.block_num,),
                                                      name="des_ub", scope=tik.scope_ubuf)
                    work_ub = self.tik_instance.Tensor(self.x_dtype, (self.block_num,),
                                                       name="work_ub", scope=tik.scope_ubuf)
                    self.tik_instance.vec_reduce_add(self.vector_num, des_ub, src_ub, work_ub,
                                                     1, 8)
                    self.tik_instance.vmuls(1, ping_ub, des_ub, -1.0 * self.reduce_scalar, 1, 1, 1, 8, 8)

            scalar_sum = self.tik_instance.Scalar(self.x_dtype, name="scalar_sum")
            scalar_sum.set_as(ping_ub[0])

            self.tik_instance.vadds(self.vector_num, ping_ub_1, ping_ub_1, scalar_sum, vector_repeat, 1, 1, 8, 8)
            self._do_mask_for_fz(ping_ub_1, mask_offset, ori_c_dim % shape_c0_value)
            self.tik_instance.data_move(self.y_gm[copy_new_offset], ping_ub_1,
                                        0, copy_burst_num, copy_burst_len_block, 0, copy_burst_offset_block)

        with self.tik_instance.for_range(0, core_pro_dims) as idx:
            _run_idx(idx)

    def _compute_fz_one_core(self, core_idx, core_pro_dims, dim_offset):
        copy_offset = core_idx * dim_offset
        self._compute_fz_one_by_one(core_pro_dims, copy_offset)

    def compute_fz(self):
        """
        compute_fz
        """
        dict_zip_shape = dict(zip(list(self.input_ori_foramt), self.x_ori_shape))
        self.input_ori_n = dict_zip_shape["N"]
        self.input_ori_c = dict_zip_shape["C"]
        self.input_ori_h = dict_zip_shape["H"]
        self.input_ori_w = dict_zip_shape["W"]
        # calcu core
        self.x_shape, _ = redifine_reduce_case(self.x_shape, [0, 3])
        second_dim = self.x_shape[1]
        ori_second_dim = self.input_ori_n
        if second_dim != ori_second_dim:
            self.y_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape,
                                                 name="y_gm", scope=tik.scope_gm,
                                                 is_atomic_add=True)
            second_dim = ori_second_dim
        one_core_dim = (second_dim + self.ai_core_num - 1) // self.ai_core_num
        used_core_num = (second_dim + one_core_dim - 1) // one_core_dim
        tail_core = second_dim - (used_core_num - 1) * one_core_dim
        with self.tik_instance.for_range(0, used_core_num, block_num=used_core_num) as _core_idx:
            with self.tik_instance.if_scope(_core_idx < used_core_num - 1):
                self._compute_fz_one_core(_core_idx, one_core_dim, one_core_dim)
            with self.tik_instance.else_scope():
                self._compute_fz_one_core(_core_idx, tail_core, one_core_dim)

    def centralization_compute(self):
        """
        The tik implementation of operator centralization
        """
        if self.input_foramt in ("FRACTAL_Z",):
            # FRACTAL_Z, mean: C1HWNiNoC0
            self.compute_fz()
        elif self.input_foramt in ("FRACTAL_NZ",):
            # FRACTAL_NZ, mean: [A, B] -> [ceil(B//16), ceil(A//16), 16, 16]
            # [A, B] with axis = 0 --> nz dim with axis = 1, 2
            # [A, B] with axis = 1 --> nz dim with axis = 0, 3
            if self.axes[0] == 0:
                self.compute_nz()
            else:
                self.input_ori_foramt = "NCHW"
                self.x_ori_shape = list(self.x_ori_shape) + [1, 1]
                self.compute_fz()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x_gm,),
                                   outputs=(self.y_gm,))


def centralization_default(x, axes, kernel_name):
    """
    centralization_default function
     use DSL to calcu
    """
    x_shape = x.get("shape")
    x_dtype = x.get("dtype")
    x_data = tvm.placeholder(x_shape, name="x_data", dtype=x_dtype)
    reduce_elts = 1.0
    for axis in axes:
        reduce_elts = reduce_elts * x_shape[axis]
    cof = reduce_elts ** (-1)
    x_data_vmul = tbe.vmuls(x_data, cof)
    x_reduce = tbe.sum(x_data_vmul, axis=axes, keepdims=True)
    x_reduce_broadcast = tbe.broadcast(x_reduce, x_shape)
    res = tbe.vsub(x_data, x_reduce_broadcast)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [x_data, res]}
    tbe.build(sch, config)


def centralization(x, y, axes, kernel_name="centralization"):
    """
    algorithm: assign
    calculating: update 'ref' by assigning 'value' to it

    Parameters
    ----------
    x: dict
        dict of input_ref, include shape and dtype,
    y: dict
        dict of input_value, include shape and dtype,
        Must have the same shape and dtype as input_ref
    axes: axes
        list for reduce axis
    kernel_name : str
        cce kernel name, default value is assign

    Returns
    -------
    None
    """
    input_foramt = x.get("format")
    if input_foramt in ("FRACTAL_NZ", "FRACTAL_Z"):
        gc_object = Centralization(x, y, axes, kernel_name)
        gc_object.centralization_compute()
    else:
        centralization_default(x, axes, kernel_name)
