#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
ln_dropout_grad
"""
import functools

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import OpImplMode
from impl.util import util_tik_comm_func
from impl.util import util_select_op_base
from impl.util.util_common import check_op_impl_mode
from impl.util.util_binary import get_bit_len


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes,too-many-arguments,unused-argument
class Constant:
    """
    The class for constant
    """
    # `1 Byte = 8 bit'
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = 32
    # elements of one block for float32
    BLOCK_NUM_FP32 = 8
    # elements of one block for float16
    BLOCK_NUM_FP16 = 16
    # mask for float32
    MASK_FP32 = 64
    # mask for float32
    MASK_FP16 = 128
    # bytes of one block
    BLOCK_BYTES = 32
    FP_32 = "float32"
    FP_16 = "float16"


class TilingParams:
    def __init__(self):
        self.batch_dim = 1
        self.c1 = 1
        self.mid_dim = 1
        self.c0 = 16
        self.block_num = 1
        self.block_factor = 1
        self.block_tail = 0
        self.ub_factor = 16 * 1024
        self.loop_num_pre = 0
        self.tail_num_pre = 0
        self.loop_num_post = 0
        self.tail_num_post = 0
        self.mid_factor = 1
        self.c0_block_num = 1
        self.c0_factor = 1
        self.thread_num = 2
        self.x_ub_dims = (1,)
        self.var_ub_dims = (1,)
        self.brd_var_ub_dims = (1,)
        self.beta_ub_dims = (1,)

    def set_dims(self, shape_x):
        self.batch_dim = shape_x[0]
        self.c1 = shape_x[1]
        self.mid_dim = shape_x[2]
        self.c0 = shape_x[-1]

    def refine_mid_factor(self):
        self.mid_factor = self.ub_factor // (self.c1 * self.c0)
        if self.mid_factor <= 0:
            self.mid_factor = 1
        if self.mid_factor > Constant.BLOCK_NUM_FP16:
            self.mid_factor = Constant.BLOCK_NUM_FP16

    def calc_tiling_params(self, logic_core_num, ub_size, ori_dtype, reduce_axis_size, act_format):
        self._calc_block_tiling_params(logic_core_num)
        self._calc_ub_tiling_params(ub_size, ori_dtype, reduce_axis_size, act_format)

    def set_ub_dims(self, act_format, d_type):
        mask_num = Constant.MASK_FP16 if d_type == "float16" else Constant.MASK_FP32
        self.c0_factor = self.c0 if act_format == "FRACTAL_NZ" else mask_num
        self.x_ub_dims = (self.c1 * self.mid_factor * self.c0,)
        self.var_ub_dims = (1 * self.mid_factor * 1,)
        self.brd_var_ub_dims = (1 * self.mid_factor * self.c0_factor,)
        self.beta_ub_dims = (self.c1 * 1 * self.c0,)

    def _calc_block_tiling_params(self, logic_core_num):
        if self.mid_dim >= logic_core_num:
            self.block_num = logic_core_num
            self.block_factor = self.mid_dim // logic_core_num
            self.block_tail = self.mid_dim % logic_core_num
        else:
            self.block_num = self.mid_dim
            self.block_factor = 1
            self.block_tail = 0

    def _calc_ub_factor(self, ub_size, ori_dtype, reduce_axis_size, act_format):
        inner_bytes_size = get_bit_len(ori_dtype) // Constant.EIGHT_BIT
        self.c0_block_num = (self.c0 * inner_bytes_size) // Constant.BLOCK_BYTES
        # Calculated according to the op compute, there are 4 copies coexist at least.
        max_survival_nodes = 4
        inner_bytes_size = get_bit_len(Constant.FP_32) // Constant.EIGHT_BIT
        # Considering that 16 * 16 fractal is required to enable mad instruction,
        # the mad instruction can be enabled with the double buffer enabled when the reduce size is less than
        # or equal to 512, the mad instruction can be enabled without the double buffer enabled when the reduce
        # size is between 512 and 1024, the mad instruction cannot be enabled when the reduce size exceeds 1024.
        min_size = 512
        max_size = 1024
        if act_format == "FRACTAL_NZ":
            if min_size < reduce_axis_size <= max_size:
                self.thread_num = 1

        self.ub_factor = ub_size // inner_bytes_size // max_survival_nodes // self.thread_num
        self.ub_factor = util_tik_comm_func.floor_align(self.ub_factor, self.c1 * self.c0)

    def _calc_ub_tiling_params(self, ub_size, ori_dtype, reduce_axis_size, act_format):
        self._calc_ub_factor(ub_size, ori_dtype, reduce_axis_size, act_format)
        self.refine_mid_factor()

        self.loop_num_pre = (self.block_factor + 1) // self.mid_factor
        self.tail_num_pre = (self.block_factor + 1) % self.mid_factor
        self.loop_num_post = self.block_factor // self.mid_factor
        self.tail_num_post = self.block_factor % self.mid_factor


class LayerNormGradV2(util_tik_comm_func.OpBase):
    """
        Function: use to store LayerNormGradV2 base parameters
    """

    def __init__(self, keep_prob, kernel_name):
        """
        constructor of LayerNormGradV2
        """
        util_tik_comm_func.OpBase.__init__(self)

        self.keep_prop = keep_prob
        self.ori_dtype = Constant.FP_16
        self.act_dtype = Constant.FP_32
        self.params_axis = 0
        self.reduce_axis = 0
        self.mean_num = 1.0
        self.kernel_name = kernel_name
        self.impl_mode = "high_performance"
        self.act_format = "ND"
        self.share_l0b = None
        self.temp_fp16_ub = None
        self._mask = None
        self.tiling_params = TilingParams()

    def ln_dropout_grad_compute(self):
        with self.tik_instance.for_range(0, self.tiling_params.block_num, name="core_idx",
                                         block_num=self.tiling_params.block_num) as core_idx:
            with self.tik_instance.if_scope(core_idx < self.tiling_params.block_tail):
                core_offset = core_idx * (self.tiling_params.block_factor + 1)
                self._loop_func(self.tiling_params.loop_num_pre, self.tiling_params.tail_num_pre, core_offset)
            with self.tik_instance.else_scope():
                core_offset = (self.tiling_params.block_tail * (self.tiling_params.block_factor + 1) +
                               (core_idx - self.tiling_params.block_tail) * self.tiling_params.block_factor)
                self._loop_func(self.tiling_params.loop_num_post, self.tiling_params.tail_num_post, core_offset)

        self.op_build_cce()

    def prepare(self, input_dict_list, output_dict_list, act_format, impl_mode):
        self._init_src_dst_gm(input_dict_list, output_dict_list)
        shape_x = input_dict_list[1].get("shape")
        shape_mean = input_dict_list[3].get("shape")
        shape_gamma = input_dict_list[4].get("shape")
        self.impl_mode = impl_mode
        self.act_format = act_format
        self.ori_dtype = input_dict_list[1].get("dtype").lower()
        self.act_dtype = Constant.FP_32 if self._is_cast_to_fp32() else self.ori_dtype
        reduce_axis_size = int(functools.reduce(lambda i, j: i * j, shape_gamma))

        self._get_params_axis(shape_x, shape_gamma)
        self._get_reduce_params(shape_x, shape_mean)

        self.tiling_params.set_dims(shape_x)
        self.tiling_params.calc_tiling_params(self.core_nums, self.ub_size_bytes,
                                              self.ori_dtype, reduce_axis_size, self.act_format)
        self.tiling_params.set_ub_dims(self.act_format, self.act_dtype)

        self._mask = input_dict_list[5]

    def _init_src_dst_gm(self, input_dict_list, output_dict_list):
        output_dict_list[2]["is_atomic_add"] = True
        output_dict_list[3]["is_atomic_add"] = True
        self.op_init_gm(input_dict_list, output_dict_list)

    def _get_params_axis(self, shape_x, shape_gamma):
        params_axis_tmp = []
        if len(shape_x) != len(shape_gamma):
            sub = len(shape_x) - len(shape_gamma)
            for i in range(sub):
                params_axis_tmp.append(i)

        self.params_axis = tuple(params_axis_tmp)

    def _get_reduce_params(self, shape_x, shape_mean):
        """
        compute parameters including reduce_axis and mean_num
        """
        reduce_axis_tmp = []
        for i, (x_size, mean_size) in enumerate(zip(shape_x, shape_mean)):
            if x_size != mean_size:
                reduce_axis_tmp.append(i)
        reduce_axis = tuple(reduce_axis_tmp)

        mean_num = 1.0
        for i in reduce_axis:
            mean_num *= shape_x[i]

        self.reduce_axis = reduce_axis
        self.mean_num = mean_num

    def _dichotomy_add_at_norm_axis(self, dst_ub, dichotomy_ub, src_ub):
        if self._is_nz():
            dichotomy_num = self.tiling_params.c1
            dichotomy_repeat = self.tiling_params.c1
            dichotomy_tail = self.tiling_params.c1 % 2
            dichotomy_factor = self.tiling_params.mid_factor
        else:
            dichotomy_num = self.tiling_params.mid_factor
            dichotomy_repeat = self.tiling_params.mid_factor
            dichotomy_tail = self.tiling_params.mid_factor % 2
            dichotomy_factor = 1

        if dichotomy_repeat > 1:
            dichotomy_repeat //= 2
            cnt = dichotomy_repeat * dichotomy_factor * self.tiling_params.c0
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", dichotomy_ub,
                                                src_ub[dichotomy_repeat * dichotomy_factor *
                                                       self.tiling_params.c0:],
                                                src_ub, cnt)
        while dichotomy_repeat > 1:
            # Judge whether dichotomy_repeat is power to 2 and is odd number
            if dichotomy_repeat & (dichotomy_repeat - 1) and dichotomy_repeat % 2:
                cnt = dichotomy_factor * self.tiling_params.c0
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", dichotomy_ub,
                                                    dichotomy_ub[(dichotomy_repeat - 1) * dichotomy_factor *
                                                                 self.tiling_params.c0:],
                                                    dichotomy_ub, cnt)
            dichotomy_repeat //= 2
            cnt = dichotomy_repeat * dichotomy_factor * self.tiling_params.c0
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", dichotomy_ub,
                                                dichotomy_ub[dichotomy_repeat * dichotomy_factor *
                                                             self.tiling_params.c0:],
                                                dichotomy_ub, cnt)
        is_need_dichotomy_add = True if dichotomy_num > 1 else False
        if dichotomy_tail > 0 and is_need_dichotomy_add:
            cnt = dichotomy_factor * self.tiling_params.c0
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", dichotomy_ub,
                                                src_ub[(dichotomy_num - 1) * dichotomy_factor *
                                                       self.tiling_params.c0:],
                                                dichotomy_ub, cnt)
        temp_res_ub = dichotomy_ub if is_need_dichotomy_add else src_ub
        if self._is_nz():
            self.tik_instance.vcadd(self.tiling_params.c0,
                                    dst_ub, temp_res_ub,
                                    self.tiling_params.mid_factor, 1, 1,
                                    self.tiling_params.c0 // Constant.BLOCK_NUM_FP32)
        else:
            util_tik_comm_func.tik_func_vadds(self.tik_instance, dst_ub, temp_res_ub, 0,
                                              self.tiling_params.c0)

    def _dichotomy_add_at_last_axis(self, dst_ub, dichotomy_ub, src_ub):
        mask_num = Constant.MASK_FP16 if dst_ub.dtype.lower() == "float16" else Constant.MASK_FP32
        block_num = Constant.BLOCK_NUM_FP16 if dst_ub.dtype.lower() == "float16" else Constant.BLOCK_NUM_FP32
        vcadd_repeat_num = self.tiling_params.c0 // mask_num
        vcadd_tail = self.tiling_params.c0 % mask_num
        vcadd_stride = mask_num // block_num
        final_num = vcadd_repeat_num

        with self.tik_instance.for_range(0, self.tiling_params.mid_factor) as i:
            src_offset = i * self.tiling_params.c0
            if vcadd_repeat_num > 0:
                self.tik_instance.vcadd(mask_num,
                                        dichotomy_ub, src_ub[src_offset:],
                                        vcadd_repeat_num, 1, 1, vcadd_stride)
            if vcadd_tail > 0:
                final_num += 1
                src_offset = i * self.tiling_params.c0 + vcadd_repeat_num * mask_num
                self.tik_instance.vcadd(vcadd_tail,
                                        dichotomy_ub[vcadd_repeat_num:], src_ub[src_offset:],
                                        1, 1, 1, vcadd_stride)
            self.tik_instance.vcadd(final_num,
                                    dst_ub[i], dichotomy_ub,
                                    1, 1, 1, vcadd_stride)

    def _brd_at_last_axis(self, dst_ub, src_ub):
        if self._is_support_mmad() and self.impl_mode == "high_performance":
            block_num = Constant.BLOCK_NUM_FP16 if src_ub.dtype.lower() == "float16" else Constant.BLOCK_NUM_FP32
            var_stride = self.tiling_params.c0_factor // block_num
            self.tik_instance.vadds(self.tiling_params.mid_factor, dst_ub, src_ub, 0,
                                    self.tiling_params.c0_factor, 1, 1, var_stride, 0)
            if src_ub.dtype.lower() != "float16":
                util_tik_comm_func.tik_func_vconv(self.tik_instance, self.temp_fp16_ub, dst_ub,
                                                  self.tiling_params.mid_factor * self.tiling_params.c0)
                self.tik_instance.vtranspose(self.temp_fp16_ub, self.temp_fp16_ub)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, dst_ub, self.temp_fp16_ub,
                                                  self.tiling_params.mid_factor * self.tiling_params.c0)
            else:
                self.tik_instance.vtranspose(dst_ub, dst_ub)
        else:
            scalar_array = self.tik_instance.ScalarArray(self.act_dtype, self.tiling_params.mid_factor,
                                                         "scalar_array")
            for i in range(self.tiling_params.mid_factor):
                scalar_array[i].set_as(src_ub[i])
            for i in range(self.tiling_params.mid_factor):
                util_tik_comm_func.tik_func_vector(self.tik_instance, dst_ub[i * self.tiling_params.c0_factor:],
                                                   scalar_array[i], self.tiling_params.c0_factor)

    def _vec_with_brd_at_norm_axis(self, vec_cmd, dst_ub, src0_ub, src1_ub):
        if self._is_nz():
            brd_factor = self.tiling_params.mid_factor * self.tiling_params.c0
            repeat_num = self.tiling_params.c1
        else:
            brd_factor = self.tiling_params.c0
            repeat_num = self.tiling_params.mid_factor
        mask_num = Constant.MASK_FP16 if src0_ub.dtype.lower() == "float16" else Constant.MASK_FP32
        block_num = Constant.BLOCK_NUM_FP16 if src0_ub.dtype.lower() == "float16" else Constant.BLOCK_NUM_FP32
        var_loop_num = brd_factor // mask_num
        var_tail = brd_factor % mask_num
        var_stride = brd_factor // block_num

        if vec_cmd == "vadd":
            tik_fun = self.tik_instance.vadd
        elif vec_cmd == "vmul":
            tik_fun = self.tik_instance.vmul
        elif vec_cmd == "vsub":
            tik_fun = self.tik_instance.vsub

        with self.tik_instance.for_range(0, var_loop_num) as i:
            tik_fun(mask_num,
                    dst_ub[i * mask_num:],
                    src0_ub[i * mask_num:],
                    src1_ub[i * mask_num:],
                    repeat_num, 1, 1, 1,
                    var_stride, var_stride, 0)
        if var_tail > 0:
            tik_fun(var_tail,
                    dst_ub[var_loop_num * mask_num:],
                    src0_ub[var_loop_num * mask_num:],
                    src1_ub[var_loop_num * mask_num:],
                    repeat_num, 1, 1, 1,
                    var_stride, var_stride, 0)

    def _vec_with_brd_at_last_axis(self, vec_cmd, dst_ub, src0_ub, src1_ub):
        block_num = Constant.BLOCK_NUM_FP16 if src0_ub.dtype.lower() == "float16" else Constant.BLOCK_NUM_FP32
        var_repeat_num = self.tiling_params.c0 // self.tiling_params.c0_factor
        var_tail = self.tiling_params.c0 % self.tiling_params.c0_factor
        var_stride = self.tiling_params.c0_factor // block_num

        if vec_cmd == "vadd":
            tik_fun = self.tik_instance.vadd
        elif vec_cmd == "vmul":
            tik_fun = self.tik_instance.vmul
        elif vec_cmd == "vsub":
            tik_fun = self.tik_instance.vsub

        with self.tik_instance.for_range(0, self.tiling_params.mid_factor) as i:
            if var_repeat_num > 0:
                tik_fun(self.tiling_params.c0_factor,
                        dst_ub[i * self.tiling_params.c0:],
                        src0_ub[i * self.tiling_params.c0:],
                        src1_ub[i * self.tiling_params.c0_factor:],
                        var_repeat_num, 1, 1, 1,
                        var_stride, var_stride, 0)
            if var_tail > 0:
                tik_fun(var_tail,
                        dst_ub[i * self.tiling_params.c0 + var_repeat_num * self.tiling_params.c0_factor:],
                        src0_ub[i * self.tiling_params.c0 + var_repeat_num * self.tiling_params.c0_factor:],
                        src1_ub[i * self.tiling_params.c0_factor:],
                        1, 1, 1, 1, 0, 0, 0)

    def _load_2d(self, src, dst, instr_params):
        """
        load_2d instr is different in different platforms
        """
        start_index, repeat, repeat_stride, sid, is_transpose = instr_params
        if tbe_platform.api_check_support("tik.load2dv2"):
            self.tik_instance.load2dv2(src, dst, start_index, repeat, 0, repeat_stride, sid, is_transpose)
        elif tbe_platform.api_check_support("tik.load2dv1"):
            self.tik_instance.load2dv1(src, dst, start_index, repeat, repeat_stride, sid, is_transpose)
        else:
            error_manager_vector.raise_err_specific_reson("ln_dropout_grad", "load2d instr is unsupported.")

    def _generate_assist_data(self):
        with self.tik_instance.new_stmt_scope():
            cnt = self.tiling_params.c1 * self.tiling_params.mid_factor * self.tiling_params.c0
            tensor_l1b = self._apply_tensor(Constant.FP_16, self.tiling_params.x_ub_dims, "tensor_l1b",
                                            tbe_platform.scope_cbuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, self.temp_fp16_ub, 1, cnt)
            self.tik_instance.data_move(tensor_l1b, self.temp_fp16_ub, 0, 1,
                                        cnt // Constant.BLOCK_NUM_FP16, 0, 0)
            self._load_2d(self.share_l0b, tensor_l1b, [0,
                                                       cnt // (Constant.BLOCK_NUM_FP16 * Constant.BLOCK_NUM_FP16),
                                                       1, 0, False])

    def _mad_compute(self, dst_ub, l0a_ub, is_pd_gamma):
        with self.tik_instance.new_stmt_scope():
            if is_pd_gamma:
                ele_m = self.tiling_params.c1 * self.tiling_params.c0
                ele_k = self.tiling_params.mid_factor
                ele_n = self.tiling_params.c0
                if_transpose = True
            else:
                ele_m = self.tiling_params.mid_factor
                ele_k = self.tiling_params.c1 * self.tiling_params.c0
                ele_n = self.tiling_params.c0
                if_transpose = False
            mad_dims = (ele_m * ele_k,)
            tensor_l1a = self._apply_tensor(Constant.FP_16, mad_dims, "tensor_l1a", tbe_platform.scope_cbuf)
            tensor_l0a = self._apply_tensor(Constant.FP_16, mad_dims, "tensor_l0a", tbe_platform.scope_ca)
            tensor_l0c = self._apply_tensor(Constant.FP_32, mad_dims, "tensor_l0c", tbe_platform.scope_cc)

            if l0a_ub.dtype != Constant.FP_16:
                util_tik_comm_func.tik_func_vconv(self.tik_instance, self.temp_fp16_ub, l0a_ub, ele_m * ele_k)
                self.tik_instance.data_move(tensor_l1a, self.temp_fp16_ub, 0, 1,
                                            (ele_m * ele_k) // Constant.BLOCK_NUM_FP16, 0, 0)
            else:
                self.tik_instance.data_move(tensor_l1a, l0a_ub, 0, 1,
                                            (ele_m * ele_k) // Constant.BLOCK_NUM_FP16, 0, 0)
            self._load_2d(tensor_l0a, tensor_l1a,
                          [0, (ele_m * ele_k) // (Constant.BLOCK_NUM_FP16 * Constant.BLOCK_NUM_FP16),
                           1, 0, if_transpose])
            self.tik_instance.mmad(tensor_l0c, tensor_l0a, self.share_l0b, ele_m, ele_k, ele_n, 0)

            output_dims = ele_m * ele_n
            if not is_pd_gamma:
                if dst_ub.dtype != Constant.FP_32:
                    temp_ub = self._apply_tensor(Constant.FP_32, (output_dims,), "temp_ub")
                    self.tik_instance.tensor_mov(temp_ub, tensor_l0c, 'm', 1,
                                                 output_dims // (Constant.BLOCK_NUM_FP16 * Constant.BLOCK_NUM_FP16),
                                                 0, 0)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance, temp_ub, temp_ub,
                                                      -1.0 / self.mean_num, output_dims)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, dst_ub, temp_ub, output_dims)
                else:
                    self.tik_instance.tensor_mov(dst_ub, tensor_l0c, 'm', 1,
                                                 output_dims // (Constant.BLOCK_NUM_FP16 * Constant.BLOCK_NUM_FP16),
                                                 0, 0)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance, dst_ub, dst_ub,
                                                      -1.0 / self.mean_num, output_dims)
            else:
                self.tik_instance.tensor_mov(dst_ub, tensor_l0c, 'm', 1,
                                             output_dims // (Constant.BLOCK_NUM_FP16 * Constant.BLOCK_NUM_FP16), 0, 0)

    def _transpose_compute(self, dst_ub, src_ub, temp_src_ub_fp16):
        vec_ub = src_ub
        if src_ub.dtype != Constant.FP_16:
            util_tik_comm_func.tik_func_vconv(self.tik_instance, temp_src_ub_fp16, src_ub,
                                              self.tiling_params.c1 * self.tiling_params.c0 * self.tiling_params.c0)
            vec_ub = temp_src_ub_fp16
        src_addr_list = [vec_ub[self.tiling_params.c0 * i] for i in range(self.tiling_params.mid_factor)]
        dst_addr_list = [vec_ub[self.tiling_params.c0 * i] for i in range(self.tiling_params.mid_factor)]
        self.tik_instance.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                    self.tiling_params.c1, 1, self.tiling_params.mid_factor)
        util_tik_comm_func.tik_func_vconv(self.tik_instance, dst_ub, vec_ub,
                                          self.tiling_params.c1 * self.tiling_params.c0)

    def _is_cast_to_fp32(self):
        return self.ori_dtype == Constant.FP_16 and self.impl_mode == "high_precision"

    def _is_nz(self):
        return self.act_format == "FRACTAL_NZ"

    def _is_support_mmad(self):
        return (self._is_nz() and self.tiling_params.mid_factor == Constant.BLOCK_NUM_FP16 and
                self.impl_mode != "high_precision")

    def _apply_tensor(self, dtype, dims, name, scope=tbe_platform.scope_ubuf):
        return self.tik_instance.Tensor(dtype, dims, name=name, scope=scope)

    def _calc_sub_x_mean(self, sub_x_mean_ub, var_elt_ub, broadcast_var_elt_ub, x_input_offset, mean_input_offset):
        with self.tik_instance.new_stmt_scope():
            entire_num = self.tiling_params.c1 * self.tiling_params.mid_factor * self.tiling_params.c0
            if self._is_nz():
                gm_stride = ((self.tiling_params.mid_dim - self.tiling_params.mid_factor) *
                             self.tiling_params.c0_block_num)
            else:
                gm_stride = 1
            if self._is_cast_to_fp32():
                util_tik_comm_func.gm2ub(self.tik_instance, self.temp_fp16_ub, self.input_gm_list[1][x_input_offset],
                                         self.tiling_params.mid_factor * self.tiling_params.c0, None,
                                         self.tiling_params.c1, gm_stride, 0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, sub_x_mean_ub, self.temp_fp16_ub, entire_num)
                util_tik_comm_func.gm2ub(self.tik_instance, self.temp_fp16_ub,
                                         self.input_gm_list[3][mean_input_offset],
                                         self.tiling_params.mid_factor)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, var_elt_ub, self.temp_fp16_ub,
                                                  self.tiling_params.mid_factor)
            else:
                util_tik_comm_func.gm2ub(self.tik_instance, sub_x_mean_ub, self.input_gm_list[1][x_input_offset],
                                         self.tiling_params.mid_factor * self.tiling_params.c0, None,
                                         self.tiling_params.c1, gm_stride, 0)
                util_tik_comm_func.gm2ub(self.tik_instance, var_elt_ub,
                                         self.input_gm_list[3][mean_input_offset],
                                         self.tiling_params.mid_factor)
            self._brd_at_last_axis(broadcast_var_elt_ub, var_elt_ub)
            if self._is_nz():
                self._vec_with_brd_at_norm_axis("vsub", sub_x_mean_ub, sub_x_mean_ub, broadcast_var_elt_ub)
            else:
                self._vec_with_brd_at_last_axis("vsub", sub_x_mean_ub, sub_x_mean_ub, broadcast_var_elt_ub)

    def _calc_var_etl_2(self, var_elt_ub, broadcast_var_elt_ub, mean_input_offset):
        with self.tik_instance.new_stmt_scope():
            if self._is_cast_to_fp32():
                util_tik_comm_func.gm2ub(self.tik_instance, self.temp_fp16_ub,
                                         self.input_gm_list[2][mean_input_offset],
                                         self.tiling_params.mid_factor)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, var_elt_ub, self.temp_fp16_ub,
                                                  self.tiling_params.mid_factor)
            else:
                util_tik_comm_func.gm2ub(self.tik_instance, var_elt_ub,
                                         self.input_gm_list[2][mean_input_offset],
                                         self.tiling_params.mid_factor)

            # 'func: var_elt_2 = broadcast(power((var + epsilon), -0.5))
            epsilon = 1e-5 if self.ori_dtype == Constant.FP_16 else 1e-12
            util_tik_comm_func.tik_func_vadds(self.tik_instance, var_elt_ub, var_elt_ub, epsilon,
                                              self.tiling_params.mid_factor)
            util_tik_comm_func.tik_func_vsingle(self.tik_instance, "vln", var_elt_ub, var_elt_ub,
                                                self.tiling_params.mid_factor)
            util_tik_comm_func.tik_func_vmuls(self.tik_instance, var_elt_ub, var_elt_ub, -0.5,
                                              self.tiling_params.mid_factor)
            util_tik_comm_func.tik_func_vsingle(self.tik_instance, "vexp", var_elt_ub, var_elt_ub,
                                                self.tiling_params.mid_factor)
            self._brd_at_last_axis(broadcast_var_elt_ub, var_elt_ub)

    def _calc_pd_gamma_beta(self, cast_dy_ub, sub_x_mean_ub, broadcast_var_elt_ub, x_input_offset):
        def _reduce_params_axis(dst_ub, src_ub):
            if self.tiling_params.mid_factor > 1:
                block_num = Constant.BLOCK_NUM_FP16 if src_ub.dtype.lower() == "float16" else Constant.BLOCK_NUM_FP32
                src_stride = self.tiling_params.mid_factor * self.tiling_params.c0_factor // block_num
                dst_stride = self.tiling_params.c0_factor // block_num
                self.tik_instance.vadds(self.tiling_params.c0_factor, dst_ub, src_ub, 0,
                                        self.tiling_params.c1, 1, 1, dst_stride, src_stride)
                with self.tik_instance.for_range(1, self.tiling_params.mid_factor) as i:
                    self.tik_instance.vadd(self.tiling_params.c0_factor,
                                           dst_ub,
                                           src_ub[i * self.tiling_params.c0_factor:],
                                           dst_ub,
                                           self.tiling_params.c1, 1, 1, 1, dst_stride, src_stride, dst_stride)
            else:
                util_tik_comm_func.tik_func_vadds(self.tik_instance, dst_ub, src_ub, 0,
                                                  self.tiling_params.c1 * self.tiling_params.c0)

        with self.tik_instance.new_stmt_scope():
            entire_num = self.tiling_params.c1 * self.tiling_params.mid_factor * self.tiling_params.c0
            if self._is_nz():
                gm_stride = ((self.tiling_params.mid_dim - self.tiling_params.mid_factor) *
                             self.tiling_params.c0_block_num)
            else:
                gm_stride = 1
            temp_res_ub = self._apply_tensor(self.act_dtype, self.tiling_params.x_ub_dims, "temp_res_ub")
            output_gamma_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.beta_ub_dims,
                                                 "output_gamma_ub")
            output_beta_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.beta_ub_dims,
                                                "output_beta_ub")
            if self._is_cast_to_fp32():
                util_tik_comm_func.gm2ub(self.tik_instance, self.temp_fp16_ub, self.input_gm_list[0][x_input_offset],
                                         self.tiling_params.mid_factor * self.tiling_params.c0, None,
                                         self.tiling_params.c1, gm_stride, 0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, cast_dy_ub, self.temp_fp16_ub, entire_num)
            else:
                util_tik_comm_func.gm2ub(self.tik_instance, cast_dy_ub, self.input_gm_list[0][x_input_offset],
                                         self.tiling_params.mid_factor * self.tiling_params.c0, None,
                                         self.tiling_params.c1, gm_stride, 0)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul", temp_res_ub,
                                                cast_dy_ub, sub_x_mean_ub, entire_num)

            if self._is_nz():
                self._vec_with_brd_at_norm_axis("vmul", temp_res_ub, temp_res_ub, broadcast_var_elt_ub)
                if self._is_support_mmad():
                    self._mad_compute(self.temp_fp16_ub, temp_res_ub, True)
                    self._transpose_compute(output_gamma_ub, self.temp_fp16_ub, self.temp_fp16_ub)
                    self._mad_compute(self.temp_fp16_ub, cast_dy_ub, True)
                    self._transpose_compute(output_beta_ub, self.temp_fp16_ub, self.temp_fp16_ub)
                else:
                    if temp_res_ub.dtype != Constant.FP_32:
                        temp_fp32_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.x_ub_dims, "temp_fp32_ub")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance, temp_fp32_ub, temp_res_ub,
                                                          entire_num)
                        _reduce_params_axis(output_gamma_ub, temp_fp32_ub)
                        util_tik_comm_func.tik_func_vconv(self.tik_instance, temp_fp32_ub, cast_dy_ub,
                                                          entire_num)
                        _reduce_params_axis(output_beta_ub, temp_fp32_ub)
                    else:
                        _reduce_params_axis(output_gamma_ub, temp_res_ub)
                        _reduce_params_axis(output_beta_ub, cast_dy_ub)
            else:
                self._vec_with_brd_at_last_axis("vmul", temp_res_ub, temp_res_ub, broadcast_var_elt_ub)
                if temp_res_ub.dtype != Constant.FP_32:
                    temp_fp32_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.x_ub_dims, "temp_fp32_ub")
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, temp_fp32_ub, temp_res_ub,
                                                      entire_num)
                    self._dichotomy_add_at_norm_axis(output_gamma_ub, temp_fp32_ub, temp_fp32_ub)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, temp_fp32_ub, cast_dy_ub,
                                                      entire_num)
                    self._dichotomy_add_at_norm_axis(output_beta_ub, temp_fp32_ub, temp_fp32_ub)
                else:
                    self._dichotomy_add_at_norm_axis(output_gamma_ub, temp_res_ub, temp_res_ub)
                    self._dichotomy_add_at_norm_axis(output_beta_ub, temp_res_ub, cast_dy_ub)

            self.tik_instance.set_atomic_add(1)
            util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[2], output_gamma_ub,
                                     self.tiling_params.c1 * self.tiling_params.c0)
            util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[3], output_beta_ub,
                                     self.tiling_params.c1 * self.tiling_params.c0)
            self.tik_instance.set_atomic_add(0)

    def _calc_pd_var(self, cast_dy_ub, var_elt_ub, broadcast_var_elt_ub, sub_x_mean_ub, pd_var_ub):
        with self.tik_instance.new_stmt_scope():
            entire_num = self.tiling_params.c1 * self.tiling_params.mid_factor * self.tiling_params.c0
            cast_gamma_ub = self._apply_tensor(self.act_dtype, self.tiling_params.beta_ub_dims, "cast_gamma_ub")
            brd_gamma_ub = self._apply_tensor(self.act_dtype, self.tiling_params.x_ub_dims,
                                              "brd_gamma_ub")
            var_ub = var_elt_ub
            temp_factor = 1

            if self._is_cast_to_fp32():
                util_tik_comm_func.gm2ub(self.tik_instance, self.temp_fp16_ub, self.input_gm_list[4],
                                         self.tiling_params.c1 * self.tiling_params.c0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, cast_gamma_ub, self.temp_fp16_ub,
                                                  self.tiling_params.c1 * self.tiling_params.c0)
            else:
                util_tik_comm_func.gm2ub(self.tik_instance, cast_gamma_ub, self.input_gm_list[4],
                                         self.tiling_params.c1 * self.tiling_params.c0)
            if self._is_nz():
                block_num = (Constant.BLOCK_NUM_FP16 if cast_dy_ub.dtype.lower() == "float16"
                             else Constant.BLOCK_NUM_FP32)
                rep_stride = self.tiling_params.mid_factor * self.tiling_params.c0_factor // block_num
                with self.tik_instance.for_range(0, self.tiling_params.mid_factor) as i:
                    self.tik_instance.vmul(self.tiling_params.c0_factor,
                                           cast_dy_ub[i * self.tiling_params.c0_factor],
                                           cast_dy_ub[i * self.tiling_params.c0_factor],
                                           cast_gamma_ub,
                                           self.tiling_params.c1, 1, 1, 1,
                                           rep_stride, rep_stride, self.tiling_params.c0_factor // block_num)

                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul", brd_gamma_ub,
                                                    cast_dy_ub, sub_x_mean_ub, entire_num)
                if self._is_support_mmad():
                    var_ub = broadcast_var_elt_ub
                    temp_factor = self.tiling_params.c0
                    self._mad_compute(pd_var_ub, brd_gamma_ub, False)
                else:
                    if pd_var_ub.dtype == Constant.FP_16:
                        temp_pd_var_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.var_ub_dims,
                                                            "temp_pd_var_ub")
                        dichotomy_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.x_ub_dims, "dichotomy_ub")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance, dichotomy_ub, brd_gamma_ub,
                                                          entire_num)
                        util_tik_comm_func.tik_func_vmuls(self.tik_instance, dichotomy_ub, dichotomy_ub,
                                                          -1.0 / self.mean_num,
                                                          entire_num)
                        self._dichotomy_add_at_norm_axis(temp_pd_var_ub, dichotomy_ub, dichotomy_ub)
                        util_tik_comm_func.tik_func_vconv(self.tik_instance, pd_var_ub, temp_pd_var_ub,
                                                          self.tiling_params.mid_factor * temp_factor)
                    else:
                        self._dichotomy_add_at_norm_axis(pd_var_ub, brd_gamma_ub, brd_gamma_ub)
                        util_tik_comm_func.tik_func_vmuls(self.tik_instance, pd_var_ub, pd_var_ub,
                                                          -1.0 / self.mean_num,
                                                          self.tiling_params.mid_factor * temp_factor)
            else:
                temp_reduce_ub = self._apply_tensor(Constant.FP_32, (Constant.MASK_FP32,),
                                                    "temp_reduce_ub")
                self._vec_with_brd_at_norm_axis("vmul", cast_dy_ub, cast_dy_ub, cast_gamma_ub)
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul", brd_gamma_ub,
                                                    cast_dy_ub, sub_x_mean_ub, entire_num)
                if pd_var_ub.dtype == Constant.FP_16:
                    temp_fp32_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.x_ub_dims, "temp_fp32_ub")
                    temp_pd_var_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.var_ub_dims,
                                                        "temp_pd_var_ub")
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, temp_fp32_ub, brd_gamma_ub,
                                                      entire_num)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance, temp_fp32_ub, temp_fp32_ub,
                                                      -1.0 / self.mean_num,
                                                      entire_num)
                    self._dichotomy_add_at_last_axis(temp_pd_var_ub, temp_reduce_ub, temp_fp32_ub)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, pd_var_ub, temp_pd_var_ub,
                                                      self.tiling_params.mid_factor * temp_factor)
                else:
                    self._dichotomy_add_at_last_axis(pd_var_ub, temp_reduce_ub, brd_gamma_ub)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance, pd_var_ub, pd_var_ub, -1.0 / self.mean_num,
                                                      self.tiling_params.mid_factor * temp_factor)

            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul", pd_var_ub,
                                                pd_var_ub, var_ub, self.tiling_params.mid_factor * temp_factor)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul", pd_var_ub,
                                                pd_var_ub, var_ub, self.tiling_params.mid_factor * temp_factor)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul", pd_var_ub,
                                                pd_var_ub, var_ub, self.tiling_params.mid_factor * temp_factor)

    def _calc_pd_mean(self, cast_dy_ub, var_elt_ub, broadcast_var_elt_ub, pd_mean_ub):
        with self.tik_instance.new_stmt_scope():
            temp_factor = 1
            var_ub = var_elt_ub
            if self._is_nz():
                if self._is_support_mmad():
                    temp_factor = self.tiling_params.c0
                    var_ub = broadcast_var_elt_ub
                    self._mad_compute(pd_mean_ub, cast_dy_ub, False)
                else:
                    dichotomy_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.x_ub_dims, "dichotomy_ub")
                    if pd_mean_ub.dtype == Constant.FP_16:
                        temp_pd_mean_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.var_ub_dims,
                                                             "temp_pd_mean_ub")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance, dichotomy_ub, cast_dy_ub,
                                                          self.tiling_params.c1 * self.tiling_params.mid_factor *
                                                          self.tiling_params.c0)
                        util_tik_comm_func.tik_func_vmuls(self.tik_instance, dichotomy_ub, dichotomy_ub,
                                                          -1.0 / self.mean_num,
                                                          self.tiling_params.c1 * self.tiling_params.mid_factor *
                                                          self.tiling_params.c0)
                        self._dichotomy_add_at_norm_axis(temp_pd_mean_ub, dichotomy_ub, dichotomy_ub)
                        util_tik_comm_func.tik_func_vconv(self.tik_instance, pd_mean_ub, temp_pd_mean_ub,
                                                          self.tiling_params.mid_factor * temp_factor)
                    else:
                        self._dichotomy_add_at_norm_axis(pd_mean_ub, dichotomy_ub, cast_dy_ub)
                        util_tik_comm_func.tik_func_vmuls(self.tik_instance, pd_mean_ub, pd_mean_ub,
                                                          -1.0 / self.mean_num,
                                                          self.tiling_params.mid_factor * temp_factor)
            else:
                temp_reduce_ub = self._apply_tensor(Constant.FP_32, (Constant.MASK_FP32,),
                                                    "temp_reduce_ub")
                if pd_mean_ub.dtype == Constant.FP_16:
                    temp_pd_mean_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.var_ub_dims,
                                                         "temp_pd_mean_ub")
                    temp_fp32_ub = self._apply_tensor(Constant.FP_32, self.tiling_params.x_ub_dims, "temp_fp32_ub")
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, temp_fp32_ub, cast_dy_ub,
                                                      self.tiling_params.c1 * self.tiling_params.mid_factor *
                                                      self.tiling_params.c0)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance, temp_fp32_ub, temp_fp32_ub,
                                                      -1.0 / self.mean_num,
                                                      self.tiling_params.c1 * self.tiling_params.mid_factor *
                                                      self.tiling_params.c0)
                    self._dichotomy_add_at_last_axis(temp_pd_mean_ub, temp_fp32_ub, temp_fp32_ub)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, pd_mean_ub, temp_pd_mean_ub,
                                                      self.tiling_params.mid_factor * temp_factor)
                else:
                    self._dichotomy_add_at_last_axis(pd_mean_ub, temp_reduce_ub, cast_dy_ub)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance, pd_mean_ub, pd_mean_ub, -1.0 / self.mean_num,
                                                      self.tiling_params.mid_factor * temp_factor)

            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul", pd_mean_ub,
                                                pd_mean_ub, var_ub, self.tiling_params.mid_factor * temp_factor)

    def _calc_pd_x(self, cast_dy_ub, broadcast_var_elt_ub, sub_x_mean_ub, pd_var_ub, pd_mean_ub):
        entire_num = self.tiling_params.c1 * self.tiling_params.mid_factor * self.tiling_params.c0
        with self.tik_instance.new_stmt_scope():
            if self._is_nz():
                self._vec_with_brd_at_norm_axis("vmul", cast_dy_ub, cast_dy_ub, broadcast_var_elt_ub)
                if self._is_support_mmad():
                    self._vec_with_brd_at_norm_axis("vmul", sub_x_mean_ub, sub_x_mean_ub, pd_var_ub)
                else:
                    self._brd_at_last_axis(broadcast_var_elt_ub, pd_var_ub)
                    self._vec_with_brd_at_norm_axis("vmul", sub_x_mean_ub, sub_x_mean_ub, broadcast_var_elt_ub)
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", cast_dy_ub,
                                                    cast_dy_ub, sub_x_mean_ub, entire_num)
                if self._is_support_mmad():
                    self._vec_with_brd_at_norm_axis("vadd", cast_dy_ub, cast_dy_ub, pd_mean_ub)
                else:
                    self._brd_at_last_axis(broadcast_var_elt_ub, pd_mean_ub)
                    self._vec_with_brd_at_norm_axis("vadd", cast_dy_ub, cast_dy_ub, broadcast_var_elt_ub)
            else:
                self._vec_with_brd_at_last_axis("vmul", cast_dy_ub, cast_dy_ub, broadcast_var_elt_ub)
                self._brd_at_last_axis(broadcast_var_elt_ub, pd_var_ub)
                self._vec_with_brd_at_last_axis("vmul", sub_x_mean_ub, sub_x_mean_ub, broadcast_var_elt_ub)
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", cast_dy_ub,
                                                    cast_dy_ub, sub_x_mean_ub, entire_num)
                self._brd_at_last_axis(broadcast_var_elt_ub, pd_mean_ub)
                self._vec_with_brd_at_last_axis("vadd", cast_dy_ub, cast_dy_ub, broadcast_var_elt_ub)

    def _calc_drop_out(self, cast_dy_ub, x_input_offset, mask_input_offset):
        with self.tik_instance.new_stmt_scope():
            mask_ub = self._apply_tensor(self._mask.get("dtype"), self.tiling_params.x_ub_dims, "mask_ub")
            cast_mask_ub = self._apply_tensor(self.ori_dtype, self.tiling_params.x_ub_dims, "cast_mask_ub")
            entire_num = self.tiling_params.c1 * self.tiling_params.mid_factor * self.tiling_params.c0
            if self._is_nz():
                gm_stride = ((self.tiling_params.mid_dim - self.tiling_params.mid_factor) *
                             self.tiling_params.c0_block_num)
            else:
                gm_stride = 1

            even_base = 2
            if self._is_nz() and self.tiling_params.mid_factor % even_base == 0:
                mask_stride = (self.tiling_params.mid_dim - self.tiling_params.mid_factor) // even_base
                util_tik_comm_func.gm2ub(self.tik_instance, mask_ub, self.input_gm_list[5][x_input_offset],
                                         self.tiling_params.mid_factor * self.tiling_params.c0, None,
                                         self.tiling_params.c1, mask_stride, 0)
            else:
                util_tik_comm_func.gm2ub(self.tik_instance, mask_ub, self.input_gm_list[5][mask_input_offset],
                                         entire_num)

            if self.ori_dtype == Constant.FP_16:
                util_tik_comm_func.tik_func_vconv(self.tik_instance, cast_mask_ub, mask_ub,
                                                  entire_num)
            else:
                util_tik_comm_func.tik_func_vconv(self.tik_instance, self.temp_fp16_ub, mask_ub,
                                                  entire_num)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, cast_mask_ub, self.temp_fp16_ub,
                                                  entire_num)
            if self._is_cast_to_fp32():
                util_tik_comm_func.tik_func_vconv(self.tik_instance, self.temp_fp16_ub, cast_dy_ub,
                                                  entire_num)
                util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[0][x_input_offset], self.temp_fp16_ub,
                                         self.tiling_params.mid_factor * self.tiling_params.c0, None,
                                         self.tiling_params.c1, 0, gm_stride)
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul", cast_mask_ub,
                                                    cast_mask_ub, self.temp_fp16_ub, entire_num)
            else:
                util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[0][x_input_offset], cast_dy_ub,
                                         self.tiling_params.mid_factor * self.tiling_params.c0, None,
                                         self.tiling_params.c1, 0, gm_stride)
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul", cast_mask_ub,
                                                    cast_mask_ub, cast_dy_ub, entire_num)
            util_tik_comm_func.tik_func_vmuls(self.tik_instance, cast_mask_ub, cast_mask_ub,
                                              1.0 / self.keep_prop, entire_num)
            util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[1][x_input_offset], cast_mask_ub,
                                     self.tiling_params.mid_factor * self.tiling_params.c0, None,
                                     self.tiling_params.c1, 0, gm_stride)

    def _main_compute_func(self, x_input_offset, mean_input_offset, mask_input_offset):
        with self.tik_instance.new_stmt_scope():
            cast_dy_ub = self._apply_tensor(self.act_dtype, self.tiling_params.x_ub_dims, "cast_dy_ub")
            with self.tik_instance.new_stmt_scope():
                sub_x_mean_ub = self._apply_tensor(self.act_dtype, self.tiling_params.x_ub_dims, "sub_x_mean_ub")
                var_elt_ub = self._apply_tensor(self.act_dtype, self.tiling_params.var_ub_dims, "var_elt_ub")
                broadcast_var_elt_ub = self._apply_tensor(self.act_dtype, self.tiling_params.brd_var_ub_dims,
                                                          "broadcast_var_elt_ub")
                self._calc_sub_x_mean(sub_x_mean_ub, var_elt_ub, broadcast_var_elt_ub,
                                      x_input_offset, mean_input_offset)
                self._calc_var_etl_2(var_elt_ub, broadcast_var_elt_ub, mean_input_offset)
                self._calc_pd_gamma_beta(cast_dy_ub, sub_x_mean_ub, broadcast_var_elt_ub, x_input_offset)

                if self._is_support_mmad():
                    pd_var_ub = self._apply_tensor(self.act_dtype, self.tiling_params.brd_var_ub_dims, "pd_var_ub")
                    pd_mean_ub = self._apply_tensor(self.act_dtype, self.tiling_params.brd_var_ub_dims, "pd_mean_ub")
                else:
                    pd_var_ub = self._apply_tensor(self.act_dtype, self.tiling_params.var_ub_dims, "pd_var_ub")
                    pd_mean_ub = self._apply_tensor(self.act_dtype, self.tiling_params.var_ub_dims, "pd_mean_ub")
                self._calc_pd_var(cast_dy_ub, var_elt_ub, broadcast_var_elt_ub, sub_x_mean_ub, pd_var_ub)
                self._calc_pd_mean(cast_dy_ub, var_elt_ub, broadcast_var_elt_ub, pd_mean_ub)
                self._calc_pd_x(cast_dy_ub, broadcast_var_elt_ub, sub_x_mean_ub, pd_var_ub, pd_mean_ub)

            self._calc_drop_out(cast_dy_ub, x_input_offset, mask_input_offset)

    def _loop_func(self, loop_num, tail_num, core_offset):
        with self.tik_instance.new_stmt_scope():
            self.tiling_params.refine_mid_factor()
            self.temp_fp16_ub = self._apply_tensor(Constant.FP_16, self.tiling_params.x_ub_dims, "temp_fp16_ub")
            if self._is_support_mmad():
                self.share_l0b = self._apply_tensor(Constant.FP_16, self.tiling_params.x_ub_dims, "tensor_l0b",
                                                    tbe_platform.scope_cb)
                self._generate_assist_data()
            with self.tik_instance.for_range(0, self.tiling_params.batch_dim) as batch_idx:
                batch_offset = batch_idx * self.tiling_params.c1 * self.tiling_params.mid_dim * self.tiling_params.c0
                thread_num = self.tiling_params.thread_num if loop_num > 1 else 1
                with self.tik_instance.for_range(0, loop_num, thread_num=thread_num) as loop_idx:
                    x_input_offset = batch_offset + (
                            core_offset + loop_idx * self.tiling_params.mid_factor) * self.tiling_params.c0
                    mean_input_offset = (batch_idx * self.tiling_params.mid_dim + core_offset +
                                         loop_idx * self.tiling_params.mid_factor)
                    mask_input_offset = (batch_offset + (core_offset + loop_idx * self.tiling_params.mid_factor) *
                                         self.tiling_params.c1 * self.tiling_params.c0)
                    self._main_compute_func(x_input_offset, mean_input_offset, mask_input_offset)

                if tail_num > 0:
                    x_input_offset = batch_offset + (
                            core_offset + loop_num * self.tiling_params.mid_factor) * self.tiling_params.c0
                    mean_input_offset = (batch_idx * self.tiling_params.mid_dim + core_offset +
                                         loop_num * self.tiling_params.mid_factor)
                    mask_input_offset = (batch_offset + (core_offset + loop_num * self.tiling_params.mid_factor) *
                                         self.tiling_params.c1 * self.tiling_params.c0)
                    self.tiling_params.mid_factor = tail_num
                    self._main_compute_func(x_input_offset, mean_input_offset, mask_input_offset)


def _update_shape_nz(shape_x, shape_var):
    """
    function of updating Nz shape

    """
    # Nz shape of x >= four dim
    len_x = len(shape_x)
    nz_begin = len_x - 4
    shape_x_refine = []
    temp = 1
    for i in range(0, nz_begin):
        temp *= shape_x[i]
    shape_x_refine.append(temp)
    shape_x_refine.append(shape_x[nz_begin])
    shape_x_refine.append(shape_x[nz_begin + 1] * shape_x[nz_begin + 2])
    shape_x_refine.append(shape_x[nz_begin + 3])

    # ND shape of var >= two dim
    shape_var_refine = []
    len_var = len(shape_var)
    var_nz_begin = len_var - 2
    shape_var_refine.append(temp)
    shape_var_refine.append(1)
    shape_var_refine.append(shape_var[var_nz_begin])
    shape_var_refine.append(1)

    # ND shape of gamma is one dim
    shape_gamma_refine = [1, shape_x[nz_begin], 1, shape_x[nz_begin + 2]]

    return tuple(shape_x_refine), tuple(shape_var_refine), tuple(shape_gamma_refine)


def _update_shape_nd(shape_x, shape_gamma):
    """
    update shape for subsequent calculation
    """
    len_x = len(shape_x)
    shape_x_refine = [1, 1]
    shape_var_refine = [1, 1]
    temp = 1
    for i in range(0, len_x - 1):
        temp *= shape_x[i]
    shape_x_refine.append(temp)
    shape_x_refine.append(shape_x[-1])

    shape_var_refine.append(temp)
    shape_var_refine.append(1)

    if len(shape_x_refine) != len(shape_gamma):
        sub = len(shape_x_refine) - len(shape_gamma)
        shape_gamma = list(shape_gamma)
        for i in range(sub):
            shape_gamma.insert(0, 1)

    return tuple(shape_x_refine), tuple(shape_var_refine), tuple(shape_gamma)


def _is_aligned(shape_dy):
    c_0 = 16
    if len(shape_dy) < 2:
        return False

    return shape_dy[-1] % c_0 == 0 and shape_dy[-2] % c_0 == 0


def op_select_format(input_dy, input_x, input_variance, input_mean,
                     input_gamma, input_mask, output_pd_x, output_pd_x_dropout, output_pd_gamma,
                     output_pd_beta, keep_prob, kernel_name="ln_dropout_grad"):
    """
    function of selecting dynamic format
    """
    shape_dy = input_dy.get("ori_shape")
    shape_dy = shape_util.scalar2tensor_one(shape_dy)

    supported_datatype = "float16,float16,float16,float32,float32,float32,float16,float32"
    supported_datatype_mask = "uint8,uint8,uint8,uint8,uint8,uint8,bool,bool"
    supported_datatype_beta = "float,float,float,float,float,float,float,float"
    supported_format = "NCHW,NHWC,ND,NCHW,NHWC,ND,NHWC,NHWC"
    supported_format_with_reduce = "NCHW,NHWC,ND,NCHW,NHWC,ND,NHWC,NHWC"

    if _is_aligned(shape_dy):
        supported_datatype += ",float16,float32"
        supported_datatype_mask += ",uint8,uint8"
        supported_datatype_beta += ",float,float"
        supported_format += ",FRACTAL_NZ,FRACTAL_NZ"
        supported_format_with_reduce += ",ND,ND"

    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="dy",
                                           datatype=supported_datatype,
                                           format=supported_format)
    input1 = util_select_op_base.gen_param(classify="input1",
                                           name="x",
                                           datatype=supported_datatype,
                                           format=supported_format)
    input2 = util_select_op_base.gen_param(classify="input2",
                                           name="variance",
                                           datatype=supported_datatype,
                                           format=supported_format_with_reduce)
    input3 = util_select_op_base.gen_param(classify="input3",
                                           name="mean",
                                           datatype=supported_datatype,
                                           format=supported_format_with_reduce)
    input4 = util_select_op_base.gen_param(classify="input4",
                                           name="gamma",
                                           datatype=supported_datatype,
                                           format=supported_format_with_reduce)
    input5 = util_select_op_base.gen_param(classify="input5",
                                           name="mask",
                                           datatype=supported_datatype_mask,
                                           format=supported_format_with_reduce)
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="pd_x",
                                            datatype=supported_datatype,
                                            format=supported_format)
    output1 = util_select_op_base.gen_param(classify="output1",
                                            name="pd_x_dropout",
                                            datatype=supported_datatype,
                                            format=supported_format)
    output2 = util_select_op_base.gen_param(classify="output2",
                                            name="pd_gamma",
                                            datatype=supported_datatype_beta,
                                            format=supported_format_with_reduce)
    output3 = util_select_op_base.gen_param(classify="output3",
                                            name="pd_beta",
                                            datatype=supported_datatype_beta,
                                            format=supported_format_with_reduce)

    param_list = [input0, input1, input2, input3, input4, input5, output0, output1, output2, output3]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def ln_dropout_grad(input_dy, input_x, input_variance, input_mean,
                    input_gamma, input_mask, output_pd_x, output_pd_x_dropout, output_pd_gamma,
                    output_pd_beta, keep_prob, kernel_name="ln_dropout_grad", impl_mode="high_precision"):
    """
    algorithm: ln_dropout_grad
    calculating: gradient of ln_dropout
                 compute partial derivation of x, gamma and beta
    pd_xl = input_dy * input_gamma
    sub_x_mean = input_x - input_mean
    var_elta_2 = np.power((variance + EPSLON), (-0.5))

    pd_var = sum(pd_xl * sub_x_mean, reduce_axis, keepdims=True) * var_elta_2 * var_elta_2 * var_elta_2 * (-0.5)
    pd_mean = sum(pd_xl, reduce_axis, keepdims=True) * var_elta_2 * (-1.0)
    pd_x = pd_xl * var_elta_2 + pd_var * (2.0 / m) * sub_x_mean + pd_mean * (1.0 / m)
    pd_x_dropout = pd_x * mask * (1 / keep_prob)
    pd_gamma = sum(input_dy * sub_x_mean * var_elta_2, param_axis, keepdims=True)
    pd_beta = sum(input_dy, param_axis, keepdims=True)

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    input_mask: dict
        shape and dtype of input gamma, only support float16, float32
    output_pd_x: dict
        shape and dtype of output, only support float16, float32
    output_pd_x_dropout: dict
        shape and dtype of output, only support float16, float32
    output_pd_gamma: dict
        shape and dtype of output, only support float16, float32
    output_pd_beta: dict
        shape and dtype of output, only support float16, float32
    keep_prob: float
        prob scale (0.0,1.0] NOTICE: type is same as the dtype of input_x
    kernel_name: str
        cce kernel name, default value is "ln_dropout_grad"
    impl_mode: str
        support modes: high_performance and high_precision

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    format_x = input_x.get("format")
    if format_x.upper() == "FRACTAL_NZ":
        shape_x_refine, shape_var_refine, shape_gamma_refine = _update_shape_nz(input_x.get("shape"),
                                                                                input_variance.get("shape"))
    else:
        shape_x_refine, shape_var_refine, shape_gamma_refine = _update_shape_nd(input_x.get("shape"),
                                                                                input_gamma.get("shape"))
    input_dy.update({"shape": shape_x_refine})
    input_x.update({"shape": shape_x_refine})
    input_variance.update({"shape": shape_var_refine})
    input_mean.update({"shape": shape_var_refine})
    input_gamma.update({"shape": shape_gamma_refine})
    input_mask.update({"shape": shape_x_refine})
    output_pd_x.update({"shape": shape_x_refine})
    output_pd_x_dropout.update({"shape": shape_x_refine})

    obj = LayerNormGradV2(keep_prob, kernel_name)
    input_dict_list = [input_dy, input_x, input_variance, input_mean, input_gamma, input_mask]
    output_dict_list = [output_pd_x, output_pd_x_dropout, output_pd_gamma, output_pd_beta]
    obj.prepare(input_dict_list, output_dict_list, format_x.upper(), impl_mode)

    obj.ln_dropout_grad_compute()
