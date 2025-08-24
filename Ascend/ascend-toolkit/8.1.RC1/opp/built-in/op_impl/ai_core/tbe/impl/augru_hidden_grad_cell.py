#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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

# 'pylint: disable=locally-disabled,import-error,unused-import,ungrouped-imports
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.tik_op_base import TikOpBase


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    OP_NAME = "AUGRUHiddenGradCell"


# 'pylint: disable=too-many-arguments,invalid-name
def _check_dtype(dh_pre_t, h, dy, dh, update, reset, new, hidden_new):
    """
    check parameters type
    """
    para_check.check_dtype(dh_pre_t["dtype"], ["float16", "float32"], "dh_pre_t")
    para_check.check_dtype(h["dtype"], ["float16", "float32"], "h")
    para_check.check_dtype(dy["dtype"], ["float16", "float32"], "dy")
    para_check.check_dtype(dh["dtype"], ["float16", "float32"], "dh")
    para_check.check_dtype(update["dtype"], ["float16", "float32"], "update")
    para_check.check_dtype(reset["dtype"], ["float16", "float32"], "reset")
    para_check.check_dtype(new["dtype"], ["float16", "float32"], "new")
    para_check.check_dtype(hidden_new["dtype"], ["float16", "float32"], "hidden_new")


# 'pylint: disable=too-many-arguments,invalid-name
def _check_param(dh_pre_t, h, dy, dh, update, reset, new, hidden_new):
    """
    check parameters
    """
    para_check.check_shape_size(dh_pre_t["shape"])
    para_check.check_shape_size(h["shape"])
    para_check.check_shape_size(dy["shape"])
    para_check.check_shape_size(dh["shape"])
    para_check.check_shape_size(update["shape"])
    para_check.check_shape_size(reset["shape"])
    para_check.check_shape_size(new["shape"])
    para_check.check_shape_size(hidden_new["shape"])


def _check_attr(gate_order):
    """
    check attr
    """
    if gate_order not in ['zrh', 'rzh']:
        rule_desc = "gate_order should be zrh or rzh, but current attr is " + gate_order
        error_manager_vector.raise_err_check_params_rules(Constant.OP_NAME, rule_desc, 'gate_order', gate_order)


# 'pylint: disable=too-many-instance-attributes
class AUGRUHiddenGradCell(TikOpBase):
    """ AUGRUHiddenGradCell
    """
    # 'pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda
    # 'pylint: disable=too-many-locals,invalid-name,too-many-arguments
    def __init__(self, tik_instance, h, dy, dnt_x, seq_mask, t_state, gate_order, kernel_name):
        """ init AUGRUHiddenGradCell
        """
        super(AUGRUHiddenGradCell, self).__init__(tik_instance)
        self.t_state = t_state
        self.gate_order = gate_order
        self.kernel_name = kernel_name
        self.device_aicore_num = self.tik_instance.d_profiling.get_aicore_num()
        self.ub_byte_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        # `[1, output_dim, batch, 16, 16]`
        shape = dnt_x["shape"]
        self.t_size = dy["shape"][0]
        self.b_size = dy["shape"][2]
        self.fuse_size = self.get_shape_size(shape)
        fuse_shape = (self.fuse_size,)
        t_fuse_shape = (self.t_size, self.fuse_size)
        self.dtype = h["dtype"]
        self.input_data_size = self.get_data_size(self.dtype)
        # t offset for dy, update, reset, new, hidden_new
        self.t_offset = self.t_size - self.t_state - 1
        # ht offset for h
        self.ht_offset = 0 if self.t_size == 1 else self.t_size - self.t_state - 2
        self.has_seq_mask = True if seq_mask else False

        # input
        self.weight_att = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "weight_att")
        self.dh_pre_t = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "dh_pre_t")
        if self.t_state == self.t_size - 1:
            self.h = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "h")
        else:
            self.h = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "h")
        self.dy = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "dy")
        self.dh = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "dh")
        self.i2 = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "update")
        self.i2_att = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "update_att")
        self.r2 = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "reset")
        self.n2 = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "n2")
        self.n2_mid = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "hidden_new")
        if self.has_seq_mask:
            self.seq_mask_t = self.tik_instance.Tensor(self.dtype, t_fuse_shape, tbe_platform.scope_gm, "seq_mask")


        # output
        self.dh_prev = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "dh_prev")
        d_gate_shape = (3 * self.fuse_size,)
        self.d_gate_h = self.tik_instance.Tensor(self.dtype, d_gate_shape, tbe_platform.scope_gm, "d_gate_h")
        self.dnt_x = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "dnt_x")
        self.dw_att_t = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "dw_att_t")

    def build(self):
        """
        build cce
        """
        config_map = {"dump_cce_code": False}
        if self.has_seq_mask:
            input_list = (self.weight_att, self.dh_pre_t, self.h, self.dy, self.dh, self.i2, self.i2_att,
                          self.r2, self.n2, self.n2_mid, self.seq_mask_t)
        else:
            input_list = (self.weight_att, self.dh_pre_t, self.h, self.dy, self.dh, self.i2, self.i2_att,
                          self.r2, self.n2, self.n2_mid)
        output_list = (self.dh_prev, self.d_gate_h, self.dnt_x, self.dw_att_t)
        self.tik_instance.BuildCCE(self.kernel_name, input_list, output_list, config=config_map)

    def compute(self):
        """ do compute
        """
        tiling = self._get_tiling()
        core_num = tiling["core_num"]
        tail_core_num = tiling["tail_core_num"]
        ele_num = tiling["loop_ele"]
        tail_ele_num = tiling["tail_loop_ele"]
        offset = tiling["block_size"] * core_num
        max_core_num = max(core_num, tail_core_num)
        with self.tik_instance.for_range(0, max_core_num, block_num=max_core_num) as block_idx:
            if tiling["loop_num"] > 0:
                with self.tik_instance.if_scope(block_idx < core_num):
                    with self.tik_instance.for_range(0, tiling["loop_num"]) as loop_idx:
                        base_offset = tiling["block_size"] * block_idx + ele_num * loop_idx
                        self._do_compute(base_offset, ele_num)
            if tiling["tail_num"] > 0:
                with self.tik_instance.if_scope(block_idx < tail_core_num):
                    base_offset = offset + block_idx * tail_ele_num
                    with self.tik_instance.if_scope(block_idx < tail_core_num - 1):
                        self._do_compute(base_offset, tail_ele_num)
                    with self.tik_instance.else_scope():
                        self._do_compute(base_offset, tiling["tail_last_ele"])

    # 'pylint: disable=too-many-locals,too-many-statements,invalid-name
    def _do_compute(self, input_offset, ele_num):
        """
        do compute
        """
        shape = (ele_num, )
        dh = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "dh")
        self.move_data(dh, self.dh[input_offset], self.dtype, shape)
        if self.t_state > 0:
            dh_pre_t = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "dh_pre_t")
            self.move_data(dh_pre_t, self.dh_pre_t[input_offset], self.dtype, shape)
            self.vadd_func(dh, dh, dh_pre_t, shape)
        if self.has_seq_mask:
            seq_mask_ub = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "seq_mask")
            self.move_data(seq_mask_ub, self.seq_mask_t[input_offset], self.dtype, shape)
            self.vmul_func(dh, dh, seq_mask_ub, shape)

        if self.t_state == self.t_size:
            # just cal dh + dh_pre_t in last cell
            self.move_data(self.dh_prev[input_offset], dh, self.dtype, shape)
            return

        dy = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "dy")
        self.move_data(dy, self.dy[self.t_offset, input_offset], self.dtype, shape)
        dh_add_dy = dh
        self.vadd_func(dh_add_dy, dh, dy, shape)    # free dy
        i2_att = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "i2_att")
        self.move_data(i2_att, self.i2_att[self.t_offset, input_offset], self.dtype, shape)

        # cal dh_pre_t for next cell, output to dh_prev
        dh_pre_t = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "dh_pre_t")
        self.vmul_func(dh_pre_t, dh_add_dy, i2_att, shape)
        self.move_data(self.dh_prev[input_offset], dh_pre_t, self.dtype, shape)

        # cal concat
        one = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "one")
        self.vector_dup_func(one, 1, shape)
        n2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "n2")
        self.move_data(n2, self.n2[self.t_offset, input_offset], self.dtype, shape)
        power_n2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "power_n2")
        self.vmul_func(power_n2, n2, n2, shape)
        one_sub_power_n2 = power_n2
        self.vsub_func(one_sub_power_n2, one, power_n2, shape)
        one_sub_i2_att = i2_att
        self.vsub_func(one_sub_i2_att, one, i2_att, shape)
        n2_mul_i2_att = one_sub_power_n2
        self.vmul_func(n2_mul_i2_att, one_sub_power_n2, one_sub_i2_att, shape)
        dn2i = n2_mul_i2_att
        self.vmul_func(dn2i, n2_mul_i2_att, dh_add_dy, shape)
        # dn2i -> out
        self.move_data(self.dnt_x[input_offset], dn2i, self.dtype, shape)

        # cal di2
        h1 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "h1")
        if self.t_state == self.t_size - 1:
            self.move_data(h1, self.h[input_offset], self.dtype, shape)
        else:
            self.move_data(h1, self.h[self.ht_offset, input_offset], self.dtype, shape)
        h1_sub_n2 = h1
        self.vsub_func(h1_sub_n2, h1, n2, shape)

        dh2_mul_h1_sub_n2 = h1_sub_n2
        self.vmul_func(dh2_mul_h1_sub_n2, h1_sub_n2, dh_add_dy, shape)

        # `(1-i2)*i2`
        i2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "i2")
        self.move_data(i2, self.i2[self.t_offset, input_offset], self.dtype, shape)
        one_sub_i2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "one_sub_i2")
        self.vsub_func(one_sub_i2, one, i2, shape)
        one_i2_mul_i2 = one_sub_i2
        self.vmul_func(one_i2_mul_i2, one_sub_i2, i2, shape)  # free i2
        # `(h1-n2)*(dh2+dy)*(1-i2)*i2`
        di2_mid = one_i2_mul_i2
        self.vmul_func(di2_mid, dh2_mul_h1_sub_n2, one_i2_mul_i2, shape)
        # `(h1-n2)*(dh2+dy)*(1-i2)*i2*(1-a2)`
        a2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "a2")
        self.move_data(a2, self.weight_att[self.t_offset, input_offset], self.dtype, shape)
        one_sub_a2 = a2
        self.vsub_func(one_sub_a2, one, a2, shape)
        di2 = di2_mid
        self.vmul_func(di2, di2_mid, one_sub_a2, shape)

        # di2 -> out
        if self.gate_order == "zrh":
            offset = input_offset
        else:
            offset = self.fuse_size + input_offset
        self.move_data(self.d_gate_h[offset], di2, self.dtype, shape)

        r2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "r2")
        self.move_data(r2, self.r2[self.t_offset, input_offset], self.dtype, shape)
        dn2h = dn2i
        self.vmul_func(dn2h, dn2i, r2, shape)
        # dn2h -> out
        self.move_data(self.d_gate_h[self.fuse_size * 2 + input_offset], dn2h, self.dtype, shape)

        one_sub_r2 = r2
        self.vsub_func(one_sub_r2, one, r2, shape)
        n2_mid = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "n2_mid")
        self.move_data(n2_mid, self.n2_mid[self.t_offset, input_offset], self.dtype, shape)
        mid_mul_r2 = one_sub_r2
        self.vmul_func(mid_mul_r2, one_sub_r2, n2_mid, shape)
        dr2 = mid_mul_r2
        self.vmul_func(dr2, mid_mul_r2, dn2h, shape)
        # dr2 -> out
        if self.gate_order == "zrh":
            offset = self.fuse_size + input_offset
        else:
            offset = input_offset
        self.move_data(self.d_gate_h[offset], dr2, self.dtype, shape)

        zero = one
        self.vector_dup_func(zero, 0, shape)
        zero_sub_i2 = zero
        self.vsub_func(zero_sub_i2, zero, i2, shape)
        dw_att_t = zero_sub_i2
        self.vmul_func(dw_att_t, dh2_mul_h1_sub_n2, zero_sub_i2, shape)
        self.move_data(self.dw_att_t[input_offset], dw_att_t, self.dtype, shape)

    def _get_tiling(self):
        """ get tiling
        """
        ub_max_ele_num = self.ub_byte_size // self.input_data_size
        # align 128 or 64
        align = 256 // self.input_data_size
        if self.fuse_size < align * self.device_aicore_num:
            # fuse align 256
            core_num = self.fuse_size // align
            return {
                "core_num": core_num,
                "loop_num": 1,
                "loop_ele": align,
                "block_size": align,
                "tail_num": 0,
                "tail_core_num": 0,
                "tail_loop_ele": 0,
                "tail_last_ele": 0
            }

        core_num = self.device_aicore_num
        max_block_ele_num = (ub_max_ele_num // 8 // 2 // align) * align
        # fuse align 256
        loop_num = self.fuse_size // (max_block_ele_num * core_num)
        loop_ele = max_block_ele_num
        block_size = loop_ele * loop_num

        tail_num = self.fuse_size - block_size * core_num
        if tail_num == 0:
            return {
                "core_num": core_num,
                "loop_num": loop_num,
                "loop_ele": loop_ele,
                "block_size": block_size,
                "tail_num": tail_num,
                "tail_core_num": 0,
                "tail_loop_ele": 0,
                "tail_last_ele": 0
            }
        tail_loop_ele = (tail_num // core_num + align - 1) // align * align
        tail_core_num = (tail_num + tail_loop_ele - 1) // tail_loop_ele
        tail_last_ele = tail_num % tail_loop_ele if tail_num % tail_loop_ele != 0 else tail_loop_ele
        return {
            "core_num": core_num,
            "loop_num": loop_num,
            "loop_ele": loop_ele,
            "block_size": block_size,
            "tail_num": tail_num,
            "tail_core_num": tail_core_num,
            "tail_loop_ele": tail_loop_ele,
            "tail_last_ele": tail_last_ele
        }


# 'pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda
# 'pylint: disable=too-many-locals,invalid-name,too-many-arguments,unused-argument,huawei-too-many-arguments
def augru_hidden_grad_cell(weight_att, dh_pre_t, h, dy, dh, update, update_att, reset, new, hidden_new, seq_length,
                            dh_prev, dgate_h, dnt_x, dw_att_t, t_state=0, gate_order="zrh",
                            kernel_name="augru_hidden_grad_cell"):
    """
    Calculate the gradient
    Parameters
    -----------
    :param weight_att:[t, n]
    :param dh_pre_t: result of (dh2 + dy) * i2 at (cur_t -1)
        when t_state > 0, dh = dh + dh_pre_t
    :param h: [t, n, out]; set init_h(t_state = t_size - 1) [n, out]
    :param dy: [t, n, out]
    :param dh: [n, out]
    :param update: [t, n, out]
    :param update_att: [t, n, out]
    :param reset: [t, n, out]
    :param new: [t, n, out]
    :param hidden_new: [t, n, out]
    :param seq_length: [t, n, out]
    :param dh_prev:
        output real dh_prev when cur_t == t
        otherwise, output dh_pre_t for next cell
    :param dgate_h:
    :param dnt_x:
    :param t_state: means cur_t
    :param gate_order:
    :param kernel_name:
    :return:
    """
    _check_dtype(dh_pre_t, h, dy, dh, update, reset, new, hidden_new)
    _check_param(dh_pre_t, h, dy, dh, update, reset, new, hidden_new)
    _check_attr(gate_order)

    tik_instance = tik.Tik(tik.Dprofile())
    cell = AUGRUHiddenGradCell(tik_instance, h, dy, dnt_x, seq_length, t_state, gate_order, kernel_name)
    cell.compute()
    cell.build()
