"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

lstm_grad
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.utils import errormgr
from .dynamic_gru_cell_base import TikOpBase


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    OP_NAME = "DynamicGRUGradCell"
    TILING_ARG_NUM = 12
    INT64 = 'int64'
    INT32 = 'int32'
    INT32_MAX_NUM = 2**32 - 1


# 'pylint: disable=too-many-arguments,invalid-name
def _check_dtype(dh_pre_t, h, dy, dh, update, reset, new, hidden_new, init_h, t_state):
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
    para_check.check_dtype(init_h["dtype"], ["float16", "float32"], "init_h")
    para_check.check_dtype(t_state["dtype"], ["int32"], "t_state")


# 'pylint: disable=too-many-arguments,invalid-name
def _check_param(dh_pre_t, h, dy, dh, update, reset, new, hidden_new, init_h):
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
    para_check.check_shape_size(init_h["shape"])


def _check_attr(gate_order):
    """
    check attr
    """
    if gate_order not in ['zrh', 'rzh']:
        rule_desc = "gate_order should be zrh or rzh, but current attr is " + gate_order
        errormgr.raise_err_check_params_rules(Constant.OP_NAME, rule_desc, 'gate_order', gate_order)


# 'pylint: disable=too-many-statements,too-many-locals,unused-argument,invalid-name,too-many-instance-attributes
class DynamicGRUCellGrad(TikOpBase):
    """ DynamicGRUCellGrad
    """
    # 'pylint: disable=too-many-statements,too-many-locals,unused-argument,invalid-name
    def __init__(self, tik_instance, h, seq_mask, gate_order, kernel_name):
        """ init DynamicGRUCellGrad
        """
        super(DynamicGRUCellGrad, self).__init__(tik_instance)
        self.tiling_dtype = Constant.INT64
        self.t_size = None
        self.fuse_size = None
        self.max_core_num = None
        self.t_state = None
        self.tiling_map = {}

        self.gate_order = gate_order
        self.kernel_name = kernel_name
        self.device_aicore_num = self.tik_instance.d_profiling.get_aicore_num()
        self.ub_byte_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.dtype = h["dtype"]
        self.input_data_size = self.get_data_size(self.dtype)
        t_state_and_tiling_size = (16 * self.input_data_size + 16 * 8) * 2
        align = 256 // self.input_data_size
        ub_max_ele_num = (self.ub_byte_size - t_state_and_tiling_size) // self.input_data_size
        self.max_block_ele_num = (ub_max_ele_num // 8 // 2 // align) * align
        self.max_mem_size = self.input_data_size * self.max_block_ele_num

        # t offset for dy, update, reset, new, hidden_new
        self.t_offset = self.tik_instance.Scalar(self.tiling_dtype, name="t_offset", init_value=0)
        # ht offset for h
        self.ht_offset = self.tik_instance.Scalar(self.tiling_dtype, name="ht_offset", init_value=0)
        self.has_seq_mask = True if seq_mask else False

        # input
        self.dh_pre_t = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), \
        tbe_platform.scope_gm, "dh_pre_t")
        self.h = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), tbe_platform.scope_gm, "h")
        self.dy = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), tbe_platform.scope_gm, "dy")
        self.dh = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), tbe_platform.scope_gm, "dh")
        self.i2 = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), tbe_platform.scope_gm, "update")
        self.r2 = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), tbe_platform.scope_gm, "reset")
        self.n2 = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), tbe_platform.scope_gm, "n2")
        self.n2_mid = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), \
        tbe_platform.scope_gm, "hidden_new")
        self.init_h = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), tbe_platform.scope_gm, "init_h")
        self.t_state_gm = self.tik_instance.Tensor(Constant.INT32, (1,), tbe_platform.scope_gm, "t_state")
        if self.has_seq_mask:
            self.seq_mask_t = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), \
                                                       tbe_platform.scope_gm, "seq_mask")

        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), \
        tbe_platform.scope_gm, "ddr_arg")

        # output
        self.dh_prev = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), tbe_platform.scope_gm, "dh_prev")
        self.d_gate_h = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), \
        tbe_platform.scope_gm, "d_gate_h")
        self.dnt_x = self.tik_instance.Tensor(self.dtype, (Constant.INT32_MAX_NUM,), tbe_platform.scope_gm, "dnt_x")

    def build(self):
        """
        build cce
        """
        config_map = {"dump_cce_code": False}
        if self.has_seq_mask:
            input_list = (self.dh_pre_t, self.h, self.dy, self.dh, self.i2, self.r2, self.n2, self.n2_mid,
                          self.init_h, self.t_state_gm, self.seq_mask_t)
        else:
            input_list = (self.dh_pre_t, self.h, self.dy, self.dh, self.i2, self.r2, self.n2, self.n2_mid,
                          self.init_h, self.t_state_gm)

        output_list = (self.dh_prev, self.d_gate_h, self.dnt_x)
        self.tik_instance.BuildCCE(self.kernel_name,
                                   input_list,
                                   output_list,
                                   flowtable=(self.tiling_gm,),
                                   config=config_map)
        tbe_context.get_context().add_compile_info("vars", {
            "device_aicore_num": self.device_aicore_num,
            "ub_size": self.ub_byte_size
        })

    def compute(self):
        """ do compute
        """
        self._build_tiling_info()
        tiling = self.tiling_map
        core_num = tiling["core_num"]
        with self.tik_instance.for_range(0, core_num, block_num=core_num) as block_idx:
            tail_core_num = tiling["tail_core_num"]
            ele_num = tiling["loop_ele"]
            tail_ele_num = tiling["tail_loop_ele"]
            offset = tiling["block_size"] * core_num
            tail_num = tiling["tail_num"]
            with self.tik_instance.if_scope(tiling["loop_num"] > 0):
                with self.tik_instance.if_scope(block_idx < core_num):
                    with self.tik_instance.for_range(0, tiling["loop_num"]) as loop_idx:
                        base_offset = tiling["block_size"] * block_idx + ele_num * loop_idx
                        self._do_compute(base_offset, ele_num)
            with self.tik_instance.if_scope(tail_num > 0):
                with self.tik_instance.if_scope(block_idx < tail_core_num):
                    base_offset = offset + block_idx * tail_ele_num
                    with self.tik_instance.if_scope(block_idx < tail_core_num - 1):
                        self._do_compute(base_offset, tail_ele_num)
                    with self.tik_instance.else_scope():
                        self._do_compute(base_offset, tiling["tail_last_ele"])

    def _build_tiling_info(self):
        tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), \
        tik.scope_ubuf, "tiling_ub")
        self.move_data(tiling_ub, self.tiling_gm, self.tiling_dtype, (Constant.TILING_ARG_NUM,))

        self.fuse_size = self.tik_instance.Scalar(self.tiling_dtype, name='fuse_size')
        self.fuse_size.set_as(tiling_ub[9])
        self.t_size = self.tik_instance.Scalar(self.tiling_dtype, name='t_size')
        self.t_size.set_as(tiling_ub[8])

        core_num = self.tik_instance.Scalar(self.tiling_dtype, name='core_num')
        loop_num = self.tik_instance.Scalar(self.tiling_dtype, name='loop_num')
        loop_ele = self.tik_instance.Scalar(self.tiling_dtype, name='loop_ele')
        block_size = self.tik_instance.Scalar(self.tiling_dtype, name='block_size')
        tail_num = self.tik_instance.Scalar(self.tiling_dtype, name='tail_num')
        tail_core_num = self.tik_instance.Scalar(self.tiling_dtype, name='tail_core_num')
        tail_loop_ele = self.tik_instance.Scalar(self.tiling_dtype, name='tail_loop_ele')
        tail_last_ele = self.tik_instance.Scalar(self.tiling_dtype, name='tail_last_ele')
        for index, arg in enumerate(
            [core_num, loop_num, loop_ele, block_size, tail_num, tail_core_num, tail_loop_ele, tail_last_ele]):
            arg.set_as(tiling_ub[index])
        self._set_t_state()
        # scalar t offset for dy, update, reset, new, hidden new
        self.t_offset.set_as(self.t_size - self.t_state - 1)

        with self.tik_instance.if_scope(self.t_size != 1):
            self.ht_offset.set_as(self.t_size - self.t_state - 2)

        self.tiling_map = {
            "core_num": core_num,
            "loop_num": loop_num,
            "loop_ele": loop_ele,
            "block_size": block_size,
            "tail_num": tail_num,
            "tail_core_num": tail_core_num,
            "tail_loop_ele": tail_loop_ele,
            "tail_last_ele": tail_last_ele
        }

    def _set_t_state(self):
        t_state_ub = self.tik_instance.Tensor(Constant.INT32, (4,), tbe_platform.scope_ubuf, "t_state_ub")
        self.tik_instance.data_move(t_state_ub, self.t_state_gm, 0, 1, 1, 0, 0)
        self.t_state = self.tik_instance.Scalar(Constant.INT32, name="t_state")
        self.t_state.set_as(t_state_ub[0])

    # 'pylint: disable=too-many-statements,too-many-locals,unused-argument,invalid-name
    def _do_compute(self, input_offset, ele_num):
        """
        do compute
        """
        shape = (ele_num,)
        dh = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "dh", max_mem_size=self.max_mem_size)
        self.move_data(dh, self.dh[input_offset], self.dtype, shape)
        with self.tik_instance.if_scope(self.t_state > 0):
            dh_pre_t = self.tik_instance.Tensor(self.dtype,
                                                shape,
                                                tbe_platform.scope_ubuf,
                                                "dh_pre_t",
                                                max_mem_size=self.max_mem_size)
            self.move_data(dh_pre_t, self.dh_pre_t[input_offset], self.dtype, shape)
            self.vadd_func(dh, dh, dh_pre_t, shape)
        if self.has_seq_mask:
            seq_mask_ub = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "seq_mask")
            self.move_data(seq_mask_ub, self.seq_mask_t[input_offset], self.dtype, shape)
            self.vmul_func(dh, dh, seq_mask_ub, shape)

        with self.tik_instance.if_scope(self.t_state == self.t_size):
            # just cal dh + dh_pre_t in last cell
            self.move_data(self.dh_prev[input_offset], dh, self.dtype, shape)
        with self.tik_instance.else_scope():
            dy = self.tik_instance.Tensor(self.dtype,
                                          shape,
                                          tbe_platform.scope_ubuf,
                                          "dy",
                                          max_mem_size=self.max_mem_size)
            self.move_data(dy, self.dy[self.t_offset * self.fuse_size + input_offset], self.dtype, shape)
            dh_add_dy = dh
            self.vadd_func(dh_add_dy, dh, dy, shape)  # free dy
            i2 = self.tik_instance.Tensor(self.dtype,
                                          shape,
                                          tbe_platform.scope_ubuf,
                                          "i2",
                                          max_mem_size=self.max_mem_size)
            self.move_data(i2, self.i2[self.t_offset * self.fuse_size + input_offset], self.dtype, shape)

            # cal dh_pre_t for next cell, output to dh_prev
            dh_pre_t = self.tik_instance.Tensor(self.dtype,
                                                shape,
                                                tbe_platform.scope_ubuf,
                                                "dh_pre_t",
                                                max_mem_size=self.max_mem_size)
            self.vmul_func(dh_pre_t, dh_add_dy, i2, shape)
            self.move_data(self.dh_prev[input_offset], dh_pre_t, self.dtype, shape)

            # cal concat
            one = self.tik_instance.Tensor(self.dtype,
                                           shape,
                                           tbe_platform.scope_ubuf,
                                           "one",
                                           max_mem_size=self.max_mem_size)
            self.vector_dup_func(one, 1, shape)
            n2 = self.tik_instance.Tensor(self.dtype,
                                          shape,
                                          tbe_platform.scope_ubuf,
                                          "n2",
                                          max_mem_size=self.max_mem_size)
            self.move_data(n2, self.n2[self.t_offset * self.fuse_size + input_offset], self.dtype, shape)
            power_n2 = self.tik_instance.Tensor(self.dtype,
                                                shape,
                                                tbe_platform.scope_ubuf,
                                                "power_n2",
                                                max_mem_size=self.max_mem_size)
            self.vmul_func(power_n2, n2, n2, shape)
            one_sub_power_n2 = power_n2
            self.vsub_func(one_sub_power_n2, one, power_n2, shape)
            one_sub_i2 = self.tik_instance.Tensor(self.dtype,
                                                  shape,
                                                  tbe_platform.scope_ubuf,
                                                  "one_sub_i2",
                                                  max_mem_size=self.max_mem_size)
            self.vsub_func(one_sub_i2, one, i2, shape)
            n2_mul_i2 = one_sub_power_n2
            self.vmul_func(n2_mul_i2, one_sub_power_n2, one_sub_i2, shape)
            dn2i = n2_mul_i2
            self.vmul_func(dn2i, n2_mul_i2, dh_add_dy, shape)
            # dn2i -> out
            self.move_data(self.dnt_x[input_offset], dn2i, self.dtype, shape)

            # cal di2
            h1 = self.tik_instance.Tensor(self.dtype,
                                          shape,
                                          tbe_platform.scope_ubuf,
                                          "h1",
                                          max_mem_size=self.max_mem_size)
            with self.tik_instance.if_scope(self.t_state == self.t_size - 1):
                self.move_data(h1, self.init_h[input_offset], self.dtype, shape)
            with self.tik_instance.else_scope():
                self.move_data(h1, self.h[self.ht_offset * self.fuse_size + input_offset], self.dtype, shape)
            h1_sub_n2 = h1
            self.vsub_func(h1_sub_n2, h1, n2, shape)
            # `(1-i2)*i2`
            one_i2_mul_i2 = one_sub_i2
            self.vmul_func(one_i2_mul_i2, one_sub_i2, i2, shape)  # free i2
            # `(dh2 + dy)*(1-i2)*i2`
            dh2_mul_i2 = one_i2_mul_i2
            self.vmul_func(dh2_mul_i2, one_i2_mul_i2, dh_add_dy, shape)
            # `(h1-n2)*(dh2 + dy)*(1-i2)*i2`
            di2 = dh2_mul_i2
            self.vmul_func(di2, dh2_mul_i2, h1_sub_n2, shape)  # free h1
            # `di2 -> out`
            if self.gate_order == "zrh":
                offset = input_offset
            else:
                offset = self.fuse_size + input_offset
            self.move_data(self.d_gate_h[offset], di2, self.dtype, shape)

            r2 = self.tik_instance.Tensor(self.dtype,
                                          shape,
                                          tbe_platform.scope_ubuf,
                                          "r2",
                                          max_mem_size=self.max_mem_size)
            self.move_data(r2, self.r2[self.t_offset * self.fuse_size + input_offset], self.dtype, shape)
            dn2h = dn2i
            self.vmul_func(dn2h, dn2i, r2, shape)
            # dn2h -> out
            self.move_data(self.d_gate_h[self.fuse_size * 2 + input_offset], dn2h, self.dtype, shape)

            one_sub_r2 = r2
            self.vsub_func(one_sub_r2, one, r2, shape)
            n2_mid = self.tik_instance.Tensor(self.dtype,
                                              shape,
                                              tbe_platform.scope_ubuf,
                                              "n2_mid",
                                              max_mem_size=self.max_mem_size)
            self.move_data(n2_mid, self.n2_mid[self.t_offset * self.fuse_size + input_offset], self.dtype, shape)
            mid_mul_r2 = one_sub_i2
            self.vmul_func(mid_mul_r2, one_sub_r2, n2_mid, shape)
            dr2 = mid_mul_r2
            self.vmul_func(dr2, mid_mul_r2, dn2h, shape)
            # dr2 -> out
            if self.gate_order == "zrh":
                offset = self.fuse_size + input_offset
            else:
                offset = input_offset
            self.move_data(self.d_gate_h[offset], dr2, self.dtype, shape)


# 'pylint: disable=too-many-arguments,too-many-locals,unused-argument
@register_operator('DynamicGRUCellGrad')
def dynamic_gru_cell_grad(dh_pre_t,
                          h,
                          dy,
                          dh,
                          update,
                          reset,
                          new,
                          hidden_new,
                          init_h,
                          t_state,
                          seq_length,
                          dh_prev,
                          dgate_h,
                          dnt_x,
                          gate_order="zrh",
                          kernel_name="dynamic_gru_cell_grad"):
    """
    Calculate the gradient
    Parameters
    -----------
    :param dh_pre_t: result of (dh2 + dy) * i2 at (cur_t -1)
        when t_state > 0, dh = dh + dh_pre_t
    :param h: [t, n, out]; set init_h(t_state = t_size - 1) [n, out]
    :param dy: [t, n, out]
    :param dh: [n, out]
    :param update: [t, n, out]
    :param reset: [t, n, out]
    :param new: [t, n, out]
    :param hidden_new: [t, n, out]
    :param init_h: [t, n, out];
    :param t_state: means cur_t
    :param seq_length: [t, n, out]
    :param dh_prev:
        output real dh_prev when cur_t == t
        otherwise, output dh_pre_t for next cell
    :param dgate_h:
    :param dnt_x:
    :param gate_order:
    :param kernel_name:
    :return:
    """
    _check_dtype(dh_pre_t, h, dy, dh, update, reset, new, hidden_new, init_h, t_state)
    _check_param(dh_pre_t, h, dy, dh, update, reset, new, hidden_new, init_h)
    _check_attr(gate_order)

    tik_instance = tik.Tik(tik.Dprofile())
    cell = DynamicGRUCellGrad(tik_instance, h, seq_length, gate_order, kernel_name)
    cell.compute()
    cell.build()
    return cell
