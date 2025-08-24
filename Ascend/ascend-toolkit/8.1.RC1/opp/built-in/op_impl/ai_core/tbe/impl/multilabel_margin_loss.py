#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
multilabel_margin_loss
"""
import operator
import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check


def _check_shape(shape_predict, shape_label):
    if operator.ne(list(shape_predict), list(shape_label)):
        raise RuntimeError("predict and label must have the same shape !")


class MultilabelMarginLoss:
    """
        object
    """
    def __init__(self, x, target, reduction, kernel_name):
        """
        init data

        Parameters
        ----------
        x : dict
        include keys(shape and dtype)
        target : dict
        include keys(shape and dtype)
        reduction : str
        mean
        kernel_name : str
        kernel name, default value is "multilabel_margin_loss"

        Returns
        -------
        """
        self.ori_type = x.get("dtype")

        self.shape_input = x.get("shape")
        self.dtype_input = "float32"

        self.shape_target = target.get("shape")
        self.dtype_target = target.get("dtype")

        self.reduction = reduction
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()

        self.nd_flag = len(self.shape_target) == 2

        # check
        _check_shape(self.shape_input, self.shape_target)

        check_input_tuple = ("float16", "float32")
        para_check.check_dtype_rule(self.ori_type, check_input_tuple)

        check_target_tuple = ("int32")
        para_check.check_dtype_rule(self.dtype_target, check_target_tuple)

        check_redution_tuple = ("none", "mean", "sum")
        para_check.check_dtype_rule(self.reduction, check_redution_tuple)

        # cal multi core
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 8192)
        self.var_dtype_bytes_size = tbe_platform.get_bit_len(self.dtype_input) // 8
        self.target_dtype_bytes_size = tbe_platform.get_bit_len(self.dtype_target) // 8
        self.var_data_each_block = 32 // self.var_dtype_bytes_size
        self.target_data_each_block = 32 // self.target_dtype_bytes_size

        if self.nd_flag:
            self.nframe = self.shape_input[0]
            self.update_data_num = self.shape_input[1]
        else:
            self.nframe = 1
            self.update_data_num = self.shape_input[0]

        if self.update_data_num < self.var_data_each_block or self.nframe == 1 or self.ori_type == "float16":
            self.block_num = 1
            self.frame_step = self.nframe
        else:
            ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
            self.frame_step = math.ceil(self.nframe / ai_core_num)
            self.block_num = math.ceil(self.nframe / self.frame_step)

        # output_size
        if len(self.shape_input) == 1 or reduction == "mean" or reduction == "sum":
            self.total_weight_size = 1
        else:
            self.total_weight_size = self.shape_input[0]

        # scope_gm
        self.input_gm = self.tik_instance.Tensor(
            self.ori_type, self.shape_input, name="input_gm", scope=tik.scope_gm)
        self.target_gm = self.tik_instance.Tensor(
            self.dtype_target, self.shape_target, name="target_gm", scope=tik.scope_gm)

        self.total_weight = self.tik_instance.Tensor(
            self.ori_type, [self.total_weight_size], name="total_weight", scope=tik.scope_gm, is_atomic_add=True)
        self.is_target = self.tik_instance.Tensor(
            self.dtype_target, self.shape_target, name="is_target_gm", scope=tik.scope_gm, is_atomic_add=True)

        # init target
        self.target_ub_tensor_size = None
        self.is_target_ub = None
        self.target_ub = None
        self.flag = None
        self.frame_index = None

    def init_target_ub(self):
        """
        init_target_ub
        """
        self.target_ub_tensor_size = (
                ((self.ub_size_bytes // self.target_dtype_bytes_size) - self.target_data_each_block) //
                self.target_data_each_block * self.target_data_each_block)

        self.is_target_ub = self.tik_instance.Tensor(
            self.dtype_target, (self.target_data_each_block,),
            name="is_target_ub",
            scope=tik.scope_ubuf)

        self.target_ub = self.tik_instance.Tensor(
            self.dtype_target, (self.target_ub_tensor_size,),
            name="target_ub",
            scope=tik.scope_ubuf)

    def init_ub(self, target, value, move_num, data_each_block):
        """
        init target by value
        """
        vector_mask_max = 8 * data_each_block
        vadd_loop = move_num // (vector_mask_max * 255)
        add_offset = 0
        if vadd_loop > 0:
            with self.tik_instance.for_range(0, vadd_loop) as add_index:
                add_offset = add_index * vector_mask_max * 255
                self.tik_instance.vec_dup(vector_mask_max, target[add_offset], value, 255, 8)

        repeat_time = (move_num % (vector_mask_max * 255) // vector_mask_max)
        if repeat_time > 0:
            add_offset = vadd_loop * vector_mask_max * 255
            self.tik_instance.vec_dup(vector_mask_max, target[add_offset], value, repeat_time, 8)

        last_num = move_num % vector_mask_max
        if last_num > 0:
            add_offset += repeat_time * vector_mask_max
            self.tik_instance.vec_dup(last_num, target[add_offset], value, 1, 8)

    def compute_is_target_each_loop(self, move_offset, need_num, data_each_block, dim):
        """
        compute_is_target_each_loop
        Parameters
        ----------
        move_offset : int

        need_num : int

        data_each_block : int

        dim: int
        """
        burse_len = (need_num + data_each_block) // data_each_block
        self.tik_instance.data_move(self.target_ub, self.target_gm[move_offset], 0, 1, burse_len, 0, 0)

        one_reg = self.tik_instance.Scalar(dtype=self.dtype_target)
        one_reg.set_as(1)

        with self.tik_instance.for_range(0, need_num) as index:
            block_start = self.tik_instance.Scalar(dtype=self.dtype_target)
            block_index = self.tik_instance.Scalar(dtype=self.dtype_target)
            is_target_s = self.tik_instance.Scalar(dtype=self.dtype_target)
            need_back = self.tik_instance.Scalar(dtype=self.dtype_target)
            with self.tik_instance.if_scope((index + move_offset) % dim == 0):
                self.flag.set_as(1)
                self.frame_index.set_as(self.frame_index + dim)
            with self.tik_instance.if_scope(self.flag == 1):
                with self.tik_instance.if_scope(self.target_ub[index] > -1):
                    block_index.set_as(self.target_ub[index])
                    block_index.set_as(block_index % data_each_block)

                    block_start.set_as(self.target_ub[index])
                    block_start.set_as(block_start // data_each_block * data_each_block)
                    block_start.set_as(block_start + self.frame_index)

                    with self.tik_instance.if_scope(block_start + data_each_block > (self.frame_index + dim)):
                        need_back.set_as((block_start + data_each_block) - (self.frame_index + dim))
                        block_start.set_as(block_start - need_back)
                        block_index.set_as(block_index + need_back)

                    self.tik_instance.data_move(self.is_target_ub, self.is_target[block_start], 0, 1, 1, 0, 0)
                    is_target_s.set_as(self.is_target_ub[block_index])
                    is_target_s.set_as(is_target_s + one_reg)
                    self.is_target_ub[block_index].set_as(is_target_s)
                    self.tik_instance.data_move(self.is_target[block_start], self.is_target_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.flag.set_as(0)

    def compute_is_target_per_core(self, target_loop_index):
        """
        compute_is_target_per_core

        Parameters
        target_loop_index: int
            the index of target

        return

        """
        move_begin = self.tik_instance.Scalar(dtype=self.dtype_target)
        move_offset = self.tik_instance.Scalar(dtype=self.dtype_target)
        move_num = self.tik_instance.Scalar(dtype=self.dtype_target)

        move_begin.set_as(target_loop_index * self.frame_step * self.update_data_num)

        with self.tik_instance.if_scope(target_loop_index == self.block_num - 1):
            move_num.set_as(self.nframe * self.update_data_num - move_begin)
        with self.tik_instance.else_scope():
            move_num.set_as(self.frame_step * self.update_data_num)

        self.flag = self.tik_instance.Scalar(dtype=self.dtype_target)
        self.flag.set_as(1)
        self.frame_index = self.tik_instance.Scalar(dtype=self.dtype_target)
        self.frame_index.set_as(move_begin - self.update_data_num)

        loop_time = move_num // self.target_ub_tensor_size
        with self.tik_instance.if_scope(loop_time > 0):
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                move_offset.set_as(move_begin + loop_index * self.target_ub_tensor_size)
                self.compute_is_target_each_loop(
                    move_offset, self.target_ub_tensor_size, self.target_data_each_block, self.update_data_num)

        move_offset.set_as(move_begin + loop_time * self.target_ub_tensor_size)
        last_num = move_num % self.target_ub_tensor_size

        with self.tik_instance.if_scope(last_num > 0):
            self.compute_is_target_each_loop(move_offset, last_num, self.target_data_each_block, self.update_data_num)

    # 'pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def compute_total_weight_per_core(self, target_loop_index):
        """
        compute_is_target_per_core
        Parameters
        target_loop_index: int
        """
        block_bite_size = 32

        dtype_input_bytes_size = tbe_platform.get_bit_len(self.dtype_input) // 8
        data_input_each_block = block_bite_size // dtype_input_bytes_size
        vec_data_input_each_block = 8 * data_input_each_block

        dtype_target_bytes_size = tbe_platform.get_bit_len(self.dtype_target) // 8
        data_target_each_block = block_bite_size // dtype_target_bytes_size
        vec_data_target_each_block = 8 * data_target_each_block

        dtype_ori_bytes_size = tbe_platform.get_bit_len(self.ori_type) // 8
        data_ori_each_block = block_bite_size // dtype_ori_bytes_size

        sum_loss_ub = self.tik_instance.Tensor(
            self.dtype_input, (vec_data_input_each_block,),
            name="sum_loss_ub",
            scope=tik.scope_ubuf)
        total_weight_ub = self.tik_instance.Tensor(
            self.dtype_input, (vec_data_input_each_block,),
            name="total_weight_ub",
            scope=tik.scope_ubuf)
        total_weight_16_ub = self.tik_instance.Tensor(
            self.ori_type, (vec_data_input_each_block,),
            name="total_weight_16_ub",
            scope=tik.scope_ubuf)
        sum_loss_16_ub = self.tik_instance.Tensor(
            self.ori_type, (vec_data_input_each_block,),
            name="sum_loss_16_ub",
            scope=tik.scope_ubuf)

        ans = self.tik_instance.Scalar(dtype=self.dtype_input)
        ans.set_as(0)
        ans_16 = self.tik_instance.Scalar(dtype=self.ori_type)
        ans_16.set_as(0)

        self.init_ub(sum_loss_ub, ans, vec_data_input_each_block, data_input_each_block)

        zero_ub = self.tik_instance.Tensor(
            self.dtype_input, (vec_data_input_each_block,),
            name="zero_ub",
            scope=tik.scope_ubuf)
        self.init_ub(zero_ub, ans, vec_data_input_each_block, data_input_each_block)

        dim = self.update_data_num

        end = self.tik_instance.Scalar(dtype=self.dtype_target)
        with self.tik_instance.if_scope(target_loop_index == self.block_num - 1):
            end.set_as(self.nframe)
        with self.tik_instance.else_scope():
            end.set_as((target_loop_index + 1) * self.frame_step)

        with self.tik_instance.for_range(
                target_loop_index * self.frame_step, end) as frame_idx:
            with self.tik_instance.if_scope(frame_idx < self.nframe):
                input_l_ub = self.tik_instance.Tensor(
                    self.dtype_input, (vec_data_input_each_block,),
                    name="input_l_ub",
                    scope=tik.scope_ubuf)
                is_target_l_ub = self.tik_instance.Tensor(
                    self.dtype_target, (vec_data_target_each_block,),
                    name="is_target_l_ub",
                    scope=tik.scope_ubuf)
                is_target_input_ub = self.tik_instance.Tensor(
                    self.dtype_input, (vec_data_input_each_block,),
                    name="is_target_input_ub",
                    scope=tik.scope_ubuf)

                input_r_ub = self.tik_instance.Tensor(
                    self.dtype_input, (vec_data_input_each_block,),
                    name="input_r_ub",
                    scope=tik.scope_ubuf)
                loss_ub = self.tik_instance.Tensor(
                    self.dtype_input, (vec_data_input_each_block,),
                    name="loss_ub",
                    scope=tik.scope_ubuf)
                is_target_r_ub = self.tik_instance.Tensor(
                    self.dtype_target, (vec_data_target_each_block,),
                    name="is_target_r_ub",
                    scope=tik.scope_ubuf)

                input_r_16_ub = self.tik_instance.Tensor(
                    self.ori_type, (vec_data_input_each_block,),
                    name="input_r_16_ub",
                    scope=tik.scope_ubuf)
                input_l_16_ub = self.tik_instance.Tensor(
                    self.ori_type, (vec_data_input_each_block,),
                    name="input_l_16_ub",
                    scope=tik.scope_ubuf)
                loss_16_ub = self.tik_instance.Tensor(
                    self.ori_type, (vec_data_input_each_block,),
                    name="loss_16_ub",
                    scope=tik.scope_ubuf)

                self.init_ub(loss_ub, ans, vec_data_input_each_block, data_input_each_block)

                is_target_s = self.tik_instance.Scalar(dtype=self.dtype_input)

                last_target_r_start = self.tik_instance.Scalar(dtype=self.dtype_target)
                last_target_r_start.set_as(-1)
                last_input_r_start = self.tik_instance.Scalar(dtype=self.dtype_target)
                last_input_r_start.set_as(-1)
                tmp = self.tik_instance.Scalar(dtype=self.dtype_input)
                add_tmp = self.tik_instance.Scalar(dtype=self.dtype_input)

                with self.tik_instance.for_range(0, dim) as dim_r_idx:
                    r_start = frame_idx * dim + dim_r_idx

                    input_r_start = r_start // vec_data_input_each_block * vec_data_input_each_block
                    input_r_index = r_start % vec_data_input_each_block

                    target_r_start = r_start // vec_data_target_each_block * vec_data_target_each_block
                    target_r_index = r_start % vec_data_target_each_block

                    with self.tik_instance.if_scope(last_target_r_start != target_r_start):
                        self.tik_instance.data_move(is_target_r_ub, self.is_target[target_r_start], 0, 1, 8, 0, 0)
                        last_target_r_start.set_as(target_r_start)
                    with self.tik_instance.if_scope(last_input_r_start != input_r_start):
                        if self.ori_type == "float16":
                            self.tik_instance.data_move(input_r_16_ub, self.input_gm[input_r_start], 0, 1, 8, 0, 0)
                            self.tik_instance.vec_conv(vec_data_input_each_block, '',
                                                       input_r_ub, input_r_16_ub, 1, 8, 8)
                        else:
                            self.tik_instance.data_move(input_r_ub, self.input_gm[input_r_start], 0, 1, 8, 0, 0)
                        self.tik_instance.vec_adds(vec_data_input_each_block, input_r_ub[0], input_r_ub[0], 1, 1, 8, 8)
                        last_input_r_start.set_as(input_r_start)

                    with self.tik_instance.if_scope(is_target_r_ub[target_r_index] == 0):
                        tmp.set_as(input_r_ub[input_r_index])
                        last_target_l_start = self.tik_instance.Scalar(dtype=self.dtype_target)
                        last_target_l_start.set_as(-1)
                        last_input_l_start = self.tik_instance.Scalar(dtype=self.dtype_target)
                        last_input_l_start.set_as(-1)
                        with self.tik_instance.for_range(0, dim) as dim_l_idx:
                            l_start = frame_idx * dim + dim_l_idx

                            input_l_start = l_start // vec_data_input_each_block * vec_data_input_each_block
                            input_l_index = l_start % vec_data_input_each_block

                            target_l_start = l_start // vec_data_target_each_block * vec_data_target_each_block
                            target_l_index = l_start % vec_data_target_each_block

                            with self.tik_instance.if_scope(last_target_l_start != target_l_start):
                                self.tik_instance.data_move(is_target_l_ub, self.is_target[target_l_start],
                                                            0, 1, 8, 0, 0)
                                if self.dtype_input == 'float16':
                                    self.tik_instance.vec_conv(vec_data_target_each_block, '',
                                                               is_target_input_ub, is_target_l_ub, 1, 8, 8, 1.0)
                                else:
                                    self.tik_instance.vec_conv(vec_data_target_each_block, '',
                                                               is_target_input_ub, is_target_l_ub, 1, 8, 8)
                                last_target_l_start.set_as(target_l_start)
                            with self.tik_instance.if_scope(last_input_l_start != input_l_start):
                                if self.ori_type == "float16":
                                    self.tik_instance.data_move(input_l_16_ub, self.input_gm[input_l_start],
                                                                0, 1, 8, 0, 0)
                                    self.tik_instance.vec_conv(vec_data_input_each_block, '',
                                                               input_l_ub, input_l_16_ub, 1, 8, 8)
                                else:
                                    self.tik_instance.data_move(input_l_ub, self.input_gm[input_l_start], 0, 1, 8, 0, 0)
                                self.tik_instance.vec_muls(
                                    vec_data_input_each_block, input_l_ub[0], input_l_ub[0], -1, 1, 8, 8)
                                self.tik_instance.vec_adds(
                                    vec_data_input_each_block, input_l_ub[0], input_l_ub[0], tmp, 1, 8, 8)
                                self.tik_instance.vec_max(
                                    vec_data_input_each_block, input_l_ub[0], input_l_ub[0], zero_ub[0], 1, 8, 8, 8)
                                last_input_l_start.set_as(input_l_start)

                            with self.tik_instance.if_scope(is_target_l_ub[target_l_index] != 0):
                                is_target_s.set_as(is_target_input_ub[target_l_index])
                                add_tmp.set_as(input_l_ub[input_l_index])
                                add_tmp.set_as(add_tmp * is_target_s)
                                self.tik_instance.vec_adds(
                                    vec_data_input_each_block, loss_ub[0], loss_ub[0], add_tmp, 1, 8, 8)

                self.tik_instance.vec_muls(
                    vec_data_input_each_block, loss_ub[0], loss_ub[0], 1.0 / dim, 1, 8, 8)
                self.tik_instance.vec_add(
                    vec_data_input_each_block, sum_loss_ub[0], loss_ub[0], sum_loss_ub[0], 1, 8, 8, 8)

                self.init_ub(total_weight_ub, ans, vec_data_input_each_block, data_input_each_block)
                self.init_ub(total_weight_16_ub, ans_16, vec_data_input_each_block, data_ori_each_block)

                if self.reduction == "none":
                    if self.ori_type == "float32":
                        total_weight_ub[0].set_as(loss_ub[0])
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(
                            self.total_weight[frame_idx], total_weight_ub, 0, 1, 1, 0, 0)
                        self.tik_instance.set_atomic_add(0)
                    else:
                        total_weight_start = frame_idx // vec_data_input_each_block * vec_data_input_each_block
                        total_weight_index = frame_idx % vec_data_input_each_block

                        self.tik_instance.data_move(
                            total_weight_16_ub, self.total_weight[total_weight_start], 0, 1, 8, 0, 0)
                        self.tik_instance.vec_conv(vec_data_input_each_block, '',
                                                   loss_16_ub, loss_ub, 1, 8, 8)
                        total_weight_16_ub[total_weight_index].set_as(loss_16_ub[0])
                        self.tik_instance.data_move(
                            self.total_weight[total_weight_start], total_weight_16_ub, 0, 1, 8, 0, 0)

        if self.reduction == "mean":
            self.tik_instance.vec_muls(
                vec_data_input_each_block, sum_loss_ub[0], sum_loss_ub[0], 1.0 / self.nframe, 1, 8, 8)

        if self.reduction == "sum" or self.reduction == "mean":
            if self.ori_type == "float16":
                self.tik_instance.vec_conv(vec_data_input_each_block, '',
                                           sum_loss_16_ub, sum_loss_ub, 1, 8, 8)
                self.tik_instance.data_move(self.total_weight[0], sum_loss_16_ub[0], 0, 1, 8, 0, 0)
            else:
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.total_weight[0], sum_loss_ub[0], 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)

    def compute(self):
        """
        calculating data
        """
        with self.tik_instance.for_range(
                0, self.block_num,
                block_num=self.block_num) as target_loop_index:
            flag = self.tik_instance.Scalar(dtype=self.dtype_input)
            flag.set_as(1)
            with self.tik_instance.if_scope(flag == 1):  # split scope
                self.init_target_ub()
                self.compute_is_target_per_core(target_loop_index)
            self.compute_total_weight_per_core(target_loop_index)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_gm, self.target_gm],
            outputs=[self.total_weight, self.is_target]
        )

        return self.tik_instance


# 'pylint: disable=unused-argument,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def multilabel_margin_loss(x, target, y, is_target,
                           reduction="mean", kernel_name="multilabel_margin_loss"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of x tensor, only support float16, float32
    target : dict
        shape and dtype of target, only support int64, int32
    y : dict
        shape and dtype of y tensor, only support float16, float32
    is_target : dict
        shape and dtype of is_target
    reduction : str
        support "none", "mean", "sum", default value is "mean"
    kernel_name : str
        kernel name, default value is "multilabel_margin_loss"

    Returns
    -------
    None
    """
    multilabel_margin_loss_instance = MultilabelMarginLoss(x, target, reduction, kernel_name)
    tik_instance = multilabel_margin_loss_instance.compute()
    return tik_instance
