#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
fixpipe process for conv3d_backprop_input fusion
"""
from typing import List
from tbe import tvm
from tbe.tvm import Tensor
from impl.fixpipe_op.fixpipe_base import FixpipeBase
from tbe.common.utils.errormgr import error_manager_cube
from tbe.dsl.base.operation import in_dynamic

LENGTH_FORMAT_NDC1HWC0 = 6
LENGTH_FORMAT_NDHWC = 5


class FixpipeConv3dBackpropInput(FixpipeBase):
    """
    Fixpipe for conv3d_backprop_input
    """

    def __init__(self, op_type: str, x1: Tensor, x2: (Tensor, None), quant_scale_0: (Tensor, None),
                 relu_weight_0: (Tensor, None), clip_value_0: (Tensor, None), quant_scale_1: (Tensor, None),
                 relu_weight_1: (Tensor, None), clip_value_1: (Tensor, None), anti_quant_scale: (Tensor, None),
                 anti_quant_offset: (Tensor, None), output: dict, fusion_op_list: List[str], unit_list: List[str],
                 eltwise_mode: str):
        """
        and assign its original properties to the dictionary additional_attrs
        """

        super().__init__(op_type, x1, x2, quant_scale_0, relu_weight_0, clip_value_0, quant_scale_1, relu_weight_1,
                         clip_value_1, anti_quant_scale, anti_quant_offset, output, fusion_op_list, unit_list,
                         eltwise_mode)
        self.additional_attrs = self.x1.op.attrs
        self.op_tag = "conv3d_backprop_input_"

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        fixpipe_reform_tag = "fixpipe_reform"
        self.additional_attrs["6HD_TRANS_NDHWC"] = False
        if self._is_nz2nd():
            self.additional_attrs["6HD_TRANS_NDHWC"] = True
            res_reform = tvm.compute(
                self.output_shape,
                lambda n, d, hw, c: res(n, d, c // self.input_shape[-1], hw, c % self.input_shape[-1]),
                name=fixpipe_reform_tag,
                tag=fixpipe_reform_tag,
                attrs=self.attrs)
        else:
            res_reform = tvm.compute(self.output_shape,
                                     lambda n, d, c1, hw, c0: res(n, d, c1, hw, c0),
                                     name=fixpipe_reform_tag,
                                     tag=fixpipe_reform_tag,
                                     attrs=self.attrs)

        return self.get_c_ddr_vn(res_reform)

    def get_dx_ddr_zero_padding_tensor(self):
        _stride_d = self.additional_attrs.get("stride_d")
        _kernel_d, _, _ = self.additional_attrs.get("kernels")
        if _stride_d >= _kernel_d:
            dx_filing_zero = tvm.compute(self.output_shape,
                                         lambda *index: tvm.select(self.filling_zero_coordinates(index, True),
                                                                   tvm.const(0, dtype=self.output_dtype)),
                                         name="dx_filing_zero",
                                         tag=self.op_tag + "dx_filing_zero")
            dx_ddr_zero = tvm.compute(
                self.output_shape,
                lambda *index: tvm.select(self.filling_zero_coordinates(index, True), dx_filing_zero[index]),
                name="c_ddr_zero",
                tag=self.op_tag + "c_ddr_zero")
        else:
            dx_filing_zero = tvm.compute(
                self.output_shape,
                lambda *index: tvm.select(self.filling_zero_coordinates(index), tvm.const(0, dtype=self.output_dtype)),
                name="dx_filing_zero",
                tag=self.op_tag + "dx_filing_zero")
            dx_ddr_zero = tvm.compute(
                self.output_shape,
                lambda *index: tvm.select(self.filling_zero_coordinates(index), dx_filing_zero[index]),
                name="c_ddr_zero",
                tag=self.op_tag + "c_ddr_zero")

        return dx_ddr_zero

    def get_dx_ddr_add_tensor(self):
        _stride_d = self.additional_attrs.get("stride_d")
        _kernel_d, _, _ = self.additional_attrs.get("kernels")
        _d_dim = self.output_shape[1]

        fixpipe_add = tvm.compute(self.input_shape,
                                  lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx: self.x2[
                                      dx_batch_idx * _d_dim + dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx],
                                  name="fixpipe_add",
                                  tag=self.op_tag + "fixpipe_add")

        if self._is_nz2nd():
            dx_ddr_add = tvm.compute(self.output_shape,
                                     lambda dx_batch_idx, dx_deep_idx, dx_hw_idx, dx_cin_idx: tvm.select(
                                         self.filling_zero_coordinates(
                                             (dx_batch_idx, dx_deep_idx, dx_hw_idx, dx_cin_idx), _stride_d > _kernel_d),
                                         fixpipe_add[dx_batch_idx, dx_deep_idx, dx_cin_idx // self.input_shape[-1],
                                                     dx_hw_idx, dx_cin_idx % self.input_shape[-1]]),
                                     name="c_ddr_add",
                                     tag=self.op_tag + "c_ddr_add")
        else:
            dx_ddr_add = tvm.compute(
                self.output_shape,
                lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx: tvm.select(
                    self.filling_zero_coordinates(
                        (dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx), _stride_d > _kernel_d),
                    fixpipe_add[dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx]),
                name="c_ddr_add",
                tag=self.op_tag + "c_ddr_add")

        return dx_ddr_add

    def get_c_ddr_vn(self, res_reform):
        if self.x2 is None:
            dx_l12out_tensor = self.get_dx_ddr_zero_padding_tensor()
        else:
            dx_l12out_tensor = self.get_dx_ddr_add_tensor()

        if self._is_nz2nd():
            c_ddr_vn = tvm.compute(self.output_shape,
                                   lambda dx_batch_idx, dx_deep_idx, dx_hw_idx, dx_cin_idx: res_reform[
                                       dx_batch_idx, dx_deep_idx, dx_hw_idx, dx_cin_idx] + dx_l12out_tensor[
                                           dx_batch_idx, dx_deep_idx, dx_hw_idx, dx_cin_idx],
                                   name="c_ddr_vn",
                                   tag=self.op_tag + "c_ddr_vn",
                                   attrs=self.additional_attrs)
        else:
            c_ddr_vn = tvm.compute(self.output_shape,
                                   lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx: res_reform[
                                       dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx] +
                                   dx_l12out_tensor[dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx],
                                   name="c_ddr_vn",
                                   tag=self.op_tag + "c_ddr_vn",
                                   attrs=self.additional_attrs)
        return c_ddr_vn

    def filling_zero_coordinates(self, index, interval=False):
        _stride_d = self.additional_attrs.get("stride_d")
        _kernel_d, _, _ = self.additional_attrs.get("kernels")
        _pad_head, _ = self.additional_attrs.get("depth_pad")
        _dedy_d = self.additional_attrs.get("dedy_d")

        if len(index) == 5:
            #(N，D, C1, H*W, C0)
            dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx = index
        elif len(index) == 4:
            #(N，D, H*W, C):
            dx_batch_idx, dx_deep_idx, dx_hw_idx, dx_cin_idx = index
        if interval:
            return tvm.any(
                (dx_deep_idx + _pad_head - (dx_deep_idx + _pad_head) // _stride_d * _stride_d - _kernel_d) >= 0,
                dx_deep_idx + _pad_head >= _stride_d * _dedy_d)
        else:
            return dx_deep_idx + _pad_head >= _stride_d * (_dedy_d - 1) + _kernel_d

    def _get_c0_c1_index(self):
        """
        get c0 c1 index according to format
        """
        c1_index = 2
        c0_index = -1
        return c0_index, c1_index

    def _get_input_dtype(self):
        """
        get fixpipe op input dtype (result type of mad1)
        """
        return self.x1.op.input_tensors[0].dtype

    def _get_output_shape(self):
        """
        get output shape
        """
        if in_dynamic():
            error_manager_cube.raise_err_specific("conv3d_backprop_input", "not support dynamic shape")
        else:
            out_shape = self._get_output_shape_static()
        if out_shape is None:
            error_manager_cube.raise_err_specific("conv3d_backprop_input", "error output shape or format")
        return out_shape

    def _get_output_shape_static(self):
        shape = self.output.get("shape")
        format_out = self.output.get("format")
        out_shape = None
        if len(shape) == LENGTH_FORMAT_NDC1HWC0 and format_out == "NDC1HWC0":
            out_shape = [shape[0], shape[1], shape[2], shape[3] * shape[4], shape[5]]
        elif len(shape) == LENGTH_FORMAT_NDHWC and format_out == "NDHWC":
            out_shape = [shape[0], shape[1], shape[2] * shape[3], shape[4]]
        return out_shape

    def _update_inputs(self):
        """
        locate the dx_ddr tensor
        """
        for x1_input_tensor in self.x1.op.input_tensors:
            if "dx_ddr" in x1_input_tensor.op.tag:
                self.x1 = x1_input_tensor

    def _x2_reform_generate_func_splitnd(self, x2, input_shape):
        dim_num = len(input_shape)
        add_tensor_d_dim = input_shape[1]
        x2_l1 = tvm.compute(input_shape,
                            lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx: x2[
                                dx_batch_idx * add_tensor_d_dim + dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx],
                            name="fixpipe_trans_eltwise",
                            tag="fixpipe_trans_eltwise")

        def lamda_func(*indice):
            new_indice = [0] * dim_num
            for i in range(dim_num):
                new_indice[i] = indice[i]

            return x2_l1(*new_indice)

        return lamda_func

    def _x2_reform_generate_func(self, x2, input_shape):
        """
        x2 index reform
        """
        return self._x2_reform_generate_func_splitnd(x2, input_shape)
