#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
fixpipe fusion with matmul
"""
from functools import reduce

from impl.fixpipe_op.fixpipe_base import FixpipeBase
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.util_common import ceil
from impl.fixpipe_op import fixpipe_util
from tbe import tvm
from tbe.common.utils import shape_to_list
from tbe.common.platform import platform_info
from tbe.dsl.base.operation import in_dynamic


class FixpipeMatmul(FixpipeBase):
    """
    matmul Fixpipe
    """
    batch_nd_len = 3
    batch_frac_len = 5

    @staticmethod
    def _x_reform_generate_func_broadcast(x, input_shape):
        """
        Consider broadcasting x.shape to input_shape and get x index.

        Paremeters:
        -----------------------
        x: input tensor.
        input_shape: tuple or list, boradcast shape
        """
        x_shape = shape_to_list(x.shape)
        dim_num = len(x_shape)

        def lamda_func(*indice):
            new_indice = [0] * dim_num
            for i in range(dim_num):
                if x_shape[-i - 1] == input_shape[-i - 1]:
                    new_indice[-i - 1] = indice[-i - 1]

            return x(*new_indice)
        return lamda_func

    @staticmethod
    def _x_reform_generate_func_broadcast_v2(self, x, input_shape):
        """
        Consider broadcasting x.shape to input_shape and get x index.

        Paremeters:
        -----------------------
        x: input tensor.
        input_shape: tuple or list, boradcast shape
        """
        x_shape = shape_to_list(x.shape)
        dim_num = len(x_shape)
        ori_shape = x.op.attrs["ori_shape"]
        data_format = x.op.attrs["format"]
        x1_batch_shape = self.x1.op.attrs["batch_shape"]

        def lamda_func(*indice):
            new_indice = [0] * dim_num
            for i in range(dim_num):
                if x_shape[-i - 1] == input_shape[-i - 1]:
                    new_indice[-i - 1] = indice[-i - 1]
            if data_format == "FRACTAL_NZ" and (len(x1_batch_shape) > 0):
                new_indice[0] = 0
                reduce_indice = 1
                reduce_new_indice = 1
                for i in reversed(range(len(x1_batch_shape))):
                    new_indice[0] = new_indice[0] + \
                        ((indice[0] // reduce_indice) % x1_batch_shape[i]) % ori_shape[i] * reduce_new_indice
                    reduce_indice = reduce_indice * x1_batch_shape[i]
                    reduce_new_indice = reduce_new_indice * ori_shape[i]

            return x(*new_indice)
        return lamda_func

    def get_bm_fusion_shape(self):
        res_reshape_shape = self.output.get("shape")
        if self.x1.op.attrs.get("bm_fusion_flag") == True:
            batch_prod = reduce(lambda x, y:x * y, res_reshape_shape[:-4])
            res_reshape_shape = [batch_prod, *res_reshape_shape[-4:]]
        return res_reshape_shape

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        def bm_fusion_reshape_compute(res_reform, res_shape):
            res_fixpipe = tvm.compute(res_shape,
                                     lambda *indices: res_reform(0, indices[-4],
                                        indices[-5] * self.output.get("shape")[-3] + indices[-3], *indices[-2:]),
                                     name=fixpipe_name + "_res",
                                     tag=fixpipe_tag,
                                     attrs=attrs_dict)
            return res_fixpipe

        res_reshape_shape = self.get_bm_fusion_shape()
        fixpipe_name = "fixpipe"
        fixpipe_tag = "fixpipe_reform"
        format_out = "ND" if self._check_fc_nd_out() or self._is_nz2nd() else self.output.get("format")
        attrs_dict = {
            "format": format_out,
            "batch_shape": self.x1.op.attrs["batch_shape"],
            "ori_shape": self.output.get("ori_shape"),
            "shape": self.output.get("shape"),
            "ori_format": self.output.get("ori_format")
        }
        if self._is_nz2nd():
            fixpipe_name += "_nz2nd"
            m_block = self.input_shape[-1]
            n_block = self.input_shape[-2]
            res_reform = tvm.compute(self.output_shape,
                                     lambda *indices: res(*indices[:-2], indices[-1] // m_block,
                                                          indices[-2] // n_block, indices[-2] % n_block,
                                                          indices[-1] % m_block),
                                     name=fixpipe_name,
                                     tag=fixpipe_tag,
                                     attrs=attrs_dict)
        elif self._is_channel_merge() or self._is_channel_split():
            fixpipe_name += "_channel_merge_split"
            output_n_block = self.output_shape[-1]
            input_n_block = self.input_shape[-1]
            res_reform = tvm.compute(self.output_shape,
                                     lambda *indices:
                                     res(*indices[:-4],
                                         (indices[-4] * output_n_block + indices[-1]) // input_n_block,
                                         indices[-3], indices[-2],
                                         (indices[-4] * output_n_block + indices[-1]) % input_n_block),
                                     name=fixpipe_name,
                                     tag=fixpipe_tag,
                                     attrs=attrs_dict)
        else:
            if self.x1.op.attrs.get("bm_fusion_flag") == True:
                res = bm_fusion_reshape_compute(res, res_reshape_shape)
                self.output_shape = res_reshape_shape
            fixpipe_name += "_out"
            res_reform = tvm.compute(self.output_shape,
                                     lambda *indice: res(*indice),
                                     name=fixpipe_name,
                                     tag=fixpipe_tag,
                                     attrs=attrs_dict)
            return res_reform
        if self.x1.op.attrs.get("bm_fusion_flag") == True:
            return bm_fusion_reshape_compute(res_reform, res_reshape_shape)
        return res_reform


    def _check_fc_nd_out(self):
        """
        when op type is fc,then fc_flag is True
        """
        is_fc = False
        if self.x1.op.attrs and "fc_flag" in self.x1.op.attrs:
            is_fc = self.x1.op.attrs["fc_flag"]
        out_format = self.output.get("format")
        return is_fc and out_format in ["NHWC", "NC1HWC0", "NCHW"]

    def _get_post_transform(self):
        """
        get post_transform for op_dict

        Returns
        -------
        string
        """
        if self._is_nz2nd():
            return "NZ2ND"
        return ""

    def _get_output_shape(self):
        """
        get output shape
        """
        shape = self.output.get("shape")
        out_shape = shape
        if self._check_fc_nd_out():
            out_shape = (shape[0], reduce(lambda x, y: x * y, shape[1:]))
        elif len(shape) > self.batch_frac_len and self.output.get("format") == "FRACTAL_NZ":
            out_shape = [reduce(lambda x, y: x * y, shape[:-4])] + list(shape[-4:])
        elif len(shape) > self.batch_nd_len and self.output.get("format") != "FRACTAL_NZ":
            out_shape = [reduce(lambda x, y: x * y, shape[:-2])] + list(shape[-2:])
        if len(shape) == 1:
            reduce_axis = [1, 1] if len(self.input_shape) in (self.batch_frac_len, self.batch_nd_len) else [1]
            out_shape = reduce_axis + [shape[0]]
        if self.x1.op.attrs.get("bm_fusion_flag", False):
            # reshape to [1, n1, m1 * batch, m0, n0]
            out_shape = (1, shape[-4], reduce(lambda x, y: x * y, shape[:-4]) * shape[-3], *shape[-2:])
        return out_shape

    def _get_c0_c1_index(self):
        """
        get c0 c1 index according to format
        """
        nz_c0_idx = -1
        nz_c1_idx = -4
        return nz_c0_idx, nz_c1_idx

    def _update_inputs(self):
        """
        skip matmul ddr tensor
        """
        while self.x1.op.name not in  ["tensor_mmad", "matmul_op"]:
            self.x1 = self.x1.op.input_tensors[0]

    def _is_nz2nd(self):
        """
        check nz2nd scene

        Returns
        -------
        bool
        """
        return self.output.get("format") in ("NHWC", "ND", "NC1HWC0", "NCHW")

    def _get_const_value(self):
        """
        get input const value
        """
        if not in_dynamic() or platform_info.intrinsic_check_support("Intrinsic_matmul_ub_to_ub"):
            return fixpipe_util.get_input_scalar_value(self.quant_scale_0)
        return self.quant_scale_0(0, 0, 0, 0, 0)

    def _get_input_shape(self):
        """
        get input tensor shape
        """
        shape_x1 = shape_to_list(self.x1.shape)
        if self.x2 is None or self._check_fc_nd_out():
            return shape_x1
        else:
            shape_x2 = shape_to_list(self.x2.shape)
            try:
                shape_x1[-4] = ceil(self.output_shape[-1], shape_x1[-1]) if self._is_nz2nd() else self.output_shape[-4]
                shape_x1[-3] = ceil(self.output_shape[-2], shape_x1[-2]) if self._is_nz2nd() else self.output_shape[-3]
                _, _, shape_max = shape_util.broadcast_shapes(shape_x1, shape_x2, param_name_input1="input_x1",
                                                              param_name_input2="shape_x2")
                return shape_max
            except RuntimeError:
                return shape_x1

    def _x2_reform_generate_func(self, x2, input_shape):
        """
        x2 index reform

        Parameters
        ----------
        x2 : tensor
            elewise input
        input_shape : tuple or list
            shape of x1

        Returns
        -------
        lambda description
            new description for elewise input
        """
        if not self._check_fc_nd_out():
            return self._x_reform_generate_func_broadcast_v2(self, x2, input_shape)
        # (N,C1,H,W,C0) -> (C1HW,N1,N0,C0)
        x2_n, x2_c1, x2_h, x2_w, x2_c0 = shape_util.shape_to_list(x2.shape)
        x2_l1_shape = (x2_c1 * x2_h * x2_w,
                       ceil(x2_n, tbe_platform.BLOCK_IN),
                       tbe_platform.BLOCK_IN,
                       x2_c0)
        x2_l1 = tvm.compute(
            x2_l1_shape,
            lambda * indice: tvm.select(
                tvm.all(indice[-3] * tbe_platform.BLOCK_IN + indice[-2] < x2_n),
                x2(indice[-3] * tbe_platform.BLOCK_IN + indice[-2],
                   indice[-4] // (x2_h * x2_w),
                   indice[-4] // x2_w % x2_h,
                   indice[-4] % x2_w,
                   indice[-1])
            ),
            name="fixpipe_trans_eltwise"
        )
        return self._x_reform_generate_func_broadcast(x2_l1, input_shape)

    def _is_channel_split(self):
        if "ops_data_flow_mode" not in self.x1.op.attrs:
            return super()._is_channel_split()
        ops_data_flow_mode = self.x1.op.attrs["ops_data_flow_mode"]
        return super()._is_channel_split() and ops_data_flow_mode == "fp322fp32"
