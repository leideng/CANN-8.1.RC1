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
fixpipe process for conv3d fusion
"""
from impl.fixpipe_op.fixpipe_base import FixpipeBase
from impl.fixpipe_op import fixpipe_util
from tbe import tvm
from tbe.common.utils.errormgr import error_manager_cube
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import in_dynamic


LENGTH_FORMAT_NDC1HWC0 = 6
LENGTH_FORMAT_NDHWC = 5


class FixpipeConv3d(FixpipeBase):
    """
    Fixpipe for conv3d
    """
    def fixpipe_op_compute(self):
        """
        main fixpipe compute
        default input format is NC1HWC0
        """
        c0_index, c1_index = self._get_c0_c1_index()
        max_index = len(self.input_shape) - 1

        fixpipe_op = tvm.compute(self.input_shape,
                                 lambda *indices:
                                 self._get_fixpipe_op_compute(indices, c0_index, c1_index, max_index),
                                 name=fixpipe_util.FIXPIPE_OP_TAG,
                                 tag=fixpipe_util.FIXPIPE_OP_TAG,
                                 attrs=self.attrs)
        return fixpipe_op

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        fixpipe_reform_tag = "fixpipe_reform"
        self.attrs["6HD_TRANS_NDHWC"] = False
        if self._is_nz2nd():
            self.attrs["6HD_TRANS_NDHWC"] = True
            res_reform = tvm.compute(self.output_shape,
                                     lambda nd, hw, c: res(nd, c // self.input_shape[-1],
                                                           hw, c % self.input_shape[-1]),
                                     name=fixpipe_reform_tag,
                                     tag=fixpipe_reform_tag,
                                     attrs=self.attrs)
            return res_reform

        if self._is_channel_merge() or self._is_channel_split():
            name = "fixpipe_channel_merge"
            if self._is_channel_split():
                name = "fixpipe_channel_split"
            res_reform = tvm.compute(self.output_shape,
                                     lambda nd, c1, hw, c0:
                                     res(nd, (c1 * self.output_shape[-1] + c0) // self.input_shape[-1],
                                         hw,
                                         (c1 * self.output_shape[-1] + c0) % self.input_shape[-1]),
                                     name=name,
                                     tag=fixpipe_util.FIXPIPE_REFORM_TAG,
                                     attrs=self.attrs)
            return res_reform

        res_reform = tvm.compute(self.output_shape,
                                 lambda *indice: res(*indice),
                                 name=fixpipe_reform_tag,
                                 tag=fixpipe_reform_tag,
                                 attrs=self.attrs)
        return res_reform

    def _get_c0_c1_index(self):
        """
        get c0 c1 index according to format
        """
        c1_index = 1 # n and d fused into one axis, c1 = 1
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
            error_manager_cube.raise_err_specific("Conv3d", "not support dynamic shape")
        else:
            out_shape = self._get_output_shape_static()
        if out_shape is None:
            error_manager_cube.raise_err_specific("Conv3d", "error output shape or format")
        return out_shape

    def _get_output_shape_static(self):
        shape = self.output.get("shape")
        format_out = self.output.get("format")
        out_shape = None
        if len(shape) == LENGTH_FORMAT_NDC1HWC0 and format_out == "NDC1HWC0":
            out_shape = [shape[0] * shape[1], shape[2], shape[3] * shape[4], shape[5]]
        elif len(shape) == LENGTH_FORMAT_NDHWC and format_out == "NDHWC":
            out_shape = [shape[0] * shape[1], shape[2] * shape[3], shape[4]]
        return out_shape

    def _get_fixpipe_op_compute(self, indices, c0_index, c1_index, max_index):
        if c0_index > max_index or c1_index > max_index:
            error_manager_cube.raise_err_specific("FixpipeConv3d", "c0_index or c1_index is out of range")
        if self.quant_scale_0_vector_flag:
            pre_conv_param = self.quant_scale_0(0, 0, indices[c1_index], 0, 0, indices[c0_index])
        else:
            pre_conv_param = fixpipe_util.get_input_scalar_value(self.quant_scale_0)
        if self.relu_weight_0_vector_flag:
            pre_relu_param = self.relu_weight_0(0, 0, indices[c1_index], 0, 0, indices[c0_index])
        else:
            pre_relu_param = fixpipe_util.get_input_scalar_value(self.relu_weight_0)
        pre_clip_relu_param = fixpipe_util.get_input_scalar_value(self.clip_value_0)
        if self.x2 is not None:
            post_eltwise_src = self._x2_reform_generate_func(self.x2, self.input_shape)(*indices)
        else:
            post_eltwise_src = self.x2
        post_anti_quant_scale = fixpipe_util.get_input_scalar_value(self.anti_quant_scale)
        post_anti_quant_offset = fixpipe_util.get_input_scalar_value(self.anti_quant_offset)
        post_clip_relu_param = fixpipe_util.get_input_scalar_value(self.clip_value_1)
        if self.quant_scale_1_vector_flag:
            post_quant_param = self.quant_scale_1(0, 0, indices[c1_index], 0, 0, indices[c0_index])
        else:
            post_quant_param = fixpipe_util.get_input_scalar_value(self.quant_scale_1)
        if self.relu_weight_1_vector_flag:
            post_relu_param = self.relu_weight_1(0, 0, indices[c1_index], 0, 0, indices[c0_index])
        else:
            post_relu_param = fixpipe_util.get_input_scalar_value(self.relu_weight_1)

        return tvm.fixpipe_op(self._x1_reform_generate_func(self.x1)(*indices),
                              self.output_dtype,
                              pre_conv_param,
                              pre_relu_param,
                              pre_clip_relu_param,
                              post_eltwise_src,
                              post_anti_quant_scale,
                              post_anti_quant_offset,
                              post_clip_relu_param,
                              post_quant_param,
                              post_relu_param,
                              op_dict=self.op_dict)