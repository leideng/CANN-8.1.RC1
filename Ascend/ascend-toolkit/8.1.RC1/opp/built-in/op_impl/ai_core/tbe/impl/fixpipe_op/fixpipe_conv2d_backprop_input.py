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
fixpipe process for conv2d fusion
"""

from tbe import tvm
from impl.fixpipe_op.fixpipe_base import FixpipeBase
from tbe import tvm
from tbe.common.utils.errormgr import error_manager_cube
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import in_dynamic
from tbe.common.utils.op_util import op_util_conv2d

LENGTH_FORMAT_NC1HWC0 = 5
LENGTH_FORMAT_NHWC = 4
FIXPIPE_REFORM_INDEX = [0]


class FixpipeConv2dBackpropInput(FixpipeBase):
    """
    Fixpipe for conv2d dx
    """
    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        fixpipe_reform_tag = "fixpipe_reform"
        self.attrs["5HD_TRANS_NHWC"] = "False"
        if self._is_nz2nd():
            self.attrs["5HD_TRANS_NHWC"] = "True"
            res_reform = tvm.compute(self.output_shape,
                                     lambda n, hw, c: res(n, c // self.input_shape[-1],
                                                          hw, c % self.input_shape[-1]),
                                     name=fixpipe_reform_tag + "_" + str(FIXPIPE_REFORM_INDEX[0]),
                                     tag=fixpipe_reform_tag,
                                     attrs=self.attrs)
            FIXPIPE_REFORM_INDEX[0] += 1
            return res_reform

        if self._is_channel_merge() or self._is_channel_split():
            if op_util_conv2d.support_conv_instruction():
                res_reform = tvm.compute(self.output_shape,
                                         lambda n, c1, h, w, c0:
                                         res(n, (c1 * self.output_shape[-1] + c0) // self.input_shape[-1],
                                             h, w, (c1 * self.output_shape[-1] + c0) % self.input_shape[-1]),
                                         name=fixpipe_reform_tag,
                                         tag=fixpipe_reform_tag,
                                         attrs=self.attrs)
            else:
                res_reform = tvm.compute(self.output_shape,
                                         lambda n, c1, hw, c0:
                                         res(n, (c1 * self.output_shape[-1] + c0) // self.input_shape[-1],
                                             hw, (c1 * self.output_shape[-1] + c0) % self.input_shape[-1]),
                                         name=fixpipe_reform_tag + "_" + str(FIXPIPE_REFORM_INDEX[0]),
                                         tag=fixpipe_reform_tag,
                                         attrs=self.attrs)
            FIXPIPE_REFORM_INDEX[0] += 1
            return res_reform

        res_reform = tvm.compute(self.output_shape,
                                 lambda *indice: res(*indice),
                                 name=fixpipe_reform_tag + "_" + str(FIXPIPE_REFORM_INDEX[0]),
                                 tag=fixpipe_reform_tag,
                                 attrs=self.attrs)
        FIXPIPE_REFORM_INDEX[0] += 1
        return res_reform

    def _get_c0_c1_index(self):
        """
        get c0 c1 index according to format
        """
        NC1MC0_C1_IDX = 1
        NC1MC0_C0_IDX = -1
        return NC1MC0_C0_IDX, NC1MC0_C1_IDX

    def _get_input_dtype(self):
        """
        get fixpipe op input dtype (result type of mad1)
        """
        if op_util_conv2d.support_conv_instruction():
            return self.x1.dtype
        return self.x1.op.input_tensors[0].dtype

    def _get_output_shape(self):
        """
        get output shape
        """
        if in_dynamic() and not op_util_conv2d.support_conv_instruction():
            out_shape = self._get_output_shape_dynamic()
        else:
            out_shape = self._get_output_shape_static()
        if out_shape is None:
            error_manager_cube.raise_err_specific("Conv2dBackptopInput", "error output shape or format")
        return out_shape

    def _get_output_shape_dynamic(self):
        src_n, src_c1, src_hw, src_c0 = self.x1.shape
        real_c = get_te_var("dx_c").get_tvm_var() if isinstance(src_c1, tvm.Var) else self.output.get("shape")[-1]
        out_shape = None
        format_out = self.output.get("format")
        if format_out == "NC1HWC0":
            out_shape = [src_n, src_c1, src_hw, src_c0]
        elif format_out == "NHWC":
            out_shape = [src_n, src_hw, real_c]
        return  out_shape

    def _get_output_shape_static(self):
        shape = self.output.get("shape")
        format_out = self.output.get("format")
        out_shape = None
        if len(shape) == LENGTH_FORMAT_NC1HWC0 and format_out == "NC1HWC0":
            if op_util_conv2d.support_conv_instruction():
                out_shape = [shape[0], shape[1], shape[2], shape[3], shape[4]]
            else:
                out_shape = [shape[0], shape[1], shape[2] * shape[3], shape[4]]
        elif len(shape) == LENGTH_FORMAT_NHWC and format_out == "NHWC":
            out_shape = [shape[0], shape[1] * shape[2], shape[3]]
        return out_shape
