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
fixpipe base functions
"""
from typing import List
from tbe import tvm
from tbe.tvm import Tensor
from tbe.common.utils import log
from tbe.common.utils import shape_to_list
from tbe.common.platform import get_cube_mkn
from impl.fixpipe_op import fixpipe_util
from tbe.common.utils.op_util import op_util_conv2d
from tbe.common import platform as tbe_platform

M_INDEX = 0
C_INDEX = 1
N_INDEX = 2


class FixpipeBase(object):
    """
    fixpipe base class
    """
    def __init__(self, op_type: str, x1: Tensor, x2: (Tensor, None), quant_scale_0: (Tensor, None),
                 relu_weight_0: (Tensor, None), clip_value_0: (Tensor, None),
                 quant_scale_1: (Tensor, None), relu_weight_1: (Tensor, None),
                 clip_value_1: (Tensor, None),
                 anti_quant_scale: (Tensor, None), anti_quant_offset: (Tensor, None),
                 output: dict, fusion_op_list: List[str], unit_list: List[str], eltwise_mode: str):
        """
        FixpipeBase init func
        """
        # set op input params
        self.op_type = op_type
        self.x1 = x1
        self.x2 = x2
        self.quant_scale_0 = quant_scale_0
        self.relu_weight_0 = relu_weight_0
        self.clip_value_0 = clip_value_0
        self.quant_scale_1 = quant_scale_1
        self.relu_weight_1 = relu_weight_1
        self.clip_value_1 = clip_value_1
        self.anti_quant_scale = anti_quant_scale
        self.anti_quant_offset = anti_quant_offset
        self.output = output
        self.fusion_op_list = fusion_op_list
        self.unit_list = unit_list
        self.eltwise_mode = eltwise_mode

        self.vector_inputs_dict = {
            fixpipe_util.QUANT_SCALE_0_STR: self.quant_scale_0,
            fixpipe_util.RELU_WEIGHT_0_STR: self.relu_weight_0,
            fixpipe_util.QUANT_SCALE_1_STR: self.quant_scale_1,
            fixpipe_util.RELU_WEIGHT_1_STR: self.relu_weight_1,
            fixpipe_util.ELTWISE_SRC_STR: self.x2
        }

        # set vector tendor flag
        self.quant_scale_0_vector_flag = fixpipe_util.is_vector_input(self.quant_scale_0)
        self.relu_weight_0_vector_flag = fixpipe_util.is_vector_input(self.relu_weight_0)
        self.quant_scale_1_vector_flag = fixpipe_util.is_vector_input(self.quant_scale_1)
        self.relu_weight_1_vector_flag = fixpipe_util.is_vector_input(self.relu_weight_1)

        self.attrs = {}
        self.op_dict = {}

        self.output_dtype = ""
        self.output_shape = []
        self.input_dtype = ""
        self.input_shape = []

    def fixpipe_op_compute(self):
        """
        main fixpipe compute
        default input format is NC1HWC0
        """
        c0_index, c1_index = self._get_c0_c1_index()
        max_index = len(self.input_shape) - 1
        if c0_index > max_index or c1_index > max_index:
            raise RuntimeError("c0_index or c1_index is out of range")

        fixpipe_op = tvm.compute(
            self.input_shape,
            lambda *indices:
            tvm.fixpipe_op(self._x1_reform_generate_func(self.x1)(*indices),
                self.output_dtype,
                pre_conv_param=self.quant_scale_0(0, indices[c1_index], 0, 0, indices[c0_index]) \
                    if self.quant_scale_0_vector_flag else self._get_const_value(),
                pre_relu_param=self.relu_weight_0(0, indices[c1_index], 0, 0, indices[c0_index]) \
                    if self.relu_weight_0_vector_flag else fixpipe_util.get_input_scalar_value(self.relu_weight_0),
                pre_clip_relu_param=fixpipe_util.get_input_scalar_value(self.clip_value_0),
                post_eltwise_src=self._x2_reform_generate_func(self.x2, self.input_shape)(*indices) \
                    if self.x2 is not None else self.x2,
                post_anti_quant_scale=fixpipe_util.get_input_scalar_value(self.anti_quant_scale),
                post_anti_quant_offset=fixpipe_util.get_input_scalar_value(self.anti_quant_offset),
                post_clip_relu_param=fixpipe_util.get_input_scalar_value(self.clip_value_1),
                post_quant_param=self.quant_scale_1(0, indices[c1_index], 0, 0, indices[c0_index]) \
                    if self.quant_scale_1_vector_flag else fixpipe_util.get_input_scalar_value(self.quant_scale_1),
                post_relu_param=self.relu_weight_1(0, indices[c1_index], 0, 0, indices[c0_index]) \
                    if self.relu_weight_1_vector_flag else fixpipe_util.get_input_scalar_value(self.relu_weight_1),
                op_dict=self.op_dict
            ),
            name=fixpipe_util.FIXPIPE_OP_TAG,
            tag=fixpipe_util.FIXPIPE_OP_TAG,
            attrs=self.attrs)
        return fixpipe_op

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        res_reform = tvm.compute(self.output_shape,
                                 lambda *indice: res(*indice),
                                 name=fixpipe_util.FIXPIPE_REFORM_TAG,
                                 tag=fixpipe_util.FIXPIPE_REFORM_TAG)
        return res_reform

    def fixpipe_compute(self):
        """
        fixpipe compute
        """
        self._update_inputs()
        self._get_params()
        self._param_check()
        self._update_flag_and_pre_conv()

        fixpipe_op = self.fixpipe_op_compute()
        fixpipe_reform = self.fixpipe_reform(fixpipe_op)

        return fixpipe_reform

    def _get_params(self):
        """
        get necessary info for fixpipe_op
        """
        self.input_dtype = self._get_input_dtype()
        self.output_dtype = self._get_output_dtype()
        self.output_shape = self._get_output_shape()
        self.input_shape = self._get_input_shape()
        self.op_dict = self._get_op_dict()
        self.attrs = self._get_attrs()

    def _get_output_shape(self):
        """
        get output shape
        """
        return self.output.get("shape")

    def _get_c0_c1_index(self):
        """
        get c0 c1 index
        """
        return fixpipe_util.NC1HWC0_C0_IDX, fixpipe_util.NC1HWC0_C1_IDX

    def _get_const_value(self):
        return fixpipe_util.get_input_scalar_value(self.quant_scale_0)

    def _get_input_shape(self):
        """
        get input tensor shape
        """
        return shape_to_list(self.x1.shape)

    def _get_output_dtype(self):
        """
        get output dtype
        """
        return self.output.get("dtype")

    def _get_input_dtype(self):
        return self.x1.dtype

    def _get_pre_conv(self):
        """
        get pre_conv for op_dict
        """
        def _get_pre_dst_dtype(quant_scale_1, output_dtype):
            dst_pre_conv_dtype_ = output_dtype
            if quant_scale_1 is not None:
                dst_pre_conv_dtype_ = fixpipe_util.DTYPE_FLOAT16
            return dst_pre_conv_dtype_

        conv_mode = ""
        if fixpipe_util.is_vector_input(self.quant_scale_0):
            conv_mode += "V"

        dst_pre_conv_dtype = _get_pre_dst_dtype(self.quant_scale_1, self.output_dtype)
        conv_mode += fixpipe_util.DTYPE_TRANS_MAP.get(self.input_dtype) + "2" + \
                     fixpipe_util.DTYPE_TRANS_MAP.get(dst_pre_conv_dtype)

        if conv_mode in fixpipe_util.PASS_PRE_CONVERT_MODE:
            return ""

        if conv_mode not in fixpipe_util.PRE_CONVERT_MODE:
            raise RuntimeError("{} is not supported for fixpipe pre_conv".format(conv_mode))

        return conv_mode

    def _get_pre_activation(self):
        """
        get pre_activation for op_dict
        """
        if self.relu_weight_0 is not None:
            if fixpipe_util.is_scaler_input(self.relu_weight_0):
                return fixpipe_util.SCALAR_RELU_MODE
            return fixpipe_util.VECTOR_RELU_MODE

        for lut_str in fixpipe_util.LUT_MODE_MAP.keys():
            if lut_str in self.fusion_op_list:
                return fixpipe_util.LUT_MODE_MAP.get(lut_str)

        if fixpipe_util.PRE_ACT_UNIT_STR in self.unit_list:
            if op_util_conv2d.support_conv_instruction():
                if fixpipe_util.RELU_STR in self.fusion_op_list:
                    return fixpipe_util.NORMAL_RELU_MODE
                return fixpipe_util.VECTOR_RELU_MODE
            return fixpipe_util.NORMAL_RELU_MODE

        return ""

    def _get_post_anti_quant(self):
        """
        get post_anti_quant for op_dict
        """
        if self.anti_quant_scale is None:
            return ""

        anti_quant_dtype = self.x2.dtype
        if anti_quant_dtype not in fixpipe_util.ANTI_QUANT_MAP.keys():
            raise RuntimeError("{} is not supported for fixpipe anti_quant".format(anti_quant_dtype))

        return fixpipe_util.ANTI_QUANT_MAP.get(anti_quant_dtype)

    def _get_post_eltwise(self):
        """
        get post_eltwise for op_dict
        """
        if self.x2 is None:
            if self.eltwise_mode != "":
                raise RuntimeError("eltwise_mode should be SUB or ADD when x2 is not None")

            return ""

        return self.eltwise_mode

    def _get_post_activation(self):
        """
        get post_activation for op_dict
        """
        if self.relu_weight_1 is not None:
            if fixpipe_util.is_scaler_input(self.relu_weight_1):
                return fixpipe_util.SCALAR_RELU_MODE
            return fixpipe_util.VECTOR_RELU_MODE

        if fixpipe_util.POST_ACT_UNIT_STR in self.unit_list:
            return fixpipe_util.NORMAL_RELU_MODE

        return ""

    def _get_post_quant(self):
        """
        get post_quant for op_dict
        """
        if self.quant_scale_1 is None:
            return ""

        conv_mode = ""
        if fixpipe_util.is_vector_input(self.quant_scale_1):
            conv_mode += "V"

        conv_mode += fixpipe_util.DTYPE_TRANS_MAP.get(fixpipe_util.DTYPE_FLOAT16) + "2" + \
                     fixpipe_util.DTYPE_TRANS_MAP.get(self.output_dtype)
        if conv_mode not in fixpipe_util.POST_QUANT_MODE:
            raise RuntimeError("{} is not supported for fixpipe post_quant".format(conv_mode))

        return conv_mode

    def _get_post_transform(self):
        """
        get post_transform for op_dict
        """
        if self.output.get("format") in ("NHWC", "ND", "NDHWC"):
            return "NZ2ND"
        return ""

    def _is_nz2nd(self):
        """
        check nz2nd scene
        """
        if self.output.get("format") in ("NHWC", "ND", "NDHWC"):
            return True
        return False

    def _is_channel_split(self):
        """
        check channel spilt scene
        """
        if self._is_nz2nd():
            return False

        block_n0 = get_cube_mkn(self.output_dtype)[N_INDEX]
        block_c0 = get_cube_mkn(self.output_dtype)[C_INDEX]

        return block_n0 % block_c0 == 0 and block_n0 != block_c0

    def _is_channel_merge(self):
        """
        check channel merge scene
        """
        if self._is_nz2nd():
            return False

        block_n0 = get_cube_mkn(self.output_dtype)[N_INDEX]
        block_c0 = get_cube_mkn(self.output_dtype)[C_INDEX]

        return block_c0 % block_n0 == 0 and block_n0 != block_c0

    def _get_vector_tensors(self):
        """
        get vector tensors from inputs
        """
        vector_params = []
        vector_tensors = []

        for input_name in self.vector_inputs_dict.keys():
            input_tensor = self.vector_inputs_dict.get(input_name)
            if input_name == fixpipe_util.ELTWISE_SRC_STR and input_tensor is not None:
                vector_params.append(input_name)
                vector_tensors.append(input_tensor)
                continue

            if fixpipe_util.is_vector_input(input_tensor, self.op_dict.get("pre_activation")):
                vector_params.append(input_name)
                vector_tensors.append(input_tensor)

        return vector_params, vector_tensors

    def _get_op_dict(self):
        """
        get op_dict for tvm.fixpipe_op
        """
        op_dict = {
            "pre_conv": self._get_pre_conv(),
            "pre_activation": self._get_pre_activation(),
            "post_anti_quant": self._get_post_anti_quant(),
            "post_eltwise": self._get_post_eltwise(),
            "post_activation": self._get_post_activation(),
            "post_quant": self._get_post_quant(),
            "post_transform": self._get_post_transform()
        }
        log.debug("fixpipe op_dict:{}".format(op_dict))
        return op_dict

    def _get_attrs(self):
        """
        get attrs for fixpipe
        """
        attrs = {}
        vector_params, vector_tensors = self._get_vector_tensors()
        attrs["vector_params"] = vector_params
        attrs["vector_tensors"] = vector_tensors
        attrs["nz2nd_flag"] = self._is_nz2nd()
        attrs["anti_quant_flag"] = True if self.anti_quant_scale is not None else False
        attrs["op_dict"] = self.op_dict
        log.debug("fixpipe attrs:{}".format(attrs))
        return attrs

    def _update_inputs(self):
        """
        update op input for special scenes
        """
        pass

    def _update_flag_and_pre_conv(self):
        """
        update quant_scale_0 flag and pre conv mode
        """
        self.quant_scale_0_vector_flag = fixpipe_util.is_vector_input(self.quant_scale_0,
                                                                      self.op_dict.get("pre_activation"))
        pre_conv_str = self.op_dict.get("pre_conv")
        if self.quant_scale_0_vector_flag and "V" not in pre_conv_str:
            self.op_dict["pre_conv"] = "V" + pre_conv_str

    def _param_check(self):
        """
        check op input params
        """
        pass

    def _x2_reform_generate_func_default(self, x2, input_shape):
        """
        x2 index reform default
        """
        dim_num = len(input_shape)

        def lamda_func(*indice):
            new_indice = [0] * dim_num
            for i in range(dim_num):
                new_indice[i] = indice[i]

            return x2(*new_indice)

        return lamda_func

    def _x2_reform_generate_func(self, x2, input_shape):
        """
        x2 index reform
        """
        return self._x2_reform_generate_func_default(x2, input_shape)

    def _x1_reform_generate_func(self, x1):
        """
        x1 index reform.
        The parent class returns x1, the subclass returns self define function.
        """
        return x1