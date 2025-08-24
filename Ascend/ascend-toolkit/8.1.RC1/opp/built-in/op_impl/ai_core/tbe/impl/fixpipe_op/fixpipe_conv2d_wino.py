#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2024 Huawei Technologies Co., Ltd
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
fixpipe process for conv2d winograd fusion
"""
from tbe import tvm
from impl.fixpipe_op.fixpipe_base import FixpipeBase
from impl.fixpipe_op import fixpipe_util

NC1P1MP2C0_C1_IDX = 1
NC1P1MP2C0_C0_IDX = 5
C0_16 = 16
C0_32 = 32
C0_64 = 64
FIXPIPE_INDEX = [0]
FIXPIPE_REFORM_INDEX = [0]
WINO_OUT_TILE_HW = 2


class FixpipeConv2dWino(FixpipeBase):
    """
    conv2d winograd Fixpipe
    """
    def fixpipe_op_compute(self):
        """
        main fixpipe compute
        default input format is NC1P1MP2C0
        """
        c0_index, c1_index = self._get_c0_c1_index()
        max_index = len(self.input_shape) - 1
        if c0_index > max_index or c1_index > max_index:
            raise RuntimeError("c0_index or c1_index is out of range")

        self.attrs["winograd_conv_flag"] = True

        fixpipe_op = tvm.compute(
            self.input_shape,
            lambda *indices:
            tvm.fixpipe_op(self._x1_reform_generate_func(self.x1)(*indices),
                self.output_dtype,
                pre_conv_param=self.get_input_vector_tensor(self.quant_scale_0,
                                                            fixpipe_util.QUANT_SCALE_0_STR)(*indices) \
                    if self.quant_scale_0_vector_flag else fixpipe_util.get_input_scalar_value(self.quant_scale_0),
                pre_relu_param=self.get_input_vector_tensor(self.relu_weight_0,
                                                            fixpipe_util.RELU_WEIGHT_0_STR)(*indices) \
                    if self.relu_weight_0_vector_flag else fixpipe_util.get_input_scalar_value(self.relu_weight_0),
                pre_clip_relu_param=fixpipe_util.get_input_scalar_value(self.clip_value_0),
                post_eltwise_src=self._x2_reform_generate_func(self.x2, self.input_shape)(*indices) \
                    if self.x2 is not None else self.x2,
                post_anti_quant_scale=fixpipe_util.get_input_scalar_value(self.anti_quant_scale),
                post_anti_quant_offset=fixpipe_util.get_conv_antiq_offset(self.anti_quant_offset, self.x2),
                post_clip_relu_param=fixpipe_util.get_input_scalar_value(self.clip_value_1),
                post_quant_param=self.get_input_vector_tensor(self.quant_scale_1,
                                                              fixpipe_util.QUANT_SCALE_1_STR)(*indices) \
                    if self.quant_scale_1_vector_flag else fixpipe_util.get_input_scalar_value(self.quant_scale_1),
                post_relu_param=self.get_input_vector_tensor(self.relu_weight_1,
                                                             fixpipe_util.RELU_WEIGHT_1_STR)(*indices) \
                    if self.relu_weight_1_vector_flag else fixpipe_util.get_input_scalar_value(self.relu_weight_1),
                op_dict=self.op_dict
            ),
            name=fixpipe_util.FIXPIPE_OP_TAG + "_" + str(FIXPIPE_INDEX[0]),
            tag=fixpipe_util.FIXPIPE_OP_TAG + "_" + str(FIXPIPE_INDEX[0]),
            attrs=self.attrs)
        FIXPIPE_INDEX[0] += 1
        return fixpipe_op

    def get_input_vector_tensor(self, vector_tensor, tensor_name):
        """
        get input vector tensor for quant_scale_0/1, relu_weight_0/1
        """
        _, c1, _, _, c0 = fixpipe_util.shape_to_list(vector_tensor.shape)
        vector_shape_fb = (1, WINO_OUT_TILE_HW, WINO_OUT_TILE_HW, c1, 1, 1, c0)
        vector_tensor_fb = tvm.compute(vector_shape_fb,
                                       lambda n_idx, p1_idx, p2_idx, c1_idx, h_idx, w_idx, c0_idx:
                                           vector_tensor(n_idx, c1_idx, h_idx, w_idx, c0_idx),
                                           name=tensor_name + "_" + str(FIXPIPE_INDEX[0]),
                                           tag=tensor_name + "_" + str(FIXPIPE_INDEX[0]))
        if "vector_fb_tensors" not in self.attrs:
            self.attrs["vector_fb_tensors"] = {}
        self.attrs["vector_fb_tensors"][tensor_name] = vector_tensor_fb

        def lambda_func(*indice):
            _, c1_idx, p1_idx, _, p2_idx, c0_idx = indice
            return vector_tensor_fb(0, p1_idx, p2_idx, c1_idx, 0, 0, c0_idx)

        return lambda_func

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        batch, co1, out_height, out_width, co0 = self.output_shape
        wino_res_ho = (out_height + 1) // WINO_OUT_TILE_HW
        wino_res_wo = (out_width + 1) // WINO_OUT_TILE_HW

        reform_shape = (batch, co1, WINO_OUT_TILE_HW, wino_res_ho * wino_res_wo, WINO_OUT_TILE_HW, co0)
        if self._is_channel_merge():
            res_reform = tvm.compute(reform_shape,
                                     lambda n_idx, co1_idx, p1_idx, hw_idx, p2_idx, co0_idx:
                                         res[n_idx, 
                                             (co1_idx * co0 + co0_idx) // self.input_shape[-1],
                                             p1_idx,
                                             hw_idx,
                                             p2_idx,
                                             (co1_idx * co0 + co0_idx) % self.input_shape[-1]],
                                         name="wino_channel_merge" + "_" + str(FIXPIPE_REFORM_INDEX[0]),
                                         tag=fixpipe_util.FIXPIPE_REFORM_TAG + "_" + str(FIXPIPE_REFORM_INDEX[0]),
                                         attrs=self.attrs)
            FIXPIPE_REFORM_INDEX[0] += 1
        else:
            res_reform = tvm.compute(reform_shape,
                                     lambda *indice: res(*indice),
                                     name="fixpipe_reform_default" + "_" + str(FIXPIPE_REFORM_INDEX[0]),
                                     tag=fixpipe_util.FIXPIPE_REFORM_TAG + "_" + str(FIXPIPE_REFORM_INDEX[0]),
                                     attrs=self.attrs)
            FIXPIPE_REFORM_INDEX[0] += 1

        cub_wino_post_shape = (batch, co1, wino_res_ho * WINO_OUT_TILE_HW, wino_res_wo * WINO_OUT_TILE_HW, co0)
        cub_wino_post = tvm.compute(cub_wino_post_shape,
                                    lambda n_idx, co1_idx, ho_idx, wo_idx, co0_idx:
                                        res_reform[n_idx,
                                                   co1_idx,
                                                   ho_idx % WINO_OUT_TILE_HW,
                                                   (ho_idx // WINO_OUT_TILE_HW) * wino_res_wo +
                                                   wo_idx // WINO_OUT_TILE_HW,
                                                   wo_idx % WINO_OUT_TILE_HW,
                                                   co0_idx],
                                    name=fixpipe_util.WINOGRAD_POST_TAG + "_" + str(FIXPIPE_INDEX[0]),
                                    tag=fixpipe_util.WINOGRAD_POST_TAG + "_" + str(FIXPIPE_INDEX[0]))
        conv_res_shape = (batch, co1, out_height * out_width, co0)
        conv_res = tvm.compute(conv_res_shape,
                               lambda n_idx, co1_idx, howo_idx, co0_idx:
                                   cub_wino_post[n_idx,
                                                 co1_idx,
                                                 howo_idx // out_width,
                                                 howo_idx % out_width,
                                                 co0_idx],
                               name=fixpipe_util.WINOGRAD_RES_TAG + "_" + str(FIXPIPE_INDEX[0]),
                               tag=fixpipe_util.WINOGRAD_RES_TAG + "_" + str(FIXPIPE_INDEX[0]))
        return conv_res

    def _get_post_transform(self):
        """
        get post transform for op_dict
        """
        return "WINO_POST"

    def _get_c0_c1_index(self):
        """
        get c0 c1 index according to format
        """
        return NC1P1MP2C0_C0_IDX, NC1P1MP2C0_C1_IDX

    def _get_input_dtype(self):
        """
        get fixpipe op input dtype (result type of mad1)
        """
        return self.x1.op.input_tensors[0].dtype

    def _x2_reform_generate_func_anti_quant(self, x2, input_shape):
        """
        x2 index reform with anti_qunt unit
        """
        dim_num = len(input_shape)
        c0_index, c1_index = self._get_c0_c1_index()

        x2_shape = fixpipe_util.shape_to_list(x2.shape)
        x2_c0 = x2_shape[-1]

        if x2_c0 not in [C0_32, C0_64]:
            raise RuntimeError("c0 of x2 should be 32 or 64")

        def lamda_func(*indice):
            new_indice = [0] * dim_num
            for i in range(dim_num):
                if i == c0_index:
                    new_indice[i] = (indice[c1_index] * C0_16 + indice[c0_index]) % x2_c0
                    continue

                if i == c1_index:
                    new_indice[i] = (indice[c1_index] * C0_16 + indice[c0_index]) // x2_c0
                    continue

                new_indice[i] = indice[i]

            return x2(*new_indice)

        return lamda_func

    def _x2_reform_generate_func_wino(self, x2, input_shape):
        """
        x2 reform in winograd scenario
        """
        _, _, _, out_width, _ = self.output_shape
        wino_res_wo = (out_width + 1) // WINO_OUT_TILE_HW
        batch, _, p1, hw_wino, p2, _ = input_shape
        _, x2_c1, _, x2_c0 = fixpipe_util.shape_to_list(x2.shape)
        x2_reform_shape = (batch, x2_c1, p1, hw_wino, p2, x2_c0)

        x2_reform = tvm.compute(x2_reform_shape,
                                lambda n_idx, co1_idx, p1_idx, hw_idx, p2_idx, co0_idx:
                                    x2[n_idx,
                                       co1_idx,
                                       (hw_idx // wino_res_wo) * WINO_OUT_TILE_HW * wino_res_wo * WINO_OUT_TILE_HW +
                                       p1_idx * wino_res_wo * WINO_OUT_TILE_HW +
                                       (hw_idx % wino_res_wo) * WINO_OUT_TILE_HW + p2_idx,
                                       co0_idx],
                                    name="x2_reform_l1" + "_" + str(FIXPIPE_INDEX[0]),
                                    tag="x2_reform_l1" + "_" + str(FIXPIPE_INDEX[0]))
        if "vector_tensors" not in self.attrs:
            raise RuntimeError("get vector_tensors from fixpipe attrs fail")
        self.attrs["vector_tensors"].remove(x2)
        self.attrs["vector_tensors"].append(x2_reform)
        return x2_reform

    def _x2_reform_generate_func(self, x2, input_shape):
        """
        x2 index reform
        """
        x2 = self._x2_reform_generate_func_wino(x2, input_shape)
        if self.anti_quant_scale is None:
            return self._x2_reform_generate_func_default(x2, input_shape)

        return self._x2_reform_generate_func_anti_quant(x2, input_shape)
