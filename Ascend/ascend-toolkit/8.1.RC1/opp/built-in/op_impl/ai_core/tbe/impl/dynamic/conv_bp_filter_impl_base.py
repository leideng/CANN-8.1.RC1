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
conv2d/conv3d backprop filter base class
"""
import copy

from impl.util import util_select_op_base
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from tbe.common.utils import log
from tbe.common.utils.conv_util import CubeConstantConfig
from tbe.dsl.base.operation import get_op_context
from tbe.dsl.base.operation import is_unify
from tbe.dsl.classifier.conv2d_bp_filter_classifier import Conv2dBpFilterClassifier
from tbe.dsl.classifier.conv_bp_filter_base import ConvBpFilterBase
from tbe.dsl.unify_schedule.conv2d_bp_filter_tilingcase import ATTR_VARS
from tbe.dsl.unify_schedule.conv2d_bp_filter_tilingcase import SHAPE_VARS
from tbe.dsl.unify_schedule.conv2d_bp_filter_tilingcase import SPECIAL_SCENE_VAR
from tbe.dsl.unify_schedule.conv2d_bp_filter_tilingcase import TILING_VARS


class ConvBpFilterImplBase(ConvBpFilterBase):

    def __init__(self, inputs_list, op_name, fusion_mode=False, options=None) -> None:
        super().__init__(inputs_list, op_name, fusion_mode, options)
        self.var_map = {}
        self.binary_flag = False

    @staticmethod
    def gen_conv_default_shape_range(ori_format, ori_shape):
        """
        change ori_shape and ori_range to enter binary mode
        """
        shape_len = len(ori_shape)
        if list(ori_shape) == CubeConstantConfig.DYNAMIC_RANK_SHAPE:
            shape_len = CubeConstantConfig.FROMAT_TO_FIX_DIMS.get(ori_format)
        new_shape = list([-1 for _ in range(shape_len)])
        new_range = list([[1, -1] for _ in range(shape_len)])
        return new_shape, new_range

    def need_exchange_hw(self):
        return False

    def exchange_hw(self):
        pass

    def check_inputs_logic(self):
        pass

    def save_input_info(self):
        x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name = self.inputs_list
        attrs = {
            "strides": strides,
            "pads": pads,
            "dilations": dilations,
            "groups": groups,
            "data_format": data_format,
            "kernel_name": kernel_name
        }
        # change attr to input tensor for filter_size
        filter_size_len = len(filter_size)
        filter_size = {
            "shape": [filter_size_len],
            "ori_shape": [filter_size_len],
            'dtype': "int32",
            "format": data_format,
            "ori_format": data_format,
            "const_value": filter_size
        }
        context = get_op_context()
        context.add_addition("x", copy.deepcopy(x))
        context.add_addition("filter_size", copy.deepcopy(filter_size))
        context.add_addition("out_backprop", copy.deepcopy(out_backprop))
        context.add_addition("y", copy.deepcopy(y))
        context.add_addition("attrs", copy.deepcopy(attrs))

    def do_classify(self):
        x, filter_size, out_backprop, y, strides, pads, dilations, groups, data_format, kernel_name = self.inputs_list
        input_list = [x, filter_size, out_backprop, y]
        attr_list = [strides, pads, dilations, groups, data_format, kernel_name]
        option_list = []
        extra_params = {}
        ins = (input_list, attr_list, option_list)
        classifier = Conv2dBpFilterClassifier(ins, self.op_name, self.fusion_mode, extra_params)
        return classifier.classify()

    def define_vars(self):
        '''
        variablization
        '''

        def _define_optional_vars(var_name):
            if operation.get_te_var(var_name) is None:
                return operation.var(var_name)
            return operation.get_te_var(var_name).get_tvm_var()

        shape_var_map = {}
        attr_var_map = {}
        tiling_var_map = {}
        special_scene_var_map = {}

        for var in SHAPE_VARS:
            shape_var_map[var] = _define_optional_vars(var)

        for var in ATTR_VARS:
            attr_var_map[var] = operation.var(var)

        for var in TILING_VARS:
            tiling_var_map[var] = operation.var(var)

        for var in SPECIAL_SCENE_VAR:
            special_scene_var_map[var] = operation.var(var)

        c0_size = tbe_platform.CUBE_MKN.get(self.fm.dtype).get("mac")[1]
        var_shape_map = {}
        var_shape_map["fmap_nchw"] = (shape_var_map.get("batch"), shape_var_map.get("fmap_c"),
                                      shape_var_map.get("fmap_h"), shape_var_map.get("fmap_w"))
        var_shape_map["dedy_nchw"] = (shape_var_map.get("batch"), shape_var_map.get("dedy_c"),
                                      shape_var_map.get("dedy_h"), shape_var_map.get("dedy_w"))
        var_shape_map["dedw_nchw"] = (shape_var_map.get("dedy_c"), shape_var_map.get("fmap_c"),
                                      attr_var_map.get("kernel_h"), attr_var_map.get("kernel_w"))
        var_shape_map["fmap_nc1hwc0"] = (shape_var_map.get("batch"), attr_var_map.get("fmap_c1"),
                                         shape_var_map.get("fmap_h"), shape_var_map.get("fmap_w"), c0_size)
        var_shape_map["dedy_nc1hwc0"] = (shape_var_map.get("batch"), attr_var_map.get("dedy_c1"),
                                         shape_var_map.get("dedy_h"), shape_var_map.get("dedy_w"), c0_size)
        var_shape_map["strides"] = (attr_var_map.get("stride_h"), attr_var_map.get("stride_w"))
        var_shape_map["pads"] = (attr_var_map.get("padt"), attr_var_map.get("padb"), attr_var_map.get("padl"),
                                 attr_var_map.get("padr"))
        var_shape_map["dilations"] = (1, 1, attr_var_map.get("dilation_h"), attr_var_map.get("dilation_w"))
        var_shape_map["groups"] = attr_var_map.get("groups")

        self.var_map.update(var_shape_map)

    def new_placeholder(self):
        fmap_shape = self.var_map.get("fmap_nc1hwc0")
        dedy_shape = self.var_map.get("dedy_nc1hwc0")

        fmap = tvm.placeholder(fmap_shape, name="fmap", dtype=self.fm.dtype)
        filter_size = tvm.placeholder(self.filter_size.shape, name="filter_size", dtype=self.filter_size.dtype)
        dedy = tvm.placeholder(dedy_shape, name="dedy", dtype=self.grads.dtype)

        if get_op_context().get_addition("is_dynamic_constantization") is True or not is_unify():
            return [fmap, dedy]
        return [fmap, filter_size, dedy]

    def do_compute(self, tensor_list, options):
        if len(tensor_list) == 2:
            fmap_tensor, dedy_tensor = tensor_list
        else:
            fmap_tensor, _, dedy_tensor = tensor_list

        fmap_format_in_gm = self.fm.ori_format if self.fusion_mode else self.fm.format
        para_dict = {
            "strides": self.var_map.get("strides"),
            "padding": self.var_map.get("pads"),
            "dilations": self.var_map.get("dilations"),
            "groups": self.var_map.get("groups"),
            "data_format": self.data_format,
            "fmap_format_in_gm": fmap_format_in_gm,
            "kernel_name": self.kernel_name,
            "res_dtype": self.kernel.dtype,
            "binary_flag": self.binary_flag,
            "ori_tensors": {
                "x": self.fm.ori_info,
                "filter_size": self.filter_size.ori_info,
                "out_backprop": self.grads.ori_info,
                "y": self.kernel.ori_info
            }
        }
        if options:
            para_dict.update(options)
            log.debug("[ComputeTemplate] {}".format(options.get("compute_template").get_debug_info()))
        log.debug("[{}] compute param para_dict: {}".format(self.op_name, para_dict))
        return tbe.conv2d_backprop_filter(input_x=fmap_tensor,
                                          out_backprop=dedy_tensor,
                                          filter_sizes=self.var_map.get("dedw_nchw"),
                                          para_dict=para_dict)

    def do_build(self, tensor_list, sch_list):
        config = {'print_ir': False, 'name': self.kernel_name, 'tensor_list': tensor_list}
        build_args = {
            'constant_realize_extent_in_infer_bound': False,
            'predicate_total_out_of_bound': False,
            'enable_db_fold': True
        }
        if is_unify():
            config["build_args"] = build_args

        if get_op_context().get_addition("is_dynamic_constantization") is True:
            get_op_context().set_op_mode("static")

        log.debug("[{}] build start. kernel_name = {}".format(self.op_name, self.kernel_name))
        tbe.build(sch_list, config)
        log.debug("[{}] build end. kernel_name = {}".format(self.op_name, self.kernel_name))

    def get_op_split_info(self):
        """
        get the conv2d_backprop_filter split info
        """
        axis_split_matrix = None
        axis_reduce_list = None
        if self.fm.format == "NC1HWC0":
            # only Cout1 can be cut without overlap
            axis_split_matrix = [[
                util_select_op_base.SplitInput([1, [1], [-1], [-1]]),
                util_select_op_base.SplitOutput([0, [1]])
            ]]
            axis_reduce_list = [[
                util_select_op_base.ReduceInput([0, [0]], [1, [0]]),
                util_select_op_base.ReduceOutput([0, 1, False])
            ]]
        return axis_split_matrix, axis_reduce_list

    def select_format(self):
        pass
