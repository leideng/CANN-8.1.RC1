#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

trans_data_dsl
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import operation


def _push(var, info_list):
    if var is not None:
        info_list.append(var)


def _set_compile_info(key, value):
    if value is not None:
        operation.add_compile_info_inner(key, value)


def _get_compile_info(key):
    return operation.get_compile_info().get(key)


def _del_compile_info(key):
    if key in operation.get_compile_info().keys():
        operation.get_compile_info().pop(key)


def _classify_c0(infos, axes_map):
    # Classify by c0 that maybe 8,16 in FP32
    ins, src_pad_list, data_move_pad_list, c0_list = [], [], [], []
    for src, dst, c0 in infos:
        ins.append(classify([src, dst, axes_map], OpPatternMode.TRANSDATA))
        _push(_get_compile_info("_data_move_src_pad_var"), data_move_pad_list)
        _push(_get_compile_info("_src_pad_var"), src_pad_list)
        c0_list.append(c0)

    # Update New CompileInfo
    _set_compile_info("_src_pad_list", src_pad_list)
    _set_compile_info("_data_move_pad_list", data_move_pad_list)
    _set_compile_info("_c0_list", c0_list)
    # Delete Old CompileInfo
    _del_compile_info("_src_pad_var")
    _del_compile_info("_data_move_src_pad_var")
    return ins, c0_list


def _add_context_info(context, c0):
    # convert compile message
    c0_list = operation.get_compile_info().get("_c0_list")
    src_pad_list = operation.get_compile_info().get("_src_pad_list")
    data_move_pad_list = operation.get_compile_info().get("_data_move_pad_list")
    src_pad_mode = operation.get_compile_info().get("_src_pad_mode")
    data_move_src_pad_mode = operation.get_compile_info().get("_data_move_src_pad_mode")
    remove_size_one_axis_src_pad_mode = operation.get_compile_info().get("_remove_size_one_axis_src_pad_mode")
    remove_size_one_axis_src_pad_var = operation.get_compile_info().get("_remove_size_one_axis_src_pad_var")

    if c0 not in c0_list:
        raise RuntimeError("C0 not in c0_list is illegal")
    context.add("_c0", c0)

    index = c0_list.index(c0)
    if src_pad_list:
        context.add("_src_pad_var", src_pad_list[index])
    if data_move_pad_list:
        context.add("_data_move_src_pad_var", data_move_pad_list[index])
    if src_pad_mode:
        context.add("_src_pad_mode", src_pad_mode)
    if data_move_src_pad_mode:
        context.add("_data_move_src_pad_mode", data_move_src_pad_mode)
    if remove_size_one_axis_src_pad_mode:
        context.add("_remove_size_one_axis_src_pad_mode", remove_size_one_axis_src_pad_mode)
    if remove_size_one_axis_src_pad_var:
        context.add("_remove_size_one_axis_src_pad_var", remove_size_one_axis_src_pad_var)


# 'pylint: disable=too-many-locals,invalid-name
def trans_data_dsl(infos, axes_map, pad_value=0, kernel_name="trans_data_dsl"):
    """
    :param infos: [[src0, dst0], [src1, dst1], ....] that srcX is message of inputs, dstX is message of outputs.
    :param axes_map: transformer-regulation.
    :param pad_value: pack value, default 0.
    :param kernel_name: kernel_name.
    :return:
    """
    sch_list, tensors = [], []
    ins, c0_list = _classify_c0(infos, axes_map)

    for _ins, c0 in zip(ins, c0_list):
        for (x, dst_shape, axes_map) in _ins:
            with tbe.compute():
                _add_context_info(operation.get_context().get_current_compute(), c0)
                src_shape, dst_shape = shape_util.variable_shape([x, dst_shape, axes_map], op_mode="transdata")
                data_input = tvm.placeholder(src_shape, name="data_input", dtype=x.get("dtype"))
                res = tbe.transdata(data_input, dst_shape, axes_map, pad_value)
                tensor_list = [data_input, res]
                tensors.append(tensor_list)
                with tvm.target.cce():
                    sch = tbe.auto_schedule(res)
            sch_list.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors,
              "build_args": {"constant_realize_extent_in_infer_bound": False,
                             "enable_dma_optimizer": False},
              }
    tbe.build(sch_list, config)
