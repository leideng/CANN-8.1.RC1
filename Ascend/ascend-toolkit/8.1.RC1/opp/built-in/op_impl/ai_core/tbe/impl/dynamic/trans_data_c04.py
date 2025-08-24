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


# 'pylint: disable=too-many-locals,invalid-name
def trans_data_c04(infos, pad_value=0, kernel_name="trans_data_dsl"):
    """
    :param infos: [[src0, dst0], [src1, dst1], ....] that srcX is message of inputs, dstX is message of outputs.
    :param axes_map: transformer-regulation.
    :param pad_value: pack value, default 0.
    :param kernel_name: kernel_name.
    :return:
    """
    ins = classify(infos, OpPatternMode.TRANSDATA)
    tensors = []
    sch_list = []
    for (x, dst_shape, axes_map) in ins:
        with tbe.compute():
            src_shape, dst_shape = shape_util.variable_shape([x, dst_shape, axes_map], op_mode="transdatac04")
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
