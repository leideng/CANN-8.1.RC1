#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
"""
from mindspore.ops import prim_attr_register
from mindspore.ops import PrimitiveWithInfer


class Square(PrimitiveWithInfer):
    """CusSquare definition"""
    from square_impl import square_impl

    @prim_attr_register
    def __init__(self):
        """init CusSquare"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, data_shape):
        """infer shape
        arguments:
        data_shape: the original data shape of Operator.
        returns: the data shape after converted.
        """
        return data_shape

    def infer_dtype(self, data_dtype):
        """infer shape
        arguments:
        data_dtype: the original data dtype of Operator.
        returns: the data dtype after converted.
        """
        return data_dtype
