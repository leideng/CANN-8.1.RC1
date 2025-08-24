#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021-2021 Huawei Technologies Co., Ltd
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
classifier of shape in transpose
"""
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation

from . import shape_classifier
from . import util


@shape_classifier.register_classifier(shape_classifier.TRANSPOSE)
def classify(ins: list, extra_params: dict):
    """
    classify
    :param ins: input list
    :param extra_params: include transpose list
    :return:
    """
    def mul(x, y):
        if 0 in (x, y):
            return 0
        if -1 in (x, y):
            return -1
        return x * y

    def maybe_pure_copy(shape):
        count = 0
        for value in shape:
            if value > 0:
                count += 1
        return count <= 1 and len(shape) > 1

    def remove_one_axis():
        _no_one_shape, _no_one_range, _no_one_index, _one_index = [], [], [], []
        for index, (_shape_v, _range) in enumerate(zip(shape_x, range_x)):
            if _shape_v != 1:
                _no_one_shape.append(_shape_v)
                _no_one_range.append(_range)
                _no_one_index.append(index)
            else:
                _one_index.append(index)

        if len(_no_one_shape) == 0:
            _no_one_shape = [shape_x[0]]
            _no_one_range = [range_x[0]]
            _no_one_index = [0]
            _one_index.remove(0)
        return [_no_one_shape, _no_one_range, _no_one_index, _one_index]

    def merge_axis():
        mergeable = [0]
        for i in range(len(no_one_perm) - 1):
            if no_one_perm.index(i) + 1 == no_one_perm.index(i + 1):
                mergeable.append(1)
            else:
                mergeable.append(0)

        new_shape = []
        new_range = []
        for i, merge in enumerate(mergeable):
            if merge == 1:
                new_shape[-1] = mul(new_shape[-1], no_one_shape[i])
                new_range[-1] = util.combine_range([new_range[-1], no_one_range[i]])
            else:
                new_shape.append(no_one_shape[i])
                new_range.append(no_one_range[i])

        unmerge_index = [i for i, merge in enumerate(mergeable) if not merge]
        new_permute = [unmerge_index.index(v) for v in no_one_perm if v in unmerge_index]

        for i in one_index:
            mergeable.insert(i, 2)
        operation.add_compile_info_inner("_mergeable", mergeable)
        return new_shape, new_range, new_permute

    def _update_shape_range(_shape, _range):
        new_shape, new_range = [], []
        for dim_value, dim_range in zip(_shape, _range):
            if dim_value > 0:
                dim_range = (dim_value, dim_value)
            elif dim_range[0] == dim_range[1] and not dim_range[0] is None and dim_range[0] > 0:
                dim_value = dim_range[0]
            new_shape.append(dim_value)
            new_range.append(dim_range)
        return new_shape, new_range

    if extra_params is None or "axes" not in extra_params:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "inputs of classify must include the dict extra_params with the key axes " \
                                      "when mode is transpose"
        raise RuntimeError(dict_args, get_error_message(dict_args))
    if len(ins) != 1:
        dict_args = {"errCode": "E90001", "detailed_cause": "input numbers error"}
        raise RuntimeError(dict_args, get_error_message(dict_args))
    permute = extra_params.get("axes")
    input_x = ins[0]
    shape_x = input_x.get("shape")
    range_x = input_x.get("range")
    shape_x, range_x = _update_shape_range(shape_x, range_x)

    no_one_shape, no_one_range, no_one_index, one_index = remove_one_axis()

    no_one_perm = [no_one_index.index(v) for v in permute if v in no_one_index]

    merge_shape, merge_range, merge_permute = merge_axis()

    new_input_x = {
        "shape": merge_shape,
        "range": merge_range,
        "dtype": input_x.get("dtype")
    }

    if maybe_pure_copy(merge_shape):
        copy_input_x = {
            "shape": (-1,),
            "range": ((1, None),),
            "dtype": input_x.get("dtype")
        }
        copy_permute = [0]
        return [[new_input_x, merge_permute], [copy_input_x, copy_permute]]
    return [[new_input_x, merge_permute]]
