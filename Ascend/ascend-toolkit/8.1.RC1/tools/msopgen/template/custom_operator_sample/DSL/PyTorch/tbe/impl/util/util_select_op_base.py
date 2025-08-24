# Copyright 2019 Huawei Technologies Co., Ltd
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
util_select_op_base
"""
import json


def get_op_cal_info(axis_split_list, axis_reduce_list=None, l1FusionEnable=0, minTbeL1Space=0):
    """
    Function
    --------
    generate the op split info by split info matrix
    """
    op_cal_info = {}
    op_slice_info = {}
    def _get_axis_split_in_json(axis_split_list):
        splitMaps = []
        if axis_split_list is None:
            return splitMaps

        split_info = _gen_multi_split_list(axis_split_list)
        for item in split_info:
            splitMaps.append(item)
        return splitMaps

    def _get_axis_reduce_in_json(axis_reduce_list):
        reduceMaps = []
        if axis_reduce_list is None:
            return reduceMaps

        reduce_info = _gen_multi_reduce_list(axis_reduce_list)
        for item in reduce_info:
            reduceMaps.append(item)
        return reduceMaps

    op_slice_info["splitMaps"] = _get_axis_split_in_json(axis_split_list)
    op_slice_info["reduceMaps"] = _get_axis_reduce_in_json(axis_reduce_list)
    op_slice_info["l1FusionEnable"] = l1FusionEnable
    op_slice_info["minTbeL1Space"] = minTbeL1Space
    op_cal_info["_op_slice_info"] = op_slice_info
    op_cal_info_in_json = json.dumps(op_cal_info)
    return op_cal_info_in_json


def _gen_multi_split_list(axis_split_list):
    """
    generate final list of op split info by multi split info
    """
    split_list_res = []
    split_info_num = len(axis_split_list)

    for k in range(split_info_num):
        input_list = axis_split_list[k][0]
        output_list = axis_split_list[k][1]
        split_list = {}
        info_res = []

        for i in range(input_list.input_split_num):
            info_temp = _split_multi_input_list(input_list.input_array[i])
            info_res.append(info_temp)

        split_list["inputList"] = info_res
        info_res = []
        for j in range(output_list.output_split_num):
            info_temp = _split_multi_output_list(output_list.output_array[j])
            info_res.append(info_temp)
        split_list["outputList"] = info_res

        split_list_res.append(split_list)

    return split_list_res


def _gen_multi_reduce_list(axis_reduce_list):
    """
    generate final list of op reduce info by multi reduce info
    """
    reduce_list_res = []
    reduce_info_num = len(axis_reduce_list)
    for k in range(reduce_info_num):
        input_list = axis_reduce_list[k][0]
        output_list = axis_reduce_list[k][1]
        reduce_list = {}
        info_res = []
        for i in range(input_list.reduce_input_num):
            info_temp = _multi_reduce_input_list(input_list.reduce_input[i])
            info_res.append(info_temp)
        reduce_list["inputList"] = info_res

        info_res = []
        for j in range(output_list.reduce_output_num):
            info_temp = _multi_reduce_output_list(output_list.reduce_output[j])
            info_res.append(info_temp)
        reduce_list["outputList"] = info_res

        reduce_list_res.append(reduce_list)

    return reduce_list_res


def _split_multi_input_list(one_input_list):
    """
    Function
    --------
    generate input info of input tensors by multi split info

    Parameters
    ----------
    idx: num
        input tensor index,the tensor has reduce axis
    axis: num
        the axis could be sliced
    headOverLap:
        the headOverLap value when cut the axis
    tailOverLap:
        the tailOverLap value when cut the axis

    Return
    ------
    dict
    """

    return {"idx":one_input_list[0],"axis":one_input_list[1],"headOverLap":one_input_list[2], \
            "tailOverLap":one_input_list[3]}


def _split_multi_output_list(one_output_list):
    """
    Function
    --------
    generate output info of output tensors by multi split info

    Parameters
    ----------
    idx: num
        output tensor index
    axis: num
        the axis could be sliced

    Return
    ------
    dict
    """
    return {"idx":one_output_list[0],"axis":one_output_list[1]}


def _multi_reduce_input_list(one_input_list):
    """
    Function
    --------
    generate input info of input tensors by multi reduce info

    Parameters
    ----------
    idx: num
        input tensor index,the tensor has reduce axis
    axis: num
        the axis be sliced

    Return
    ------
    dict
    """
    return {"idx":one_input_list[0],"axis":one_input_list[1]}


def _multi_reduce_output_list(one_output_list):
    """
    Function
    --------
    generate output info of output tensors by multi reduce info

    Parameters
    ----------
    idx: num
        oputput tensor index,the tensor has reduce axis
    reduceType: str
        only surpport  REDUCE_MEAN, REDUCE_ADD, REDUCE_MAX, REDUCE_MIN
    isAtomic: bool
        if true,compute with atomic add

    Return
    ------
    dict
    """
    return {"idx":one_output_list[0],"reduceType":one_output_list[1],"isAtomic":one_output_list[2]}


def get_dynamic_param_in_json(param_desc_list):
    param_dynamic = {}
    for item in param_desc_list:
        param_dict = {}
        param_dict["name"] = item.element.name
        param_dict["dtype"] = item.element.datatype
        if item.element.format is not None:
            param_dict["format"] = item.element.format
        if item.element.unknownshape_format is not None:
            param_dict["unknownshape_format"] = \
                item.element.unknownshape_format
        param_dynamic[item.classify] = param_dict
    param_dynamic_in_json = json.dumps(param_dynamic, indent=4)
    return param_dynamic_in_json


# pylint: disable=locally-disabled,redefined-builtin
def gen_param(classify, name, datatype, format, unknownshape_format=None):
    return ParamItem(classify=classify,
                     element=Element(name=name,
                                     datatype=datatype,
                                     format=format,
                                     unknownshape_format=unknownshape_format))


class Element:
    def __init__(self, name, datatype, format, unknownshape_format):
        self.name = name
        self.datatype = datatype
        self.format = format
        self.unknownshape_format = unknownshape_format

class ParamItem:
    def __init__(self, classify, element):
        self.classify = classify
        self.element = element

class SplitInput:
    def __init__(self, *input_array):
        self.input_array = input_array
        self.input_split_num = len(input_array)

class SplitOutput:
    def __init__(self, *output_array):
        self.output_array = output_array
        self.output_split_num = len(output_array)

class ReduceInput:
    def __init__(self, *input_array):
        self.reduce_input = input_array
        self.reduce_input_num = len(input_array)

class ReduceOutput:
    def __init__(self, *output_array):
        self.reduce_output = output_array
        self.reduce_output_num = len(output_array)


def get_split_n_info(inputs, outputs):
    """
    only support split axis N
    """
    split_inputs = []
    for i in inputs:
        split_inputs.append([i, [0], [-1], [-1]])
    split_outputs = []
    for i in outputs:
        split_outputs.append([i, [0], [-1], [-1]])
    axis_split_matrix = [[SplitInput(*split_inputs), SplitOutput(*split_outputs)]]
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, None, 0, 0)
    return op_cal_info_in_json
