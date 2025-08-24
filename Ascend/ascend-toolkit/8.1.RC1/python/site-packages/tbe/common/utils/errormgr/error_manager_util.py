#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
error manager util
"""
import json
import os


def get_error_message(args):
    """
    :param args: dict
    keys in dict must be in accordance with xlsx
    :return: string
    formatted message
    """
    error_code = args.get("errCode")
    file_name = os.path.dirname(os.path.abspath(__file__))
    with open("{}/errormanager.json".format(file_name)) as file_content:
        data = json.load(file_content)
        error_dict = {}
        for error_message in data:
            error_dict[error_message['errCode']] = error_message
        error_json = error_dict
    error_stmt = error_json.get(error_code)
    if error_stmt is None:
        return "errCode = {} has not been defined".format(error_code)
    arg_list = error_stmt.get("argList").split(",")
    arg_value = []
    for arg_name in arg_list:
        if arg_name.strip() not in args:
            arg_value.append("")
        else:
            arg_value.append(args.get(arg_name.strip()))
    msg = error_json.get(error_code).get("errMessage") % tuple(arg_value)
    msg = msg.replace("[]", "")
    return msg


def raise_runtime_error(dict_args):
    """
    raise runtime error
    :param dict_args: error message dict
    """
    raise RuntimeError(dict_args,
                       get_error_message(dict_args))


def raise_runtime_error_cube(args_dict, *msgs):
    """
    raise runtime error
    :param args_dict: input dict
    :param msg: error message
    """
    raise RuntimeError(args_dict, *msgs)
