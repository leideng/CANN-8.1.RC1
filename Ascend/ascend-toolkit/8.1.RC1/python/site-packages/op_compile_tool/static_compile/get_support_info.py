#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
# ----------------------------------------------------------------------------


import json
import os
import sys
from tbe.common.utils import log as logger



# 运行时禁止生成__pycache__/文件夹
sys.dont_write_bytecode = True

STATIC_KERNEL_PATH = 'static_kernel/ai_core'


def _search_key_kernel_path_in_op(kernel_name, op_binary_info):
    static_list = op_binary_info.get('staticList')
    for kernel_info in static_list:
        bin_desc = kernel_info.get('binDesc')
        if bin_desc is None:
            continue
        kernel_name_here = bin_desc.get('kernelName')
        if kernel_name_here != kernel_name:
            continue
        bin_path = kernel_info.get('binPath')
        if bin_path is None:
            raise Exception(f"error: cannot find field op_type in dump shape json: {json_file}")
        ext = os.path.splitext(bin_path)
        if len(ext) != 2:
            raise Exception(f"error: binPath error of kernel name: {kernel_name}, which is {bin_path}")
        json_path = ext[0] + '.json'
        return True, json_path
    return False, None


def _quick_search_key_kernel_path(kernel_name: str, binary_info):
    kernel_name_first = kernel_name.split('_')
    if len(kernel_name_first) == 0:
        return False, None
    op_name = kernel_name_first[0]
    if op_name not in binary_info:
        return False, None
    return _search_key_kernel_path_in_op(kernel_name, binary_info.get(op_name))


def _search_key_kernel_path(kernel_name, binary_info):
    for op in binary_info:
        op_info = binary_info.get(op)
        res, json_path = _search_key_kernel_path_in_op(kernel_name, op_info)
        if res:
            return True, json_path
    return False, None


def _get_static_binary_info_in_json(json_path: str):
    if not os.path.exists(json_path):
        raise Exception(f"error: json path is not exists: {json_path}")
    with open(json_path, mode='r', encoding='utf-8') as f:
        js = json.load(f)
        return js


def get_static_binary_info_by_kernelname(kernel_name: str, short_soc_version: str):
    ascend_opp_path = os.environ.get('ASCEND_OPP_PATH')
    if not os.path.isdir(ascend_opp_path):
        logger.error(f'please setting ASCEND_OPP_PATH env')
        return False, None, None
    binary_info_config_path = os.path.join(ascend_opp_path, STATIC_KERNEL_PATH, 'config', short_soc_version, \
        'binary_info_config.json')
    if not os.path.exists(binary_info_config_path):
        logger.warn(f'have not static kernel in short soc version: {short_soc_version}')
        return False, None, None
    with open(binary_info_config_path, mode='r', encoding='utf-8') as f:
        binary_info = json.load(f)
        res, json_path = _search_key_kernel_path(kernel_name, binary_info)
        if res:
            return True, os.path.join(ascend_opp_path, STATIC_KERNEL_PATH, json_path) , \
                _get_static_binary_info_in_json(os.path.join(ascend_opp_path, STATIC_KERNEL_PATH, json_path))
        logger.warn(f'cannot find kernelname {kernel_name} in static kernel, '
                      f'short soc version is {short_soc_version}' )
    return False, None, None


def get_static_binary_info_by_binpath(bin_path:str):
    bin_path_try = bin_path
    if not os.path.exists(bin_path):
        ascend_opp_path = os.environ.get('ASCEND_OPP_PATH')
        if ascend_opp_path is None:
            logger.error(f'cannot find binPath {bin_path}, please try set OPP env path' )
            return None
        elif os.path.isdir(ascend_opp_path):
            bin_path_try = os.path.join(ascend_opp_path, STATIC_KERNEL_PATH, bin_path)
            if not os.path.exists(bin_path_try):
                logger.error(f'cannot find binPath {bin_path} in static kernel' )
                return None
        else:
            logger.error(f'cannot find binPath {bin_path}, env ASCEND_OPP_PATH is wrong: {ascend_opp_path}' )
            return None
    ext = os.path.splitext(bin_path_try)
    if len(ext) != 2:
        raise Exception(f"error: binPath error, cannot split into 2 section, which is {bin_path}")
    json_path = ext[0] + '.json'
    return _get_static_binary_info_in_json(json_path)
