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


import os
import sys
import json
import copy
from typing import List, Tuple, Dict
from collections import namedtuple

from tbe.common.utils import log as logger
from op_compiler_common import verify_pathfile_exist_permission
from op_compiler_common import corrections_op_type

# 运行时禁止生成__pycache__/文件夹
sys.dont_write_bytecode = True


def _get_key_binpath(json_path: str, soc_version: str) -> Tuple[bool, str, str, str]:
    """
    :params: json_path: 
    :return:  True/False, binpath, binary_info_config_json_path, binary_info_list_key
    """
    flag, binpath, binary_info_config_json_path, binary_info_list_key = False, None, None, None 
    binpath_info = namedtuple("binpath_info", \
                              "flag, binpath, binary_info_config_json_path, binary_info_list_key")
    try:
        base_path = json_path.split("ai_core")[0].rstrip(os.sep)
        op_name = os.path.basename(os.path.dirname(json_path)) 
        bin_name = os.path.basename(json_path).replace(".json", ".o")
        if "static" in json_path:
            static_version = json_path.split("ai_core")[-1].split(os.sep)[1]
            binpath = os.path.join(static_version, soc_version, op_name, bin_name)
            binary_info_config_json_path = \
                os.path.join(base_path, "ai_core", static_version, "config", soc_version, "binary_info_config.json")
            binary_info_list_key = "staticList"
        else:
            binpath = os.path.join(soc_version, op_name, bin_name)
            binary_info_config_json_path = \
                os.path.join(base_path, "ai_core/tbe/kernel/config", soc_version, "binary_info_config.json")
            binary_info_list_key = "binaryList"
    except Exception as e:
        logger.error(f"failed to get binpath, binary_info_config_json_path, binary_info_list_key from {json_path}")
        return binpath_info(flag, binpath, binary_info_config_json_path, binary_info_list_key)

    flag = True
    return binpath_info(flag, binpath, binary_info_config_json_path, binary_info_list_key)


def _get_binary_info(op_type: str, binary_info_config_json: dict, \
                     binary_info_list_key: str, binpath: str) -> Tuple[bool, dict]:
    """
    :params: op_type
             binary_info_config_json
             binary_info_list_key
             binpath
    :return:  True/False, binary_info
    """
    if op_type in binary_info_config_json:
        binary_info_list = binary_info_config_json[op_type][binary_info_list_key]
        for binary_info in binary_info_list:
            if binary_info["binPath"] == binpath:
                return True, binary_info
    binary_info = {}
    binary_info["binPath"] = binpath
    binary_info["jsonPath"] = binpath.replace(".o", ".json")
    return True, binary_info 


def _get_debug_op_binary_info(op_type, soc_version, json_path: str=None) -> Tuple[bool, dict]:
    """
    :params: op_type
             soc_version: 芯片名代号
             json_path: 
    :return:  True/False, binary_info
    """
    binpath_info = _get_key_binpath(json_path, soc_version)
    if not binpath_info.flag:
        return False, None
    if not verify_pathfile_exist_permission(binpath_info.binary_info_config_json_path, "file", "e").exist_state and \
        not verify_pathfile_exist_permission(binpath_info.binary_info_config_json_path, "file", "r").r_permission_state:
        logger.error(f"[{binpath_info.binary_info_config_json_path}] \
            does not exist or do not have the read permission.")
        return False, None
    try:
        with open(binpath_info.binary_info_config_json_path, mode='r', encoding='utf-8') as fd:
            binary_info_config_json = json.load(fd)
    except Exception as e:
        logger.error(f"failed to open {binpath_info.binary_info_config_json_path}.")
        return False, None 
    # 获取op_binary_info
    get_binary_info_flag, binary_info = \
        _get_binary_info(op_type, binary_info_config_json, binpath_info.binary_info_list_key, binpath_info.binpath)
    return get_binary_info_flag, binary_info


def _corrections_binary_info(op_type, op_binary_info, binary_info_list_key, compile_res_path, binary_info_config_json):
    """
    :params: op_type:
             op_binary_info
             binary_info_list_key
             compile_res_path
             binary_info_config_json
    :return: binary_info_config_json
    """
    try:
        binary_info_index_list = list(range(len(op_binary_info[binary_info_list_key])))
        binary_info_index_list.reverse()
        for binary_info_index in binary_info_index_list:
            bin_name = os.path.basename(op_binary_info[binary_info_list_key][binary_info_index]["binPath"])
            compile_res_bin_path = os.path.join(compile_res_path, bin_name)
            if not verify_pathfile_exist_permission(compile_res_bin_path, "file", "e").exist_state:
                logger.warn(f"the compilation result does not exist [{bin_name}]")
                binary_info_config_json[op_type][binary_info_list_key].pop(binary_info_index) # 删除不存在编译.o的info
        if not binary_info_config_json[op_type][binary_info_list_key]: # 如果不存在对应的binary删除对应的binary_info_list_key
            binary_info_config_json[op_type].remove(binary_info_list_key)
    except Exception as e:
        return False, None
    return True, binary_info_config_json


def _corrections_binary_info_config(compile_res_path: str, binary_info_config_json: dict) -> dict:
    """
    :params: compile_res_path:
             binary_info_config_json
    :return: binary_info_config_json
    """
    binary_info_config_json_deepcopy = copy.deepcopy(binary_info_config_json)
    for op_type, op_binary_info in binary_info_config_json_deepcopy.items():
        for binary_info_list_key in ["staticList", "binaryList"]:
            if binary_info_list_key not in op_binary_info:
                continue
            flag, binary_info_config_json = _corrections_binary_info(op_type, op_binary_info, \
                binary_info_list_key, compile_res_path, binary_info_config_json)
            if not flag:
                continue
    return binary_info_config_json


def _unified_binpath_style(binary_info_config_json: dict) -> dict:
    """
    :params: binary_info_config_json
    :return: binary_info_config_json
    """
    for op_type, op_binary_info in binary_info_config_json.items():
        if "staticList" in op_binary_info:
            for binary_info in op_binary_info["staticList"]:
                static_version = binary_info["binPath"].split(os.sep)[0]
                binary_info["binPath"] = binary_info["binPath"].replace(static_version, "")
                binary_info["jsonPath"] = binary_info["binPath"].replace(static_version, "").replace('.o', '.json')
    return binary_info_config_json



def _binary_info_config_json_add_op(op_type: str, op_binary_info: dict,  \
                                    binary_info_config_json: dict, json_path: str) -> dict:
    """
    :params: op_type
             binary_info_config_json
    :return: binary_info_config_json
    """
    if op_type not in binary_info_config_json:
        binary_info_config_json[op_type] = {   
            }
    if "static" in json_path:
        if "staticList" not in binary_info_config_json[op_type]:
            binary_info_config_json[op_type]["staticList"] = [op_binary_info]
        else:
            binary_info_config_json[op_type]["staticList"].append(op_binary_info)
    else:
        if "binaryList" not in binary_info_config_json[op_type]:
            binary_info_config_json[op_type]["binaryList"] = [op_binary_info]
        else:
            binary_info_config_json[op_type]["binaryList"].append(op_binary_info)     
    return binary_info_config_json


def gen_debug_binary_info_config(soc_version, compile_res_path: str, json_path_list: list=None) -> Tuple[bool, dict]:
    """
    :params: soc_version
             compile_res_path
             json_path_list
    :return: True/False, binary_info_config_json
    """
    # 生成全量的binary_info_config_json
    binary_info_config_json = {}  
    for json_path in json_path_list:
        op_type = corrections_op_type(os.path.basename(json_path).split("_")[0]) # 获取op_type
        flag, op_binary_info = _get_debug_op_binary_info(op_type, soc_version, json_path)
        if not flag:
            logger.error(f"failed to search for binary_info based on json_path, [{json_path}]")
            return False, binary_info_config_json
        binary_info_config_json = \
            _binary_info_config_json_add_op(op_type, op_binary_info, binary_info_config_json, json_path)
    if not binary_info_config_json:
        logger.info(f"failed to gen full binary_info_config json.")
        return False, binary_info_config_json
    logger.info(f"gen full binary_info_config json.")

    # 根据opc编译结果修正binary_info_config_json
    binary_info_config_json = _corrections_binary_info_config(compile_res_path, binary_info_config_json)
    if not binary_info_config_json:
        logger.info(f"failed to corrections binary_info_config json.")
        return False, binary_info_config_json
    logger.info(f"modify binary_info_config json based on the opc compilation result.")

    # 统一staticList中binPath风格
    binary_info_config_json = _unified_binpath_style(binary_info_config_json)
    logger.info(f"unified binary_info_config json format.")

    return True, binary_info_config_json
