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
from typing import List, Tuple, Dict
from tbe.common.utils import log as logger
from op_compiler_common import verify_pathfile_exist_permission
from op_compiler_common import get_opp_path


FLAG, ASCEND_OPP_PATH = get_opp_path()
DYNAMIC_KERNEL_PATH = 'built-in/op_impl/ai_core/tbe/kernel'

# 运行时禁止生成__pycache__/文件夹
sys.dont_write_bytecode = True


def _parse_kernel_name_from_json(kernel_dyn_binary_json: str) -> Tuple[bool, str]:
    """
    :params: kernel_dyn_binary_json: 
    :return: True/False, kernel_name
    """
    try:
        with open(kernel_dyn_binary_json, mode='r', encoding='utf-8') as fd:
            kernel_dyn_binary_info = json.load(fd)
    except Exception as e:
        logger.exception(f"failed to open {kernel_dyn_binary_json}")
        return False, None
    if "kernelName" in kernel_dyn_binary_info:
        kernel_name = kernel_dyn_binary_info["kernelName"]
        return True, kernel_name
    return False, None


def _verify_bin_json_path(dir_: str, file_: str) -> Tuple[bool, str, str]:
    """
    :params: dir 
             file
    :return: True/False, dyn_binary_info 
    """
    bin_path = os.path.join(dir_, file_)
    json_path = os.path.splitext(bin_path)[0] + ".json"
    if not verify_pathfile_exist_permission(bin_path, "path", "e").exist_state \
        and not verify_pathfile_exist_permission(bin_path, "path", "r").r_permission_state:
        logger.error(f"[{bin_path}] does not exist or do not have the read permission")
        return False, None, None
    if not verify_pathfile_exist_permission(json_path, "path", "e").exist_state \
        and not verify_pathfile_exist_permission(json_path, "path", "r").r_permission_state:
        logger.error(f"[{json_path}] does not exist or do not have the read permission")
        return False, None, None
    return True, bin_path, json_path


def init_dyn_binary_info(short_soc_version: str, dyn_binary_path: str) -> Tuple[bool, dict]:
    """
    :params: short_soc_version: 
                dyn_binary_path
    :return: True/False, dyn_binary_info 
    """
    dyn_binary_info = {}
    dyn_socversion_path = os.path.join(dyn_binary_path, short_soc_version)
    try:
        for root, dirs, files in os.walk(dyn_socversion_path):
            if root == dyn_socversion_path:
                continue
            op_name = os.path.basename(root)
            if op_name not in dyn_binary_info:
                dyn_binary_info[op_name] = {}
            for file_ in files:
                if not file_.endswith(".o"):
                    continue
                verify_bin_json_path_flag, bin_path, json_path = _verify_bin_json_path(root, file_)
                if not verify_bin_json_path_flag:
                    continue
                parse_kernel_dyn_binary_json_flag, kernel_name = _parse_kernel_name_from_json(json_path)
                if not parse_kernel_dyn_binary_json_flag:
                    continue
                dyn_binary_info[op_name][kernel_name] = {}
                dyn_binary_info[op_name][kernel_name]["binPath"] = bin_path
                dyn_binary_info[op_name][kernel_name]["jsonPath"] = json_path
                try:
                    with open(json_path, mode='r', encoding='utf-8') as fd:
                        kernel_json_info = json.load(fd)
                except Exception as e:
                    logger.exception(f"failed to open {kernel_json_path}")
                dyn_binary_info[op_name][kernel_name]["jsonInfo"] = kernel_json_info
    except Exception as e:
        logger.exception(f"failed to init dyn_binary_info")
        return False, dyn_binary_info
    return True, dyn_binary_info


def search_kernelname_for_kerneljsoninfo(kernel_name: str, kernel_json_info: dict) -> bool:
    """
    search_kernel_name for kernel_json_info.
    :param key: kernel_name
    :return: True/False
    """
    if kernel_name == kernel_json_info["kernelName"]: # search kernelName
        return True
    if "kernelList" not in  kernel_json_info:
        return False
    for subkernel_name_info in kernel_json_info.get("kernelList"): # search kernelList
        if kernel_name == subkernel_name_info["kernelName"]:
            return True
        elif kernel_name.endswith("_mix_aic"):
            kernel_name = kernel_name.replace("_mix_aic", "")
            if kernel_name == subkernel_name_info["kernelName"]:
                return True
        elif kernel_name.endswith("_mix_aiv"):
            kernel_name = kernel_name.replace("_mix_aiv", "")
            if kernel_name == subkernel_name_info["kernelName"]:
                return True
    return False


def get_dyn_binpath_by_kernelname(kernel_name: str, dyn_binary_info: dict, soc_version: str) -> Tuple[bool, str]:
    """
    search for support_info based on the kernel name.
    :param key: kernel_name
    :return: True/False, bin_path, json_path
    """
    for op_name, kernel_info in dyn_binary_info.items():
        for _, info in kernel_info.items():   
            if search_kernelname_for_kerneljsoninfo(kernel_name, info["jsonInfo"]):
                logger.debug(f"kernel name is found in {info['jsonPath']}.")
                return True, info["binPath"], info["jsonPath"]
    logger.warn(f"cannot find kernelname {kernel_name} in dynamic kernel, short soc version is {soc_version}")
    return False, None, None


def get_binary_info_by_binpath(json_path: str) -> Tuple[bool, dict]:
    """
    search for support_info based for json_path.
    :params: json_path
    :return: True/False, binary_info_config_json
    """
    binary_json_info = None
    if json_path.endswith(".o"):
        json_path = os.path.splitext(json_path)[0] + ".json"
    if not verify_pathfile_exist_permission(json_path, "file", "e").exist_state:
        logger.error(f"[{json_path}] path does not exist. Please check.")
        return False, binary_json_info
    if not verify_pathfile_exist_permission(json_path, "file", "r").r_permission_state:
        logger.error(f"[{json_path}] do not have the read permission., please setting.")
        return False, binary_json_info
    try:
        with open(json_path, mode='r', encoding='utf-8') as fd:
            binary_json_info = json.load(fd)
            logger.debug(f"the binary_info_config file is opened successfully, {json_path}.")
    except Exception as e:
        logger.exception(f"{json_path}, failed to open the op_debug_config file.")
        return False, binary_json_info
    return True, binary_json_info


