#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import json
import os.path
import re

from params import ParamDict
from common import consts
from common import log_debug, log_warning
from common import FileOperate as f
from common.file_operate import COPY_MODE, MOVE_MODE
from drv import EnvVarName

__all__ = ["collect_ops"]


def get_fault_kernel_name(output_root_path):
    """
    Obtains fault_kernel_name from plog.
    """
    error_msg = ["Aicore kernel execute failed", "fftsplus task execute failed"]
    run_plog = os.path.join(output_root_path, "dfx", "log", "host", "cann", "run", "plog")
    debug_plog = os.path.join(output_root_path, "dfx", "log", "host", "cann", "debug", "plog")
    plog_files = [run_plog, debug_plog]
    for plog in plog_files:
        if not f.check_dir(plog):
            continue
        plog_lines = []
        for msg in error_msg:
            cmd_ret = os.popen(f"grep '{msg}' -inrE {plog}")
            plog_lines += cmd_ret.readlines()
            cmd_ret.close()
        if len(plog_lines) == 0:
            continue

        static_regexp = r" stream_id=\d+,.*?task_id=\d+,.*?fault kernel_name=.*?,.*?" \
                        r" fault kernel info ext=(.*?),"
        dynamic_regexp = r" stream_id=\d+,.*?task_id=\d+,.*?fault kernel_name=(.*?),"

        for regexp in [static_regexp, dynamic_regexp]:
            kernel_name_ret = re.findall(regexp, plog_lines[0])
            if not kernel_name_ret or kernel_name_ret[0] == 'none':
                continue
            kernel_name = kernel_name_ret[0]
            return kernel_name.replace("_mix_aic", "").replace("_mix_aiv", "")
    return None


def get_all_kernel_name_from_file(file_path):
    """
    read JSON file and check whether they contain kernel_name.
    """
    all_kernel_name = []
    try:
        with open(file_path, 'r') as json_file:
            dict_obj = json.load(json_file)
        all_kernel_name.append(dict_obj.get("binFileName"))
        all_kernel_name.append(dict_obj.get("kernelName"))
        all_kernel_name += [kernel.get("kernelName") for kernel in dict_obj.get("kernelList", [])]
    except Exception as e:
        log_debug(f"failed to load the '{file_path}', {e}")
        return all_kernel_name

    # remove its 'None' elements
    return list(filter(None, all_kernel_name))


def get_fault_kernel_name_files(collect_path, kernel_name):
    """
    Obtain the .o.json file corresponding to fault_kernel_name.
    """
    collect_files = []
    if not kernel_name:
        return collect_files
    opp_path = EnvVarName().opp_path
    for path, _, files in os.walk(collect_path):
        for file in files:
            file_path = os.path.join(path, file)
            # ASCEND_OPP_PATH only needs to read files in the '/kernel/'.
            if not file.endswith(".json") or (collect_path == opp_path and "kernel" not in path.split("/")):
                continue
            # read all JSON files and check whether they contain kernel_name.
            if kernel_name not in get_all_kernel_name_from_file(file_path):
                continue
            # collect the .json file and .o file that contain kernel_name.
            collect_files.append(file_path)
            o_file_path = os.path.join(path, file.replace(".json", ".o"))
            if os.path.isfile(o_file_path):
                collect_files.append(o_file_path)

    return collect_files


def collect_op_files(ops_res, target_dir, mode=COPY_MODE):
    """
    Collect the .o.json file corresponding to fault_kernel_name.
    """
    ret = True
    for file_path in ops_res:
        op_file_ret = f.collect_file_to_dir(file_path, target_dir, mode)
        ret = ret and op_file_ret
    return ret


def collect_ops_from_dump(output_root_path):
    """
    Collect ops files from the dump directory.
    """
    target_dir = os.path.join(output_root_path, "dfx", "ops")
    dump_path = os.path.join(output_root_path, "dfx", "data-dump")
    if not f.check_dir(dump_path):
        return False
    ops_files = []
    for path, _, files in os.walk(dump_path):
        for file in files:
            if file.endswith(".o") or file.endswith(".json"):
                ops_files.append(os.path.join(path, file))
    if len(ops_files) >= 3:  # two .o, one .json
        return collect_op_files(ops_files, target_dir, MOVE_MODE)

    return False


def collect_ops_files_env_var(output_root_path, ops_target_dir):

    collect_path_list = []
    task_dir = ParamDict().get_arg("task_dir")
    # ops files priority: NPU_COLLECT_PATH > ASCEND_CACHE_PATH > ASCEND_WORK_PATH > $HOME/atc_data >
    # ASCEND_CUSTOM_OPP_PATH > ASCEND_OPP_PATH > ./
    env_var = EnvVarName()
    for collect_path in [task_dir, env_var.npu_collect_path, env_var.cache_path, env_var.work_path,
                         os.path.join(env_var.home_path, "atc_data"), env_var.custom_opp_path, env_var.opp_path,
                         env_var.current_path]:
        if collect_path and f.check_dir(collect_path):
            collect_path_list.append(collect_path)

    kernel_name = get_fault_kernel_name(output_root_path)
    for path in collect_path_list:
        collect_files = get_fault_kernel_name_files(path, kernel_name)
        if collect_files:
            return collect_op_files(collect_files, ops_target_dir, COPY_MODE)

    log_warning("The JSON file of the fault kernel_name is not found.")
    return False


def check_launch_ops():
    if (ParamDict().get_command() == consts.launch_cmd) and (not ParamDict().get_ini("ops") == "1"):
        log_debug("ops is not set on, not collect ops files")
        return False
    return True


def collect_debug_kernel(output_root_path):
    ops_target_dir = os.path.join(output_root_path, "dfx", "ops")
    opp_path = EnvVarName().opp_path
    if opp_path is None:
        log_debug("ASCEND_OPP_PATH not set")
        return
    debug_kernel_path = os.path.join(opp_path, "debug_kernel")
    if debug_kernel_path and f.check_access(debug_kernel_path) and f.check_dir(debug_kernel_path):
        if f.list_dir(debug_kernel_path):
            debug_kernel_target_path = os.path.join(ops_target_dir, os.path.basename(debug_kernel_path))
            if debug_kernel_target_path.startswith(debug_kernel_path):
                log_debug("Cannot copy debug_kernel to %s" % debug_kernel_target_path)
            else:
                f.copy_dir(debug_kernel_path, debug_kernel_target_path)


def collect_file(output_root_path):

    ops_target_dir = os.path.join(output_root_path, "dfx", "ops")
    ret = False
    if ParamDict().get_command() == consts.launch_cmd:
        ops_source_dir = os.path.join(
            ParamDict().asys_output_timestamp_dir, "npu_collect_intermediates", "extra-info", "ops")
        if f.check_dir(ops_source_dir):
            ret = f.collect_dir(ops_source_dir, ops_target_dir, MOVE_MODE)
    else:
        ret = collect_ops_files_env_var(output_root_path, ops_target_dir)
    if not ret:
        log_warning("ops collect failed.")


def collect_cfg_json(output_root_path, cfg_dir, json_dir, config):
    if not os.path.exists(json_dir):
        return False

    ret = True
    for path, _, files in os.walk(json_dir):
        if "/config/" not in path:
            continue
        for file in files:
            if not file.endswith(".json"):
                continue
            dst_dir = os.path.join(output_root_path, "dfx", "ops", config, cfg_dir, os.path.relpath(path, json_dir))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            ret = ret and f.copy_file_to_dir(os.path.join(path, file), dst_dir)
    return ret


def collect_opp_config(output_root_path):
    opp_path = EnvVarName().opp_path
    if opp_path is None:
        log_debug("ASCEND_OPP_PATH is not set.")
        return False

    config_path = os.path.join(opp_path, "vendors", "config.ini")
    if not os.path.isfile(config_path):
        log_debug(f"The {config_path} is not a file.")
        return False
    try:
        with open(config_path, 'r') as cfg:
            cfg_content = cfg.read()
    except PermissionError:
        log_warning(f"The {config_path} file does not have the read permission.")
        return False

    load_priority = re.search("load_priority=(.+?)\n", cfg_content)
    if not load_priority:
        log_warning(f"The {config_path} file does not contain the load_priority field.")
        return False
    load_priority = load_priority.group(1).split(",")
    ret = True
    for cfg_dir in load_priority:
        # remove front and back spaces
        _cfg_dir = cfg_dir.strip()
        json_dir = os.path.join(opp_path, "vendors", _cfg_dir)
        ret = ret and collect_cfg_json(output_root_path, _cfg_dir, json_dir, "vendor_config")

    ret = ret and f.copy_file_to_dir(config_path, os.path.join(output_root_path, "dfx", "ops", "vendor_config"))
    return ret


def collect_custom_opp_config(output_root_path):
    custom_opp_path = EnvVarName().custom_opp_path
    if custom_opp_path is None:
        log_debug("ASCEND_CUSTOM_OPP_PATH is not set.")
        return False
    return collect_cfg_json(output_root_path, "", custom_opp_path, "custom_config")


def collect_ops(output_root_path):
    if not check_launch_ops():
        return

    if not collect_ops_from_dump(output_root_path):
        collect_file(output_root_path)
    collect_debug_kernel(output_root_path)
    collect_opp_config(output_root_path)
    collect_custom_opp_config(output_root_path)
