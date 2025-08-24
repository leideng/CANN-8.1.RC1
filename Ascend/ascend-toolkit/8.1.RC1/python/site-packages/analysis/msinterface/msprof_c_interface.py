#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import importlib
import logging
import os
import sys
import multiprocessing

from common_func.config_mgr import ConfigMgr
from common_func.info_conf_reader import InfoConfReader
from common_func.platform.chip_manager import ChipManager
from common_func.profiling_scene import ProfilingScene

SO_DIR = os.path.join(os.path.dirname(__file__), "..", "lib64")


def run_in_subprocess(func, *args: any):
    """
    拉起多进程，确保每个python调c进程均为独立进程
    """
    proc = multiprocessing.Process(target=func, args=args)
    proc.start()
    proc.join()


def _dump_cann_trace(project_path: str):
    sys.path.append(os.path.realpath(SO_DIR))
    logging.info("Data will be parsed by msprof_analysis.so!")
    msprof_analysis_module = importlib.import_module("msprof_analysis")
    msprof_analysis_module.parser.dump_cann_trace(project_path)


def _dump_device_data(device_path: str):
    sys.path.append(os.path.realpath(SO_DIR))
    logging.info("Device Data will be parsed by msprof_analysis.so!")
    msprof_analysis_module = importlib.import_module("msprof_analysis")
    msprof_analysis_module.parser.dump_device_data(os.path.dirname(device_path))


def _export_unified_db(project_path: str):
    sys.path.append(os.path.realpath(SO_DIR))
    logging.info("Data will be parsed by msprof_analysis.so!")
    msprof_analysis_module = importlib.import_module("msprof_analysis")
    msprof_analysis_module.parser.export_unified_db(project_path)


def _export_timeline(project_path: str, report_json_path: str):
    sys.path.append(os.path.realpath(SO_DIR))
    logging.info("Data will be export by msprof_analysis.so!")
    msprof_analysis_module = importlib.import_module("msprof_analysis")
    msprof_analysis_module.parser.export_timeline(project_path, report_json_path)


def _export_op_summary(project_path: str):
    sys.path.append(os.path.realpath(SO_DIR))
    logging.info("Op Summary will be export by msprof_analysis.so!")
    msprof_analysis_module = importlib.import_module("msprof_analysis")
    msprof_analysis_module.parser.export_op_summary(project_path)


def dump_cann_trace(project_path: str):
    """
    调用host c化
    """
    run_in_subprocess(_dump_cann_trace, project_path)


def export_timeline(project_path: str, report_json_path: str):
    """
    调用viewer C化导出
    """
    run_in_subprocess(_export_timeline, project_path, report_json_path)


def export_op_summary(project_path: str):
    run_in_subprocess(_export_op_summary, project_path)


def dump_device_data(device_path: str) -> None:
    """
    调用device c化
    """
    if not ChipManager().is_chip_v4():
        logging.info("Do not support parsing by msprof_analysis.so!")
        return
    if ConfigMgr.is_ai_core_sample_based(device_path):
        logging.warning("Device Data in sample-based will not be parsed by msprof_analysis.so!")
        return
    all_export_flag = ProfilingScene().is_all_export() and InfoConfReader().is_all_export_version()
    if ProfilingScene().is_cpp_parse_enable() and all_export_flag:
        run_in_subprocess(_dump_device_data, device_path)
    else:
        logging.warning("Device Data will not be parsed by msprof_analysis.so!")
    return


def export_unified_db(project_path: str):
    """
    调用统一db
    """
    if not ProfilingScene().is_cpp_parse_enable():
        logging.warning("Does not support exporting the msprof.db!")
        return
    run_in_subprocess(_export_unified_db, project_path)
