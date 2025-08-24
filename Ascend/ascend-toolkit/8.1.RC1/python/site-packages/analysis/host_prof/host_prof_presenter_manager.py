#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import json
import logging
from enum import Enum

from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from common_func.trace_view_header_constant import TraceViewHeaderConstant
from common_func.trace_view_manager import TraceViewManager
from host_prof.host_cpu_usage.presenter.host_cpu_usage_presenter import HostCpuUsagePresenter
from host_prof.host_disk_usage.presenter.host_disk_usage_presenter import HostDiskUsagePresenter
from host_prof.host_mem_usage.presenter.host_mem_usage_presenter import HostMemUsagePresenter
from host_prof.host_network_usage.presenter.host_network_usage_presenter import \
    HostNetworkUsagePresenter
from host_prof.host_syscall.presenter.host_syscall_presenter import HostSyscallPresenter
from viewer.get_trace_timeline import TraceViewer


class HostExportType(Enum):
    """
    export type enum
    """
    CPU_USAGE = 1
    MEM_USAGE = 2
    DISK_USAGE = 3
    NETWORK_USAGE = 4
    HOST_RUNTIME_API = 5


def init_presenter(result_dir: str, export_type: int) -> any:
    host_presenter_dict = {
        HostExportType.CPU_USAGE: HostCpuUsagePresenter,
        HostExportType.MEM_USAGE: HostMemUsagePresenter,
        HostExportType.DISK_USAGE: HostDiskUsagePresenter,
        HostExportType.NETWORK_USAGE: HostNetworkUsagePresenter,
        HostExportType.HOST_RUNTIME_API: HostSyscallPresenter
    }
    presenter = host_presenter_dict.get(export_type)(result_dir)
    presenter.init()
    return presenter


def get_host_prof_timeline(result_dir: str, export_type: int) -> list:
    """
    Return trace-viewer json format host prof data timeline
    """
    presenter = init_presenter(result_dir, export_type)
    header = presenter.get_timeline_header()
    data_list = presenter.get_timeline_data()
    if not data_list:
        logging.warning("failed to get os runtime data, may be the pid set is invalid, please check.")
        return []
    if export_type == HostExportType.HOST_RUNTIME_API:
        return get_time_data(data_list, header)
    return get_column_data(data_list, header)


def get_host_prof_summary(result_dir: str, export_type: int, configs: dict) -> tuple:
    """
    Return csv format of host prof data (summary)
    """
    presenter = init_presenter(result_dir, export_type)
    headers = configs.get('headers')
    data = presenter.get_summary_data()
    return headers, data, len(data)


def get_time_data(data_list: list, header: list) -> list:
    """
    get time timeline
    """
    trace_parser = TraceViewer('HostProf')
    _trace = TraceViewManager.time_graph_trace(TraceViewHeaderConstant.TASK_TIME_GRAPH_HEAD, data_list)
    result = TraceViewManager.metadata_event(header)
    result.extend(_trace)
    return result


def get_column_data(data_list: list, header: list) -> list:
    """
    usage timeline
    """
    json_header = TraceViewHeaderConstant.COLUMN_GRAPH_HEAD_LEAST
    trace_parser = TraceViewer('HostProf')
    pid = InfoConfReader().get_json_pid_data()
    tid = InfoConfReader().get_json_tid_data()
    # data example [name, ts, usage]
    for data_item in data_list:
        data_item.insert(2, pid)
        data_item.insert(3, tid)
    _trace = TraceViewManager.column_graph_trace(json_header, data_list)
    result = TraceViewManager.metadata_event(header)
    result.extend(_trace)
    return result
