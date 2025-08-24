#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from collections import OrderedDict

from common_func.db_manager import DBManager
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.str_constant import StrConstant
from common_func.ms_constant.number_constant import NumberConstant
from common_func.msvp_common import is_number
from common_func.trace_view_header_constant import TraceViewHeaderConstant


class TraceViewManager:
    """
    Trace view Manager object
    """

    @staticmethod
    def column_graph_trace(trace_header: list, trace_data: list) -> list:
        """
        Format column graph
        """
        result_data = [0] * len(trace_data)
        if not trace_data:
            return result_data
        try:
            for index, item_data in enumerate(trace_data):
                # name, ts, pid, args is required
                result_data_part = OrderedDict(list(zip(trace_header, item_data)))
                result_data_part['ph'] = 'C'
                result_data[index] = result_data_part
            return result_data
        except (OSError, SystemError, ValueError, TypeError, RuntimeError):
            return result_data
        finally:
            pass

    @staticmethod
    def time_graph_trace(trace_header: list, trace_data: list) -> list:
        """
        Format sequence diagram
        """
        result_data = [0] * len(trace_data)
        if not trace_data:
            return result_data
        try:
            for index, item_data in enumerate(trace_data):
                # name, pid, tid, ts, duration is required
                result_data_part = OrderedDict(list(zip(trace_header, item_data)))
                result_data_part['ph'] = 'X'
                result_data[index] = result_data_part
            return result_data
        except (OSError, SystemError, ValueError, TypeError, RuntimeError):
            return result_data
        finally:
            pass

    @staticmethod
    def metadata_event(meta_data: any) -> list:
        """
        Format metadata event
        """
        if not meta_data:
            return []
        result_data = [0] * len(meta_data)
        try:
            for index, item_data in enumerate(meta_data):
                item_data_list = list(item_data)
                # name, pid, tid, args is required
                if item_data_list[0] in ["process_sort_index", "thread_sort_index"]:
                    item_data_list[3] = OrderedDict([("sort_index", item_data_list[3])])
                elif item_data_list[0] in ["process_labels"]:
                    item_data_list[3] = OrderedDict([("labels", item_data_list[3])])
                else:
                    item_data_list[3] = OrderedDict([("name", str(item_data_list[3]))])
                result_data_part = OrderedDict(list(zip(TraceViewHeaderConstant.METADATA_HEAD, item_data_list)))
                result_data_part['ph'] = 'M'
                result_data[index] = result_data_part
            return result_data
        except (OSError, SystemError, ValueError, TypeError, RuntimeError):
            return result_data
        finally:
            pass

    @staticmethod
    def add_connect_start_point(data_dict: dict, data_list: list) -> list:
        """
        add connect start point
        :param data_dict: json_data_dict
        :param data_list: ge_data_list
        :return: None
        """
        connect_list = []
        start_time = float(data_dict.get('ts', '0'))
        end_time = start_time + float(data_dict.get('dur', '0'))
        while data_list:
            ts = float(InfoConfReader().time_from_host_syscnt(data_list[0].get('timestamp', 0)) / DBManager.NSTOUS)
            if start_time <= ts <= end_time:
                connect_dict = {
                    'name': 'acl_to_npu', 'ph': 's', 'cat': StrConstant.ASYNC_ACL_NPU,
                    'id': TraceViewManager.get_line_format_pid(data_list[0].get('stream_id'),
                                data_list[0].get('task_id'), data_list[0].get('batch_id')),
                    'pid': data_dict.get('pid'), 'tid': data_dict.get('tid'), 'ts': start_time
                }
                connect_list.append(connect_dict)
            elif ts > end_time:
                break
            data_list.pop(0)
        return connect_list

    @staticmethod
    def add_connect_end_point(json_list: list) -> list:
        """
        add connect end point
        :param json_list: json_data_dict
        :return: None
        """
        if isinstance(json_list, list):
            for data_dict in json_list:
                args = data_dict.get('args', {})
                if not all(str(args.get(id, '')) for id in ('Stream Id', 'Task Id', 'Batch Id')):
                    continue
                connect_dict = {
                    'name': 'acl_to_npu', 'ph': 'f',
                    'id': TraceViewManager.get_line_format_pid(args.get('Stream Id'), args.get('Task Id'),
                                                               args.get('Batch Id')),
                    'cat': StrConstant.ASYNC_ACL_NPU, 'pid': data_dict.get('pid'), 'tid': data_dict.get('tid'),
                    'ts': data_dict.get('ts'), 'bp': 'e'
                }
                json_list.append(connect_dict)
        return json_list

    @staticmethod
    def get_format_pid(pid: int, index_id: int) -> int:
        """
        get format_pid
        :param pid: int, index_id: int
        :return: format_pid: Uint32: pid use high 22bit, index_id use middle 5bit, device_id use low 5bit
        ps: pid_max is 10^22 - 1
        """
        if is_number(InfoConfReader().get_device_id()):
            device_id = int(InfoConfReader().get_device_id())
        else:
            # host device_id is 31, we cannot use NumberConstant.HOST_ID,
            # cause this value is alse been used in record time.
            device_id = 31
        format_pid = (pid << 10) | (index_id << 5) | device_id
        return format_pid
    
    @staticmethod
    def get_line_format_pid(stream_id: int, task_id: int, batch_id: int) -> int:
        """
        get format_pid
        :param stream_id: int, task_id: int, batch_id: int
        :return: format_pid: int
        """
        stream_id_pos = 32
        task_id_pos = 16
        format_pid = (stream_id << stream_id_pos) + (task_id << task_id_pos) + batch_id
        return format_pid
