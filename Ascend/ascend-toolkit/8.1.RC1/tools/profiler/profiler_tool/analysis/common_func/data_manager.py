#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
import logging
from common_func.db_name_constant import DBNameConstant
from common_func.info_conf_reader import InfoConfReader
from msmodel.interface.view_model import ViewModel
from viewer.runtime_report import add_mem_bound, cube_usage
from common_func.msvp_common import is_number
from common_func.ms_constant.number_constant import NumberConstant


class DataManager:
    """
    class to manage or modify data
    """

    ACCURACY = "%.6f"
    AI_CORE_TASK_TYPE = 0
    AI_CPU_TASK_TYPE = 1
    AI_VECTOR_CORE_TASK_TYPE = 66

    @staticmethod
    def get_op_dict(project_path: str) -> dict:
        """
        get operator dictionary with task id and stream id
        """
        model_view = ViewModel(project_path, DBNameConstant.DB_GE_INFO, [DBNameConstant.TABLE_GE_TASK])
        model_view.check_table()
        used_cols = "op_name, task_id, stream_id"
        search_data_sql = "select {1} from {0} order by rowid" \
            .format(DBNameConstant.TABLE_GE_TASK, used_cols)
        data = model_view.get_sql_data(search_data_sql)
        task_op_dict = {}
        for sub in data:
            key = "{}-{}".format(sub[1], sub[2])  # key is task_id-stream_id
            task_op_dict[key] = sub[0]  # value is op_name
        return task_op_dict

    @classmethod
    def add_cube_usage(cls: any, headers: list, data: list) -> None:
        """
        add cube usage data
        """
        config_dict = dict()
        total_cycles = 'total_cycles'
        total_cycles_index = 'total_cycles_index'
        mac_ratio = 'mac_ratio'
        ai_core_num = 'ai_core_num'
        aic_frequency = 'aic_frequency'
        mac_ratio_index = 'mac_ratio_index'
        task_duration_index = 'task_duration_index'
        config_dict[ai_core_num] = InfoConfReader().get_data_under_device(ai_core_num)
        config_dict[aic_frequency] = InfoConfReader().get_data_under_device(aic_frequency)
        if not is_number(config_dict[aic_frequency]):
            logging.error("aic_frequency is not number.")
            return
        config_dict[aic_frequency] = float(config_dict[aic_frequency])
        if NumberConstant.is_zero(config_dict[ai_core_num]) or NumberConstant.is_zero(config_dict[aic_frequency]):
            logging.error("ai_core_num or aic_frequency is zero, calculate cube usage failed.")
            return
        if mac_ratio in headers or "aic_mac_ratio" in headers:
            config_dict[mac_ratio_index] = \
                headers.index(mac_ratio) if mac_ratio in headers else headers.index("aic_mac_ratio")
        if total_cycles in headers or "aic_total_cycles" in headers:
            config_dict[total_cycles_index] = \
                headers.index(total_cycles) if total_cycles in headers else headers.index("aic_total_cycles")
        config_dict[task_duration_index] = \
            headers.index("Task Duration(us)") if "Task Duration(us)" in headers else None
        if config_dict.get(task_duration_index, None) and config_dict.get(total_cycles_index, None) and \
                config_dict.get(mac_ratio_index, None):
            headers.append("cube_utilization(%)")
            for index, row in enumerate(data):
                data[index] = cube_usage(config_dict, list(row))

    @classmethod
    def add_memory_bound(cls: any, headers: list, data: list) -> None:
        """
        add memory bound data
        """
        if all(header in headers for header in ("mac_ratio", "vec_ratio", "mte2_ratio")):
            mte2_index = headers.index("mte2_ratio")
            vec_index = headers.index("vec_ratio")
            mac_index = headers.index("mac_ratio")
            headers.append("memory_bound")
            for index, row in enumerate(data):
                data[index] = add_mem_bound(list(row), vec_index, mac_index, mte2_index)

        elif all(header in headers for header in ("mac_exe_ratio", "vec_exe_ratio", "mte2_exe_ratio")):
            mte2_index = headers.index("mte2_exe_ratio")
            vec_index = headers.index("vec_exe_ratio")
            mac_index = headers.index("mac_exe_ratio")
            headers.append("memory_bound")
            for index, row in enumerate(data):
                data[index] = add_mem_bound(list(row), vec_index, mac_index, mte2_index)

    @classmethod
    def add_iter_id(cls: any, *args: any, task_type_index: int = None, iter_id: int = 1) -> bool:
        """
        add iteration id
        """
        data, task_id_index, stream_id_index = args
        iter_id_record = {}
        for row_num, row in enumerate(data):
            if len(row) < task_id_index or len(row) < stream_id_index:
                return False
            if task_type_index is not None and str(task_type_index).isdigit() \
                    and row[int(task_type_index)] not in (cls.AI_CORE_TASK_TYPE,
                                                          cls.AI_CPU_TASK_TYPE,
                                                          cls.AI_VECTOR_CORE_TASK_TYPE):
                data[row_num] = list(row)
                data[row_num].append(0)
                continue
            iter_id_tag = "{}-{}".format(row[task_id_index], row[stream_id_index])
            iter_id_record[iter_id_tag] = iter_id_record.get(iter_id_tag, 0) + 1
            if iter_id_record.get(iter_id_tag) > iter_id:
                iter_id = iter_id_record.get(iter_id_tag)
            data[row_num] = list(row)
            data[row_num].append(iter_id)
        return True
