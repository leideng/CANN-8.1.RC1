#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import logging
import sqlite3

from common_func.constant import Constant
from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.str_constant import StrConstant
from mscalculate.aic.aic_utils import AicPmuUtils
from msmodel.interface.parser_model import ParserModel
from viewer.calculate_rts_data import get_metrics_from_sample_config


class FftsPmuModel(ParserModel):
    """
    ffts pmu model.
    """

    def __init__(self: any, result_dir: str, db_name: str, table_list: list) -> None:
        super().__init__(result_dir, db_name, table_list)
        self.events_name_list = []
        self.profiling_events = {'aic': [], 'aiv': []}

    def create_table(self: any) -> None:
        """
        create aic and aiv table by sample.json
        :return:
        """
        # aic metrics table
        column_list = []
        self.profiling_events['aiv'] = get_metrics_from_sample_config(self.result_dir,
                                                                      StrConstant.AIV_PROFILING_METRICS)
        self.profiling_events['aic'] = get_metrics_from_sample_config(self.result_dir)
        self.update_pmu_list(column_list)
        self._creat_metric_table_by_head(column_list, DBNameConstant.TABLE_METRIC_SUMMARY)

    def update_pmu_list(self: any, column_list: list) -> None:
        for core_type, pmu_list in self.profiling_events.items():
            core_column_list = AicPmuUtils.remove_unused_column(pmu_list)
            for column in core_column_list:
                column_list.append('{0}_{1}'.format(core_type, column))

    def flush(self: any, data_list: list) -> None:
        """
        insert data into database
        :param data_list: ffts pmu data list
        :return: None
        """
        if not data_list:
            logging.warning("ffts pmu data is empty, no data found.")
            return
        self.insert_data_to_db(DBNameConstant.TABLE_METRIC_SUMMARY, data_list)

    def _creat_metric_table_by_head(self: any, metrics: list, table_name: str) -> None:
        """
        insert event value into metric op_summary
        """
        sql = 'CREATE TABLE IF NOT EXISTS {name}({column})'.format(
            column=','.join(metric.replace('(ms)', '').replace('(GB/s)', '')
                            + ' numeric' for metric in metrics) + ', task_id INT, '
                                                                  'stream_id INT,'
                                                                  'subtask_id INT,'
                                                                  'task_type INT, '
                                                                  'start_time INT,'
                                                                  'end_time INT, '
                                                                  'ffts_type INT,'
                                                                  'core_type INT,'
                                                                  'batch_id INT', name=table_name)
        try:
            DBManager.execute_sql(self.conn, sql)
        except sqlite3.Error as err:
            logging.error(err, exc_info=Constant.TRACE_BACK_SWITCH)
