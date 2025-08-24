#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.str_constant import StrConstant
from common_func.msprof_iteration import MsprofIteration
from msmodel.interface.view_model import ViewModel
from profiling_bean.db_dto.api_data_dto import ApiDataDto


class ApiDataViewModel(ViewModel):
    """
    class for api_viewer
    """

    def __init__(self, params: dict) -> None:
        self._result_dir = params.get(StrConstant.PARAM_RESULT_DIR)
        self._iter_range = params.get(StrConstant.PARAM_ITER_ID)
        super().__init__(self._result_dir, DBNameConstant.DB_API_EVENT,
                         [DBNameConstant.TABLE_API_DATA])

    def get_timeline_data(self: any) -> list:
        sql = "select struct_type, start, (end - start), " \
              "thread_id, level, id," \
              "item_id, connection_id " \
              "from (select* from {} where start > 0) " \
              "{where_condition}".format(DBNameConstant.TABLE_API_DATA,
                                         where_condition=self._get_where_condition())
        return DBManager.fetch_all_data(self.cur, sql)

    def get_api_total_time(self):
        search_data_sql = f"select sum(end - start) " \
                          f"from {DBNameConstant.TABLE_API_DATA} {self._get_where_condition()}"
        total_time = DBManager.fetch_all_data(self.cur, search_data_sql)
        return total_time[0] if total_time else []

    def get_api_statistic_data(self):
        total_time = self.get_api_total_time()
        if not (total_time and total_time[0]):
            return []

        search_data_sql = "select CASE WHEN level = 'acl' THEN id " \
                          "WHEN level IN ('runtime', 'model','node') THEN struct_type " \
                          "ELSE item_id END AS api_name, " \
                          "(end-start) as duration, level   " \
                          "from {1} {where_condition} " \
                          "ORDER BY api_name, level".format(total_time[0],
                                                            DBNameConstant.TABLE_API_DATA,
                                                            where_condition=self._get_where_condition())
        return DBManager.fetch_all_data(self.cur, search_data_sql)

    def get_earliests_api(self):
        search_data_sql = "select * from {} ORDER BY " \
            "start LIMIT 1".format(DBNameConstant.TABLE_API_DATA)
        return DBManager.fetch_all_data(self.cur, search_data_sql, dto_class=ApiDataDto)

    def _get_where_condition(self):
        return MsprofIteration(self._result_dir).get_condition_within_iteration(self._iter_range,
                                                                                time_start_key='start',
                                                                                time_end_key='end')