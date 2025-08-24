#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.str_constant import StrConstant
from common_func.msprof_iteration import MsprofIteration
from msmodel.interface.view_model import ViewModel
from profiling_bean.db_dto.api_data_dto import ApiDataDto, generate_api_data_from_event
from profiling_bean.db_dto.event_data_dto import EventDataDto


class EventDataViewModel(ViewModel):
    """
    class for api_viewer
    """

    def __init__(self, params: dict) -> None:
        self._result_dir = params.get(StrConstant.PARAM_RESULT_DIR)
        self._iter_range = params.get(StrConstant.PARAM_ITER_ID)
        super().__init__(self._result_dir, DBNameConstant.DB_API_EVENT,
                         [DBNameConstant.TABLE_EVENT_DATA])

    def get_timeline_data(self: any) -> list:
        event_data = self.get_all_data(DBNameConstant.TABLE_EVENT_DATA, EventDataDto)
        return self._generate_timeline_data_from_event(event_data)

    def get_all_data(self: any, table_name: str, dto_class: any = None) -> list:
        if not DBManager.judge_table_exist(self.cur, table_name):
            return []
        sql = "select struct_type, request_id, (timestamp), " \
              "thread_id, level, connection_id, " \
              "item_id from {} {where_condition}".format(DBNameConstant.TABLE_EVENT_DATA,
                                                         where_condition=self._get_where_condition())
        return DBManager.fetch_all_data(self.cur, sql, dto_class=dto_class)

    def _get_where_condition(self):
        return MsprofIteration(self._result_dir).get_condition_within_iteration(self._iter_range,
                                                                                time_start_key='timestamp',
                                                                                time_end_key='timestamp')

    def _generate_timeline_data_from_event(self, event_data: list):
        timeline_data = []
        event_logger = dict()
        for event_data_dto in event_data:
            # matching, then generate event
            friend_timestamp = event_logger.get(
                (event_data_dto.level, event_data_dto.thread_id, event_data_dto.struct_type, event_data_dto.request_id,
                 event_data_dto.item_id),
                None
            )
            if not friend_timestamp:
                event_logger[
                    (event_data_dto.level, event_data_dto.thread_id, event_data_dto.struct_type,
                     event_data_dto.request_id, event_data_dto.item_id)
                ] = event_data_dto.timestamp
                continue
            else:
                # represent 2 event with an api
                equal_api = generate_api_data_from_event(friend_timestamp, event_data_dto)
                timeline_data.append(equal_api)
                # for sub graph scene
                event_logger.pop((event_data_dto.level, event_data_dto.thread_id, event_data_dto.struct_type,
                                  event_data_dto.request_id, event_data_dto.item_id))
        return timeline_data
