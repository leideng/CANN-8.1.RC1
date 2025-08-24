#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msmodel.ge.ge_hash_model import GeHashViewModel
from msmodel.interface.parser_model import ParserModel
from profiling_bean.struct_info.event_data_bean import EventDataBean


class EventDataModel(ParserModel):
    """
    event model class
    """

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_API_EVENT, [DBNameConstant.TABLE_EVENT_DATA])

    @staticmethod
    def update_hash_value(data: EventDataBean, hash_dict: dict):
        if data.level not in hash_dict:
            return data.struct_type
        return hash_dict[data.level].get(data.struct_type, data.struct_type)

    def flush(self: any, data_list: list, table_name: str = DBNameConstant.TABLE_EVENT_DATA) -> None:
        """
        insert data to table
        :param data_list: event data
        :param table_name: table name
        :return:
        """
        data_list = self.reformat_data(data_list)
        self.insert_data_to_db(table_name, data_list)

    def reformat_data(self: any, data_list: list) -> list:
        with GeHashViewModel(self.result_dir) as _model:
            hash_dict = _model.get_type_hash_data()
        return [
            [
                self.update_hash_value(data, hash_dict), data.level, data.thread_id,
                data.item_id, data.request_id, data.timestamp, connection_id,
            ]
            for connection_id, data in data_list
        ]
