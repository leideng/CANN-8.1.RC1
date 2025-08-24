#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging

from common_func.constant import Constant
from common_func.db_name_constant import DBNameConstant
from common_func.hash_dict_constant import HashDictData
from msmodel.interface.parser_model import ParserModel
from profiling_bean.prof_enum.data_tag import AclApiTag
from profiling_bean.struct_info.api_data_bean import ApiDataBean


class ApiDataModel(ParserModel):
    """
    api model class
    """

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_API_EVENT, [DBNameConstant.TABLE_API_DATA])

    @staticmethod
    def update_type_hash_value(data: ApiDataBean, hash_dict: dict) -> tuple:
        if data.level not in hash_dict:
            return data.struct_type, 0
        # acl and hccl have two hash values, other type set default second value：0
        if data.level == 'acl':
            return AclApiTag(data.acl_type).name, \
                   hash_dict[data.level].get(data.struct_type, data.struct_type)
        return hash_dict[data.level].get(data.struct_type, data.struct_type), 0

    def flush(self: any, data_list: list, table_name: str = DBNameConstant.TABLE_API_DATA) -> None:
        """
        insert data to table
        :param data_list: api data
        :param table_name: table name
        :return:
        """
        try:
            data_list = self.reformat_data(data_list)
        except (IndexError, ValueError) as _err:
            logging.error(str(_err), exc_info=Constant.TRACE_BACK_SWITCH)
            return
        self.insert_data_to_db(table_name, data_list)

    def reformat_data(self: any, data_list: list) -> list:
        hash_dict_data = HashDictData(self.result_dir)
        type_dict = hash_dict_data.get_type_hash_dict()
        ge_dict = hash_dict_data.get_ge_hash_dict()
        return [
            [
                *self.update_type_hash_value(data, type_dict), data.level, data.thread_id,
                ge_dict.get(data.item_id, data.item_id), data.start, data.end, connection_id,
            ]
            for connection_id, data in data_list
        ]
