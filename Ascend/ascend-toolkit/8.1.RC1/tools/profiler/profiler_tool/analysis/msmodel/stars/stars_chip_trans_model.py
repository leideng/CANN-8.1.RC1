#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.stars_constant import StarsConstant
from msmodel.interface.parser_model import ParserModel


class StarsChipTransModel(ParserModel):
    """
    stars model class
    """

    TYPE_TABLE_MAP = {
        StarsConstant.TYPE_STARS_PA: DBNameConstant.TABLE_STARS_PA_LINK,
        StarsConstant.TYPE_STARS_PCIE: DBNameConstant.TABLE_STARS_PCIE
    }

    def __init__(self: any, result_dir: str, db: str, table_list: list) -> None:
        super().__init__(result_dir, db, table_list)

    def flush(self: any, data_dict: dict) -> None:
        """
        insert stars data into database
        """
        for _type in data_dict.keys():
            if not self.TYPE_TABLE_MAP.get(_type):
                continue
            self.insert_data_to_db(self.TYPE_TABLE_MAP.get(_type), data_dict.get(_type))
