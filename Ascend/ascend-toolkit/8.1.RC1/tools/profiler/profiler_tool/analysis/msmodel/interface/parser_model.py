#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from abc import abstractmethod

from msmodel.interface.base_model import BaseModel


class ParserModel(BaseModel):
    """
    class used to calculate
    """

    @abstractmethod
    def flush(self: any, data_list: list) -> None:
        """
        base method to insert data into database
        """

    def init(self: any) -> bool:
        """
        create db and tables
        """
        if not super().init():
            return False
        self.create_table()
        return True
