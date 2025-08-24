#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from msmodel.stars.inter_soc_model import InterSocModel
from msparser.interface.istars_parser import IStarsParser
from profiling_bean.stars.inter_soc import InterSoc


class InterSocParser(IStarsParser):
    """
    class used to parse inter soc Transmission type data"
    """

    def __init__(self: any, result_dir: str, db: str, table_list: list) -> None:
        super().__init__()
        self._model = InterSocModel(result_dir, db, table_list)
        self._decoder = InterSoc
        self._data_list = []

    @property
    def decoder(self: any) -> any:
        """
        get decoder
        :return: class decoder
        """
        return self._decoder

    def preprocess_data(self: any) -> None:
        """
        process data list before save to db
        :return: None
        """
        inter_soc_data = []
        for _data_bean in self._data_list:
            inter_soc_data.append([_data_bean.mata_bw_level, _data_bean.l2_buffer_bw_level, _data_bean.sys_time])
        self._data_list = inter_soc_data
