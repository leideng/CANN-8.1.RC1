#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from abc import abstractmethod

from msmodel.stars.sio_model import SioModel
from msparser.interface.istars_parser import IStarsParser
from profiling_bean.stars.sio_bean import SioDecoder


class SioParser(IStarsParser):
    """
    class used to parse sio data"
    """

    def __init__(self: any, result_dir: str, db: str, table_list: list) -> None:
        super().__init__()
        self._model = SioModel(result_dir, db, table_list)
        self._decoder = SioDecoder
        self._data_list = []

    @property
    def decoder(self: any) -> any:
        """
        get decoder
        :return: class decoder
        """
        return self._decoder

    @abstractmethod
    def preprocess_data(self: any) -> None:
        """
        process data list before save to db
        :return: None
        """
        sio_data = []
        for _data_bean in self._data_list:
            bandwidth = _data_bean.data_size
            sio_data.append([_data_bean.acc_id, *bandwidth, _data_bean.timestamp])
        self._data_list = sio_data
