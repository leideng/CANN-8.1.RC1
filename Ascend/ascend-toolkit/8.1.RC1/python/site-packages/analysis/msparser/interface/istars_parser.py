#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from abc import abstractmethod


class IStarsParser:
    """
    parser interface
    """

    MAX_DATA_LEN = 1000000000

    def __init__(self: any) -> None:
        self._model = None
        self._decoder = None
        self._data_list = []

    def handle(self: any, _: any, data: bytes) -> None:
        """
        Process a single piece of binary data, and the subclass can also overwrite it.
        :param _: sqe_type some parser need it
        :param data: binary data
        :return:
        """
        if len(self._data_list) >= self.MAX_DATA_LEN:
            self.flush()
        self._data_list.append(self._decoder.decode(data))

    @abstractmethod
    def preprocess_data(self: any) -> None:
        """
        Before saving to the database, subclasses can implement this method.
        Do another layer of preprocessing on the data in the buffer
        :return:result data list
        """

    def flush(self: any) -> None:
        """
        flush all buffer data to db
        :return: NA
        """
        if not self._data_list:
            return
        if self._model.init():
            self.preprocess_data()
            self._model.flush(self._data_list)
            self._model.finalize()
            self._data_list.clear()
