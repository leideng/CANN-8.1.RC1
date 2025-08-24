#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from abc import abstractmethod

from msmodel.stars.acc_pmu_model import AccPmuModel
from msparser.interface.istars_parser import IStarsParser
from profiling_bean.stars.acc_pmu import AccPmuDecoder


class AccPmuParser(IStarsParser):
    """
    class used to parse acc_pmu type data"
    """

    SAMPLE_BASED = 'sample_based'

    def __init__(self: any, result_dir: str, db: str, table_list: list) -> None:
        super().__init__()
        self._model = AccPmuModel(result_dir, db, table_list)
        self._decoder = AccPmuDecoder

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
