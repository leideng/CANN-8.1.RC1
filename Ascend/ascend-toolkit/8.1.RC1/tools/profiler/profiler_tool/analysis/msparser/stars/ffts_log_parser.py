#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from abc import abstractmethod

from common_func.platform.chip_manager import ChipManager
from msmodel.stars.ffts_log_model import FftsLogModel
from msparser.interface.istars_parser import IStarsParser
from profiling_bean.stars.ffts_log import FftsLogDecoder
from profiling_bean.stars.ffts_plus_log import FftsPlusLogDecoder


class FftsLogParser(IStarsParser):
    """
    class used to parse ffts thread log type data"
    """

    def __init__(self: any, result_dir: str, db: str, table_list: list) -> None:
        super().__init__()
        self._model = FftsLogModel(result_dir, db, table_list)
        self._decoder = self._get_log_decoder()

    @property
    def decoder(self: any) -> any:
        """
        get decoder
        :return: class decoder
        """
        return self._decoder

    @classmethod
    def _get_log_decoder(cls: any) -> any:
        if ChipManager().is_ffts_plus_type():
            return FftsPlusLogDecoder
        return FftsLogDecoder

    @abstractmethod
    def preprocess_data(self: any) -> None:
        """
        process data list before save to db
        :return: None
        """
