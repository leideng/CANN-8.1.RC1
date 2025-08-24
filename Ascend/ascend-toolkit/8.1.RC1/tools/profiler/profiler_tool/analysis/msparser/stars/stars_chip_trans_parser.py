#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from msmodel.stars.stars_chip_trans_model import StarsChipTransModel
from msparser.interface.istars_parser import IStarsParser
from profiling_bean.stars.stars_chip_trans_bean import StarsChipTransBean


class StarsChipTransParser(IStarsParser):
    """
    stars chip trans data parser
    """

    def __init__(self: any, result_dir: str, db: str, table_list: list) -> None:
        super().__init__()
        self._model = StarsChipTransModel(result_dir, db, table_list)
        self._decoder = StarsChipTransBean
        self._data_list = []
        self._data_dict = {}

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
        for bean_data in self._data_list:
            self._data_dict.setdefault(bean_data.acc_type, []) \
                .append([bean_data.event_id, str(bean_data.pa_rx_or_pcie_write_bw),
                         str(bean_data.pa_tx_or_pcie_read_bw), str(bean_data.sys_time)])

    def flush(self: any) -> None:
        """
        flush all buffer data to db
        :return: NA
        """
        if not self._data_list:
            return
        if self._model.init():
            self.preprocess_data()
            self._model.flush(self._data_dict)
            self._model.finalize()
            self._data_list.clear()
