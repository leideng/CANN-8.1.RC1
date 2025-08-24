#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import struct

from common_func.info_conf_reader import InfoConfReader
from msmodel.stars.lowpower_model import LowPowerModel
from msparser.interface.istars_parser import IStarsParser
from profiling_bean.stars.lowpower_bean import LowPowerBean


class LowPowerParser(IStarsParser):
    """
    stars chip trans data parser
    """

    def __init__(self: any, result_dir: str, db: str, table_list: list) -> None:
        super().__init__()
        self._model = LowPowerModel(result_dir, db, table_list)
        self._decoder = LowPowerBean
        self._data_list = {}

    def handle(self: any, _: any, data: bytes) -> None:
        """
        Process a single piece of binary data, and the subclass can also overwrite it.
        :param _: sqe_type some parser need it
        :param data: binary data
        :return:
        """
        if len(self._data_list) >= self.MAX_DATA_LEN:
            self.flush()
            self._data_list = {}
        _, sys_time = struct.unpack("=QQ", data[:16])
        self._data_list.setdefault(sys_time, self._decoder()).decode(data)

    def preprocess_data(self: any) -> None:
        """
        process data list before save to db
        :return: None
        """
        data_list = []
        for timestamp, low_power_data in self._data_list.items():
            voltage = 0
            if low_power_data.aic_current_sd5003:
                voltage = low_power_data.power_consumption_sd5003 // low_power_data.aic_current_sd5003
            data_list.append(
                [InfoConfReader().time_from_syscnt(timestamp), low_power_data.vf_samp_cnt,
                 low_power_data.pwr_samp_cnt, low_power_data.temp_samp_cnt, low_power_data.tem_of_ai_core,
                 low_power_data.tem_of_hbm, low_power_data.tem_of_hbm_granularity, low_power_data.tem_of_3ds_ram,
                 low_power_data.tem_of_cpu, low_power_data.tem_of_peripherals, low_power_data.tem_of_l2_buff,
                 low_power_data.aic_current_dpm, low_power_data.power_consumption_dpm,
                 low_power_data.aic_current_sd5003, low_power_data.power_consumption_sd5003,
                 low_power_data.pwr2prof_dat_imon, low_power_data.pmc2prof_freq_prof, low_power_data.tem_warn_cnt0,
                 low_power_data.tem_warn_cnt1, low_power_data.tem_warn_cnt2,
                 low_power_data.tem_warn_cnt3, low_power_data.cos_warn_cnt0, low_power_data.cos_warn_cnt1,
                 low_power_data.cos_warn_cnt2, low_power_data.cos_warn_cnt3, low_power_data.io_warn_cnt0,
                 low_power_data.io_warn_cnt1, low_power_data.io_warn_cnt2, low_power_data.io_warn_cnt3,
                 low_power_data.epd_warn, low_power_data.sft_samp_cfg, voltage])

        self._data_list.clear()
        self._data_list['data_list'] = data_list
