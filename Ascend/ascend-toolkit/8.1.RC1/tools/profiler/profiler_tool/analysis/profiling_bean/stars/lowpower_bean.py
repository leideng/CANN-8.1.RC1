#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import struct
from enum import IntEnum
from enum import unique

from msparser.data_struct_size_constant import StructFmt
from profiling_bean.struct_info.struct_decoder import StructDecoder


@unique
class LowPowerType(IntEnum):
    """
    DataType enum for ge graph operator data
    """
    SAMPLE_CNT = 0
    TEMP_INFO = 1
    PWR_INFO = 2
    VF_INFO = 3


class LowPowerBean(StructDecoder):
    """
    bean for lowpower sample data
    """

    DATA_TYPE_DICT = {
        '1-0': 'tem_of_ai_core', '1-1': 'tem_of_hbm', '1-2': 'tem_of_hbm_granularity', '1-3': 'tem_of_3ds_ram',
        '1-4': 'tem_of_cpu', '1-5': 'tem_of_peripherals', '1-6': 'tem_of_l2_buff', '2-0': 'aic_current_dpm',
        '2-1': 'power_consumption_dpm', '2-2': 'aic_current_sd5003', '2-3': 'power_consumption_sd5003',
        '2-4': 'pwr2prof_dat_imon', '3-0': 'pmc2prof_freq_prof', '3-1': 'tem_warn_cnt0', '3-2': 'tem_warn_cnt1',
        '3-3': 'tem_warn_cnt2', '3-4': 'tem_warn_cnt3', '3-5': 'cos_warn_cnt0', '3-6': 'cos_warn_cnt1',
        '3-7': 'cos_warn_cnt2', '3-8': 'cos_warn_cnt3', '3-9': 'io_warn_cnt0', '3-10': 'io_warn_cnt1',
        '3-11': 'io_warn_cnt2', '3-12': 'io_warn_cnt3', '3-13': 'epd_warn', '3-15': 'sft_samp_cfg',
    }

    TEMP_SUM_WITH_THREE_BITS = 8.0
    TEMP_SUM_WITH_FOUR_BITS = 16.0
    TWENTY_SIX_BITS_OF_FOUR_BYTES = 67108863

    def __init__(self: any) -> None:
        self._lp_info = {}

    @property
    def tem_of_ai_core(self: any) -> int:
        """
        for tem_of_ai_core
        """
        return self._get_temp_info('tem_of_ai_core')

    @property
    def tem_of_hbm(self: any) -> float:
        """
        for tem_of_hbm
        """
        return self._get_temp_info('tem_of_hbm')

    @property
    def tem_of_hbm_granularity(self: any) -> float:
        """
        for tem_of_hbm_granularity
        """
        return self._get_temp_info('tem_of_hbm_granularity')

    @property
    def tem_of_3ds_ram(self: any) -> float:
        """
        for tem_of_3ds_ram
        """
        return self._get_temp_info('tem_of_3ds_ram')

    @property
    def tem_of_cpu(self: any) -> float:
        """
        for tem_of_cpu
        """
        return self._get_temp_info('tem_of_cpu')

    @property
    def tem_of_peripherals(self: any) -> float:
        """
        for tem_of_peripherals
        """
        return self._get_temp_info('tem_of_peripherals')

    @property
    def tem_of_l2_buff(self: any) -> float:
        """
        for tem_of_l2_buff
        """
        return self._get_temp_info('tem_of_l2_buff')

    @property
    def aic_current_dpm(self: any) -> float:
        """
        for aic_current_dpm
        """
        return self._lp_info.get('aic_current_dpm', 0.0)

    @property
    def power_consumption_dpm(self: any) -> float:
        """
        for power_consumption_dpm
        """
        return self._lp_info.get('power_consumption_dpm', 0.0)

    @property
    def aic_current_sd5003(self: any) -> int:
        """
        for aic_current_sd5003
        """
        return self._lp_info.get('aic_current_sd5003', 0)

    @property
    def power_consumption_sd5003(self: any) -> float:
        """
        for power_consumption_sd5003
        """
        if not self._lp_info.get('pwr_samp_cnt'):
            return 0
        return self._lp_info.get('power_consumption_sd5003', 0.0) / self._lp_info.get('pwr_samp_cnt') / \
            self.TEMP_SUM_WITH_FOUR_BITS

    @property
    def pwr2prof_dat_imon(self: any) -> float:
        """
        for pwr2prof_dat_imon
        """
        return self._lp_info.get('pwr2prof_dat_imon', 0.0)

    @property
    def pmc2prof_freq_prof(self: any) -> float:
        """
        for pmc2prof_freq_prof
        """
        if not self._lp_info.get('vf_samp_cnt'):
            return 0
        return self._lp_info.get('pmc2prof_freq_prof', 0.0) / self._lp_info.get('vf_samp_cnt') / \
            self.TEMP_SUM_WITH_THREE_BITS

    @property
    def tem_warn_cnt0(self: any) -> int:
        """
        for tem_warn_cnt0
        """
        return self._lp_info.get('tem_warn_cnt0', 0)

    @property
    def tem_warn_cnt1(self: any) -> int:
        """
        for tem_warn_cnt1
        """
        return self._lp_info.get('tem_warn_cnt1', 0)

    @property
    def tem_warn_cnt2(self: any) -> int:
        """
        for tem_warn_cnt2
        """
        return self._lp_info.get('tem_warn_cnt2', 0)

    @property
    def tem_warn_cnt3(self: any) -> int:
        """
        for tem_warn_cnt3
        """
        return self._lp_info.get('tem_warn_cnt3', 0)

    @property
    def cos_warn_cnt0(self: any) -> int:
        """
        for cos_warn_cnt0
        """
        return self._lp_info.get('cos_warn_cnt0', 0)

    @property
    def cos_warn_cnt1(self: any) -> int:
        """
        for cos_warn_cnt1
        """
        return self._lp_info.get('cos_warn_cnt1', 0)

    @property
    def cos_warn_cnt2(self: any) -> int:
        """
        for cos_warn_cnt2
        """
        return self._lp_info.get('cos_warn_cnt2', 0)

    @property
    def cos_warn_cnt3(self: any) -> int:
        """
        for cos_warn_cnt3
        """
        return self._lp_info.get('cos_warn_cnt3', 0)

    @property
    def io_warn_cnt0(self: any) -> int:
        """
        for io_warn_cnt0
        """
        return self._lp_info.get('io_warn_cnt0', 0)

    @property
    def io_warn_cnt1(self: any) -> int:
        """
        for io_warn_cnt1
        """
        return self._lp_info.get('io_warn_cnt1', 0)

    @property
    def io_warn_cnt2(self: any) -> int:
        """
        for io_warn_cnt2
        """
        return self._lp_info.get('io_warn_cnt2', 0)

    @property
    def io_warn_cnt3(self: any) -> int:
        """
        for io_warn_cnt3
        """
        return self._lp_info.get('io_warn_cnt3', 0)

    @property
    def epd_warn(self: any) -> int:
        """
        for epd_warn
        """
        return self._lp_info.get('epd_warn', 0)

    @property
    def sft_samp_cfg(self: any) -> float:
        """
        for sft_samp_cfg
        """
        return self._lp_info.get('sft_samp_cfg', 0.0)

    @property
    def vf_samp_cnt(self: any) -> int:
        """
        for vf_samp_cnt
        """
        return self._lp_info.get('vf_samp_cnt', 0)

    @property
    def pwr_samp_cnt(self: any) -> int:
        """
        for pwr_samp_cnt
        """
        return self._lp_info.get('pwr_samp_cnt', 0)

    @property
    def temp_samp_cnt(self: any) -> int:
        """
        for temp_samp_cnt
        """
        return self._lp_info.get('temp_samp_cnt', 0)

    @staticmethod
    def get_low_power_id(data) -> int:
        """
        get temp_type 2-6 bits of 32bits
        """
        return data >> 26 & 15

    @staticmethod
    def get_low_power_data_header(data: int) -> LowPowerType:
        """
        get highest 2bits of 32bits
        """
        return LowPowerType(data >> 30)

    def decode(self: any, binary_data: bytes, additional_fmt: str = "") -> any:
        """
        decode binary dato to class
        :param binary_data:
        :param additional_fmt:
        :return:
        """
        header_handle_dict = {
            LowPowerType.SAMPLE_CNT: self._calculate_sample_data,
            LowPowerType.TEMP_INFO: self._calculate_temp_data,
            LowPowerType.PWR_INFO: self._calculate_pwr_data,
            LowPowerType.VF_INFO: self._calculate_vf_data
        }
        fmt = StructFmt.BYTE_ORDER_CHAR + self.get_fmt() + additional_fmt
        struct_data = struct.unpack_from(fmt, binary_data)
        for data in struct_data[4:]:
            header = self.get_low_power_data_header(data)
            header_handle_dict.get(header)(data)

    def update_lp_info(self, data, low_power_type):
        """
        The last 26 bits of the 32-bit pwr are used.
        """
        data_type = self.get_low_power_id(data)
        data_key = "{}-{}".format(low_power_type.value, data_type)
        if self.DATA_TYPE_DICT.get(data_key, '') in self._lp_info:
            raise ValueError
        self._lp_info[self.DATA_TYPE_DICT.get(data_key)] = data & self.TWENTY_SIX_BITS_OF_FOUR_BYTES

    def _calculate_sample_data(self, data):
        vf_samp_cnt = data >> 20 & 1023  # vf_samp_cnt is 2-12bits of 32bits
        pwr_samp_cnt = data >> 10 & 1023  # pwr_samp_cnt is 12-22bits of 32bits
        temp_samp_cnt = data & 1023  # temp_samp_cnt is 22-32bits of 32bits
        if not all([vf_samp_cnt, pwr_samp_cnt, temp_samp_cnt]):
            return
        self._lp_info['vf_samp_cnt'] = vf_samp_cnt
        self._lp_info['pwr_samp_cnt'] = pwr_samp_cnt
        self._lp_info['temp_samp_cnt'] = temp_samp_cnt

    def _calculate_temp_data(self, data):
        self.update_lp_info(data, LowPowerType.TEMP_INFO)

    def _calculate_pwr_data(self, data):
        self.update_lp_info(data, LowPowerType.PWR_INFO)

    def _calculate_vf_data(self, data):
        self.update_lp_info(data, LowPowerType.VF_INFO)

    def _get_temp_info(self, temp_type):
        temp_samp_cnt = self._lp_info.get('temp_samp_cnt')
        if not temp_samp_cnt:
            return 0
        return self._lp_info.get(temp_type, 0.0) / temp_samp_cnt / self.TEMP_SUM_WITH_THREE_BITS
