#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from msconfig.meta_config import MetaConfig


class StarsConfig(MetaConfig):
    DATA = {
        'AcsqTaskParser': [
            ('fmt', '000000, 000001'),
            ('db', 'soc_log.db'),
            ('table', 'AcsqTask')
        ],
        'FftsLogParser': [
            ('fmt', '100010, 100011'),
            ('db', 'soc_log.db'),
            ('table', 'FftsLog')
        ],
        'SioParser': [
            ('fmt', '011001'),
            ('db', 'sio.db'),
            ('table', 'Sio')
        ],
        'AccPmuParser': [
            ('fmt', '011010'),
            ('db', 'acc_pmu.db'),
            ('table', 'AccPmu')
        ],
        'InterSocParser': [
            ('fmt', '011100'),
            ('db', 'soc_profiler.db'),
            ('table', 'InterSoc')
        ],
        'StarsChipTransParser': [
            ('fmt', '011011'),
            ('db', 'chip_trans.db'),
            ('table', 'PaLinkInfo,PcieInfo')
        ],
        'LowPowerParser': [
            ('fmt', '011101'),
            ('db', 'lowpower.db'),
            ('table', 'LowPower')
        ]
    }
