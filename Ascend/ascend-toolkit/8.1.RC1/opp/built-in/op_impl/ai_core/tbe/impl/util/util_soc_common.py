#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
util_soc_common
"""
from impl.util.platform_adapter import tbe_platform


def is_support_inf_nan():
    """
    is support inf nan or not

    Parameters
    ----------

    Return
    -------
    is support inf nan
    """
    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    inf_nan_soc_list = ("Ascend910B", "Ascend910_93", "Ascend310B", "BS9SX1A")
    return cur_cce_product in inf_nan_soc_list


def is_v200():
    """
    is v200 or not

    Parameters
    ----------

    Return
    -------
    is v200
    """
    # 1911 is v200, but has a bug, need return false temporarily
    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cur_cce_product in ('Ascend310B', "AS31XM1"):
        return False

    return tbe_platform.api_check_support("tik.vcopy")


def after_v200():
    """
    after v200 or not

    Parameters
    ----------

    Return
    -------
    is v200
    """
    return tbe_platform.api_check_support("tik.vcopy")


def is_v220():
    """
    is v220 or not

    Parameters
    ----------

    Return
    -------
    is v220
    """
    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    return cur_cce_product in ('Ascend910B', 'Ascend910_93')


def support_inf_nan():
    """
    support inf nan mode

    Parameters
    ----------

    Return
    -------
    support inf nan mode
    """
    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    return cur_cce_product not in ('Ascend310', 'Ascend910', 'Ascend610', 'Ascend310P')


def is_v300():
    """
    is v300 or not

    Parameters
    ----------

    Return
    -------
    is v300
    """
    ub_size = tbe_platform.get_soc_spec("UB_SIZE")
    return ub_size == 30720


def is_v310():
    """
    is v310 or not

    Parameters
    ----------

    Return
    -------
    is v310
    """
    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    return cur_cce_product in ("Ascend610Lite", "BS9SX2A", "MC61AM21A")
