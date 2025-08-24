#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
DRV Interface
"""
# Standard Packages
import ctypes
import logging
from enum import Enum
from typing import Optional, Tuple


class DsmiErrorCode(Enum):
    DSMI_ERROR_NONE = 0
    DSMI_ERROR_NO_DEVICE = 1
    DSMI_ERROR_INVALID_DEVICE = 2
    DSMI_ERROR_INVALID_HANDLE = 3
    DSMI_ERROR_INNER_ERR = 7
    DSMI_ERROR_PARA_ERROR = 8
    DSMI_ERROR_NOT_EXIST = 11
    DSMI_ERROR_BUSY = 13
    DSMI_ERROR_WAIT_TIMEOUT = 16
    DSMI_ERROR_IOCRL_FAIL = 17
    DSMI_ERROR_SEND_MESG = 27
    DSMI_ERROR_OPER_NOT_PERMITTED = 46
    DSMI_ERROR_TRY_AGAIN = 51
    DSMI_ERROR_MEMORY_OPT_FAIL = 58
    DSMI_ERROR_PARTITION_NOT_RIGHT = 86
    DSMI_ERROR_RESOURCE_OCCUPIED = 87
    DSMI_ERROR_NOT_SUPPORT = 0xFFFE


class DsmiChipInfoStru(ctypes.Structure):
    _fields_ = [('chip_type', ctypes.c_char * 32),
                ('chip_name', ctypes.c_char * 32),
                ('chip_ver', ctypes.c_char * 32)]

    def get_complete_platform(self) -> str:
        res = self.chip_type + self.chip_name
        return res.decode("UTF-8")

    def get_ver(self) -> str:
        return self.chip_ver.decode("UTF-8")


class DSMIInterface:
    """
    DRV Function Wrappers
    """
    prof_online: dict = {}

    def __init__(self):
        self.dsmidll = ctypes.CDLL("libdrvdsmi_host.so")

    def get_device_count(self) -> int:
        device_count = (ctypes.c_int * 1)()
        self.dsmidll.dsmi_get_device_count.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_count(device_count)
        if self._parse_error(error_code, "dsmi_get_device_count"):
            return 0
        return device_count[0]

    def get_chip_info(self, device_id: int) -> Optional[DsmiChipInfoStru]:
        device_id = ctypes.c_int(device_id)
        result_struct = DsmiChipInfoStru()
        self.dsmidll.dsmi_get_chip_info.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_chip_info(device_id, ctypes.c_void_p(ctypes.addressof(result_struct)))
        if self._parse_error(error_code, "dsmi_get_chip_info"):
            return None
        return result_struct

    @staticmethod
    def _parse_error(error_code: int, function_name: str, allow_positive=False) -> bool:
        if error_code != 0:
            if allow_positive and error_code > 0:
                logging.debug("DRV API Call %s() Success with return code %d" % (function_name, error_code))
            else:
                try:
                    logging.error(f"DSMI API Call {function_name} failed: {DsmiErrorCode(error_code).name}")
                    return True
                except ValueError:
                    pass
                logging.error(f"DSMI API Call {function_name} failed with unknown code: {error_code}")
                return True
        else:
            logging.debug("DSMI API Call %s() Success" % function_name)
        return False
