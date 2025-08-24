#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import ctypes
from common import RetCode
from common import log_debug, log_error
from common.const import NOT_SUPPORT, UNKNOWN
from drv import LoadSoType

MAX_CHIP_NAME = 32
MODULE_TYPE_AICORE = 4
MODULE_TYPE_VECTOR_CORE = 7
MODULE_TYPE_AICPU = 1
MODULE_TYPE_CCPU = 2
INFO_TYPE_CORE_NUM = 3

DSMI_DEVICE_TYPE_HBM = 2

MODULE_TYPE_SYSTEM = 0
INFO_TYPE_MASTERID = 2

DSMI_ERROR_CORE = {
    1: "the device does not exist.",
    2: "invalid device ID.",
    3: "invalid value",
    4: "invalid handle",
    6: "out of memory",
    7: "inner err",
    8: "para error",
    10: "repeated init",
    11: "not exist",
    13: "device busy",
    14: "device nor resources",
    16: "wait timeout",
    17: "iocrl failed",
    27: "send mesg failed",
    37: "over limit",
    38: "file ops",
    46: "oper not permitted",
    51: "try again",
    58: "memory opt fai",
    86: "partitio not right",
    87: "resource occupied",
    89: "resume",
    91: "bist hw err",
    92: "bist sw err",
    93: "dup config",
    94: "power of fail",
    65534: "not support"
}


class DsmiChipInfoStru(ctypes.Structure):
    _fields_ = [
        ("chip_type", ctypes.c_char * MAX_CHIP_NAME),
        ("chip_name", ctypes.c_char * MAX_CHIP_NAME),
        ("chip_ver", ctypes.c_char * MAX_CHIP_NAME),
    ]


class DsmiAicpuInfoStru(ctypes.Structure):
    _fields_ = [
        ("maxFreq", ctypes.c_int),
        ("curFreq", ctypes.c_int),
        ("aicpuNum", ctypes.c_int),
        ("utilRate", ctypes.c_int * 16)
    ]


class DsmiMemoryInfoStru(ctypes.Structure):
    _fields_ = [
        ("memory_size", ctypes.c_ulonglong),
        ("freq", ctypes.c_uint),
        ("utiliza", ctypes.c_uint),
    ]


class DsmiHBMInfoStru(ctypes.Structure):
    _fields_ = [
        ("memory_size", ctypes.c_ulonglong),
        ("freq", ctypes.c_uint),
        ("memory_usage", ctypes.c_ulonglong),
        ("temp", ctypes.c_int),
        ("bandwith_util_rate", ctypes.c_uint),
    ]


class DsmiPowerInfoStru(ctypes.Structure):
    _fields_ = [
        ("power", ctypes.c_short)
    ]


class AmlCpuInfo(ctypes.Structure):
    _fields_ = [
        ("aicpuCount", ctypes.c_uint16),
        ("ccpuCount", ctypes.c_uint16),
        ("ccpuVoltage", ctypes.c_uint16),
        ("ccpuFrequency", ctypes.c_uint16)
    ]


class AmlAicoreInfo(ctypes.Structure):
    _fields_ = [
        ("aicCount", ctypes.c_uint16),
        ("aicVoltage", ctypes.c_uint16),
        ("aicFrequency", ctypes.c_uint16)
    ]


class AmlBusInfo(ctypes.Structure):
    _fields_ = [
        ("busVoltage", ctypes.c_uint16),
        ("ringFrequency", ctypes.c_uint16),
        ("cpuFrequency", ctypes.c_uint16),
        ("mataFrequency", ctypes.c_uint16),
        ("l2bufferFrequency", ctypes.c_uint16),
    ]


class AmlHbmInfo(ctypes.Structure):
    _fields_ = [
        ("hbmVoltage", ctypes.c_uint16),
        ("hbmFrequency", ctypes.c_uint16)
    ]


class DsmiEccPageStru(ctypes.Structure):
    _fields_ = [
        ("correctedEccErrorsTotal", ctypes.c_int),
        ("ucorrectedEccErrorsTotal", ctypes.c_int),
        ("isolatedSingleBitError", ctypes.c_int),
        ("isolatedDoubleBitError", ctypes.c_int),
    ]


class DeviceInfo:
    def __init__(self):
        self.dsmi_handle = LoadSoType().get_drvdsmi_env_type()
        self.hal_handle = LoadSoType().get_drvhal_env_type()
        self.ascend_ml = LoadSoType().get_ascend_ml()

    @staticmethod
    def check_status(ret, msg="Failed to query data"):
        if ret == 0:
            return True
        msg += ", %s" % DSMI_ERROR_CORE.get(ret)
        log_debug(msg)
        return False

    @staticmethod
    def get_device_info_loop(device_num, func, err):
        for device_id in range(device_num):
            ret = func(device_id)
            if ret and ret != err:
                return ret
        return err

    def get_device_cpu_info(self, device_id):
        if self.ascend_ml == RetCode.FAILED:
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]
        cpu_info = ctypes.pointer(AmlCpuInfo())
        try:
            ret = self.ascend_ml.AmlDeviceGetCpuInfo(device_id, cpu_info)
        except Exception as e:
            log_error(f"get device cpu info fail, error_msg: {e}")
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]

        if not self.check_status(ret, "get device cpu info fail"):
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]

        ai_cpu_c = cpu_info.contents.aicpuCount
        c_cpu_c = cpu_info.contents.ccpuCount
        c_cpu_v = cpu_info.contents.ccpuVoltage
        c_cpu_f = cpu_info.contents.ccpuFrequency

        return [ai_cpu_c, c_cpu_c, c_cpu_v, c_cpu_f]

    def get_device_aic_info(self, device_id):
        if self.ascend_ml == RetCode.FAILED:
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]
        aic_info = ctypes.pointer(AmlAicoreInfo())
        try:
            ret = self.ascend_ml.AmlDeviceGetAicoreInfo(device_id, aic_info)
        except Exception as e:
            log_error(f"get device aic info fail, error_msg: {e}")
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]

        if not self.check_status(ret, "get device aic info fail"):
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]

        aic_c = aic_info.contents.aicCount
        aic_v = aic_info.contents.aicVoltage
        aic_f = aic_info.contents.aicFrequency

        return [aic_c, aic_v, aic_f]

    def get_device_bus_info(self, device_id):
        if self.ascend_ml == RetCode.FAILED:
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]
        bus_info = ctypes.pointer(AmlBusInfo())
        try:
            ret = self.ascend_ml.AmlDeviceGetBusInfo(device_id, bus_info)
        except Exception as e:
            log_error(f"get device bus info fail, error_msg: {e}")
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]

        if not self.check_status(ret, "get device bus info fail"):
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]

        bus_v = bus_info.contents.busVoltage
        ring_f = bus_info.contents.ringFrequency
        cpu_f = bus_info.contents.cpuFrequency
        mate_f = bus_info.contents.mataFrequency
        l2_buf_f = bus_info.contents.l2bufferFrequency

        return [bus_v, ring_f, cpu_f, mate_f, l2_buf_f]

    def get_device_hbm_volt_freq(self, device_id):
        if self.ascend_ml == RetCode.FAILED:
            return [NOT_SUPPORT, NOT_SUPPORT]
        hbm_info = ctypes.pointer(AmlHbmInfo())
        try:
            ret = self.ascend_ml.AmlDeviceGetHbmInfo(device_id, hbm_info)
        except Exception as e:
            log_error(f"get device hbm volt & freq fail, error_msg: {e}")
            return [NOT_SUPPORT, NOT_SUPPORT]

        if not self.check_status(ret, "get device hbm volt & freq fail"):
            return [NOT_SUPPORT, NOT_SUPPORT]

        hbm_v = hbm_info.contents.hbmVoltage
        hbm_f = hbm_info.contents.hbmFrequency

        return [hbm_v, hbm_f]

    def get_device_count(self):
        """Obtains the number of devices."""
        p_device_count = ctypes.pointer(ctypes.c_int())
        try:
            ret = self.dsmi_handle.dsmi_get_device_count(p_device_count)
        except AttributeError:
            return 0
        if not self.check_status(ret, "get device count fail"):
            return 0
        return p_device_count.contents.value

    def get_chip_info(self, device_id):
        """Obtains device chip information."""
        p_chip_info = ctypes.pointer(DsmiChipInfoStru())
        try:
            ret = self.hal_handle.halGetChipInfo(device_id, p_chip_info)
        except AttributeError:
            return UNKNOWN
        if not self.check_status(ret, "get chip info fail"):
            return UNKNOWN
        chip_type = p_chip_info.contents.chip_type.decode()
        chip_name = p_chip_info.contents.chip_name.decode()
        chip_ver = p_chip_info.contents.chip_ver.decode()
        return " ".join([chip_type, chip_name, chip_ver])

    def get_aicpu_count(self, device_id):
        """Obtains device aicpu information."""
        module_type_aicpu = ctypes.c_int(MODULE_TYPE_AICPU)
        type_core_num = ctypes.c_int(INFO_TYPE_CORE_NUM)
        p_aicpu_count = ctypes.pointer(ctypes.c_int())
        try:
            ret = self.hal_handle.halGetDeviceInfo(device_id, module_type_aicpu, type_core_num, p_aicpu_count)
        except AttributeError:
            return NOT_SUPPORT
        if not self.check_status(ret, "get aicpu count fail"):
            return NOT_SUPPORT
        return p_aicpu_count.contents.value

    def get_ccpu_count(self, device_id):
        module_type_ccpu = ctypes.c_int(MODULE_TYPE_CCPU)
        type_core_num = ctypes.c_int(INFO_TYPE_CORE_NUM)
        p_ccpu_count = ctypes.pointer(ctypes.c_int())
        try:
            ret = self.hal_handle.halGetDeviceInfo(device_id, module_type_ccpu, type_core_num, p_ccpu_count)
        except AttributeError:
            return NOT_SUPPORT

        if not self.check_status(ret, "get device cpu count fail"):
            return NOT_SUPPORT

        return p_ccpu_count.contents.value

    def get_device_health(self, device_id):
        """Obtaining the device health status"""
        device_health_status = {0: "Healthy", 1: "Warning", 2: "Alarm", 3: "Critical"}

        p_health_count = ctypes.pointer(ctypes.c_int())
        try:
            ret = self.dsmi_handle.dsmi_get_device_health(device_id, p_health_count)
        except AttributeError:
            return UNKNOWN
        if not self.check_status(ret, "get device health fail"):
            return UNKNOWN
        device_health_count = p_health_count.contents.value
        if device_health_count in device_health_status.keys():
            return device_health_status.get(device_health_count)
        return UNKNOWN

    def get_device_errorcode(self, device_id):
        """Obtaining the device error code"""
        error_list = list()
        pyarray = [0]
        perrorcode = (ctypes.c_uint * 128)(*pyarray)
        perrorinfo = (ctypes.c_char * 255)(*pyarray)
        p_error_count = ctypes.pointer(ctypes.c_int())
        try:
            ret = self.dsmi_handle.dsmi_get_device_errorcode(device_id, p_error_count, perrorcode)
        except AttributeError:
            return error_list
        if not self.check_status(ret, "get errorcode fail"):
            return error_list
        error_code = p_error_count.contents.value
        for i in range(error_code):
            ret = self.dsmi_handle.dsmi_query_errorstring(device_id, perrorcode[i], perrorinfo, 256)
            error_info = ""
            for info in perrorinfo:
                error_info += str(info.decode())
            if not self.check_status(ret, "get errorcode fail"):
                error_list.append([hex(perrorcode[i]), "NA"])
            else:
                error_list.append([hex(perrorcode[i]), str(error_info.strip("\x00"))])
        return error_list

    def get_aicore_count(self, device_id):
        module_type_aicore = ctypes.c_int(MODULE_TYPE_AICORE)
        type_core_num = ctypes.c_int(INFO_TYPE_CORE_NUM)
        p_aicore_count = ctypes.pointer(ctypes.c_int())
        try:
            ret = self.hal_handle.halGetDeviceInfo(device_id, module_type_aicore, type_core_num, p_aicore_count)
        except AttributeError:
            return NOT_SUPPORT

        if not self.check_status(ret, "git device aicore count fail"):
            return NOT_SUPPORT

        return p_aicore_count.contents.value

    def get_veccore_count(self, device_id):
        type_vector_core = ctypes.c_int(MODULE_TYPE_VECTOR_CORE)
        type_core_num = ctypes.c_int(INFO_TYPE_CORE_NUM)
        p_veccore_count = ctypes.pointer(ctypes.c_int())
        try:
            ret = self.hal_handle.halGetDeviceInfo(device_id, type_vector_core, type_core_num, p_veccore_count)
        except AttributeError:
            return NOT_SUPPORT

        if not self.check_status(ret, "git device veccore count fail"):
            return NOT_SUPPORT

        return p_veccore_count.contents.value

    def get_device_power(self, device_id):
        p_power_info = ctypes.pointer(DsmiPowerInfoStru())
        try:
            ret = self.dsmi_handle.dsmi_get_device_power_info(device_id, p_power_info)
        except AttributeError:
            return NOT_SUPPORT
        if not self.check_status(ret, "get power info fail"):
            return NOT_SUPPORT
        return p_power_info.contents.power

    def get_device_temperature(self, device_id):
        p_temperature = ctypes.pointer(ctypes.c_int())
        try:
            ret = self.dsmi_handle.dsmi_get_device_temperature(device_id, p_temperature)
        except AttributeError:
            return NOT_SUPPORT
        if not self.check_status(ret, "get temperature info fail"):
            return NOT_SUPPORT
        return p_temperature.contents.value

    def get_device_frequency(self, device_id, device_type):
        """
        device_id: device id
        device_type:
            1  memory Frequency
            2  Control CPU Frequency
            3  HBM
            4  AI CORE Current frequency
            5  AI CORE rated frequency
            6  Vector CORE Current frequency
        """
        p_frequency = ctypes.pointer(ctypes.c_int())
        try:
            ret = self.dsmi_handle.dsmi_get_device_frequency(device_id, device_type, p_frequency)
        except AttributeError:
            return NOT_SUPPORT
        if not self.check_status(ret, "get frequency info fail"):
            return NOT_SUPPORT
        return p_frequency.contents.value

    def get_device_voltage(self, device_id):
        """
        device_id device_id
        return voltage pvoltage * 0.01V  *1000 MV
        """
        p_voltage = ctypes.pointer(ctypes.c_uint())
        try:
            ret = self.dsmi_handle.dsmi_get_device_voltage(device_id, p_voltage)
        except AttributeError:
            return NOT_SUPPORT
        if not self.check_status(ret, "get voltage info fail"):
            return NOT_SUPPORT
        return round(p_voltage.contents.value * 0.01 * 1000, 2)

    def get_device_utilization_rate(self, device_id, device_type):
        """
        device_id: Device Id
        device_type:
            1: memory 2: Ai Core 3: Ai Cpu 4: Control CPU
            5: memory Bandwidth  6: HBM 8: DDR 10: HBM Bandwidth
            12: Vector Core
        return:
            utilization
        """

        p_utilization = ctypes.pointer(ctypes.c_uint())
        try:
            ret = self.dsmi_handle.dsmi_get_device_utilization_rate(device_id, device_type, p_utilization)
        except AttributeError:
            return NOT_SUPPORT
        if not self.check_status(ret, "get utilization info fail"):
            return NOT_SUPPORT
        return p_utilization.contents.value

    def get_device_memory_info(self, device_id):
        p_memory_info = ctypes.pointer(DsmiMemoryInfoStru())
        try:
            ret = self.dsmi_handle.dsmi_get_memory_info(device_id, p_memory_info)
        except AttributeError:
            return NOT_SUPPORT, NOT_SUPPORT
        if not self.check_status(ret, "get ddr memory info fail"):
            return NOT_SUPPORT, NOT_SUPPORT

        if "310 " in self.get_chip_info(device_id):
            memory_size = p_memory_info.contents.memory_size
        else:
            memory_size = p_memory_info.contents.memory_size // 1024
        utiliza = p_memory_info.contents.utiliza
        return memory_size, round(memory_size * utiliza / 100, 2)

    def get_device_hbm_info(self, device_id):
        p_memory_info = ctypes.pointer(DsmiHBMInfoStru())
        try:
            ret = self.dsmi_handle.dsmi_get_hbm_info(device_id, p_memory_info)
        except AttributeError:
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]
        if not self.check_status(ret, "get hbm memory info fail"):
            return [NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT, NOT_SUPPORT]

        memory_size = p_memory_info.contents.memory_size // 1024
        usage = p_memory_info.contents.memory_usage
        bandwidth = p_memory_info.contents.bandwith_util_rate
        temp = p_memory_info.contents.temp
        return [memory_size, round(usage / 1024, 2), temp, bandwidth]

    def get_ecc_isolated_page(self, device_id):
        p_device_ecc_info = ctypes.pointer(DsmiEccPageStru())
        dsmi_device_type_hbm = ctypes.c_int(DSMI_DEVICE_TYPE_HBM)
        try:
            ret = self.dsmi_handle.dsmi_get_total_ecc_isolated_pages_info(device_id,
                                                                          dsmi_device_type_hbm, p_device_ecc_info)
        except AttributeError:
            return "-"
        if not self.check_status(ret, "uncorrected ecc errors aggregate total fail"):
            return "-"
        return p_device_ecc_info.contents.ucorrectedEccErrorsTotal

    def clear_ecc_isolated(self, device_id):
        """
        Clears historical ECC error statistics and isolation page information.
        """
        try:
            ret = self.dsmi_handle.dsmi_clear_ecc_isolated_statistics_info(device_id)
            if ret != 0:
                log_error("clear ecc isolated failed")
        except AttributeError as e:
            log_error(e)

    def get_phyid_from_logicid(self, device_id):
        phyid = ctypes.pointer(ctypes.c_uint32(0))
        try:
            ret = self.hal_handle.drvDeviceGetPhyIdByIndex(ctypes.c_int32(device_id), phyid)
        except AttributeError as e:
            log_error(f"get PhyId by device:{device_id} fail, error_msg: {e}")
            return RetCode.FAILED

        if ret != 0:
            log_error(f"get PhyId by device:{device_id} fail")
            return RetCode.FAILED

        return phyid.contents.value

    def get_masterid_from_phyid(self, phyid):
        master_id = ctypes.pointer(ctypes.c_int64(0))
        try:
            ret = self.hal_handle.halGetDeviceInfo(
                ctypes.c_uint32(phyid),
                ctypes.c_int32(MODULE_TYPE_SYSTEM),
                ctypes.c_int32(INFO_TYPE_MASTERID),
                master_id
            )
        except AttributeError as e:
            log_error(f"get MasterId by PhyId:{phyid} fail, error_msg: {e}")
            return RetCode.FAILED

        if ret != 0:
            log_error(f"get MasterId by PhyId:{phyid} fail")
            return RetCode.FAILED

        return master_id.contents.value
