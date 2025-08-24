#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys

from common import run_command
from common import get_project_conf, get_ascend_home
from common import DeviceInfo
from common import FileOperate as f
from common import log_error
from common.const import NONE, MEMORY_FREQUENCY, HBM_FREQUENCY, CONTROL_CPU_FREQUENCY, AI_CORE_FREQUENCY, CannPkg
from common.const import AI_CORE_USE, AI_CPU_USE, CONTROL_CPU_USE, MEM_BANDWIDTH_USE, NOT_SUPPORT, MAX_CHAR_LINE
from common.const import UNKNOWN
from params import ParamDict
from view import generate_report

CPU_TABLE_TITLE = " CPU Information "
AIC_TABLE_TITLE = " AI Core Information "
BUS_TABLE_TITLE = " Bus Information "
MEM_TABLE_TITLE = " Memory Information "

HOST_VERSION = " Host Version "
DEVICE_VERSION = " Device Version "

PCIE_INFO = " PCIe Info "

LSPCI_GREP_VERSION = "lspci | grep -E 'd100|d500|d801|d802'"
GET_COUNT = "wc -l"


class AsysInfo:
    def __init__(self):
        self.device_info = DeviceInfo()
        self.device_num = self.device_info.get_device_count()
        self.output_root_path = ParamDict().asys_output_timestamp_dir

    @staticmethod
    def __get_pcie_info(table_data):
        pcie_info_query_cmds = {
            "PCIe Dev Count": f"{LSPCI_GREP_VERSION} | {GET_COUNT}",
            "PCIe Dev Count(normal)": f"{LSPCI_GREP_VERSION} | {GET_COUNT}",
            "PCIe Dev Count(abnormal)": f"{LSPCI_GREP_VERSION} | grep 'rev ff' | {GET_COUNT}",
        }
        for query_name, query_cmds in pcie_info_query_cmds.items():
            count = run_command(query_cmds)
            if not count.isdigit() or (query_name == "PCIe Dev Count" and count == "0"):
                table_data[PCIE_INFO] = []
                break
            table_data[PCIE_INFO].append([query_name, count])
        if table_data[PCIE_INFO]:
            table_data[PCIE_INFO][1][1] = int(table_data[PCIE_INFO][0][1]) \
                                              - int(table_data[PCIE_INFO][2][1])
        else:
            del table_data[PCIE_INFO]

    def __get_hardware_info(self, write_file=False):
        """
        return hardware info report
        """
        table_data = {
            " Host Info ": [],
            " Device Info ": [],
            PCIE_INFO: []
        }
        host_info_query_cmds = {
            "Cpu Info": "cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq",
            "Cpu Physical Count": "lscpu | grep 'Socket(s):' | cut -f2 -d: | uniq",  # arm cpuinfo no 'physical id'
            "Cpu Logical Count": "cat /proc/cpuinfo| grep 'processor' | wc -l",
            "Memory Total Size": "cat /proc/meminfo | sed -n '1p' | awk 'NR=2{print $2, $3}'",
            "Disk Total Size": """df -k / |sed -n '2p' | awk 'NR=2{printf $2}END{print " kB"}'""",
        }
        for query_name, query_cmds in host_info_query_cmds.items():
            host_info = run_command(query_cmds)
            table_data[" Host Info "].append([query_name, host_info])

        ccpu_count = self.device_info.get_device_info_loop(
            self.device_num, self.device_info.get_ccpu_count, NOT_SUPPORT
        )
        aicpu_count = self.device_info.get_device_info_loop(
            self.device_num, self.device_info.get_aicpu_count, NOT_SUPPORT
        )
        aicore_count = self.device_info.get_device_info_loop(
            self.device_num, self.device_info.get_aicore_count, NOT_SUPPORT
        )
        vector_count = self.device_info.get_device_info_loop(
            self.device_num, self.device_info.get_veccore_count, NOT_SUPPORT
        )

        device_info_query_cmds = {
            "NPU Count": self.device_num,
            "Chip Info": self.device_info.get_device_info_loop(
                self.device_num, self.device_info.get_chip_info, UNKNOWN
            ),
            "Control CPU Count": str(ccpu_count * self.device_num) + f" ({ccpu_count} * {self.device_num})",
            "AI CPU Count": str(aicpu_count * self.device_num) + f" ({aicpu_count} * {self.device_num})",
            "AI Core Count": str(aicore_count * self.device_num) + f" ({aicore_count} * {self.device_num})",
            "AI Vector Count": str(vector_count * self.device_num) + f" ({vector_count} * {self.device_num})",
        }
        for query_name, query_cmds in device_info_query_cmds.items():
            if isinstance(query_cmds, str) and NOT_SUPPORT in query_cmds:
                query_cmds = NOT_SUPPORT
            table_data[" Device Info "].append([query_name, query_cmds])
        # PCIe Info
        self.__get_pcie_info(table_data)
        table_header = [[f"Group of {self.device_num} Device", "INFORMATION"]]
        table_string = generate_report(table_header, table_data)
        if write_file:
            hardware_file = os.path.join(self.output_root_path, "hardware_info.txt")
            f.write_file(hardware_file, table_string)
        else:
            sys.stdout.write(table_string)

    @staticmethod
    def __software_set_env(table_data):
        env_info = []
        envs = os.environ
        for env_name, env_value in envs.items():
            if env_name == "LS_COLORS" or env_value == "":
                continue
            if len(env_value) > MAX_CHAR_LINE:
                env_info.append([env_name, env_value[:MAX_CHAR_LINE]])
                for i in range(MAX_CHAR_LINE, len(env_value), MAX_CHAR_LINE):
                    env_info.append(["", env_value[i:i + MAX_CHAR_LINE]])
            else:
                env_info.append([env_name, env_value])
        if env_info:
            table_data[" Env Information "] = env_info

    @staticmethod
    def __software_set_dep(table_data):
        dependent_packet = []
        dep_info = f.read_file(os.path.join(get_project_conf(), "dependent_package.csv"))
        for item in dep_info:
            info = run_command(item[1])
            if info == "NONE":
                continue
            dependent_packet.append([item[0], info])
        if dependent_packet:
            table_data[" Dependent Packet "] = dependent_packet

    @staticmethod
    def __table_data_append(table, value):
        if value[1] != NOT_SUPPORT:
            table.append(value)

    @staticmethod
    def __software_set_pkg(table_data):
        grep_version = "| grep Version | awk -v FS='=' '{print $2}'"
        if ParamDict().get_env_type() == "EP":
            install_path = get_ascend_home()
            for pag_name in CannPkg.get_all_pkg_list():
                if pag_name in [CannPkg.firmware, CannPkg.driver]:
                    version = run_command('cat {}/{}/version.info {}'.format(install_path, pag_name, grep_version))
                elif pag_name in [CannPkg.aoe, CannPkg.ncs]:
                    version = run_command('cat {}/latest/tools/{}/version.info {}'.format(install_path, pag_name,
                                                                                          grep_version))
                else:
                    version = run_command('cat {}/latest/{}/version.info {}'.format(install_path, pag_name,
                                                                                    grep_version))
                if version == "":
                    version = "None"
                table_data[DEVICE_VERSION].append([pag_name, version])
        else:
            driver_version = run_command(f"cat /var/davinci/driver/version.info {grep_version}")
            table_data[DEVICE_VERSION].append([CannPkg.driver, driver_version])
            firmware_version = run_command(f"cat /fw/version.info {grep_version}")
            table_data[DEVICE_VERSION].append([CannPkg.firmware, firmware_version])
            runtime_version = run_command(f"cat /usr/local/Ascend/latest/runtime/version.info {grep_version}")
            if not runtime_version or runtime_version == "NONE":
                runtime_version = run_command(f"cat /usr/local/Ascend/runtime/version.info {grep_version}")
            table_data[DEVICE_VERSION].append(["runtime", runtime_version])

    def get_software_info(self, write_file=False):
        """
        return software info report
        """
        table_data = {
            HOST_VERSION: [],
            DEVICE_VERSION: []
        }
        os_version_path = os.sep + "etc/*release"
        table_data[HOST_VERSION].append(["Kernel", run_command('uname -r')])
        os_version = run_command("cat " + os_version_path + """ | grep PRETTY_NAME | awk -v FS='"' '{print $2}'""")
        table_data[HOST_VERSION].append(["OS", os_version])

        self.__software_set_pkg(table_data)
        table_header = [[f"Group of {self.device_num} Device", "INFORMATION"]]
        if write_file:
            self.__software_set_dep(table_data)
            self.__software_set_env(table_data)
            table_string = generate_report(table_header, table_data)
            software_file = os.path.join(self.output_root_path, "software_info.txt")
            f.write_file(software_file, table_string)
        else:
            table_string = generate_report(table_header, table_data)
            sys.stdout.write(table_string)

    def __add_status_cpu_info(self, table_data, device_id):
        cpu_info = self.device_info.get_device_cpu_info(device_id)
        if NOT_SUPPORT in cpu_info:
            ai_cpu_c = self.device_info.get_aicpu_count(device_id)
            c_cpu_c = self.device_info.get_ccpu_count(device_id)
            c_cpu_v = NOT_SUPPORT
            c_cpu_f = self.device_info.get_device_frequency(device_id, CONTROL_CPU_FREQUENCY)
        else:
            ai_cpu_c, c_cpu_c, c_cpu_v, c_cpu_f = cpu_info

        cpu_info_list = []
        self.__table_data_append(cpu_info_list, ["AI CPU Count", ai_cpu_c])
        self.__table_data_append(
            cpu_info_list,
            ["AI CPU Usage (%)", self.device_info.get_device_utilization_rate(device_id, AI_CPU_USE)])
        self.__table_data_append(cpu_info_list, ["Control CPU Count", c_cpu_c])
        self.__table_data_append(
            cpu_info_list,
            ["Control CPU Usage (%)", self.device_info.get_device_utilization_rate(device_id, CONTROL_CPU_USE)])
        self.__table_data_append(cpu_info_list, ["Control CPU Frequency (MHZ)", c_cpu_f])
        self.__table_data_append(cpu_info_list, ["Control CPU Voltage (MV)", c_cpu_v])
        if cpu_info_list:
            table_data[CPU_TABLE_TITLE] = cpu_info_list
        else:
            table_data.pop(CPU_TABLE_TITLE)

    def __add_status_aic_info(self, table_data, device_id):
        aic_info = self.device_info.get_device_aic_info(device_id)
        if NOT_SUPPORT in aic_info:
            aic_c = self.device_info.get_aicore_count(device_id)
            aic_v = self.device_info.get_device_voltage(device_id)
            aic_f = self.device_info.get_device_frequency(device_id, AI_CORE_FREQUENCY)
        else:
            aic_c, aic_v, aic_f = aic_info
        aic_info_list = []
        self.__table_data_append(aic_info_list, ["AI Core Count", aic_c])
        self.__table_data_append(
            aic_info_list,
            ["AI Core Usage (%)", self.device_info.get_device_utilization_rate(device_id, AI_CORE_USE)])
        self.__table_data_append(aic_info_list, ["AI Core Frequency (MHZ)", aic_f])
        self.__table_data_append(aic_info_list, ["AI Core Voltage (MV)", aic_v])
        if aic_info_list:
            table_data[AIC_TABLE_TITLE] = aic_info_list
        else:
            table_data.pop(AIC_TABLE_TITLE)

    def __add_status_bus_info(self, table_data, device_id):
        bus_v, ring_f, cpu_f, mate_f, l2_buf_f = self.device_info.get_device_bus_info(device_id)
        bus_info_list = []
        self.__table_data_append(bus_info_list, ["Bus Voltage (MV)", bus_v])
        self.__table_data_append(bus_info_list, ["Ring Frequency (MHZ)", ring_f])
        self.__table_data_append(bus_info_list, ["CPU Frequency (MHZ)", cpu_f])
        self.__table_data_append(bus_info_list, ["Mata Frequency (MHZ)", mate_f])
        self.__table_data_append(bus_info_list, ["L2buffer Frequency (MHZ)", l2_buf_f])
        if bus_info_list:
            table_data[BUS_TABLE_TITLE] = bus_info_list
        else:
            table_data.pop(BUS_TABLE_TITLE)

    def __add_status_memory_info(self, table_data, device_id):
        ddr_total, ddr_use = self.device_info.get_device_memory_info(device_id)
        hbm_total, hbm_use, _, hbm_bandwidth = self.device_info.get_device_hbm_info(device_id)

        memory_info_list = []
        if ddr_total != NOT_SUPPORT:
            ddr_bandwidth = self.device_info.get_device_utilization_rate(device_id, MEM_BANDWIDTH_USE)
            ddr_frequency = self.device_info.get_device_frequency(device_id, MEMORY_FREQUENCY)
            self.__table_data_append(memory_info_list, ["DDR Total (MB)", ddr_total])
            self.__table_data_append(memory_info_list, ["DDR Used (MB)", ddr_use])
            self.__table_data_append(memory_info_list, ["DDR Bandwidth Usage (%)", ddr_bandwidth])
            self.__table_data_append(memory_info_list, ["DDR Frequency (MHZ)", ddr_frequency])

        if hbm_total != NOT_SUPPORT:
            hbm_v, hbm_f = self.device_info.get_device_hbm_volt_freq(device_id)
            if hbm_f == NOT_SUPPORT:
                hbm_f = self.device_info.get_device_frequency(device_id, HBM_FREQUENCY)
            self.__table_data_append(memory_info_list, ["HBM Total (MB)", hbm_total])
            self.__table_data_append(memory_info_list, ["HBM Used (MB)", hbm_use])
            self.__table_data_append(memory_info_list, ["HBM Bandwidth Usage (%)", hbm_bandwidth])
            self.__table_data_append(memory_info_list, ["HBM Frequency (MHZ)", hbm_f])
            self.__table_data_append(memory_info_list, ["HBM Voltage (MV)", hbm_v])

        if memory_info_list:
            table_data[MEM_TABLE_TITLE] = memory_info_list
        else:
            table_data.pop(MEM_TABLE_TITLE)

    def __get_status_info(self, device_id, write_file=False):
        """
        return status info report
        """
        table_data = {
            NONE: [],
            CPU_TABLE_TITLE: [],
            AIC_TABLE_TITLE: [],
            BUS_TABLE_TITLE: [],
            MEM_TABLE_TITLE: []
        }
        # add public information
        self.__table_data_append(table_data[NONE], ["Chip Name", self.device_info.get_chip_info(device_id)])
        self.__table_data_append(table_data[NONE], ["Power (W)", self.device_info.get_device_power(device_id)])
        self.__table_data_append(table_data[NONE], ["Temperature (C)",
                                                    self.device_info.get_device_temperature(device_id)])
        self.__table_data_append(table_data[NONE], ["health", self.device_info.get_device_health(device_id)])

        # add cpu information
        self.__add_status_cpu_info(table_data=table_data, device_id=device_id)

        # add aicore information
        self.__add_status_aic_info(table_data=table_data, device_id=device_id)

        # add bus information
        self.__add_status_bus_info(table_data=table_data, device_id=device_id)

        # add memory information
        self.__add_status_memory_info(table_data=table_data, device_id=device_id)

        # Delete the key whose value is empty.
        for key, value in table_data.items():
            if not value:
                table_data.pop(key)

        table_header = [[f"Device ID: {device_id}", "INFORMATION"]]
        table_string = generate_report(table_header, table_data)
        if write_file:
            status_file = os.path.join(self.output_root_path, "status_info.txt")
            f.append_write_file(status_file, table_string)
        else:
            sys.stdout.write(table_string)

    def run_info(self):
        run_mode = ParamDict().get_arg('run_mode')
        if run_mode == "hardware":
            self.__get_hardware_info()
        elif run_mode == "software":
            self.get_software_info()
        elif run_mode == "status":
            device_id = ParamDict().get_arg('device_id') if ParamDict().get_arg('device_id') else 0
            if device_id >= self.device_num:
                log_error("'-d' value should be in [0, {}), input {}".format(self.device_num, device_id))
                return False
            self.__get_status_info(device_id)
        return True

    def write_info(self):
        self.__get_hardware_info(write_file=True)
        self.get_software_info(write_file=True)
        for device_id in range(self.device_num):
            self.__get_status_info(device_id, write_file=True)
