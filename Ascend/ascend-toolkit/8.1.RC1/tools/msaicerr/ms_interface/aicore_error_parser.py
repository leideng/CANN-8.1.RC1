#!/usr/bin/env python
# coding=utf-8
"""
Function:
AicoreErrorParser class. This file mainly involves the parse function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import re
import os
import subprocess
import time
import json
import platform
import shutil
from collections import namedtuple
from functools import reduce
import numpy as np
from ms_interface import utils
from ms_interface.tiling_data_parser import TilingDataParser
from ms_interface.constant import Constant
from ms_interface.aic_error_info import AicErrorInfo
from ms_interface.dump_data_parser import DumpDataParser
from ms_interface.single_op_test_frame.utils import shape_utils
from ms_interface.dsmi_interface import DSMIInterface
from ms_interface.single_op_test_frame.single_op_case import SingleOpCase


def _find_runtime_so():
    ld_lib_paths = os.environ['LD_LIBRARY_PATH']
    if not ld_lib_paths:
        raise RuntimeError("LD_LIBRARY_PATH is empty")
    ld_lib_path_list = ld_lib_paths.split(":")
    
    for ld_lib_path in ld_lib_path_list:
        runtime_so_path_tmp = os.path.join(ld_lib_path, "libruntime.so")
        if os.path.exists(runtime_so_path_tmp):
            return runtime_so_path_tmp
    return ""


class AicoreErrorParser:
    """
    The AicoreErrorParser class, parse aicore error info
    """

    def __init__(self: any, collect_path: str, device_id=0) -> None:
        self.collect_path = collect_path
        self.parse_level = 0
        self.ffts_flag = False
        self.device_id = device_id

    @staticmethod
    def _check_args(args_before, args_after) -> bool:
        if sum(args_before) == 0:
            utils.print_warn_log("args_before is empty, maybe GE ubable to print it. Skip args check!")
            return True
        for arg_after in args_after:
            for arg_before in args_before:
                if arg_after == arg_before:
                    return True
            return False

    def get_node_and_kernel_name_l1(self: any) -> list:
        plog_dir = os.path.join(self.collect_path, 'collection', 'plog')
        # 获取kernel_name
        kernel_name_cmd = ['grep', '\[AIC_INFO\] dev_func:', '-inrE', plog_dir]
        kernel_name_regexp = r"dev_func:([a-zA-Z0-9_]{0,})$"
        kernel_name_ret = utils.get_inquire_result(kernel_name_cmd, kernel_name_regexp)
        if not kernel_name_ret:
            utils.print_error_log(f"Failed to get \"[AIC_INFO] dev_func:\" in plog. Cannot run L1 test.")
            return None

        if "__" in kernel_name_ret[0]:
            kernel_name_list = kernel_name_ret[0].split('__')
            kernel_name = kernel_name_list[0]
        else:
            kernel_name = kernel_name_ret[0]

        # 获取node_name、stream_id、task_id
        node_name_cmd = ['grep', '\[AIC_INFO\] node_name:', '-inrE', plog_dir]
        regexp = r".+?node_name:(.*?),.*stream_id:(\d+)\s*.+?\s*task_id:(\d+)\s*"
        result = utils.get_inquire_result(node_name_cmd, regexp)
        if not result:
            utils.print_error_log(f"Failed to get node name in plog. Cannot run L1 test.")
        node_name, stream_id, task_id = result[0]
        node_name = node_name.replace('/', '_').replace('.', '_')
        hash_id_cmd = ['grep', 'hash=', '-inrE', plog_dir]
        hash_id_regexp = r" stream_id=\d+,.*?task_id=\d+,.*?fault kernel_name=.*?,.*?hash=(\d+)"
        hash_id_ret = utils.get_inquire_result(hash_id_cmd, hash_id_regexp)
        if not hash_id_ret:
            utils.print_error_log(f"Cannot get hash id in plog.")
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_SLOG_DATA_ERROR)
        hash_id = list(set(hash_id_ret))
        hash_id = hash_id_ret[0]
        AicoreErrorInfo = namedtuple("AicoreErrorInfo",
                                     ["stream_id", "task_id", "node_name", "kernel_name", "hash_id"])
        # 使用具名元组
        error_info = AicoreErrorInfo(stream_id, task_id, node_name, kernel_name, hash_id)
        return error_info

    def get_kernel_name_l0(self: any, data_name) -> list:
        # 获取kernel_name
        plog_dir = os.path.join(self.collect_path, 'collection', 'plog')
        if not self.ffts_flag:
            kernel_name_cmd = ['grep', 'Aicore kernel execute failed', '-inrE', plog_dir]

            kernel_name_regexp = r" stream_id=(\d+),.*?task_id=(\d+),.*?fault kernel_name=(.*?),.*?" \
                                r"fault kernel info ext=(.*?),.*?hash=(\d+)"
            kernel_name_ret = utils.get_inquire_result(kernel_name_cmd, kernel_name_regexp)

            if kernel_name_ret and kernel_name_ret[0][3] != "none":
                stream_id = kernel_name_ret[0][0]
                task_id = kernel_name_ret[0][1]
                kernel_name = kernel_name_ret[0][3]
                hash_id = kernel_name_ret[0][4]
                node_name = data_name

                AicoreErrorInfo = namedtuple("AicoreErrorInfo",
                                            ["stream_id", "task_id", "node_name", "kernel_name", "hash_id"])
                error_info = AicoreErrorInfo(stream_id, task_id, node_name, kernel_name, hash_id)
                return error_info

            kernel_name_regexp = r" stream_id=(\d+),.*?task_id=(\d+),.*?fault kernel_name=(.*?),.*?hash=(\d+)"
            kernel_name_ret = utils.get_inquire_result(kernel_name_cmd, kernel_name_regexp)
            if not kernel_name_ret:
                utils.print_error_log(f"Failed to get \"Aicore kernel execute failed\" in plog.")
                raise utils.AicErrException(Constant.MS_AICERR_INVALID_SLOG_DATA_ERROR)
            stream_id = kernel_name_ret[0][0]
            task_id = kernel_name_ret[0][1]
            kernel_name = kernel_name_ret[0][2]
            hash_id = kernel_name_ret[0][3]
            utils.print_debug_log(f"AicoreError Found, Stream id: {stream_id}, task_id: {task_id},"
                                 f"kernel_name: {kernel_name}")

            node_name = data_name

            AicoreErrorInfo = namedtuple("AicoreErrorInfo",
                                        ["stream_id", "task_id", "node_name", "kernel_name", "hash_id"])

            # 使用具名元组
            error_info = AicoreErrorInfo(stream_id, task_id, node_name, kernel_name, hash_id)
            return error_info
        else:
            kernel_name_cmd = ['grep', 'fftsplus task execute failed', '-inrE', plog_dir]

            kernel_name_regexp = r" stream_id=(\d+),.*?task_id=(\d+),.*?fault kernel_name=(.*?),.*?hash=(\d+)"
            kernel_name_ret = utils.get_inquire_result(kernel_name_cmd, kernel_name_regexp)
            if not kernel_name_ret:
                utils.print_error_log(f"Failed to get \"fftsplus task execute failed\" in plog.")
                raise utils.AicErrException(Constant.MS_AICERR_INVALID_SLOG_DATA_ERROR)
            stream_id = kernel_name_ret[0][0]
            task_id = kernel_name_ret[0][1]
            kernel_name = kernel_name_ret[0][2]
            hash_id = kernel_name_ret[0][3]
            utils.print_debug_log(f"AicoreError Found, Stream id: {stream_id}, task_id: {task_id},"
                                f"kernel_name: {kernel_name}")

            node_name = data_name

            AicoreErrorInfo = namedtuple("AicoreErrorInfo",
                                        ["stream_id", "task_id", "node_name", "kernel_name", "hash_id"])

            # 使用具名元组
            error_info = AicoreErrorInfo(stream_id, task_id, node_name, kernel_name, hash_id)
            return error_info

    def _get_node_and_kernel_name(self: any, data_name) -> list:
        if self.parse_level == 1:
            return self.get_node_and_kernel_name_l1()
        else:
            return self.get_kernel_name_l0(data_name)

    def check_plog_info(self: any):
        find_path_cmd = ['grep', "\[AIC_INFO\] dev_func:", '-inrE', self.collect_path]
        find_path_regexp = r"(/[_\-/0-9a-zA-Z.]{1,}.[log|txt]):"
        plog_path_ret = utils.get_inquire_result(find_path_cmd, find_path_regexp)

        ffts_check_path_cmd = ['grep',
                               'fftsplus task execute failed',
                               '-inrE', self.collect_path]
        ffts_check_path_regexp = r"(/[_\-/0-9a-zA-Z.]{1,}.[log|txt]):"
        ffts_check_path_ret = utils.get_inquire_result(ffts_check_path_cmd, ffts_check_path_regexp)
        if ffts_check_path_ret:
            self.ffts_flag = True
        if plog_path_ret:
            self.parse_level = 1
        else:
            self.parse_level = 0

    @staticmethod
    def _get_extra_info(aic_error):
        '''
         为兼容原device log框架生成一个extra info参数
        :param aic_error: 正则匹配结果
        :return: 返回的extra info
        '''

        result = "extra info:\n"
        result += "IFU_ERR_INFO={}\n".format(aic_error[8])
        result += "CCU_ERR_INFO={}\n".format(aic_error[9])
        result += "BIU_ERR_INFO={}\n".format(aic_error[11])
        result += "CUBE_ERR_INFO={}\n".format(aic_error[10])
        result += "MTE_ERR_INFO={}\n".format(aic_error[7])
        result += "VEC_ERR_INFO={}\n".format(aic_error[6])
        return result

    def _get_v300_error_code(self: any) -> list:
        plog_path = os.path.join(self.collect_path, "collection", "plog")
        cmd = ['grep', 'The extend info: errcode:', '-nr', plog_path]
        regexp = r"\(([0-9xa-eA-E]+),\s*([0-9xa-eA-E]+),\s*([0-9xa-eA-E]+)\)"
        ret = utils.get_inquire_result(cmd, regexp)
        new_code = 0
        if ret:
            code0, code1, code2 = ret[0]
            code0_int = utils.get_hexstr_value(code0)
            code1_int = utils.get_hexstr_value(code1)
            code1_int = code1_int << 64
            code2_int = utils.get_hexstr_value(code2)
            code2_int = (((code2_int >> 32) << 17) & (code2_int & 0x1FFFF)) << 128
            new_code = code0_int | code1_int | code2_int
        return str(hex(new_code))

    def _get_kernel_and_json_file(self: any, kernel_name: str, tiling_key: str):
        kernel_path = os.path.join(self.collect_path, "collection", "compile")
        kernel_name = kernel_name.replace("_mix_aic", "").replace("_mix_aiv", "")
        if os.path.exists(os.path.join(kernel_path, kernel_name + ".o")):
            bin_file = os.path.join(kernel_path, kernel_name + ".o")
            json_file = os.path.join(kernel_path, kernel_name + ".json")
            cce_file = os.path.join(kernel_path, kernel_name + "_" + str(tiling_key) + ".cce")
            if not os.path.exists(cce_file) :
                cce_file = os.path.join(kernel_path, kernel_name + ".cce")
            KernelFile = namedtuple("KernelFile",
                                    ["bin_file", "json_file", "cce_file"])
            kernel_file = KernelFile(bin_file, json_file, cce_file)
            return kernel_file

        find_path_cmd = ['grep', kernel_name, '-inrE', self.collect_path]
        regexp = r"([_\-/0-9a-zA-Z.]{1,}\.json|[_\-/0-9a-zA-Z.]{1,}\.o|[_\-/0-9a-zA-Z.]{1,}\.cce)"
        kernel_file_list = utils.get_inquire_result(find_path_cmd, regexp)
        if not kernel_file_list:
            utils.print_error_log(f"The {kernel_name}.o or {kernel_name}.json cannot be found in {self.collect_path}.")
            return None

        bin_file = ""
        json_file = ""
        cce_file = ""
        for file_name in kernel_file_list:
            if (not os.path.exists(file_name)) or file_name.endswith("_loc.json") or (not file_name.endswith(".json")):
                continue
            json_file = os.path.join(kernel_path, file_name)
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                if not json_data:
                    continue
                bin_file = json_data.get("binFileName")
                if not bin_file:
                    continue
                bin_file = os.path.join(kernel_path, bin_file + ".o")
                json_file = os.path.join(kernel_path, bin_file + ".json")
                if not os.path.exists(bin_file) or not os.path.exists(json_file):
                    continue
                else:
                    break
        if os.path.exists(bin_file) and os.path.exists(json_file):
            utils.print_inf_log(f"kernel_file {bin_file}, json_file: {json_file} found.")

        bin_file_backup = ""
        json_file_backup = ""
        cce_file_backup = ""
        for file_name in kernel_file_list:
            if not os.path.exists(file_name):
                continue
            if file_name.endswith(".o"):
                bin_file_backup = file_name
            elif file_name.endswith(".json") and (not file_name.endswith("_loc.json")):
                json_file_backup = file_name
            elif file_name.endswith(".cce"):
                tiling_key = "_" + str(tiling_key) + "."
                if tiling_key in file_name:
                    cce_file = file_name
                cce_file_backup = file_name
        if not os.path.exists(bin_file):
            bin_file = bin_file_backup
        if not os.path.exists(json_file):
            json_file = json_file_backup
        if not os.path.exists(cce_file):
            cce_file = cce_file_backup

        KernelFile = namedtuple("KernelFile",
                                ["bin_file", "json_file", "cce_file"])

        # 使用具名元组
        kernel_file = KernelFile(bin_file, json_file, cce_file)
        return kernel_file

    @staticmethod
    def collect_driver_aicore_number():
        import ctypes

        rts_so_path = _find_runtime_so()
        lib = ctypes.cdll.LoadLibrary(rts_so_path)
        driver_aicore_number = ctypes.c_int()
        get_core_num = ctypes.byref(driver_aicore_number)
        result = lib.rtGetAiCoreCount(get_core_num)
        if result != 0:
            utils.print_error_log("get driver aicore number failed")
            raise utils.AicErrException(Constant.MS_AICERR_GET_DRIVER_AICORE_NUMBER_ERROR)
        return driver_aicore_number.value

    @staticmethod
    def get_workspace_info(parse_level, workspace_list):
        if parse_level == 0:
            return 0
        else:
            workspace_info = 1
            for woskspace_npy in workspace_list:
                workspace = np.load(woskspace_npy)
                workspace_info = shape_utils.calc_shape_size(workspace.shape) * workspace.itemsize
            return workspace_info

    def get_dump_data_info(self: any):
        plog_dir = os.path.join(self.collect_path, 'collection', 'plog')
        if self.parse_level == 1:
            dump_data_cmd = ['grep', 'dump exception to file', '-inrE', plog_dir]
            adump_dump_data_regexp = r"tid\:(\d+).*?extra-info\/data-dump\/\d+\/([\w.]+)"
            ge_dump_data_regexp = r"(\d+) DumpNodeInfo:.*?extra-info\/data-dump\/\d+\/([\w.]+)"
            adump_dump_data_ret = utils.get_inquire_result(dump_data_cmd, adump_dump_data_regexp)
            ge_dump_data_ret = utils.get_inquire_result(dump_data_cmd, ge_dump_data_regexp)
            if not adump_dump_data_ret and not ge_dump_data_ret:
                utils.print_error_log(f"Dump file cannot be found in {self.collect_path}.")
                raise utils.AicErrException(Constant.MS_AICERR_INVALID_PATH_ERROR)
            if adump_dump_data_ret:
                thread_id, data_name = adump_dump_data_ret[0]
                return thread_id, data_name
            else:
                thread_id, data_name = ge_dump_data_ret[0]
                return thread_id, data_name
        else:
            dump_data_cmd = ['grep', 'dump exception to file', '-inrE', plog_dir]

            dump_data_regexp = r"tid\:(\d+).*?extra-info\/data-dump\/\d+\/([\w.]+)"
            dump_data_ret = utils.get_inquire_result(dump_data_cmd, dump_data_regexp)
            if not dump_data_ret:
                utils.print_error_log(f"Dump file cannot be found in {self.collect_path}.")
                raise utils.AicErrException(Constant.MS_AICERR_INVALID_PATH_ERROR)
            thread_id, data_name = dump_data_ret[0]
            return thread_id, data_name

    def get_op_info(self: any) -> AicErrorInfo:
        thread_id, data_name = self.get_dump_data_info()
        plog_path = os.path.join(self.collect_path, "collection", "plog")
        aicore_err_cmd = ['grep', 'there is an .*aicore.* error|there is an .*aivec.* error', '-inrE',
                          plog_path]
        aicore_err_regexp = r"(\d+-\d+-\d+-\d+:\d+:\d+\.\d+\.\d+).+?device_error_proc.cc\:\d+\](\d+)" \
                            r".+?device\(([a-zA-Z0-9\s,:]{1,})\),\s" \
                            r"[a-zA-Z0-9\s,]{1,},\score id is (\d+),\s+error code = (\S+),.*?pc start:\s(\S+)," \
                            r"\scurrent:\s(\S+),\svec error info:\s(\S+),\smte error info:\s(\S+)," \
                            r"\sifu error info:\s(\S+),\sccu error info:\s(\S+),\scube error info:\s(\S+)," \
                            r"\sbiu error info:\s(\S+),\saic error mask:\s(\S+),\spara base:\s(\S+)."
        aic_err_rets = utils.get_inquire_result(aicore_err_cmd, aicore_err_regexp)
        if not aic_err_rets:
            utils.print_error_log("aicore error exception does not match.")
            raise utils.AicErrException(Constant.MS_AICERR_FIND_DATA_ERROR)

        aic_err_ret = aic_err_rets
        for aic_err_info in aic_err_rets:
            if thread_id == aic_err_info[1]:
                aic_err_ret = aic_err_info
                break
            else:
                utils.print_error_log("dump data pid is not the same with rts pid.")
                raise utils.AicErrException(Constant.MS_AICERR_FIND_DATA_ERROR)

        rts_info_cmd = ['grep', '-r', "RUNTIME", plog_path]
        _, rts_info = utils.execute_command(rts_info_cmd)
        block_dim_regexp = r"blockDim=(\d+)"
        block_dim_ret = re.findall(block_dim_regexp, rts_info, re.M)
        if not block_dim_ret:
            utils.print_error_log("get runtime blockdim failed")
            raise utils.AicErrException(Constant.MS_AICERR_GET_RUNTIME_BLOCKDIM_ERROR)
        block_dim = 0
        for block_dim_num in block_dim_ret:
            if block_dim <= int(block_dim_num):
                block_dim = int(block_dim_num)
        rts_block_dim = block_dim

        info = AicErrorInfo()
        info.err_time = aic_err_ret[0]
        info.dev_id = aic_err_ret[2]
        info.core_id = aic_err_ret[3]
        info.error_code = aic_err_ret[4]
        info.start_pc = aic_err_ret[5]
        info.current_pc = aic_err_ret[6]
        # 此处附加判断是L0还是L1
        error_info = self._get_node_and_kernel_name(data_name)
        # 访问具名元组的属性
        info.stream_id = error_info.stream_id
        info.task_id = error_info.task_id
        info.node_name = error_info.node_name
        info.kernel_name = error_info.kernel_name
        info.hash_id = error_info.hash_id

        info.kernel_path = os.path.join(self.collect_path, 'collection', 'compile')
        info.tiling_key, tiling_data_bytes, info.block_dim = self.get_tiling(info.kernel_name)
        info.rts_block_dim = rts_block_dim
        info.driver_aicore_num = self.collect_driver_aicore_number()
        if tiling_data_bytes:
            target_bin = os.path.join(info.kernel_path, info.kernel_name + "_tiling.bin")
            with open(target_bin, "wb") as f:
                f.write(tiling_data_bytes)
            info.tiling_data = target_bin
            utils.print_debug_log(f"tiling data is saved to {target_bin}")

        kernel_file = self._get_kernel_and_json_file(info.kernel_name, info.tiling_key)
        info.bin_file = kernel_file.bin_file
        info.json_file = kernel_file.json_file
        info.cce_file = kernel_file.cce_file

        utils.print_debug_log(f"err_time: {info.err_time}, dev_id: {info.dev_id}, core_id: {info.core_id}, "
                             f"error_code: {info.error_code}, start_pc: {info.start_pc}, "
                             f"current_pc: {info.current_pc}, stream_id: {info.stream_id}, "
                             f"task_id: {info.task_id}, node_name: {info.node_name}, kernel_name: {info.kernel_name}, "
                             f"bin_file: {info.bin_file}, json_file: {info.json_file}, cce_file: {info.cce_file}, "
                             f"rts_block_dim: {info.rts_block_dim}, driver_aicore_num: {info.driver_aicore_num}.")

        if info.error_code == "0" or info.error_code == "0x0":
            info.error_code = self._get_v300_error_code()

        # extra_info 包括各寄存器信息ifu、ccu、biu、cube、mte、vec的寄存器错误码
        info.extra_info = self._get_extra_info(aic_err_ret)
        return info
    
    @staticmethod
    def _get_args(plog_path) -> list:
        key_word_list = []
        get_args_cmd = ['grep', 'after execute:', '-inrE', plog_path]
        get_args_regexp = r"(args\(0 to .*?\) after execute:.*?)after execute:args print end"
        get_args_ret = utils.get_inquire_result(get_args_cmd, get_args_regexp)
        if not get_args_ret or len(get_args_ret) == 0:
            utils.print_warn_log("Failed to get all args after execute.")
            return []

        get_args_ret = get_args_ret[0]
        get_arg_regexp = r" after execute:(.*?)$"
        get_arg_ret = re.findall(get_arg_regexp, get_args_ret, re.M | re.S)
        if not get_arg_ret:
            utils.print_warn_log("Failed to get arg after execute.")
            return []

        for args in get_arg_ret:
            split_tmp = args.split(",")
            for str2 in split_tmp:
                if str2.find("\n") != -1:
                    continue
                str2 = str2.replace(' ', '')
                if not str2:
                    continue
                key_word_list.append(str2)
        return key_word_list
    
    @staticmethod
    def _get_para_base(plog_path) -> int:
        para_base_cmd = ['grep', 'para base', '-inrE', plog_path]
        para_base_regexp = r"para base:\s+(.*?)\."
        parm_base_ret = utils.get_inquire_result(para_base_cmd, para_base_regexp)
        if not parm_base_ret or len(parm_base_ret) == 0:
            utils.print_warn_log("Failed to get para_base after execute.")
            return -1
        utils.print_debug_log(f"para_base address is {parm_base_ret[0]}")
        return utils.get_hexstr_value(parm_base_ret[0])
    
    def _get_sub_ptr(self: any, info) -> dict:
        utils.print_debug_log("Start to get sub ptr.")
        plog_path = os.path.join(self.collect_path, "collection", "plog")
        sub_ptr_rst = {}
        sub_ptr_index = []
        sub_ptr_addrs = []
        dynamic_tensor_count = []
        de_deplicate = []
        get_io_cmd = ['grep', '\[Dump\]\[Exception\]', '-inrE', plog_path]
        get_io_regexp = r"begin to load .*? tensor.*?end to load .*? tensor"
        get_io_ret = utils.get_inquire_result(get_io_cmd, get_io_regexp)
        idx = 0
        
        for ret in get_io_ret:
            if idx > len(info.bin_list) - 1:
                break
            get_tensor_regexp = r"exception info dump args data, addr:(.*?);"
            get_tensor_ret = re.findall(get_tensor_regexp, ret, re.M | re.S)
            tensor_cnt = len(get_tensor_ret)
            if "pointer tensor" in ret:
                get_sub_ptr_regexp = r"begin to load .*? pointer tensor.*?addr:(.*?)$"
                get_sub_ptr_ret = re.findall(get_sub_ptr_regexp, ret, re.M | re.S)
                sub_ptr_addr = get_sub_ptr_ret[0]
                # is pointer tensor
                if sub_ptr_addr not in sub_ptr_addrs:
                    sub_ptr_index.append(idx)
                    sub_ptr_addrs.append(sub_ptr_addr)
                    dynamic_tensor_count.append(tensor_cnt)
            de_deplicate.extend(get_tensor_ret)
            idx = len(list(de_deplicate))
        key_word_list = self._get_args(plog_path)
        if key_word_list is None or len(key_word_list) == 0:
            utils.print_warn_log("Failed to get args after execute.")
            return {}
        para_base = self._get_para_base(plog_path)
        if para_base == -1:
            utils.print_warn_log("Failed to get para base. The sub ptr process ends.")
            return {}

        for idx, addr, tensor_cnt in zip(sub_ptr_index, sub_ptr_addrs, dynamic_tensor_count):
            addr = utils.get_hexstr_value(addr)
            offset = (addr - para_base) // 8
            if offset >= len(key_word_list):
                utils.print_warn_log(f"Tensor with index {idx} is sub_ptr tensor,"
                                     f"sub_ptr offset is {offset}, out of bounds in args")
                continue
            sub_addr = key_word_list[offset]
            utils.print_debug_log(f"Tensor with index {idx} is sub_ptr tensor,"
                                 f"sub_ptr offset is {offset}, and sub_ptr address is {sub_addr}")
            clip_offset = utils.get_hexstr_value(sub_addr) // 8
            if (offset + clip_offset) >= len(key_word_list):
                utils.print_warn_log(f"Tensor with index {idx} is sub_ptr tensor,"
                                     f"sub_ptr offset is {offset}, clip_offset is {clip_offset},"
                                     f"out of bounds in args")
                continue
            clip_args = key_word_list[offset:offset+clip_offset]
            utils.print_debug_log(f"clip_offset is {clip_offset}, and clip_args is {clip_args}")
            sub_ptr_rst[idx] = {"args_list": clip_args}
            sub_ptr_rst[idx]["dynamic_tensor_count"] = tensor_cnt
        return sub_ptr_rst

    def _get_graph_file(self: any) -> str:
        utils.print_debug_log("Start to get graph file.")
        match_list = []
        for top, _, files in os.walk(os.path.join(self.collect_path, 'collection', 'graph')):
            for name in files:
                file_name_pattern = re.compile(r"^ge_proto_(.*?)_Build\.txt$")
                pattern_match = file_name_pattern.match(name)
                if pattern_match:
                    match_list.append((pattern_match.group(1), os.path.join(top, name)))
        new_match_list = sorted(match_list, key=lambda s: s[0], reverse=True)
        if len(new_match_list) > 0 and len(new_match_list[0]) > 0:
            choose_file = new_match_list[0][1]
        else:
            choose_file = ""
        utils.print_debug_log(f'Choose {choose_file} to read op info.')
        return choose_file

    @staticmethod
    def _get_op_by_graph(graph_file: str, node_name, kernel_name):
        try:
            if not isinstance(graph_file, str):
                utils.print_warn_log(f'graph_file: {graph_file} cannot found.')
                return None
            if not os.path.exists(graph_file):
                utils.print_warn_log(f'Failed to find graph_file: {graph_file}.')
                return None
            with open(graph_file, 'r') as graph:
                text = graph.read()
                regexp = r'(op\s+\{\s+name:\s+"%s".+?%s.+?\})\s+op\s+\{' % (node_name, kernel_name)
                ret = re.findall(regexp, text, re.M | re.S)
                if not ret:
                    utils.print_warn_log(f'Failed to get op for node({node_name}) kernel({kernel_name}).')
                    return None
                return ret[0]
        except BaseException as io_error:
            utils.print_warn_log(f'Failed to open file graph_file: {graph_file}.')
            return None

    def _write_summary_file(self: any, summary_info_list: list) -> None:
        summary = """本次信息收集发生于%s，只收集第1个AICERROR，概要如下：
        ***************************************************************************************************
        %s
        ***************************************************************************************************
        建议选择最近发生的AICERROR，查看其中的info.txt， 日志打印信息查看其中的debug_info.txt。

        """ % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
       "\n".join(summary_info_list))
        summary_file = os.path.join(self.collect_path, "README.txt")
        utils.write_file(summary_file, summary)
        utils.print_info_log('The summary info is saved in %s' % summary_file)

        utils.print_info_log('Analysis finished, please check %s, you can '
                             'view README.txt first.' % self.collect_path)
        current_path = os.getcwd()
        print_info_path = current_path + '/debug_info.txt'
        print_info_cmd = ['mv', print_info_path, self.collect_path]
        _, rts_info = utils.execute_command(print_info_cmd)

    def _get_atomic_err_log(self: any) -> list:
        cmd = ['grep', 'dha status 1', '-nr', self.collect_path]
        status, _ = utils.execute_command(cmd)
        if status == 0:
            utils.print_debug_log("#" * 70)
            utils.print_debug_log("Find \"dha status 1\" in plogs. Maybe atomic add error happened!")
            utils.print_debug_log("#" * 70)
            return True
        return False

    def _get_tiling_l1(self, kernel_name) -> tuple:
        plog_path = os.path.join(self.collect_path, "collection", "plog")
        aic_info_cmd = ['grep', '-r', '-C', '1024', f"\[AIC_INFO\] dev_func:{kernel_name}", plog_path]
        _, aic_info = utils.execute_command(aic_info_cmd)
        tiling_data_regexp = r"\[AIC_INFO\]\stiling_data:(.*)"
        tiling_data_ret = re.findall(tiling_data_regexp, aic_info, re.M)
        if len(tiling_data_ret) == 0:
            utils.print_warn_log(f"Failed to get {tiling_data_regexp}")
            tiling_data = None
        else:
            if tiling_data_ret[0].startswith("0x"):
                temp_tiling_data = ""
                for tiling_data in tiling_data_ret:
                    temp_tiling_data += tiling_data.replace(" 0x", "").replace("0x", "").strip()
                try:
                    tiling_data = bytes.fromhex(temp_tiling_data)
                except Exception as e:
                    utils.print_warn_log(f"Failed to decode tiling_data {temp_tiling_data}")
                    tiling_data = bytes.fromhex("0000")
            else:
                temp_tiling_data = ""
                for tiling_data in tiling_data_ret:
                    temp_tiling_data += tiling_data.strip()
                tiling_data = bytes(temp_tiling_data, 'utf-8')
        tiling_key_regexp = r"\[AIC_INFO\]\stiling_key:([0-9]{1,})"
        tiling_key_ret = re.findall(tiling_key_regexp, aic_info, re.M)
        if len(tiling_key_ret) == 0:
            utils.print_warn_log(f"Failed to get {tiling_key_regexp}")
            tiling_key = "0"
        else:
            tiling_key = utils.get_str_value(tiling_key_ret[0])

        block_dim_regexp = r"\[AIC_INFO\]\sblock_dim:(\d+)"
        block_dim_ret = re.findall(block_dim_regexp, aic_info, re.M)
        if not block_dim_ret:
            utils.print_warn_log(f"Failed to get {block_dim_regexp} is null.")
            block_dim = -1
        else:
            block_dim = utils.get_str_value(block_dim_ret[0])
        return tiling_key, tiling_data, block_dim

    def _get_tiling_l0(self) -> tuple:
        if not self.ffts_flag:
            plog_path = os.path.join(self.collect_path, "collection", "plog")
            aic_info_cmd = ['grep', '-r', f"tilingKey =", plog_path]
            _, aic_info = utils.execute_command(aic_info_cmd)
            tiling_key_regexp = r"tilingKey = (.*?),"
            tiling_key_ret = re.findall(tiling_key_regexp, aic_info, re.M)
            if not tiling_key_ret:
                utils.print_warn_log("Unable to get tilingKey in plog, set tilingKey=0")
                tiling_key = 0
            else:
                tiling_key = utils.get_str_value(tiling_key_ret[0])
                tiling_key = 0 if tiling_key == -1 else tiling_key
            block_dim_regexp = r"blockDim=(\d+)"
            block_dim_ret = re.findall(block_dim_regexp, aic_info, re.M)
            if not block_dim_ret:
                utils.print_warn_log("Unable to get blockDim in plog, set block_dim=0")
                block_dim = 0
            else:
                block_dim = utils.get_str_value(block_dim_ret[0])
            tiling_data = TilingDataParser(plog_path).parse()
            return tiling_key, tiling_data, block_dim
        else:
            plog_path = os.path.join(self.collect_path, "collection", "plog")
            aic_info_cmd = ['grep', '-r', f"fftsplus aivector error", plog_path]
            _, aic_info = utils.execute_command(aic_info_cmd)
            kernel_name_regexp = r"kernel_name=(.*?),"
            kernel_name_ret = re.findall(kernel_name_regexp, aic_info, re.M)
            if not kernel_name_ret:
                utils.print_warn_log("Unable to get tilingKey in plog, set tilingKey=0")
                tiling_key = 0
            else:
                kernel_name = kernel_name_ret[0].replace("_mix_aic", "").replace("_mix_aiv", "")
                tiling_key = utils.get_str_value((kernel_name.split("_")[-1]))
                tiling_key = 0 if tiling_key == -1 else tiling_key

            block_dim_regexp = r"blockDim=(\d+)"
            block_dim_ret = re.findall(block_dim_regexp, aic_info, re.M)
            if not block_dim_ret:
                utils.print_warn_log("Unable to get blockDim in plog, set block_dim=0")
                block_dim = 0
            else:
                block_dim = utils.get_str_value(block_dim_ret[0])
            tiling_data = TilingDataParser(plog_path).parse()
            return tiling_key, tiling_data, block_dim

    def get_tiling(self, kernel_name=""):
        if self.parse_level == 0:
            return self._get_tiling_l0()
        else:
            return self._get_tiling_l1(kernel_name)

    def get_ffts_addrs_num(self):
        plog_path = os.path.join(self.collect_path, "collection", "plog")
        get_idx_cmd = ["grep", "begin to load normal tensor", "-inrE", plog_path]
        get_idx_regexp = r".*?index:(\d+)"
        get_idx_ret = utils.get_inquire_result(get_idx_cmd, get_idx_regexp)
        if not get_idx_ret:
            return 0
        return int(get_idx_ret[0])

    @staticmethod
    def _cal_shape_size(shape_str):
        utils.print_debug_log("shape_str is {}".format(shape_str))
        if shape_str == "[]":
            return 1
        shape_str_list = shape_str.replace("[", "").replace("]", "").split(",")
        return reduce(lambda x, y: int(x) * int(y), shape_str_list)

    @staticmethod
    def _check_addr_in_range(addr, size, ranges):
        if not isinstance(addr, int):
            addr = int(addr)

        for addr_range in ranges:
            if "0x" in addr_range[0]:
                range_left = utils.get_hexstr_value(addr_range[0])
                range_right = utils.get_hexstr_value(addr_range[0]) + addr_range[1]
            else:
                range_left = int(addr_range[0])
                range_right = int(addr_range[0]) + addr_range[1]
            if range_left <= addr <= addr + size <= range_right:
                return True
        return False

    def _check_addr(self, avaliable_addrs, used_addrs):
        input_params = used_addrs.get("input_addr")
        output_params = used_addrs.get("output_addr")
        need_check_args = used_addrs.get("need_check_args")
        if not input_params and not output_params:
            utils.print_error_log("Unable to get input parameters and output parameters.")
            raise utils.AicErrException(Constant.MS_AICERR_FIND_DATA_ERROR)

        for input_param in input_params:
            if input_param.get("addr").startswith("0x"):
                start_addr = int(input_param.get("addr"), 16)
            else:
                start_addr = int(input_param.get("addr"))
            shape_size = self._cal_shape_size(input_param.get("shape"))
            size_of_dtype = Constant.SIZE_OF_DTYPE.get(input_param.get("dtype"))
            input_param["size"] = int(shape_size) * int(size_of_dtype)
            utils.print_info_log(f"shape_size is {shape_size}, size_of_dtype is {size_of_dtype}")
            input_param["in_range"] = self._check_addr_in_range(start_addr, input_param["size"], avaliable_addrs)

        for output_param in output_params:
            if output_param.get("addr").startswith("0x"):
                start_addr = int(output_param.get("addr"), 16)
            else:
                start_addr = int(output_param.get("addr"))
            shape_size = self._cal_shape_size(output_param.get("shape"))
            size_of_dtype = Constant.SIZE_OF_DTYPE.get(output_param.get("dtype"))
            utils.print_info_log(f"shape_size is {shape_size}, size_of_dtype is {size_of_dtype}")
            output_param["size"] = int(shape_size) * int(size_of_dtype)
            output_param["in_range"] = self._check_addr_in_range(start_addr, output_param["size"], avaliable_addrs)

        used_addrs["fault_arg_index"] = []
        if need_check_args:
            for i, arg in enumerate(need_check_args):
                if not self._check_addr_in_range(arg, 0, avaliable_addrs):
                    used_addrs["fault_arg_index"].append(i)

    def _get_info_for_decompile(self: any, info: any) -> tuple:
        info.instr = ""
        # 最后一条指令 算差值
        utils.print_debug_log("Calculate the address offset.")
        current_pc5 = '0' + info.current_pc[-5:]
        start_pc5 = '0' + info.start_pc[-5:]
        current_pc5_value = utils.get_hexstr_value(current_pc5)
        start_pc5_value = utils.get_hexstr_value(start_pc5)
        diff_str = hex(current_pc5_value - start_pc5_value)[2:]
        utils.print_debug_log(f"The address offset is {diff_str}.")

        # 估算err pc
        err_pc = ""
        try:
            err_pc = self._get_err_pc(info, current_pc5_value, start_pc5_value)
        except BaseException:
            utils.print_debug_log("_get_err_pc failed")
        if err_pc == "":
            utils.print_debug_log("This aicore error has no estimated offset.")
        else:
            utils.print_debug_log(f"Estimated error address offset is {err_pc}.")

        return diff_str, err_pc

    @staticmethod
    def _get_decompile_status(o_file: str, decompile_file: str) -> int:
        if shutil.which(Constant.OBJ_DUMP_FILE):
            objdump = Constant.OBJ_DUMP_FILE
        elif shutil.which(Constant.NEW_DUMP_FILE):
            objdump = Constant.NEW_DUMP_FILE
        else:
            utils.print_error_log("No cce-objdump or llvm-objdump found.")
        cmd = [objdump, '-d', '--line-numbers', o_file]
        status, _ = utils.execute_command(cmd, file_out=decompile_file)
        return status

    @staticmethod
    def _update_err_pc(err_pc: str, decompile_file: str, kernel_name: str) -> None:
        # 模板类算子是多个.o合成需找到对应的行号
        utils.print_debug_log(f"Start to update err pc: {err_pc}.")
        update_pc_cmd = ['grep', f'{kernel_name}$local', decompile_file]
        update_pc_regexp = r"([0-9A-Za-z]*?)\s+<{}\$local>".format(kernel_name)
        update_pc_ret = utils.get_inquire_result(update_pc_cmd, update_pc_regexp)
        if not update_pc_ret:
            utils.print_debug_log("No need to update err pc.")
            return err_pc
        else:
            err_pc = hex(int(err_pc, 16) + int(update_pc_ret[0], 16))[2:]
            utils.print_debug_log(f"find base pc is 0x{update_pc_ret[0]}, err pc after update is  0x{err_pc}.")
            return err_pc

    def _decompile(self: any, kernel_meta_path: str, dir_path: str, info: AicErrorInfo) -> bool:
        kernel_name = info.kernel_name

        tiling_key = info.tiling_key

        # 获取cce, o, json路径
        cce_file = info.cce_file
        o_file = info.bin_file
        json_file = info.json_file
        if not os.path.exists(o_file) and not os.path.exists(json_file):
            utils.print_error_log("The file that needs to be decompiled does not exist.")
            return False

        # decompile .o file
        decompile_file = o_file + ".txt"
        status = self._get_decompile_status(o_file, decompile_file)
        if status != 0:
            utils.print_error_log(f"Failed to decompile {o_file}, you can fix problem according to the message above, "
                                  f"or copy {Constant.OBJ_DUMP_FILE} and {o_file} to another host and execute : "
                                  f"{Constant.OBJ_DUMP_FILE} -d {kernel_name}.o > {kernel_name}.o.txt")
            return False
        utils.copy_src_to_dest([cce_file, o_file, json_file], dir_path)
        loc_json_file = os.path.join(kernel_meta_path, kernel_name + "_loc.json")
        diff_str, err_pc = self._get_info_for_decompile(info)
        if self.parse_level == 1:
            err_pc = self._update_err_pc(err_pc, decompile_file, f"{kernel_name}_{tiling_key}")
            cce_tbe_result = self._get_cce_tbe_code_number(decompile_file, loc_json_file, err_pc, info)
            occur_result = self._get_occur_before_mark(decompile_file, diff_str, info)

            return cce_tbe_result and occur_result
        else:
            return True

    @staticmethod
    def _get_err_pc(info: any, current_pc5_value: int, start_pc5_value: int) -> str:
        # 估算err pc
        extra_pc = info.find_extra_pc()  # [9:2]
        if extra_pc != "":
            ori_pc = bin(current_pc5_value)[2:]
            if len(extra_pc) == 16:
                new_pc_bin = extra_pc + "00"
            elif len(ori_pc) == 1:
                new_pc_bin = extra_pc + '0' + ori_pc
            elif len(ori_pc) <= 10:
                new_pc_bin = extra_pc + ori_pc[-2:]
            else:
                new_pc_bin = ori_pc[:-10] + extra_pc + ori_pc[-2:]
            new_pc_value = int(new_pc_bin, 2)
            if new_pc_value - 1024 > start_pc5_value and new_pc_value > current_pc5_value and len(extra_pc) != 16:
                new_pc_value -= 1024
            if new_pc_value - start_pc5_value > 0:
                err_pc = hex(new_pc_value - start_pc5_value)[2:]
            else:
                err_pc = hex(current_pc5_value - start_pc5_value)[2:]
        else:
            err_pc = hex(current_pc5_value - start_pc5_value)[2:]
        info.instr += "\nError occured most likely at line: %s\n\n" % err_pc
        return err_pc

    @staticmethod
    def _read_decompile_file(decompile_file: str, err_pc: str, info: any) -> str:
        with open(decompile_file, 'r') as fo_file:
            cce_code = ""
            cce_code_num = ""
            for line in fo_file.readlines():
                regexp = r"cce:(\d+)"
                ret = re.findall(regexp, line, re.M)
                if len(ret) > 0:
                    cce_code = line
                    cce_code_num = ret[0]
                elif err_pc + ':' in line:
                    info.instr += "%s:%s\n" % (fo_file.name, err_pc)
                    break
            cce_line_number = cce_code.split(os.sep)[-1]
            utils.print_debug_log(f"Maybe find cce code line number is {cce_line_number}")
            info.instr += "%s" % cce_line_number
        return cce_code_num

    @staticmethod
    def _read_loc_json_file(loc_json_file: str, cce_code_num: str, info: any) -> None:
        with open(loc_json_file, 'r') as load_f:
            load_dict = json.load(load_f)
            for line in load_dict[0]['cce_line2loc']:
                if str(line['cce_line']) == cce_code_num \
                        and line['loc'][0] != "":
                    info.instr += "%s:%s\n" % (line['loc'][0], line['loc'][1])

    def _get_cce_tbe_code_number(self: any, decompile_file: str, loc_json_file: str, err_pc: str, info: any) -> bool:
        # txt code to cce number
        if os.path.exists(decompile_file) is False:
            utils.print_error_log("The decompile file does not exist.")
            return False

        if err_pc != "":
            cce_code_num = self._read_decompile_file(decompile_file, err_pc, info)
            # cce to tbe code number
            if not os.path.exists(loc_json_file) or os.stat(loc_json_file).st_size == 0:
                utils.print_warn_log(f"The file {loc_json_file} is not exist or file is empty.")
                return True
            self._read_loc_json_file(loc_json_file, cce_code_num, info)

            cce_file = info.cce_file
            if os.path.exists(cce_file):
                with open(cce_file, 'r') as f:
                    for index, line in enumerate(f):
                        if int(cce_code_num) - 1 == index:
                            if "PIPE_ALL" in line:
                                info.flag_check = "Please check the set_flag/wait_flag is match or not!!!."
                                break
            else:
                utils.print_error_log("The cce file does not exist.")
                return False
        return True

    @staticmethod
    def __generate_case(config, case_path, op_test):
        config_str = json.dumps(config, indent=4)
        op_test_str = json.dumps(op_test, indent=4)
        case_content = f"""
from ms_interface.single_op_test_frame.single_op_case import SingleOpCase
config = {config_str}
OP_TEST = {op_test_str}
SingleOpCase.run(config, OP_TEST)
exit()"""

        case_file = os.path.join(case_path, f"test_{op_test}.py")
        utils.print_debug_log(f"Generate case file {case_file}")
        with open(case_file, 'w') as f:
            f.write(case_content)
        return case_file

    @staticmethod
    def _test_single_op(aic_info: AicErrorInfo, case_path: str, op_test: str):
        single_op_case = SingleOpCase(aic_info, op_test)
        config = single_op_case.generate_config()
        case_file = AicoreErrorParser.__generate_case(config, case_path, op_test)

        date_string = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
        single_op_log_path = os.path.join(case_path, f"{op_test}_{date_string}")
        utils.print_debug_log(f"The single_op_log_path is {single_op_log_path}")
        current_path = os.path.dirname(os.path.abspath(__file__))
        new_env = os.environ.copy()
        new_env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
        new_env['ASCEND_PROCESS_LOG_PATH'] = single_op_log_path
        new_env['PYTHONPATH'] = new_env.get('PYTHONPATH') + ":" + os.path.dirname(current_path)

        AicoreErrorParser.comment_cce_in_case(case_file)
        _exec_run = subprocess.run(['python3', case_file], env=new_env, stdout=subprocess.PIPE)
        _len = _exec_run.stdout.decode('gbk').find("SingleOpCase.run Execute Info")
        test_info: str = _exec_run.stdout.decode('gbk')[_len:]
        single_op_ret = not AicoreErrorParser.search_aicerr_log(aic_info.kernel_name,
                                                                single_op_log_path)
        if not single_op_ret:
            AicoreErrorParser.print_single_op_result(case_file)
            return False, test_info, single_op_log_path
        return True, test_info, single_op_log_path

    @staticmethod
    def comment_cce_in_case(case_file):
        with open(case_file, "r+") as f:
            content = f.read().replace("\"cce_file\"", "# \"cce_file\"")
            f.seek(0)
            f.truncate(0)
            f.write(content)

    @staticmethod
    def print_single_op_result(case_file):
        split_line = "#" * 50
        utils.print_debug_log(split_line)
        utils.print_debug_log("Single op test failed! Please check OP or input data!")
        current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        utils.print_debug_log(f"Run 'export PYTHONPATH={current_path}:$PYTHONPATH;cd {current_path};"
                             f"python3 {case_file}' can test op!")
        utils.print_debug_log(split_line)

    @staticmethod
    def _get_occur_before_mark(decompile_file: str, diff_str: str, info: any) -> bool:
        #      504:    04c20000    ST.b64         X1, [X0], #0
        with open(decompile_file, "r") as fo_file:
            text = fo_file.read()

        regexp = r'(^\s+(\S+):\s+\S+\s+\S.+$)'
        ret = re.findall(regexp, text, re.M)
        find_i = -1
        for i, (_, line_diff) in enumerate(ret):
            if line_diff == diff_str:
                find_i = i
                break

        if find_i == -1:
            utils.print_warn_log(f"Get fault instruction failed, file({decompile_file}) diff({diff_str}).")
            return False

        begin_i = 0 if find_i < 9 else find_i - 9
        instr_str_list = []
        for i in range(begin_i, find_i + 1):
            instr_str_list.append(ret[i][0] + "\n")
        instr_str = "".join(instr_str_list).strip("\n")

        info.instr += "\nrelated instructions (error occured before the mark *):\n\n"
        info.instr += instr_str[:instr_str.rfind('\n') + 1] + '*'
        info.instr += instr_str[instr_str.rfind('\n') + 2:]
        info.instr += "\n\nFor complete instructions, please view %s" % decompile_file

        return True

    @staticmethod
    def _write_errorinfo_file(err_i_folder: str, info: AicErrorInfo) -> None:
        info_file = os.path.join(err_i_folder, "info.txt")
        utils.write_file(info_file, info.analyse())
        utils.print_info_log(f'Analysis info is saved in {info_file}')

    def _get_data_dump_result(self: any):
        try:
            data_dump_failed_cmd = ['grep', 'exception_dumper.cc.*Dump exception.*failed', '-nr',
                                    self.collect_path]
            data_dump_ret, _ = utils.execute_command(data_dump_failed_cmd)
            if data_dump_ret == 0:
                utils.print_warn_log("Data dump failed in exception dump. Please contact GE to resolve it!")
                return False
        except utils.AicErrException as e:
            utils.print_error_log("Failed to dump data!")
        try:
            data_dump_failed_cmd1 = ['grep', '\[Dump\]\[Exception\] D2H failed', '-nr', self.collect_path]
            data_dump_failed_cmd2 = ['grep', '\[Exception\] the address maybe invalid', '-nr', self.collect_path]
            data_dump_ret1, _ = utils.execute_command(data_dump_failed_cmd1)
            data_dump_ret2, _ = utils.execute_command(data_dump_failed_cmd2)
            if data_dump_ret1 == 0 or data_dump_ret2 == 0:
                utils.print_warn_log("Data dump failed. Maybe memory is invalid. Search 'D2H failed' in plog!")
                return False
        except utils.AicErrException as e:
            utils.print_error_log("Failed to dump data!")
        return True

    def check_dump_result(self: any, dfx_message: str, info: AicErrorInfo) -> bool:
        data_dump_failed_cmd = ['grep', 'exception_dumper.cc.*Dump exception.*failed', '-nr',
                                self.collect_path]
        data_dump_ret, _ = utils.execute_command(data_dump_failed_cmd)
        if data_dump_ret == 0:
            utils.print_error_log("Data dump failed in exception dump. Please contact GE to resolve it!")
            return False
        data_dump_copy_failed_cmd = ['grep', '\[Dump\]\[Exception\] D2H failed', '-nr', self.collect_path]
        data_dump_copy_ret, _ = utils.execute_command(data_dump_copy_failed_cmd)
        if data_dump_copy_ret == 0:
            utils.print_error_log("Data dump failed. Copy data from device to host fail!")
            return False
        memory_failed_str = "\[Dump\]\[Exception\] the address maybe invalid"
        memory_faided_ret = re.findall(memory_failed_str, dfx_message)
        if memory_faided_ret:
            utils.print_error_log("Data dump failed. Maybe memory is invalid!")
            return False
        ffts_addr_num_str = "begin to load normal tensor, index:(\d+)"
        ffts_addr_num_ret = re.findall(ffts_addr_num_str, dfx_message)
        if not ffts_addr_num_ret:
            info.ffts_addrs_num = 0
            return True
        info.ffts_addrs_num = int(ffts_addr_num_ret[0])
        return True

    def _need_atomic_clean(self: any, kernel_meta_path: str, info: any) -> bool:
        kernel_name = info.kernel_name
        json_file = os.path.join(kernel_meta_path, kernel_name + ".json")
        if not os.path.exists(json_file):
            utils.print_warn_log(f"Can not find {json_file}!")
            return False
        with open(json_file, "r") as f:
            json_obj = json.load(f)
            # compileInfo in json file means the kernel is dynamic
            if json_obj.get("compileInfo") is None and json_obj.get("compile_info") is None:
                utils.print_debug_log(f"No compile_info found in json file, no need to check atomic clean!")
                return False
            parameters = json_obj.get("parameters")
            for param in parameters:
                if param is not None:
                    return True
        return False

    def _check_atomic_clean(self: any, kernel_meta_path: str, info: AicErrorInfo) -> bool:
        need_atomic_clean = self._need_atomic_clean(kernel_meta_path, info)
        if need_atomic_clean:
            cmd = ['grep', f'AtomicLaunchKernelWithFlag_{info.node_name}', '-nr', self.collect_path]
            status, _ = utils.execute_command(cmd)
            if status == 0:
                return True
            utils.print_warn_log(f"Can not find AtomicLaunchKernelWithFlag_{info.node_name} in plog!")
            return False
        return True

    def _get_args_from_info(self: any, key_words: str) -> list:
        args_exec_cmd = ['grep', key_words, '-nr', self.collect_path]
        args_exec_regexp = r":([x0-9a-fA-F,\s]+)addr"
        args_exec_rets = utils.get_inquire_result(args_exec_cmd, args_exec_regexp)

        if not args_exec_rets:
            args_exec_cmd = ['grep', key_words, '-Enr', self.collect_path]
            args_exec_regexp = r"args.*?after execute:([x0-9a-fA-F,\s]+)"
            args_exec_rets = utils.get_inquire_result(args_exec_cmd, args_exec_regexp)

        if not args_exec_rets:
            args_exec_rets = []

        args_exec_result = []
        result = []
        for args_exec_ret in args_exec_rets:
            args_array = re.split(",|\\s", args_exec_ret)
            args_list = []
            for arg in args_array:
                if (not isinstance(arg, str)) or (not arg.strip()):
                    continue
                arg = arg.strip()
                if (not arg.startswith("0x")) and (arg != "0"):
                    continue
                if arg.startswith("0x") or arg == "0":
                    args_list.append(utils.get_hexstr_value(arg))
                else:
                    args_list.append(int(arg))
            args_tuple = tuple(args_list)
            if args_tuple not in args_exec_result:
                args_exec_result.append(args_tuple)
                result.extend(args_list)
        return result

    def _get_args_after_exc(self: any) -> list:
        after_key = '\[AIC_INFO\] args.*after execute'
        return self._get_args_from_info(after_key)

    def _get_args_before_exc(self: any) -> list:
        before_key = '\[AIC_INFO\] args before execute'
        return self._get_args_from_info(before_key)

    @staticmethod
    def _check_file_content(kernel_name, content):
        error_strings = [
            "there is an aivec error exception",
            "there is an aicore error exception",
            "aicore exception"
        ]
        for s in error_strings:
            if s in content and kernel_name in content:
                return True
        return False

    @staticmethod
    def _wait_for_log_stabilization(log_path):
        log_size = os.path.getsize(log_path)
        while True:
            time.sleep(0.2)
            current_log_size = os.path.getsize(log_path)
            if current_log_size == log_size:
                break
            log_size = current_log_size

    @staticmethod
    def search_aicerr_log(kernel_name, path):
        kernel_name = kernel_name.replace("_mix_aic", "").replace("_mix_aiv", "")
        for root, _, files in os.walk(path):
            for file in files:
                if not file.endswith(".log"):
                    continue
                log_path = os.path.abspath(os.path.join(root, file))
                AicoreErrorParser._wait_for_log_stabilization(log_path)
                with open(log_path, "r") as f:
                    content = f.read()
                if AicoreErrorParser._check_file_content(kernel_name, content):
                    return True
        return False

    @staticmethod
    def get_soc_version_from_cce(cce_file):
        try:
            with open(cce_file, 'r') as f:
                content = f.read()
            soc_version_ret = re.findall(r'//.*?(Ascend.*?)"', content)
            if soc_version_ret:
                utils.print_debug_log(f"Get soc_version {soc_version_ret[0]} from cce file {cce_file}")
                if soc_version_ret[0] == "Ascend910B":
                    return "Ascend910B2"
                elif soc_version_ret[0] == "Ascend310B":
                    return "Ascend310B1"
                return soc_version_ret[0]
            else:
                utils.print_warn_log('Can not get soc_version from cce file {cce_file}')
                return "Ascend310"
        except Exception as e:
            utils.print_warn_log('Can not get soc_version from cce file {cce_file}')
            utils.GLOBAL_RESULT = False
            return "Ascend310"

    @staticmethod
    def run_test_env(soc_version, device_id=0):
        date_string = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
        kernel_name = f"golden_op_{soc_version}_{date_string}"
        golden_op_path = os.path.abspath(f"golden_op_{date_string}")
        current_path = os.path.dirname(os.path.abspath(__file__))
        new_env = os.environ.copy()
        new_env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
        new_env['ASCEND_PROCESS_LOG_PATH'] = golden_op_path
        new_env['PYTHONPATH'] = new_env.get('PYTHONPATH') + ":" + os.path.dirname(current_path)

        cmd = ['python3', f'{current_path}/golden_op.py', soc_version, kernel_name, str(device_id)]
        subprocess.run(cmd, env=new_env)
        result = not AicoreErrorParser.search_aicerr_log(kernel_name, golden_op_path)
        if not result:
            utils.print_error_log("there is an error exception. Please check npu device!")
            return False
        else:
            if os.path.exists(golden_op_path):
                shutil.rmtree(golden_op_path)
            if os.path.exists("kernel_meta"):
                shutil.rmtree("kernel_meta")
            return True

    @staticmethod
    def add_objdump_to_path():
        # decompile
        utils.print_debug_log("Start looking for the location of cce-objdump.")
        obj_dump_file = "cce-objdump_aarch64" if "aarch64" in platform.machine() else "cce-objdump"
        obj_dump_file = os.path.join(os.getcwd(), "tools", obj_dump_file)
        if os.path.exists(obj_dump_file):
            os.system("chmod 755 " + obj_dump_file)
            os.environ["PATH"] = os.path.join(os.getcwd(), "tools") + ":" + os.environ["PATH"]
        else:
            cce_dump = shutil.which("cce-objdump") or shutil.which("llvm-objdump")
            if not cce_dump:
                # guess where is cce-objdump
                parent_path = "aarch64-linux" if "aarch64" in platform.machine() else "x86_64-linux"
                cce_dump_guess = os.path.join("/usr/local/Ascend/latest", parent_path, "ccec_compiler/bin/cce-objdump")
                llvm_dump_guess = os.path.join("/usr/local/Ascend/latest", parent_path,
                                               "ccec_compiler/bin/llvm-objdump")
                if os.path.exists(cce_dump_guess):
                    cce_dump = cce_dump_guess
                elif os.path.exists(llvm_dump_guess):
                    cce_dump = llvm_dump_guess
                else:
                    utils.print_error_log("Cannot find cce-objdump! please add cce-objdump path in env PATH.")
                    raise utils.AicErrException(Constant.MS_AICERR_EXECUTE_COMMAND_ERROR)
            os.environ["PATH"] = os.path.dirname(cce_dump) + ":" + os.environ["PATH"]

    def check_hash_id(self, hash_id, single_op_log_path):
        kernel_name_cmd = ['grep', 'Aicore kernel execute failed', '-inrE', single_op_log_path]

        kernel_name_regexp = r".*?hash=(\d+)"
        kernel_name_ret = utils.get_inquire_result(kernel_name_cmd, kernel_name_regexp)

        if not kernel_name_ret:
            utils.print_warn_log(f"Can't find hash id in single_op_log_path:{single_op_log_path}.")
            return
            
        if hash_id != kernel_name_ret[0]:
            utils.print_warn_log(f"The hash_id from plog is different with the hash_id which from the device.")
            return

    @staticmethod
    def get_soc_version(cce_file):
        try:
            soc_version = DSMIInterface().get_chip_info(0).get_complete_platform()
        except BaseException:
            utils.print_error_log("get soc_version form platform failed!")
            soc_version = None
        if not soc_version:
            soc_version = AicoreErrorParser.get_soc_version_from_cce(cce_file)
        utils.print_debug_log(f"Get soc_version of {soc_version}.")
        return soc_version

    def parse(self: any) -> None:
        """
        parse by collection info
        """
        utils.print_info_log('*******************Analysis*******************')
        summary_info_list = []

        self.add_objdump_to_path()
        # 检查日志是L0还是L1，以及是否为ffts+场景
        self.check_plog_info()

        # 1.收集aicore error的基本信息
        utils.print_info_log("Step 1. Extract operator information, including registers, tiling, and operator files.")
        info = self.get_op_info()

        # 2.创建对应的aicore error文件夹
        info.err_time_obj = utils.strplogtime(info.err_time)
        err_i_folder_name = f"aicerror_0_{time.strftime('%Y%m%d%H%M%S', info.err_time_obj.timetuple())}"
        err_i_folder = os.path.join(self.collect_path, err_i_folder_name)
        utils.print_info_log(f"Step 2. Create a directory for storing parsing result files."
                             f"The directory is {err_i_folder}")
        utils.check_path_valid(err_i_folder, isdir=True, output=True)

        # 3. 分析图中的算子信息(非必须)
        utils.print_info_log(f"Step 3. Extract the node information of an operator in the GE graph.")
        graph_file = self._get_graph_file()
        info.op_in_graph = self._get_op_by_graph(graph_file, info.node_name, info.kernel_name)

        # 4. 分析args before 以及 args after的区别(非必须)
        utils.print_info_log(f"Step 4. Extract and compare the data between 'args before' and 'args after'.")
        info.args_after_list = self._get_args_after_exc()
        info.args_before_list = self._get_args_before_exc()
        info.check_args_result = self._check_args(info.args_before_list, info.args_after_list)

        # 5. 反编译kernel文件
        utils.print_info_log(f"Step 5. Decompile the operator file, which triggers an instruction.")
        kernel_meta_path = os.path.join(self.collect_path, "collection", "compile")
        # 反编译  出错指令
        result = self._decompile(kernel_meta_path, err_i_folder, info)
        if not result:
            utils.print_warn_log(f"decompile kernel_meta file \
                {os.path.join(kernel_meta_path, info.kernel_name)}.o failed.")

        # 6. 检查框架是否正确插入memset
        utils.print_info_log(f"Step 6. Check whether memset or atomic_clean is correctly inserted"
                             f" before the operator in the graph.")
        info.atomic_clean_check = self._check_atomic_clean(kernel_meta_path, info)
        # 日志报错有0x800000，并且插入了memset才进行累加误差相关的检查
        if ("0x800000" == info.error_code) and (not info.atomic_clean_check):
            info.atomic_add_err = self._get_atomic_err_log()

        # 7. 原方法检查输入输出地址是否在合法范围已放弃，使用exceptiondump是否成功来判断内存问题并解析dump数据
        utils.print_info_log(f"Step 7. Parse dump data and check whether flushing data to disk is normal.")
        collect_dump_data = os.path.join(self.collect_path, "collection", "dump")
        dump_parser = DumpDataParser(collect_dump_data, info.node_name, info.kernel_name)
        info.dump_info = dump_parser.parse()
        if self.parse_level == 1:
            info.data_dump_result = self._get_data_dump_result()
            info.ffts_addrs_num = self.get_ffts_addrs_num()
        else:
            dfx_message = dump_parser.get_dfx_message()
            info.data_dump_result = self.check_dump_result(dfx_message, info)
        if info.data_dump_result:
            info.input_list = dump_parser.get_input_data()
            info.output_list = dump_parser.get_output_data()
            info.workspace_list = dump_parser.get_workspace_data()
            info.bin_list = dump_parser.get_bin_data()
            info.workspace = self.get_workspace_info(self.parse_level, info.workspace_list)
        else:
            info.dump_info = "Failed to get dump data of error op!"

        # 8. 解析二级指针
        utils.print_info_log(f"Step 8. Check whether the pointer tensor exists.")
        if self.parse_level == 0:
            info.sub_ptr_addrs = self._get_sub_ptr(info)
        else:
            utils.print_warn_log(f"The current Log is L1 exception."
                                  "If the operator contains pointer tensors,"
                                  "msaicerr tool does not support the operator.")
        
        # 9. 单算子验证
        utils.print_info_log(f"Step 9. Verify a single operator.")
        info.run_device_id = self.device_id
        if info.data_dump_result:
            info.single_op_test_result, info.single_op_mem_monitor, info.single_op_log_path = \
                                                           self._test_single_op(info, err_i_folder, "single_op")
        if not info.single_op_test_result:
            self.check_hash_id(info.hash_id, info.single_op_log_path)
        else:
            error_op_test_result, error_op_test_mem_monitor, error_op_test_log_path = \
                                                            self._test_single_op(info, err_i_folder, "error_single_op")
            self.check_hash_id(info.hash_id, error_op_test_log_path)

        if not info.single_op_test_result:
            utils.print_error_log(f"Exec single op case failed!")
        else:
            utils.print_debug_log(f"Exec single op case succ!")

        # 10. 使用标杆算子测试环境
        utils.print_info_log(f"Step 10. Verify the environment using the sample operator.")
        soc_version = self.get_soc_version(info.cce_file)
        try:
            info.env_available = AicoreErrorParser.run_test_env(soc_version, self.device_id)
        except Exception as e:
            utils.print_error_log("run golden op failed.Test env skip!")

        # write info file
        utils.print_info_log(f"Step 11. Write the parsing result."
                             f" The result file is saved in the {self.collect_path} directory.")
        self._write_errorinfo_file(err_i_folder, info)

        summary_info_list.append(
            "%s   %s   device_id=%s   core_id=%s   task_id=%s   node=%s   "
            "kernel=%s" % (err_i_folder_name, info.error_code, info.dev_id,
                           info.core_id, info.task_id, info.node_name,
                           info.kernel_name))

        # write summary info
        self._write_summary_file(summary_info_list)

        return self.get_return_code(info)

    @staticmethod
    def get_return_code(aic_info: AicErrorInfo):
        if not aic_info.data_dump_result:
            return Constant.MS_AICERR_MEMORY_ALLOCATION_ERR
        elif not aic_info.env_available:
            return Constant.MS_AICERR_HARDWARE_ERR  # 103
        elif not aic_info.single_op_test_result:
            return Constant.MS_AICERR_SINGLE_OP_ERR
        elif "data invalid" in aic_info.dump_info:
            return Constant.MS_AICERR_OPERATOR_INPUT_DATA_ERR
        elif not aic_info.atomic_clean_check:
            return Constant.MS_AICERR_FRAMEWORK_MEMSET_MISSING
        elif not aic_info.check_args_result:
            return Constant.MS_AICERR_OPERATOR_ARGS_OVERWRITTEN
        elif not aic_info.atomic_add_err:
            return Constant.MS_AICERR_ATOMIC_OPERATOR_OVERFLOW
        else:
            return Constant.MS_AICERR_NONE_ERROR
