#!/usr/bin/env python
# coding=utf-8
"""
Function:
Collection class. This file mainly involves the collect function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import hashlib
from ms_interface import utils
from ms_interface.constant import Constant


class Collection:
    def __init__(self: any, report_path: str, output_path: str) -> None:
        self.report_path = os.path.realpath(report_path)
        self.output_path = os.path.realpath(output_path)
        self.collect_level = 0
        self.ffts_flag = False

    def check_argument_valid(self: any) -> None:
        utils.check_path_valid(self.report_path, isdir=True)
        utils.check_path_valid(self.output_path, isdir=True, output=True)

    def get_node_and_kernel_name_l1(self: any) -> list:
        plog_dir = os.path.join(self.output_path, 'collection', 'plog')
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
        regexp = r".+?node_name:(.*?),"
        result = utils.get_inquire_result(node_name_cmd, regexp)
        if not result:
            utils.print_error_log(f"Failed to get node name in plog. Cannot run L1 test.")
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_SLOG_DATA_ERROR)
        node_name = result[0]
        node_name = node_name.replace('/', '_').replace('.', '_')
        return kernel_name, node_name

    def get_kernel_name_l0(self: any, data_name) -> list:
        # 获取kernel_name
        plog_dir = os.path.join(self.output_path, 'collection', 'plog')
        if not self.ffts_flag:
            kernel_name_cmd = ['grep', 'Aicore kernel execute failed', '-inrE', plog_dir]
            kernel_name_regexp = r".*?fault kernel_name=(.*?),.*?fault kernel info ext=(.*?),"
            kernel_name_ret = utils.get_inquire_result(kernel_name_cmd, kernel_name_regexp)
            if kernel_name_ret and kernel_name_ret[0][1] != "none":
                kernel_name = kernel_name_ret[0][1]
                node_name = data_name
                utils.print_debug_log(f"AicoreError Found, kernel_name {kernel_name}, node_name {node_name}")
                return kernel_name, node_name

            kernel_name_regexp = r" .*?fault kernel_name=(.*?),"
            kernel_name_ret = utils.get_inquire_result(kernel_name_cmd, kernel_name_regexp)
            if not kernel_name_ret:
                utils.print_error_log(f"Failed to get \"Aicore kernel execute failed\" in plog.")
                raise utils.AicErrException(Constant.MS_AICERR_INVALID_SLOG_DATA_ERROR)

            kernel_name = kernel_name_ret[0]
            node_name = data_name
            utils.print_debug_log(f"AicoreError Found, kernel_name {kernel_name}, node_name {node_name}")

            
            return kernel_name, node_name
        else:
            kernel_name_cmd = ['grep', 'fftsplus task execute failed', '-inrE', plog_dir]

            kernel_name_regexp = r".*?fault kernel_name=(.*?),"
            kernel_name_ret = utils.get_inquire_result(kernel_name_cmd, kernel_name_regexp)
            if not kernel_name_ret:
                utils.print_error_log(f"Failed to get \"fftsplus task execute failed\" in plog.")
                raise utils.AicErrException(Constant.MS_AICERR_INVALID_SLOG_DATA_ERROR)

            kernel_name = kernel_name_ret[0]
            node_name = data_name
            utils.print_debug_log(f"AicoreError Found, kernel_name {kernel_name}, node_name {node_name}")

            return kernel_name, node_name

    def _get_node_and_kernel_name(self: any, data_name) -> list:
        if self.collect_level == 1:
            kernel_name, node_name = self.get_node_and_kernel_name_l1()
        else:
            kernel_name, node_name = self.get_kernel_name_l0(data_name)
        return kernel_name, node_name

    def get_dump_data_info(self):
        plog_dir = os.path.join(self.output_path, 'collection', 'plog')
        if self.collect_level == 1:
            dump_data_cmd = ['grep', 'dump exception to file', '-inrE', plog_dir]
            adump_dump_data_regexp = r"(\d+-\d+-\d+-\d+:\d+:\d+\.\d+\.\d+).+?tid\:\d+" \
                                     r".*?extra-info\/data-dump\/(\d+)\/([\w.]+)"
            ge_dump_data_regexp = r"(\d+) DumpNodeInfo:.*?extra-info\/data-dump\/(\d+)\/([\w.]+)"
            adump_dump_data_ret = utils.get_inquire_result(dump_data_cmd, adump_dump_data_regexp)
            ge_dump_data_ret = utils.get_inquire_result(dump_data_cmd, ge_dump_data_regexp)
            if not adump_dump_data_ret and not ge_dump_data_ret:
                utils.print_error_log(f"Check whether open exception dump.")
                raise utils.AicErrException(Constant.MS_AICERR_INVALID_PATH_ERROR)
            if adump_dump_data_ret:
                err_time, device_id, data_name = adump_dump_data_ret[0]
                return err_time, device_id, data_name
            else:
                err_time, device_id, data_name = ge_dump_data_ret[0]
                return err_time, device_id, data_name

        else:
            dump_data_cmd = ['grep', 'dump exception to file', '-inrE', plog_dir]

            dump_data_regexp = r"(\d+-\d+-\d+-\d+:\d+:\d+\.\d+\.\d+).+?tid\:\d+" \
                               r".*?extra-info\/data-dump\/(\d)+\/([\w.]+)"
            dump_data_ret = utils.get_inquire_result(dump_data_cmd, dump_data_regexp)
            if not dump_data_ret:
                utils.print_error_log(f"Check whether open exception dump.")
                raise utils.AicErrException(Constant.MS_AICERR_INVALID_PATH_ERROR)
            err_time, device_id, data_name = dump_data_ret[0]
            return err_time, device_id, data_name

    def collect_plog_file(self):
        find_path_cmd = ['grep',
                         'there is an .*aicore.* error|there is an .*aivec.* error',
                         '-inrE', self.report_path]
        find_path_regexp = r"(/[_\-/0-9a-zA-Z.]{1,}.[log|txt]):"
        plog_path_ret = utils.get_inquire_result(find_path_cmd, find_path_regexp)

        if plog_path_ret:
            original_files = plog_path_ret
        else:
            utils.print_error_log(f"Aicore error log 'there is an' cannot be found in {self.report_path}.")
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_PATH_ERROR)

        ffts_check_path_cmd = ['grep',
                               'fftsplus task execute failed',
                               '-inrE', self.report_path]
        ffts_check_path_regexp = r"(/[_\-/0-9a-zA-Z.]{1,}.[log|txt]):"
        ffts_check_path_ret = utils.get_inquire_result(ffts_check_path_cmd, ffts_check_path_regexp)
        if ffts_check_path_ret:
            self.ffts_flag = True

        original_files = list(set(original_files))
        dest_path = os.path.join(self.output_path, 'collection', 'plog')
        utils.check_path_valid(dest_path, isdir=True, output=True)
        utils.copy_src_to_dest(original_files, os.path.join(dest_path, "aicore_error"))

        find_path_cmd = ['grep', "\[AIC_INFO\] dev_func:", '-inrE', self.report_path]
        find_path_regexp = r"(/[_\-/0-9a-zA-Z.]{1,}.[log|txt]):"
        plog_path_ret_1 = utils.get_inquire_result(find_path_cmd, find_path_regexp)

        if plog_path_ret_1:
            self.collect_level = 1
            original_file = sorted(plog_path_ret_1)[0]
            if original_file not in plog_path_ret:
                utils.copy_src_to_dest([original_file, ], os.path.join(dest_path, "exception_dump"))
        else:
            utils.print_debug_log(f"'[AIC_INFO] dev_func:' cannot be found in {self.report_path}. "
                                 "Only run L0 parse")
        utils.print_debug_log(f"Debug Level is {self.collect_level}")

        find_path_cmd = ['grep', "exception info dump args data", '-inrE', self.report_path]
        find_path_regexp = r"(/[_\-/0-9a-zA-Z.]{1,}.[log|txt]):"
        plog_path_ret_2 = utils.get_inquire_result(find_path_cmd, find_path_regexp)

        if plog_path_ret_2:
            original_file = sorted(plog_path_ret_2)[0]
            if original_file not in plog_path_ret:
                utils.copy_src_to_dest([original_file, ], os.path.join(dest_path, "exception_dump"))

        return dest_path

    def collect_kernel_file(self, kernel_name):
        original_files = []
        op_json = False
        op_kernel = False
        kernel_name = kernel_name.replace("__kernel0", "").replace("_mix_aic", "") \
                                 .replace("_mix_aiv", "")
        find_path_cmd = ['grep', kernel_name, '-inrE', self.report_path]
        regexp = r"([_\-/0-9a-zA-Z.]{1,}\.json|[_\-/0-9a-zA-Z.]{1,}\.o|[_\-/0-9a-zA-Z.]{1,}\.cce)"
        kernel_file_list = utils.get_inquire_result(find_path_cmd, regexp)
        if not kernel_file_list:
            utils.print_error_log(f"Kernel file cannnot find. "
                                  f"Please move {kernel_name}`s related file to {self.report_path}.")
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_PATH_ERROR)
        kernel_file_list = list(set(kernel_file_list))
        for kernel_file in kernel_file_list:
            if os.path.exists(kernel_file):
                original_files.append(kernel_file)
            if not kernel_file.endswith("host.o") and kernel_file.endswith(".o"):
                op_kernel = True
            if kernel_file.endswith(".json"):
                op_json = True
        if (not op_json) or (not op_kernel):
            utils.print_error_log(f"The {kernel_name}`s related file cannot be found in {self.report_path}.")
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_PATH_ERROR)
        original_files = list(set(original_files))
        dest_path = os.path.join(self.output_path, "collection", "compile")
        utils.check_path_valid(dest_path, isdir=True, output=True)
        utils.copy_src_to_dest(original_files, dest_path)
        return dest_path

    def collect_ge_graph(self):
        find_path_cmd = ['find', self.report_path, '-name', "ge_proto_*_Build.txt"]
        regexp = r"([_\-/0-9a-zA-Z.]{1,}_Build.txt)"
        graph_file_list = utils.get_inquire_result(find_path_cmd, regexp)
        if not graph_file_list:
            utils.print_warn_log(
                f"Graph file cannot be collected, the graph file cannot be found in {self.report_path}.")
        original_files = graph_file_list
        dest_path = os.path.join(self.output_path, "collection", "graph")
        utils.check_path_valid(dest_path, isdir=True, output=True)
        utils.copy_src_to_dest(original_files, dest_path)
        return dest_path

    def collect_data_dump(self, device_id, data_name):
        dest_path = os.path.join(self.output_path, "collection", "dump")
        find_path_cmd = ['find', self.report_path, '-name',
                         f"{data_name}"]
        regexp = r"[_\.\-/0-9a-zA-Z.]{1,}"
        original_files = utils.get_inquire_result(find_path_cmd, regexp)
        if not original_files:
            utils.print_error_log(
                f"Dump file cannot be collected, the dump file cannot be found in {self.report_path}.")
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_PATH_ERROR)

        # 如果找到大于1个data, 则匹配日志中的data_dump和device_id
        if len(original_files) > 1:
            plog_dir = os.path.join(self.output_path, 'collection', 'plog')
            for file in original_files:
                data_dump_cmd = ['grep', os.path.basename(file), '-nr', plog_dir]
                dump_data_regexp = r".*?extra-info\/data-dump\/(\d+)\/[\w.]+"
                data_dump_ret = utils.get_inquire_result(data_dump_cmd, dump_data_regexp)
                if (device_id != data_dump_ret[0]):
                    continue
                utils.print_info_log(f"Find dump file {os.path.basename(file)}.")
                original_files = [file]

        utils.check_path_valid(dest_path, isdir=True, output=True)
        utils.copy_src_to_dest(original_files, dest_path)
        return dest_path

    def check_dump_data_is_valid(self, err_time, data_name):
        find_dump_data_cmd = ['find', self.report_path, '-name', data_name]
        regexp = r".*?\/data-dump\/\d+\/([\w.]+)"
        dump_data_file_list = utils.get_inquire_result(find_dump_data_cmd, regexp)
        home = os.environ.get("HOME")
        if not dump_data_file_list:
            utils.print_error_log(f"Cannot find dump file {data_name} when analyzing the AI Core error"
                                  f" generated at {err_time}. Possible causes: 1. The dump file is missing in the"
                                  f" {self.report_path} directory. Add it to the directory."
                                  f" 2. The analyzed log is not the one for the AI core error."
                                  f" According to the time window of the training task,"
                                  f" retain the log file within the time window"
                                  f"({home}/ascend/log/debug/plog/plog-pid_FileCreationTimestamp.log file),"
                                  f" and delete the one beyond the time window"
                                  f"({home}/ascend/log/debug/plog/plog-pid_FileCreationTimestamp.log)")
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR)

    def check_host_and_device_hash(self, data_name):
        kernel_cmd = ['find', self.report_path, '-name', data_name]
        _, kernel_info = utils.execute_command(kernel_cmd)
        kernel_path = kernel_info.split(data_name)[0]
        res = os.listdir(kernel_path)
        host_kernel_name = ''
        device_kernel_name = ''
        device_kernel_json = ''
        for file in res:
            if file.endswith('.o') and not file.endswith('host.o'):
                device_kernel_name = file
            if file.endswith('.o') and file.endswith('host.o'):
                host_kernel_name = file
            if file.endswith('.json'):
                device_kernel_json = file
        if not host_kernel_name or not device_kernel_name:
            utils.print_warn_log("Cannot find host kernel or device kernel.")
            return True
        if self.calculate_hash(os.path.join(kernel_path, device_kernel_name)) \
             != self.calculate_hash(os.path.join(kernel_path, host_kernel_name)):
            return False
        return True

    def calculate_hash(self, file_path):
        hash_algorithm = hashlib.md5()
        with open(file_path, 'rb') as file:
            for chunk in iter(lambda: file.read(4096), b''):
                hash_algorithm.update(chunk)
        hash_value = hash_algorithm.hexdigest()
        return hash_value

    def collect(self: any):
        """
        collect info
        """
        utils.print_info_log(f'Check the validity of the input and output paths for file parsing.')
        self.check_argument_valid()
        utils.print_info_log('******************Collection******************')
        collect_path = os.path.join(self.output_path, 'collection')
        utils.check_path_valid(collect_path, isdir=True, output=True)

        # collect plog
        utils.print_info_log(f'Step 1. Check key information in the log and copy the log.')
        plog_dest_path = self.collect_plog_file()

        # get dump data 
        utils.print_info_log('Step 2. Obtain the name and path of the flushed data file from the log.')
        err_time, device_id, data_name = self.get_dump_data_info()
        self.check_dump_data_is_valid(err_time, data_name)
        chech_result = self.check_host_and_device_hash(data_name)
        if not chech_result:
            utils.print_error_log(f"The kernel load on the host is different from the device.")
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR)

        # collect dump
        utils.print_info_log('Step 3. Obtain the operator name from the log.')
        dump_dest_path = self.collect_data_dump(device_id, data_name)

        # get kernel_name
        utils.print_info_log('Step 4. Obtain the compilation file based on the operator name.')
        kernel_name, node_name = self._get_node_and_kernel_name(data_name)

        # collect compile
        utils.print_info_log('Step 5. Start to collect compile file.')
        kernel_dest_path = self.collect_kernel_file(kernel_name)

        # collect_ge_proto_graph
        utils.print_info_log('Step 6. Collect Graph Engine files.')
        proto_dest_path = self.collect_ge_graph()
