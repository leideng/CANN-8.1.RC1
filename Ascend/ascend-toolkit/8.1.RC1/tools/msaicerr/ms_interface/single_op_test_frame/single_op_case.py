#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves the common function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import re
import os
import subprocess
from time import sleep
import shutil
import platform
from ms_interface.aic_error_info import AicErrorInfo
from ms_interface.dsmi_interface import DSMIInterface
import numpy as np
from ms_interface import utils
from ms_interface.single_op_test_frame.common.ascend_tbe_op import AscendOpKernel, AscendOpKernelRunner


class SingleOpCase:

    def __init__(self, aic_info: AicErrorInfo, op_test: "single_op_test") -> None:
        self.aic_info = aic_info

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
            sleep(0.2)
            current_log_size = os.path.getsize(log_path)
            if current_log_size == log_size:
                break
            log_size = current_log_size

    @staticmethod
    def search_aicerr_log(kernel_name, path):
        for root, _, files in os.walk(path):
            for file in files:
                if not file.endswith(".log"):
                    continue
                log_path = os.path.abspath(os.path.join(root, file))
                utils.print_info_log(f"The find single op log {log_path}")
                SingleOpCase._wait_for_log_stabilization(log_path)
                with open(log_path, "r") as f:
                    content = f.read()
                if SingleOpCase._check_file_content(kernel_name, content):
                    return True
        return False

    def generate_config(self):
        config = {
            "cce_file": self.get_cce_file(),
            "bin_path": self.aic_info.bin_file,
            "json_path": self.aic_info.json_file,
            "tiling_data": self.aic_info.tiling_data,
            "tiling_key": self.aic_info.tiling_key,
            "block_dim": self.aic_info.block_dim,
            "input_file_list": self.aic_info.input_list,
            "output_file_list": self.aic_info.output_list,
            "workspace_file_list": self.aic_info.workspace_list,
            "bin_file_list": self.aic_info.bin_list,
            "kernel_name": self.aic_info.kernel_name,
            "device_id": self.aic_info.run_device_id,
            "sub_ptr_addrs": self.aic_info.sub_ptr_addrs,
            "ffts_addrs_num": self.aic_info.ffts_addrs_num,
            "workspace": self.aic_info.workspace
        }
        return config

    @staticmethod
    def get_soc_version_from_cce(cce_file):
        try:
            with open(cce_file, 'r') as f:
                content = f.read()
            soc_version_ret = re.findall(r'//.*?(Ascend.*?)"', content)
            if soc_version_ret:
                utils.print_info_log(
                    f"get soc_version {soc_version_ret[0]} from cce file {cce_file}")
                if soc_version_ret[0] == "Ascend910B":
                    return "Ascend910B1"
                elif soc_version_ret[0] == "Ascend310B":
                    return "Ascend310B1"
                return soc_version_ret[0]
            else:
                utils.print_warn_log(
                    'Can not get soc_version from cce file {cce_file}')
                return "Ascend310"
        except Exception as e:
            utils.print_warn_log(
                'Can not get soc_version from cce file {cce_file}')
            return "Ascend310"

    def get_cce_file(self):
        kernel_path = self.aic_info.kernel_path
        kernel_name = self.aic_info.kernel_name
        tiling_key = self.aic_info.tiling_key
        cce_file = self.aic_info.cce_file
        if not os.path.exists(cce_file):
            cce_file = os.path.join(
                kernel_path, f"{kernel_name}_{tiling_key}.cce")
            if not os.path.exists(cce_file):
                utils.print_warn_log(
                    f"The cce file:{cce_file} does not exist")
                return ""
        return cce_file

    @staticmethod
    def update_kernel_by_cce(cce_file, kernel_name):
        if not os.path.exists(cce_file):
            utils.print_info_log("Does not get cce file !!!")
            return None

        with open(cce_file, 'r') as f:
            content = f.read()
        cce_pattern = "(?<=//\s).+$"
        re_result = re.findall(cce_pattern, content)
        if not re_result:
            utils.print_info_log(
                "Does not match ccec command, use origin kernel to run single op!")
            return None
        cmd = re_result[0].split(" ")
        ccec_file = shutil.which("ccec")
        if not ccec_file:
            # guess where is ccec
            parent_path = "aarch64-linux" if "aarch64" in platform.machine() else "x86_64-linux"
            ccec_file_guess = os.path.join(
                "usr", "local" "Ascend", "latest", parent_path, "ccec_compiler", "bin" "ccec")
            if shutil.which(ccec_file_guess):
                ccec_file = ccec_file_guess
            else:
                utils.print_warn_log(
                    "Cannot find ccec! please add ccec path in env PATH.")
                return None
        os.environ['PATH'] = os.path.dirname(
            ccec_file) + ":" + os.environ["PATH"]
        rename_o_file = os.path.join(
            os.getcwd(), kernel_name + "_new.o")
        cmd[3] = cce_file
        dst_bin_index = cmd.index("-o") + 1
        cmd[dst_bin_index] = rename_o_file
        subprocess.run(cmd)
        return rename_o_file

    @staticmethod
    def run_dirty_ub(configs):
        # Step 1. get soc_version to compile dirty_ub
        try:
            soc_version = DSMIInterface().get_chip_info(0).get_complete_platform()
        except BaseException:
            utils.print_warn_log("get soc_version form platform failed!")
            soc_version = None
        if not soc_version:
            soc_version = SingleOpCase.get_soc_version_from_cce(
                configs.get("cce_file"))
        utils.print_info_log(f"get soc_version of {soc_version}.")

        # Step 2. compile dirty_ub kernel
        kernel_name = f"dirty_ub_{soc_version}"
        kernel_name = kernel_name.replace('-', '_')
        find_path_cmd = ["find", "./kernel_meta", "-name", f"{kernel_name}*"]
        try:
            SingleOpCase.dirty_ub(
                soc_version, kernel_name=kernel_name)
        except Exception as e:
            utils.print_warn_log("compile diry_ub op failed, skip dirty ub")
            return

        # Step 3. find dirty_ub kernel
        regexp = r"([_\-/0-9a-zA-Z.]{1,}\.json|[_\-/0-9a-zA-Z.]{1,}\.o|[_\-/0-9a-zA-Z.]{1,}\.cce)"
        kernel_file_list = utils.get_inquire_result(find_path_cmd, regexp)
        if not kernel_file_list:
            utils.print_warn_log(
                f"The {kernel_name} file path cannot be found.")
        for file in kernel_file_list:
            if file.endswith(".o"):
                bin_path = file
            elif file.endswith(".json"):
                json_path = file
            else:
                continue
        if not os.path.exists(bin_path) or not os.path.exists(json_path):
            utils.print_info_log(f"Can not find bin_file  and json_file ")
            return

        device_id = configs.get("device_id")
        try:
            device_id = int(device_id)
        except ValueError:
            utils.print_warn_log("device_id should be an integer, device set default 0")
            device_id = 0

        # Step 4. run dirty_ub kernel
        utils.print_info_log(
            f"Find bin_file {bin_path} and json_file {json_path}")
        op_kernel = AscendOpKernel(bin_path, json_path)
        # kernel without output can not run
        output_info = {}
        output_info["size"] = 4
        output_info["dtype"] = "float32"
        output_info["shape"] = (1,)
        with AscendOpKernelRunner(device_id=device_id) as runner:
            runner.run(op_kernel, inputs=[], actual_output_info=(output_info,))

    @staticmethod
    def dirty_ub(soc_version, kernel_name="dirty_ub"):
        try:
            from te import tik
            from tbe.common import platform as cce
            from tbe.common.platform import set_current_compile_soc_info as te_set_version
        except ImportError as e:
            utils.print_warn_log(
                "failed to import te or tbe to compile op dirty_ub, skipped it. error:", e)
            return
        te_set_version(soc_version)
        ub_size = cce.get_soc_spec("UB_SIZE")

        tik_instance = tik.Tik()

        output_gm = tik_instance.Tensor(
            "float32", (1,), name="output_gm", scope=tik.scope_gm)
        all_ub = tik_instance.Tensor(
            "float32", (ub_size // 4,), name="all_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, ub_size // 256) as loop_idx:
            tik_instance.vec_dup(
                64, all_ub[loop_idx * 64], 1.7976931348623157e+30, 1, 8)
        tik_instance.BuildCCE(kernel_name=kernel_name,
                              inputs=[], outputs=[output_gm])

    @staticmethod
    def read_bin_file(bin_name):
        with open(bin_name, 'rb') as f:
            data = f.read()
        return data

    @staticmethod
    def get_io_data_list(data):
        input_file_list = data.get("input_file_list", [])
        output_file_list = data.get("output_file_list", [])
        input_data_list = []
        for file in input_file_list:
            input_data = np.load(file)
            input_data_list.append(input_data)
        output_info_list = []
        for file in output_file_list:
            output_info = {}
            np_data = np.load(file)
            output_info["size"] = np_data.nbytes
            if str(np_data.dtype) == "|V2":
                utils.print_warn_log(
                    "np_data.dtype is V2, maybe bfloat16, same size with float16")
                output_info["dtype"] = "float16"
            else:
                output_info["dtype"] = str(np_data.dtype)
            output_info["shape"] = np_data.shape
            output_info_list.append(output_info)
        return input_data_list, output_info_list


    @staticmethod
    def run_kernel(data, op_test) -> str:
        kernel_name = data.get('kernel_name')
        cce_file = data.get('cce_file')
        if cce_file and os.path.exists(cce_file):
            utils.print_info_log("Generate new kernel by cce_file")
            bin_path = SingleOpCase.update_kernel_by_cce(cce_file, kernel_name)
            if bin_path is None:
                utils.print_warn_log("update_kernel_by_cce failed, please check cce file.")
                bin_path = data.get("bin_path")
        else:
            bin_path = data.get("bin_path")
        json_path = data.get("json_path")
        tiling_data = data.get("tiling_data")
        if tiling_data and isinstance(tiling_data, str) and tiling_data.endswith(".bin"):
            tiling_data = SingleOpCase.read_bin_file(tiling_data)
        else:
            tiling_data = tiling_data.encode("utf-8")
        tiling_key = data.get("tiling_key")
        block_dim = data.get("block_dim")
        input_data_list, output_info_list = SingleOpCase.get_io_data_list(data)
        device_id = data.get("device_id")
        try:
            device_id = int(device_id)
        except ValueError:
            utils.print_warn_log("device_id should be an integer, device set default 0")
            device_id = 0

        op_kernel = AscendOpKernel(bin_path, json_path)
        with AscendOpKernelRunner(device_id=device_id) as runner:
            ret = runner.run(op_kernel,
                       inputs=input_data_list,
                       tiling_data=tiling_data,
                       block_dim=block_dim,
                       tiling_key=tiling_key,
                       actual_output_info=output_info_list,
                       bin_list=data.get("bin_file_list"),
                       sub_ptr_addrs=data.get("sub_ptr_addrs"),
                       ffts_addrs_num=data.get("ffts_addrs_num"),
                       workspace=data.get("workspace"),
                       op_test=op_test)
        return ret

    @staticmethod
    def run(configs: dict, op_test: str) -> str:
        # set single op log path
        utils.print_info_log(f"Start run dirtyub test case...")
        SingleOpCase.run_dirty_ub(configs)

        utils.print_info_log(f"Start run kernel test case...")

        ret = SingleOpCase.run_kernel(configs, op_test)
        ret_str = f"Execute {op_test} SingleOpCase.run_kernel result: \r\n{ret}"
        return f"{ret_str}"

