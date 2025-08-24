#!/usr/bin/env python
# coding=utf-8
"""
Function:
The file mainly involves main function of parsing input arguments.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import sys
import os
import time
import argparse
import tarfile
import traceback
from ms_interface import utils
from ms_interface.tiling_data_parser import TilingDataParser
from ms_interface.collection import Collection
from ms_interface.constant import Constant
from ms_interface.aicore_error_parser import AicoreErrorParser
from ms_interface.dsmi_interface import DSMIInterface
from ms_interface.dump_data_parser import DumpDataParser


def handle_exception(exc_type, exc_value, exc_traceback):
    utils.GLOBAL_RESULT = False
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print('Uncaught exception:')
    traceback.print_exception(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_exception


def extract_tar(tar_file, path):
    tar = tarfile.open(tar_file, "r")
    tar.extractall(path)
    tar.close()


def get_select_dir(path):
    subdir = os.listdir(path)
    if len(subdir) != 1:
        raise ValueError("[ERROR] found more than one subdir in collect tar")
    report_path = os.path.join(path, subdir[0])
    return report_path


def analyse_report_path(args):
    try:
        current_path = os.getcwd()
        input_path = os.path.abspath(args.report_path)
        if current_path.find(input_path) >= 0:
            utils.print_error_log("Do not run msaicerr in the directory specified by -p or its subdirectory." \
                        " Make sure -out specifies a different directory (including its subdirectory) from -p.")
            return Constant.MS_AICERR_INVALID_PATH_ERROR
        collect_time = time.localtime()
        cur_time_str = time.strftime("%Y%m%d%H%M%S", collect_time)
        utils.check_path_valid(os.path.realpath(args.output_path), isdir=True, output=True)
        output_path = os.path.join(os.path.realpath(args.output_path), "info_" + cur_time_str)
        utils.check_path_valid(output_path, isdir=True, output=True)
        # 解压路径存在就不需要再次解压了
        if not args.report_path and args.tar_file:
            utils.print_info_log("Start to unzip tar.gz package.")
            extract_path = "extract_" + cur_time_str
            extract_tar(args.tar_file, extract_path)
            args.report_path = get_select_dir(extract_path)

        # collect info
        collection = Collection(args.report_path, output_path)
        collection.collect()

        # parse ai core error
        parser = AicoreErrorParser(output_path, args.device_id)
        return parser.parse()

    except utils.AicErrException as error:
        utils.print_error_log(
            f"The aicore error analysis tool has an exception, and error code is {error.error_info}.")
        return Constant.MS_AICERR_INVALID_PATH_ERROR


# noinspection PyBroadException
def convert_dump_data(data_path):
    try:
        DumpDataParser(data_path).parse()
        return Constant.MS_AICERR_NONE_ERROR
    except BaseException:
        utils.print_error_log("Enter a path where data exists.")
        return Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR


def parse_tiling_data(plog):
    try:
        tiling_data = TilingDataParser(plog).parse()
        tiling_data_file = f"tilingdata_{int(time.time())}.bin"
        with open(tiling_data_file, "wb") as f:
            f.write(tiling_data)
        utils.print_info_log(f"Tiling data saved in {tiling_data_file}")
        return Constant.MS_AICERR_NONE_ERROR
    except BaseException:
        return Constant.MS_AICERR_INVALID_SLOG_DATA_ERROR


def verify_device_id(device_id):
    total_device_count = DSMIInterface().get_device_count()
    utils.print_info_log(f"Total device count: {total_device_count}")
    if device_id < 0 or device_id >= total_device_count:
        return False
    return True


def test_env(device_id=0):
    try:
        soc_version = DSMIInterface().get_chip_info(0).get_complete_platform()
        utils.print_info_log(f"Get soc_version: {soc_version}")
        utils.print_info_log("Start to test env with golden op.")
        result = AicoreErrorParser.run_test_env(soc_version, device_id=device_id)
        if result:
            return Constant.MS_AICERR_NONE_ERROR
        else:
            return Constant.MS_AICERR_HARDWARE_ERR
    except BaseException:
        return Constant.MS_AICERR_HARDWARE_ERR


def main() -> int:
    """
    main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--report_path", dest="report_path", default="",
        help="The decompression directory of the tar package from npucollector.", required=False)
    parser.add_argument(
        "-out", "--output_path", dest="output_path", default="",
        help="The output address of the analysis report.", required=False)
    parser.add_argument(
        "-d", "--data", dest="data", default="",
        help="The data to be parsed as npy or bin file.", required=False)
    parser.add_argument(
        "-t", "--tiling_data", dest="tiling_data", default="",
        help="Get tiling data from plog.", required=False)
    parser.add_argument(
        "-env", "--test_env", action="store_true",
        help="The test environment of the npu.", required=False)
    parser.add_argument(
        "-dev", "--device_id", dest="device_id", default=0, type=int,
        help="Sets the device id for the operator. Default value is 0.", required=False)

    ascend_opp_path = os.environ.get("ASCEND_OPP_PATH")
    if not ascend_opp_path:
        utils.print_error_log("Please execute : source ${install_path}/latest/bin/setenv.bash")
        return Constant.MS_AICERR_INVALID_PATH_ERROR
    
    if len(sys.argv) <= 1:
        utils.print_error_log("Please execute : python msaicerr.py -h")
        parser.print_usage()
        return Constant.MS_AICERR_INVALID_PARAM_ERROR
    args = parser.parse_args(sys.argv[1:])

    if not args.output_path:
        utils.print_info_log("The tool directory will be used to as the output address of the analysis report.")
        args.output_path = os.getcwd()

    if not verify_device_id(args.device_id):
        utils.print_error_log(f"Invalid device_id {args.device_id}")
        return Constant.MS_AICERR_INVALID_PARAM_ERROR

    if args.data:
        return convert_dump_data(args.data)
    elif args.report_path:
        return analyse_report_path(args)
    elif args.tiling_data:
        utils.print_info_log("Start to get tiling data.")
        return parse_tiling_data(args.tiling_data)
    elif args.test_env:
        return test_env(args.device_id)
    else:
        utils.print_error_log("Invalid argument, please run help to check the usage.")
        return Constant.MS_AICERR_INVALID_PARAM_ERROR


if __name__ == '__main__':
    sys.exit(main())
