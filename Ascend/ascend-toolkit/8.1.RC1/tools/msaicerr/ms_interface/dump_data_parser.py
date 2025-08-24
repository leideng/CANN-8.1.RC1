#!/usr/bin/env python
# coding=utf-8
"""
Function:
DumpDataParser class. This class mainly involves the parser_dump_data function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import struct
import traceback
from typing import BinaryIO
from google.protobuf.message import DecodeError
import numpy as np
from ms_interface import utils
from ms_interface.constant import Constant
import ms_interface.dump_data_pb2 as DD


class ConstManager:
    UINT64_SIZE = 8
    UINT64_FMT = 'Q'
    ONE_GB = 1 * 1024 * 1024 * 1024
    DATA_TYPE_TO_DTYPE_MAP = {
        DD.DT_UNDEFINED: {
            Constant.DTYPE: "undefined",
            Constant.STRUCT_FORMAT_KEY: 'b'
        },
        DD.DT_FLOAT: {
            Constant.DTYPE: np.float32,
            Constant.STRUCT_FORMAT_KEY: 'f'
        },
        DD.DT_FLOAT16: {
            Constant.DTYPE: np.float16,
            Constant.STRUCT_FORMAT_KEY: 'e'
        },
        DD.DT_DOUBLE: {
            Constant.DTYPE: np.float64,
            Constant.STRUCT_FORMAT_KEY: 'd'
        },
        DD.DT_INT8: {
            Constant.DTYPE: np.int8,
            Constant.STRUCT_FORMAT_KEY: 'b'
        },
        DD.DT_INT16: {
            Constant.DTYPE: np.int16,
            Constant.STRUCT_FORMAT_KEY: 'h'
        },
        DD.DT_INT32: {
            Constant.DTYPE: np.int32,
            Constant.STRUCT_FORMAT_KEY: 'i'
        },
        DD.DT_INT64: {
            Constant.DTYPE: np.int64,
            Constant.STRUCT_FORMAT_KEY: 'q'
        },
        DD.DT_UINT8: {
            Constant.DTYPE: np.uint8,
            Constant.STRUCT_FORMAT_KEY: 'B'
        },
        DD.DT_UINT16: {
            Constant.DTYPE: np.uint16,
            Constant.STRUCT_FORMAT_KEY: 'H'
        },
        DD.DT_UINT32: {
            Constant.DTYPE: np.uint32,
            Constant.STRUCT_FORMAT_KEY: 'I'
        },
        DD.DT_UINT64: {
            Constant.DTYPE: np.uint64,
            Constant.STRUCT_FORMAT_KEY: 'Q'
        },
        DD.DT_BOOL: {
            Constant.DTYPE: np.bool_,
            Constant.STRUCT_FORMAT_KEY: '?'
        },
        DD.DT_BF16: {
            Constant.DTYPE: 'bfloat16',
            Constant.STRUCT_FORMAT_KEY: 'e'
        }
    }

    COMMON_DTYPE = ["float32", "float16", "bfloat16", "int32", "int64"]


class DumpDataParser:
    """
    The class for dump data parser
    """

    def __init__(self, dump_path, node_name=None, kernel_name=None):
        self.dump_path = dump_path
        self.node_name = node_name
        self.kernel_name = kernel_name
        self.input_data_list = []
        self.output_data_list = []
        self.workspace_data_list = []
        self.bin_data_list = []
        self.dfx_message = ""

    def get_input_data(self):
        return self.input_data_list

    def get_output_data(self):
        return self.output_data_list

    def get_workspace_data(self):
        return self.workspace_data_list

    def get_bin_data(self):
        return self.bin_data_list
    
    def get_dfx_message(self):
        return self.dfx_message

    @staticmethod
    def _get_dtype_by_data_type(data_type):
        if data_type not in ConstManager.DATA_TYPE_TO_DTYPE_MAP:
            utils.print_error_log(f"The output data type({data_type}) does not support.")
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR)
        dtype = ConstManager.DATA_TYPE_TO_DTYPE_MAP.get(data_type).get(Constant.DTYPE)
        if dtype == 'bfloat16':
            try:
                from bfloat16ext import bfloat16
                dtype = bfloat16
            except ModuleNotFoundError as ee:
                raise TypeError(
                    'bfloat16 is not supported in numpy, run `pip3 install bfloat16ext` for support.') from ee
        return dtype

    @staticmethod
    def _check_tensor_data(index, array, data_dtype):
        result_info = ""
        if (np.isinf(array).any() or np.isnan(array).any()):
            result_info = f'input[{index}] NaN/INF. Input data invalid. Please check!\n'
            utils.print_error_log(result_info)
        else:
            if data_dtype in (np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64):
                dtype_max = np.iinfo(data_dtype).max
                dtype_min = np.iinfo(data_dtype).min
            elif data_dtype in (np.float16, np.float32, np.float64):
                dtype_max = np.finfo(data_dtype).max
                dtype_min = np.finfo(data_dtype).min
            else:
                return ""
            if (np.max(array) > 0.9 * dtype_max) or (np.min(array) < 0.9 * dtype_min):
                result_info = f'input[{index}] max {np.max(array)} or min {np.min(array)}. \
                    Input data maybe invalid. Please check!\n'
                utils.print_error_log(result_info)

        return result_info

    @staticmethod
    def _summary_tensor_without_dtype(tensor_file):
        result_info = ""
        from bfloat16ext import bfloat16
        for dtype in ConstManager.COMMON_DTYPE:
            result_info += " " * 4
            result_info += f"If dtype is {dtype}, summary is: "
            try:
                arr = np.fromfile(tensor_file, dtype=dtype)
                if arr.dtype == bfloat16:
                    arr = arr.astype(np.float32)
                result_info += f"Max: {np.max(arr)}, Min: {np.min(arr)}, Mean: {np.mean(arr)}, Std: {np.std(arr)}\n"
            except BaseException:
                result_info += f"Can not read with dtype {dtype}!\n"
        return result_info

    @staticmethod
    def _summary_tensor_with_dtype(array: np.ndarray):
        result_info = " " * 4
        from bfloat16ext import bfloat16
        result_info += f"Dtype is {array.dtype}, summary is: "
        if array.dtype == bfloat16:
            array = array.astype(np.float32)
        result_info += f"Max:{np.max(array)}, Min:{np.min(array)}, Mean: {np.mean(array)}, Std: {np.std(array)}\n"
        return result_info

    def _save_tensor_to_file(self, tensor_list, tensor_type, dump_file):
        result_info = ''

        if len(tensor_list) == 0:
            utils.print_warn_log(f'There is no {tensor_type} in "{dump_file}".')
            return result_info

        dump_file_path, dump_file_name = os.path.split(dump_file)
        if not self.kernel_name:
            self.kernel_name = dump_file_name

        for (index, tensor) in enumerate(tensor_list):
            try:
                if tensor_type == "workspace":
                    data_dtype = np.int8
                else:
                    data_dtype = self._get_dtype_by_data_type(tensor.data_type)
                if data_dtype == "undefined":
                    array = np.frombuffer(tensor.data, dtype=np.int8)
                    dst_file_name = ".".join([self.kernel_name, tensor_type, str(index), "bin"])
                    dst_file_name = os.path.join(dump_file_path, dst_file_name)
                    self.bin_data_list.append(dst_file_name)
                    array.tofile(dst_file_name)
                    result_info += f'{dst_file_name}\n'
                    result_info += self._summary_tensor_without_dtype(dst_file_name)
                else:
                    array = np.frombuffer(tensor.data, dtype=data_dtype)
                    dst_file_name = ""
                    if "workspace" == tensor_type:
                        dst_file_name = os.path.join(dump_file_path, \
                                        ".".join([self.kernel_name, tensor_type, str(index), str(array.dtype), "npy"]))
                        np.save(dst_file_name, array)
                    else:
                        dst_file_name = os.path.join(dump_file_path, \
                                        ".".join([self.kernel_name, tensor_type, str(index), str(array.dtype), "bin"]))
                        array.tofile(dst_file_name)
                    result_info += f'{dst_file_name}\n'
                    result_info += self._check_tensor_data(index, array, data_dtype)
                    result_info += self._summary_tensor_with_dtype(array)
                    if "workspace" == tensor_type:
                        self.workspace_data_list.append(os.path.join(dump_file_path, dst_file_name))
                    else:
                        self.bin_data_list.append(dst_file_name)
            except (ValueError, IOError, OSError, MemoryError) as error:
                utils.print_error_log(f'Failed to parse the data of {tensor_type}:{index} of "{dump_file}". {error}')
                raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR)
            finally:
                pass
        return result_info

    def _save_dfx_message(self, dump_data):
        self.dfx_message = dump_data.dfx_message
        result_info = f"Dump exception info : {dump_data.dfx_message}"
        return result_info

    def parse_dump_data(self, dump_file):
        """
        Function Description: convert dump data to numpy and bin file
        @param dump_file: the dump file
        """
        try:
            import sys
            current_dir = os.path.abspath(os.path.dirname(__file__))
            compare_dir = os.path.join(current_dir, '..', '..', 'operator_cmp', 'compare')
            sys.path.append(compare_dir)
            dump_data = BigDumpDataParser(dump_file).parse()
            # 2. parse dump data
            result_info = self._save_tensor_to_file(dump_data.input, 'input', dump_file)
            result_info += self._save_tensor_to_file(dump_data.output, "output", dump_file)
            result_info += self._save_tensor_to_file(dump_data.space, "workspace", dump_file)
            result_info += self._save_dfx_message(dump_data)
        except BaseException as e:
            traceback.print_exc()
            utils.print_error_log(f"Failed to parse the data of dump file:{dump_file}, Error Detail: {e}")
        return result_info

    def parse(self):
        """
        Function Description: dump data parse.
        """
        # get parse data list
        if os.path.isfile(self.dump_path):
            match_dump_list = [self.dump_path]
        else:
            match_dump_list = []
            match_name = self.node_name
            for top, _, files in os.walk(self.dump_path):
                for name in files:
                    if match_name in name:
                        match_dump_list.append(os.path.join(top, name))

        # parse data
        result_info_list = []
        for dump_file in match_dump_list:
            if isinstance(dump_file, str) and (dump_file.endswith(".npy") or dump_file.endswith(".bin")):
                continue
            result_info_list.extend([f'{dump_file}\n', "after convert:\n", self.parse_dump_data(dump_file)])
        result_info = "".join(result_info_list)
        if len(match_dump_list) == 0:
            utils.print_warn_log(f'There is no dump file for "{self.node_name}". Please check the dump path.')
        dump_file_path, dump_file_name = os.path.split(dump_file)
        utils.print_info_log(f"Parse dump file finished, result path is : {os.path.abspath(dump_file_path)}")
        utils.print_debug_log(f"Parse dump file finished, result_info:{result_info}")
        return match_dump_list


class BigDumpDataParser:
    """
    The class for big dump data parser
    """

    def __init__(self: any, dump_file_path: str) -> None:
        self.dump_file_path = dump_file_path
        self.file_size = 0
        self.header_length = 0
        self.dump_data = None

    def parse(self: any) -> DD.DumpData:
        """
        Parse the dump file path by big dump data format
        :return: DumpData
        :exception when read or parse file error
        """
        self.check_argument_valid()
        try:
            with open(self.dump_file_path, 'rb') as dump_file:
                # read header length
                self._read_header_length(dump_file)
                # read dump data proto
                self._read_dump_data(dump_file)
                self._check_size_match()
                # read tensor data
                self._read_input_data(dump_file)
                self._read_output_data(dump_file)
                self._read_buffer_data(dump_file)
                self._read_space_data(dump_file)
                return self.dump_data
        except (OSError, IOError) as io_error:
            utils.print_error_log('Failed to read the dump file %s. %s'
                                  % (self.dump_file_path, str(io_error)))
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR) from io_error
        finally:
            pass

    def check_argument_valid(self: any) -> None:
        """
        check argument valid
        :exception when invalid
        """
        utils.check_path_valid(self.dump_file_path, False)
        # get file size
        try:
            self.file_size = os.path.getsize(self.dump_file_path)
        except (OSError, IOError) as error:
            utils.print_error_log('get the size of dump file %s failed. %s'
                                  % (self.dump_file_path, str(error)))
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR) from error
        finally:
            pass

        if self.file_size <= ConstManager.UINT64_SIZE:
            utils.print_warn_log(
                'The size of %s must be greater than %d, but the file size'
                ' is %d. Please check the dump file.'
                % (self.dump_file_path, ConstManager.UINT64_SIZE, self.file_size))
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR)
        if self.file_size > ConstManager.ONE_GB:
            utils.print_warn_log(
                'The size (%d) of %s exceeds 1GB, it may task more time to run, please wait.'
                % (self.file_size, self.dump_file_path))

    def _check_size_match(self: any) -> None:
        input_data_size = 0
        for item in self.dump_data.input:
            input_data_size += item.size
        output_data_size = 0
        for item in self.dump_data.output:
            output_data_size += item.size
        buffer_data_size = 0
        for item in self.dump_data.buffer:
            buffer_data_size += item.size
        space_data_size = 0
        for item in self.dump_data.space:
            space_data_size += item.size
        # check 8 + content size + sum(input.data) + sum(output.data)
        # + sum(buffer.data) equal to file size
        if self.header_length + ConstManager.UINT64_SIZE + input_data_size \
                + output_data_size + buffer_data_size + space_data_size != self.file_size:
            utils.print_warn_log(
                'The file size (%d) of %s is not equal to %d (header length)'
                ' + %d(the size of header content) '
                '+ %d(the sum of input data) + %d(the sum of output data) '
                '+ %d(the sum of buffer data) + %d(the sum of space data). Please check the dump file.'
                % (self.file_size, self.dump_file_path, ConstManager.UINT64_SIZE, self.header_length,
                   input_data_size, output_data_size, buffer_data_size, space_data_size))
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR)

    def _read_header_length(self: any, dump_file: BinaryIO) -> None:
        # read header length
        header_length = dump_file.read(ConstManager.UINT64_SIZE)
        self.header_length = struct.unpack(ConstManager.UINT64_FMT, header_length)[0]
        # check header_length <= file_size - 8
        if self.header_length > self.file_size - ConstManager.UINT64_SIZE:
            utils.print_warn_log(
                'The header content size (%d) of %s must be less than or'
                ' equal to %d (file size) - %d (header length).'
                ' Please check the dump file.'
                % (self.header_length, self.dump_file_path, self.file_size, ConstManager.UINT64_SIZE))
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR)

    def _read_dump_data(self: any, dump_file: BinaryIO) -> None:
        content = dump_file.read(self.header_length)
        self.dump_data = DD.DumpData()
        try:
            self.dump_data.ParseFromString(content)
        except DecodeError as de_error:
            utils.print_warn_log(
                'Failed to parse the serialized header content of %s. '
                'Please check the dump file. %s '
                % (self.dump_file_path, str(de_error)))
            raise utils.AicErrException(Constant.MS_AICERR_INVALID_DUMP_DATA_ERROR) from de_error
        finally:
            pass

    def _read_input_data(self: any, dump_file: BinaryIO) -> None:
        for data_input in self.dump_data.input:
            data_input.data = dump_file.read(data_input.size)

    def _read_output_data(self: any, dump_file: BinaryIO) -> None:
        for data_output in self.dump_data.output:
            data_output.data = dump_file.read(data_output.size)

    def _read_buffer_data(self: any, dump_file: BinaryIO) -> None:
        for data_buffer in self.dump_data.buffer:
            data_buffer.data = dump_file.read(data_buffer.size)

    def _read_space_data(self: any, dump_file: BinaryIO) -> None:
        for data_space in self.dump_data.space:
            data_space.data = dump_file.read(data_space.size)
