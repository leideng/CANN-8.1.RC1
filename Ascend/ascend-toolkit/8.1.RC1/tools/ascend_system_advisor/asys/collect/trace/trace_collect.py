#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import time
import struct
import os.path
from threading import Thread
from datetime import datetime


from common import FileOperate as f
from common import log_error, log_warning
from common.task_common import out_progress_bar

MAGIC_VERSION_INFO = {
    "2": "0xd928",
    "3": "0xd928"
}

NS_TO_S = 1000000000
FREQ_GHZ_TO_KHZ = 1000000

TRACE_STRUCT_FIELD_TYPE_CHAR = 0
TRACE_STRUCT_FIELD_TYPE_INT8 = 1
TRACE_STRUCT_FIELD_TYPE_UINT8 = 2
TRACE_STRUCT_FIELD_TYPE_INT16 = 3
TRACE_STRUCT_FIELD_TYPE_UINT16 = 4
TRACE_STRUCT_FIELD_TYPE_INT32 = 5
TRACE_STRUCT_FIELD_TYPE_UINT32 = 6
TRACE_STRUCT_FIELD_TYPE_INT64 = 7
TRACE_STRUCT_FIELD_TYPE_UINT64 = 8
TRACE_STRUCT_ARRAY_TYPE_CHAR = 100
TRACE_STRUCT_ARRAY_TYPE_INT8 = 101
TRACE_STRUCT_ARRAY_TYPE_UINT8 = 102
TRACE_STRUCT_ARRAY_TYPE_INT16 = 103
TRACE_STRUCT_ARRAY_TYPE_UINT16 = 104
TRACE_STRUCT_ARRAY_TYPE_INT32 = 105
TRACE_STRUCT_ARRAY_TYPE_UINT32 = 106
TRACE_STRUCT_ARRAY_TYPE_INT64 = 107
TRACE_STRUCT_ARRAY_TYPE_UINT64 = 108
TRACE_STRUCT_BOOL = 10001

TRACE_STRUCT_SHOW_MODE_DEC = 0
TRACE_STRUCT_SHOW_MODE_BIN = 1
TRACE_STRUCT_SHOW_MODE_HEX = 2
TRACE_STRUCT_SHOW_MODE_CHAR = 3

UNPACK = {
    TRACE_STRUCT_FIELD_TYPE_CHAR: ['s', 1],
    TRACE_STRUCT_FIELD_TYPE_INT8: ['b', 1],
    TRACE_STRUCT_FIELD_TYPE_UINT8: ['B', 1],
    TRACE_STRUCT_FIELD_TYPE_INT16: ['h', 2],
    TRACE_STRUCT_FIELD_TYPE_UINT16: ['H', 2],
    TRACE_STRUCT_FIELD_TYPE_INT32: ['i', 4],
    TRACE_STRUCT_FIELD_TYPE_UINT32: ['I', 4],
    TRACE_STRUCT_FIELD_TYPE_INT64: ['q', 8],
    TRACE_STRUCT_FIELD_TYPE_UINT64: ['Q', 8],
    TRACE_STRUCT_ARRAY_TYPE_CHAR: ['s', 1],
    TRACE_STRUCT_ARRAY_TYPE_INT8: ['b', 1],
    TRACE_STRUCT_ARRAY_TYPE_UINT8: ['B', 1],
    TRACE_STRUCT_ARRAY_TYPE_INT16: ['h', 2],
    TRACE_STRUCT_ARRAY_TYPE_UINT16: ['H', 2],
    TRACE_STRUCT_ARRAY_TYPE_INT32: ['i', 4],
    TRACE_STRUCT_ARRAY_TYPE_UINT32: ['I', 4],
    TRACE_STRUCT_ARRAY_TYPE_INT64: ['q', 8],
    TRACE_STRUCT_ARRAY_TYPE_UINT64: ['Q', 8],
    TRACE_STRUCT_BOOL: ['?', 1]
}


def trace_show_mode(mode, value):
    if mode == TRACE_STRUCT_SHOW_MODE_DEC:
        return str(value)
    elif mode == TRACE_STRUCT_SHOW_MODE_BIN:
        return str(bin(int(value)))
    elif mode == TRACE_STRUCT_SHOW_MODE_HEX:
        return str(hex(int(value)))
    else:
        return str(value)


class ParseTrace:
    def __init__(self, is_file=False):
        self.is_file = is_file
        self.real_time = 0
        self.tz_offset = 0
        self.cpu_freq = 0

    def error(self, msg):
        if self.is_file:
            log_error(msg)

    def warning(self, msg):
        if self.is_file:
            log_warning(msg)

    @staticmethod
    def write_res_txt(msg_txt, trace_file):
        trace_file = trace_file.replace(".bin", ".txt")
        with open(trace_file, "w") as fw:
            fw.write(msg_txt.replace('\x00', ''))

    @staticmethod
    def time_zone_calculation(tz_offset):
        date_now = time.localtime()
        date_utc = time.gmtime()
        date_utc = datetime(date_utc.tm_year, date_utc.tm_mon, date_utc.tm_mday, date_utc.tm_hour, date_utc.tm_min)
        date_now = datetime(date_now.tm_year, date_now.tm_mon, date_now.tm_mday, date_now.tm_hour, date_now.tm_min)
        return ((date_now.timestamp() - date_utc.timestamp()) // 60 - tz_offset) * NS_TO_S

    @staticmethod
    def get_struct_data(fp, num, num_type):
        try:
            unpack_list = UNPACK.get(num_type)
            data = struct.unpack(f'{num}{unpack_list[0]}', fp.read(unpack_list[1] * num))
        except Exception as e:
            raise ValueError("Unable to parse data. Check whether the version matches "
                             "or whether the file content is complete.") from e
        if num_type in [TRACE_STRUCT_FIELD_TYPE_CHAR, TRACE_STRUCT_ARRAY_TYPE_CHAR]:
            return data[0].decode()
        if num == 1:
            return data[0]
        return data

    @staticmethod
    def get_res_data(fp, byte_len):
        try:
            fp.read(byte_len)
        except Exception as e:
            raise ValueError("Unable to parse data. Check whether the version matches "
                             "or whether the file content is complete.") from e

    def parse_ctrl_head(self, fp, trace_file):
        """
        This is a parse control header information.
        """
        trace_file_name = trace_file.split(os.sep)[-1]
        # Obtains the magic and version information.
        magic, version = self.get_struct_data(fp, 2, TRACE_STRUCT_FIELD_TYPE_UINT32)
        if str(version) not in MAGIC_VERSION_INFO.keys() or MAGIC_VERSION_INFO[str(version)] != str(hex(magic)):
            raise ValueError(f"the {trace_file_name} cannot be parsed, Check the version.")
        
        _, _, _, trace_type = self.get_struct_data(fp, 4, TRACE_STRUCT_FIELD_TYPE_UINT8)
        if trace_type != 0:
            raise ValueError(f"the {trace_file_name} cannot be parsed, Check trace type.")
        # Obtains the structSize and dataSize information.
        struct_size, data_size = self.get_struct_data(fp, 2, TRACE_STRUCT_FIELD_TYPE_UINT32)

        # Obtains the realTime and minutestWest information.
        self.tz_offset = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT32)
        self.real_time = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT64)
        if str(version) == "3":
            self.cpu_freq = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT64)
            self.get_res_data(fp, 8)
        else:
            # reserve 16byte
            self.get_res_data(fp, 16)
        # record current location
        current = fp.tell()
        # skip to the end
        fp.seek(0, 2)
        offset_end = fp.tell() - current
        fp.seek(current)
        if offset_end <= struct_size:
            raise ValueError(f"the {trace_file_name} is incomplete and cannot be parsed.")
        if offset_end < (struct_size + data_size):
            self.warning(f"the {trace_file_name} data is incomplete, which may cause data loss.")

    def parse_struct_segment(self, fp):
        """
         This is a parse data structure body information.
        """
        struct_dict = dict()
        # get struct count
        struct_count = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT32)
        # reserve 36byte
        self.get_res_data(fp, 36)
        for _ in range(struct_count):
            # get struct segment information
            struct_name = self.get_struct_data(fp, 32, TRACE_STRUCT_FIELD_TYPE_CHAR).replace('\x00', '')
            item_num = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT32)
            struct_type = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT8)
            self.get_res_data(fp, 3)
            item_lists = []
            for _ in range(item_num):
                item_name = self.get_struct_data(fp, 32, TRACE_STRUCT_FIELD_TYPE_CHAR).replace('\x00', '')
                item_type, item_mode = self.get_struct_data(fp, 2, TRACE_STRUCT_FIELD_TYPE_UINT8)
                item_length = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT16)
                self.get_res_data(fp, 4)
                item_lists.append([item_name, item_type, item_mode, item_length])
            struct_dict[struct_type] = {"struct_name": struct_name, "item_lists": item_lists}
        return struct_dict

    def parse_msg_data(self, fp, item_list, txt_size):
        """
        Parse the data based on the item parameter.
        """
        item_name, item_type, item_mode, item_length = item_list
        msg_byte = UNPACK.get(item_type)[1]
        data_list = ""
        use_byte = 0
        while item_length > 0 and txt_size > 0:
            if item_length < msg_byte or txt_size < msg_byte:
                raise ValueError("the data type or data length is incorrect and cannot be parsed.")
            data = self.get_struct_data(fp, 1, item_type)
            item_length -= msg_byte
            txt_size -= msg_byte
            use_byte += msg_byte
            data = trace_show_mode(item_mode, data)
            if item_type < TRACE_STRUCT_ARRAY_TYPE_CHAR:
                return f"{item_name}[{data}], ", use_byte
            else:
                if data_list == "" or item_type == TRACE_STRUCT_ARRAY_TYPE_CHAR:
                    data_list += f"{data}"
                else:
                    data_list += f", {data}"
        return f"{item_name}[{data_list}], ", use_byte

    def parse_data_segment(self, fp, trace_file):
        """
        This is a data parsing function that returns the parsed txt string.
        """
        self.parse_ctrl_head(fp, trace_file)
        struct_dict = self.parse_struct_segment(fp)
        if not struct_dict:
            raise ValueError("failed to parse the data. check whether the file is complete.")
        offset_time_ns = self.time_zone_calculation(self.tz_offset)
        msg_size, msg_txt_size, msg_num, _ = self.get_struct_data(fp, 4, TRACE_STRUCT_FIELD_TYPE_UINT32)
        msg_txt = ""
        for _ in range(msg_num):
            # data head
            cycle = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT64)
            txt_size = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT32)
            busy = self.get_struct_data(fp, 1, TRACE_STRUCT_BOOL)
            struct_type = self.get_struct_data(fp, 1, TRACE_STRUCT_FIELD_TYPE_UINT8)
            self.get_res_data(fp, 2)
            if busy:
                self.get_res_data(fp, msg_txt_size)
                continue
            if self.cpu_freq != 0:
                time_str = datetime.fromtimestamp((self.real_time + (cycle / self.cpu_freq) * FREQ_GHZ_TO_KHZ +
                                                   offset_time_ns) / NS_TO_S)
            else:
                time_str = datetime.fromtimestamp((self.real_time + cycle + offset_time_ns) / NS_TO_S)
            time_str = time_str.strftime("%Y-%m-%d %H:%M:%S.%f")
            # data
            struct_info = struct_dict.get(struct_type)
            if not struct_info:
                self.get_res_data(fp, msg_txt_size)
                continue
            msg_txt += "%s.%s %s: " % (time_str[:-3], time_str[-3:], struct_info.get("struct_name"))
            use_msg_data = 0
            for item_list in struct_info.get("item_lists"):
                item_txt, use_byte = self.parse_msg_data(fp, item_list, txt_size)
                msg_txt += item_txt
                use_msg_data += use_byte
            # txt_size not used up, skipping byte length
            if msg_txt_size - use_msg_data > 0:
                self.get_res_data(fp, msg_txt_size - use_msg_data)
            # remove end of line ', '
            msg_txt = msg_txt[:-2] + "\n"
        return msg_txt

    def start_parse_file(self, trace_file, count=0, num=0):
        out_progress_bar(count, num)
        msg_txt = self.parse(trace_file)
        if msg_txt:
            self.write_res_txt(msg_txt, trace_file)
            os.remove(trace_file)
            return True
        return False

    def parse(self, trace_file):
        msg_txt = ""
        try:
            with open(trace_file, "rb") as fp:
                msg_txt = self.parse_data_segment(fp, trace_file)
        except ValueError as e:
            self.error(e)
        except IOError:
            self.error(f"the {trace_file} cannot be read or cannot be found.")
        return msg_txt

    def run(self, trace_path, count=0):
        atrace_dirs = f.walk_dir(trace_path)
        if not atrace_dirs:
            return False
        num = 0
        threads = []
        for dirs, _, files in atrace_dirs:
            for file in files:
                trace_file = os.path.join(dirs, file)
                num += 1
                if file.endswith(".bin"):
                    t = Thread(target=self.start_parse_file, args=(trace_file, count, num), daemon=True)
                    t.start()
                    threads.append(t)

        out_progress_bar(count, count)
        # wait for all threads to end.
        for t in threads:
            t.join()
        return True


def collect_trace(output_root_path):
    trace_path = os.path.join(output_root_path, "dfx", "atrace")
    parse_trace = ParseTrace()
    parse_trace.run(trace_path)
