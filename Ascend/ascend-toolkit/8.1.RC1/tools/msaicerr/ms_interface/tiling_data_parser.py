"""Module providing a function printing tiling data."""

import os.path
import re
import struct
from ast import literal_eval
from ms_interface import utils


class TilingDataParser:
    def __init__(self, path):
        self.file_path = path

    @staticmethod
    def __get_tiling_str(tiling_datas) -> str:
        result_str = ""
        int32_size = struct.calcsize('i')
        int64_size = struct.calcsize('q')
        float16_size = struct.calcsize('e')

        def parse_data(data, size, self_format):
            try:
                result = [struct.unpack(self_format, data[i:i + size])[0]
                          for i in range(0, len(data), size)]
            except Exception:
                result = "Cannot decode in this dtype"
            return result

        int32_values = parse_data(tiling_datas, int32_size, 'I')
        result_str += "tiling data in uint32: "
        result_str += f"tiling data: {int32_values}\n"
        int64_values = parse_data(tiling_datas, int64_size, 'q')
        result_str += "tiling data in int64: "
        result_str += f"tiling data: {int64_values}\n"
        float16_values = parse_data(tiling_datas, float16_size, 'e')
        result_str += "tiling data in float16: "
        result_str += f"tiling data: {float16_values}\n"
        return result_str

    @staticmethod
    def __get_args(path):
        key_word_list = []

        get_args_cmd = ['grep', 'after execute:', '-inrE', path]
        get_args_regexp = r"(args\(0 to .*?\) after execute:.*?)after execute:args print end"
        get_args_ret = utils.get_inquire_result(get_args_cmd, get_args_regexp)
        if not get_args_ret:
            utils.print_warn_log("Failed to get all args after execute.")
            return []

        get_args_ret = get_args_ret[0]
        get_arg_regexp =  r" after execute:(.*?)$"
        get_arg_ret = re.findall(get_arg_regexp, get_args_ret, re.M | re.S)
        if not get_arg_ret:
            utils.print_warn_log("Failed to get arg after execute.")
            return [], -1

        for args in get_arg_ret:
            split_tmp = args.split(",")
            for str2 in split_tmp:
                if str2.find("\n") != -1:
                    continue
                str2 = str2.replace(' ', '')
                if not str2:
                    continue
                key_word_list.append(str2)
        get_io_ptr_cmd = ['grep', 'exception info dump args data', '-inrE', path]
        get_io_ptr_regexp = r"exception info dump args data, addr:(.*?);"
        get_io_ptr_ret = utils.get_inquire_result(get_io_ptr_cmd, get_io_ptr_regexp)
        if not get_io_ptr_ret:
            get_io_ptr_ret = []

        get_io_cmd = ['grep', '\[Dump\]\[Exception\]', '-inrE', path]
        get_io_regexp = r"begin to load .*? pointer tensor.*?end to load .*? pointer tensor"
        get_io_ret = utils.get_inquire_result(get_io_cmd, get_io_regexp)

        if get_io_ret:
            utils.print_info_log(f"The operator has sub_pointer tensor.")
            for ret in get_io_ret:
                get_key_regexp = r"begin to load .*? pointer tensor.*?addr:(.*?)$"
                get_key_ret = re.findall(get_key_regexp, ret, re.M | re.S)

                get_value_regexp = r"exception info dump args data, addr:(.*?);"
                get_value_ret = re.findall(get_value_regexp, ret, re.M | re.S)
                for value in get_value_ret:
                    if value in get_io_ptr_ret:
                        i = get_io_ptr_ret.index(value)
                        get_io_ptr_ret[i] = get_key_ret[0]
            get_io_ptr_ret = list(set(get_io_ptr_ret))
        guess_tiling_index = -1
        for io_ptr in get_io_ptr_ret:
            if io_ptr in key_word_list:
                _cnt = key_word_list.count(io_ptr)
                i = key_word_list.index(io_ptr, _cnt - 1)
                if i + 1 > guess_tiling_index:
                    guess_tiling_index = i + 1
        if guess_tiling_index >= len(key_word_list):
            utils.print_warn_log("Failed to get tiling_ptr after execute, maybe static shape case.")
            return key_word_list, -1

        para_base_cmd = ['grep', 'para base', '-inrE', path]
        para_base_regexp = r"para base:\s+(.*?)\."
        parm_base_ret = utils.get_inquire_result(para_base_cmd, para_base_regexp)
        if not parm_base_ret or not parm_base_ret[0].startswith("0x"):
            utils.print_warn_log("Failed to get para_base after execute.")
            return key_word_list, -1
        _end_offset = len("0x")
        for c in parm_base_ret[0][_end_offset:]:
            if c not in "0123456789abcdefABCDEF":
                break
            _end_offset = _end_offset + 1
        para_base = utils.get_hexstr_value(parm_base_ret[0][:_end_offset])

        if guess_tiling_index == -1:
            utils.print_info_log("Unable to get L0 dump log, start to guess tilingptr index.")
            for io_ptr in key_word_list:
                guess_tiling_ptr = utils.get_hexstr_value(io_ptr)
                offset = (guess_tiling_ptr - para_base) // 8
                if 0 < offset < len(key_word_list):
                    utils.print_info_log(f"Find tiling_data index: {offset}")
                    guess_tiling_index = key_word_list.index(io_ptr)
                    break

        get_idx_cmd = ["grep", "begin to load normal tensor", "-inrE", path]
        get_idx_regexp = r".*?index:(\d+)"
        get_idx_ret = utils.get_inquire_result(get_idx_cmd, get_idx_regexp)
        if not get_idx_ret:
            utils.print_warn_log("Failed to get the begin tensor's index of args.")
            get_idx_ret = ['0']
        begin_idx = int(get_idx_ret[0])

        tiling_ptr = utils.get_hexstr_value(key_word_list[begin_idx + guess_tiling_index])
        offset = (tiling_ptr - para_base) // 8
        utils.print_debug_log(f"get args: {key_word_list}, "
                             f"tiling_ptr:{hex(tiling_ptr)}, para_base: {hex(para_base)}, offset: {offset}")
        return key_word_list, offset


    @staticmethod
    def __reverse_str(sss):
        ss = ""
        for i in range(7, -1, -1):
            ss = ss + sss[2 * i: 2 * i + 2]
        return ss

    def __gen_tiling_data(self, key_word_list, offset):
        ss_concate = ""
        for key in key_word_list[offset:]:
            # remove "0x"
            str_tmp = key.replace("0x", '')
            str_tmp = str_tmp.zfill(16)
            str_tmp = self.__reverse_str(str_tmp)
            ss_concate = ss_concate + str_tmp
        tiling_datas = bytes.fromhex(ss_concate)
        return tiling_datas

    def __get_files(self):
        file_list = []
        for filepath, _, filenames in os.walk(self.file_path):
            for filename in filenames:
                file_list.append(os.path.join(filepath, filename))
        return file_list

    def parse(self):
        key_word_list, offset = self.__get_args(self.file_path)
        if offset == -1:
            return ""
        tiling_data = self.__gen_tiling_data(key_word_list, int(offset))
        result_info = self.__get_tiling_str(tiling_data)
        utils.print_debug_log(f'Get tiling data success!')
        utils.print_debug_log(result_info)
        return tiling_data
