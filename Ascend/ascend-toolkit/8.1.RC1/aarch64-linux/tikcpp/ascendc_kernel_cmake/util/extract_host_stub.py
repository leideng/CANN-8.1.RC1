#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#----------------------------------------------------------------------------
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
#----------------------------------------------------------------------------

"""Extract and generate host_stub.cpp and headers."""


import argparse
import json
import io
import itertools
import os
import re
import sys
import stat
from enum import Enum
from functools import partial
from itertools import chain, repeat, starmap
from operator import attrgetter
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

from channel import CODE_AIC, CODE_AIV, CODE_DEFAULT, get_code_channel
from struct_parser import ParseError, parse_struct_by_str


FUNC_SIGNATURE = re.compile(
    r'(?:template\s*<\s*([^>]+?)\s*>\s*\n?\s*)?__attribute__\(\(cce_kernel\)\)\s*\[aicore\]\s*(.+?)\s*\{', re.DOTALL
)
FUNC_PARAMS = re.compile(r'\((.+)\)', re.DOTALL)

STRUCT = re.compile(r'^\s*struct\s+(\w+)\s*\{', re.MULTILINE)

CCE_GLOBAL = '__attribute__((cce_global))'
FUNC_PARAM_SKIPS = ('__restrict__', '__restrict', 'struct')

ATTRIBUTE = re.compile(r'__attribute__\(\(\w+\)\)')

COMMENT = re.compile(r'[ \t]*//.*')

TYPE_BASENAME = re.compile(r'\w+')

IS_V220_MODE = False
MIX_CORE_MACRO = '__MIX_CORE_MACRO__'

FUN_TEMPLATE_HASH_TILING_KEY_BASE = 1000000

class CodeMode(Enum):
    """Code mode."""
    MIX = 0
    AIV = 1
    AIC = 2
    NORMAL = 3
    MIX_VECTOR_CORE = 4
    KERNEL_TYPE_AIV_ONLY = 5
    KERNEL_TYPE_AIC_ONLY = 6
    KERNEL_TYPE_MIX_AIV_1_0 = 7
    KERNEL_TYPE_MIX_AIC_1_0 = 8
    KERNEL_TYPE_MIX_AIC_1_1 = 9
    KERNEL_TYPE_MIX_AIC_1_2 = 10


class FuncMetaType(Enum):
    """function meta type."""
    F_TYPE_KTYPE = 1
    F_TYPE_CROSS_CORE_SYNC = 2
    F_TYPE_MAX = 3


STR_TO_KERNEL_TYPE_V200 = {
    "KERNEL_TYPE_AICORE" : CodeMode.AIC,
    "KERNEL_TYPE_VECTORCORE" : CodeMode.AIC,
    "KERNEL_TYPE_MIX_VECTOR_CORE" : CodeMode.MIX_VECTOR_CORE,
}

STR_TO_KERNEL_TYPE_V220 = {
    "KERNEL_TYPE_AIV_ONLY" : CodeMode.KERNEL_TYPE_AIV_ONLY,
    "KERNEL_TYPE_AIC_ONLY" : CodeMode.KERNEL_TYPE_AIC_ONLY,
    "KERNEL_TYPE_MIX_AIV_1_0" : CodeMode.KERNEL_TYPE_MIX_AIV_1_0,
    "KERNEL_TYPE_MIX_AIC_1_0" : CodeMode.KERNEL_TYPE_MIX_AIC_1_0,
    "KERNEL_TYPE_MIX_AIC_1_1" : CodeMode.KERNEL_TYPE_MIX_AIC_1_1,
    "KERNEL_TYPE_MIX_AIC_1_2" : CodeMode.KERNEL_TYPE_MIX_AIC_1_2,
}


class ExtractError(Exception):
    """Extract host stub base exception."""


class ParseFuncSignatureError(ExtractError):
    """Parse function signature error."""


class FuncNameNotFound(ParseFuncSignatureError):
    """Function name not found."""


class MultiFuncNameFound(ParseFuncSignatureError):
    """Multiple function name not found."""


class FuncParamsNotFound(ParseFuncSignatureError):
    """Function parameters in function signature not found."""


class TooFewFuncParamParts(ParseFuncSignatureError):
    """Function parameter parts is too few."""


class TooMuchFuncParamParts(ParseFuncSignatureError):
    """Function parameter parts is too much."""


class GetFuncParamPartError(ExtractError):
    """Get function parameter part error."""
    def __init__(self, func_param: 'FuncParam'):
        super().__init__(func_param)
        self.func_param = func_param


class GetFuncParamTypeError(GetFuncParamPartError):
    """Get function parameter type error."""


class GetFuncParamNameError(GetFuncParamPartError):
    """Get function parameter type error."""


class ArgumentError(ExtractError):
    """Argument error."""


class GetOfileModeError(ExtractError):
    """Get ofile mode error."""


class CxxfiltCMDError(ExtractError):
    """C++filt command error."""


def merge_dict(base: Dict, *news: List[Dict]):
    """Merge dicts."""
    result = base.copy()
    for new in news:
        result.update(new)
    return result


class FuncParam(NamedTuple):
    """Function parameter."""
    parts: Tuple[str, ...]
    cce_global: Optional[bool] = False
    tiling_struct: Optional[bool] = False  # is tiling struct parameter.


class FuncSign(NamedTuple):
    """Function signature."""
    return_type: str
    func_name: str
    func_template_decl: str
    func_template_specilization_args: Tuple[str, ...]
    func_params_specilization_args: Tuple[str, ...]
    func_params: Tuple[FuncParam, ...]


class StructRaw(NamedTuple):
    """Raw struct."""
    name: str
    content: str


class FuncSignGroup(NamedTuple):
    """Function signature group."""
    filepath: str
    func_signs: Tuple[FuncSign, ...]
    dump_info: dict
    kernel_type: CodeMode
    structs: Dict[str, StructRaw]


class FuncSignGroupWithBase(NamedTuple):
    """Function signature group."""
    filepath: str
    func_signs: Tuple[FuncSign, ...]
    dump_info: dict
    base_key: int


class FuncSignGroupWithModeBase(NamedTuple):
    """Function signature group."""
    filepath: str
    func_signs: Tuple[FuncSign, ...]
    dump_info: dict
    mode: CodeMode
    base_key: int


def is_v220_mode() -> bool:
    return IS_V220_MODE


def get_func_param_type_by_parts(parts: Iterator[str]) -> Tuple[int, str]:
    """Get function parameter type by parts."""
    for idx, part in enumerate(parts):
        if ATTRIBUTE.fullmatch(part):
            continue
        if part == 'const':
            continue
        return idx, part
    raise GetFuncParamTypeError(parts)


def get_func_param_name_by_parts(parts: Iterator[str]) -> Tuple[int, str]:
    """Get function parameter name by parts."""
    first = True
    for idx, part in enumerate(parts):
        if ATTRIBUTE.fullmatch(part):
            continue
        if part == 'const':
            continue
        if first:
            first = False
            continue
        return idx, part
    raise GetFuncParamNameError(parts)


def get_func_param_type(func_param: FuncParam) -> str:
    """Get function parameter type."""
    return get_func_param_type_by_parts(func_param.parts)[1]


def get_func_param_type_and_idx(func_param: FuncParam) -> Tuple[int, str]:
    """Get function parameter type and index."""
    return get_func_param_type_by_parts(func_param.parts)


def get_func_param_name(func_param: FuncParam) -> str:
    """Get function parameter name."""
    return get_func_param_name_by_parts(func_param.parts)[1]


def func_param_to_string(func_param: FuncParam, with_cce_attr: bool = False) -> str:
    """Function parameter convert to string."""
    if with_cce_attr and func_param.cce_global:
        parts = chain((CCE_GLOBAL,), func_param.parts)
    else:
        parts = func_param.parts
    return f'{" ".join(parts)}'


def func_sign_to_string(func_sign: FuncSign,
                        extra_param="",
                        suffix_extra_param="",
                        with_cce_attr: bool = False,
                        remove_aclrt_prefix: bool = True) -> str:
    """Function signature convert to string."""
    func_params_str = ', '.join(
        map(partial(func_param_to_string, with_cce_attr=with_cce_attr), func_sign.func_params)
    )

    if remove_aclrt_prefix:
        func_name = func_sign.func_name.replace("aclrtlaunch_", "")
    else:
        func_name = func_sign.func_name
    return f'{func_sign.return_type} {func_name}({extra_param}{func_params_str}{suffix_extra_param})'

def get_param_names_by_func_params(func_params: Iterator[FuncParam]) -> Iterator[str]:
    """Get parameters name by function parameters."""
    yield from (get_func_param_name(func_param) for func_param in func_params)


def get_param_names_by_func_sign(func_sign: FuncSign) -> Iterator[str]:
    """Get parameters name by function signature."""
    yield from get_param_names_by_func_params(func_sign.func_params)



def add_extern_c(func_sign_str: str) -> str:
    """Add extern "C" prefix."""
    return f'extern "C" {func_sign_str}'


def func_signs_in_groups(groups: List[FuncSignGroup]) -> Iterator[FuncSign]:
    """Function signatures in group."""
    return chain.from_iterable(map(attrgetter('func_signs'), groups))


def typenames_in_func_signs(func_signs: Iterator[FuncSign]) -> Iterator[str]:
    """Types in function signatures."""
    for func_sign in func_signs:
        for param in func_sign.func_params:
            param_type = get_func_param_type(param)
            yield param_type


def typenames_in_func_groups(func_sign_groups: Iterator[FuncSignGroup]) -> Set[str]:
    """Types in function signatures."""
    return set(
        typenames_in_func_signs(func_signs_in_groups(func_sign_groups))
    )


def compose_mode_base(func_group: FuncSignGroup,
                      mode: CodeMode,
                      base_key: int) -> FuncSignGroupWithModeBase:
    """Compose mode, base_key into FuncSignGroup."""
    return FuncSignGroupWithModeBase(
        func_group.filepath, func_group.func_signs, func_group.dump_info, mode, base_key
    )


def compose_mode_base_func_groups(func_groups: Iterator[FuncSignGroup],
                                  modes: Iterator[CodeMode],
                                  base_keys: Iterator[str]) -> List[FuncSignGroupWithModeBase]:
    """Function groups compose mode and base."""
    func_groups = list(
        starmap(
            compose_mode_base,
            zip(func_groups, modes, base_keys)
        )
    )
    return func_groups


def grouptodict(objs: Iterator, key: Callable) -> Dict:
    """Group data to a dictionary."""
    results = {}
    for obj in objs:
        group = results.setdefault(key(obj), [])
        group.append(obj)

    return results


def cut_func_signature(instr: str) -> Iterator[str]:
    """Cut function signature from preprocessed code."""
    yield from FUNC_SIGNATURE.findall(instr)


def cut_struct(instr: str) -> Iterator[StructRaw]:
    """Cut struct from preprocessed code."""
    for match in STRUCT.finditer(instr):
        name = match.group(1)
        pos = match.end()
        cnt = 1
        while cnt > 0 and pos < len(instr):
            if instr[pos] == '{':
                cnt += 1
            elif instr[pos] == '}':
                cnt -= 1
            pos += 1
        content = instr[match.start():pos+1]
        yield StructRaw(name, content)


def map_structs_to_dict(structs: Iterator[StructRaw]) -> Dict[str, StructRaw]:
    """Map structs to dict."""
    aiter, biter = itertools.tee(structs)
    return dict(zip(map(attrgetter('name'), aiter), biter))


def parse_func_signature_parts(func_sign: str) -> Tuple[str, str, str]:
    """Parse function signature.
    Return a tuple of return type, function name and function parameters.
    """
    params_obj = FUNC_PARAMS.search(func_sign)
    if params_obj is None:
        # in no param case, to convert "void xxx()" into "void xxx", use [:-2]
        func_sign_without_params = func_sign[:-2]
        func_params = ""
    else:
        func_sign_without_params = func_sign[:params_obj.start(0)]
        func_params = params_obj.group(1).strip()

    # convert params like "xxx(void)", "xxx(  void  )" into "xxx()"
    if func_params == "void":
        func_params = ""
    parts = re.split(r'\s+', func_sign_without_params.strip())
    if len(parts) <= 1:
        raise FuncNameNotFound()
    if len(parts) > 2:
        raise MultiFuncNameFound()
    return_type, func_name = parts

    return return_type, func_name, func_params


def split_func_params(funcs_params_str: str) -> List[str]:
    """Split function parameters by comma and whitespaces."""
    return re.split(r',\s*', funcs_params_str)


def parse_func_param(func_param: str) -> FuncParam:
    """Parse function parameter to Function Parameter struct."""
    func_param_parts = re.split(r'\s+', func_param)

    if len(func_param_parts) < 2:
        raise TooFewFuncParamParts()

    cce_global = CCE_GLOBAL == func_param_parts[0]
    parts = tuple(
        part
        for part in func_param_parts
        if part != CCE_GLOBAL and part not in FUNC_PARAM_SKIPS
    )

    return FuncParam(parts, cce_global=cce_global)


def match_tiling_struct(func_param: FuncParam, structs: Dict[str, StructRaw]) -> bool:
    """Match tiling parameter."""
    return not func_param.cce_global and get_func_param_type(func_param) in structs


def parse_func_signature(func_sign_str: Tuple[str, str], structs: Dict[str, StructRaw]) -> FuncSign:
    """
    Parse function signature. Return FuncSing struct.
    """
    func_template_decl = func_sign_str[0]
    return_type, func_name, func_params = parse_func_signature_parts(func_sign_str[1])
    if func_params == "":
        return FuncSign(return_type, func_name, func_template_decl, tuple(), tuple(), tuple())
    func_params = tuple(map(parse_func_param, split_func_params(func_params)))
    # any struct arg could be tiling struct
    for idx, func_param in enumerate(func_params):
        if match_tiling_struct(func_param, structs):
            func_params = func_params[:idx] + (func_params[idx]._replace(tiling_struct=True),) + func_params[idx+1:]
    return FuncSign(return_type, func_name, func_template_decl, tuple(), tuple(), func_params)


def parse_func_signature_by_source(data: str, structs: Dict[str, StructRaw]) -> Iterator[FuncSign]:
    """Parse function signature by source code.
    Return iterator of function signature.
    """
    for func_sign_str in cut_func_signature(data):
        yield parse_func_signature(func_sign_str, structs)


def get_dump_info_by_source(data: str):
    dump_info = {"dump_type" : "", "dump_size" : 1048576}

    match_printf = re.search(r"__enable_feature_for_compile_printf = 1", data)
    match_assert = re.search(r"__enable_feature_for_compile_assert = 1;", data)
    if match_printf and match_assert:
        dump_info["dump_type"] = "printf,assert"
    elif match_printf:
        dump_info["dump_type"] = "printf"
    elif match_assert:
        dump_info["dump_type"] = "assert"

    match = re.search(r"__enable_feature_for_compile_printfBufSize = \s*([0-9]{1,})", data)
    if match:
        dump_info["dump_size"] = int(match.group(1))
    else:
        match = re.search(r"__enable_feature_for_compile_assertBufSize = \s*([0-9]{1,})", data)
        if match:
            dump_info["dump_size"] = int(match.group(1))

    return dump_info


def find_kernel_type_by_source(path: str, data: str, build_mode: str) -> CodeMode:
    is_m200 = build_mode == "m200"
    is_v220 = build_mode == "aiv"
    match = re.search(r"__enable_feature_for_compile_default\s*=\s*([0-9a-zA-Z_]{1,})\s*;", data)
    if match:
        kernel_type = match.group(1)
        if (kernel_type in STR_TO_KERNEL_TYPE_V200) and is_m200:
            return STR_TO_KERNEL_TYPE_V200[kernel_type]
        elif (kernel_type in STR_TO_KERNEL_TYPE_V220) and is_v220:
            return STR_TO_KERNEL_TYPE_V220[kernel_type]
        else:
            print(f"[WARNING]: {kernel_type} in path: {path} do not support in soc {build_mode}!")
            if is_v220:
                return None
            else:
                return CodeMode.AIC
    else:
        if is_v220:
            return None
        else:
            return CodeMode.AIC


def parse_func_signature_group_by_source(path: str, data: str, build_mode: str) -> FuncSignGroup:
    """Parse function signature group by source."""
    structs = map_structs_to_dict(cut_struct(data))
    func_signs = list(parse_func_signature_by_source(data, structs))
    func_signs = sorted(func_signs, key=attrgetter('func_name'))
    func_signs = tuple(func_signs)
    dump_info = get_dump_info_by_source(data)
    if build_mode == "m200" or build_mode == "aiv":
        kernel_type = find_kernel_type_by_source(path, data, build_mode)
    else:
        kernel_type = CodeMode.NORMAL
    return FuncSignGroup(path, func_signs, dump_info, kernel_type, structs)


def parse_func_signature_by_filepaths(filepaths: List[str], build_mode: str) -> Iterator[FuncSignGroup]:
    """Parse function signature by source file paths.
    Return iterator of function signature.
    """
    for path in filepaths:
        try:
            with open(path, encoding='utf-8') as file:
                data = file.read()
                func_sign_group = parse_func_signature_group_by_source(path, data, build_mode)
                yield func_sign_group
        except Exception as err:
            print("[ERROR]: read file failed, filename is: {}".format(path))
            raise err


def merge_func_sign_groups_structs(func_sign_groups: Iterator[FuncSignGroup]
                                   ) -> Dict[str, StructRaw]:
    """Merge function signature groups structs."""
    return merge_dict(*map(attrgetter('structs'), func_sign_groups))


def is_pointer(type_str: str) -> bool:
    """Is pointer type?"""
    return type_str.endswith('*')


def convert_to_void(param_type: str) -> str:
    """Convert to void."""
    if is_pointer(param_type):
        return 'void*'
    return 'void'


def convert_param_parts_type(parts: Tuple[str, ...],
                             convert_func: Callable[[str], str]) -> Tuple[str, ...]:
    """Convert type in function parameter parts."""
    idx, param_type = get_func_param_type_by_parts(parts)
    new_param_type = convert_func(param_type)
    return parts[:idx] + (new_param_type,) + parts[idx+1:]


def convert_func_param_cce_param_type(func_param: FuncParam) -> FuncParam:
    """Convert type in function cce parameter.

    Convert pointer type to void*, non-pointer type to void.
    """
    if not func_param.cce_global:
        return func_param

    parts = convert_param_parts_type(func_param.parts, convert_to_void)
    return func_param._replace(parts=parts)


def add_block_dim_and_stream_func_params(func_params: Tuple[FuncParam, ...]
                                         ) -> Tuple[FuncParam, ...]:
    """Add blockDim and stream to function parameters."""
    param_names = set(get_param_names_by_func_params(func_params))

    if 'blockDim' in param_names:
        block_dim_param = FuncParam(('uint32_t', '_blockDim'))
    else:
        block_dim_param = FuncParam(('uint32_t', 'blockDim'))

    if 'stream' in param_names:
        stream_param = FuncParam(('aclrtStream', '_stream'))
    else:
        stream_param = FuncParam(('aclrtStream', 'stream'))

    added_func_params = (block_dim_param, stream_param)
    return added_func_params + func_params


def tiling_add_pointer(func_param: FuncParam) -> FuncParam:
    """Tiling parameter type add pointer."""
    if func_param.tiling_struct:
        idx, param_type = get_func_param_type_and_idx(func_param)
        parts = func_param.parts[:idx] + (f'{param_type}*',) + func_param.parts[idx+1:]
        return func_param._replace(parts=parts)
    return func_param


def tiling_remove_pointer(func_param: FuncParam) -> FuncParam:
    """Tiling parameter type remove pointer."""
    if func_param.tiling_struct:
        idx, param_type = get_func_param_type_and_idx(func_param)
        if not param_type.endswith('*'):
            return func_param

        parts = func_param.parts[:idx] + (param_type[:-1],) + func_param.parts[idx+1:]
        return func_param._replace(parts=parts)
    return func_param


def tiling_add_pointer_func_params(func_params: Tuple[FuncParam, ...]) -> Tuple[FuncParam, ...]:
    """Add pointer at tiling parameter."""
    return tuple(map(tiling_add_pointer, func_params))


def tiling_remove_pointer_func_params(func_params: Tuple[FuncParam, ...]) -> Tuple[FuncParam, ...]:
    """Remove pointer at tiling parameter."""
    return tuple(map(tiling_remove_pointer, func_params))


def tiling_add_reference(func_param: FuncParam) -> FuncParam:
    """Tiling parameter type add reference."""
    if func_param.tiling_struct:
        idx, param_type = get_func_param_type_and_idx(func_param)
        parts = func_param.parts[:idx] + (f'{param_type}&',) + func_param.parts[idx+1:]
        return func_param._replace(parts=parts)
    return func_param


def tiling_remove_reference(func_param: FuncParam) -> FuncParam:
    """Tiling parameter type remove reference."""
    if func_param.tiling_struct:
        idx, param_type = get_func_param_type_and_idx(func_param)
        if not param_type.endswith('&'):
            return func_param

        parts = func_param.parts[:idx] + (param_type[:-1],) + func_param.parts[idx+1:]
        return func_param._replace(parts=parts)
    return func_param


def tiling_add_ref_or_ptr_func_params(func_sign: FuncSign,
                                      func_params: Tuple[FuncParam, ...]) -> Tuple[FuncParam, ...]:
    """Add reference for template func sign, add pointer for normal func sign at tiling parameter."""
    if func_sign.func_template_decl:
        return tuple(map(tiling_add_reference, func_params))
    else:
        return tuple(map(tiling_add_pointer, func_params))


def tiling_remove_ref_or_ptr_func_params(func_sign: FuncSign,
                                         func_params: Tuple[FuncParam, ...]) -> Tuple[FuncParam, ...]:
    """Remove reference for template func sign, remove pointer for normal func sign at tiling parameter."""
    if func_sign.func_template_decl:
        return tuple(map(tiling_remove_reference, func_params))
    else:
        return tuple(map(tiling_remove_pointer, func_params))


def replace_func_param_stream(func_params: Tuple[FuncParam, ...],
                              stream_name: str) -> Tuple[FuncParam, ...]:
    """Replace stream param type."""
    new_func_params = (func_params[0], FuncParam(('void*', stream_name))) + func_params[2:]
    return new_func_params


def remove_added_func_params(func_params: Tuple[FuncParam, ...]) -> Tuple[FuncParam, ...]:
    """Remove added function parameters."""
    return func_params[2:]


def add_ffts_addr_func_param_by_mode(mode: CodeMode,
                                     func_params: Tuple[FuncParam, ...]) -> Iterator[FuncParam]:
    """Add ffts_addr function parameter by mode."""
    if mode in (CodeMode.MIX,
                CodeMode.KERNEL_TYPE_MIX_AIV_1_0,
                CodeMode.KERNEL_TYPE_MIX_AIC_1_0,
                CodeMode.KERNEL_TYPE_MIX_AIC_1_1,
                CodeMode.KERNEL_TYPE_MIX_AIC_1_2):
        yield FuncParam(('void*', 'ffts_addr'))
    yield from func_params


def add_aclrt_prefix_snake(func_name: str) -> str:
    """Add aclrt prefix in snake style."""
    return f'aclrtlaunch_{func_name}'


def remove_aclrt_prefix_snake(func_name: str) -> str:
    """Remove aclrt prefix in snake style."""
    if func_name.startswith('aclrtlaunch_'):
        return func_name[len('aclrtlaunch_'):]
    return func_name


def add_auto_gen_prefix_and_kernel_suffix(func_name: str) -> str:
    """Add auto_gen prefix and kernel suffix."""
    return f'auto_gen_{func_name}_kernel'


def add_origin_suffix(func_name: str) -> str:
    """Add temp suffix."""
    return f'{func_name}_origin'


def trans_func_sign(func_sign: FuncSign) -> FuncSign:
    """Transform function signature.
    convert function parameters type, add function parameters, etc.
    """
    func_params = tuple(
        convert_func_param_cce_param_type(func_param)
        for func_param in func_sign.func_params
    )

    new_func_params = tiling_add_ref_or_ptr_func_params(
        func_sign,
        add_block_dim_and_stream_func_params(func_params)
    )
    new_func_name = add_aclrt_prefix_snake(func_sign.func_name)
    return func_sign._replace(
        return_type='uint32_t',
        func_name=new_func_name,
        func_params=new_func_params,
    )


def replace_func_sign_stream_param(func_sign: FuncSign, stream_name: str) -> FuncSign:
    """Replace function signature last stream param type."""
    func_params = replace_func_param_stream(func_sign.func_params, stream_name)
    return func_sign._replace(func_params=func_params)


def trans_func_sign_group(func_sign_group: FuncSignGroup) -> FuncSignGroup:
    """Transform each function signature in group."""
    func_signs = tuple(map(trans_func_sign, func_sign_group.func_signs))
    return func_sign_group._replace(func_signs=func_signs)


import subprocess


def get_func_template_specialization_mangle_name(src_file: str, kernel_sign: str) -> str:
    if not os.path.isfile(src_file):
        return False, '', f"file {src_file} doesn't exist."

    objdump_cmd = ['llvm-objdump', '-tC', src_file]
    proc = subprocess.run(
        objdump_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    out = proc.stdout.decode()

    if proc.returncode != 0:
        return False, '', f'llvm-objdump error, message is {out}'

    pattern = re.compile(rf" g .*? F \.text.*?{re.escape(kernel_sign)}<(.*?)>\((.*?)\)")
    matches = re.findall(pattern, out)

    return True, matches, ''


def add_template_specilization_args(func_sign: FuncSign, aiv_o: str, aic_o: str) -> FuncSign:
    """Add function template specilization args."""
    if func_sign.func_template_decl:
        ret, tmp_spec_names, err_msg = get_func_template_specialization_mangle_name(aiv_o,
            f'{func_sign.return_type} {func_sign.func_name}'
        )
        if not ret:
            raise GetOfileModeError(f'aiv: {err_msg}')
        func_spec_names = []
        func_params_spec_names = []
        for spec_name in tmp_spec_names:
            func_spec_names.append(spec_name[0])
            func_params_spec_names.append(spec_name[1])

        return func_sign._replace(
            func_template_specilization_args=func_spec_names,
            func_params_specilization_args=func_params_spec_names,
        )
    return func_sign


def get_func_sign_template_specilization_args(func_sign_group: FuncSignGroup,
                                              aiv_o: str,
                                              aic_o: str):
    func_signs = tuple(map(lambda x: add_template_specilization_args(x, aiv_o, aic_o), func_sign_group.func_signs))
    return func_sign_group._replace(func_signs=func_signs)


def get_header_name_by_func_sign(func_sign: FuncSign) -> str:
    """Get header filename by function signature."""
    return f'{func_sign.func_name}.h'


def get_header_file_macro(func_sign: FuncSign) -> str:
    """Get header file macro by function signature."""
    header_file_name = get_header_name_by_func_sign(func_sign)
    file_prefix = header_file_name.split(".")[0]
    upper_header_file_name = file_prefix.upper()
    return f'HEADER_{upper_header_file_name}_H'


# generate func sign like "template<int a> void func(uint8_t* input)" or "extern "C" void func(uint8_t* input)"
def generate_auto_gen_func_sign(func_sign: FuncSign, func_tmp_sign: str):
    """
    Generate auto gen function signature
    func_sign.func_template_decl: "int a" or None
    func_tmp_sign: "void func(uint8_t* input)"
    return: "template<int a> void func(uint8_t* input)" or "extern "C" void func(uint8_t* input)"
    """
    if not func_sign.func_template_decl:
        auto_gen_func_sign = add_extern_c(func_tmp_sign)
    else:
        auto_gen_func_sign = f'''template<{func_sign.func_template_decl}>
{func_tmp_sign}'''
    return auto_gen_func_sign


def generate_header_code(func_sign: FuncSign) -> str:
    """Generate header content."""
    header_file_macro = get_header_file_macro(func_sign)

    return f"""#ifndef {header_file_macro}
#define {header_file_macro}
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

{generate_auto_gen_func_sign(func_sign, func_sign_to_string(func_sign, "", "", False, False))};
#endif
"""


def replace_func_void_param(func_sign: FuncSign, stream_name: str) -> FuncSign:
    """Replace function signature last stream param type."""
    new_func_params = (func_sign.func_params[0], FuncParam(('void*', stream_name))) + func_sign.func_params[1:]
    return func_sign._replace(func_params=new_func_params)


def extract_variable_names(declarations: str):
    declarations = declarations.split(',')
    variable_names = []
    for decl in declarations:
        parts = decl.split()
        variable_names.append(parts[-1].strip())

    return ', '.join(variable_names)


def generate_kernel_header_code(func_sign: FuncSign) -> str:
    """Generate header content."""
    header_file_macro = get_header_file_macro(func_sign) + "KERNEL_H_"

    param_names = tuple(get_param_names_by_func_sign(func_sign))
    param_names_str = ', '.join(param_names)

    stream_name = param_names[1]
    new_func_sign = replace_func_sign_stream_param(func_sign, stream_name)

    extern_func_old = func_sign_to_string(new_func_sign)

    new_name = new_func_sign.func_name.replace("aclrtlaunch_", "")

    template_parts = func_sign.func_template_decl.split(",")

    func_sign_str_hold = f'inline {func_sign_to_string(replace_func_void_param(new_func_sign, "hold"))}'
    func_template_var_names = ""
    if func_sign.func_template_decl:
        func_sign_str_hold = f'template<{func_sign.func_template_decl}>\n{func_sign_str_hold}'
        func_template_var_names = f'<{extract_variable_names(func_sign.func_template_decl)}>'

    struct_pre_declaration = ""
    for func_param in tiling_remove_ref_or_ptr_func_params(func_sign,
                                                           remove_added_func_params(new_func_sign.func_params)):
        if func_param.tiling_struct:
            struct_pre_declaration += f"\nstruct {func_param.parts[0]};\n"

    return f"""
#ifndef {header_file_macro}
#define {header_file_macro}

{struct_pre_declaration}

{generate_auto_gen_func_sign(func_sign, func_sign_to_string(new_func_sign, "", "", False, False))};

{func_sign_str_hold}
{{
    (void)hold;
    return {new_func_sign.func_name}{func_template_var_names}({param_names_str});
}}

#endif
"""


def indent_code(code: str, indent: str = '    '):
    """Indent non-empty lines."""
    # use `(?=)' for lookahead assertion, see re module document.
    return re.sub(r'^(?=.+)', indent, code, flags=re.MULTILINE)


def generate_host_stub_set_exception_dump_source() -> str:
    exception_dump_source = r'''
#ifdef ASCENDC_DUMP
static void ascendc_set_exception_dump_info(uint32_t dumpSize)
{
    uint32_t atomicIndex = 0U;
    uint32_t addrNum = 1U;
    void *exceptionDumpAddr = Adx::AdumpGetSizeInfoAddr(addrNum + ASCENDC_EXCEPTION_DUMP_HEAD, atomicIndex);
    if (exceptionDumpAddr == nullptr) {
        printf("Get exceptionDumpAddr is nullptr.\n");
        return;
    }

    // atmoic index
    uint64_t *sizeInfoAddr = reinterpret_cast<uint64_t *>(exceptionDumpAddr);
    *sizeInfoAddr = static_cast<uint64_t>(atomicIndex);
    sizeInfoAddr++;

    *sizeInfoAddr = static_cast<uint64_t>(1);
    sizeInfoAddr++;

    *sizeInfoAddr = dumpSize * 75;
    constexpr uint64_t workspaceOffset = (4ULL << 56ULL);
    *sizeInfoAddr |= workspaceOffset;

    const rtArgsSizeInfo sizeInfo = {exceptionDumpAddr, atomicIndex};
    int32_t ret = rtSetExceptionExtInfo(&sizeInfo);
    if (ret != 0) {
        printf("rtSetExceptionExtInfo failed, ret = %d.\n", ret);
    }
}
#endif
'''
    return exception_dump_source


def generate_host_stub_head_code(has_mix: bool, has_aic: bool, has_aiv: bool, dump_assert: bool) -> str:
    """Generate host_stub.cpp head code."""

    type_nums = 0
    ascend_kernel_struct = []

    buff = io.StringIO()
    buff.write(r'''#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <securec.h>
''')
    buff.write('\n')

    buff.write(r'''#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif
''')

    if dump_assert:
        buff.write(r'''
#ifdef ASCENDC_DUMP
#define ASCENDC_EXCEPTION_DUMP_HEAD 2U

typedef struct rtArgsSizeInfo {
    void *infoAddr;
    uint32_t atomicIndex;
} rtArgsSizeInfo_t;
#endif
''')
    buff.write('\n')
    buff.write('''static char ascendcErrMsg[1024] = {0};
''')
    buff.write('\n')
    if has_mix:
        type_nums += 1
        buff.write('''static void *g_kernel_handle = nullptr;
''')
        buff.write('\n')

    if has_aiv:
        type_nums += 1
        buff.write('''static void *g_kernel_handle_aiv = nullptr;
''')
        buff.write('\n')

    if has_aic:
        type_nums += 1
        buff.write('''static void *g_kernel_handle_aic = nullptr;
''')
        buff.write('\n')

    buff.write(r'''struct ascend_kernels {
    uint32_t version;
    uint32_t type_cnt;
''')

    # version value
    ascend_kernel_struct.append(str(1))

    # type_cnt value
    ascend_kernel_struct.append(str(type_nums))

    if has_mix:
        buff.write('''    uint32_t mix_type;
    uint32_t mix_len;
    uint32_t mix_file_len;
    uint8_t mix_buf[__replaced_mix_len];
''')
        ascend_kernel_struct.append(str(0))
        ascend_kernel_struct.append('__replaced_mix_len')
        ascend_kernel_struct.append('__replaced_mix_file_len')
        ascend_kernel_struct.append(r'{0}')

    if has_aiv:
        buff.write('''    uint32_t aiv_type;
    uint32_t aiv_len;
    uint32_t aiv_file_len;
    uint8_t aiv_buf[__replaced_aiv_len];
''')
        ascend_kernel_struct.append(str(1))
        ascend_kernel_struct.append('__replaced_aiv_len')
        ascend_kernel_struct.append('__replaced_aiv_file_len')
        ascend_kernel_struct.append(r'{0}')

    if has_aic:
        buff.write('''    uint32_t aic_type;
    uint32_t aic_len;
    uint32_t aic_file_len;
    uint8_t aic_buf[__replaced_aic_len];
''')
        ascend_kernel_struct.append(str(2))
        ascend_kernel_struct.append('__replaced_aic_len')
        ascend_kernel_struct.append('__replaced_aic_file_len')
        ascend_kernel_struct.append(r'{0}')

    buff.write('} __replaced_ascend_kernel __attribute__ ((section ("__replaced_ascend_section"))) = {')

    ascend_kernel_str = ','.join(ascend_kernel_struct)
    buff.write(f'{ascend_kernel_str}')

    buff.write('};\n\n')

    buff.write('''extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim, void **args,
                            uint32_t size, const void *stream);
uint32_t GetAscendCoreSyncAddr(void **addr);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);
void AscendProfRegister();
uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);
uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,
    const void *stream, bool enbaleProf, uint32_t aicBlockDim, uint32_t aivBlockDim, uint32_t aivBlockDimOffset);
''')
    if dump_assert:
        buff.write('''int32_t rtSetExceptionExtInfo(const rtArgsSizeInfo_t * const sizeInfo);

namespace Adx {
    void *AdumpGetSizeInfoAddr(uint32_t space, uint32_t &atomicIndex);
}
}
namespace Adx {
''')
    else:
        buff.write('''}\n\nnamespace Adx {''')

    buff.write('''
    void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                            void *stream, const char *opType);
}
''')

    buff.write('''
    class KernelHandleGradUnregister {
    private:
        KernelHandleGradUnregister() {}

    public:
        KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;
        KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;

        static KernelHandleGradUnregister& GetInstance() {
            static KernelHandleGradUnregister instance;
            return instance;
        }
        ~KernelHandleGradUnregister(){
''')

    if has_mix:
        buff.write('''            if (g_kernel_handle) {
                UnregisterAscendBinary(g_kernel_handle);
                g_kernel_handle = nullptr;
            }
''')

    if has_aiv:
        buff.write('''            if (g_kernel_handle_aiv) {
                UnregisterAscendBinary(g_kernel_handle_aiv);
                g_kernel_handle_aiv = nullptr;
            }
''')

    if has_aic:
        buff.write('''            if (g_kernel_handle_aic) {
                UnregisterAscendBinary(g_kernel_handle_aic);
                g_kernel_handle_aic = nullptr;
            }
''')
    buff.write('''        }
    };
''')
    buff.write('\n')

    buff.write(r'''static void __register_kernels(void) __attribute__((constructor));
void __register_kernels(void)
{
    const char* compileSocVersion = "__replaced_ascend_compile_soc_version";
    uint32_t ret;
''')
    buff.write('\n')

    buff.write(r'''    bool checkSocVersion = AscendCheckSoCVersion(compileSocVersion, ascendcErrMsg);
    if (!checkSocVersion) {
        return;
    }
''')

    if has_mix:
        buff.write(r'''    ret = RegisterAscendBinary(
        (const char *)__replaced_ascend_kernel.mix_buf,
        __replaced_ascend_kernel.mix_file_len,
        0,
        &g_kernel_handle);
    if (ret != 0) {
        printf("RegisterAscendBinary mix ret %u \n", ret);
    }
''')
        buff.write('\n')

    if has_aiv:
        buff.write(r'''    ret = RegisterAscendBinary(
        (const char *)__replaced_ascend_kernel.aiv_buf,
        __replaced_ascend_kernel.aiv_file_len,
        1,
        &g_kernel_handle_aiv);
    if (ret != 0) {
        printf("RegisterAscendBinary aiv ret %u \n", ret);
    }
''')
        buff.write('\n')

    if has_aic:
        buff.write(r'''    ret = RegisterAscendBinary(
        (const char *)__replaced_ascend_kernel.aic_buf,
        __replaced_ascend_kernel.aic_file_len,
        2,
        &g_kernel_handle_aic);
    if (ret != 0) {
        printf("RegisterAscendBinary aic ret %u \n", ret);
    }
''')
        buff.write('\n')

    buff.write('''    AscendProfRegister();
}
''')

    if dump_assert:
        buff.write(generate_host_stub_set_exception_dump_source())

    return buff.getvalue()


def generate_args_declare_code(mode: CodeMode, func_params: Tuple[FuncParam, ...], dump_type: str) -> str:
    """Generate args declare code."""
    buff = io.StringIO()
    buff.write('struct {\n')
    if dump_type != "":
        buff.write(r'''#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
        void* __ascendc_dump;
#endif
''')

    # Match BISHENG ABI format, ptr of args should be aligned to alignof(arg type), minimum 4bytes aligned.
    for func_param in add_ffts_addr_func_param_by_mode(mode, func_params):
        param_type = get_func_param_type(func_param)
        param_name = get_func_param_name(func_param)
        buff.write(f'    alignas(((alignof({param_type}) + 3) >> 2) << 2) {param_type} {param_name};\n')
    buff.write(f'    alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;\n')
    buff.write('} __ascendc_args;\n')
    return buff.getvalue()


def generate_get_sync_addr_code() -> str:
    """Generate GetAscendCoreSyncAddr code."""
    return r"""// need to get the synchronization address if mix mode (stage 2)
void *ffts_addr;
__ascendc_ret = GetAscendCoreSyncAddr(&ffts_addr);
if (__ascendc_ret != 0) {
    printf("GetAscendCoreSyncAddr ret %u\n", __ascendc_ret);
    return __ascendc_ret;
}
"""


def generate_args_assign_code(mode: CodeMode, func_sign: FuncSign) -> str:
    """Generate args assign code."""
    buff = io.StringIO()
    for func_param in add_ffts_addr_func_param_by_mode(mode, remove_added_func_params(func_sign.func_params)):
        param_name = get_func_param_name(func_param)
        if func_sign.func_template_decl:
            param_name_ptr = f'&{param_name}'
        else:
            param_name_ptr = param_name
        if func_param.tiling_struct:
            buff.write(
                f'(void) memcpy_s(&__ascendc_args.{param_name}, sizeof(__ascendc_args.{param_name}), '
                f'{param_name_ptr}, sizeof(__ascendc_args.{param_name}));\n'
            )
        else:
            buff.write(f'__ascendc_args.{param_name} = {param_name};\n')

    return buff.getvalue()


def generate_launch_kernel_code(mode: CodeMode,
                                func_key: int,
                                block_dim: str,
                                stream: str) -> str:
    """Generate LaunchAscendKernel code."""
    buff = io.StringIO()

    # The key is the key specified by each kernel entry.
    # During compilation, the key value of the function is defined based on the kernel sequence.
    # sample: -Dmy_add=my_add_1. key=1. Different and unique keys must be specified for different kernels.
    kernel_handle = ""
    if mode in (CodeMode.MIX,
                CodeMode.NORMAL,
                CodeMode.MIX_VECTOR_CORE,
                CodeMode.KERNEL_TYPE_MIX_AIC_1_1,
                CodeMode.KERNEL_TYPE_MIX_AIC_1_2):
        kernel_handle = 'g_kernel_handle'
        buff.write(r'''if (g_kernel_handle == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
''')
    elif mode in (CodeMode.AIC, CodeMode.KERNEL_TYPE_AIC_ONLY, CodeMode.KERNEL_TYPE_MIX_AIC_1_0):
        kernel_handle = 'g_kernel_handle_aic'
        buff.write(r'''if (g_kernel_handle_aic == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
''')
    elif mode in (CodeMode.AIV, CodeMode.KERNEL_TYPE_AIV_ONLY, CodeMode.KERNEL_TYPE_MIX_AIV_1_0):
        kernel_handle = 'g_kernel_handle_aiv'
        buff.write(r'''if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
''')
    else:
        raise(f"[ERROR]: mode do not support!")

    buff.write('    uint32_t ret = LaunchAscendKernel(')
    buff.write(f'{kernel_handle}, func_key, {block_dim}, args, size, {stream});\n')

    buff.write(r'''    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
''')
    return buff.getvalue()


def generate_aclrtlaunch_for_normal(func_sign: FuncSign,
                            func_key: int,
                            mode: CodeMode,
                            dump_info: dict) -> str:
    param_names = tuple(get_param_names_by_func_sign(func_sign))
    block_dim_name = param_names[0]
    stream_name = param_names[1]
    name = func_sign.func_name.replace("aclrtlaunch_", "")
    launch_code = f'''
uint32_t launch_and_profiling_{name}(uint64_t func_key, uint32_t {block_dim_name}, void* {stream_name}, void **args, uint32_t size)
{{
    uint64_t startTime;
    const char *name = "{name}";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {{
        StartAscendProf(name, &startTime);
    }}
    '''

    launch_code += generate_launch_kernel_code(mode, func_key, block_dim_name, stream_name)
    launch_code += '    if (profStatus) {\n'
    launch_code += f'        ReportAscendProf(name, {block_dim_name}, {mode.value}, startTime);\n'
    launch_code += '    }\n'
    launch_code += '    return ret;\n'
    launch_code += "}\n\n"
    return launch_code


def get_template_hash_tiling_key(template_id: int,
                            base_key: int) -> int:
    return template_id * FUN_TEMPLATE_HASH_TILING_KEY_BASE + base_key


def dehash_template_id(hash_tiling_key: int) -> int:
    return hash_tiling_key // FUN_TEMPLATE_HASH_TILING_KEY_BASE


def replace_func_params_with_specilization_typename(func_sign: FuncSign,
                                                    template_key: int,
                                                    start_idx: int) -> Tuple[FuncParam, ...]:
    '''if func sign uses template typename, replace it before generating func sign'''
    split_args = func_sign.func_params_specilization_args[template_key].split(',')
    split_args = [item.strip() for item in split_args]
    replacement = tuple(split_args)
    new_func_params = (
        *func_sign.func_params[:start_idx],
        *(
            FuncParam(
                parts=(fp.parts[0] if fp.cce_global else new_def, fp.parts[1]),
                cce_global=fp.cce_global,
                tiling_struct=fp.tiling_struct
            )
            for fp, new_def in zip(func_sign.func_params[start_idx:], replacement)
        )
    )
    return new_func_params


def generate_func_impl_code(func_sign: FuncSign,
                            func_key: int,
                            mode: CodeMode,
                            dump_info: dict) -> str:
    """Generate func impl code."""
    param_names = tuple(get_param_names_by_func_sign(func_sign))
    block_dim_name = param_names[0]
    stream_name = param_names[1]
    name = func_sign.func_name.replace("aclrtlaunch_", "")
    buff = io.StringIO()

    new_func_params = func_sign.func_params
    if func_sign.func_template_decl:
        # set start index 2 to skip blockdim and stream definitions
        new_func_params = replace_func_params_with_specilization_typename(func_sign, dehash_template_id(func_key), 2)
        new_func_params = tiling_add_ref_or_ptr_func_params(func_sign, new_func_params)

        func_params_str = ', '.join(
            map(partial(func_param_to_string, with_cce_attr=False), new_func_params)
        )
        func_declare_code = f'''
template<>
{func_sign.return_type} {func_sign.func_name}\
<{func_sign.func_template_specilization_args[dehash_template_id(func_key)]}>({func_params_str})'''
    else:
        func_declare_code = add_extern_c(func_sign_to_string(func_sign, "", "", False, False))

    buff.write(f'{func_declare_code}\n')
    buff.write('{\n')
    args_declare_code = indent_code(
        generate_args_declare_code(
            mode,
            tiling_remove_ref_or_ptr_func_params(
                func_sign,
                remove_added_func_params(new_func_params)
            ),
            dump_info["dump_type"]
        )
    )
    buff.write(args_declare_code)
    buff.write('\n')
    buff.write("    uint32_t __ascendc_ret;\n")

    if dump_info["dump_type"] != "":
        buff.write(f'''#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    constexpr uint32_t __ascendc_one_core_dump_size = {str(dump_info["dump_size"])};
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * 75);
#endif
''')
    buff.write('''    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
''')

    if mode in (CodeMode.MIX,
                CodeMode.KERNEL_TYPE_MIX_AIV_1_0,
                CodeMode.KERNEL_TYPE_MIX_AIC_1_0,
                CodeMode.KERNEL_TYPE_MIX_AIC_1_1,
                CodeMode.KERNEL_TYPE_MIX_AIC_1_2):
        get_sync_addr_code = indent_code(generate_get_sync_addr_code())
        buff.write(get_sync_addr_code)
        buff.write('\n')

    args_assign_code = indent_code(
        generate_args_assign_code(
            mode, func_sign
        )
    )
    buff.write(args_assign_code)
    buff.write('\n')
    if mode == CodeMode.MIX_VECTOR_CORE or "printf" in dump_info["dump_type"] or "timestamp" in dump_info["dump_type"]:
        buff.write("    const char *__ascendc_name = \"{}\";\n".format(name))

    if "assert" in dump_info["dump_type"]:
        buff.write('    ascendc_set_exception_dump_info(__ascendc_one_core_dump_size);\n')
    if mode == CodeMode.MIX_VECTOR_CORE:
        mix_vector_core_launch_code = f'''
    uint32_t __ascendc_aicBlockDim;
    uint32_t __ascendc_aivBlockDim;
    __ascendc_ret = GetCoreNumForMixVectorCore(&__ascendc_aicBlockDim, &__ascendc_aivBlockDim);
    if ({block_dim_name} <= __ascendc_aicBlockDim) {{
        __ascendc_ret = launch_and_profiling_{name}({func_key}, {block_dim_name}, {stream_name}, (void **)&__ascendc_args, sizeof(__ascendc_args));
    }} else {{
        uint32_t __ascendc_totalCoreNum = __ascendc_aicBlockDim + __ascendc_aivBlockDim;
        if ({block_dim_name} > __ascendc_totalCoreNum) {{
            __ascendc_aicBlockDim = ({block_dim_name} * __ascendc_aicBlockDim + \\
                                    __ascendc_totalCoreNum - 1U) / __ascendc_totalCoreNum;
        }}
        __ascendc_aivBlockDim = {block_dim_name} - __ascendc_aicBlockDim;
        bool __ascendc_profStatus = GetAscendProfStatus();
        uint32_t __ascendc_aivBlockDimOffset = __ascendc_aicBlockDim;
        __ascendc_ret = LaunchAscendKernelForVectorCore(__ascendc_name, g_kernel_handle, {func_key}, \\
                        (void **)&__ascendc_args, sizeof(__ascendc_args), {stream_name}, __ascendc_profStatus,\\
                        __ascendc_aicBlockDim, __ascendc_aivBlockDim, __ascendc_aivBlockDimOffset);
    }}
'''
        buff.write(mix_vector_core_launch_code)
    else:
        buff.write(f"    __ascendc_ret = launch_and_profiling_{name}({func_key}, {block_dim_name}, \
{stream_name}, (void **)&__ascendc_args, sizeof(__ascendc_args));\n")

    buff.write('''    KernelHandleGradUnregister::GetInstance();
''')

    if dump_info["dump_type"] != "":
        buff.write('#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON\n')
        if "printf" in dump_info["dump_type"] or "timestamp" in dump_info["dump_type"]:
            buff.write(f'    Adx::AdumpPrintWorkSpace(__ascendc_args.__ascendc_dump, \
__ascendc_one_core_dump_size * 75, {stream_name}, __ascendc_name);\n')
        buff.write('    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);\n#endif\n')
    buff.write('    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);\n')

    buff.write(indent_code('return __ascendc_ret;\n'))
    buff.write('}\n')
    return buff.getvalue()


def remove_comments(code: str) -> str:
    """Remove comments"""
    return COMMENT.sub('', code)


def has_mode_in_func_groups(mode: CodeMode,
                            func_groups: Iterator[FuncSignGroupWithModeBase]) -> bool:
    """Has mode in function groups."""
    return any(map(lambda x: x.mode == mode, func_groups))


def has_assert_in_func_groups(func_groups: List[FuncSignGroupWithModeBase]) -> bool:
    for func_group in func_groups:
        if "assert" in func_group.dump_info["dump_type"]:
            return True

    return False


def generate_host_stub_code(func_groups: List[FuncSignGroupWithModeBase],
                            type_definition: str) -> str:
    """Generate host_stub.cpp code."""
    has_mode_func = partial(has_mode_in_func_groups, func_groups=func_groups)
    dump_assert = has_assert_in_func_groups(func_groups)
    has_mix = (has_mode_func(CodeMode.MIX) or
               has_mode_func(CodeMode.NORMAL) or
               has_mode_func(CodeMode.MIX_VECTOR_CORE) or
               has_mode_func(CodeMode.KERNEL_TYPE_MIX_AIC_1_1) or
               has_mode_func(CodeMode.KERNEL_TYPE_MIX_AIC_1_2))
    has_aic = (has_mode_func(CodeMode.AIC) or
               has_mode_func(CodeMode.KERNEL_TYPE_AIC_ONLY) or
               has_mode_func(CodeMode.KERNEL_TYPE_MIX_AIC_1_0))
    has_aiv = (has_mode_func(CodeMode.AIV) or
               has_mode_func(CodeMode.KERNEL_TYPE_AIV_ONLY) or
               has_mode_func(CodeMode.KERNEL_TYPE_MIX_AIV_1_0))
    head_code = generate_host_stub_head_code(has_mix, has_aic, has_aiv, dump_assert)
    buff = io.StringIO()
    buff.write(head_code)
    buff.write('\n')

    buff.write(type_definition)
    buff.write('\n\n')

    for func_group in func_groups:
        for key, func_sign in enumerate(func_group.func_signs, func_group.base_key):
            buff.write('\n')

            # use void* stream in host_stub.cpp
            normal_launch_code = generate_aclrtlaunch_for_normal(func_sign, key, func_group.mode, func_group.dump_info)
            buff.write(normal_launch_code)

            param_names = tuple(get_param_names_by_func_sign(func_sign))
            block_dim_name = param_names[0]
            stream_name = param_names[1]
            new_func_sign = replace_func_sign_stream_param(func_sign, stream_name)

            if new_func_sign.func_template_decl:
                func_template_declare_code = f'template<{new_func_sign.func_template_decl}>\n\
{func_sign_to_string(new_func_sign, "", "", False, False)};'
                buff.write(f'{func_template_declare_code}\n')
                for template_id, _ in enumerate(new_func_sign.func_template_specilization_args, 0):
                    template_key = get_template_hash_tiling_key(template_id, key)
                    func_impl_code = generate_func_impl_code(new_func_sign, template_key, func_group.mode,
                                                             func_group.dump_info)
                    buff.write(func_impl_code)
            else:
                func_impl_code = generate_func_impl_code(new_func_sign, key, func_group.mode, func_group.dump_info)
                buff.write(func_impl_code)

    return remove_comments(buff.getvalue())


def check_args(args: argparse.Namespace):
    """Check arguments."""
    if args.dynamic_mode:
        if not args.aiv_o:
            raise ArgumentError('--aiv-o is needed in dynamic mode.')
        if not args.aic_o:
            raise ArgumentError('--aic-o is needed in dynamic mode.')


def get_ket_name(name: str):
    name_map = {
        'AIC': 'AIC',
        'AIV': 'AIV',
        'MIX': 'MIX',
        'NORMAL': 'NORMAL',
        'MIX_VECTOR_CORE': 'MIX_VECTOR_CORE',
        'KERNEL_TYPE_AIV_ONLY': 'AIV',
        'KERNEL_TYPE_AIC_ONLY': 'AIC',
        'KERNEL_TYPE_MIX_AIV_1_0': 'AIV',
        'KERNEL_TYPE_MIX_AIC_1_0': 'AIC',
        'KERNEL_TYPE_MIX_AIC_1_1': 'MIX',
        'KERNEL_TYPE_MIX_AIC_1_2': 'MIX'
    }
    return name_map[name]


def get_base_keys(func_groups: Iterator[FuncSignGroup],
                  modes: Iterator[CodeMode]) -> Iterator[int]:
    """Get base keys."""
    cnt = {
        'AIC': 0,
        'AIV': 0,
        'MIX': 0,
        'NORMAL': 0,
        'MIX_VECTOR_CORE': 0
    }
    for func_group, mode in zip(func_groups, modes):
        yield cnt[get_ket_name(mode.name)]
        cnt[get_ket_name(mode.name)] += len(func_group.func_signs)


def get_stub_impl_cpp_filepath(dst_dir: str) -> str:
    """Get host_stub.cpp destination file path."""
    return os.path.join(dst_dir, 'host_stub.cpp')


def get_triple_chevrons_impl_cpp_filepath(dst_dir: str) -> str:
    """Get host_stub.cpp destination file path."""
    return os.path.join(dst_dir, 'triple_chevrons_config.cmake')


def get_normal_config_filepath(dst_dir: str) -> str:
    """Get normal_config.cmake destination file path."""
    return os.path.join(dst_dir, 'normal_config.cmake')


def get_host_config_filepath(dst_dir: str) -> str:
    """Get normal_config.cmake destination file path."""
    return os.path.join(dst_dir, 'host_config.cmake')


def get_aic_config_filepath(dst_dir: str) -> str:
    """Get aic_config.cmake destination file path."""
    return os.path.join(dst_dir, 'aic_config.cmake')


def get_aiv_config_filepath(dst_dir: str) -> str:
    """Get aiv_config.cmake destination file path."""
    return os.path.join(dst_dir, 'aiv_config.cmake')


def save_file(filepath: str, content: str):
    """Save file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            os.chmod(filepath, stat.S_IRUSR + stat.S_IWUSR)
            file.write(content)
    except Exception as err:
        print("[ERROR]: write file failed, filename is {}".format(filepath))
        raise err


def do_save_command(save_command: Tuple[str, str]):
    """Do save command."""
    filepath, content = save_command
    save_file(filepath, content)


def do_save_commands(save_commands: Iterator[Tuple[str, str]]):
    """Do save commands."""
    tuple(map(do_save_command, save_commands))


def generate_save_stub_header_commands(func_signs: Iterator[FuncSign],
                                       dst_dir: str) -> Iterator[Tuple[str, str]]:
    """Generate save stub header commands."""
    triple_chevrons_func_headers = ""
    for func_sign in func_signs:
        header_filepath = os.path.join(
            dst_dir, get_header_name_by_func_sign(func_sign)
        )
        yield header_filepath, generate_header_code(func_sign)
        triple_chevrons_func_headers += generate_kernel_header_code(func_sign)

    file_path = os.path.join(dst_dir, "aclrtlaunch_triple_chevrons_func.h")
    yield file_path, triple_chevrons_func_headers


def generate_save_stub_impl_cpp_commands(func_groups: List[FuncSignGroupWithModeBase],
                                         type_definition: str,
                                         dst_dir: str) -> Iterator[Tuple[str, str]]:
    """Generate save stub impl cpp commands."""
    yield (
        get_stub_impl_cpp_filepath(dst_dir),
        generate_host_stub_code(func_groups, type_definition)
    )


def get_mode_by_ofile(aic_o: str, aiv_o: str) -> CodeMode:
    """Get mode by aic ofile and aiv ofile.
    consider user swap aiv_o and aic_o value.
    """
    ret, aic_mode, err_msg = get_code_channel(aic_o)
    if not ret:
        raise GetOfileModeError(f'aic: {err_msg}')

    ret, aiv_mode, err_msg = get_code_channel(aiv_o)
    if not ret:
        raise GetOfileModeError(f'aiv: {err_msg}')

    modes = sorted([aic_mode, aiv_mode])

    if modes == [CODE_DEFAULT, CODE_DEFAULT]:
        # default aiv mode.
        return CodeMode.AIV

    if modes == [CODE_DEFAULT, CODE_AIC]:
        return CodeMode.AIC

    if modes == [CODE_DEFAULT, CODE_AIV]:
        return CodeMode.AIV

    if modes == [CODE_AIC, CODE_AIV]:
        return CodeMode.MIX

    if modes == [CODE_AIC, CODE_AIC]:
        raise GetOfileModeError('both aic mode is illegal.')

    if modes == [CODE_AIV, CODE_AIV]:
        raise GetOfileModeError('both aiv mode is illegal.')

    raise GetOfileModeError('unreachable scene.')


def get_mode_dynamic(aic_o: str, aiv_o: str, kernel_type:CodeMode) -> CodeMode:
    if kernel_type is None:
        return get_mode_by_ofile(aic_o, aiv_o)
    return kernel_type


def load_compile_commands(filepath: str) -> Tuple[Dict[str, str], Dict[str, bool]]:
    """Load compile_commands.json."""
    source_mapping = {}
    enable_ascendc_time_stamp = {}
    try:
        with open(filepath, encoding='utf-8') as file:
            compile_commands = json.load(file)
    except Exception as err:
        print("[ERROR]: read json file failed, json file is {}".format(filepath))
        raise err

    for compile_info in compile_commands:
        command_list = compile_info['command'].split()
        if '-o' in command_list:
            directory = compile_info['directory']
            source = compile_info['file']

            index = command_list.index('-o')
            output_file = command_list[index+1]
            absolute_obj = os.path.join(directory, output_file)

            source_mapping[absolute_obj] = source
            if "-DASCENDC_TIME_STAMP_ON" in command_list:
                enable_ascendc_time_stamp[absolute_obj] = True
            else:
                enable_ascendc_time_stamp[absolute_obj] = False
    return source_mapping, enable_ascendc_time_stamp


def get_preprocessed_source_filepath(filepath: str, source_mapping: Dict) -> str:
    """Get preprocessed file's source file path."""
    absolute_filepath = os.path.abspath(filepath)
    return source_mapping[absolute_filepath]


def trans_device_cpp_filename(filename: str) -> str:
    """Transform device cpp filename."""
    return f'auto_gen_{filename}'


def get_all_sources(func_groups: List[FuncSignGroup], source_mapping, dst_dir: str):
    """Get all sources."""
    all_sources = list(
        os.path.join(
            dst_dir,
            trans_device_cpp_filename(
                os.path.basename(
                    get_preprocessed_source_filepath(func_group.filepath, source_mapping)
                )
            )
        )
        for func_group in func_groups
    )
    return all_sources


def get_definition_suffix(gen_mode: CodeMode, func_group_mode: CodeMode) -> str:
    """Get function definition concatenation."""
    if func_group_mode in (CodeMode.MIX,
                           CodeMode.MIX_VECTOR_CORE,
                           CodeMode.KERNEL_TYPE_MIX_AIV_1_0,
                           CodeMode.KERNEL_TYPE_MIX_AIC_1_0,
                           CodeMode.KERNEL_TYPE_MIX_AIC_1_1,
                           CodeMode.KERNEL_TYPE_MIX_AIC_1_2):
        if gen_mode == CodeMode.AIC:
            return '_mix_aic'
        if gen_mode == CodeMode.AIV:
            return '_mix_aiv'
        return ''
    return ''



def source_properties(func_groups: Iterator[FuncSignGroupWithModeBase],
                      all_sources: Iterator[str],
                      gen_mode: CodeMode,
                      generate_definition: bool):
    """Get source properites."""
    for source, func_group in zip(all_sources, func_groups):
        suffix = get_definition_suffix(gen_mode, func_group.mode)
        is_one_to_one = func_group.mode == CodeMode.KERNEL_TYPE_MIX_AIC_1_1
        definitions = []
        for idx, func_sign in enumerate(func_group.func_signs, func_group.base_key):
            compile_section = ""
            if func_sign.func_template_decl:
                for template_id, _ in enumerate(func_sign.func_template_specilization_args, 0):
                    template_func_name = f'{func_sign.func_name}_template_{template_id}'
                    compile_section += f';{add_auto_gen_prefix_and_kernel_suffix(template_func_name)}='
                    template_key = get_template_hash_tiling_key(template_id, idx)
                    if generate_definition:
                        compile_section += f'{func_sign.func_name}_{template_key}{suffix}'
                    else:
                        compile_section += f'{func_sign.func_name}{suffix}'
            else:
                compile_section += f';{add_auto_gen_prefix_and_kernel_suffix(func_sign.func_name)}='
                if generate_definition:
                    compile_section += f'{func_sign.func_name}_{idx}{suffix}'
                else:
                    compile_section += f'{func_sign.func_name}{suffix}'
            compile_section += f';ONE_CORE_DUMP_SIZE={str(func_group.dump_info["dump_size"])}'
            if func_group.mode.value >= 9 and func_group.mode.value <= 10:
                compile_section += f';{MIX_CORE_MACRO}={1}'
            if is_one_to_one:
                compile_section += f';__MIX_CORE_AIC_RATION__=1'
            definitions.append(compile_section)
        yield (source, definitions)


def generate_config_content(all_sources: List[Tuple[str, List[str]]],
                            all_definitions: List[Tuple[str, List[str]]]) -> str:
    """Generate config.cmake content by all sources and all definitions."""
    buff = io.StringIO()

    for name, sources in all_sources:
        buff.write(f'set({name}\n')
        for source in sources:
            buff.write(f'    {source}\n')
        buff.write(')\n')

    all_definitions.sort()
    for source, definitions in all_definitions:
        if definitions:
            buff.write(f'set_source_files_properties({source}\n')
            definitions_str = ';'.join(definitions)
            buff.write(f'    PROPERTIES COMPILE_DEFINITIONS "{definitions_str}"\n')
            buff.write(')\n')

    return buff.getvalue()


def generate_normal_config_content(func_groups: List[FuncSignGroupWithModeBase],
                                   source_mapping,
                                   dst_dir: str,
                                   generate_definition: bool) -> str:
    """Generate normal_config.cmake content."""
    all_sources = get_all_sources(func_groups, source_mapping, dst_dir)
    all_definitions = list(source_properties(func_groups, all_sources, CodeMode.NORMAL, generate_definition))

    return generate_config_content(
        [
            ('MIX_SOURCES', all_sources),
        ],
        all_definitions
    )


def generate_aic_config_content(mix_groups: List[FuncSignGroupWithModeBase],
                                aic_groups: List[FuncSignGroupWithModeBase],
                                source_mapping,
                                dst_dir: str,
                                generate_definition: bool) -> str:
    """Generate aic_config.cmake content."""
    mix_sources = get_all_sources(mix_groups, source_mapping, dst_dir)
    aic_sources = get_all_sources(aic_groups, source_mapping, dst_dir)


    all_definitions = list(
        chain(
            source_properties(mix_groups, mix_sources, CodeMode.AIC, generate_definition),
            source_properties(aic_groups, aic_sources, CodeMode.AIC, generate_definition),
        )
    )

    return generate_config_content(
        [
            ('MIX_SOURCES', mix_sources),
            ('AIC_SOURCES', aic_sources),
        ],
        all_definitions
    )


def generate_aiv_config_content(mix_groups: List[FuncSignGroupWithModeBase],
                                aiv_groups: List[FuncSignGroupWithModeBase],
                                source_mapping,
                                dst_dir: str,
                                generate_definition: bool) -> str:
    """Generate aiv_config.cmake content."""
    mix_sources = get_all_sources(mix_groups, source_mapping, dst_dir)
    aiv_sources = get_all_sources(aiv_groups, source_mapping, dst_dir)

    all_definitions = list(
        chain(
            source_properties(mix_groups, mix_sources, CodeMode.AIV, generate_definition),
            source_properties(aiv_groups, aiv_sources, CodeMode.AIV, generate_definition),
        )
    )

    return generate_config_content(
        [
            ('MIX_SOURCES', mix_sources),
            ('AIV_SOURCES', aiv_sources),
        ],
        all_definitions
    )


def save_normal_config_cmake(func_groups: List[FuncSignGroupWithBase],
                             source_mapping,
                             dst_dir: str,
                             generate_definition: bool):
    """Save normal_config.cmake."""
    save_file(
        get_normal_config_filepath(dst_dir),
        generate_normal_config_content(func_groups, source_mapping, dst_dir, generate_definition)
    )


def save_v200_config_cmake(func_groups: List[FuncSignGroupWithBase],
                             source_mapping,
                             dst_dir: str,
                             generate_definition: bool):
    """Save v200.cmake."""
    mode_groups = grouptodict(func_groups, attrgetter('mode'))
    aicore_func_groups = mode_groups.get(CodeMode.AIC, [])
    mix_func_groups = mode_groups.get(CodeMode.MIX_VECTOR_CORE, [])

    if mix_func_groups:
        aiv_content = generate_aiv_config_content(
            mix_func_groups, [], source_mapping, dst_dir, generate_definition
        )
    else:
        aiv_content = ''
    save_file(get_aiv_config_filepath(dst_dir), aiv_content)

    if mix_func_groups or aicore_func_groups:
        aic_content = generate_aic_config_content(
            mix_func_groups, aicore_func_groups, source_mapping, dst_dir, generate_definition
        )
    else:
        aic_content = ''
    save_file(get_aic_config_filepath(dst_dir), aic_content)


def generate_host_config_content(func_groups: List[FuncSignGroupWithBase], source_mapping) -> str:
    """Generate host_config.cmake content."""
    buff = io.StringIO()

    for func_group in func_groups:
        buff.write(f'set_source_files_properties({source_mapping[func_group.filepath]}\n')
        buff.write(f'    PROPERTIES COMPILE_DEFINITIONS ')
        buff.write(f'"ONE_CORE_DUMP_SIZE={str(func_group.dump_info["dump_size"])}"\n')
        buff.write(')\n')

    return buff.getvalue()


def save_host_config_cmake(func_groups: List[FuncSignGroupWithBase], source_mapping, dst_dir: str):
    """Save host_config.cmake."""
    save_file(
        get_host_config_filepath(dst_dir),
        generate_host_config_content(func_groups, source_mapping)
    )


def save_aic_aiv_config_cmake(func_groups: List[FuncSignGroupWithModeBase],
                              source_mapping,
                              dst_dir: str,
                              generate_definition: bool):
    """Save aic_config.cmake and aiv_config.cmake."""
    mode_groups = grouptodict(func_groups, attrgetter('mode'))

    mix_func_groups = mode_groups.get(CodeMode.MIX, [])
    mix_func_groups += mode_groups.get(CodeMode.KERNEL_TYPE_MIX_AIC_1_1, [])
    mix_func_groups += mode_groups.get(CodeMode.KERNEL_TYPE_MIX_AIC_1_2, [])


    aic_func_groups = mode_groups.get(CodeMode.AIC, [])
    aic_func_groups += mode_groups.get(CodeMode.KERNEL_TYPE_AIC_ONLY, [])

    aiv_func_groups = mode_groups.get(CodeMode.AIV, [])
    aiv_func_groups += mode_groups.get(CodeMode.KERNEL_TYPE_AIV_ONLY, [])

    aic_func_groups += mode_groups.get(CodeMode.KERNEL_TYPE_MIX_AIC_1_0, [])
    aiv_func_groups += mode_groups.get(CodeMode.KERNEL_TYPE_MIX_AIV_1_0, [])
    if mix_func_groups or aic_func_groups:
        aic_content = generate_aic_config_content(
            mix_func_groups, aic_func_groups, source_mapping, dst_dir, generate_definition
        )
    else:
        aic_content = ''
    save_file(get_aic_config_filepath(dst_dir), aic_content)



    if mix_func_groups or aiv_func_groups:
        aiv_content = generate_aiv_config_content(
            mix_func_groups, aiv_func_groups, source_mapping, dst_dir, generate_definition
        )
    else:
        aiv_content = ''
    save_file(get_aiv_config_filepath(dst_dir), aiv_content)


def get_kernel_type(func_code_mode: CodeMode):
    """Get kernel type."""
    mode_to_ktype = {}
    if is_v220_mode():
        mode_to_ktype = {
            CodeMode.AIC : ["FunLevelKType", "K_TYPE_AIC"],
            CodeMode.AIV : ["FunLevelKType", "K_TYPE_AIV"],
            CodeMode.MIX : ["FunLevelMixCoreType", "K_TYPE_MIX_AIC_MAIN"],
            CodeMode.KERNEL_TYPE_AIC_ONLY : ["FunLevelKType", "K_TYPE_AIC"],
            CodeMode.KERNEL_TYPE_AIV_ONLY : ["FunLevelKType", "K_TYPE_AIV"],
            CodeMode.KERNEL_TYPE_MIX_AIV_1_0 : ["FunLevelMixCoreType", "K_TYPE_MIX_AIV_MAIN"],
            CodeMode.KERNEL_TYPE_MIX_AIC_1_0 : ["FunLevelMixCoreType", "K_TYPE_MIX_AIC_MAIN"],
            CodeMode.KERNEL_TYPE_MIX_AIC_1_1 : ["FunLevelMixCoreType", "K_TYPE_MIX_AIC_MAIN"],
            CodeMode.KERNEL_TYPE_MIX_AIC_1_2 : ["FunLevelMixCoreType", "K_TYPE_MIX_AIC_MAIN"],
        }
    else:
        mode_to_ktype = {
            CodeMode.AIC : ["FunLevelKType", "K_TYPE_AIC"],
            CodeMode.AIV : ["FunLevelKType", "K_TYPE_AIV"],
            CodeMode.MIX : ["FunLevelKType", "K_TYPE_MIX_AIC_MAIN"],
            CodeMode.NORMAL : ["FunLevelKType", "K_TYPE_AICORE"],
        }
    if func_code_mode in CodeMode:
        return mode_to_ktype[func_code_mode]
    else:
        raise ArgumentError(f'Unsupported CodeMode: {func_code_mode}')


def get_func_meta_type(func_meta_type: FuncMetaType):
    """Get func meta type."""
    if func_meta_type == FuncMetaType.F_TYPE_KTYPE:
        return "F_TYPE_KTYPE"
    elif func_meta_type == FuncMetaType.F_TYPE_CROSS_CORE_SYNC:
        return "F_TYPE_CROSS_CORE_SYNC"
    elif func_meta_type == FuncMetaType.F_TYPE_MAX:
        return "F_TYPE_MAX"
    else:
        raise ArgumentError(f'Unsupported FuncMetaType: {func_meta_type}')


def get_ktype_section_variable(variable_name: str,
                               section_func_name: str,
                               func_meta_type: FuncMetaType,
                               func_code_mode: CodeMode):
    type_struc_name, k_type = get_kernel_type(func_code_mode)
    section_var = ""
    if func_code_mode in (CodeMode.KERNEL_TYPE_MIX_AIC_1_1,
                          CodeMode.KERNEL_TYPE_MIX_AIC_1_2,
                          CodeMode.KERNEL_TYPE_MIX_AIV_1_0,
                          CodeMode.KERNEL_TYPE_MIX_AIC_1_0,
                          CodeMode.MIX):
        if "mix_aic" in variable_name:
            section_var += f"#if defined(__DAV_C220_CUBE__)\n"
        elif "mix_aiv" in variable_name:
            section_var += f"#if defined(__DAV_C220_VEC__)\n"
    section_var += f"static const struct {type_struc_name} {variable_name} __attribute__ "
    section_var += f"((used, section (\".ascend.meta.{section_func_name}\"))) = "
    section_var += f"{{ {{ {{{get_func_meta_type(func_meta_type)}, sizeof(unsigned int)}}, {k_type}}}"
    if func_code_mode == CodeMode.KERNEL_TYPE_MIX_AIV_1_0:
        section_var += f", {{{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}}, 0, 1}}"
    if func_code_mode == CodeMode.KERNEL_TYPE_MIX_AIC_1_0:
        section_var += f", {{{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}}, 1, 0}}"
    if func_code_mode == CodeMode.KERNEL_TYPE_MIX_AIC_1_1:
        section_var += f", {{{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}}, 1, 1}}"
    if func_code_mode in (CodeMode.KERNEL_TYPE_MIX_AIC_1_2, CodeMode.MIX):
        section_var += f", {{{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}}, 1, 2}}"
    section_var += f" }};\n"
    if func_code_mode in (CodeMode.KERNEL_TYPE_MIX_AIC_1_1,
                          CodeMode.KERNEL_TYPE_MIX_AIC_1_2,
                          CodeMode.KERNEL_TYPE_MIX_AIV_1_0,
                          CodeMode.KERNEL_TYPE_MIX_AIC_1_0,
                          CodeMode.MIX):
        section_var += f"#endif\n"
    return section_var


def gen_ktype_section(func_sign: FuncSign,
                      base_key: int,
                      genmode: CodeMode,
                      generate_ktype_section: bool) -> str:
    cur_kernel_name = func_sign.func_name
    section = ''
    if not generate_ktype_section:
        cur_kernel_name = func_sign.func_name + f'_{base_key}'
    if genmode in (CodeMode.MIX,
                   CodeMode.KERNEL_TYPE_MIX_AIC_1_1,
                   CodeMode.KERNEL_TYPE_MIX_AIC_1_2):
        section += get_ktype_section_variable(f"{func_sign.func_name}_mix_aic_section",
                                                f"{cur_kernel_name}_mix_aic",
                                                FuncMetaType.F_TYPE_KTYPE,
                                                genmode)

        section += get_ktype_section_variable(f"{func_sign.func_name}_mix_aiv_section",
                                                f"{cur_kernel_name}_mix_aiv",
                                                FuncMetaType.F_TYPE_KTYPE,
                                                genmode)
    elif genmode == CodeMode.KERNEL_TYPE_MIX_AIV_1_0:
        section += get_ktype_section_variable(f"{func_sign.func_name}_mix_aiv_section",
                                                f"{cur_kernel_name}_mix_aiv",
                                                FuncMetaType.F_TYPE_KTYPE,
                                                genmode)
    elif genmode == CodeMode.KERNEL_TYPE_MIX_AIC_1_0:
        section += get_ktype_section_variable(f"{func_sign.func_name}_mix_aic_section",
                                                f"{cur_kernel_name}_mix_aic",
                                                FuncMetaType.F_TYPE_KTYPE,
                                                genmode)
    else:
        section += get_ktype_section_variable(f"{func_sign.func_name}_section",
                                                f"{cur_kernel_name}",
                                                FuncMetaType.F_TYPE_KTYPE,
                                                genmode)
    return section


def generate_kernel_auto_gen_func_impl(func_group: FuncSignGroupWithModeBase,
                                       func_sign: FuncSign,
                                       is_mix: bool,
                                       template_id: int = 0,
                                       generate_ktype_section: bool = False):
    source = ""
    new_func_params = func_sign.func_params
    if func_sign.func_template_decl:
        auto_gen_func_name = f'{func_sign.func_name}_template_{template_id}'
        new_func_params = replace_func_params_with_specilization_typename(func_sign, template_id, 0)
    else:
        auto_gen_func_name = func_sign.func_name

    new_func_sign = func_sign._replace(
        return_type='void',
        func_name=add_auto_gen_prefix_and_kernel_suffix(auto_gen_func_name),
        func_params=new_func_params,
    )
    if is_mix:
        new_func_sign = new_func_sign._replace(
            func_params=(FuncParam(('GM_ADDR', 'ffts_addr')),) + new_func_sign.func_params,
        )

    param_names = tuple(get_param_names_by_func_sign(func_sign))

    extra_param = "\n"
    if func_group.dump_info["dump_type"] != "":
        extra_param += "#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON\n"
        if len(new_func_sign.func_params) == 0:
            extra_param += "GM_ADDR dumpAddr\n"
        else:
            extra_param += "GM_ADDR dumpAddr,\n"
        extra_param += "#endif\n"

    suffix_extra_param = ', GM_ADDR overflow_status'
    if (extra_param == "\n") and (not new_func_sign.func_params):
        suffix_extra_param = 'GM_ADDR overflow_status'

    source += ("extern \"C\" __global__ [aicore] "
               f"{func_sign_to_string(new_func_sign, extra_param, suffix_extra_param, True)} ")
    source += "{\n"

    if func_group.dump_info["dump_type"] != "":
        source += "#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON\n"
        if is_mix:
            source += "    AscendC::InitDump(true, dumpAddr, ONE_CORE_DUMP_SIZE);\n"
        else:
            source += "    AscendC::InitDump(false, dumpAddr, ONE_CORE_DUMP_SIZE);\n"
        source += "#ifdef ASCENDC_TIME_STAMP_ON\n"
        source += "    AscendC::PrintTimeStamp(static_cast<uint32_t>\
(AscendC::TimeStampId::TIME_STAMP_WRAP_INIT_DUMP));\n"
        source += "#endif\n"
        source += "#endif\n\n"
    if is_mix:
        source += "    icache_preload(1);\n"
        source += "    if (ffts_addr != nullptr) {\n"
        source += "        set_ffts_base_addr((uint64_t)ffts_addr);\n"
        source += "    }\n"
        source += "#ifdef ASCENDC_TIME_STAMP_ON\n"
        source += "    AscendC::PrintTimeStamp(static_cast<uint32_t>\
(AscendC::TimeStampId::TIME_STAMP_WRAP_FFTS_ADDR));\n"
        source += "#endif\n"
    if "printf" in func_group.dump_info["dump_type"]:
        source += "#ifdef ASCENDC_DUMP\n"
        source += "    uint64_t __ascendc_tStamp = 0;\n"
        source += "    uint64_t __ascendc_version = 0;\n"
        source += "     __gm__ char* __ascendc_versionStr = nullptr;\n"
        source += "    GetCannVersion(__ascendc_versionStr, __ascendc_version, __ascendc_tStamp);\n"
        source += "    if (__ascendc_tStamp == 0) {\n"
        source += "        AscendC::printf(\"[WARNING]: CANN TimeStamp is invalid, \
CANN TimeStamp is %u\\n\", __ascendc_tStamp);\n"
        source += "    } else {\n"
        source += "        AscendC::printf(\"CANN Version: %s, TimeStamp: %u\\n\", \
(__gm__ const char*)(__ascendc_versionStr), __ascendc_tStamp);\n"
        source += "    }\n"
        source += "#endif\n"
    # if it has no param, there should be no asignment of workspace or tiling
    if len(param_names) > 0:
        source += "#if defined(HAVE_WORKSPACE)\n"
        source += "    GM_ADDR workspace_param;\n"
        source += "    GM_ADDR workspace_usr;\n"
        # if there is only on param, there should be no tiling input
        if len(param_names) > 1:
            source += "#if defined(HAVE_TILING)\n"
            source += f"    workspace_param = {param_names[-2]};\n"
            source += "#else\n"
            source += f"    workspace_param = {param_names[-1]};\n"
            source += "#endif\n"
        else:
            source += f"    workspace_param = {param_names[-1]};\n"


        if is_mix:
            source += "    if (workspace_param == nullptr) {\n"
            source += "        return;\n"
            source += "    }\n"

        source += "    AscendC::SetSysWorkspaceForce(workspace_param);\n"
        source += "    workspace_usr = AscendC::GetUserWorkspace(workspace_param);\n"

        if is_mix:
            source += f"#if defined(REGIST_MATMUL_OBJ) || defined({MIX_CORE_MACRO})\n"
            source += "    if constexpr (g_coreType == AscendC::AIC) {\n"
            source += "        matmul::clearWorkspace(workspace_param);\n"
            source += "#ifdef ASCENDC_TIME_STAMP_ON\n"
            source += "        AscendC::PrintTimeStamp(static_cast<uint32_t>\
(AscendC::TimeStampId::TIME_STAMP_WRAP_CLEAR_WK_SPAC));\n"
            source += "#endif\n"
            source += "    }\n"
            source += "#endif\n"

        # if there is only on param, there should be no tiling input
        if len(param_names) > 1:
            source += "#if defined(HAVE_TILING)\n"
            source += f"    {param_names[-2]} = workspace_usr;\n"
            source += "#else\n"
            source += f"    {param_names[-1]} = workspace_usr;\n"
            source += "#endif\n"
        else:
            source += f"    {param_names[-1]} = workspace_usr;\n"

        source += "#endif\n"

    param_names_str = ', '.join(param_names)
    if func_sign.func_template_decl:
        source += f"    {add_origin_suffix(func_sign.func_name)}\
<{func_sign.func_template_specilization_args[template_id]}>({param_names_str});\n"
    else:
        source += f"    {add_origin_suffix(func_sign.func_name)}({param_names_str});\n"
    source += ("#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)\n"
               "    AscendC::WriteBackOverflow(overflow_status);\n#endif\n")
    source += "}\n\n"
    # for all specialized template methods, generate one kernel type section
    if (template_id == 0) and (generate_ktype_section or func_group.mode.value >= 5):
        source += gen_ktype_section(func_sign, func_group.base_key, func_group.mode, generate_ktype_section)
    return source


def save_device_kernel_function(func_groups: List[FuncSignGroupWithModeBase],
                                source_mapping: Dict[str, str],
                                dst_dir: str,
                                generate_ktype_section: bool = False):
    """Generate kernel function."""
    for func_group in func_groups:
        src_path = get_preprocessed_source_filepath(func_group.filepath, source_mapping)
        src_filename = os.path.basename(src_path)
        src_without_ext = os.path.splitext(src_filename)[0]
        is_mix = (func_group.mode == CodeMode.MIX or
                  func_group.mode == CodeMode.KERNEL_TYPE_MIX_AIV_1_0 or
                  func_group.mode == CodeMode.KERNEL_TYPE_MIX_AIC_1_0 or
                  func_group.mode == CodeMode.KERNEL_TYPE_MIX_AIC_1_1 or
                  func_group.mode == CodeMode.KERNEL_TYPE_MIX_AIC_1_2)

        # File Isolation Macro
        source = f"#ifndef __{src_without_ext.upper()}__KERNEL_FUN_H__\n"
        source += f"#define __{src_without_ext.upper()}__KERNEL_FUN_H__\n\n"

        # replace __global micro for usr kernel function, and recover after usr kernel function
        source += "#undef __global__\n"
        source += "#define __global__ inline\n"

        for func_sign in func_group.func_signs:
            source += f"#define {func_sign.func_name} {add_origin_suffix(func_sign.func_name)}\n"

        source += f"#include \"{src_path}\"\n\n"

        for func_sign in func_group.func_signs:
            source += f"#undef {func_sign.func_name}\n"

        source += "#undef __global__\n"
        source += "#if ASCENDC_CPU_DEBUG\n"
        source += "#define __global__\n"
        source += "#else\n"
        source += "#define __global__ __attribute__((cce_kernel))\n"
        source += "#endif\n\n"

        dump_size = str(func_group.dump_info["dump_size"])
        source += "#ifndef ONE_CORE_DUMP_SIZE\n"
        source += f"#define ONE_CORE_DUMP_SIZE {dump_size} * 1\n"
        source += "#endif\n\n"

        # generate kernel function
        for func_sign in func_group.func_signs:
            if func_sign.func_template_decl:
                for template_id, _ in enumerate(func_sign.func_template_specilization_args, 0):
                    source += generate_kernel_auto_gen_func_impl(func_group, func_sign, is_mix, template_id,
                                                                 generate_ktype_section)
            else:
                source += generate_kernel_auto_gen_func_impl(func_group, func_sign, is_mix, 0, generate_ktype_section)


        source += "#endif\n"
        source += "#include \"inner_interface/inner_kernel_operator_intf.h\"\n"

        # write code into file
        dst_filepath = os.path.join(dst_dir, trans_device_cpp_filename(src_filename))
        try:
            with open(dst_filepath, 'w', encoding='utf-8') as file:
                os.chmod(dst_filepath, stat.S_IRUSR + stat.S_IWUSR)
                file.write(source)
        except Exception as err:
            print("[ERROR]: write file failed, filename is {}".format(dst_filepath))
            raise err


def search_undefined_types(typenames: Set[str],
                           structs: Dict[str, StructRaw]) -> Iterator[str]:
    """Search undefined types."""
    visited = set()

    def search_recursively(name: str) -> Iterator[str]:
        visited.add(name)
        if name not in structs:
            return
        try:
            type_names = parse_struct_by_str(structs[name].content)
        except ParseError as ex:
            msg = f'Parse struct error! code is \n{structs[name].content}'
            raise ParseError(msg) from ex
        for type_name in type_names:
            if type_name not in visited:
                yield from search_recursively(type_name)
        yield name

    # sorted for stable search order.
    for typename in sorted(typenames):
        if typename not in visited:
            yield from search_recursively(typename)


def generate_type_definition_content(types_need_defined: Iterator[str],
                                     structs: Dict[str, StructRaw]) -> str:
    """Generate type define content."""
    defs = (structs[name].content for name in types_need_defined)
    return '\n\n'.join(defs)


def generate_type_definition_content_by_func_sign_groups(func_sign_groups: List[FuncSignGroup]
                                                         ) -> str:
    """Generate type define content by function signature groups."""
    structs = merge_func_sign_groups_structs(func_sign_groups)
    typenames = typenames_in_func_groups(func_sign_groups)

    types_need_defined = search_undefined_types(typenames, structs)
    return generate_type_definition_content(types_need_defined, structs)


def process_with_source_mapping(func_sign_groups: List[FuncSignGroupWithBase],
                                source_mapping: Dict[str, str],
                                dst_dir: str,
                                save_device_config_cmake_func: Callable,
                                save_host_config_cmake_func: Callable,
                                generate_definition: bool = True,
                                generate_ktype_section: bool = False):
    """Process with source mapping."""
    save_device_kernel_function(func_sign_groups, source_mapping, dst_dir, generate_ktype_section)
    save_device_config_cmake_func(func_sign_groups, source_mapping, dst_dir, generate_definition)
    save_host_config_cmake_func(func_sign_groups, source_mapping, dst_dir)


def main(argv: List[str]):
    """Main process."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filepaths', nargs='+', help='Preprocessed file paths.')
    parser.add_argument('--dynamic-mode', action='store_true', help='Get code mode dynamically.')
    parser.add_argument('-d', '--dst-dir', default='.', help='Destination directory.')
    parser.add_argument('-hd', '--header-dir', default='.', help='Header destination directory.')
    parser.add_argument('--aiv-o', nargs='+', help='Aiv ofile paths.')
    parser.add_argument('--aic-o', nargs='+', help='Aic ofile paths.')
    parser.add_argument('--compile-commands',
                        required=True,
                        help='compile_commands.json file path.')
    parser.add_argument('--generate-definition',
                        action='store_true',
                        help='generate definition')
    parser.add_argument('--generate-ktype-section',
                        action='store_true',
                        help='generate ktype section')
    parser.add_argument('--build-mode', help="Get chip type.")

    args = parser.parse_args(argv)

    dst_dir = os.path.realpath(args.dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if not os.path.exists(args.header_dir):
        os.makedirs(args.header_dir)

    check_args(args)
    global IS_V220_MODE
    IS_V220_MODE = args.build_mode == "aiv"
    try:
        func_sign_groups = list(parse_func_signature_by_filepaths(args.filepaths, args.build_mode))
    except OSError as ex:
        print(f"error: {ex.strerror}: '{ex.filename}'")
        return False

    type_definition = generate_type_definition_content_by_func_sign_groups(func_sign_groups)
    func_sign_groups = list(map(get_func_sign_template_specilization_args, func_sign_groups, args.aiv_o, args.aic_o))

    new_func_sign_groups = list(map(trans_func_sign_group, func_sign_groups))

    try:
        source_mapping, enable_ascendc_time_stamp = load_compile_commands(args.compile_commands)
    except FileNotFoundError as ex:
        print(f'error: {ex}')
        return False

    for func_group in func_sign_groups:
        enbale_flag = enable_ascendc_time_stamp[func_group.filepath]
        if enbale_flag is True:
            if func_group.dump_info["dump_type"] != "":
                func_group.dump_info["dump_type"] = func_group.dump_info["dump_type"] + ",timestamp"
            else:
                func_group.dump_info["dump_type"] = "timestamp"
            func_group.dump_info["dump_size"] = 1048576

    if args.dynamic_mode:
        # Each preprocessed file corresponds to two .o files: aiv.o and aic.o.
        kernel_types = [func_sign_group.kernel_type for func_sign_group in func_sign_groups]
        modes = list(starmap(get_mode_dynamic, zip(args.aic_o, args.aiv_o, kernel_types)))
        save_device_config_cmake_func = save_aic_aiv_config_cmake
    elif args.build_mode == "m200" and args.generate_definition:
        modes = [func_sign_group.kernel_type for func_sign_group in func_sign_groups]
        save_device_config_cmake_func = save_v200_config_cmake
    else :
        modes = repeat(CodeMode.NORMAL)
        save_device_config_cmake_func = save_normal_config_cmake

    base_keys = list(get_base_keys(func_sign_groups, modes))
    func_sign_groups = compose_mode_base_func_groups(
        func_sign_groups, modes, base_keys
    )

    new_func_sign_groups = compose_mode_base_func_groups(
        new_func_sign_groups, modes, base_keys
    )

    do_save_commands(
        generate_save_stub_header_commands(
            func_signs_in_groups(new_func_sign_groups), args.header_dir
        )
    )
    do_save_commands(
        generate_save_stub_impl_cpp_commands(
            new_func_sign_groups, type_definition, dst_dir
        )
    )

    process_with_source_mapping(
        func_sign_groups,
        source_mapping,
        dst_dir,
        save_device_config_cmake_func,
        save_host_config_cmake,
        args.generate_definition,
        args.generate_ktype_section,
    )

    return True


def main_with_except(argv: List[str]):
    """Main process with except exceptions."""
    try:
        return main(argv)
    except ArgumentError as ex:
        print(f'error: check arguments error, {ex}')
        return False


if __name__ == '__main__':
    if not main_with_except(sys.argv[1:]):
        sys.exit(1)
