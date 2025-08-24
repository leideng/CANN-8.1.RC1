#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
ascendc_npuchk_report.py
"""
import os
import sys
import glob
import subprocess


def get_error_type(info_input):
    err_start = info_input.find('[Error')
    if err_start < 0:
        return None
    err_info_str = info_input[err_start + 1:]
    err_stop = err_info_str.find(']')
    if err_stop < 0:
        return None
    return err_info_str[:err_stop]


def parse_log(file, stack):
    cce_intri = ''
    bs_start = False
    err_info = []
    key = ''
    with open(file, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            err_type_line = get_error_type(line)
            if err_type_line is not None:
                err_info.append(line.strip())
                err_info.append(cce_intri)
                key += err_type_line
                continue
            if line.startswith('### '):
                cce_intri = line.strip()
                continue
            if not bs_start and line.find('# BackTrace #') > 0:
                bs_start = True
                continue
            if bs_start and not line.startswith('  '):
                bs_start = False
                if stack.get(key) is None:
                    stack[key] = err_info
                err_info = []
                key = ''
                continue
            if bs_start:
                if line.find('.so') > 0:
                    continue
                line = line.strip()
                binfile = line.split('(')[0]
                if line.find('+') < 0:
                    continue
                addr = line.split('+')[1].split(')')[0]
                info_tmp = binfile + ':' + addr
                err_info.append(info_tmp)
                key += addr


def execute_cmd(cmds):
    proc = subprocess.Popen(cmds, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, encoding='utf-8')
    try:
        outs, errs = proc.communicate(timeout=10)
        if len(errs) > 0:
            print(errs)
        return outs.strip()
    except TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        print("Error:\n", errs)
    return ''


def addr_to_line(bin_file, addr):
    res = execute_cmd(['addr2line', '-f', '-e', bin_file, addr])
    fun_line = res.split('\n')
    fun = ''
    line = ''
    if len(fun_line) > 0:
        fun = fun_line[0]
        fun = execute_cmd(['c++filt', fun])
    if len(fun_line) > 1:
        line = fun_line[1]
    return '{} at {}'.format(fun, line)


if __name__ == "__main__":
    """
    使用场景1: 单log文件解析
              python3 ascendc_npuchk_report.py xxxx/xxx_npuchk.log
    使用场景2: 无log文件输入, 脚本自动在当前路径下找xxx_npuchk.log, 并解析
              python3 ascendc_npuchk_report.py
    使用场景3: 指定路径和cpu bin路径, 在该路径下找log文件, 并解析
              python3 ascendc_npuchk_report.py log_path bin_path
    """
    err_details = {
        'ErrorRead1': '非法内存读取数据: 整段内存未经过AscendC框架的alloc_buf申请或者已free',
        'ErrorRead2': '[可疑问题]读取无效数据：读取的内存部分/全部从未被写过，读取的数据可能是无效数据',
        'ErrorRead3': '读取越界, 长度超出经AscendC框架的alloc_buf申请实际有效的数据(开始/结尾)',
        'ErrorRead4': '读取地址非32字节对齐',
        'ErrorWrite1': '非法内存写入数据: 未经过AscendC框架的alloc_buf申请过或者已经free了',
        'ErrorWrite2': '写入越界, 长度超出经AscendC框架的alloc_buf申请实际有效的数据(开始/结尾)',
        'ErrorWrite3': '[可疑问题]重复写入，前一次写入的内存没有被读取走，重复写入',
        'ErrorWrite4': '写入地址非32字节对齐',
        'ErrorSync1': '写入存在同步问题, pipe内缺少pipe barrier/pipe间缺少set/wait',
        'ErrorSync2': '读取存在同步问题, pipe内缺少pipe barrier/pipe间缺少set/wait',
        'ErrorSync3': 'set/wait使用不配对, 缺少set或者wait',
        'ErrorSync4': '出现set/wait的eventID重复, 比如mte2: set0/set0, vector: wait0/wait0',
        'ErrorLeak': '内存泄露，存在申请内存未释放问题，详细见*_npuchk.log日志分析',
        'ErrorFree': '内存重复释放，调用free_buf释放过，再次调用free_buf',
        'ErrorBuffer1': 'tensor的que类型与初始化时不一致',
        'ErrorBuffer2': 'VECIN/VECOUT/VECCALC的操作不合规',
        'ErrorBuffer0': 'tensor内存未使用Ascendc框架的bufInit',
        'ErrorBuffer3': 'tensor的操作内存不合法, 可能原因: 内存未alloc/内存越界',
        'ErrorBuffer4': 'ButPool资源池未使用Ascendc框架的InitBufPool接口初始化'
    }
    stats = {}
    cpu_bin_path = None
    if len(sys.argv) == 2:
        all_npuchk_files = [sys.argv[1]]
    elif len(sys.argv) > 2:
        all_npuchk_files = glob.glob(sys.argv[1] + '/*_npuchk.log', recursive=True)
        cpu_bin_path = os.path.realpath(sys.argv[2])
    else:
        all_npuchk_files = glob.glob('**/*_npuchk.log', recursive=True)
    err_stack = {}
    for file_var in all_npuchk_files:
        parse_log(file_var, err_stack)
    for err_key in err_stack.keys():
        stack_info = []
        cur_err_info = err_stack.get(err_key)
        if len(cur_err_info) <= 0:
            continue
        err_type = get_error_type(cur_err_info[0])
        if stats.get(err_type) is None:
            stats[err_type] = 1
        else:
            stats[err_type] += 1
        stack_info.append(cur_err_info[0])
        stack_info.append('Rule: ' + str(err_details.get(err_type)))
        if len(cur_err_info) <= 1:
            continue
        stack_info.append(cur_err_info[1])
        if len(cur_err_info) <= 2:
            continue
        for frame in cur_err_info[2:]:
            info = frame.split(':')
            if len(info) < 2:
                stack_info.append('  ' + info[0])
                continue
            if cpu_bin_path:
                info[0] = os.path.join(cpu_bin_path, info[0])
            stack_info.append('  ' + addr_to_line(info[0], info[1]))
        stack_info.append('')
        LOG = '\n'.join(stack_info)
        if LOG.find('PostMessage') > 0:
            continue
        print(LOG)
    if stats:
        print('---------------------- ERROR STATISTICS ----------------------')
    for err_key in stats.keys():
        print('{}, {}, {}'.format(stats[err_key], err_key, err_details.get(err_key)))
