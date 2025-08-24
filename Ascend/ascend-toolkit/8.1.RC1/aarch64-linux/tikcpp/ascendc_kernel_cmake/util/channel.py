#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#----------------------------------------------------------------------------
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
#----------------------------------------------------------------------------

import os
import subprocess
from typing import Tuple

CODE_DEFAULT = 0
CODE_AIC = 1
CODE_AIV = 2
CODE_MAX = 3


def v220_mode(inst) -> int:
    if inst[6] == 'f':
        # MATRIX
        return CODE_AIC
    elif inst[6] == 'c':
        # FIXP
        return CODE_AIC
    elif inst[6] == '8':
        if inst[4] == '4' and inst[5] == '0' and inst[7] == '0':
            # MOVEMASK
            return 0
        # SIMD INST
        return CODE_AIV
    elif inst[6] == '9':
        # SIMD INST
        return CODE_AIV
    # DMA MOVE
    elif inst[6] == '6':
        if inst[7] == 'b' and (int(inst[4], 16) & 0x8) == 0x8:
            # MOV_{SRC}_TO_{DST}_ALIGN
            return CODE_AIV
        else:
            # MOV_CUB1
            return CODE_AIC
    # DMA MOVE
    elif inst[6] == '7':
        if inst[7] == '0' and (int(inst[4], 16) & 0x8) == 0x8:
            # MOV_UB_TO_XX
            return CODE_AIV
        elif (int(inst[0], 16) & 0x7) == 0 and (int(inst[1], 16) & 0x8) == 0x8:
            # MOV_XX_TO_UB
            return CODE_AIV
        else:
            # MOV_CUB2
            return CODE_AIC
    # SCALAR
    return 0


def get_code_channel(dst_file: str) -> Tuple[bool, int, str]:
    if not os.path.isfile(dst_file):
        return False, CODE_DEFAULT, f"file {dst_file} doesn't exist."

    objdump_cmd = ['llvm-objdump', '-s', '-j', '.text', dst_file]
    proc = subprocess.run(
        objdump_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    out = proc.stdout.decode()
    if proc.returncode != 0:
        return False, CODE_DEFAULT, f'llvm-objdump error, message is {out}'
    mode = 0
    lines = out.split('\n')
    for line in lines:
        insts = line.strip().split()
        if len(insts) < 5:
            continue
        for inst in insts[1:5]:
            if len(inst) != 8:
                continue
            mode |= v220_mode(inst)
    if mode >= CODE_MAX:
        return False, mode, f'unknown code mode {mode}.'
    return True, mode, ''
