#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#----------------------------------------------------------------------------
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
#----------------------------------------------------------------------------

"""Struct parser."""

import re
from typing import Tuple


STRUCT_CONTENT = re.compile(r'\{(.+)\}', re.DOTALL)

class ParseError(Exception):
    """Parse error."""


def extract_type_name(line: str) -> str:
    """Extract struct member type name."""
    line = line.split(',')[0].strip()
    line = line.split('=')[0].strip()
    parts = line.split()
    # drop the last part to get type name.
    return ' '.join(parts[:-1])


def parse_struct_by_str(struct_str: str) -> Tuple[str, ...]:
    """Parse STRUCT by string."""
    search_obj = STRUCT_CONTENT.search(struct_str)
    if not search_obj:
        raise ParseError()

    struct_content = search_obj.group(1)
    struct_mems = struct_content.split(';')
    struct_mems = [mem.strip() for mem in struct_mems if mem.strip()]
    struct_types = tuple(extract_type_name(mem) for mem in struct_mems)
    return struct_types
