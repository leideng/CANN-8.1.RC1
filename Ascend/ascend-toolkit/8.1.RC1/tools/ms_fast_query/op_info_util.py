#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.

import json
import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass

from generic_utils import check_input_file, MAX_HEADER_FILE_SIZE, MAX_JSON_FILE_SIZE, MAX_FILE_COUNT


class OppParseException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


@dataclass
class OpBaseInfo:
    hardware_type: str
    execution_unit: str
    attr: dict
    is_tf_kernel: bool


class OpInfoUtil:
    _ignored_op_pattern = ['/vector_core/cce', '/ai_core/cce', '/ai_core/tbe/kernel/']
    _not_published_op_pattern = '@par Restrictions'
    _op_name_pattern = re.compile(r"(/\*\*\s*\n(?<!\*/).*?\*/)\s*\n\s{,10}REG_OP\((\w+)\)", re.DOTALL)
    _hardware_pattern = re.compile(re.escape(os.path.sep).join(['ai_core', 'tbe', 'config', '(.*?)', '']))
    _horovod_hardware = ['ascend310p', 'ascend910']
    _hardware_name_mapper = {
        'ascend310': 'Ascend310',
        'ascend310p': 'Ascend310P',
        'ascend910': 'Ascend910',
        'ascend310b': 'Ascend310B'
    }

    def __init__(self, opp_path, fused_rule_path=None):
        self._opp_path = opp_path
        self.check_opp_path_validation()
        fused_rule_path = fused_rule_path or os.path.join(os.path.dirname(__file__), 'op_fused_rules.json')
        self.fused_rules = OpInfoUtil.load_json(fused_rule_path).get('fused_rules')

        self._published_op_names = self._get_published_op_names()
        self._op_base_info = self._parse_op_base_info()
        self._all_hardware_types = self._get_all_hardware_types()

        self._doc = self._generate_op_document()
        self._update_fused_ops(self._doc)
        self._update_empty_hardware(self._doc)
        self._hardware_to_op = self._group_op_by_hardware()

    @property
    def header_path(self):
        old_version_path = os.path.join(self._opp_path, 'op_proto/built-in/inc')
        if not os.path.exists(old_version_path):
            return os.path.join(self._opp_path, 'built-in/op_proto/inc')
        return old_version_path

    @property
    def op_impl_path(self):
        old_version_path = os.path.join(self._opp_path, 'op_impl/built-in')
        if not os.path.exists(old_version_path):
            return os.path.join(self._opp_path, 'built-in/op_impl')
        return old_version_path

    @staticmethod
    def load_json(path):
        check_input_file(str(path), MAX_JSON_FILE_SIZE)
        with open(path, 'r') as file:
            return json.load(file)

    @staticmethod
    def _update_by_special_rules(doc_op, op_name):
        # Collective communication
        if op_name.startswith('Hcom') or op_name.startswith('Horovod'):
            doc_op['hardwares'].update(OpInfoUtil._horovod_hardware)

    def get_supported_ops(self, interested_hardwares=None):
        interested_hardwares = interested_hardwares or (
            'ascend310', 'ascend910', 'ascend710', 'ascend310p', 'ascend310b')
        ops = []
        for hw_type in interested_hardwares:
            op_info = self._hardware_to_op.get(hw_type, {})
            if hw_type == 'ascend710':
                hw_type = 'ascend310p'
            for op_name, attr in op_info.items():
                hw_type = OpInfoUtil._hardware_name_mapper.get(hw_type, hw_type)
                ops.append({'op_type': op_name, 'hardware_type': hw_type, 'attr': attr})

        return ops

    def check_opp_path_validation(self):
        if not os.path.exists(self.header_path) or not os.path.exists(self.op_impl_path):
            raise OppParseException('Invalid opp path.')

    def _generate_op_document(self):
        doc = {}
        for op_name in self._published_op_names:
            doc_op = {'hardwares': set(), 'units': set(), 'attr': {}}
            if op_name in self._op_base_info:
                self._update_by_base_info(doc_op, op_name)
            # update op by enhanced op which used in contest
            if op_name + 'D' in self._op_base_info:
                self._update_by_base_info(doc_op, op_name + 'D')
            self._update_by_special_rules(doc_op, op_name)
            doc[op_name] = doc_op

        return doc

    def _get_published_op_names(self):
        published_op_names = []
        count = 0
        for file in Path(self.header_path).rglob('*.h'):
            count += 1
            if count > MAX_FILE_COUNT:
                raise ValueError(f'Number of files in {self.header_path} exceeds {MAX_FILE_COUNT}.')
            check_input_file(str(file), MAX_HEADER_FILE_SIZE)
            with open(file, 'r', errors='ignore') as fp:
                matches = OpInfoUtil._op_name_pattern.findall(fp.read())
            for match in matches:
                op_name = match[1]
                comment = match[0]
                if comment and self._not_published_op_pattern not in comment:
                    published_op_names.append(op_name)
        return published_op_names

    def _parse_op_base_info(self):
        def is_ignored_file(info_file_path):
            return any(pattern in str(info_file_path) for pattern in OpInfoUtil._ignored_op_pattern)

        def get_hardware_type(path):
            match = OpInfoUtil._hardware_pattern.search(path)
            return match[1] if match else ""

        def get_execution_unit(path):
            return path[:path.find(os.path.sep)]

        result = {}
        op_impl_path = self.op_impl_path
        count = 0
        for info_file in Path(op_impl_path).rglob('*.json'):
            if count > MAX_FILE_COUNT:
                raise ValueError(f'Number of files in {op_impl_path} exceeds {MAX_FILE_COUNT}.')
            if is_ignored_file(info_file):
                continue
            count += 1
            relative_path = os.path.relpath(str(info_file), op_impl_path)
            hardware = get_hardware_type(relative_path)
            execution_unit = get_execution_unit(relative_path)
            content = OpInfoUtil.load_json(info_file)
            for op_name in content:
                value = OpBaseInfo(hardware_type=hardware,
                                   execution_unit=execution_unit,
                                   attr=content[op_name],
                                   is_tf_kernel='tf_kernel' in relative_path)
                result.setdefault(op_name, []).append(value)
        return result

    def _get_all_hardware_types(self):
        types = set()
        for op_base_infos in self._op_base_info.values():
            for op_base_info in op_base_infos:
                hardware_type = op_base_info.hardware_type
                if hardware_type:
                    types.add(hardware_type)
        return list(types)

    def _update_fused_ops(self, doc):
        for op_name, mapping in self.fused_rules.items():
            if op_name not in doc:
                continue
            hardwares = set(self._all_hardware_types)
            for sub_op in mapping:
                if sub_op in doc:
                    hardwares = hardwares.intersection(doc[sub_op]['hardwares'])
                elif sub_op in self._op_base_info:
                    hardwares = hardwares.intersection(set(
                        attr.hardware_type
                        for attr in self._op_base_info.get(sub_op)
                        if attr.hardware_type
                    ))

            doc[op_name]['hardwares'] = hardwares

    def _update_aicpu_op_hardware(self, op_info, is_tf_kernel):
        if is_tf_kernel:
            op_info['hardwares'].update((
                hardware
                for hardware in self._all_hardware_types
                if not hardware.startswith('hi')
            ))
        else:
            op_info['hardwares'].update(self._all_hardware_types)

    def _update_by_base_info(self, doc_op, target):
        for base_info in self._op_base_info.get(target):
            if base_info.hardware_type:
                doc_op['hardwares'].add(base_info.hardware_type)
            if base_info.execution_unit == 'aicpu':
                self._update_aicpu_op_hardware(doc_op, base_info.is_tf_kernel)
            doc_op['units'].add(base_info.execution_unit)
            if not doc_op['attr']:
                doc_op['attr'] = base_info.attr

    def _update_empty_hardware(self, doc):
        for attr in doc.values():
            if not attr['hardwares']:
                attr['hardwares'].update(self._all_hardware_types)

    def _group_op_by_hardware(self):
        group_by_hardware = {hardware: {} for hardware in self._all_hardware_types}
        for op_name, value in self._doc.items():
            for hardware in value['hardwares']:
                group_by_hardware.setdefault(hardware, {}).update({op_name: {'attr': value['attr']}})
        return group_by_hardware


def _get_opp_result(opp_path, result):
    util = OpInfoUtil(opp_path)
    result['ops'] = util.get_supported_ops()
    result['result'] = 'success'


def get_opp_result(opp_path, result):
    try:
        _get_opp_result(opp_path, result)
    except OppParseException as ex:
        logging.error(ex.message)
