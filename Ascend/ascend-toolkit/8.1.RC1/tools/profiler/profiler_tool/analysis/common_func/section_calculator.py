#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from common_func.constant import Constant
from profiling_bean.db_dto.time_section_dto import CommunicationTimeSection
from profiling_bean.db_dto.time_section_dto import TimeSectionDto


class SectionCalculator:
    @staticmethod
    def _generate_time_section(start, end, class_section=TimeSectionDto):
        time_section = class_section()
        time_section.start_time, time_section.end_time = start, end
        return time_section

    @staticmethod
    def _has_section_overlapping(first_section, second_section):
        return first_section.start_time < second_section.start_time < first_section.end_time \
            or second_section.start_time < first_section.start_time < second_section.end_time

    @classmethod
    def merge_continuous_intervals(cls: any, time_section_list: list) -> list:
        result = []
        if not time_section_list:
            return result
        time_section_list = sorted(time_section_list, key=lambda x: x.start_time)
        current_section = time_section_list[0]
        for time_section in time_section_list:
            if time_section.start_time <= current_section.end_time:
                current_section = current_section.replace(
                    end_time=max(current_section.end_time, time_section.end_time))
            else:
                result.append(current_section)
                current_section = time_section
        result.append(current_section)
        return result

    @classmethod
    def compute_overlap_time(cls: any, master_time_section_list: list, slave_time_section_list: list) -> list:
        current_slava_key = Constant.DEFAULT_VALUE
        for i, master_time_section in enumerate(master_time_section_list):
            overlap_time = Constant.DEFAULT_VALUE
            while current_slava_key < len(slave_time_section_list):
                if slave_time_section_list[current_slava_key].end_time <= master_time_section.start_time:
                    current_slava_key += 1
                elif slave_time_section_list[current_slava_key].start_time >= master_time_section.end_time:
                    break
                elif slave_time_section_list[current_slava_key].end_time > master_time_section.end_time:
                    overlap_time = overlap_time + (master_time_section.end_time - max(
                        slave_time_section_list[current_slava_key].start_time, master_time_section.start_time))
                    break
                else:
                    overlap_time = overlap_time + (slave_time_section_list[current_slava_key].end_time - max(
                        slave_time_section_list[current_slava_key].start_time, master_time_section.start_time))
                    current_slava_key += 1
            master_time_section_list[i] = master_time_section.replace(overlap_time=overlap_time)
        return master_time_section_list

    @classmethod
    def compute_pipeline_overlap(cls, communication_section, compute_section):
        """
        calculate  pure communication and free time section.
        :param compute_section:|---------|    |-----XXX|
        :param communication_section:              |XXX----|
        pure communication section:                    |---|
        free time section:               |----|
        """
        free_time_section = []
        pure_communication_section = []
        time_section_list = sorted(communication_section + compute_section, key=lambda x: x.start_time)

        min_section = time_section_list.pop(0)
        for time_section in time_section_list:
            if min_section.end_time - time_section.start_time < 0:  # without overlapping
                free_time_section.append(cls._generate_time_section(min_section.end_time, time_section.start_time))
                if isinstance(min_section, CommunicationTimeSection):
                    pure_communication_section.append(
                        cls._generate_time_section(min_section.start_time, min_section.end_time))
                min_section = time_section
                continue
            if min_section.end_time - time_section.end_time < 0:  # with overlapping but no containment
                if isinstance(min_section, CommunicationTimeSection):
                    pure_communication_section.append(
                        cls._generate_time_section(min_section.start_time, time_section.start_time))
                    min_section = cls._generate_time_section(min_section.end_time, time_section.end_time)
                if isinstance(time_section, CommunicationTimeSection):
                    min_section = cls._generate_time_section(min_section.end_time, time_section.end_time,
                                                             class_section=CommunicationTimeSection)
            else:  # with containment
                if isinstance(min_section, CommunicationTimeSection):
                    pure_communication_section.append(
                        cls._generate_time_section(min_section.start_time, time_section.start_time))
                    min_section = cls._generate_time_section(time_section.end_time, min_section.end_time,
                                                             class_section=CommunicationTimeSection)
                if isinstance(time_section, CommunicationTimeSection):
                    min_section = cls._generate_time_section(time_section.end_time, min_section.end_time)
        if isinstance(min_section, CommunicationTimeSection):
            pure_communication_section.append(min_section)
        return pure_communication_section, free_time_section
