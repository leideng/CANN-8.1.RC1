# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

from typing import Any

from ms_service_profiler.utils.check.path_checker import PathChecker
from ms_service_profiler.utils.check.checker import Checker


class Rule:
    @staticmethod
    def none() -> Checker:
        return Checker().is_none()

    @staticmethod
    def path() -> PathChecker:
        return PathChecker()

    @staticmethod
    def config_file() -> PathChecker:
        return (
            PathChecker()
            .exists()
            .is_file()
            .is_readable()
            .is_not_writable_to_others()
            .is_safe_parent_dir()
            .max_size(10 * 1000 * 1000)
            .as_default()
        )

    @staticmethod
    def input_file() -> PathChecker:
        return (
            PathChecker()
            .exists()
            .forbidden_softlink()
            .is_file()
            .is_readable()
            .is_owner()
            .is_not_writable_to_others()
            .is_safe_parent_dir()
            .max_size(2 * 1000 * 1000 * 1000)
            .as_default()
        )

    @staticmethod
    def input_dir() -> PathChecker:
        return (
            PathChecker()
            .exists()
            .forbidden_softlink()
            .is_dir()
            .is_readable()
            .is_owner()
            .is_not_writable_to_others()
            .as_default()
        )

    @staticmethod
    def output_dir() -> PathChecker:
        return (
            Rule.path()
            .any(Rule.anti(PathChecker().exists()), PathChecker().is_dir().is_writeable().is_not_writable_to_others())
            .as_default()
        )

    @staticmethod
    def any(*rules: Checker) -> Checker:
        return Checker().any(*rules)

    @staticmethod
    def anti(rule: Checker) -> Checker:
        return Checker().anti(rule)

