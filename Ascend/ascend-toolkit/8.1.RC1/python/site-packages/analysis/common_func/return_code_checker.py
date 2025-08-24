#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.

import json
import logging

from common_func.common import call_sys_exit
from common_func.common import print_msg
from common_func.ms_constant.number_constant import NumberConstant


class ReturnCodeCheck:
    """
    check interface return code
    """

    RETURN_CODE_KEY = "status"

    @classmethod
    def print_and_return_status(cls: any, json_dump: any) -> None:
        """
        print and return error code
        """
        print_msg(json_dump)
        cls.finish_with_error_code(json_dump)

    @classmethod
    def finish_with_error_code(cls: any, json_dump: any) -> None:
        """
        finish with the return code
        """
        try:
            call_sys_exit(json.loads(json_dump).get(cls.RETURN_CODE_KEY, NumberConstant.SUCCESS))
        except (ValueError, TypeError) as err:
            logging.error(err)
            call_sys_exit(NumberConstant.ERROR)
        finally:
            pass
