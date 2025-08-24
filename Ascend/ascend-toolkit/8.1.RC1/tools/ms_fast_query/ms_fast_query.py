#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.

import argparse
import json
import logging
import os
import sys

from model_info_util import ModelInfoUtil
from op_info_util import get_opp_result
from generic_utils import check_input_file, check_output_file


class GlobalVariables:
    LOG_FORMAT = '%(asctime)s %(filename)s [line:%(lineno)d] [%(levelname)s] %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def parse_command_line_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', choices=['op', 'model'], help='fast query type', required=True)
    parser.add_argument('--opp_path', help='opp path, required when type is op')
    parser.add_argument('-o', '--output', action='store', metavar='DIR', help='output file path', required=True)
    return parser.parse_args(argv)


def main(argv):
    logging.basicConfig(level=logging.INFO, format=GlobalVariables.LOG_FORMAT,
                        datefmt=GlobalVariables.DATE_FORMAT)
    args = parse_command_line_arguments(argv)
    check_output_file(args.output)

    result = {'result': 'fail'}
    if args.type == 'op':
        check_input_file(args.opp_path)
        get_opp_result(args.opp_path, result)
    else:
        util = ModelInfoUtil()
        result = util.get_from_model_zoo()

    output_path = os.path.realpath(args.output)
    with os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT, 0o640), 'w', encoding='utf8', newline='') as output:
        json.dump(result, output)


def main_without_exception(argv):
    try:
        main(argv)
    except KeyboardInterrupt:
        logging.error('User canceled.')
        sys.exit(1)
    except SystemExit as exp:
        sys.exit(exp.code)
    except BaseException as exp:
        logging.error(exp)
        sys.exit(1)


if __name__ == '__main__':
    main_without_exception(sys.argv[1:])
