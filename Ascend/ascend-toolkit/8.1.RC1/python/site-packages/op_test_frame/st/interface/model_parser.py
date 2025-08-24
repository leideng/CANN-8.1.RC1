#!/usr/bin/env python
# coding=utf-8
"""
Function:
AclOpGenerator class. This class mainly implements acl op src code generation.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
Change History: 2020-07-11 file Created
"""

import os
import importlib

from op_test_frame.st.interface import utils
from op_test_frame.st.interface.const_manager import ConstManager


def _get_framework_type(path):
    cur_dir = os.path.split(os.path.realpath(__file__))[0]
    config_path = os.path.join(cur_dir, ConstManager.FRAMEWORK_CONFIG_PATH)
    framework_dict = utils.load_json_file(config_path)
    suffix_list = []
    for (key, value) in list(framework_dict.items()):
        for item in value:
            suffix_list.append(item)
            if path.endswith(item):
                return key
    utils.print_error_log(
        'The model file "%s" is invalid, only supports %s file. '
        'Please modify it.' % (path, suffix_list))
    raise utils.OpTestGenException(
        ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)


def _function_call(args, op_type, func_name):
    framework = _get_framework_type(args.model_path)
    module_name = 'op_test_frame.st.interface.framework.%s_model_parser' % \
                  framework
    utils.print_info_log("Start to import %s." % module_name)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    try:
        return func(args, op_type)
    except Exception as ex:
        utils.print_error_log(
            'Failed to execute "%s". %s' % (func_name, str(ex)))
        raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR) from ex
    finally:
        pass


def get_model_nodes(args, op_type):
    """
    get model nodes by framework
    :param op_type: the op type
    :param args: the argument
    :return: the list of nodes.
    eg:
    node_list:
    [{"op_type": 'Add',
    "layer": 'fp32_vars/add',
    "input_dtype": ['float','float'],
    "input_shape": [[8,56,56,256],[8,56,56,256]],
    "output_dtype": ['float'],
    "output_shape": [[8,56,56,256]],
    "attr": [{'name :'T', type:'type', value:'AT_FLOAT'}]
    }]
    """
    return _function_call(args, op_type, ConstManager.GET_MODEL_NODES_FUNC)


def get_shape(args):
    """
    get shape by framework
    :param args: the argument
    :return: the shape list
    """
    return _function_call(args, '', ConstManager.GET_SHAPE_FUNC)


def change_shape(args):
    """
    change shape by framework
    :param args: the argument
    :return: the shape list
    """
    return _function_call(args, '', ConstManager.CHANGE_SHAPE_FUNC)
