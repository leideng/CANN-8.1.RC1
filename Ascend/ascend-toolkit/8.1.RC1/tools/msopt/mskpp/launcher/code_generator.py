#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
import inspect
import ctypes
import numpy as np
from .config import KernelInvokeConfig
from .context import context
from ..utils import logger
from ..utils import safe_check
from ..core.metric.file_system import FileChecker


def is_builtin_basic_type_instance(obj):
    return isinstance(obj, int) or isinstance(obj, float)


def is_ctypes_class_instance(obj):
    return isinstance(obj, ctypes._SimpleCData)


def has_get_namespace(obj):
    return hasattr(obj, 'get_namespace') and callable(getattr(obj, 'get_namespace'))


def pytype_to_cpp(pytype):
    type_map = {
        # ctypes
        "c_char": "char",
        "c_byte": "char",
        "c_ubyte": "unsigned char",
        "c_short": "short",
        "c_ushort": "unsigned short",
        "c_int": "int",
        "c_uint": "unsigned int",
        "c_long": "long",
        "c_ulong": "unsigned long",
        "c_longlong": "long long",
        "c_ulonglong": "unsigned long long",
        "c_float": "float",
        "c_double": "double",
        "c_size_t": "size_t",
        "c_ssize_t": "ssize_t",
        "c_int8": "int8_t",
        "c_int16": "int16_t",
        "c_int32": "int32_t",
        "c_int64": "int64_t",
        "c_uint8": "uint8_t",
        "c_uint16": "uint16_t",
        "c_uint32": "uint32_t",
        "c_uint64": "uint64_t",

        # py built-in types
        "int": "int",
        "float": "float",
    }
    if pytype in type_map:
        return type_map[pytype]
    else:
        raise Exception("unsupported type")


def format_of(pytype):
    type_map = {
        # ctypes
        "c_char": "c",
        "c_byte": "b",
        "c_ubyte": "B",
        "c_short": "h",
        "c_ushort": "H",
        "c_int": "i",
        "c_uint": "I",
        "c_long": "l",
        "c_ulong": "k",
        "c_longlong": "L",
        "c_ulonglong": "K",
        "c_float": "f",
        "c_double": "d",
        "c_size_t": "n",
        "c_ssize_t": "N",
        "c_int8": "b",
        "c_int16": "h",
        "c_int32": "i",
        "c_int64": "L",
        "c_uint8": "B",
        "c_uint16": "H",
        "c_uint32": "I",
        "c_uint64": "K",

        # py built-in types
        "int": "i",
        "float": "f",
    }
    if pytype in type_map:
        return type_map[pytype]
    else:
        raise Exception("unsupported type")


class Launcher:

    def __init__(self, config: KernelInvokeConfig):
        """
        a class that generates launch source code for a kernel

        Args:
            config (KernelInvokeConfig): An configuration descriptor for a kernel
        """
        self.config = config
        self.kernel_name = None
        self.kernel_src_file = None
        self.kernel_args = None
        self.template_params = []
        if isinstance(config, KernelInvokeConfig):
            self.kernel_name = config.kernel_name
            self.kernel_src_file = config.kernel_src_file
            if context.kernel_args is None:
                self.kernel_args, self.template_params = self._parse_kernel_args_by_stack()
                context.kernel_args = self.kernel_args
                context.template_params = self.template_params
            else:
                self.kernel_args = context.kernel_args
                self.template_params = context.template_params
        else:
            raise Exception(f"unsupported config type: {type(config)}")

        # save params in context
        context.kernel_name = self.kernel_name
        context.kernel_src_file = self.kernel_src_file

    @staticmethod
    def _parse_kernel_args_by_stack():
        """
        Parse out the kernel args by tracing back the callstack.

        Returns:
            list: A list of arguments from user input.
        """
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back.f_back.f_back
        args, _, _, values = inspect.getargvalues(caller_frame)
        kernel_args = tuple(values[arg] for arg in args if arg != "template_param")
        if not kernel_args:
            raise Exception("kernel args not found")
        logger.debug(f"kernel_args: {kernel_args}")

        template_params = []
        if "template_param" in args:
            if not isinstance(values["template_param"], list):
                raise Exception("template_param shall be list")

            for arg in values["template_param"]:
                if isinstance(arg, str):
                    template_params.append(arg)
                else:
                    namespace = arg.get_namespace() if has_get_namespace(arg) else ''
                    template_params.append(namespace + arg.__name__)

        logger.debug(f"template_params: {template_params}")
        return kernel_args, template_params

    def code_gen(self, gen_file : str = "_gen_launch.cpp"):
        """
        Generate launch source code (glue code) for a kernel.
        Support the following launch mode: 1. kernel invocation <<<>>>

        Args:
            gen_file (str, optional): Specify the generated launch source code file path for kernel.
                                      Defaults to "__gen_launch.cpp".

        Returns:
            str: The file path of generated launch source file.

        Note:
        """
        def _check_input(file_path : str):
            safe_check.check_variable_type(file_path, str)
            checker = None
            if os.path.isfile(file_path):
                checker = FileChecker(file_path, "file")
            else:
                checker = FileChecker(os.path.dirname(file_path), "dir")
            if not checker.check_input_file():
                raise Exception("Check file_path {} permission failed.".format(file_path))

        _check_input(gen_file)
        context.launch_src_file = gen_file
        new_line = '\n    '
        args_decl = []
        args_deref = []
        arg_parse_list = []
        args_format = ['K', 'K', 'K'] # for blockdim, l2ctrl, stream

        for i, arg in enumerate(self.kernel_args):
            # template_param
            arg_parse_list.append(f"&arg{i}")
            if is_builtin_basic_type_instance(arg) or is_ctypes_class_instance(arg):
                # 基础类型，传递scalar值
                pytype = arg.__class__.__name__
                if isinstance(arg, int) and arg.bit_length() > 32:
                    # 认为大整数传递的是指针
                    pytype = "c_ulonglong"
                cpp_type = pytype_to_cpp(pytype)
                args_decl.append(f"{cpp_type} arg{i};")
                args_deref.append(f"arg{i}")
                args_format.append(format_of(pytype))
            elif isinstance(arg, ctypes.Structure):
                # 外部导入Structure数据类型，变量声明时考虑命名空间，形如{namespace}{type} *，传递cpu内存地址
                namespace = arg.get_namespace() if has_get_namespace(arg) else ''
                args_decl.append(namespace + arg.__class__.__name__ + f" *arg{i};")
                args_deref.append(f"*arg{i}")
                args_format.append(f"K")
            elif isinstance(arg, np.ndarray):
                # numpy数组，申请GM空间，传递指针地址
                args_decl.append(f"__gm__ uint8_t *arg{i};")
                args_deref.append(f"arg{i}")
                args_format.append(f"K")
            elif isinstance(arg, list):
                raise Exception(f"type \"list\" is unsupported yet. use ctypes array instead.")
            else:
                # 用户自定义类型，包括模板库数据类型数组，申请GM空间，传递指针地址
                args_decl.append(f"__gm__ uint8_t *arg{i};")
                args_deref.append(f"arg{i}")
                args_format.append(f"K")

        template_params = '<' + ', '.join(self.template_params) + '>' if len(self.template_params) > 0 else ''

        src = KERNEL_TEMPLATE.format(kernel_src_file=self.kernel_src_file,
                              kernel_name=self.kernel_name,
                              args_decl=new_line.join(e for e in args_decl if e is not None),
                              args_format=''.join(e for e in args_format if e is not None),
                              args_ref=', ' + ', '.join(arg_parse_list) if len(arg_parse_list) > 0 else '',
                              template_params=template_params,
                              args=', '.join(args_deref) if len(args_deref) > 0 else '')


        with os.fdopen(os.open(gen_file, safe_check.OPEN_FLAGS, safe_check.SAVE_DATA_FILE_AUTHORITY), 'w') as f:
            f.truncate()
            f.write(src)
            logger.debug(f"launch src file generated in {gen_file}")
        return gen_file

KERNEL_TEMPLATE = """
#include <iostream>
#include <Python.h>

#include "acl/acl.h"

// kernel src file
#include "{kernel_src_file}"

static PyObject* _launch_{kernel_name}(PyObject* self, PyObject* args) {{

    uint64_t blockdim;
    void *l2ctrl;
    void *stream;

    // args decl
    {args_decl}

    // args parse
    if (!PyArg_ParseTuple(args, "{args_format}", &blockdim, &l2ctrl, &stream{args_ref})) {{
        std::cout << "PyArg_ParseTuple failed" << std::endl;
        Py_RETURN_NONE;
    }}

    // launch here
    {kernel_name}{template_params}<<<blockdim, l2ctrl, stream>>>({args});

    Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"{kernel_name}", _launch_{kernel_name}, METH_VARARGS, "Entry point for kernel {kernel_name}"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  "_mskpp_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit__mskpp_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""