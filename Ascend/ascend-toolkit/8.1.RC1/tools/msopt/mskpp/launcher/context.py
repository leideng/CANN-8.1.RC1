#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

class Context:
    """
    Maintain context of modules(code_generator, compile, etc.)
    """

    def __init__(self):
        self._config = None
        self._kernel_name = None
        self._kernel_src_file = None
        self._kernel_args = None
        self._launch_src_file = None
        self._build_script = None
        self._blockdim = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @property
    def kernel_name(self):
        return self._kernel_name

    @kernel_name.setter
    def kernel_name(self, value):
        self._kernel_name = value

    @property
    def kernel_src_file(self):
        return self._kernel_src_file

    @kernel_src_file.setter
    def kernel_src_file(self, value):
        self._kernel_src_file = value

    @property
    def kernel_args(self):
        return self._kernel_args

    @kernel_args.setter
    def kernel_args(self, value):
        self._kernel_args = value

    @property
    def launch_src_file(self):
        return self._launch_src_file

    @launch_src_file.setter
    def launch_src_file(self, value):
        self._launch_src_file = value

    @property
    def build_script(self):
        return self._build_script

    @build_script.setter
    def build_script(self, value):
        self._build_script = value

    @property
    def blockdim(self):
        return self._blockdim

    @blockdim.setter
    def blockdim(self, value):
        self._blockdim = value

    @property
    def template_params(self):
        return self._template_params

    @template_params.setter
    def template_params(self, value):
        self._template_params = value


context = Context()