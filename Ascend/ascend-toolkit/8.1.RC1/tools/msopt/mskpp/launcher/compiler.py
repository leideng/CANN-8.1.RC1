#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import subprocess
import os
from typing import Generic, TypeVar
from .context import context
from .driver import NPULauncher
from ..utils import safe_check
from ..core.metric.file_system import FileChecker

T = TypeVar("T")


class KernelInterface(Generic[T]):
    """
    Kernel interface class, providing a way to launch function in the form: kernel[blockdim](x, y, z)
    """
    launch: T

    def __getitem__(self, blockdim) -> T:
        return lambda *args, **kwargs: self.launch(blockdim=blockdim, *args, **kwargs)


class CompiledKernel(KernelInterface[T]):
    """
    Object class representing a kernel, provides an interface to launching kernel.
    Support kernel invocation in form of "kernel[blockdim](a, b, c)".
    """
    def __init__(self, output_bin_path : str, kernel_name : str):
        self._check_constructor_input(output_bin_path, kernel_name)
        self.module_path = output_bin_path
        self.kernel_name = kernel_name
        self.__run__ = NPULauncher(self.module_path)


    def launch(self, *args, blockdim : int, **kwargs):
        """
        Launch the kernel on NPU device.

        Args:
            blockdim (int): Blocks allocated for the kernel.
            stream (int, optional): Stream that the kernel assigned to.
            device_id (int, optional): Device that launched the kernel.
            timeout (int, optional): Stream synchronization timeout in miliseconds.
            kernel_name (int, optional): Name of the to be launched kernel.
        """
        stream = kwargs.get('stream', None)
        device_id = kwargs.get('device_id', None)
        timeout = kwargs.get('timeout', 5000)
        kernel_name = kwargs.get('kernel_name', self.kernel_name)
        self._check_launch_input(blockdim, stream, device_id, timeout, kernel_name)

        self.__run__(blockdim=blockdim, l2ctrl=0, stream=stream, warmup=None, device_id=device_id,
            profiling=False, timeout=timeout, kernel_name=kernel_name, *args)


    def _check_constructor_input(self, output_bin_path : str, kernel_name : str):
        safe_check.check_variable_type(output_bin_path, str)
        safe_check.check_variable_type(kernel_name, str)
        checker = FileChecker(output_bin_path, "file")
        if not checker.check_input_file():
            raise Exception("Check the output_bin_path {} permission failed.".format(output_bin_path))


    def _check_launch_input(self, blockdim : int, stream : int, device_id : int, timeout : int, kernel_name : str):
        safe_check.check_variable_type(blockdim, int)
        if blockdim <= 0:
            raise Exception(f"Blockdim must be a positive integer but got {blockdim}")
        if stream is not None:
            safe_check.check_variable_type(stream, int)
        if device_id is not None:
            safe_check.check_variable_type(device_id, int)

        safe_check.check_variable_type(timeout, int)
        safe_check.check_variable_type(kernel_name, str)


def _check_compie_input(build_script : str,
                        launch_src_file : str,
                        output_bin_path : str,
                        use_cache : bool):
    safe_check.check_variable_type(build_script, str)
    checker = FileChecker(build_script, "file")
    if not checker.check_input_file():
        raise Exception("Check the build_script {} permission failed.".format(build_script))

    safe_check.check_variable_type(launch_src_file, str)
    checker = FileChecker(launch_src_file, "file")
    if not checker.check_input_file():
        raise Exception("Check the launch_src_file {} permission failed.".format(build_script))

    safe_check.check_variable_type(output_bin_path, str)
    safe_check.check_variable_type(use_cache, bool)


def compile(build_script : str,
            launch_src_file : str,
            output_bin_path : str = "_gen_module.so",
            use_cache : bool = False) -> CompiledKernel:
    """
    Compile a kernel and return a launchable kernel object.

    Args:
        build_script (str): The script for compiling the kernel that requires two input args
        launch_src_file (str): The launch source code file for kernel.
        output_bin_path (str, optional): Specify the output file generated from the build script.
                                         Defaults to "_gen_module.so".
        use_cache (bool, optional): Skip compiling and use the existed compiled module specified
                                    by "output_bin_path" instead.
                                    Defaults to False.

    Returns:
        CompiledKernel: A kernel object that can be launched.
    """
    _check_compie_input(build_script, launch_src_file, output_bin_path, use_cache)
    abs_launch_src_path = os.path.realpath(launch_src_file)
    abs_output_bin_path = os.path.realpath(output_bin_path)
    context.build_script = build_script
    context.launch_src_file = abs_launch_src_path

    if use_cache:
        if not os.path.exists(abs_output_bin_path):
            raise Exception("The excutable generated from build script does not exist.")
        return CompiledKernel(abs_output_bin_path, context.kernel_name)

    compile_cmd = ["bash", build_script, abs_launch_src_path, abs_output_bin_path]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Compile failed.\nCommand info: " + ' '.join(compile_cmd) + "\n{}".format(result.stderr))

    if result.stdout.strip() != "":
        print(result.stdout)

    return CompiledKernel(abs_output_bin_path, context.kernel_name)
