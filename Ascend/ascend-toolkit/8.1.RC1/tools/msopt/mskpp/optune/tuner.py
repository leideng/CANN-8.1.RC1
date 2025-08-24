#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from math import ceil
from multiprocessing import Manager, Pool
import os
from pathlib import Path
import threading
from typing import List, Dict

from mskpp.launcher.compiler import compile, CompiledKernel
from mskpp.launcher.config import KernelInvokeConfig
from mskpp.launcher.code_generator import Launcher

from mskpp.launcher.context import context as launch_context
from mskpp.launcher.driver import driver
from mskpp.optune.kernel_modifier import Replacer
from mskpp.optune.kernel_prof import Monitor
from mskpp.utils import logger, autotune_utils


class Autotuner:

    def __init__(self):
        self.context = None
        self.kernel_file = ''
        self.replacer = None

    @staticmethod
    def pre_launch(user_func, device_id, *args, **kwargs):
        # pre launch
        try:
            driver.set_device(device_id)
            logger.debug('starting kernel pre-launch...')
            user_func(*args, **kwargs)
            logger.debug('kernel pre-launch completed...')
        except Exception as exp:
            raise Exception(f'pre-launch operator failed! Error: {exp}.')

    @staticmethod
    def launch(kernel: CompiledKernel, context, device_id, warmup, repeat):
        kernel_monitor = Monitor()
        kernel_monitor.start(device_id)
        # get single launch time
        kernel[context.blockdim](*context.kernel_args, device_id=device_id)
        kernel_monitor.stop(device_id)
        one_time_duration = kernel_monitor.get_task_duration()
        if one_time_duration <= 0:
            raise ValueError(f'The running time of this operator is less than or equal to 0. The device {device_id} '
                             f'might be occupied by another process. Please use the command `npu-smi info` to '
                             f'check the device status.')
        if one_time_duration < warmup * 10 ** 3:
            warmup_times = ceil(warmup * 10 ** 3 / one_time_duration)
            # warm up
            for _ in range(warmup_times):
                kernel[context.blockdim](*context.kernel_args, device_id=device_id)
        # repeat
        kernel_monitor.start(device_id)
        for _ in range(repeat):
            kernel[context.blockdim](*context.kernel_args, device_id=device_id)
        kernel_monitor.stop(device_id)
        task_duration = kernel_monitor.get_task_duration()

        return task_duration / repeat

    @staticmethod
    def clean_files(context):
        if logger.log_level == '0':
            return
        autotun_files = [context.kernel_src_file, context.launch_src_file,
                         Path(context.launch_src_file).with_suffix('.so'),
                         Path(context.launch_src_file).with_suffix('.o')]
        for file in autotun_files:
            if os.path.isfile(file):
                os.remove(file)

    @staticmethod
    def _modify_file_path_with_index(file, index):
        path = Path(file)
        prefix = '' if path.stem.startswith('_gen_') else "_gen_"
        new_name = f'{prefix}{path.stem}_{index}{path.suffix}'
        return os.path.join(os.path.dirname(file), new_name)

    def gen_context(self, index, config):
        # modify kernel src file
        self.context.kernel_src_file = self._modify_file_path_with_index(self.context.kernel_src_file, index)
        self.replacer.replace_config(config, self.context.kernel_src_file)

        # modify code gen file path
        self.context.launch_src_file = self._modify_file_path_with_index(self.context.launch_src_file, index)

    def code_gen(self):
        config = KernelInvokeConfig(self.context.kernel_src_file, self.context.kernel_name)
        launcher = Launcher(config)
        launcher.code_gen(gen_file=self.context.launch_src_file)

    def compile_op(self, index):
        output_so_path = str(Path(self.context.launch_src_file).with_suffix('.so'))
        return compile(self.context.build_script, self.context.launch_src_file, output_so_path)


class Executor:
    def __init__(self, configs, device_ids, warmup, repeat, compile_processes=16):
        self._compile_processes = compile_processes
        multiprocessing_manager = Manager()
        self.task_queue = multiprocessing_manager.Queue()
        self.logging_queue = multiprocessing_manager.Queue()
        self._warmup = warmup
        self._repeat = repeat
        self._best_config = None
        self._best_index = None
        self._best_execution_time = 0
        self._auto_tuner = Autotuner()
        self._configs = configs
        self._device_id = device_ids[0]

    def execute(self):
        compile_pool = Pool(min(self._compile_processes, len(self._configs)))

        # start logging monitor
        log_thread = threading.Thread(target=self._log_listener)
        log_thread.start()

        # start compile task and launch task
        compile_tasks = [compile_pool.apply_async(self._compile_task, (index,)) for index in range(len(self._configs))]
        launch_thread = threading.Thread(target=self._launch_task)
        launch_thread.start()

        # wait for compile task finish
        for compile_task in compile_tasks:
            compile_task.get()
        compile_pool.close()
        compile_pool.join()

        self.task_queue.put((None, None, None))

        launch_thread.join()

        self.logging_queue.put((None, None))
        log_thread.join()
        if self._best_index is not None:
            print(f'Best config: No.{self._best_index}')
        logger.debug('kernel autotune end...')

    def _compile_task(self, index):
        self._auto_tuner.context = launch_context
        self._auto_tuner.replacer = Replacer(launch_context.kernel_src_file)
        config = self._configs[index]
        try:
            logger.debug(f'start to compile op for the {index}th config:{config}.')
            self._auto_tuner.gen_context(index, config)
            self._auto_tuner.code_gen()
            kernel = self._auto_tuner.compile_op(index)
            # ctypes.sharedctypes.RawArray object should only be shared between processes through inheritance
            self._auto_tuner.context.kernel_args = None
            self.task_queue.put((index, kernel, self._auto_tuner.context))
            logger.debug(f'Successfully compiled op for the {index}th config.')
        except Exception as exp:
            logger.error(f"compilation failed for the {index}th config {config}: {exp}")
            self.task_queue.put((index, None, None))
            self._auto_tuner.clean_files(self._auto_tuner.context)

    def _launch_task(self):
        while True:
            index, kernel, context = self.task_queue.get()
            if index is None:
                break
            if kernel is None:
                continue
            self._auto_tuner.context = context
            self._auto_tuner.context.kernel_args = launch_context.kernel_args
            logger.debug(f'start to launch op for the {index}th config on device {self._device_id}.')
            try:
                task_duration = self._auto_tuner.launch(kernel, context, self._device_id, self._warmup,
                                                        self._repeat)
                self.logging_queue.put((index, task_duration))
                logger.debug(f'Successfully launched for {index}th config.')
            except Exception as exp:
                logger.error(f'failed to launch for {index}th config {self._configs[index]}: {exp}')
                self.logging_queue.put((index, None))
            finally:
                self._auto_tuner.clean_files(context)

    def _log_listener(self):
        while True:
            index, task_duration = self.logging_queue.get()
            if index is None:
                break
            if task_duration is None:
                continue
            config = self._configs[index]
            print(f'No.{index}: {task_duration / 10 ** 3:.3f}μs, {config}')
            if self._best_config is None or task_duration < self._best_execution_time:
                self._best_config = config
                self._best_index = index
                self._best_execution_time = task_duration


def autotune(configs: List[Dict], warmup: int = 300, repeat: int = 1, device_ids=None):
    """Decorator for auto-tuning a kernel. Evaluate the configs and present the best one.

    Args:
        configs (List[Dict]): list of multiple key-value pairs.
        warmup (int, optional): Number of warmup iterations before measurement. Defaults to 300μs.
        repeat (int, optional): Number of repetitions for each configuration. Defaults to 1.
        device_ids (List[int], optional): Target device ID list for execution.
        Multi-device parallel execution is not yet supported.
        Only the first device id will be used currently. Defaults to [0].
    """

    if device_ids is None:
        device_ids = [0]

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                logger.debug('starting kernel autotune... ')
                autotune_utils.check_params(configs, warmup, repeat, device_ids)
                Autotuner().pre_launch(func, device_ids[0], *args, **kwargs)
                executor = Executor(configs, device_ids, warmup, repeat)
                executor.execute()
            except Exception as exp:
                logger.error(str(exp))

        return wrapper

    return decorator
