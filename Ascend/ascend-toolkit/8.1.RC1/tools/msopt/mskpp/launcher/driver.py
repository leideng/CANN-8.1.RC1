#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import importlib.util
import ctypes
import os
import numpy as np

from .context import context
from .code_generator import is_builtin_basic_type_instance, is_ctypes_class_instance


def load_mspti_so():
    cann_path = os.getenv('ASCEND_HOME_PATH')
    if cann_path and os.path.isdir(cann_path):
        lib = ctypes.CDLL(os.path.join(cann_path, "lib64/libmspti.so"), mode=ctypes.RTLD_GLOBAL)


load_mspti_so()


class NPULauncher(object):

    def __init__(self, module : str):
        self._module = module
        self._args_info = []
        self._kernel_meta = []
        self._host_to_gm_map = {}

    def __call__(self, *args,
                       blockdim : int,
                       l2ctrl : int,
                       stream : int,
                       warmup : int,
                       profiling : bool,
                       device_id : int,
                       timeout : int,
                       kernel_name : str
                ):
        if device_id is not None:
            driver.set_device(device_id)
        elif driver.get_active_device() is None:
            driver.set_device(0)

        context.blockdim = blockdim

        self._arg_preprocess(*args)

        if warmup is not None or profiling:
            # not implemented
            pass

        func_name = kernel_name

        spec = importlib.util.spec_from_file_location("_mskpp_launcher", self._module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, func_name):
            self._free_all_dev_ptr()
            raise Exception("Can't find function: {} in module: {}".format(func_name, self._module))

        new_stream_flag = False
        if stream is None:
            stream = driver.create_stream()
            new_stream_flag = True

        func = getattr(module, func_name)
        func(blockdim, l2ctrl, stream, *self._kernel_meta)
        ret = driver.synchronize_stream(stream, timeout)

        if new_stream_flag is True:
            driver.destroy_stream(stream)

        if ret != 0:
            self._free_all_dev_ptr()
            raise Exception("Synchronize stream failed, ret={}".format(ret))
        else:
            # args copy from gm back to host
            self._arg_postprocess()

    def _arg_preprocess(self, *args):

        def malloc_and_copy_to_device(addr, size):
            dev_ptr, ret = driver.malloc(size)
            if ret != 0:
                self._free_all_dev_ptr()
                raise Exception("Malloc failed. ret={} size={}".format(ret, size))
            # copy from host to device
            ret = driver.memcpy(dev_ptr, size, addr, size, 1)
            if ret != 0:
                self._free_all_dev_ptr()
                raise Exception("Memcpy failed. ret={}".format(ret))
            return dev_ptr

        args_info = []
        kernel_meta = []
        for arg in args:
            arg_type = type(arg)
            dev_ptr = None
            size = None
            addr = None
            if is_builtin_basic_type_instance(arg) or (is_ctypes_class_instance(arg)):
                # 传递scalar值
                kernel_meta.append(arg)
            elif isinstance(arg, ctypes.Structure):
                # 传递arg对象的host内存指针
                kernel_meta.append(ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p).value)
            elif isinstance(arg, np.ndarray):
                # 传递GM内存指针
                size = arg.nbytes
                addr = arg.ctypes.data
                if addr not in self._host_to_gm_map:
                    dev_ptr = malloc_and_copy_to_device(addr, size)
                    self._host_to_gm_map[addr] = dev_ptr
                kernel_meta.append(self._host_to_gm_map[addr])

            elif isinstance(arg, ctypes.Array):
                # 传递GM内存指针
                size = len(arg) * ctypes.sizeof(arg[0])
                addr = ctypes.addressof(arg)
                if addr not in self._host_to_gm_map:
                    dev_ptr = malloc_and_copy_to_device(addr, size)
                    self._host_to_gm_map[addr] = dev_ptr
                kernel_meta.append(self._host_to_gm_map[addr])
            else:
                self._free_all_dev_ptr()
                raise Exception("unsupported arg type {}".format(type(arg)))

            args_info.append({
                "type": arg_type,
                "size": size,
                "value": arg,
                "addr": addr,
                "dev_addr": dev_ptr
            })
        self._args_info = args_info
        self._kernel_meta = kernel_meta

    def _arg_postprocess(self):
        for arg_info in self._args_info:
            dev_ptr = arg_info["dev_addr"]
            size = arg_info["size"]
            addr = arg_info["addr"]
            if (dev_ptr is not None) and (addr in self._host_to_gm_map):
                # copy from device back to host
                ret = driver.memcpy(addr, size, dev_ptr, size, 2)
                if ret != 0:
                    self._free_all_dev_ptr()
                    raise Exception("Memcpy failed. ret={}".format(ret))
                driver.free(dev_ptr)
                arg_info["dev_addr"] = None
                self._host_to_gm_map.pop(addr)

    def _free_all_dev_ptr(self):
        for _, dev_ptr in self._host_to_gm_map.items():
            driver.free(dev_ptr)


class NPUDeviceContext:
    import acl

    def __init__(self):
        self.active_device = None
        self.acl.init()

    def __exit__(self):
        if self.active_device is not None:
            self.acl.rt.reset_device(self.active_device)
        self.acl.finalize()

    def set_device(self, devid : int):
        if not isinstance(devid, int) or devid < 0:
            raise Exception("Invalid devid, got".format(devid))
        self.acl.rt.set_device(devid)
        self.active_device = devid

    def get_active_device(self):
        return self.active_device

    def create_stream(self):
        stream, ret = self.acl.rt.create_stream()
        if ret != 0:
            raise Exception("Create stream failed. ret={}".format(ret))
        return stream

    def destroy_stream(self, stream : int):
        if stream is None:
            raise Exception("Stream is None")
        return self.acl.rt.destroy_stream_force(stream)

    def synchronize_stream(self, stream, timeout : int = -1):
        if stream is None:
            raise Exception("Stream is None")
        return self.acl.rt.synchronize_stream_with_timeout(stream, timeout)

    def malloc(self, size : int, policy : int = 0):
        '''
        MemMallocPolicy:
            ACL_MEM_MALLOC_HUGE_FIRST = 0
            ACL_MEM_MALLOC_HUGE_ONLY
            ACL_MEM_MALLOC_NORMAL_ONLY
            ACL_MEM_MALLOC_HUGE_FIRST_P2P
            ACL_MEM_MALLOC_HUGE_ONLY_P2P
            ACL_MEM_MALLOC_NORMAL_ONLY_P2P
        '''
        return self.acl.rt.malloc(size, policy)

    def free(self, dev_ptr : int):
        return self.acl.rt.free(dev_ptr)

    def memcpy(self, dst : int, dst_size : int, src : int, count : int, direction : int):
        '''
        memcpy mode:
            ACL_MEMCPY_HOST_TO_HOST: 0
            ACL_MEMCPY_HOST_TO_DEVICE: 1
            ACL_MEMCPY_DEVICE_TO_HOST: 2
            ACL_MEMCPY_DEVICE_TO_DEVICE: 3
        '''
        return self.acl.rt.memcpy(dst, dst_size, src, count, direction)


driver = NPUDeviceContext()
