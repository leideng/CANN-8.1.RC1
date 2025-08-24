#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
extract_image_patches support format NCHW
"""
from tbe.common.utils import decode
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import var
from impl.util.platform_adapter import add_compile_info
from impl.util.platform_adapter import op_tiling
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm


def _calc_output_size_and_pad(input_size, k, s, r, padding):
    k = (k - 1) * r + 1
    if padding == "VALID":
        output_size = (input_size - k + s) // s
        pad_before = pad_after = 0
    else:
        output_size = (input_size + s - 1) // s
        pad = tvm.max(0, (output_size - 1) * s + k - input_size)
        pad_before = pad // 2
        pad_after = pad - pad_before

    return output_size, pad_before, pad_after


class ExtractImagePatchesNCHW:
    def __init__(self, x, ksizes, strides, rates, padding, pads: list = None):
        self.padding = padding.upper()
        self.pads = pads

        dtype = x.get("dtype").lower()
        dtype_dict = {
            "int8": "int8", "uint8": "uint8",
            "float16": "float16", "fp16": "float16", "bfloat16": "int16",
            "float32": "float32", "fp32": "float32", "float": "float32"}
        self.dtype = dtype_dict.get(dtype)

        x_shape = x.get("shape")
        if -1 not in x_shape and -2 not in x_shape:
            self.is_const = True
            self.n, self.c, self.hi, self.wi = x_shape
        else:
            self.is_const = False
            self.n = var("n", (1, None), "int64")
            self.c = var("c", (1, None), "int64")
            self.hi = var("hi", (1, None), "int64")
            self.wi = var("wi", (1, None), "int64")

        self.is_binary = None in (ksizes, strides, rates, pads)
        if self.is_binary:
            self.kh = var("kh", (1, None), "int64")
            self.kw = var("kw", (1, None), "int64")
            self.sh = var("sh", (1, None), "int64")
            self.sw = var("sw", (1, None), "int64")
            self.rh = var("rh", (1, None), "int64")
            self.rw = var("rw", (1, None), "int64")
            self.ho = var("ho", (1, None), "int64")
            self.pt = var("pt", (1, None), "int64")
            self.pb = var("pb", (1, None), "int64")
            self.wo = var("wo", (1, None), "int64")
            self.pl = var("pl", (1, None), "int64")
            self.pr = var("pr", (1, None), "int64")

        else:
            _, _, self.kh, self.kw = ksizes
            _, _, self.sh, self.sw = strides
            _, _, self.rh, self.rw = rates
            self.pt, self.pb, self.pl, self.pr = pads

        if self.padding == "CALCULATED":
            if not self.is_binary:
                self.ho = (self.hi + self.pt + self.pb - (self.rh * (self.kh - 1) + 1)) / self.sh + 1
                self.wo = (self.wi + self.pl + self.pr - (self.rw * (self.kw - 1) + 1)) / self.sw + 1
        else:
            self.ho, self.pt, self.pb = _calc_output_size_and_pad(self.hi, self.kh, self.sh, self.rh, self.padding)
            self.wo, self.pl, self.pr = _calc_output_size_and_pad(self.wi, self.kw, self.sw, self.rw, self.padding)

        self.hi_p = self.hi + self.pt + self.pb
        self.wi_p = self.wi + self.pl + self.pr

        if self.is_const:
            run_info = self._get_run_info(x_shape, ksizes, strides, rates, self.padding, pads)
            self.tiling_key = run_info.get("tiling_key")

            tiling_format = {
                "n_factor": "int64",
                "c_factor": "int64",
                "kh_factor": "int64",
                "kw_factor": "int64",
                "ho_factor": "int64",
                "wo_factor": "int64",
                "ub_factor": "int64"
            }
            tiling_data = decode(run_info.get("tiling_data"), tiling_format)

            self.n_factor = tiling_data.get("n_factor")
            self.c_factor = tiling_data.get("c_factor")
            self.kh_factor = tiling_data.get("kh_factor")
            self.kw_factor = tiling_data.get("kw_factor")
            self.ho_factor = tiling_data.get("ho_factor")
            self.wo_factor = tiling_data.get("wo_factor")
            self.ub_factor = tiling_data.get("ub_factor")
        else:
            self.tiling_key = -1
            self.n_factor = var("n_factor", (1, None), "int64")
            self.c_factor = var("c_factor", (1, None), "int64")
            self.kh_factor = var("kh_factor", (1, None), "int64")
            self.kw_factor = var("kw_factor", (1, None), "int64")
            self.ho_factor = var("ho_factor", (1, None), "int64")
            self.wo_factor = var("wo_factor", (1, None), "int64")
            self.ub_factor = var("ub_factor", (1, None), "int64")

    def build(self, kernel_name):
        with tbe.compute():
            x, y = self._compute()
        with tvm.target.cce():
            self._add_compile_info()
            sch = tbe.auto_schedule(y)

        tbe.build(sch, {"tensor_list": [x, y], "name": kernel_name})

    def _padding(self, x_nhwc):
        x_data = tvm.compute(
            (self.n, self.hi_p, self.wi_p, self.c),
            lambda i0, i1, i2, i3:
                tvm.select(
                    i1 >= self.pt,
                    tvm.select(
                        i1 < self.hi + self.pt,
                        tvm.select(
                            i2 >= self.pl,
                            tvm.select(
                                i2 < self.wi + self.pl,
                                x_nhwc[i0, i1 - self.pt, i2 - self.pl, i3])))),
            name="x_data")

        top = tvm.compute(
            (self.n, self.hi_p, self.wi_p, self.c),
            lambda i0, i1, i2, i3:
                tvm.select(i1 < self.pt, tvm.const(0, self.dtype)),
            name="top")

        bottom = tvm.compute(
            (self.n, self.hi_p, self.wi_p, self.c),
            lambda i0, i1, i2, i3:
                tvm.select(i1 >= self.pt + self.hi, tvm.const(0, self.dtype)),
            name="bottom")

        left = tvm.compute(
            (self.n, self.hi_p, self.wi_p, self.c),
            lambda i0, i1, i2, i3:
                tvm.select(i2 < self.pl, tvm.const(0, self.dtype)),
            name="left")

        right = tvm.compute(
            (self.n, self.hi_p, self.wi_p, self.c),
            lambda i0, i1, i2, i3:
                tvm.select(i2 >= self.pl + self.wi, tvm.const(0, self.dtype)),
            name="right")

        padding_computes = (x_data, top, bottom, left, right)

        x_p = tvm.compute(
            (self.n, self.hi_p, self.wi_p, self.c),
            lambda i0, i1, i2, i3:
                tvm.select(
                    i1 < self.pt,
                    top[i0, i1, i2, i3],
                    tvm.select(
                        i1 >= self.pt + self.hi,
                        bottom[i0, i1, i2, i3],
                        tvm.select(
                            i2 < self.pl,
                            left[i0, i1, i2, i3],
                            tvm.select(
                                i2 >= self.pl + self.wi,
                                right[i0, i1, i2, i3],
                                x_data[i0, i1, i2, i3])))),
            name="x_p")

        return padding_computes, x_p

    def _get_run_info(self, x_shape, ksizes, strides, rates, padding, pads):
        input_list = [
            {"shape": x_shape, "ori_shape": x_shape, "format": "NCHW", "ori_format": "NCHW", "dtype": self.dtype}
        ]

        output_list = []

        attr_list = [
            {"name": "ksizes", "dtype": "list_int", "value": ksizes},
            {"name": "strides", "dtype": "list_int", "value": strides},
            {"name": "rates", "dtype": "list_int", "value": rates},
            {"name": "padding", "dtype": "str", "value": padding},
            {"name": "pads", "dtype": "list_int", "value": pads}
        ]

        compile_info = {
            "coreNum": tbe_platform.get_soc_spec("CORE_NUM"),
            "SIZE_UB": tbe_platform.get_soc_spec("UB_SIZE"),
            "nchw_format": True,
            "dtypeInput": self.dtype,
            "paddingType": self.padding,
            "isBinary": False,
            "isConst": True,
            "pads": self.pads
        }

        return op_tiling.do_op_tiling(get_context().get_op_type(),
                                      compile_info, input_list, output_list, None, None, attr_list)

    def _compute(self):
        x = tvm.placeholder((self.n, self.c, self.hi, self.wi), dtype=self.dtype, name="x")

        x_ub = tvm.compute((self.n, self.c, self.hi, self.wi), lambda *i: x[i], name="x_ub")

        x_nhwc = tvm.compute(
            (self.n, self.hi, self.wi, self.c),
            lambda i0, i1, i2, i3:
                x_ub[i0, i3, i1, i2],
            name="x_nhwc")

        if self.padding == "VALID":
            tmp_compute = x_nhwc
        else:
            padding_computes, x_p = self._padding(x_nhwc)
            tmp_compute = x_p

        y_nhwc = tvm.compute(
            (self.n, self.kh, self.kw, self.ho, self.wo, self.c),
            lambda i0, i1, i2, i3, i4, i5:
                tmp_compute[i0, i3 * self.sh + i1 * self.rh, i4 * self.sw + i2 * self.rw, i5] + \
                        tvm.const(0, self.dtype),
            name="y_nhwc")

        y_ub = tvm.compute(
            (self.n, self.c, self.kh, self.kw, self.ho, self.wo),
            lambda i0, i1, i2, i3, i4, i5:
                y_nhwc[i0, i2, i3, i4, i5, i1],
            name="y_ub")

        if self.padding == "VALID":
            computes = (x, x_ub, x_nhwc, y_nhwc, y_ub)
        else:
            computes = (x, x_ub, x_nhwc, padding_computes, x_p, y_nhwc, y_ub)

        y = tvm.compute(
            (self.n, self.c, self.kh, self.kw, self.ho, self.wo),
            lambda *i: y_ub[i],
            attrs={
                "params": (
                    self.n, self.c, self.hi, self.wi,
                    self.kh, self.kw, self.sh, self.sw, self.rh, self.rw, self.padding,
                    self.ho, self.wo,
                    self.n_factor, self.c_factor, self.kh_factor, self.kw_factor,
                    self.ho_factor, self.wo_factor, self.ub_factor
                ),
                "computes": computes,
                "tiling_key": self.tiling_key
            },
            name="y")

        return x, y

    def _add_compile_info(self):
        add_compile_info("coreNum", tbe_platform.get_soc_spec("CORE_NUM"))
        add_compile_info("SIZE_UB", tbe_platform.get_soc_spec("UB_SIZE"))
        add_compile_info("nchw_format", True)
        add_compile_info("dtypeInput", self.dtype)
        add_compile_info("paddingType", self.padding)
        add_compile_info("isBinary", self.is_binary)
        add_compile_info("isConst", self.is_const)
        add_compile_info("pads", self.pads)
