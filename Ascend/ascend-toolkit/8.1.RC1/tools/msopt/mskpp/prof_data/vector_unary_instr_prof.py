#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from math import ceil
from mskpp.core.prof_data import PrefModel, ProfDataRegister
from mskpp._C import arch, prof_data
from mskpp.core.common import checker


class VectorUnaryInstr(PrefModel):
    """
    Template class
    """
    def __init__(self, name, inputs, outputs):
        super(VectorUnaryInstr, self).__init__(name, inputs, outputs)
        tile_size = 1
        for tile in inputs[0].size:
            tile_size *= tile
        self.tile_size = tile_size
        self.dtype = self.inputs[0].dtype
        self.instr_type = self.dtype + "_" + self.outputs[0].dtype
        self.data_obj = None

    def size(self):
        return self.tile_size

    def time(self):
        if self.data_obj is None:
            raise Exception("data object is not impl")
        if not checker.check_convert_long_size(self.tile_size):
            raise Exception("The shape size is too large")
        real_perf = self.data_obj.get(self.tile_size, self.instr_type)
        if real_perf <= 0:
            raise Exception("cannot get running time of {}".format(self.name))
        cycles = ceil(arch.get_size_of(self.dtype) * self.tile_size / real_perf)
        return cycles


@ProfDataRegister.register("VABS")
class VabsPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VabsPref, self).__init__("VABS", inputs, outputs)
        self.data_obj = prof_data.VabsData()


@ProfDataRegister.register("VBRCB")
class VbrcbPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VbrcbPref, self).__init__("VBRCB", inputs, outputs)
        self.data_obj = prof_data.VbrcbData()


@ProfDataRegister.register("VCADD")
class VcaddPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VcaddPref, self).__init__("VCADD", inputs, outputs)
        self.data_obj = prof_data.VcaddData()


@ProfDataRegister.register("VCGADD")
class VcgaddPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VcgaddPref, self).__init__("VCGADD", inputs, outputs)
        self.data_obj = prof_data.VcgaddData()


@ProfDataRegister.register("VCGMAX")
class VcgmaxPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VcgmaxPref, self).__init__("VCGMAX", inputs, outputs)
        self.data_obj = prof_data.VcgmaxData()


@ProfDataRegister.register("VCGMIN")
class VcgminPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VcgminPref, self).__init__("VCGMIN", inputs, outputs)
        self.data_obj = prof_data.VcgminData()


@ProfDataRegister.register("VCMAX")
class VcmaxPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VcmaxPref, self).__init__("VCMAX", inputs, outputs)
        self.data_obj = prof_data.VcmaxData()


@ProfDataRegister.register("VCMIN")
class VcminPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VcminPref, self).__init__("VCMIN", inputs, outputs)
        self.data_obj = prof_data.VcminData()


@ProfDataRegister.register("VCMP")
class VcmpPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VcmpPref, self).__init__("VCMP", inputs, outputs)
        self.data_obj = prof_data.VcmpData()


@ProfDataRegister.register("VCONV")
class VconvPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VconvPref, self).__init__("VCONV", inputs, outputs)
        self.data_obj = prof_data.VconvData()


@ProfDataRegister.register("VCONVDEQ")
class VconvdeqPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VconvdeqPref, self).__init__("VCONVDEQ", inputs, outputs)
        self.data_obj = prof_data.VconvdeqData()


@ProfDataRegister.register("VCONVVDEQ")
class VconvvdeqPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VconvvdeqPref, self).__init__("VCONVVDEQ", inputs, outputs)
        self.data_obj = prof_data.VconvvdeqData()


@ProfDataRegister.register("VCOPY")
class VcopyPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VcopyPref, self).__init__("VCOPY", inputs, outputs)
        self.data_obj = prof_data.VcopyData()


@ProfDataRegister.register("VCPADD")
class VcpaddPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VcpaddPref, self).__init__("VCPADD", inputs, outputs)
        self.data_obj = prof_data.VcpaddData()


@ProfDataRegister.register("VECTORDUP")
class VectorDupPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VectorDupPref, self).__init__("VECTORDUP", inputs, outputs)
        tile_size = 1
        y = outputs[0]
        for tile in y.size:
            tile_size *= tile
        self.tile_size = tile_size
        self.data_obj = prof_data.VectorDupData()


@ProfDataRegister.register("VEXP")
class VexpPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VexpPref, self).__init__("VEXP", inputs, outputs)
        self.data_obj = prof_data.VexpData()


@ProfDataRegister.register("VLN")
class VlnPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VlnPref, self).__init__("VLN", inputs, outputs)
        self.data_obj = prof_data.VlnData()


@ProfDataRegister.register("VNOT")
class VnotPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VnotPref, self).__init__("VNOT", inputs, outputs)
        self.data_obj = prof_data.VnotData()


@ProfDataRegister.register("VREC")
class VrecPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VrecPref, self).__init__("VREC", inputs, outputs)
        self.data_obj = prof_data.VrecData()


@ProfDataRegister.register("VTRANSPOSE")
class VtransposePref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VtransposePref, self).__init__("VTRANSPOSE", inputs, outputs)
        self.tile_size = 256  # transpose的该值固定

    def time(self):
        # 21.168为transpose实测固定值
        cycles = ceil(21.168)
        return cycles


@ProfDataRegister.register("VRELU")
class VreluPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VreluPref, self).__init__("VRELU", inputs, outputs)
        self.data_obj = prof_data.VreluData()


@ProfDataRegister.register("VSQRT")
class VsqrtPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VsqrtPref, self).__init__("VSQRT", inputs, outputs)
        self.data_obj = prof_data.VsqrtData()


@ProfDataRegister.register("VRSQRT")
class VrsqrtPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VrsqrtPref, self).__init__("VRSQRT", inputs, outputs)
        self.data_obj = prof_data.VrsqrtData()


@ProfDataRegister.register("VSHL")
class VshlPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VshlPref, self).__init__("VSHL", inputs, outputs)
        self.data_obj = prof_data.VshlData()


@ProfDataRegister.register("VSHR")
class VshrPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VshrPref, self).__init__("VSHR", inputs, outputs)
        self.data_obj = prof_data.VshrData()


@ProfDataRegister.register("VGATHER")
class VgatherPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VgatherPref, self).__init__("VGATHER", inputs, outputs)
        self.data_obj = prof_data.VgatherData()


@ProfDataRegister.register("VGATHERB")
class VgatherbPref(VectorUnaryInstr):
    def __init__(self, inputs, outputs):
        super(VgatherbPref, self).__init__("VGATHERB", inputs, outputs)
        self.data_obj = prof_data.VgatherbData()
