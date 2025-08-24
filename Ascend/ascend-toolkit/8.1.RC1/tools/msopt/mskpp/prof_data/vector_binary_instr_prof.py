#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from math import ceil
from mskpp.core.prof_data import PrefModel, ProfDataRegister
from mskpp._C import arch, prof_data
from mskpp.core.common import checker


class VectorBinaryInstr(PrefModel):
    """
    Template class
    """
    def __init__(self, name, inputs, outputs):
        super(VectorBinaryInstr, self).__init__(name, inputs, outputs)
        tile_size = 1
        for tile in inputs[0].size:
            tile_size *= tile
        self.tile_size = tile_size
        self.dtype = self.inputs[0].dtype
        self.instr_type = self.dtype + "_" + self.dtype + "_" + self.outputs[0].dtype
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


@ProfDataRegister.register("VADD")
class VaddPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VaddPref, self).__init__("VADD", inputs, outputs)
        self.data_obj = prof_data.VaddData()


@ProfDataRegister.register("VADDRELU")
class VaddreluPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VaddreluPref, self).__init__("VADDRELU", inputs, outputs)
        self.data_obj = prof_data.VaddreluData()


@ProfDataRegister.register("VADDRELUCONV")
class VaddreluconvPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VaddreluconvPref, self).__init__("VADDRELUCONV", inputs, outputs)
        self.data_obj = prof_data.VaddreluconvData()


@ProfDataRegister.register("VADDS")
class VaddsPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VaddsPref, self).__init__("VADDS", inputs, outputs)
        self.data_obj = prof_data.VaddsData()


@ProfDataRegister.register("VAND")
class VandPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VandPref, self).__init__("VAND", inputs, outputs)
        self.data_obj = prof_data.VandData()


@ProfDataRegister.register("VAXPY")
class VaxpyPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VaxpyPref, self).__init__("VAXPY", inputs, outputs)
        self.data_obj = prof_data.VaxpyData()


@ProfDataRegister.register("VBITSORT")
class VbitsortPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VbitsortPref, self).__init__("VBITSORT", inputs, outputs)

    def time(self):
        # 15.970为vbitsort实测固定值
        cycles = ceil(15.970)
        return cycles


@ProfDataRegister.register("VCMPV")
class VcmpvPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VcmpvPref, self).__init__("VCMPV", inputs, outputs)
        self.data_obj = prof_data.VcmpvData()


@ProfDataRegister.register("VCMPVS")
class VcmpvsPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VcmpvsPref, self).__init__("VCMPVS", inputs, outputs)
        self.data_obj = prof_data.VcmpvsData()


@ProfDataRegister.register("VDIV")
class VdivPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VdivPref, self).__init__("VDIV", inputs, outputs)
        self.data_obj = prof_data.VdivData()


@ProfDataRegister.register("VLRELU")
class VlreluPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VlreluPref, self).__init__("VLRELU", inputs, outputs)
        self.data_obj = prof_data.VlreluData()


@ProfDataRegister.register("VMADD")
class VmaddPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VmaddPref, self).__init__("VMADD", inputs, outputs)
        self.data_obj = prof_data.VmaddData()


@ProfDataRegister.register("VMAX")
class VmaxPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VmaxPref, self).__init__("VMAX", inputs, outputs)
        self.data_obj = prof_data.VmaxData()


@ProfDataRegister.register("VMAXS")
class VmaxsPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VmaxsPref, self).__init__("VMAXS", inputs, outputs)
        self.data_obj = prof_data.VmaxsData()


@ProfDataRegister.register("VMIN")
class VminPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VminPref, self).__init__("VMIN", inputs, outputs)
        self.data_obj = prof_data.VminData()


@ProfDataRegister.register("VMINS")
class VminsPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VminsPref, self).__init__("VMINS", inputs, outputs)
        self.data_obj = prof_data.VminsData()


@ProfDataRegister.register("VMRGSORT")
class VmrgsortPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VmrgsortPref, self).__init__("VMRGSORT", inputs, outputs)
        self.instr_type = inputs[0].dtype + "_" + inputs[1].dtype + "_" + self.outputs[0].dtype
        self.data_obj = prof_data.VmrgsortData()


@ProfDataRegister.register("VMUL")
class VmulPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VmulPref, self).__init__("VMUL", inputs, outputs)
        self.data_obj = prof_data.VmulData()


@ProfDataRegister.register("VMULS")
class VmulsPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VmulsPref, self).__init__("VMULS", inputs, outputs)
        self.data_obj = prof_data.VmulsData()


@ProfDataRegister.register("VSUB")
class VsubPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VsubPref, self).__init__("VSUB", inputs, outputs)
        self.data_obj = prof_data.VsubData()


@ProfDataRegister.register("VSEL")
class VselPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VselPref, self).__init__("VSEL", inputs, outputs)
        self.data_obj = prof_data.VselData()


@ProfDataRegister.register("VSUBRELUCONV")
class VsubReluConvPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VsubReluConvPref, self).__init__("VSUBRELUCONV", inputs, outputs)
        self.data_obj = prof_data.VsubReluConvData()


@ProfDataRegister.register("VSUBRELU")
class VsubReluPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VsubReluPref, self).__init__("VSUBRELU", inputs, outputs)
        self.data_obj = prof_data.VsubReluData()


@ProfDataRegister.register("VREDUCE")
class VreducePref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VreducePref, self).__init__("VREDUCE", inputs, outputs)
        self.data_obj = prof_data.VreduceData()


@ProfDataRegister.register("VREDUCEV2")
class Vreducev2Pref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(Vreducev2Pref, self).__init__("VREDUCEV2", inputs, outputs)
        self.data_obj = prof_data.Vreducev2Data()


@ProfDataRegister.register("VMLA")
class VmlaPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VmlaPref, self).__init__("VMLA", inputs, outputs)
        self.data_obj = prof_data.VmlaData()


@ProfDataRegister.register("VMADDRELU")
class VmaddReluPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VmaddReluPref, self).__init__("VMADDRELU", inputs, outputs)
        self.data_obj = prof_data.VmaddReluData()


@ProfDataRegister.register("VOR")
class VorPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VorPref, self).__init__("VOR", inputs, outputs)
        self.data_obj = prof_data.VorData()


@ProfDataRegister.register("VMULCONV")
class VmulconvPref(VectorBinaryInstr):
    def __init__(self, inputs, outputs):
        super(VmulconvPref, self).__init__("VMULCONV", inputs, outputs)
        self.data_obj = prof_data.VmulconvData()