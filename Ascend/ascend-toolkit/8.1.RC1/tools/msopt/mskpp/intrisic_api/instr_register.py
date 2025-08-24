#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

# 新增的指令仅需补充初始化函数__init__()和推导函数infer_shape();__init__用于使用输入的tensor以及attr传递至父类构造计算类指令
# infer_shape利用输入推导出经过调度的指令的输出;本文件定义了不同指令的初始化以及推导函数，其实现的策略定义在instr_strategy.py中

from ..core.api_register import InstrApiRegister
from .instr_strategy import InstrInitHandle
from .instr_strategy import MmadInit, VecUnaryInit, VecBinaryInit, VecBinaryInitV2, VecBinaryInitV2ByAttr, \
    VecUnaryInitByAttr, VecBinaryInitByAttr
from .infer_invoke import MmadInferInvoke, VecInferByTypeInvoke, TransInferInvoke, VecInferBySizeInvoke

instr_init_handle = InstrInitHandle(None)


@InstrApiRegister.register("MMAD")
class Mmad(MmadInferInvoke):
    # 指令原型：z = x * y + b
    # 在910b情况下，bias只能从bias table获取。bias table上的shape是[n]，在加的过程中会自动广播，
    # 且据芯片同事了解也没有gm-l1-l0c的通路，因此在指令这里只需要了解偏置项可以是[n]（真实意义上的偏置项）也可以是[m,n]（初始化的矩阵）即可。
    # 实测性能并不影响，当m是32倍数时，增大n一般可达到cube单元理论峰值。
    # infer_shape:
    # 理论上进入到mmad指令的tensor的数据排布格式都不会再是ND格式了，芯片内部数据格式不同位置的矩阵有差异，但在指令层面都叫做NZ或NC1HWC0。
    # 目前输入进来的tensor format还可以是nd的，后续期望用户在main函数里，910b的ND tensor用随路转换指令写法，910a需要加一个transdata。
    # 由于910b偏置项可以是ND排布，因此目前加个判断，如果b是ND排布，使用x的format
    def __init__(self, x, y, b, is_inited=False):
        # 该函数求值： x * y + b
        # :param is_inited:  标识b是否已经被初始化
        instr_init_handle.set_instr_strategy(MmadInit())
        instr_init_handle.instr_init(x, y, self, b=b, is_inited=is_inited)
        super(Mmad, self).__init__("MMAD", self.inputs, self.outputs, {}, self.instr_type)


@InstrApiRegister.register("VABS")
class Vabs(VecInferByTypeInvoke):
    # y = |x|    按元素取绝对值
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vabs, self).__init__("VABS", self.inputs, self.outputs, {})


@InstrApiRegister.register("VCMP")
class Vcmp(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vcmp, self).__init__("VCMP", self.inputs, self.outputs, {})


@InstrApiRegister.register("VCOPY")
class Vcopy(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vcopy, self).__init__("VCOPY", self.inputs, self.outputs, {})


@InstrApiRegister.register("VEXP")
class Vexp(VecInferByTypeInvoke):
    # y = exp(x)    x, y 按元素取指数
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vexp, self).__init__("VEXP", self.inputs, self.outputs, {})


@InstrApiRegister.register("VGATHER")
class Vgather(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vgather, self).__init__("VGATHER", self.inputs, self.outputs, {})


@InstrApiRegister.register("VGATHERB")
class Vgatherb(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vgatherb, self).__init__("VGATHERB", self.inputs, self.outputs, {})


@InstrApiRegister.register("VLN")
class Vln(VecInferByTypeInvoke):
    # y = ln(x)    x, y 按元素取对数
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vln, self).__init__("VLN", self.inputs, self.outputs, {})


@InstrApiRegister.register("VNOT")
class Vnot(VecInferByTypeInvoke):
    # y = vnot(x)    按元素取非
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vnot, self).__init__("VNOT", self.inputs, self.outputs, {})


@InstrApiRegister.register("VREC")
class Vrec(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vrec, self).__init__("VREC", self.inputs, self.outputs, {})


@InstrApiRegister.register("VRELU")
class Vrelu(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vrelu, self).__init__("VRELU", self.inputs, self.outputs, {})


@InstrApiRegister.register("VRSQRT")
class Vrsqrt(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vrsqrt, self).__init__("VRSQRT", self.inputs, self.outputs, {})


@InstrApiRegister.register("VSHL")
class Vshl(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vshl, self).__init__("VSHL", self.inputs, self.outputs, {})


@InstrApiRegister.register("VSHR")
class Vshr(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vshr, self).__init__("VSHR", self.inputs, self.outputs, {})


@InstrApiRegister.register("VSQRT")
class Vsqrt(VecInferByTypeInvoke):
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vsqrt, self).__init__("VSQRT", self.inputs, self.outputs, {})


@InstrApiRegister.register("VTRANSPOSE")
class Vtranspose(TransInferInvoke):
    # y = x^T    将输入数据x进行转置
    def __init__(self, x, y):
        instr_init_handle.set_instr_strategy(VecUnaryInit())
        instr_init_handle.instr_init(x, y, self)
        super(Vtranspose, self).__init__("VTRANSPOSE", self.inputs, self.outputs, {})


@InstrApiRegister.register("VADD")
class Vadd(VecInferByTypeInvoke):
    # z = x + y   x, y 按元素相加
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vadd, self).__init__("VADD", self.inputs, self.outputs, {})


@InstrApiRegister.register("VADDRELU")
class Vaddrelu(VecInferByTypeInvoke):
    # z = vaddrelu(x, y)    按元素取加法后做relu操作
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vaddrelu, self).__init__("VADDRELU", self.inputs, self.outputs, {})


@InstrApiRegister.register("VAND")
class Vand(VecInferByTypeInvoke):
    # z = vand(x, y)    x, y 按元素取与
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vand, self).__init__("VAND", self.inputs, self.outputs, {})


@InstrApiRegister.register("VCMPV")
class Vcmpv(VecInferByTypeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vcmpv, self).__init__("VCMPV", self.inputs, self.outputs, {})


@InstrApiRegister.register("VDIV")
class Vdiv(VecInferByTypeInvoke):
    # z = div(x, y)    x, y 按元素取除法
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vdiv, self).__init__("VDIV", self.inputs, self.outputs, {})


@InstrApiRegister.register("VMADD")
class Vmadd(VecInferByTypeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vmadd, self).__init__("VMADD", self.inputs, self.outputs, {})


@InstrApiRegister.register("VMADDRELU")
class VmaddRelu(VecInferByTypeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(VmaddRelu, self).__init__("VMADDRELU", self.inputs, self.outputs, {})


@InstrApiRegister.register("VMAX")
class Vmax(VecInferByTypeInvoke):
    # z = max(x, y)    x, y 按元素取最大
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vmax, self).__init__("VMAX", self.inputs, self.outputs, {})


@InstrApiRegister.register("VMIN")
class Vmin(VecInferByTypeInvoke):
    # z = min(x, y)    x, y 按元素取最小
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vmin, self).__init__("VMIN", self.inputs, self.outputs, {})


@InstrApiRegister.register("VMRGSORT")
class Vmrgsort(VecInferByTypeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vmrgsort, self).__init__("VMRGSORT", self.inputs, self.outputs, {})


@InstrApiRegister.register("VMUL")
class Vmul(VecInferByTypeInvoke):
    # z = x * y    x, y 按元素相乘
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vmul, self).__init__("VMUL", self.inputs, self.outputs, {})


@InstrApiRegister.register("VOR")
class Vor(VecInferByTypeInvoke):
    # z = vor(x, y)    x, y 按元素取或
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vor, self).__init__("VOR", self.inputs, self.outputs, {})


@InstrApiRegister.register("VSEL")
class Vsel(VecInferByTypeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vsel, self).__init__("VSEL", self.inputs, self.outputs, {})


@InstrApiRegister.register("VSUB")
class Vsub(VecInferByTypeInvoke):
    # z = x - y    x, y 按元素相减
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vsub, self).__init__("VSUB", self.inputs, self.outputs, {})


@InstrApiRegister.register("VSUBRELU")
class VsubRelu(VecInferByTypeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(VsubRelu, self).__init__("VSUBRELU", self.inputs, self.outputs, {})


@InstrApiRegister.register("VADDS")
class Vadds(VecInferByTypeInvoke):
    # z = vadds(x, y) vadds求值矢量x与标量y的乘积
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInitV2())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vadds, self).__init__("VADDS", self.inputs, self.outputs, {})


@InstrApiRegister.register("VCMPVS")
class Vcmpvs(VecInferByTypeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInitV2())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vcmpvs, self).__init__("VCMPVS", self.inputs, self.outputs, {})


@InstrApiRegister.register("VLRELU")
class Vlrelu(VecInferByTypeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInitV2())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vlrelu, self).__init__("VLRELU", self.inputs, self.outputs, {})


@InstrApiRegister.register("VMAXS")
class Vmaxs(VecInferByTypeInvoke):
    # z = vmaxs(x, y)    vmaxs求矢量x与标量y的最大
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInitV2())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vmaxs, self).__init__("VMAXS", self.inputs, self.outputs, {})


@InstrApiRegister.register("VMINS")
class Vmins(VecInferByTypeInvoke):
    # z = vmins(x, y)    vmaxs求矢量x与标量y的最小
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInitV2())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vmins, self).__init__("VMINS", self.inputs, self.outputs, {})


@InstrApiRegister.register("VMULS")
class Vmuls(VecInferByTypeInvoke):
    # z = vmuls(x, y)    vmuls求值矢量x与标量y的乘积
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInitV2())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vmuls, self).__init__("VMULS", self.inputs, self.outputs, {})


@InstrApiRegister.register("VADDRELUCONV")
class Vaddreluconv(VecInferByTypeInvoke):
    # z = vaddreluconv(x, y)    按元素取加法后做relu+量化操作
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vaddreluconv, self).__init__("VADDRELUCONV", self.inputs, self.outputs, {})


@InstrApiRegister.register("VAXPY")
class Vaxpy(VecInferByTypeInvoke):
    # z = vaxpy(x, y)    vaxpy求值矢量x与标量y的乘积与y的和
    def __init__(self, x, y, z, if_mix=False):
        instr_init_handle.set_instr_strategy(VecBinaryInitV2ByAttr())
        instr_init_handle.instr_init(x, y, self, z=z, instr_name="VAXPY", if_mix=if_mix)
        super(Vaxpy, self).__init__("VAXPY", self.inputs, self.outputs, {"if_mix": if_mix})


@InstrApiRegister.register("VBITSORT")
class Vbitsort(VecInferBySizeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(Vbitsort, self).__init__("VBITSORT", self.inputs, self.outputs, {})


@InstrApiRegister.register("VBRCB")
class Vbrcb(VecInferBySizeInvoke):
    # 根据指令的stride将tensor进行扩维，由于目前mskpp指令体系里并没有stride的概念，需要用户填写如何扩维倍数，
    # 并保持输入输出tensor的shape维度一致
    def __init__(self, x, y, broadcast_num):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VBRCB", broadcast_num=broadcast_num)
        super(Vbrcb, self).__init__("VBRCB", self.inputs, self.outputs, {"broadcast_num": broadcast_num})


@InstrApiRegister.register("VCADD")
class Vcadd(VecInferBySizeInvoke):
    # 根据指令的stride将tensor进行reduce，由于目前mskpp指令体系里并没有stride的概念，需要用户填写如何shape缩减倍数，
    # 并保持输入输出tensor的shape维度一致
    # 当shape最后一维reduce到1，则将该维度消除，以便后续用户能够继续将shape进行reduce
    # 目前需保证shape中最后一维能够被reduce_num整除且不为0
    def __init__(self, x, y, reduce_num):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCADD", reduce_num=reduce_num)
        super(Vcadd, self).__init__("VCADD", self.inputs, self.outputs, {"reduce_num": reduce_num})


@InstrApiRegister.register("VCGADD")
class Vcgadd(VecInferBySizeInvoke):
    def __init__(self, x, y, reduce_num):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCGADD", reduce_num=reduce_num)
        super(Vcgadd, self).__init__("VCGADD", self.inputs, self.outputs, {"reduce_num": reduce_num})


@InstrApiRegister.register("VCGMAX")
class Vcgmax(VecInferBySizeInvoke):
    def __init__(self, x, y, reduce_num):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCGMAX", reduce_num=reduce_num)
        super(Vcgmax, self).__init__("VCGMAX", self.inputs, self.outputs, {"reduce_num": reduce_num})


@InstrApiRegister.register("VCGMIN")
class Vcgmin(VecInferBySizeInvoke):
    def __init__(self, x, y, reduce_num):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCGMIN", reduce_num=reduce_num)
        super(Vcgmin, self).__init__("VCGMIN", self.inputs, self.outputs, {"reduce_num": reduce_num})


@InstrApiRegister.register("VCMAX")
class Vcmax(VecInferBySizeInvoke):
    def __init__(self, x, y, reduce_num):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCMAX", reduce_num=reduce_num)
        super(Vcmax, self).__init__("VCMAX", self.inputs, self.outputs, {"reduce_num": reduce_num})


@InstrApiRegister.register("VCMIN")
class Vcmin(VecInferBySizeInvoke):
    def __init__(self, x, y, reduce_num):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCMIN", reduce_num=reduce_num)
        super(Vcmin, self).__init__("VCMIN", self.inputs, self.outputs, {"reduce_num": reduce_num})


@InstrApiRegister.register("VCPADD")
class Vcpadd(VecInferBySizeInvoke):
    def __init__(self, x, y, reduce_num):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCPADD", reduce_num=reduce_num)
        super(Vcpadd, self).__init__("VCPADD", self.inputs, self.outputs, {"reduce_num": reduce_num})


@InstrApiRegister.register("VCONV")
class Vconv(VecInferByTypeInvoke):
    # y = vconv(x, dtype)    vconv表示对输入数据进行类型转换的向量计算
    def __init__(self, x, y, dtype):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCONV", dtype=dtype)
        super(Vconv, self).__init__("VCONV", self.inputs, self.outputs, {"dtype": dtype})


@InstrApiRegister.register("VECTORDUP")
class VectorDup(VecInferBySizeInvoke):
    # y = fill(x)    x, y 按元素取指数
    def __init__(self, x, y, fill_shape):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VECTORDUP", fill_shape=fill_shape)
        super(VectorDup, self).__init__("VECTORDUP", self.inputs, self.outputs, {"fill_shape": fill_shape})


@InstrApiRegister.register("VMLA")
class Vmla(VecInferByTypeInvoke):
    def __init__(self, x, y, z, if_mix=False):
        instr_init_handle.set_instr_strategy(VecBinaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, z=z, instr_name="VMLA", if_mix=if_mix)
        super(Vmla, self).__init__("VMLA", self.inputs, self.outputs, {"if_mix": if_mix})


@InstrApiRegister.register("VMULCONV")
class VmulConv(VecInferByTypeInvoke):
    def __init__(self, x, y, z, dtype):
        instr_init_handle.set_instr_strategy(VecBinaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, z=z, instr_name="VMULCONV", dtype=dtype)
        super(VmulConv, self).__init__("VMULCONV", self.inputs, self.outputs, {"dtype": dtype})


@InstrApiRegister.register("VCONVDEQ")
class Vconvdeq(VecInferByTypeInvoke):
    def __init__(self, x, y, dtype):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCONVDEQ")
        super(Vconvdeq, self).__init__("VCONVDEQ", (x,), (y,), {"dtype": dtype})


@InstrApiRegister.register("VCONVVDEQ")
class Vconvvdeq(VecInferByTypeInvoke):
    def __init__(self, x, y, dtype):
        instr_init_handle.set_instr_strategy(VecUnaryInitByAttr())
        instr_init_handle.instr_init(x, y, self, instr_name="VCONVVDEQ")
        super(Vconvvdeq, self).__init__("VCONVVDEQ", (x,), (y,), {"dtype": dtype})


@InstrApiRegister.register("VREDUCE")
class Vreduce(VecInferBySizeInvoke):
    def __init__(self, x, y, z, reserve_num):
        instr_init_handle.set_instr_strategy(VecBinaryInitV2ByAttr())
        instr_init_handle.instr_init(x, y, self, z=z, instr_name="VREDUCE", reserve_num=reserve_num)
        super(Vreduce, self).__init__("VREDUCE", self.inputs, self.outputs, {"reserve_num": reserve_num})


@InstrApiRegister.register("VREDUCEV2")
class Vreducev2(VecInferBySizeInvoke):
    def __init__(self, x, y, z, reserve_num):
        instr_init_handle.set_instr_strategy(VecBinaryInitV2ByAttr())
        instr_init_handle.instr_init(x, y, self, z=z, instr_name="VREDUCEV2", reserve_num=reserve_num)
        super(Vreducev2, self).__init__("VREDUCEV2", self.inputs, self.outputs, {"reserve_num": reserve_num})


@InstrApiRegister.register("VSUBRELUCONV")
class VsubReluConv(VecInferByTypeInvoke):
    def __init__(self, x, y, z):
        instr_init_handle.set_instr_strategy(VecBinaryInit())
        instr_init_handle.instr_init(x, y, self, z=z)
        super(VsubReluConv, self).__init__("VSUBRELUCONV", self.inputs, self.outputs, {})
