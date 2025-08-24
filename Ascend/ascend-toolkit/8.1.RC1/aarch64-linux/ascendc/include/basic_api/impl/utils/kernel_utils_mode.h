/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file kernel_utils_mode.h
 * \brief
 */
#ifndef ASCENDC_MODULE_UTILS_MODE_H
#define ASCENDC_MODULE_UTILS_MODE_H
#include "utils/kernel_utils_constants.h"

namespace AscendC {
/*
input_format -> output_format; new_input_format -> new_output_format; new_input_shape -> new_output_shape;
NHWC         -> NC1HWC0         NHC   -> NCHT  [0,1*2,3]      -> [0,1,2*3,4]
ND           -> FRACTAL_NZ      HNC   -> HCNT  [0:-2,-2,-1]   -> [0:-4,-4,-3*-2,-1]
NDHWC        -> FRACTAL_Z_3D    NDHC  -> DCHNT [0,1,2*3,4]    -> [0,1,2*3,4*5,6]
NC1HWC0      -> FRACTAL_Z       NDHC  -> DCHNT [0,(1),1*2*3,4]-> [(1),(1),0*1*2,3*4,5]
NCDHW        -> NDC1HWC0        NCDH  -> NDCHT [0,1,2,3*4]    -> [0,1,2,3*4,5]
NCHW         -> NC1HWC0         NCH   -> NCHT  [0,1,2*3]      -> [0,1,2*3,4]
HWCN         -> FRACTAL_Z       HCN   -> CHNT  [0*1,2,3]      -> [0,1*2,3*4,5]
DHWCN        -> FRACTAL_Z_3D    DHCN  -> DCHNT [0,1*2,3,4]    -> [0,1,2*3,4*5,6]
ND           -> FRACTAL_Z       HCN   -> HCNT  [0:-2,-2,-1]   -> [0:-4,-4,-3*-2,-1]
NCHW         -> FRACTAL_Z       NCH   -> CHNT  [0,1,2*3]      -> [0,1*2,3*4,5]
NCDHW        -> FRACTAL_Z_3D    NCDH  -> DCHNT [0,1,2,3*4]    -> [0,1,2*3,4*5,6]
NC1HWC0      -> NHWC            NCHT  -> NHC   [0,1,2*3,4]    -> [0,1*2,3]
NDC1HWC0     -> NDHWC           NCHT  -> NHC   [0*1,2,3*4,5]   -> [0*1,2*3,4]
FRACTAL_Z_3D -> NDHWC           DCHNT -> NDHC  [0,1,2*3,4*5,6] -> [0,1,2*3,4]
FRACTAL_NZ   -> NC1HWC0         DCHNT -> NDHC  [(1),(1),0*1*2,3*4,5]-> [0,(1),1*2*3,4]
NDC1HWC0     -> NCDHW           NCHT  -> NCDH  [0,1,2,3*4,5]      -> [0,1,2,3*4]
NC1HWC0      -> NCHW            NCHT  -> NCH   [0,1,2*3,4]        -> [0,1,2*3]
FRACTAL_Z    -> HWCN            CHNT  -> HCN   [0,1*2,3*4,5]      -> [0*1,2,3]
FRACTAL_Z_3D -> DHWCN           DCHNT -> DHCN  [0,1,2*3,4*5,6]    -> [0,1*2,3,4]
FRACTAL_Z    -> NCHW            CHNT  -> NCH   [0,1*2,3*4,5]      -> [0,1,2*3]
FRACTAL_Z_3D -> NCDHW           DCHNT -> NCDH  [0,1,2*3,4*5,6]    -> [0,1,2,3*4]
FRACTAL_Z    -> ND              HCNT  -> HCN   [0:-4,-4,-3*-2,-1] -> [0:-2,-2,-1]
*/

class MaskSetter {
public:
    static MaskSetter& Instance()
    {
        static MaskSetter instance;
        return instance;
    };

    void SetMask(bool setMask)
    {
        isSetMask = setMask;
    }

    bool GetMask() const
    {
        return isSetMask;
    }

private:
    MaskSetter(){};
    ~MaskSetter(){};
    bool isSetMask = true;
};

class Int4Setter {
public:
    static Int4Setter& Instance()
    {
        static Int4Setter instance;
        return instance;
    };

    void SetInt4()
    {
        isInt4 = true;
    }

    void SetDstInt4()
    {
        isDstInt4 = true;
    }

    void SetSrcInt4()
    {
        isSrcInt4 = true;
    }

    void ResetInt4()
    {
        isInt4 = false;
    }

    void ResetDstSrcInt4()
    {
        isDstInt4 = false;
        isSrcInt4 = false;
    }

    bool GetInt4() const
    {
        return isInt4;
    }

    bool GetDstInt4() const
    {
        return isDstInt4;
    }

    bool GetSrcInt4() const
    {
        return isSrcInt4;
    }

private:
    Int4Setter(){};
    ~Int4Setter(){};

    bool isInt4 = false;
    bool isDstInt4 = false;
    bool isSrcInt4 = false;
};

union NotNumUnion {
    __aicore__ NotNumUnion() {}
    float f;
    uint32_t i;
};

enum class TShapeType : uint8_t {
    DEFAULT,
    NHWC,
    NC1HWC0,
    NHC,
    NCHT,
    ND,
    FRACTAL_NZ,
    HNC,
    HCNT,
    NDHWC,
    FRACTAL_Z_3D,
    NDHC,
    DCHNT,
    FRACTAL_Z,
    NCDHW,
    NDC1HWC0,
    NCDH,
    NDCHT,
    NCHW,
    NCH,
    HWCN,
    HCN,
    CHNT,
    DHWCN,
    DHCN
};

enum class RoundMode : uint8_t {
    CAST_NONE = 0,
    CAST_RINT, // round
    CAST_FLOOR,
    CAST_CEIL,
    CAST_ROUND, // away-zero
    CAST_TRUNC, // to-zero
    CAST_ODD,   // Von Neumann rounding
};

enum class CMPMODE : uint8_t {
    LT = 0,
    GT,
    EQ,
    LE,
    GE,
    NE,
};

enum class SELMODE : uint8_t {
    VSEL_CMPMASK_SPR = 0,
    VSEL_TENSOR_SCALAR_MODE,
    VSEL_TENSOR_TENSOR_MODE,
};

enum class BlockMode : uint8_t {
    BLOCK_MODE_NORMAL = 0,
    BLOCK_MODE_MATRIX,
    BLOCK_MODE_VECTOR,
    BLOCK_MODE_SMALL_CHANNEL,
    BLOCK_MODE_DEPTHWISE,
};

enum class DeqScale : uint8_t {
    DEQ_NONE = 0,
    DEQ,
    VDEQ,
    DEQ8,
    VDEQ8,
    DEQ16,
    VDEQ16,
};

enum class ReduceMode : uint8_t {
    REDUCE_MAX = 0,
    REDUCE_MIN,
    REDUCE_SUM,
};

enum class ReduceOrder : uint8_t {
    ORDER_VALUE_INDEX = 0,
    ORDER_INDEX_VALUE,
    ORDER_ONLY_VALUE,
    ORDER_ONLY_INDEX,
};

enum class DumpType : uint8_t {
    DUMP_DEFAULT = 0,
    DUMP_SCALAR,
    DUMP_TENSOR,
    DUMP_SHAPE,
    DUMP_ASSERT,
    DUMP_META,
    DUMP_TIME_STAMP,
};

enum class CLAMPMODE {
    CLAMP_MAX = 0,
    CLAMP_MIN,
};

enum class PcieCtrl : uint64_t {
    WR = 0,
    RD
};

enum class DeQuantMode : uint8_t {
    DEQUANT_WITH_SINGLE_ROW = 0,    // {1, m * n, n}  = {m, n, n}
    DEQUANT_WITH_MULTI_ROW,         // {1, m * n, n} != {m, n, n}
};

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
class ConstDefiner {
public:
    static ConstDefiner& Instance()
    {
        static ConstDefiner instance;
        return instance;
    };

    uint8_t* GetHardwareBaseAddr(Hardware hardPos)
    {
        ASCENDC_ASSERT((hardwareCpuBufferMap.find(hardPos) != hardwareCpuBufferMap.end()),
                       { KERNEL_LOG(KERNEL_ERROR, "illegal hardPos %d", static_cast<int>(hardPos)); });
        return hardwareCpuBufferMap[hardPos];
    }

    const std::map<uint8_t, std::string> logicNameMap{
        { static_cast<uint8_t>(TPosition::GM), "GM" },
        { static_cast<uint8_t>(TPosition::A1), "A1" },
        { static_cast<uint8_t>(TPosition::A2), "A2" },
        { static_cast<uint8_t>(TPosition::B1), "B1" },
        { static_cast<uint8_t>(TPosition::B2), "B2" },
        { static_cast<uint8_t>(TPosition::C1), "C1" },
        { static_cast<uint8_t>(TPosition::C2), "C2" },
        { static_cast<uint8_t>(TPosition::CO1), "CO1" },
        { static_cast<uint8_t>(TPosition::CO2), "CO2" },
        { static_cast<uint8_t>(TPosition::VECIN), "VECIN" },
        { static_cast<uint8_t>(TPosition::VECOUT), "VECOUT" },
        { static_cast<uint8_t>(TPosition::VECCALC), "VECCALC" },
        { static_cast<uint8_t>(TPosition::LCM), "LCM" },
        { static_cast<uint8_t>(TPosition::SPM), "SPM" },
        { static_cast<uint8_t>(TPosition::SHM), "SHM" },
        { static_cast<uint8_t>(TPosition::TSCM), "TSCM" },
        { static_cast<uint8_t>(TPosition::C2PIPE2GM), "C2PIPE2GM" },
        { static_cast<uint8_t>(TPosition::C2PIPE2LOCAL), "C2PIPE2LOCAL" },
        { static_cast<uint8_t>(TPosition::MAX), "MAX" },
    };

    const std::set<std::vector<Hardware>> quantDataCopy = {
        { Hardware::UB, Hardware::L0C }, { Hardware::L0C, Hardware::UB }
    };

    const std::set<std::vector<Hardware>> normalDataCopy = {
        { Hardware::L1, Hardware::UB }, { Hardware::GM, Hardware::L1 }, { Hardware::GM, Hardware::UB },
        { Hardware::UB, Hardware::L1 }, { Hardware::UB, Hardware::GM }, { Hardware::UB, Hardware::UB },
        { Hardware::L1, Hardware::GM }
    };

    const std::set<std::vector<Hardware>> biasDataCopy = { { Hardware::L1, Hardware::BIAS } };

    const std::set<std::vector<Hardware>> matDataCopy = { { Hardware::L1, Hardware::L0C } };

    const std::map<BlockMode, std::string> blockModeMap = {
        { BlockMode::BLOCK_MODE_NORMAL, "" },      { BlockMode::BLOCK_MODE_MATRIX, "" },
        { BlockMode::BLOCK_MODE_VECTOR, "V" },     { BlockMode::BLOCK_MODE_SMALL_CHANNEL, "SC" },
        { BlockMode::BLOCK_MODE_DEPTHWISE, "DP" },
    };

    const std::map<Hardware, std::string> hardwareMap = {
        { Hardware::GM, "OUT" }, { Hardware::L1, "L1" },   { Hardware::L0A, "L0A" },
        { Hardware::L0B, "L0B" }, { Hardware::L0C, "L0C" }, { Hardware::UB, "UB" },
#if __CCE_AICORE__ >= 220
        { Hardware::BIAS, "BT" }, { Hardware::FIXBUF, "FB" },
#endif
    };

    const std::map<std::string, uint16_t> dstBurstLenUnitMap = {
        { "L0C16UB", 512 },    { "L0C32UB", 1024 },    { "UBL0C16", 512 },   { "UBL0C32", 1024 },
        { "L1L0C16", 512 },    { "L1L0C32", 1024 },    { "L0CV16UB", 32 },   { "L0CV32UB", 64 },
        { "UBL0CV16", 512 },   { "UBL0CV32", 1024 },   { "L0CSC32UB", 256 }, { "UBL0CSC32", 256 },
        { "L0CDPf16UB", 512 }, { "L0CDPf32UB", 1024 }, { "L1BT", 64 },      { "L1FB", 128 },
    };

    const std::map<std::string, uint16_t> srcBurstLenUnitMap = {
        { "L0C16UB", 512 },   { "L0C32UB", 1024 },  { "UBL0C16", 512 },    { "UBL0C32", 1024 },    { "L1L0C16", 512 },
        { "L1L0C32", 1024 },  { "L0CV16UB", 512 },  { "L0CV32UB", 1024 },  { "UBL0CV16", 32 },     { "UBL0CV32", 64 },
        { "L0CSC32UB", 256 }, { "UBL0CSC32", 256 }, { "L0CDPf16UB", 512 }, { "L0CDPf32UB", 1024 }, { "L1BT", 64 },
        { "L1FB", 128 },
    };

    const std::map<std::string, uint16_t> dstStrideUnitMap = {
        { "UBL0C16", 512 },  { "UBL0C32", 1024 },  { "L1L0C16", 512 },   { "L1L0C32", 1024 },
        { "UBL0CV16", 512 }, { "UBL0CV32", 1024 }, { "UBL0CSC32", 256 }, { "L1BT", 64 },
        { "L1FB", 128 },
    };

    const std::map<std::string, uint16_t> srcStrideUnitMap = {
        { "L1L0C16", 32 },    { "L1L0C32", 32 },    { "L0C16UB", 512 },    { "L0C32UB", 1024 },    { "L0CV16UB", 512 },
        { "L0CV32UB", 1024 }, { "L0CSC32UB", 256 }, { "L0CDPf16UB", 512 }, { "L0CDPf32UB", 1024 }, { "L1BT", 32 },
        { "L1FB", 32 },
    };

#if __CCE_AICORE__ <= 200
    const std::map<TPosition, Hardware> positionHardMap = {
        { TPosition::GM, Hardware::GM },      { TPosition::A1, Hardware::L1 },    { TPosition::B1, Hardware::L1 },
        { TPosition::TSCM, Hardware::L1 },    { TPosition::VECIN, Hardware::UB }, { TPosition::VECOUT, Hardware::UB },
        { TPosition::VECCALC, Hardware::UB }, { TPosition::A2, Hardware::L0A },   { TPosition::B2, Hardware::L0B },
        { TPosition::C1, Hardware::L1 },      { TPosition::C2, Hardware::BIAS },  { TPosition::CO1, Hardware::L0C },
        { TPosition::CO2, Hardware::UB },
    };
#elif __CCE_AICORE__ == 220
    const std::map<TPosition, Hardware> positionHardMap = {
        { TPosition::GM, Hardware::GM },      { TPosition::A1, Hardware::L1 },    { TPosition::B1, Hardware::L1 },
        { TPosition::TSCM, Hardware::L1 },    { TPosition::VECIN, Hardware::UB }, { TPosition::VECOUT, Hardware::UB },
        { TPosition::VECCALC, Hardware::UB }, { TPosition::A2, Hardware::L0A },   { TPosition::B2, Hardware::L0B },
        { TPosition::C1, Hardware::L1 },      { TPosition::C2, Hardware::BIAS },  { TPosition::CO1, Hardware::L0C },
        { TPosition::CO2, Hardware::GM }, { TPosition::C2PIPE2GM, Hardware::FIXBUF },
    };
#elif __CCE_AICORE__ == 300
    const std::map<TPosition, Hardware> positionHardMap = {
        { TPosition::GM, Hardware::GM },      { TPosition::A1, Hardware::L1 },    { TPosition::B1, Hardware::L1 },
        { TPosition::TSCM, Hardware::L1 },    { TPosition::VECIN, Hardware::UB }, { TPosition::VECOUT, Hardware::UB },
        { TPosition::VECCALC, Hardware::UB }, { TPosition::A2, Hardware::L0A },   { TPosition::B2, Hardware::L0B },
        { TPosition::C1, Hardware::L1 },      { TPosition::C2, Hardware::BIAS },  { TPosition::CO1, Hardware::L0C },
        { TPosition::CO2, Hardware::GM }, { TPosition::C2PIPE2GM, Hardware::FIXBUF },
    };
#elif defined(__DAV_M310__)
    const std::map<TPosition, Hardware> positionHardMap = {
        { TPosition::GM, Hardware::GM },      { TPosition::A1, Hardware::L1 },    { TPosition::B1, Hardware::L1 },
        { TPosition::TSCM, Hardware::L1 },    { TPosition::VECIN, Hardware::UB }, { TPosition::VECOUT, Hardware::UB },
        { TPosition::VECCALC, Hardware::UB }, { TPosition::A2, Hardware::L0A },   { TPosition::B2, Hardware::L0B },
        { TPosition::C1, Hardware::L1 },      { TPosition::C2, Hardware::BIAS },  { TPosition::CO1, Hardware::L0C },
        { TPosition::CO2, Hardware::GM },
    };
#endif
#if __CCE_AICORE__ <= 200
    const std::map<Hardware, uint32_t> bufferInitLen = {
        { Hardware::GM, 1024 * 1024 }, { Hardware::UB, 1024 * 256 },    { Hardware::L1, 1024 * 1024 },
        { Hardware::L0A, 1024 * 64 },  { Hardware::L0B, 1024 * 64 },    { Hardware::L0C, 1024 * 256 },
        { Hardware::BIAS, 1024 * 64 }, { Hardware::FIXBUF, 1024 * 64 },
    };
#elif (__CCE_AICORE__ == 220)
    const std::map<Hardware, uint32_t> bufferInitLen = {
        { Hardware::GM, 1024 * 1024 }, { Hardware::UB, 1024 * 192 },   { Hardware::L1, 1024 * 512 },
        { Hardware::L0A, 1024 * 64 },  { Hardware::L0B, 1024 * 64 },   { Hardware::L0C, 1024 * 128 },
        { Hardware::BIAS, 1024 * 1 },  { Hardware::FIXBUF, 1024 * 7 },
    };
#elif (__CCE_AICORE__ == 300)
    const std::map<Hardware, uint32_t> bufferInitLen = {
        { Hardware::GM, 1024 * 1024 }, { Hardware::UB, 1024 * 256 },   { Hardware::L1, 1024 * 1024 },
        { Hardware::L0A, 1024 * 64 },  { Hardware::L0B, 1024 * 64 },   { Hardware::L0C, 1024 * 128 },
        { Hardware::BIAS, 1024 * 1 },  { Hardware::FIXBUF, 1024 * 7 },
    };
#elif defined(__DAV_M310__)
    const std::map<Hardware, uint32_t> bufferInitLen = {
        { Hardware::GM, 1024 * 1024 }, { Hardware::UB, 1024 * 256 },   { Hardware::L1, 1024 * 1024 },
        { Hardware::L0A, 1024 * 64 },  { Hardware::L0B, 1024 * 64 },   { Hardware::L0C, 1024 * 128 },
        { Hardware::BIAS, 1024 * 1 },  { Hardware::FIXBUF, 1024 * 7 },
    };
#endif
    uint8_t* cpuGM;
    uint8_t* cpuUB;
    uint8_t* cpuL1;
    uint8_t* cpuL0A;
    uint8_t* cpuL0B;
    uint8_t* cpuL0C;
    uint8_t* cpuBIAS;
    uint8_t* cpuFIXBUF;
    std::map<Hardware, uint8_t*> hardwareCpuBufferMap;

private:
    ConstDefiner() : cpuGM(new uint8_t[bufferInitLen.at(Hardware::GM)]),
        cpuUB(new uint8_t[bufferInitLen.at(Hardware::UB)]), cpuL1(new uint8_t[bufferInitLen.at(Hardware::L1)]),
        cpuL0A(new uint8_t[bufferInitLen.at(Hardware::L0A)]), cpuL0B(new uint8_t[bufferInitLen.at(Hardware::L0B)]),
        cpuL0C(new uint8_t[bufferInitLen.at(Hardware::L0C)]), cpuBIAS(new uint8_t[bufferInitLen.at(Hardware::BIAS)]),
        cpuFIXBUF(new uint8_t[bufferInitLen.at(Hardware::FIXBUF)]),
        hardwareCpuBufferMap({ { Hardware::UB, cpuUB }, { Hardware::L1, cpuL1 }, { Hardware::L0A, cpuL0A },
            { Hardware::L0B, cpuL0B }, { Hardware::L0C, cpuL0C }, { Hardware::BIAS, cpuBIAS },
            { Hardware::FIXBUF, cpuFIXBUF }, }) {}

    ~ConstDefiner()
    {
        if (cpuGM != nullptr) {
            delete[] cpuGM;
            cpuGM = nullptr;
        }
        if (cpuUB != nullptr) {
            delete[] cpuUB;
            cpuUB = nullptr;
        }
        if (cpuL1 != nullptr) {
            delete[] cpuL1;
            cpuL1 = nullptr;
        }
        if (cpuL0A != nullptr) {
            delete[] cpuL0A;
            cpuL0A = nullptr;
        }
        if (cpuL0B != nullptr) {
            delete[] cpuL0B;
            cpuL0B = nullptr;
        }
        if (cpuL0C != nullptr) {
            delete[] cpuL0C;
            cpuL0C = nullptr;
        }
        if (cpuBIAS != nullptr) {
            delete[] cpuBIAS;
            cpuBIAS = nullptr;
        }
        if (cpuFIXBUF != nullptr) {
            delete[] cpuFIXBUF;
            cpuFIXBUF = nullptr;
        }
    }
};
#endif
} // namespace AscendC
#endif // ASCENDC_MODULE_UTILS_MODE_H