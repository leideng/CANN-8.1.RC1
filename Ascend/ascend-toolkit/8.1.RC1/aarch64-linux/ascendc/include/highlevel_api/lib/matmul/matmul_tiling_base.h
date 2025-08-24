/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_tiling_base.h
 * \brief
 */

#ifndef LIB_MATMUL_MATMUL_TILING_BASE_H
#define LIB_MATMUL_MATMUL_TILING_BASE_H

#include "matmul_tilingdata.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace matmul_tiling {
using half = double;
constexpr int32_t UINT8_BYTES = 1;
constexpr int32_t INT8_BYTES = 1;
constexpr int32_t FP32_BYTES = 4;
constexpr int32_t FP16_BYTES = 2;
constexpr int32_t C0_SIZE = 16;
constexpr int32_t C0_BYTE_SIZE = 32;
constexpr int32_t BITS_PER_BYTE = 8;
enum class DataType : int32_t {
    DT_FLOAT = 0,           // float type
    DT_FLOAT16 = 1,         // fp16 type
    DT_INT8 = 2,            // int8 type
    DT_INT16 = 6,           // int16 type
    DT_UINT16 = 7,          // uint16 type
    DT_UINT8 = 4,           // uint8 type
    DT_INT32 = 3,           // int32 type
    DT_INT64 = 9,           // int64 type
    DT_UINT32 = 8,          // unsigned int32
    DT_UINT64 = 10,         // unsigned int64
    DT_BOOL = 12,           // bool type
    DT_DOUBLE = 11,         // double type
    DT_STRING = 13,         // std::string type
    DT_DUAL_SUB_INT8 = 14,  // dual output int8 type
    DT_DUAL_SUB_UINT8 = 15, // dual output uint8 type
    DT_COMPLEX64 = 16,      // complex64 type
    DT_COMPLEX128 = 17,     // complex128 type
    DT_QINT8 = 18,          // qint8 type
    DT_QINT16 = 19,         // qint16 type
    DT_QINT32 = 20,         // qint32 type
    DT_QUINT8 = 21,         // quint8 type
    DT_QUINT16 = 22,        // quint16 type
    DT_RESOURCE = 23,       // resource type
    DT_STRING_REF = 24,     // std::string ref type
    DT_DUAL = 25,           // dual output type
    DT_VARIANT = 26,        // dt_variant type
    DT_BF16 = 27,           // bf16 type
    DT_UNDEFINED = 28,      // Used to indicate a DataType field has not been set.
    DT_INT4 = 29,           // int4 type
    DT_UINT1 = 30,          // uint1 type
    DT_INT2 = 31,           // int2 type
    DT_UINT2 = 32,          // uint2 type
    DT_BFLOAT16 = 33,       // bf16 type
    DT_MAX = 34             // Mark the boundaries of data types
};

const std::map<DataType, uint32_t> DTYPE_BYTE_TAB = {
    {DataType::DT_FLOAT, 4}, {DataType::DT_FLOAT16, 2}, {DataType::DT_INT8, 1}, {DataType::DT_INT16, 2},
    {DataType::DT_UINT16, 2}, {DataType::DT_UINT8, 1}, {DataType::DT_INT32, 4}, {DataType::DT_INT64, 8},
    {DataType::DT_UINT32, 4}, {DataType::DT_UINT64, 8}, {DataType::DT_BF16, 2}, {DataType::DT_BFLOAT16, 2},
    {DataType::DT_INT4, 1}
};

const std::map<DataType, uint32_t> DTYPE_BIT_TAB = {
    {DataType::DT_FLOAT, 32}, {DataType::DT_FLOAT16, 16}, {DataType::DT_INT8, 8}, {DataType::DT_INT16, 16},
    {DataType::DT_UINT16, 16}, {DataType::DT_UINT8, 8}, {DataType::DT_INT32, 32}, {DataType::DT_INT64, 64},
    {DataType::DT_UINT32, 32}, {DataType::DT_UINT64, 64}, {DataType::DT_BF16, 16}, {DataType::DT_BFLOAT16, 16},
    {DataType::DT_INT4, 4}
};

enum class TPosition : int32_t {
    GM,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    MAX,
};

enum class TilingPolicy : int32_t {
    FIXED_A_TSCM,
    FIXED_B_TSCM,
    FIXED_A_B_TSCM,
    NO_POLICY
};

enum class CubeFormat : int32_t {
    ND = 0,
    NZ,
    ZN,
    ZZ,
    NN,
    ND_ALIGN,
    SCALAR,
    VECTOR,
};

enum class MatrixTraverse : int32_t {
    NOSET = 0,
    FIRSTM = 1,
    FIRSTN = 2,
};

enum class MatrixMadType : int32_t {
    NORMAL = 0,
    HF32 = 1, // V220 HF32
};

enum class DequantType : int32_t {
    SCALAR = 0,
    TENSOR = 1,
};

enum class ScheduleType : int32_t {
    INNER_PRODUCT = 0,
    OUTER_PRODUCT = 1,
};

struct SysTilingTempBufSize {
    int32_t ubSize = 0;
    int32_t l1Size = 0;
    int32_t l0cSize = 0;
};

struct MatTilingType {
    TPosition pos = TPosition::GM;
    CubeFormat type = CubeFormat::ND;
    DataType dataType = DataType::DT_FLOAT;
    bool isTrans = false;
    bool isDB = false;
};

struct BufferPool {
    int32_t l1Size;
    int32_t l0CSize;
    int32_t ubSize;
    int32_t l0ASize;
    int32_t l0BSize;
    int32_t btSize;

    int32_t l1AlignSize;
    int32_t l0CAlignSize;
    int32_t l0AAlignSize;
    int32_t l0BAlignSize;
    int32_t ubAlignSize;
};

struct PlatformInfo {
    platform_ascendc::SocVersion socVersion;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
};

struct MatmulConfigParams {
    int32_t mmConfigType;
    bool enableL1CacheUB;
    ScheduleType scheduleType;
    MatrixTraverse traverse;
    bool enVecND2NZ;
    MatmulConfigParams(int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false,
        ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT, MatrixTraverse traverseIn = MatrixTraverse::NOSET,
        bool enVecND2NZIn = false) {
        mmConfigType = mmConfigTypeIn;
        enableL1CacheUB = enableL1CacheUBIn;
        scheduleType = scheduleTypeIn;
        traverse = traverseIn;
        enVecND2NZ = enVecND2NZIn;
    }
};

class MatmulApiTilingBase {
public:
    MatmulApiTilingBase();
    explicit MatmulApiTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform);
    explicit MatmulApiTilingBase(const PlatformInfo& platform);
    virtual ~MatmulApiTilingBase();
    int32_t SetAType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false);
    int32_t SetBType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false);
    int32_t SetCType(TPosition pos, CubeFormat type, DataType dataType);
    int32_t SetBiasType(TPosition pos, CubeFormat type, DataType dataType);
    int32_t SetDequantType(DequantType dequantType)
    {
        this->deqType = dequantType;
        return 0;
    }

    virtual int32_t SetShape(int32_t m, int32_t n, int32_t k);
    int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKIn);
    int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKaIn, int32_t orgKbIn);
    int32_t SetALayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d);
    int32_t SetBLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d);
    int32_t SetCLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d);
    int32_t SetBatchInfoForNormal(int32_t batchA, int32_t batchB, int32_t m, int32_t n, int32_t k);
    int32_t SetBatchNum(int32_t batch);
    int32_t EnableBias(bool isBiasIn = false);
    int32_t SetBias(bool isBiasIn = false);
    int32_t SetFixSplit(int32_t baseMIn = -1, int32_t baseNIn = -1, int32_t baseKIn = -1);
    int32_t SetBufferSpace(int32_t l1Size = -1, int32_t l0CSize = -1, int32_t ubSize = -1, int32_t btSize = -1);
    int32_t SetTraverse(MatrixTraverse traverse); // Set the N direction first for the upper left corner matrix
    int32_t SetMadType(MatrixMadType madType);    // Set hf32 mode
    // L0C:  BaseM * baseN = GetTensorC()
    // L1 :  BaseM * BaseK + BaseK*BaseN,  --> [disable temporarily] BaseK/k(1)=k1,  BaseM/m(1)=m1, BaseN/n(1) = n1
    int32_t SetSplitRange(int32_t maxBaseM = -1, int32_t maxBaseN = -1, int32_t maxBaseK = -1, int32_t minBaseM = -1,
        int32_t minBaseN = -1, int32_t minBaseK = -1);

    int32_t SetDoubleBuffer(bool a, bool b, bool c, bool bias, bool transND2NZ = true, bool transNZ2ND = true);

    void SetMatmulConfigParams(int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false,
        ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT, MatrixTraverse traverseIn = MatrixTraverse::NOSET,
        bool enVecND2NZIn = false);
    void SetMatmulConfigParams(const MatmulConfigParams& configParams);
    int32_t SetSparse(bool isSparseIn = false);

    int32_t GetBaseM() const
    {
        return baseM;
    }
    int32_t GetBaseN() const
    {
        return baseN;
    }
    int32_t GetBaseK() const
    {
        return baseK;
    }

    virtual int64_t GetTiling(optiling::TCubeTiling& tiling) = 0;
    virtual int64_t GetTiling(TCubeTiling& tiling) = 0;

public:
    optiling::TCubeTiling tiling_;

    MatTilingType aType_;
    MatTilingType bType_;
    MatTilingType cType_;
    MatTilingType biasType_;
    bool isBias = false;
    bool isSupportL0c2Out = true;
    int32_t blockDim = 0;
    int32_t orgM = 0;
    int32_t orgN = 0;
    int32_t orgKa = 0;
    int32_t orgKb = 0;

    int32_t aLayoutInfoB = 0;
    int32_t aLayoutInfoS = 0;
    int32_t aLayoutInfoN = 0;
    int32_t aLayoutInfoG = 0;
    int32_t aLayoutInfoD = 0;
    int32_t bLayoutInfoB = 0;
    int32_t bLayoutInfoS = 0;
    int32_t bLayoutInfoN = 0;
    int32_t bLayoutInfoG = 0;
    int32_t bLayoutInfoD = 0;
    int32_t cLayoutInfoB = 0;
    int32_t cLayoutInfoS1 = 0;
    int32_t cLayoutInfoN = 0;
    int32_t cLayoutInfoG = 0;
    int32_t cLayoutInfoS2 = 0;
    int32_t batchNum = 0;

    int32_t singleM = 0;
    int32_t singleN = 0;
    int32_t singleK = 0;

    int32_t singleCoreM = 0;
    int32_t singleCoreN = 0;
    int32_t singleCoreK = 0;

    int32_t baseM = 0;
    int32_t baseN = 0;
    int32_t baseK = 0;
    int32_t batchM = 0;
    int32_t batchN = 0;
    int32_t singleBatchM = 0;
    int32_t singleBatchN = 0;
    int32_t alignSingleM = 1;
    int32_t alignSingleN = 1;
    int32_t alignSingleK = 1;

    struct MnmAdjust {
        int32_t maxBaseM;
        int32_t maxBaseN;
        int32_t maxBaseK;

        int32_t minBaseM;
        int32_t minBaseN;
        int32_t minBaseK;
    } adjust_;

    BufferPool oriBufferPool_;
    BufferPool bufferPool_;
    MatrixTraverse traverse_ = MatrixTraverse::FIRSTM;
    MatrixMadType madType_ = MatrixMadType::NORMAL;
    ScheduleType scheduleType = ScheduleType::INNER_PRODUCT;
    bool transND2NZ_ = false;
    bool transNZ2ND_ = false;
    bool isSparse_ = false;

    int32_t maxSingleM = 0;
    int32_t maxSingleN = 0;
    int32_t maxSingleK = 0;
    int32_t minSingleM = 0;
    int32_t minSingleN = 0;
    int32_t minSingleK = 0;
    DequantType deqType = DequantType::SCALAR;
    bool enableSplitK_ = false;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    int32_t mmConfigType = 1; // 0: Norm; 1: MDL
    bool enableL1CacheUB = false;
    bool enVecND2NZ = false;

protected:
    virtual int64_t Compute() = 0;
    void SetFinalTiling(optiling::TCubeTiling& tiling);
    bool CheckSetParam();
    void PrintTilingData();
    void PrintTilingDataInfo(optiling::TCubeTiling &tiling) const;
};
} // namespace matmul_tiling

#endif // LIB_MATMUL_MATMUL_TILING_BASE_H
