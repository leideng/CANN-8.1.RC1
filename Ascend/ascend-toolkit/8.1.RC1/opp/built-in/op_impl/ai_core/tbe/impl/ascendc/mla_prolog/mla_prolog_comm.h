/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file mla_prolog_comm.h
 * \brief
 */

#ifndef MLA_PROLOG_COMM_H
#define MLA_PROLOG_COMM_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;
using matmul::Matmul;
using matmul::MatmulType;

#define USE_MM_API_MLAP
// 
enum CACHE_MODE {
    CACHE_MODE_BNSD = 0,
    CACHE_MODE_PA_BSND = 1,
    CACHE_MODE_PA_NZ = 2,
    CACHE_MODE_PA_BS = 3,
};

#ifdef ENABLE_DUMP_DATA
#define DO_DUMP_DATA(srcTensor, id, len) AscendC::DumpTensor(srcTensor, id, len)
#else
#define DO_DUMP_DATA(srcTensor, id, len)
#endif

// mte2 <> mte1
#define A_EVENT0 EVENT_ID4
#define A_EVENT1 EVENT_ID5
#define B_EVENT0 EVENT_ID6
#define B_EVENT1 EVENT_ID7

// m <> mte1
#define L0A_EVENT0 EVENT_ID3
#define L0A_EVENT1 EVENT_ID4
#define L0B_EVENT0 EVENT_ID5
#define L0B_EVENT1 EVENT_ID6

// fix <> m
#define L0C_EVENT0 EVENT_ID3
#define L0C_EVENT1 EVENT_ID4

#define L1_A_SIZE   (128 * 1024) // 512 / 4
#define L1_B_SIZE   (128 * 1024) // 512 / 4
#define L0A_PP_SIZE   (32 * 1024)
#define L0B_PP_SIZE   (32 * 1024)
#define L0C_PP_SIZE   (64 * 1024)

#define L1_A_SIZE   (128 * 1024) // Ping buffer size
#define L1_B_SIZE   (128 * 1024)

template <typename X_T, typename W_T, typename C_T, CACHE_MODE C_M, typename... Args>
struct MLAPType {
    using mmInputType = bfloat16_t;        // tokenX的类型与weight的类型一致
    using mmQcQrInputType = bfloat16_t;
    using mmQnInputType = bfloat16_t;     // matmul 计算Qn的输入类型
    using mmCqOutputType = bfloat16_t;     // matmul计算Cq的输出类型
    using mmCkvKrOutputType = bfloat16_t;  // matmul计算CkvKr的输出类型
    using mmQcQrOutputType = bfloat16_t;   // matmul计算QcQr的输出类型
    using mmQnOutputType = bfloat16_t;     // matmul计算Qn的输出类型
    using rmsNormGammaType = bfloat16_t;   // gamma的输入类型
    using rmsNormComputType = float;       //
    using rmsNormCqOutputType = bfloat16_t;   //
    using rmsNormCkvOutputType = bfloat16_t;  //
    using ropeSinCosType = bfloat16_t;     // sin cos的输入类型
    using ropeComputType = float;
    using ropeOutputType = bfloat16_t;     // 
    using cacheType = bfloat16_t;          // query query_rope kvcache krcahe的类型一致
    static constexpr CACHE_MODE cacheMode = C_M;
};

template <CACHE_MODE C_M, typename... Args>
struct MLAPType<bfloat16_t, int8_t, bfloat16_t, C_M, Args...> {
    using mmInputType = bfloat16_t;
    using mmQcQrInputType = int8_t;
    using mmQnInputType = bfloat16_t;          // tokenX的类型与weight的类型一致
    using mmCqOutputType = bfloat16_t;      // matmul计算Cq的输出类型
    using mmCkvKrOutputType = bfloat16_t;   // matmul计算CkvKr的输出类型
    using mmQcQrOutputType = int32_t;    // matmul计算QcQr的输出类型
    using mmQnOutputType = bfloat16_t;      // matmul计算Qn的输出类型
    using rmsNormGammaType = bfloat16_t;    // gamma的输入类型
    using rmsNormComputType = float;  //
    using rmsNormCqOutputType = int8_t; // 支持量化
    using rmsNormCkvOutputType = bfloat16_t;  //
    using ropeSinCosType = bfloat16_t;      // sin cos的输入类型
    using ropeComputType = float;   
    using ropeOutputType = bfloat16_t;      // 
    using cacheType = bfloat16_t;           // query query_rope kvcache krcahe的类型一致
    static constexpr CACHE_MODE cacheMode = C_M;
};

template <CACHE_MODE C_M, typename... Args>
struct MLAPType<bfloat16_t, int8_t, int8_t, C_M, Args...> {
    using mmInputType = bfloat16_t;
    using mmQcQrInputType = int8_t;
    using mmQnInputType = bfloat16_t;          // tokenX的类型与weight的类型一致
    using mmCqOutputType = bfloat16_t;      // matmul计算Cq的输出类型
    using mmCkvKrOutputType = bfloat16_t;   // matmul计算CkvKr的输出类型
    using mmQcQrOutputType = int32_t;    // matmul计算QcQr的输出类型
    using mmQnOutputType = bfloat16_t;      // matmul计算Qn的输出类型
    using rmsNormGammaType = bfloat16_t;    // gamma的输入类型
    using rmsNormComputType = float;  //
    using rmsNormCqOutputType = int8_t; // 支持量化
    using rmsNormCkvOutputType = int8_t;  //
    using ropeSinCosType = bfloat16_t;      // sin cos的输入类型
    using ropeComputType = float;   
    using ropeOutputType = bfloat16_t;      // 
    using ropeKroutputType = int8_t;
    using cacheType = int8_t;           // query query_rope kvcache krcahe的类型一致
    static constexpr CACHE_MODE cacheMode = C_M;
};

#ifdef USE_MM_API_MLAP

using MMParams = struct MMParams {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t orgM;
  uint32_t orgN;
  uint32_t orgKa;
  uint32_t orgKb;
  uint32_t orgKc;
  uint32_t baseM;
  uint32_t baseN;
  uint32_t baseK;
  uint32_t stepK;
  uint32_t needSetOrgShape;
};

struct MMBufParams {
  uint32_t aL1BufIter = 0;
  uint32_t bL1BufIter = 0;
  TBuffAddr aL1BufAddr;
  TBuffAddr bL1BufAddr;
};


constexpr MatmulConfig CFG_MDL_EXCEED_INIT{.doNorm = false,
                                           .doBasicBlock = false,
                                           .doMultiDataLoad = true,
                                           .basicM = 0,
                                           .basicN = 0,
                                           .basicK = 0,
                                           .intrinsicsCheck = true,
                                           .isNBatch = false,
                                           .enVecND2NZ = false,
                                           .doSpecialBasicBlock = false,
                                           .doMTE2Preload = false,
                                           .singleCoreM = 0,
                                           .singleCoreN = 0,
                                           .singleCoreK = 0,
                                           .stepM = 0,
                                           .stepN = 0,
                                           .baseMN = 0,
                                           .singleCoreMN = 0,
                                           .enUnitFlag = false,
                                           .isPerTensor = false,
                                           .hasAntiQuantOffset = false,
                                           .doIBShareNorm = false,
                                           .doSpecialMDL = false,
                                           .enableInit = false,
                                           .batchMode = BatchMode::NONE,
                                           .enableEnd = false,
                                           .enableGetTensorC = false,
                                           .enableSetOrgShape = true,
                                           .enableSetBias = false,
                                           .enableSetTail = true,
                                           .enableQuantVector = false,
                                           .enableSetDefineData = true,
                                           .iterateMode = IterateMode::ITERATE_MODE_ALL};
#else

using MMParams = struct MMParams {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t orgM;
  uint32_t orgN;
  uint32_t orgKa;
  uint32_t orgKb;
  uint32_t orgKc;
  uint32_t baseM;
  uint32_t baseN;
  uint32_t baseK;
  uint32_t stepK;
  uint32_t needSetOrgShape;
  uint32_t kL1SplitSize;
  uint32_t kSplitSize;
  uint32_t nSplitSize;
};

template<typename I, typename O>
struct MMBufParams {
  uint32_t aL1BufIter = 0;
  uint32_t bL1BufIter = 0;
  uint32_t aL0BufIter = 0;
  uint32_t bL0BufIter = 0;
  uint32_t cL0BufIter = 0;
  LocalTensor<I> aL1Tensor;
  LocalTensor<I> bL1Tensor;
  LocalTensor<I> aL0TensorPingPong;
  LocalTensor<I> bL0TensorPingPong;
  LocalTensor<O> cL0TensorPingPong;
};
#endif
#endif
