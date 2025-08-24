/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file deep_norm_grad_common.h
 * \brief
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_DEEP_NORM_GRAD_DEEP_NORM_GRAD_COMMON_H_
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_DEEP_NORM_GRAD_DEEP_NORM_GRAD_COMMON_H_
#include <limits.h>

#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "../inc/safe_data_copy.h"

using namespace AscendC;

template <typename Tp, Tp v> struct integral_constant {
  static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename> struct is_same : public false_type {};
template <typename Tp> struct is_same<Tp, Tp> : public true_type {};

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_ALIGN_SIZE = 32;    // 32B
constexpr uint32_t REDUCE_REP_STRIDE = 8;
constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr uint32_t BRCB_ONCE_ELEM = 8;
constexpr uint32_t MAX_REP_NUM = 255;
constexpr uint32_t MAX_COPY_LENTH = 2000;

static volatile __gm__ uint32_t g_FixedOutputSync[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


template <typename T>
class KernelDeepNormGradBase {
public:
    __aicore__ inline uint32_t BlockAlign(uint32_t x, uint32_t blockElem) {
        if (blockElem > 0) {
            return (x + blockElem - 1) / blockElem * blockElem;
        }
        return 0;
    }
    __aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y) {
        if (y > 0) {return (x + y - 1) / y;}
        return 0;
    }

    /*
     * only support count <= 255 * 64 = 16320
     */
    __aicore__ inline float ReduceSumFP32WithReturn(const LocalTensor<float> &srcLocal, int32_t count)
    {
        int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
        int32_t repeatTimes = count / elementNumPerRep;
        int32_t tailCount = count % elementNumPerRep;
        int32_t bodyCount = repeatTimes * elementNumPerRep;
        #ifdef __CCE_KT_TEST__
        assert(count <= MAX_REPEAT_TIMES * elementNumPerRep);
        #endif
        float value = 0.0;
        if (g_coreType == AIV) {
            if (likely(repeatTimes > 0)) {
                AscendCUtils::SetMask<float>(elementNumPerRep);
                vcadd(nullptr, (__ubuf__ float *)srcLocal.GetPhyAddr(), repeatTimes, 1, 1, REDUCE_REP_STRIDE, true);
                event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                set_flag(PIPE_V, PIPE_S, event_v_s);
                wait_flag(PIPE_V, PIPE_S, event_v_s);
                #ifdef __CCE_KT_TEST__
                uint64_t accVal = get_acc_val();
                #else
                uint64_t accVal = GetAccVal();
                #endif
                value = *reinterpret_cast<float *>(&accVal);
            }
            if (unlikely(tailCount != 0)) {
                AscendCUtils::SetMask<float>(tailCount);
                vcadd(nullptr, (__ubuf__ float *)srcLocal[bodyCount].GetPhyAddr(), 1, 1, 1, REDUCE_REP_STRIDE, true);
                event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                set_flag(PIPE_V, PIPE_S, event_v_s);
                wait_flag(PIPE_V, PIPE_S, event_v_s);
                #ifdef __CCE_KT_TEST__
                uint64_t accVal = get_acc_val();
                #else
                uint64_t accVal = GetAccVal();
                #endif
                value += *reinterpret_cast<float *>(&accVal);
            }
        }
        return value;
    }

    __aicore__ inline float ReduceSumCustom(const LocalTensor<float>& src_local, int32_t count) {
#if __CCE_AICORE__ == 220
        return this -> ReduceSumFP32WithReturn(src_local, count);
#else
        ReduceSum(src_local, src_local, src_local, count);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        set_flag(PIPE_V, PIPE_S, event_v_s);
        wait_flag(PIPE_V, PIPE_S, event_v_s);
        float rstd_value = src_local.GetValue(0);
        return rstd_value;
#endif
    }
    
    __aicore__ inline void BlockReduceSumFp32Short(const LocalTensor<float> &dstLocal,
                                                   const LocalTensor<float> &tmpLocal,
                                                   uint32_t repeat)
    {
        uint32_t elemNum = FLOAT_BLOCK_ELEM; 
        uint32_t maxRepeat = ONE_REPEAT_BYTE_SIZE / sizeof(float);

        uint32_t repeatTimes = repeat / elemNum;
        uint32_t bodyCount = repeatTimes * elemNum;
        uint32_t repTailNum = repeat % elemNum * elemNum;

        if (repeatTimes != 0) {
            BlockReduceSum<float>(dstLocal, tmpLocal, repeatTimes, maxRepeat, 1, 1, elemNum);
        }
        if (repTailNum != 0) {
            BlockReduceSum<float>(dstLocal[bodyCount], tmpLocal[bodyCount * elemNum], 1, repTailNum , 1, 1, elemNum);
        }
    }

    __aicore__ inline void ReduceSumFp32Short(const LocalTensor<float> &dstLocal,
                                              const LocalTensor<float> &srcLocal,
                                              const LocalTensor<float> &tmpLocal,
                                              uint32_t alignElem, uint32_t repeat, uint32_t processElem)
    {
        uint32_t elemNum = FLOAT_BLOCK_ELEM;  // 8
        uint32_t tailCount = processElem % elemNum;
        uint8_t repStride = alignElem / FLOAT_BLOCK_ELEM;

        uint32_t repeatTimes = repeat / MAX_REP_NUM;

        uint32_t index = 0;
        uint32_t elemIndex = 0;
        uint32_t tmpIndex = 0;
        if (likely(repeatTimes == 0)){
            elemIndex = 0;
            for (elemIndex = 0; elemIndex + elemNum <= processElem; elemIndex += elemNum) {
                Add(tmpLocal, tmpLocal, srcLocal[elemIndex], elemNum, repeat, {1, 1, 1, 1, 1, repStride});
                pipe_barrier(PIPE_V);
            }
            if (unlikely(tailCount != 0)) {
                Add(tmpLocal, tmpLocal, srcLocal[elemIndex], tailCount, repeat, {1, 1, 1, 1, 1, repStride});
            }
        } else {
            uint32_t repTailNum = repeat % MAX_REP_NUM;
            uint32_t repIndex = 0;
            uint32_t repElem;
            for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
                elemIndex = 0;
                tmpIndex = repIndex * FLOAT_BLOCK_ELEM;
                repElem = repIndex * alignElem;
                for (elemIndex = 0; elemIndex + elemNum <= processElem; elemIndex += elemNum) {
                    index = repElem + elemIndex;
                    Add(tmpLocal[tmpIndex], tmpLocal[tmpIndex], srcLocal[index], elemNum, MAX_REP_NUM, {1, 1, 1, 1, 1, repStride});
                    pipe_barrier(PIPE_V);
                }
                if (unlikely(tailCount != 0)) {
                    index = repElem + elemIndex;
                    Add(tmpLocal[tmpIndex], tmpLocal[tmpIndex], srcLocal[index], tailCount, MAX_REP_NUM, {1, 1, 1, 1, 1, repStride});
                    pipe_barrier(PIPE_V);
                }
            }
            if (repTailNum != 0) {
                elemIndex = 0;
                tmpIndex = repIndex * FLOAT_BLOCK_ELEM;
                repElem = repIndex * alignElem;
                for (elemIndex = 0; elemIndex + elemNum <= processElem; elemIndex += elemNum) {
                    index = repElem + elemIndex;
                    Add(tmpLocal[tmpIndex], tmpLocal[tmpIndex], srcLocal[index], elemNum, repTailNum, {1, 1, 1, 1, 1, repStride});
                    pipe_barrier(PIPE_V);
                }
                if (unlikely(tailCount != 0)) {
                    index = repElem + elemIndex;
                    Add(tmpLocal[tmpIndex], tmpLocal[tmpIndex], srcLocal[index], tailCount, repTailNum, {1, 1, 1, 1, 1, repStride});
                }
            }
        }

        pipe_barrier(PIPE_V);
        BlockReduceSumFp32Short(dstLocal, tmpLocal, repeat);
    }

    __aicore__ inline void Level0MulFp32Short(const LocalTensor<float> &dstLocal,
                                              const LocalTensor<float> &srcLocal0,
                                              const LocalTensor<float> &srcLocal1,
                                              uint32_t alignElem, uint32_t repeat, uint32_t processElem)
    {
        uint32_t maxElemFp32 = ONE_REPEAT_BYTE_SIZE / sizeof(float);    // 64
        uint8_t repStride = alignElem / FLOAT_BLOCK_ELEM;
        uint32_t tailCount = processElem % maxElemFp32;

        uint32_t repeatTimes = repeat / MAX_REP_NUM;

        uint32_t index = 0;
        uint32_t elemIndex = 0;
        if (likely(repeatTimes == 0)){
            for (elemIndex = 0; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                Mul(dstLocal[elemIndex], srcLocal0[elemIndex], srcLocal1[elemIndex], maxElemFp32, repeat, {1, 1, 1, repStride, repStride, 0});
                pipe_barrier(PIPE_V);
            }
            if (tailCount != 0) {
                Mul(dstLocal[elemIndex], srcLocal0[elemIndex], srcLocal1[elemIndex], tailCount, repeat, {1, 1, 1, repStride, repStride, 0});
            }
        } else {
            uint32_t repTailNum = repeat % MAX_REP_NUM;
            uint32_t repIndex = 0;
            uint32_t repElem;
            for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
                elemIndex = 0;
                repElem = repIndex * alignElem;
                for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                    index = repElem + elemIndex;
                    Mul(dstLocal[index], srcLocal0[index], srcLocal1[elemIndex], maxElemFp32, MAX_REP_NUM, {1, 1, 1, repStride, repStride, 0});
                    pipe_barrier(PIPE_V);
                }
                if (tailCount != 0) {
                    index = repElem + elemIndex;
                    Mul(dstLocal[index], srcLocal0[index], srcLocal1[elemIndex], tailCount, MAX_REP_NUM, {1, 1, 1, repStride, repStride, 0});
                    pipe_barrier(PIPE_V);
                }
            }
            if (repTailNum != 0) {
                elemIndex = 0;
                repElem = repIndex * alignElem;
                for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                    index = repElem + elemIndex;
                    Mul(dstLocal[index], srcLocal0[index], srcLocal1[elemIndex], maxElemFp32, repTailNum, {1, 1, 1, repStride, repStride, 0});
                    pipe_barrier(PIPE_V);
                }
                if (tailCount != 0) {
                    index = repElem + elemIndex;
                    Mul(dstLocal[index], srcLocal0[index], srcLocal1[elemIndex], tailCount, repTailNum, {1, 1, 1, repStride, repStride, 0});
                }
            }
        }
    }

    __aicore__ inline void Level0AddFp32Short(const LocalTensor<float> &dstLocal,
                                              const LocalTensor<float> &srcLocal,
                                              uint32_t alignElem, uint32_t repeat, uint32_t processElem)
    {
        uint32_t maxElemFp32 = ONE_REPEAT_BYTE_SIZE / sizeof(float);    // 64
        uint8_t repStride = alignElem / FLOAT_BLOCK_ELEM;
        uint32_t tailCount = processElem % maxElemFp32;

        uint32_t repeatTimes = repeat / MAX_REP_NUM;

        uint32_t index = 0;
        uint32_t elemIndex = 0;
        if (likely(repeatTimes == 0)){
            elemIndex = 0;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                Add(dstLocal[elemIndex], srcLocal[elemIndex], dstLocal[elemIndex], maxElemFp32, repeat, {1, 1, 1, 0, repStride, 0});
                pipe_barrier(PIPE_V);
            }
            if (tailCount != 0) {
                Add(dstLocal[elemIndex], srcLocal[elemIndex], dstLocal[elemIndex], tailCount, repeat, {1, 1, 1, 0, repStride, 0});
            }
        } else {
            uint32_t repTailNum = repeat % MAX_REP_NUM;
            uint32_t repIndex = 0;
            uint32_t repElem;
            for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
                elemIndex = 0;
                repElem = repIndex * alignElem;
                for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                    index = repElem + elemIndex;
                    Add(dstLocal[elemIndex], srcLocal[index], dstLocal[elemIndex], maxElemFp32, MAX_REP_NUM, {1, 1, 1, 0, repStride, 0});
                    pipe_barrier(PIPE_V);
                }
                if (tailCount != 0) {
                    index = repElem + elemIndex;
                    Add(dstLocal[elemIndex], srcLocal[index], dstLocal[elemIndex], tailCount, MAX_REP_NUM, {1, 1, 1, 0, repStride, 0});
                    pipe_barrier(PIPE_V);
                }
            }
            if (repTailNum != 0) {
                elemIndex = 0;
                repElem = repIndex * alignElem;
                for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                    index = repElem + elemIndex;
                    Add(dstLocal[elemIndex], srcLocal[index], dstLocal[elemIndex], maxElemFp32, repTailNum, {1, 1, 1, 0, repStride, 0});
                    pipe_barrier(PIPE_V);
                }
                if (tailCount != 0) {
                    index = repElem + elemIndex;
                    Add(dstLocal[elemIndex], srcLocal[index], dstLocal[elemIndex], tailCount, repTailNum, {1, 1, 1, 0, repStride, 0});
                }
            }
        }
    }

    __aicore__ inline void InitGmData(GlobalTensor<float> outputPdGammaGm, GlobalTensor<float> outputPdBetaGm, 
    const uint32_t dDimNum, LocalTensor<float> dbeta, uint32_t elemWithDInUBFp32){
#if __CCE_AICORE__ == 220
        if (GetBlockIdx() == 0) {
        InitOutput<float>(outputPdGammaGm, dDimNum, 0);
        InitOutput<float>(outputPdBetaGm, dDimNum, 0);
        g_FixedOutputSync[0] = 0;
}
#else
        if (GetBlockIdx() == 0) {
            uint32_t maxCopyLenth = elemWithDInUBFp32;
            uint32_t loopNum = dDimNum / maxCopyLenth;
            uint32_t tail = dDimNum % maxCopyLenth;
            if (loopNum == 0){
                Duplicate(dbeta, 0.0f, this->BlockAlign(dDimNum, FLOAT_BLOCK_ELEM));
            } else {
                Duplicate(dbeta, 0.0f, maxCopyLenth);
            }
            pipe_barrier(PIPE_ALL);
            for (uint32_t idx = 0; idx < loopNum; idx++) {
                uint32_t curOffset = idx * maxCopyLenth;
                DataCopy(outputPdGammaGm[curOffset], dbeta, maxCopyLenth);
                DataCopy(outputPdBetaGm[curOffset], dbeta, maxCopyLenth);
            }
            if (tail != 0) {
                if (loopNum >= 1) {
                    uint32_t rollbackTail = maxCopyLenth + tail;
                    Duplicate(dbeta, 0.0f, this->BlockAlign(rollbackTail, FLOAT_BLOCK_ELEM));
                    SafeDataCopy(outputPdGammaGm[maxCopyLenth * (loopNum - 1)], dbeta, rollbackTail);
                    SafeDataCopy(outputPdBetaGm[maxCopyLenth * (loopNum - 1)], dbeta, rollbackTail);
                } else {
                    DataCopy(outputPdGammaGm[maxCopyLenth * loopNum], dbeta, this->BlockAlign(tail, FLOAT_BLOCK_ELEM));
                    DataCopy(outputPdBetaGm[maxCopyLenth * loopNum], dbeta, this->BlockAlign(tail, FLOAT_BLOCK_ELEM));
                }
            }
            pipe_barrier(PIPE_ALL);
        }
    #endif
    }
};
__aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t blockElem) {
    if (blockElem > 0) {
        return (x + blockElem - 1) / blockElem * blockElem;
    }
    return 0;
}

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U& dstTensor, const R& srcTensor, const uint32_t processElem, const uint32_t offset, bool is_float, 
const uint16_t blockCount) {
#if __CCE_AICORE__ == 220
    DataCopyParams dataCopyParamsND{blockCount, (uint16_t)(processElem * sizeof(T)), 0, 0};
    DataCopyPad(dstTensor[offset], srcTensor, dataCopyParamsND);
#else
    int32_t blockNumel = is_float ? BLOCK_ALIGN_SIZE / sizeof(float) : BLOCK_ALIGN_SIZE / sizeof(T);
    int32_t blockNum = processElem / blockNumel;//32/byte
    int32_t tail = processElem % blockNumel;
    int32_t blkLength = blockNum * blockNumel;
    for (uint32_t idx = 0; idx < blockCount; idx++) {
        uint32_t curOffset = offset + idx * processElem;
        if (blockNum == 0) {
            break;
        }
        DataCopy(dstTensor[curOffset], srcTensor[idx * RoundUp(processElem, blockNumel)], blkLength);
        if (tail != 0) {
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            for (uint32_t i = 0; i < blockNumel; i++) {
                T tensorValue = srcTensor.GetValue(idx * RoundUp(processElem, blockNumel) + processElem - blockNumel + i);
                srcTensor.SetValue(idx * RoundUp(processElem, blockNumel) + i, tensorValue);
            }
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            DataCopy(dstTensor[curOffset + processElem - blockNumel], srcTensor[idx * RoundUp(processElem, blockNumel)], blockNumel);
        }
    }
#endif
}

__aicore__ inline void DataCopyAutomicAdd(GlobalTensor<float> dstTensor, const LocalTensor<float> srcTensor, const uint32_t processElem, const uint32_t offset,
const uint16_t blockCount) {
#if __CCE_AICORE__ == 220
    DataCopyParams dataCopyParamsND{blockCount, (uint16_t)(processElem * sizeof(float)), 0, 0};
    DataCopyPad(dstTensor[offset], srcTensor, dataCopyParamsND);
#else
    SafeDataCopy<true>(dstTensor[offset], srcTensor, blockCount * processElem);
#endif
}

#endif
