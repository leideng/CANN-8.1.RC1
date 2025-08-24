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
 * \file kernel_utils.h
 * \brief
 */
#ifndef ASCENDC_MODULE_UTILS_H
#define ASCENDC_MODULE_UTILS_H
#include "utils/kernel_utils_macros.h"
#include "utils/kernel_utils_ceil_oom_que.h"
#include "utils/kernel_utils_constants.h"
#include "utils/kernel_utils_mode.h"
#include "utils/kernel_utils_struct_confusion_pad.h"
#include "utils/kernel_utils_struct_dma_params.h"
#include "utils/kernel_utils_struct_norm_sort.h"
#include "utils/kernel_utils_struct_param.h"

#include "kernel_struct_data_copy.h"

namespace AscendC {
class AscendCUtils {
public:
    __aicore__ static inline int32_t GetBitSize(int32_t byteSize)
    {
        return byteSize * ONE_BYTE_BIT_SIZE;
    }

    __aicore__ static inline int32_t GetC0Size()
    {
        return DEFAULT_C0_SIZE;
    }

    __aicore__ static inline int32_t GetC0Count(const int32_t dtypeSize)
    {
        ASCENDC_ASSERT((dtypeSize != 0), { KERNEL_LOG(KERNEL_ERROR, "dtypeSize can not be 0"); });
        return GetC0Size() / dtypeSize;
    }

    __aicore__ static inline int32_t GetDefaultBlockNum()
    {
        return DEFAULT_BLK_NUM;
    }

    __aicore__ static inline int64_t GetRsvdCnt()
    {
        return get_rsvd_cnt();
    }

    template <typename T, bool isSetMask = true>
    __aicore__ static inline void SetMask(const uint64_t& maskHigh, const uint64_t& maskLow)
    {
        if constexpr (!isSetMask) {
            return;
        }
        if ASCEND_IS_NOT_AIC {
            set_vector_mask(maskHigh, maskLow);
        }
    }

    template <typename T, bool isSetMask = true> __aicore__ static inline void SetMask(int32_t len)
    {
        if constexpr (!isSetMask) {
            return;
        }

        int32_t typeLen = 0;
        if constexpr (IsSameType<T, int4b_t>::value) {
            typeLen = DEFAULT_BLOCK_SIZE * INT4_TWO;
        } else {
            typeLen = DEFAULT_BLOCK_SIZE / sizeof(T);
        }
        constexpr int32_t halfTypeLen = 64;  // 1 register -> 64 bits -> 64 elements
        constexpr int32_t lenCoeff = 2;      // 2 registers for masks
        if (len == halfTypeLen) {
            SetMask<T>(0, FULL_MASK);
            return;
        } else if (len == typeLen || len >= halfTypeLen * lenCoeff) { // len = max ele per repeat / len >= 128
            SetMask<T>(FULL_MASK, FULL_MASK);
            return;
        }
        SetMask<T>(static_cast<uint64_t>(
            (len > halfTypeLen) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(len - halfTypeLen)) - 1) : 0),
            static_cast<uint64_t>(
            (len > halfTypeLen) ? FULL_MASK : (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(len)) - 1)));
    }

    template <typename T> __aicore__ static inline void SetMaskCount()
    {
        set_mask_count();
    }

    template <typename T> __aicore__ static inline void SetMaskNorm()
    {
        set_mask_norm();
    }

#if __CCE_AICORE__ >= 220
    __aicore__ static inline void SetOverflow(uint64_t ctrlValue)
    {
        // set CTRL[48] is 1 --- inf/nan mode
        // set CTRL[48] is 0 --- saturated mode
        if (ctrlValue == 1) {
            set_ctrl(sbitset1(get_ctrl(), CTRL_48_BIT));
        } else {
            set_ctrl(sbitset0(get_ctrl(), CTRL_48_BIT));
        }
    }

#elif __CCE_AICORE__ == 200
    __aicore__ static inline void SetOverflow(uint64_t ctrlValue)
    {
        // set CTRL[53] is 1 --- saturated mode
        // set CTRL[53] is 0 --- inf/nan mode
        if (ctrlValue == 0) {
            set_ctrl(sbitset1(get_ctrl(), CTRL_53_BIT));
        } else {
            set_ctrl(sbitset0(get_ctrl(), CTRL_53_BIT));
        }
    }
#endif

    template <bool isSetMask = true> __aicore__ static inline void ResetMask()
    {
        if constexpr (!isSetMask) {
            return;
        }
        if ASCEND_IS_NOT_AIC {
            set_vector_mask(FULL_MASK, FULL_MASK);
        }
    }

    template <bool isInt4 = false>
    __aicore__ inline static IntriInfo CalIntriInfo(
        const uint32_t dtypeSize, const uint32_t calCount, uint32_t repStride = DEFAULT_BLK_NUM)
    {
        IntriInfo retIntriInfo;
        retIntriInfo.c0Count = GetC0Count(dtypeSize);
        if constexpr (isInt4) {
            retIntriInfo.c0Count = GetC0Size() * INT4_TWO;
        }
        uint32_t repeatCount = repStride * retIntriInfo.c0Count;
        retIntriInfo.repeat = calCount / repeatCount;
        retIntriInfo.tail = calCount % repeatCount;
        retIntriInfo.repeatRounding = retIntriInfo.repeat / MAX_REPEAT_TIMES;
        retIntriInfo.repeatRemaining = retIntriInfo.repeat % MAX_REPEAT_TIMES;

        return retIntriInfo;
    }

    template <typename T>
    __aicore__ static inline __ubuf__ T* GetTemporaryBufferAddr(const int32_t bufferOffset, const int32_t bufferSize)
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((bufferOffset % ONE_BLK_SIZE == 0),
                       { KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %d, which must be 32B aligned", bufferOffset); });
        ASCENDC_ASSERT(
            (bufferOffset + bufferSize * sizeof(T) <= ConstDefiner::Instance().bufferInitLen.at(Hardware::UB)), {
                KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %d, bufferSize is %d, which exceed the limit of ub %d",
                    bufferOffset, bufferSize, ConstDefiner::Instance().bufferInitLen.at(Hardware::UB));
            });
        const int32_t maxTempSize = 0x100000;
        ASCENDC_ASSERT((bufferSize < maxTempSize), {
            KERNEL_LOG(KERNEL_ERROR, "bufferSize is %d, which exceed the maxTempSize limits %d", bufferSize,
                maxTempSize);
        });
        T* addr = reinterpret_cast<T*>(ConstDefiner::Instance().hardwareCpuBufferMap.at(Hardware::UB) + bufferOffset);
#else
        (void)bufferSize;
        __ubuf__ T* addr = reinterpret_cast<__ubuf__ T*>(get_imm(0) + bufferOffset);
#endif
        return addr;
    }

    template <typename T> __aicore__ static inline void FreeTemporaryBuffer(__ubuf__ T* addr)
    {
        (void)addr;
    }

#if __CCE_AICORE__ >= 220
    template <typename T>
    __aicore__ static inline __fbuf__ T* GetTemporaryFbBufferAddr(const int32_t bufferOffset, const int32_t bufferSize)
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((bufferOffset % ONE_BLK_SIZE == 0),
                       { KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %d, which must be 32B aligned", bufferOffset); });
        ASCENDC_ASSERT(
            (bufferOffset + bufferSize * sizeof(T) <= ConstDefiner::Instance().bufferInitLen.at(Hardware::FIXBUF)), {
                KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %d, bufferSize is %d, which exceed the limit of fixbuf %d",
                    bufferOffset, bufferSize, ConstDefiner::Instance().bufferInitLen.at(Hardware::FIXBUF));
            });
        T* addr =
            reinterpret_cast<T*>(ConstDefiner::Instance().hardwareCpuBufferMap.at(Hardware::FIXBUF) + bufferOffset);
#else
        (void)bufferSize;
        __fbuf__ T* addr = reinterpret_cast<__fbuf__ T*>(get_imm(0) + bufferOffset);
#endif
        return addr;
    }

    template <typename T> __aicore__ static inline void FreeTemporaryFbBuffer(__fbuf__ T* addr)
    {
        (void)addr;
    }
#endif

    __aicore__ static inline uint64_t GetGMLen(const DataCopyParams& intriParams, const bool& isSrc,
        const bool& isMovAlignIntri)
    {
        uint16_t stride = intriParams.dstStride;
        uint16_t burstLenUnit = 32;
        uint16_t strideUnit = 32;
        if (isSrc) {
            stride = intriParams.srcStride;
        }
        if (isMovAlignIntri) {
            burstLenUnit = 1;
            strideUnit = 1;
        }
        if (intriParams.blockLen == 0) {
            return 0;
        }
        uint64_t gmLen = static_cast<uint64_t>(intriParams.blockCount) * intriParams.blockLen * burstLenUnit +
            (intriParams.blockCount - 1) * stride * strideUnit;
        return gmLen;
    }

    __aicore__ static inline uint64_t GetGMLen(const DataCopyExtParams& intriParams, const bool& isSrc,
        const bool& isMovAlignIntri)
    {
        uint16_t stride = intriParams.dstStride;
        uint16_t burstLenUnit = 32;
        uint16_t strideUnit = 32;
        if (isSrc) {
            stride = intriParams.srcStride;
        }
        if (isMovAlignIntri) {
            burstLenUnit = 1;
            strideUnit = 1;
        }
        if (intriParams.blockLen == 0) {
            return 0;
        } 
        uint64_t gmLen = static_cast<uint64_t>(intriParams.blockCount) * intriParams.blockLen * burstLenUnit +
            (intriParams.blockCount - 1) * stride * strideUnit;
        return gmLen;
    }

    __aicore__ static inline uint64_t GetGMLen(const uint64_t& srcEleSize, const Nd2NzParams& intriParams)
    {
        uint64_t gmLen = (static_cast<uint64_t>(intriParams.ndNum) - 1) * srcEleSize * intriParams.srcNdMatrixStride +
            (intriParams.nValue - 1) * intriParams.srcDValue * srcEleSize + intriParams.dValue * srcEleSize;
        return gmLen;
    }

    __aicore__ static inline bool OOMCheckAddrIsOverflow(uintptr_t gmAddrConvert, const uint64_t& gmLen)
    {
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        uintptr_t inputOutputAddr = 0;
        uint64_t inputOutputLen = 0;

        for (uint64_t index = 0; index < g_oomAddrArange.count; index++) {
            if (g_oomAddrArange.addr[index] == 0 || g_oomAddrArange.len[index] == 0) {
                continue;
            }
            if (g_oomAddrArange.isLevelOnePointer[index] == 0 &&
                OOMCheckAddrInTensorList(index, gmAddrConvert, inputOutputAddr, inputOutputLen)) {
                break;
            } else {
                inputOutputAddr = g_oomAddrArange.addr[index];
                inputOutputLen = g_oomAddrArange.len[index];
                if (gmAddrConvert >= inputOutputAddr && gmAddrConvert < inputOutputAddr + inputOutputLen) {
                    break;
                }
            }
            if (index == g_oomAddrArange.count - 1) {
                return true;
            }
        }
        if (gmAddrConvert + gmLen > inputOutputAddr + inputOutputLen) {
            return true;
        }
#endif
        return false;
    }

    template <typename T>
    __aicore__ static inline void CheckGmMemOverflow(__gm__ T* gmAddr, const bool& isSrc,
        const uint64_t& gmLen)
    {
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        if (gmLen == 0) {
            return;
        }
        if (g_oomAddrArange.count == 0) {
            return;
        }
        uintptr_t gmAddrConvert = reinterpret_cast<uintptr_t>(gmAddr);
        bool status = OOMCheckAddrIsOverflow(gmAddrConvert, gmLen);
#if defined(L2_CACHE_HINT) && (__CCE_AICORE__ == 220)
        if ASCEND_IS_NOT_AIV {
            if (status) {
                uint64_t oriGmAddr = reinterpret_cast<uint64_t>(gmAddr);
                if (oriGmAddr >= g_opSystemRunCfg.l2Cacheoffset) {
                    oriGmAddr -= g_opSystemRunCfg.l2Cacheoffset;
                }
                gmAddrConvert = reinterpret_cast<uintptr_t>(oriGmAddr);
                status = OOMCheckAddrIsOverflow(gmAddrConvert, gmLen);
            }
        }
#endif // L2_CACHE_HINT
        constexpr uint64_t errCode = 0X5A5A0001;
        if (status) {
#if __CCE_AICORE__ == 300 || defined(__DAV_M310__)
            trap();
#else
            trap(errCode);
#endif
        }
#endif
    }

    template <typename T>
    __aicore__ static inline void CheckGmMemOverflowNormal(__gm__ T *gmAddr, __gm__ uint8_t *workSpace,
        const bool isSrc, const bool isMovAlignIntri, const DataCopyParams& intriParams)
    {
        (void)(workSpace);
        uint64_t gmLen = GetGMLen(intriParams, isSrc, isMovAlignIntri);
        CheckGmMemOverflow(gmAddr, isSrc, gmLen);
    }

    template <typename T>
    __aicore__ static inline void CheckGmMemOverflowNormal(__gm__ T* gmAddr, __gm__ uint8_t* workSpace,
        const bool isSrc, const bool isMovAlignIntri, const DataCopyExtParams& intriParams)
    {
        (void)(workSpace);
        uint64_t gmLen = GetGMLen(intriParams, isSrc, isMovAlignIntri);
        CheckGmMemOverflow(gmAddr, isSrc, gmLen);
    }

    template <typename T>
    __aicore__ static inline void CheckGmMemOverflowNd2Nz(__gm__ T* gmAddr, __gm__ uint8_t* workSpace,
        const bool isSrc, const Nd2NzParams& intriParams)
    {
        (void)(workSpace);
        uint64_t srcEleSize = sizeof(T);
        uint64_t gmLen = GetGMLen(srcEleSize, intriParams);
        CheckGmMemOverflow(gmAddr, isSrc, gmLen);
    }
};

#ifdef ASCENDC_CPU_DEBUG
enum AtomicType {
    SUM,
    MAX,
    MIN
};
extern bool g_isAtomic;
extern AtomicType g_atomicType;

template <typename T>
__aicore__ inline void DataCopyWithAtomic(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    const uint16_t nBurst = intriParams.blockCount;
    const uint16_t lenBurst = intriParams.blockLen;
    const uint16_t srcStride = intriParams.srcStride;
    const uint16_t dstStride = intriParams.dstStride;
    // new one buffer and do add
    uint32_t dstOffset = 0;
    uint32_t srcOffset = 0;
    const int repeat = (lenBurst * ONE_BLK_SIZE + ONE_REPEAT_BYTE_SIZE - 1) / ONE_REPEAT_BYTE_SIZE;
    for (int index = 0; index < nBurst; ++index) {
        for (int indexJ = 0; indexJ < lenBurst * ONE_BLK_SIZE / sizeof(T); ++indexJ) {
            if (g_atomicType == SUM) {
                *(static_cast<T*>(src) + srcOffset + indexJ) =
                    *(static_cast<T*>(dst) + dstOffset + indexJ) + *(static_cast<T*>(src) + srcOffset + indexJ);
            } else if (g_atomicType == MAX) {
                *(static_cast<T*>(src) + srcOffset + indexJ) = std::max(*(static_cast<T*>(dst) + dstOffset + indexJ),
                    *(static_cast<T*>(src) + srcOffset + indexJ));
            } else {
                *(static_cast<T*>(src) + srcOffset + indexJ) = std::min(*(static_cast<T*>(dst) + dstOffset + indexJ),
                    *(static_cast<T*>(src) + srcOffset + indexJ));
            }
        }
        dstOffset += ((lenBurst + dstStride) * ONE_BLK_SIZE) / sizeof(T);
        srcOffset += ((lenBurst + srcStride) * ONE_BLK_SIZE) / sizeof(T);
    }
}

template <typename T>
__aicore__ inline void DataCopyWithAtomicCom(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    const uint16_t nBurst = intriParams.blockCount;
    const uint16_t lenBurst = intriParams.blockLen;
    const uint16_t srcStride = intriParams.srcStride;
    const uint16_t dstStride = intriParams.dstStride;
    const uint16_t halfSize = sizeof(T);
    // new one buffer and do add
    uint32_t dstOffset = 0;
    uint32_t srcOffset = 0;
    const int repeat = (lenBurst * ONE_BLK_SIZE) / ONE_REPEAT_BYTE_SIZE;
    const int countInRepeat = (ONE_REPEAT_BYTE_SIZE / halfSize);
#if __CCE_AICORE__ <= 220
    const int tail = lenBurst * ONE_BLK_SIZE / halfSize - repeat * countInRepeat;
#endif
    for (int index = 0; index < nBurst; ++index) {
#if __CCE_AICORE__ <= 220
        __ubuf__ T* dstAddr = static_cast<__ubuf__ T*>(src) + srcOffset;
        __ubuf__ T* src0Addr = static_cast<__ubuf__ T*>(dst) + dstOffset;
        __ubuf__ T* src1Addr = static_cast<__ubuf__ T*>(src) + srcOffset;
        if (repeat > 0) {
            AscendCUtils::SetMask<T>(countInRepeat);
            if (g_atomicType == SUM) {
                vadd(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), repeat, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            } else if (g_atomicType == MAX) {
                vmax(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), repeat, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            } else {
                vmin(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), repeat, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            }
            AscendCUtils::ResetMask();
        }
        if (tail != 0) {
            dstAddr = dstAddr + repeat * countInRepeat;
            src0Addr = src0Addr + repeat * countInRepeat;
            src1Addr = src1Addr + repeat * countInRepeat;
            AscendCUtils::SetMask<T>(tail);
            if (g_atomicType == SUM) {
                vadd(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), 1, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            } else if (g_atomicType == MAX) {
                vmax(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), 1, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            } else {
                vmin(static_cast<T*>(dstAddr), static_cast<T*>(src0Addr), static_cast<T*>(src1Addr), 1, 1, 1, 1,
                    DEFAULT_BLK_NUM, DEFAULT_BLK_NUM, DEFAULT_BLK_NUM);
            }
            AscendCUtils::ResetMask();
        }
#endif
        dstOffset += ((lenBurst + dstStride) * ONE_BLK_SIZE) / halfSize;
        srcOffset += ((lenBurst + srcStride) * ONE_BLK_SIZE) / halfSize;
    }
}

__aicore__ inline void DataCopyWithAtomic(__gm__ half* dst, __ubuf__ half* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    DataCopyWithAtomicCom(dst, src, intriParams);
}
__aicore__ inline void DataCopyWithAtomic(__gm__ float* dst, __ubuf__ float* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    DataCopyWithAtomicCom(dst, src, intriParams);
}

#if (__CCE_AICORE__ == 300)
__aicore__ inline void DataCopyWithAtomic(__gm__ int16_t* dst, __ubuf__ int16_t* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    DataCopyWithAtomicCom(dst, src, intriParams);
}

__aicore__ inline void DataCopyWithAtomic(__gm__ int32_t* dst, __ubuf__ int32_t* src, const DataCopyParams& intriParams)
{
    if (!g_isAtomic) {
        return;
    }
    DataCopyWithAtomicCom(dst, src, intriParams);
}
#endif
#endif // ASCENDC_CPU_DEBUG
// BF16
#if __CCE_AICORE__ >= 220 && (!defined(__DAV_M310__))
constexpr uint32_t BF16_TO_FP32_MAN_LEN = 16;
__aicore__ inline bfloat16_t ToBfloat16(const float& fVal)
{
    float fNum = fVal;
    union ToBfloat16Union {
        __aicore__ ToBfloat16Union() {}
        uint16_t val;
        bfloat16_t bNum;
    } bf16Union;
    union FloattoInt32Union {
        __aicore__ FloattoInt32Union() {}
        float ftmp;
        uint32_t uret;
    } int32Union;
    int32Union.ftmp = fNum;
    bf16Union.val = int32Union.uret >> BF16_TO_FP32_MAN_LEN;
    return bf16Union.bNum;
}

__aicore__ inline float ToFloat(const bfloat16_t& bVal)
{
    bfloat16_t bNum = bVal;
    union ToFloatUnion {
        __aicore__ ToFloatUnion() {}
        uint32_t val;
        float fNum;
    } floatUnion;
    union ToUint16Union {
        __aicore__ ToUint16Union() {}
        bfloat16_t uret;
        uint16_t num;
    } u16Union;
    u16Union.uret = bNum;
    floatUnion.val = u16Union.num << BF16_TO_FP32_MAN_LEN;
    return floatUnion.fNum;
}
#endif
} // namespace AscendC
#endif // ASCENDC_MODULE_UTILS_H
