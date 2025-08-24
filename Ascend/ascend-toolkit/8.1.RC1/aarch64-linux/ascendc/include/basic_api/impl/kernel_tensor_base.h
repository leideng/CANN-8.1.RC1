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
 * \file kernel_tensor_base.h
 * \brief
 */
#ifndef ASCENDC_MODULE_TENSOR_BASE_H
#define ASCENDC_MODULE_TENSOR_BASE_H
#include <cstdint>
#include "kernel_utils.h"
#include "kernel_common.h"

namespace AscendC {
using TBufHandle = uint8_t*;
using TEventID = int8_t;
using TTagType = int32_t;

enum class TBufState : uint8_t {
    FREE = 0,
    OCCUPIED,
    ENQUE,
    DEQUE,
};

struct TBufType {
    TBufState state;
    HardEvent freeBufEvt;
    TEventID enQueEvtID;
    TEventID freeBufEvtID;
    uint32_t address;
    uint32_t dataLen;
    TTagType usertag;
    DEBUG_CODE(HardEvent userEnQueEvt);
};

struct TBuffAddr {
    uint32_t dataLen;
    uint32_t bufferAddr;
    TBufHandle bufferHandle;
    uint8_t logicPos;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    uint8_t* absAddr;
#endif
};

template <typename T> class BaseLocalTensor {
public:
    using PrimType = PrimT<T>;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    void SetAddr(const TBuffAddr& address)
    {
        this->address_ = address;
    }
    [[deprecated("NOTICE: InitBuffer has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
    void InitBuffer(const uint32_t bufferOffset, const uint32_t bufferSize)
    {
        if constexpr (IsSameType<PrimType, int4b_t>::value) {
            ASCENDC_ASSERT((bufferSize != 0),
                    { KERNEL_LOG(KERNEL_ERROR, "InitBuffer bufferSize must be larger than 0."); });

            ASCENDC_ASSERT((bufferOffset % ONE_BLK_SIZE == 0), {
                KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %u, which should be 32Bytes aligned", bufferOffset);
            });
        }

        ASCENDC_ASSERT((TPosition(this->address_.logicPos) != TPosition::GM),
                    { KERNEL_LOG(KERNEL_ERROR, "logicPos can not be gm when init buffer"); });
        auto positionHardMap = ConstDefiner::Instance().positionHardMap;
        ASCENDC_ASSERT((positionHardMap.count(AscendC::TPosition(this->address_.logicPos)) != 0),
                    { KERNEL_LOG(KERNEL_ERROR, "illegal logis pos %d", this->address_.logicPos); });
        if constexpr (IsSameType<PrimType, int4b_t>::value) {
            ASCENDC_ASSERT((bufferOffset + bufferSize * INT4_BIT_NUM / ONE_BYTE_BIT_SIZE <=
            ConstDefiner::Instance().bufferInitLen.at(positionHardMap.at(
                AscendC::TPosition(this->address_.logicPos)))),
                        {
                            KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %d, bufferSize is %d, buffer overflow",
                                bufferOffset, bufferSize);
                        });

            this->address_.absAddr = ConstDefiner::Instance().hardwareCpuBufferMap.at(
                positionHardMap.at(AscendC::TPosition(this->address_.logicPos))) +
                bufferOffset;
            this->address_.dataLen = bufferSize * INT4_BIT_NUM / ONE_BYTE_BIT_SIZE;
        } else {
            ASCENDC_ASSERT((bufferOffset % ONE_BLK_SIZE == 0), {
            KERNEL_LOG(KERNEL_ERROR, "bufferOffset is %u, which should be 32Bytes aligned", bufferOffset);
            });
            ASCENDC_ASSERT((bufferOffset + bufferSize * sizeof(PrimType) <=
                ConstDefiner::Instance().bufferInitLen.at(positionHardMap.at(
                    AscendC::TPosition(this->address_.logicPos)))),
                            {
                                KERNEL_LOG(KERNEL_ERROR,
                                    "bufferOffset is %d, bufferSize is %d, buffer overflow",
                                    bufferOffset, bufferSize);
                            });
            ASCENDC_ASSERT((bufferSize != 0),
                        { KERNEL_LOG(KERNEL_ERROR, "InitBuffer bufferSize must be larger than 0."); });
            this->address_.absAddr = ConstDefiner::Instance().hardwareCpuBufferMap.at(
                positionHardMap.at(AscendC::TPosition(this->address_.logicPos))) +
                bufferOffset;
            this->address_.dataLen = bufferSize * sizeof(PrimType);
        }
    }
#else
    __aicore__ inline void SetAddr(const TBuffAddr& bufferAddr)
    {
        this->address_ = bufferAddr;
    }
    [[deprecated("NOTICE: InitBuffer has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
    __aicore__ inline void InitBuffer(const uint32_t bufferOffset, const uint32_t bufferSize)
    {
        this->address_.bufferAddr = get_imm(0) + bufferOffset;
        if constexpr (IsSameType<PrimType, int4b_t>::value) {
            this->address_.dataLen = bufferSize * INT4_BIT_NUM / ONE_BYTE_BIT_SIZE;
        } else {
            this->address_.dataLen = bufferSize * sizeof(PrimType);
        }
    }
#endif
    __aicore__ inline TBufHandle GetBufferHandle() const
    {
        return address_.bufferHandle;
    }
public:
    TBuffAddr address_;
};

template <typename T> class BaseGlobalTensor {
public:
    using PrimType = PrimT<T>;
    __aicore__ inline void SetAddr(const uint64_t offset)
    {
        if constexpr (IsSameType<PrimType, int4b_t>::value) {
            address_ = address_ + offset / INT4_TWO;
            oriAddress_ = oriAddress_ + offset / INT4_TWO;
        } else {
            address_ = address_ + offset;
            oriAddress_ = oriAddress_ + offset;
        }
    }
public:
    __gm__ PrimType* address_;
    __gm__ PrimType* oriAddress_;
};

template <typename T> class BaseTensor {
};
}
#endif // ASCENDC_MODULE_TPIPE_BASE_H