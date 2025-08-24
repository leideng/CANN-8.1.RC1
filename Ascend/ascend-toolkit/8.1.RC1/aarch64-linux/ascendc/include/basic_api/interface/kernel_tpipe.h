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
 * \file kernel_tpipe.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_QUEUE_H
#define ASCENDC_KERNEL_QUEUE_H
#include "kernel_tpipe_base.h"
#include "kernel_struct_data_copy.h"

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <map>
#include <random>
#include <sstream>
#include <iomanip>
#endif

namespace AscendC {
class TPipe;
template <TPosition src, TPosition dst, int32_t depth, auto mask = 0> class TQueBind {
public:
    __aicore__ inline TQueBind();
    __aicore__ inline void FreeBuffer(TBufHandle buf);
    __aicore__ inline TBuffAddr GetBufferAddr(TBufHandle buf);
    template <typename T> __aicore__ inline __sync_alias__ LocalTensor<T> AllocTensor();
    template <typename T> __aicore__ inline void FreeTensor(LocalTensor<T>& tensor);
    template <typename T> __aicore__ inline bool EnQue(const LocalTensor<T>& tensor);
    __aicore__ inline bool EnQue(TBufHandle buf);
    template <TPosition srcUserPos, TPosition dstUserPos, typename T>
    __aicore__ inline bool EnQue(const LocalTensor<T>& tensor);
    template <typename T> __aicore__ inline LocalTensor<T> DeQue();
    __aicore__ inline TBufHandle DeQue();
    template <TPosition srcUserPos, TPosition dstUserPos, typename T> __aicore__ inline LocalTensor<T> DeQue();
    __aicore__ inline bool VacantInQue();
    __aicore__ inline bool HasTensorInQue();
    __aicore__ inline int32_t GetTensorCountInQue();
    __aicore__ inline bool HasIdleBuffer();
    __aicore__ inline void FreeAllEvent();
    template <typename T> __aicore__ inline TBufState GetState(const LocalTensor<T>& tensor) const;
    __aicore__ inline void InitStartBufHandle(TBufHandle startBufhandle, uint8_t num, uint32_t len);
    template <typename T>
    __aicore__ inline void InitBufHandle(T* bufPool, uint32_t index, TBufHandle bufhandle,
        uint32_t curPoolAddr, uint32_t len);
protected:
    static constexpr TQueConfig config = GetTQueConfig(mask);
    static constexpr bool nd2nz = config.nd2nz;
    static constexpr bool nz2nd = config.nz2nd;
    static constexpr bool scmBlockGroup = config.scmBlockGroup;
    static constexpr bool enableLoopQueue = config.enableLoopQueue;
    static constexpr TPosition srcPosition = src;
    static constexpr TPosition dstPosition = dst;
    static constexpr Hardware srcHardType = GetPhyType(src);
    static constexpr Hardware dstHardType = GetPhyType(dst);
    static constexpr HardEvent enQueEvt = GetQueEvt(srcHardType, dstHardType, true, nd2nz, nz2nd);
    static constexpr HardEvent freeBufEvt = GetQueEvt(srcHardType, dstHardType, false, nd2nz, nz2nd);
    static constexpr int32_t queDepth = depth;
    union {
        uint64_t value;
        struct {
            uint8_t bufNum = 0;
            uint8_t usedCount;
            uint16_t head;
            uint16_t tail;
            uint8_t bufUsedCount;
            uint8_t bufCursor;
        };
    };
    typename TBufHandleAux<depth>::T que_;
    struct TBufType* bufStart;
    DEBUG_CODE(uint32_t bufLen);
    friend class TPipe;
    template <TPosition pos, int32_t d, auto m> friend class TQue;
    template<TPosition pos, uint32_t bufIDSize> friend class TBufPool;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    uint64_t bufPoolHandle{0U};
#endif
private:
    __aicore__ inline void SetTBufPoolHandle(uint64_t bufPoolHandle);
    template <typename T> __aicore__ inline LocalTensor<T> Buf2Tensor(TBufHandle buf);
    __aicore__ inline TBufState GetState(const TBufHandle& handle) const;
    static constexpr bool isTQue = true;
    __aicore__ inline TBufHandle AllocBuffer();
    template <TPosition srcUserPos, TPosition dstUserPos> __aicore__ inline bool EnQue(TBufHandle buf);
    template <TPosition srcUserPos, TPosition dstUserPos> __aicore__ inline TBufHandle DeQue();
};

// Template Args:
// pos - position for queue, suach as VECIN/VECOUT/A1...
// mask - the 0th bit is nd2nz, 1 means data trans from nd format to nz format
//        the 1st bit is nz2nd, 1 means data trans from nz format to nd format
template <TPosition pos, int32_t depth, auto mask = 0>
class TQue : public TQueBind<GetBufferLogicPos(pos, true), GetBufferLogicPos(pos, false), depth, mask> {
public:
    __aicore__ inline TQue() = default;
private:
    friend class TPipe;
    template<TPosition bufPos, uint32_t bufIDSize> friend class TBufPool;
    static constexpr bool isTQue = true;
};

template <TPosition pos = TPosition::LCM> class TBuf : public TQueBind<pos, pos, 0, 0> {
public:
    __aicore__ inline TBuf() = default;
    template <typename T> __aicore__ inline LocalTensor<T> Get();
    template <typename T> __aicore__ inline LocalTensor<T> Get(uint32_t len);
    template <typename T> __aicore__ inline LocalTensor<T> GetWithOffset(uint32_t size, uint32_t bufOffset);
    // inheritance function from Class TQueBind
    template <typename T> __aicore__ inline void EnQue(const LocalTensor<T>& tensor);
    template <typename T> __aicore__ inline LocalTensor<T> DeQue();
    template <typename T> __aicore__ inline LocalTensor<T> AllocTensor();
    template <typename T> __aicore__ inline void FreeTensor(LocalTensor<T>& tensor);
    template <typename T> __aicore__ inline TBufState GetState(const LocalTensor<T>& tensor) const;
    __aicore__ inline bool EnQue(TBufHandle buf);
    __aicore__ inline TBufHandle DeQue();
    __aicore__ inline void FreeBuffer(TBufHandle buf);
    __aicore__ inline TBuffAddr GetBufferAddr(TBufHandle buf);
    __aicore__ inline void InitStartBufHandle(TBufHandle startBufhandle, uint8_t num, uint32_t len);

private:
    __aicore__ inline TBufHandle Get();
    __aicore__ inline TBufHandle Get(uint32_t len);
    __aicore__ inline uint32_t GetBufLen() const;
    __aicore__ inline void SetTpipeBuf(TBufType* bufStartIn, uint32_t bufLenIn);
    template <TPosition posPopBuffer>
    friend __aicore__ inline bool PopStackBuffer(TBuf<posPopBuffer> &popBuffer, TBufType &bufStart);
    __aicore__ inline TBufHandle AllocBuffer();

private:
    struct TBufType* bufStart;
    uint32_t bufLen;
    uint32_t offset;
    friend class TPipe;
    template<TPosition bufPos, uint32_t bufIDSize> friend class TBufPool;
    static constexpr bool isTQue = false;
};

template<TPosition pos, uint32_t bufIDSize = defaultBufIDSize>
class TBufPool {
public:
    static constexpr TPosition poolPos = pos;
public:
    __aicore__ inline TBufPool();
    __aicore__ inline ~TBufPool();
    template <class T> __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len);
    template <TPosition bufPos> __aicore__ inline bool InitBuffer(TBuf<bufPos>& buf, uint32_t len);
    template <class T, class U> __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf);
    template <class T> __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len);
    __aicore__ inline void Reset();
protected:
    TBufPoolImpl<bufIDSize> tBufPoolImpl;
private:
    __aicore__ inline void Init();
    __aicore__ inline void ResetPool();
private:
    friend class TPipe;
    template <TPosition src, TPosition dst, int32_t depth, auto mask> friend class TQueBind;
    template <TPosition bufPos, int32_t depth, auto mask> friend class TQue;
    template <TPosition bufPos> friend class TBuf;
    static constexpr bool isTbufPool = true;
};

#define  EXTERN_IMPL_BUFPOOL(EXT_BUFPOOL, POSITION, BUFID_SIZE)                              \
public:                                                                                      \
    static constexpr AscendC::TPosition poolPos = POSITION;                                  \
    static constexpr int bufIDSize = BUFID_SIZE;                                             \
    static constexpr bool isTbufPool = true;                                                 \
    __aicore__ inline ~EXT_BUFPOOL() {                                                       \
        Reset();                                                                             \
    }                                                                                        \
    __aicore__ inline void Reset() {                                                         \
        auto ptr = this->tBufPoolImpl.buf_;                                                  \
        for (uint8_t i = 0; i < this->tBufPoolImpl.curBufSize_; i++, ptr++) {                \
            if (ptr->freeBufEvtID != AscendC::INVALID_TEVENTID) {                            \
                AscendC::WaitFlagImpl(ptr->freeBufEvt, ptr->freeBufEvtID);                   \
                ptr->freeBufEvtID = AscendC::INVALID_TEVENTID;                               \
            }                                                                                \
        }                                                                                    \
        ResetPool();                                                                         \
    }                                                                                        \
    __aicore__ inline void Init() {                                                          \
        constexpr auto pool = AscendC::GetPhyType(poolPos);                                  \
        static_assert((pool == AscendC::Hardware::L1 || pool == AscendC::Hardware::UB),      \
            "TbufPool Position should be one of A1/B1/C1/VECIN/VECOUT/VECCALC");             \
        ResetPool();                                                                         \
        tBufPoolImpl.isReset_ = true;                                                        \
    }                                                                                        \
    __aicore__ inline AscendC::TBufHandle GetBufHandle(uint8_t offset) {                     \
        return reinterpret_cast<AscendC::TBufHandle>(this->tBufPoolImpl.buf_ + offset);      \
    }                                                                                        \
    __aicore__ inline void SetCurAddr(uint32_t curAddr) {                                    \
        this->tBufPoolImpl.maxAddr_ = curAddr;                                               \
        return;                                                                              \
    }                                                                                        \
    __aicore__ inline uint32_t GetCurAddr() {                                                \
        return this->tBufPoolImpl.maxAddr_;                                                  \
    }                                                                                        \
    __aicore__ inline void SetCurBufSize(uint8_t curBufSize) {                               \
        this->tBufPoolImpl.curBufSize_ = curBufSize;                                         \
        return;                                                                              \
    }                                                                                        \
    __aicore__ inline uint8_t GetCurBufSize() {                                              \
        return this->tBufPoolImpl.curBufSize_;                                               \
    }                                                                                        \
protected:                                                                                   \
    AscendC::TBufPoolImpl<bufIDSize> tBufPoolImpl;                                           \
private:                                                                                     \
    __aicore__ inline void ResetPool() {                                                     \
        tBufPoolImpl.curBufSize_ = 0;                                                        \
        tBufPoolImpl.startAddr_ = 0;                                                         \
        tBufPoolImpl.maxAddr_ = 0;                                                           \
        tBufPoolImpl.maxLen_ = 0;                                                            \
    }                                                                                        \
private:                                                                                     \
    friend class AscendC::TPipe;                                                             \
    template <AscendC::TPosition src, AscendC::TPosition dst, int32_t depth, auto mask>      \
    friend class AscendC::TQueBind;                                                          \
    template <AscendC::TPosition bufPos, int32_t depth, auto mask>                           \
    friend class AscendC::TQue;                                                              \
    template <AscendC::TPosition bufPos> friend class AscendC::TBuf

class TPipe : public TPipeBase {
public:
    __aicore__ inline TPipe();
    __aicore__ inline ~TPipe();
    __aicore__ inline void Init();
    template <class T> __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len);
    template <TPosition pos> __aicore__ inline bool InitBuffer(TBuf<pos>& buf, uint32_t len);
    template <class T> __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len);
    template <class T, class U> __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf);
    template <HardEvent evt> __aicore__ inline TEventID AllocEventID();
    template <HardEvent evt> __aicore__ inline void ReleaseEventID(TEventID id);
    template <HardEvent evt> __aicore__ inline TEventID FetchEventID();
    __aicore__ inline TEventID FetchEventID(HardEvent evt);
    template <TPosition pos, typename T>
    __aicore__ inline LocalTensor<T> GetAbsAddr(int32_t offset, int32_t size) const;
    template <TPosition pos> __aicore__ inline TBuffAddr GetAbsAddr(int32_t offset, int32_t len) const;
    /*
     * brief: these functions are used to use spm buffer;
     * demo case:
     * GlobalTensor<T> workTensor;
     * tpipe.InitSpmBuffer(workTensor, size);
     * LocalTensor<T> calcTensor = tpip.Get<T>(size);
     * // when local buffer is not enough, spill local to spm buffer;
     * tpipe.WriteSpmBuffer(calcTensor, size);
     * // ...
     * // read buffer from spm buffer into local
     * tpipe.ReadSpmBuffer(calcTensor, size);
     */
    template <typename T>
    __aicore__ inline void InitSpmBuffer(const GlobalTensor<T>& workspace, const int32_t bufferSize);
    __aicore__ inline void InitSpmBuffer(const int32_t bufferSize);
    template <typename T>
    __aicore__ inline void WriteSpmBuffer(const LocalTensor<T>& writeLocal, const DataCopyParams& copyParams,
        int32_t writeOffset = 0);
    template <typename T>
    __aicore__ inline void ReadSpmBuffer(const LocalTensor<T>& readLocal, const DataCopyParams& copyParams,
        int32_t readOffset = 0);
    template <typename T>
    __aicore__ inline void WriteSpmBuffer(const LocalTensor<T>& writeLocal, const int32_t writeSize,
        int32_t writeOffset = 0);
    template <typename T>
    __aicore__ inline void ReadSpmBuffer(const LocalTensor<T>& readLocal, const int32_t readSize,
        int32_t readOffset = 0);
    __aicore__ inline void Destroy();
    __aicore__ inline void Reset();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    template <typename T> inline uint64_t GetAbsAddr(const LocalTensor<T>& tensor);
    inline uint8_t* GetBaseAddr(int8_t logicPos);
#endif

protected:
    template <TPosition src, TPosition dst, int32_t depth, auto mask> friend class TQueBind;
    template <TPosition pos, int32_t depth, auto mask> friend class TQue;
    template <TPosition pos> friend class TBuf;
    template<TPosition pos, uint32_t bufIDSize> friend class TBufPool;
    template <TPosition pos> friend __aicore__ inline bool PopStackBuffer(TBuf<pos>& popBuffer, TBufType& bufStart);
    template <typename T, TPosition pos> friend __aicore__ inline bool PopStackBuffer(LocalTensor<T>& popLocal);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    void inline SetBufferCtx(Hardware hard, struct BufPoolExtra* bufPool);
#endif

private:
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    template <typename T> friend inline uint64_t GetAbsAddr(TPipe* tpipe, const LocalTensor<T>& tensor);
#endif
    friend __aicore__ inline void InitShareBufStart(TPipe* tpipe, uint32_t mode, uint32_t* shareLens,
        uint32_t lens, uint8_t subBlockIdx);
    friend __aicore__ inline void InitShareBufEnd(TPipe* tpipe);
    __aicore__ inline void InitSocState() const;
    __aicore__ inline void ResetPool();
    template <class T> __aicore__ inline bool TscmInitBuffer(T& que, uint8_t num, uint32_t len);
    /*
     * brief: these functions are used to get end and queueend addr.
     */
    template <TPosition pos> __aicore__ inline uint64_t GetQueueEndAddress();
};

template<pipe_t src, pipe_t dst>
class TQueSync {
public:
    __aicore__ inline void SetFlag(TEventID id);
    __aicore__ inline void WaitFlag(TEventID id);
};

template <TPosition pos, int32_t depth = 1, auto mask = 0>
using TSCM = TQueBind<pos, TPosition::TSCM, depth, mask>;
} // namespace AscendC

#endif // ASCENDC_KERNEL_QUEUE_H
