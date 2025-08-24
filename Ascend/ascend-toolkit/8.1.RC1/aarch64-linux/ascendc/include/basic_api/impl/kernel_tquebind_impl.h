/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file kernel_tquebind_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_TQUEBIND_IMPL_H
#define ASCENDC_MODULE_TQUEBIND_IMPL_H

namespace AscendC {
// begin impl of IsAivTscm used by tquebind
__aicore__ inline constexpr bool IsAivTscm(TPosition src, TPosition dst)
{
#if __CCE_AICORE__ == 220
    if (GetPosition(src, dst) == TPosition::TSCM) {
        return true;
    }
#else
    (void)(src);
    (void)(dst);
#endif
    return false;
}
// begin impl of tquebind
// TQueBind : this is used for off-standard queue
template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline TQueBind<src, dst, depth, mask>::TQueBind()
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    AscendCQueCreate(static_cast<uint8_t>(src), static_cast<uint8_t>(dst), depth);
#endif // ASCENDC_CPU_DEBUG
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline void TQueBind<src, dst, depth, mask>::InitStartBufHandle(
    TBufHandle startBufhandle, uint8_t num, uint32_t len)
{
    static_assert(isTQue, "InitTQueAddr only support TQue class");
    auto ptr = reinterpret_cast<TBufType*>(startBufhandle);
    this->value = num;
    this->bufStart = ptr;
    DEBUG_CODE(this->bufLen = num * len);
    return;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline void TQueBind<src, dst, depth, mask>::InitBufHandle(T* bufPool,
    uint32_t index, TBufHandle bufhandle, uint32_t curPoolAddr, uint32_t len)
{
    (void)(bufPool);
    (void)(index);
    ASCENDC_ASSERT((len > 0), { KERNEL_LOG(KERNEL_ERROR, "buffer length is %u, which shoud be larger than 0", len); });
    len = (len + ONE_BLK_SIZE - MIN_BLOCK_LEN) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    auto ptr = reinterpret_cast<TBufType*>(bufhandle);
    ptr->state = TBufState::FREE;
    ptr->freeBufEvt = freeBufEvt;
    ptr->enQueEvtID = INVALID_TEVENTID;
    ptr->freeBufEvtID = INVALID_TEVENTID;
    ptr->address = curPoolAddr;
    ptr->dataLen = len;
    ptr->usertag = -1;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline __sync_alias__ LocalTensor<T> TQueBind<src, dst, depth, mask>::AllocTensor()
{
    auto buf = AllocBuffer();
    return Buf2Tensor<T>(buf);
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline void TQueBind<src, dst, depth, mask>::FreeTensor(LocalTensor<T>& tensor)
{
    FreeBuffer(tensor.GetBufferHandle());
    return;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline __sync_alias__ bool TQueBind<src, dst, depth, mask>::EnQue(const LocalTensor<T>& tensor)
{
    auto buf = tensor.GetBufferHandle();
    return EnQue(reinterpret_cast<TBufHandle>(buf));
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <TPosition srcUserPos, TPosition dstUserPos, typename T>
__aicore__ inline __sync_alias__ bool TQueBind<src, dst, depth, mask>::EnQue(const LocalTensor<T>& tensor)
{
    auto buf = tensor.GetBufferHandle();
    return EnQue<srcUserPos, dstUserPos>(reinterpret_cast<TBufHandle>(buf));
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <TPosition srcUserPos, TPosition dstUserPos>
__aicore__ inline __sync_alias__ bool TQueBind<src, dst, depth, mask>::EnQue(TBufHandle buf)
{
    static_assert(((srcUserPos == TPosition::GM) || (srcUserPos == TPosition::VECIN) ||
                (srcUserPos == TPosition::VECOUT) || (srcUserPos == TPosition::VECCALC)) &&
                "enque only support src position GM/VECIN/VECOUT/VECCALC currently.");
    static_assert(((dstUserPos == TPosition::GM) || (dstUserPos == TPosition::VECIN) ||
                (dstUserPos == TPosition::VECOUT) || (dstUserPos == TPosition::VECCALC)) &&
                "enque only support dst position GM/VECIN/VECOUT/VECCALC currently.");
    static_assert(!((srcUserPos == TPosition::GM) && (dstUserPos == TPosition::GM)) &&
                "enque src and dst position cannot be GM at the same time.");
    constexpr Hardware srcUserHardType = GetPhyType(srcUserPos);
    constexpr Hardware dstUserHardType = GetPhyType(dstUserPos);
    constexpr HardEvent enQueUserEvt = GetQueEvt(srcUserHardType, dstUserHardType, true, false, false);

    ASCENDC_ASSERT((this->usedCount < depth), {
        KERNEL_LOG(KERNEL_ERROR, "usedCount is %d, which exceed depth limits %d",
            static_cast<int32_t>(usedCount), depth);
    });
    auto ptr = reinterpret_cast<TBufType*>(buf);
    if constexpr (depth == 1) {
        this->que_ = buf;
    } else {
        this->que_[this->tail] = buf;
    }
    this->usedCount++;

    ASCENDC_ASSERT((this->bufStart <= ptr && ptr < this->bufStart + this->bufNum), {
        KERNEL_LOG(KERNEL_ERROR, "ptr is %p, which should be in range [%p, %p)",
            ptr, this->bufStart, this->bufStart + this->bufNum);
    });
    ASCENDC_ASSERT((ptr->state == TBufState::OCCUPIED) || (ptr->state == TBufState::DEQUE), {
        KERNEL_LOG(KERNEL_ERROR, "ptr state is %d, which should be OCCUPIED / DEQUE",
            static_cast<int32_t>(ptr->state));
    });
    DEBUG_CODE(ptr->userEnQueEvt = enQueUserEvt);
    DEBUG_CODE(ptr->state = TBufState::ENQUE);

    // when src and dst both ub, should insert pipe v barrier
    if constexpr (enQueUserEvt == HardEvent::V_V) {
        SetFlag<enQueUserEvt>(0);
        ptr->enQueEvtID = 0;
    } else {
        auto enQueUserEvtID = GetTPipePtr()->AllocEventID<enQueUserEvt>();
        SetFlag<enQueUserEvt>(enQueUserEvtID);
        ptr->enQueEvtID = enQueUserEvtID;
    }
    if constexpr (depth != 1) {
        if (++this->tail >= depth) {
            this->tail = 0;
        }
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufEnque(static_cast<uint8_t>(src), static_cast<uint8_t>(dst), static_cast<uint8_t>(GetPosition(src, dst)),
        reinterpret_cast<uint64_t>(absAddr + ptr->address));
#endif // ASCENDC_CPU_DEBUG
    return true;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline __sync_alias__ bool TQueBind<src, dst, depth, mask>::EnQue(TBufHandle buf)
{
    ASCENDC_ASSERT((this->usedCount < depth), {
        KERNEL_LOG(KERNEL_ERROR, "usedCount is %d, which exceed depth limits %d", static_cast<int32_t>(usedCount),
            depth);
    });
    auto ptr = reinterpret_cast<TBufType*>(buf);
    if constexpr (depth == 1) {
        this->que_ = buf;
    } else {
        this->que_[this->tail] = buf;
    }
    this->usedCount++;
    ASCENDC_ASSERT((this->bufStart <= ptr && ptr < this->bufStart + this->bufNum), {
        KERNEL_LOG(KERNEL_ERROR, "ptr is %p, which should be in range [%p, %p)", ptr, this->bufStart,
            this->bufStart + this->bufNum);
    });
    ASCENDC_ASSERT((ptr->state == TBufState::OCCUPIED) || (ptr->state == TBufState::DEQUE), {
        KERNEL_LOG(KERNEL_ERROR, "ptr state is %d, which should be OCCUPIED / DEQUE", static_cast<int32_t>(ptr->state));
    });
    DEBUG_CODE(ptr->state = TBufState::ENQUE);
    /* Add for TSCM
     * for 220, aiv just send message, no need add this set/wait
     */
#if __CCE_AICORE__ == 220
    // If the AIC is not entered, the AIV does not process any event ID.
    if (g_coreType != AIV || (GetPosition(src, dst) != TPosition::TSCM)) {
        auto enQueEvtID = GetTPipePtr()->AllocEventID<enQueEvt>();
        SetFlag<enQueEvt>(enQueEvtID);
        ptr->enQueEvtID = enQueEvtID;
    }
#else
    auto enQueEvtID = GetTPipePtr()->AllocEventID<enQueEvt>();
    SetFlag<enQueEvt>(enQueEvtID);
    ptr->enQueEvtID = enQueEvtID;
#endif
    if constexpr (depth != 1) {
        if (++this->tail >= depth) {
            this->tail = 0;
        }
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufEnque(static_cast<uint8_t>(src), static_cast<uint8_t>(dst),
        static_cast<uint8_t>(GetPosition(src, dst)), reinterpret_cast<uint64_t>(absAddr + ptr->address));
#endif // ASCENDC_CPU_DEBUG
    return true;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline __sync_alias__ LocalTensor<T> TQueBind<src, dst, depth, mask>::DeQue()
{
    auto buf = DeQue();
    auto ret = Buf2Tensor<T>(buf);
    return ret;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <TPosition srcUserPos, TPosition dstUserPos, typename T>
__aicore__ inline __sync_alias__ LocalTensor<T> TQueBind<src, dst, depth, mask>::DeQue()
{
    auto buf = DeQue<srcUserPos, dstUserPos>();
    auto ret = Buf2Tensor<T>(buf);
    return ret;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline __sync_alias__ TBufHandle TQueBind<src, dst, depth, mask>::DeQue()
{
    TBufHandle buf;
    if constexpr (depth == 1) {
        buf = this->que_;
    } else {
        buf = this->que_[this->head];
    }
    ASCENDC_ASSERT((buf != nullptr), { KERNEL_LOG(KERNEL_ERROR, "buf can not be nullptr"); });
    auto ptr = reinterpret_cast<TBufType*>(buf);

    ASCENDC_ASSERT((ptr->state == TBufState::ENQUE), {
        KERNEL_LOG(KERNEL_ERROR, "ptr state is %d, which can only be ENQUE", static_cast<int32_t>(ptr->state));
    });
    ASCENDC_ASSERT((this->usedCount > 0), {
        KERNEL_LOG(KERNEL_ERROR, "usedCount is %d, which can only larger than 0",
            static_cast<int32_t>(this->usedCount));
    });
    this->usedCount--;
    /* Add for TSCM
     * for 220, aiv just send message, no need add this set/wait
     */
    DEBUG_CODE(ptr->state = TBufState::DEQUE);
#if __CCE_AICORE__ == 220
    if (g_coreType != AIV || (GetPosition(src, dst) != TPosition::TSCM)) {
        if (ptr->enQueEvtID != INVALID_TEVENTID) {
            WaitFlag<enQueEvt>(ptr->enQueEvtID);
            GetTPipePtr()->ReleaseEventID<enQueEvt>(ptr->enQueEvtID);
            ptr->enQueEvtID = INVALID_TEVENTID;
        }
    }
#else
    if (ptr->enQueEvtID != INVALID_TEVENTID) {
        WaitFlag<enQueEvt>(ptr->enQueEvtID);
        GetTPipePtr()->ReleaseEventID<enQueEvt>(ptr->enQueEvtID);
        ptr->enQueEvtID = INVALID_TEVENTID;
    }
#endif
    if constexpr (depth != 1) {
        if (++this->head >= depth) {
            this->head = 0;
        }
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufDeque(static_cast<uint8_t>(src), static_cast<uint8_t>(dst), static_cast<uint8_t>(GetPosition(src, dst)),
        (uint64_t)(absAddr + ptr->address));
#endif // ASCENDC_CPU_DEBUG
    return reinterpret_cast<TBufHandle>(buf);
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <TPosition srcUserPos, TPosition dstUserPos>
__aicore__ inline __sync_alias__ TBufHandle TQueBind<src, dst, depth, mask>::DeQue()
{
    static_assert(((srcUserPos == TPosition::GM) || (srcUserPos == TPosition::VECIN) ||
                (srcUserPos == TPosition::VECOUT) || (srcUserPos == TPosition::VECCALC)) &&
                "DeQue only support src position GM/VECIN/VECOUT/VECCALC currently.");
    static_assert(((dstUserPos == TPosition::GM) || (dstUserPos == TPosition::VECIN) ||
                (dstUserPos == TPosition::VECOUT) || (dstUserPos == TPosition::VECCALC)) &&
                "DeQue only support dst position GM/VECIN/VECOUT/VECCALC currently.");
    static_assert(!((srcUserPos == TPosition::GM) && (dstUserPos == TPosition::GM)) &&
                "DeQue src and dst position cannot be GM at the same time.");
    constexpr Hardware srcUserHardType = GetPhyType(srcUserPos);
    constexpr Hardware dstUserHardType = GetPhyType(dstUserPos);
    constexpr HardEvent deQueUserEvt = GetQueEvt(srcUserHardType, dstUserHardType, true, false, false);

    TBufHandle buf;
    if constexpr (depth == 1) {
        buf = this->que_;
    } else {
        buf = this->que_[this->head];
    }
    ASCENDC_ASSERT((buf != nullptr), {
        KERNEL_LOG(KERNEL_ERROR, "buf can not be nullptr");
    });
    auto ptr = reinterpret_cast<TBufType*>(buf);

    ASCENDC_ASSERT((ptr->state == TBufState::ENQUE), {
        KERNEL_LOG(KERNEL_ERROR, "ptr state is %d, which can only be ENQUE",
            static_cast<int32_t>(ptr->state));
    });
    ASCENDC_ASSERT((this->usedCount > 0), {
        KERNEL_LOG(KERNEL_ERROR, "usedCount is %d, which can only larger than 0",
            static_cast<int32_t>(this->usedCount));
    });
    this->usedCount--;
    ASCENDC_ASSERT((ptr->userEnQueEvt == deQueUserEvt), {
        KERNEL_LOG(KERNEL_ERROR, "EnQue and DeQue Event should be same.");
    });
    DEBUG_CODE(ptr->state = TBufState::DEQUE);
    // when src and dst both ub, should insert pipe v barrier
    if constexpr (deQueUserEvt == HardEvent::V_V) {
        WaitFlag<deQueUserEvt>(0);
        ptr->enQueEvtID = INVALID_TEVENTID;
    } else {
        if (ptr->enQueEvtID != INVALID_TEVENTID) {
            WaitFlag<deQueUserEvt>(ptr->enQueEvtID);
            GetTPipePtr()->ReleaseEventID<deQueUserEvt>(ptr->enQueEvtID);
            ptr->enQueEvtID = INVALID_TEVENTID;
        }
    }

    if constexpr (depth != 1) {
        if (++this->head >= depth) {
            this->head = 0;
        }
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufDeque(static_cast<uint8_t>(src), static_cast<uint8_t>(dst),
        static_cast<uint8_t>(GetPosition(src, dst)), (uint64_t)(absAddr + ptr->address));
#endif // ASCENDC_CPU_DEBUG
    return reinterpret_cast<TBufHandle>(buf);
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline void TQueBind<src, dst, depth, mask>::FreeBuffer(TBufHandle buf)
{
    auto ptr = reinterpret_cast<TBufType*>(buf);
    ASCENDC_ASSERT((this->bufStart <= ptr && ptr < this->bufStart + this->bufNum), {
        KERNEL_LOG(KERNEL_ERROR, "ptr is %p, which should be in range [%p, %p)", ptr, this->bufStart,
            this->bufStart + this->bufNum);
    });
    ASCENDC_ASSERT((ptr->state != TBufState::FREE), {
        KERNEL_LOG(KERNEL_ERROR, "ptr state is %d, which can not be FREE", static_cast<int32_t>(ptr->state));
    });
    if constexpr (!IsAivTscm(src, dst)) {
#if __CCE_AICORE__ < 220
        if constexpr (src == TPosition::C1 || (src == TPosition::CO2 && dst == TPosition::VECIN)) {
            SetFlag<freeBufEvt>(0); // insert pipe_v without eventID
            ASCENDC_ASSERT((ptr->freeBufEvtID == INVALID_TEVENTID),
                           { KERNEL_LOG(KERNEL_ERROR, "freebuf event id can not be -1"); });
        } else {
            ptr->freeBufEvtID = GetTPipePtr()->AllocEventID<freeBufEvt>();
            SetFlag<freeBufEvt>(ptr->freeBufEvtID);
        }
#else
        // in 220 version, event changed from v to M_MTE1 on condition C1 -> C2
        ptr->freeBufEvtID = GetTPipePtr()->AllocEventID<freeBufEvt>();
        SetFlag<freeBufEvt>(ptr->freeBufEvtID);
        if constexpr (enableLoopQueue) {
            ptr->freeBufEvt = freeBufEvt;
        }
#endif
    } else if constexpr (srcHardType == Hardware::GM) {
        if ASCEND_IS_AIC {
            ptr->freeBufEvtID = GetTPipePtr()->AllocEventID<freeBufEvt>();
            SetFlag<freeBufEvt>(ptr->freeBufEvtID);
            if constexpr (enableLoopQueue) {
                ptr->freeBufEvt = freeBufEvt;
            }
        }
    }
    ptr->state = TBufState::FREE;
    this->bufUsedCount--;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufFree(static_cast<uint8_t>(bufferType), static_cast<uint8_t>(GetPosition(src, dst)),
        (uint64_t)(absAddr + ptr->address), static_cast<uint64_t>(ptr->dataLen));
#endif // ASCENDC_CPU_DEBUG
    return;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline TBufHandle TQueBind<src, dst, depth, mask>::AllocBuffer()
{
    DEBUG_CODE(int32_t size = 0);
    ASCENDC_ASSERT((bufNum > 0), {
        KERNEL_LOG(KERNEL_ERROR, "bufNum is %d, which must be larger than 0", static_cast<int32_t>(bufNum));
    });
    TBufType* ret;
    do {
        ret = this->bufStart + this->bufCursor;
        if constexpr (config.bufferNumber != 1) {
            this->bufCursor += 1;
            if (this->bufCursor == this->bufNum) {
                this->bufCursor = 0;
            }
        }
        if (ret->state == TBufState::FREE) {
            ret->state = TBufState::OCCUPIED;
            if constexpr (IsAivTscm(src, dst)) {
                if constexpr (srcHardType == Hardware::UB) {
                    break;
                } else if constexpr (srcHardType == Hardware::GM) {
                    if ASCEND_IS_AIV {
                        break;
                    }
                }
            }
            if (ret->freeBufEvtID != INVALID_TEVENTID) {
                if constexpr (enableLoopQueue) {
                    if (freeBufEvt == ret->freeBufEvt) {
                        WaitFlag<freeBufEvt>(ret->freeBufEvtID);
                        GetTPipePtr()->ReleaseEventID<freeBufEvt>(ret->freeBufEvtID);
                        ret->freeBufEvtID = INVALID_TEVENTID;
                    } else if (freeBufEvt == HardEvent::V_MTE2 && ret->freeBufEvt == HardEvent::MTE3_V) {
                        WaitFlag<HardEvent::MTE3_V>(ret->freeBufEvtID);
                        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(ret->freeBufEvtID);
                        ret->freeBufEvtID = INVALID_TEVENTID;
                        TEventID evtId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
                        SetFlag<HardEvent::MTE3_MTE2>(evtId);
                        WaitFlag<HardEvent::MTE3_MTE2>(evtId);
                        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(evtId);
                    } else if (freeBufEvt == HardEvent::MTE3_V && ret->freeBufEvt == HardEvent::V_MTE2) {
                        WaitFlag<HardEvent::V_MTE2>(ret->freeBufEvtID);
                        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(ret->freeBufEvtID);
                        ret->freeBufEvtID = INVALID_TEVENTID;
                    } else {
                        ASCENDC_ASSERT(false, {
                            KERNEL_LOG(KERNEL_ERROR, "there is something wrong with free buf event");
                        });
                    }
                } else {
                    WaitFlag<freeBufEvt>(ret->freeBufEvtID);
                    GetTPipePtr()->ReleaseEventID<freeBufEvt>(ret->freeBufEvtID);
                    ret->freeBufEvtID = INVALID_TEVENTID;
                }
            }
            break;
        }
        ASCENDC_ASSERT((++size <= this->bufNum), {
            KERNEL_LOG(KERNEL_ERROR, "size is %d, which exceed limits %d", size, static_cast<int32_t>(this->bufNum));
        });
    } while (true);
    this->bufUsedCount++;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    AscendCBufAlloc(static_cast<uint8_t>(bufferType), static_cast<uint8_t>(GetPosition(src, dst)),
        reinterpret_cast<uint64_t>(absAddr + ret->address), static_cast<uint64_t>(ret->dataLen));
    if (this->bufPoolHandle != 0U) {
        AscendCUpdateTbufPoolStatus(this->bufPoolHandle, false);
        AscendCTBufPoolResetCheck(static_cast<uint8_t>(GetPosition(srcPosition, dstPosition)),
            reinterpret_cast<uint64_t>(absAddr + ret->address),
            static_cast<uint64_t>(ret->dataLen),
            this->bufPoolHandle);
    }
#endif // ASCENDC_CPU_DEBUG
    return reinterpret_cast<TBufHandle>(ret);
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline void TQueBind<src, dst, depth, mask>::FreeAllEvent()
{
    auto ptr = this->bufStart;
    for (int i = 0; i < this->bufNum; i++, ptr++) {
        // should be in deque status
        ASCENDC_ASSERT((ptr->enQueEvtID == INVALID_TEVENTID),
                       { KERNEL_LOG(KERNEL_ERROR, "enque event id can not be -1"); });
        if (ptr->freeBufEvtID != INVALID_TEVENTID) {
            WaitFlag<freeBufEvt>(ptr->freeBufEvtID);
            GetTPipePtr()->ReleaseEventID<freeBufEvt>(ptr->freeBufEvtID);
            ptr->freeBufEvtID = INVALID_TEVENTID;
        }
    }
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline void TQueBind<src, dst, depth, mask>::SetTBufPoolHandle(uint64_t bufPoolHandle)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    this->bufPoolHandle = bufPoolHandle;
#else
    (void)(bufPoolHandle);
#endif
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline int32_t TQueBind<src, dst, depth, mask>::GetTensorCountInQue()
{
    return usedCount;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline TBuffAddr TQueBind<src, dst, depth, mask>::GetBufferAddr(TBufHandle buf)
{
    ASCENDC_ASSERT((GetPosition(src, dst) != TPosition::GM), { KERNEL_LOG(KERNEL_ERROR, "buffer pos can not be GM"); });
    auto ptr = reinterpret_cast<TBufType*>(buf);
    ASCENDC_ASSERT((this->bufStart <= ptr && ptr < this->bufStart + this->bufNum), {
        KERNEL_LOG(KERNEL_ERROR, "ptr is %p, which should be in range [%p, %p)", ptr, this->bufStart,
            this->bufStart + this->bufNum);
    });

    TBuffAddr addr;
    addr.logicPos = static_cast<uint8_t>(GetPosition(src, dst));
    addr.bufferHandle = buf;
    addr.bufferAddr = ptr->address;
    addr.dataLen = ptr->dataLen;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    constexpr Hardware bufferType = GetBufferPos(src, dst);
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(bufferType)].absAddr;
    addr.absAddr = absAddr + addr.bufferAddr;
#endif
    return addr;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline TBufState TQueBind<src, dst, depth, mask>::GetState(const LocalTensor<T>& tensor) const
{
    return GetState(tensor.GetBufferHandle());
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline TBufState TQueBind<src, dst, depth, mask>::GetState(const TBufHandle& handle) const
{
    if (handle == nullptr) {
        return TBufState::FREE;
    }
    auto ptr = reinterpret_cast<TBufType*>(handle);
    ASCENDC_ASSERT((this->bufStart <= ptr && ptr < this->bufStart + this->bufNum), {
        KERNEL_LOG(KERNEL_ERROR, "ptr is %p, which should be in range [%p, %p)", ptr, this->bufStart,
            this->bufStart + this->bufNum);
    });
    return ptr->state;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline bool TQueBind<src, dst, depth, mask>::VacantInQue()
{
    return usedCount < depth;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline bool TQueBind<src, dst, depth, mask>::HasTensorInQue()
{
    return usedCount > 0;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
__aicore__ inline bool TQueBind<src, dst, depth, mask>::HasIdleBuffer()
{
    return bufUsedCount < bufNum;
}
template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline __sync_alias__ LocalTensor<T> TQueBind<src, dst, depth, mask>::Buf2Tensor(TBufHandle buf)
{
    TBuffAddr addr = GetBufferAddr(buf);
    LocalTensor<T> tensor;
    tensor.SetAddr(addr);
    return tensor;
}
}
#endif // ASCENDC_MODULE_TQUEBIND_IMPL_H