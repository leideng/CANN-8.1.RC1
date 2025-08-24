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
 * \file kernel_tbufpool_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_TBUFPOOL_IMPL_H
#define ASCENDC_MODULE_TBUFPOOL_IMPL_H

namespace AscendC {
// begin impl of tBufPool
template <TPosition pos, uint32_t bufIDSize>
__aicore__ inline TBufPool<pos, bufIDSize>::TBufPool()
{
    Init();
}

template <TPosition pos, uint32_t bufIDSize>
__aicore__ inline TBufPool<pos, bufIDSize>::~TBufPool()
{
    auto ptr = this->tBufPoolImpl.buf_;
    for (uint8_t i = 0; i < this->tBufPoolImpl.curBufSize_; i++, ptr++) {
        if (ptr->freeBufEvtID != INVALID_TEVENTID) {
            WaitFlagImpl(ptr->freeBufEvt, ptr->freeBufEvtID);
            ptr->freeBufEvtID = INVALID_TEVENTID;
        }
    }
    ResetPool();
};

template <TPosition pos, uint32_t bufIDSize>
__aicore__ inline void TBufPool<pos, bufIDSize>::ResetPool()
{
    tBufPoolImpl.curBufSize_ = 0;
    tBufPoolImpl.startAddr_ = 0;
    tBufPoolImpl.maxAddr_ = 0;
    tBufPoolImpl.maxLen_ = 0;
}

template <TPosition pos, uint32_t bufIDSize>
__aicore__ inline void TBufPool<pos, bufIDSize>::Init()
{
    constexpr auto pool = GetPhyType(pos);
    static_assert((pool == Hardware::L1 || pool == Hardware::UB),
        "TbufPool Position should be one of A1/B1/C1/VECIN/VECOUT/VECCALC");
    ResetPool();
    tBufPoolImpl.isReset_ = true;
}

template <TPosition pos, uint32_t bufIDSize>
__aicore__ inline void TBufPool<pos, bufIDSize>::Reset()
{
    auto ptr = this->tBufPoolImpl.buf_;
    for (uint8_t i = 0; i < this->tBufPoolImpl.curBufSize_; i++, ptr++) {
        if (ptr->freeBufEvtID != INVALID_TEVENTID) {
            WaitFlagImpl(ptr->freeBufEvt, ptr->freeBufEvtID);
            ptr->freeBufEvtID = INVALID_TEVENTID;
        }
    }
    ResetPool();
    tBufPoolImpl.isReset_ = true;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    AscendCUpdateTbufPoolStatus(reinterpret_cast<uint64_t>(&tBufPoolImpl), tBufPoolImpl.isReset_);
#endif
}

template <TPosition pos, uint32_t bufIDSize>
template <class T>
__aicore__ inline bool TBufPool<pos, bufIDSize>::InitBuffer(T &que, uint8_t num, uint32_t len)
{
    static_assert((T::isTQue), "TBufPool::InitBuffer(T& que, uint8_t num, uint32_t len) not supports T as TBuf");
    ASCENDC_ASSERT((len > 0), { KERNEL_LOG(KERNEL_ERROR, "buffer length is %u, which shoud be larger than 0", len); });
    len = (len + ONE_BLK_SIZE - MIN_BLOCK_LEN) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    que.value = num;
    que.bufStart = this->tBufPoolImpl.buf_ + this->tBufPoolImpl.curBufSize_;
    DEBUG_CODE(que.bufLen = num * len);
    ASCENDC_ASSERT(
        (this->tBufPoolImpl.maxAddr_ + num * len <= this->tBufPoolImpl.startAddr_ + this->tBufPoolImpl.maxLen_), {
            KERNEL_LOG(KERNEL_ERROR,
                "Buffer Init length exceeds limit of BufPool. Max Length of BufPool is %u",
                this->tBufPoolImpl.maxLen_);
        });
    auto curPoolAddr = this->tBufPoolImpl.maxAddr_;
    auto ptr = que.bufStart;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    Hardware pool = GetBufferPos(T::srcPosition, T::dstPosition);
    ASCENDC_ASSERT(
        (pool == GetPhyType(pos)), { KERNEL_LOG(KERNEL_ERROR, "buffer pos should be same with pos of TbufPool"); });
    auto bufferInitLen = ConstDefiner::Instance().bufferInitLen;
    ASCENDC_ASSERT((num * len <= bufferInitLen.at(pool)),
        { KERNEL_LOG(KERNEL_ERROR, "buffer size is %d, exceed limits %d", num * len, bufferInitLen.at(pool)); });
    auto bufPos = GetPosition(T::srcPosition, T::dstPosition);
    auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(bufPos));
    AscendCBufInit(static_cast<uint8_t>(bufPos), 0, num, reinterpret_cast<uint64_t>(curPoolAddr + absAddr), len);
    que.SetTBufPoolHandle(reinterpret_cast<uint64_t>(&tBufPoolImpl));
    ASCENDC_ASSERT((curPoolAddr + num * len <= bufferInitLen.at(pool)),
        { KERNEL_LOG(KERNEL_ERROR, "curPoolAddr is %d, limits is %d", curPoolAddr, bufferInitLen.at(pool)); });
#endif
    for (int32_t i = 0; i < num; i++, ptr++) {
        ptr->state = TBufState::FREE;
        ptr->freeBufEvt = T::freeBufEvt;
        ptr->enQueEvtID = INVALID_TEVENTID;
        ptr->freeBufEvtID = INVALID_TEVENTID;
        ptr->address = curPoolAddr;
        ptr->dataLen = len;
        ptr->usertag = -1;
        curPoolAddr += len;
    }
    this->tBufPoolImpl.maxAddr_ = curPoolAddr;
    this->tBufPoolImpl.curBufSize_ += num;
    ASCENDC_ASSERT((this->tBufPoolImpl.curBufSize_ <= QBUFPOOL_MAX_LEN), {
        KERNEL_LOG(KERNEL_ERROR, "buffer size is %d, limits is %d", this->tBufPoolImpl.curBufSize_, QBUFPOOL_MAX_LEN);
    });
    return true;
}

template <TPosition pos, uint32_t bufIDSize>
template <TPosition bufPos>
__aicore__ inline bool TBufPool<pos, bufIDSize>::InitBuffer(TBuf<bufPos> &buf, uint32_t len)
{
    ASCENDC_ASSERT((len > 0), { KERNEL_LOG(KERNEL_ERROR, "buffer length is %u, which shoud be larger than 0", len); });
    len = (len + ONE_BLK_SIZE - MIN_BLOCK_LEN) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    constexpr int32_t bufHandleSize = 1;
    buf.bufStart = this->tBufPoolImpl.buf_ + this->tBufPoolImpl.curBufSize_;
    buf.bufLen = len;
    buf.offset = 0;
    ASCENDC_ASSERT(
        (this->tBufPoolImpl.maxAddr_ + len <= this->tBufPoolImpl.startAddr_ + this->tBufPoolImpl.maxLen_), {
            KERNEL_LOG(KERNEL_ERROR,
                "Buffer Init length exceeds limit of BufPool. Max Length of BufPool is %u",
                this->tBufPoolImpl.maxLen_);
        });
    constexpr auto pool = GetPhyType(bufPos);
    ASCENDC_ASSERT((GetPhyType(bufPos) == GetPhyType(pos)),
        { KERNEL_LOG(KERNEL_ERROR, "buffer pos should be same with pos of TBufPool"); });
    auto curPoolAddr = this->tBufPoolImpl.maxAddr_;
    auto ptr = buf.bufStart;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    auto bufferInitLen = ConstDefiner::Instance().bufferInitLen;
    ASCENDC_ASSERT((len <= bufferInitLen.at(pool)),
        { KERNEL_LOG(KERNEL_ERROR, "len is %u, exceed limits %d", len, bufferInitLen.at(pool)); });
    auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(bufPos));
    AscendCBufInit(static_cast<uint8_t>(bufPos), 1, 1, reinterpret_cast<uint64_t>(curPoolAddr + absAddr), len);
    buf.SetTBufPoolHandle(reinterpret_cast<uint64_t>(&tBufPoolImpl));
#endif
    for (uint8_t i = 0; i < bufHandleSize; i++, ptr++) {
        ptr->state = TBufState::FREE;
        ptr->enQueEvtID = INVALID_TEVENTID;
        ptr->freeBufEvtID = INVALID_TEVENTID;
        ptr->address = curPoolAddr;
        ptr->dataLen = len;
        ptr->usertag = -1;
        curPoolAddr += len;
    }
    ASCENDC_ASSERT((curPoolAddr <= bufferInitLen.at(pool)),
        { KERNEL_LOG(KERNEL_ERROR, "curPoolAddr is %d, exceed limits %d", curPoolAddr, bufferInitLen.at(pool)); });
    this->tBufPoolImpl.maxAddr_ = curPoolAddr;
    this->tBufPoolImpl.curBufSize_ += bufHandleSize;
    ASCENDC_ASSERT((this->tBufPoolImpl.curBufSize_ <= QBUFPOOL_MAX_LEN), {
        KERNEL_LOG(KERNEL_ERROR,
            "current total buffer num is %d, exceed limits %d",
            this->tBufPoolImpl.curBufSize_,
            QBUFPOOL_MAX_LEN);
    });
    return true;
}

template <TPosition pos, uint32_t bufIDSize>
template <class T>
__aicore__ inline bool TBufPool<pos, bufIDSize>::InitBufPool(T &bufPool, uint32_t len)
{
    static_assert(
        (T::isTbufPool), "TBufPool::InitBufPool(T& bufPool, uint32_t len, U& shareBuf) only supports T as TbufPool");
    ASCENDC_ASSERT((len > 0), { KERNEL_LOG(KERNEL_ERROR, "buffer length is %u, which shoud be larger than 0", len); });
    len = (len + ONE_BLK_SIZE - MIN_BLOCK_LEN) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    constexpr auto pool = GetPhyType(T::poolPos);
    bufPool.tBufPoolImpl.startAddr_ = this->tBufPoolImpl.maxAddr_;
    bufPool.tBufPoolImpl.maxAddr_ = bufPool.tBufPoolImpl.startAddr_;
    bufPool.tBufPoolImpl.maxLen_ = len;
    ASCENDC_ASSERT(
        (this->tBufPoolImpl.maxAddr_ + len <= this->tBufPoolImpl.startAddr_ + this->tBufPoolImpl.maxLen_), {
            KERNEL_LOG(KERNEL_ERROR,
                "Buffer Init length exceeds limit of BufPool. Max Length of BufPool is %u",
                this->tBufPoolImpl.maxLen_);
        });
    auto curPoolAddr = this->tBufPoolImpl.maxAddr_;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    auto bufferInitLen = ConstDefiner::Instance().bufferInitLen;
    ASCENDC_ASSERT((len <= bufferInitLen.at(pool)),
        { KERNEL_LOG(KERNEL_ERROR, "buffer size is %d, exceed limits %d", len, bufferInitLen.at(pool)); });
    auto bufPos = T::poolPos;
    auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(bufPos));
    AscendCTBufPoolInit(static_cast<uint8_t>(bufPos),
        reinterpret_cast<uint64_t>(curPoolAddr + absAddr),
        len,
        reinterpret_cast<uint64_t>(&bufPool.tBufPoolImpl));
    AscendCRecordPoolHierarchy(
        reinterpret_cast<uint64_t>(&this->tBufPoolImpl), reinterpret_cast<uint64_t>(&bufPool.tBufPoolImpl));
#endif
    curPoolAddr += len;
    ASCENDC_ASSERT((curPoolAddr <= bufferInitLen.at(pool)),
        { KERNEL_LOG(KERNEL_ERROR, "curPoolAddr is %d, limits is %d", curPoolAddr, bufferInitLen.at(pool)); });
    this->tBufPoolImpl.maxAddr_ = curPoolAddr;
    return true;
}

template <TPosition pos, uint32_t bufIDSize>
template <class T, class U>
__aicore__ inline bool TBufPool<pos, bufIDSize>::InitBufPool(T &bufPool, uint32_t len, U &shareBuf)
{
    static_assert((T::isTbufPool && U::isTbufPool),
        "TBufPool::InitBufPool(T& bufPool, uint32_t len, U& shareBuf) only supports T and U as TBufPool");
    ASCENDC_ASSERT((len > 0), { KERNEL_LOG(KERNEL_ERROR, "buffer length is %u, which shoud be larger than 0", len); });
    len = (len + ONE_BLK_SIZE - MIN_BLOCK_LEN) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    constexpr auto pool = GetPhyType(T::poolPos);
    constexpr auto sharedPool = GetPhyType(U::poolPos);
    ASCENDC_ASSERT((pool == sharedPool),
        { KERNEL_LOG(KERNEL_ERROR, "Position of input bufPool should be same with position of shareBuf"); });
    bufPool.tBufPoolImpl.startAddr_ = shareBuf.tBufPoolImpl.startAddr_;
    bufPool.tBufPoolImpl.maxAddr_ = bufPool.tBufPoolImpl.startAddr_;
    bufPool.tBufPoolImpl.maxLen_ = shareBuf.tBufPoolImpl.maxLen_;
    ASCENDC_ASSERT((len <= shareBuf.tBufPoolImpl.maxLen_), {
        KERNEL_LOG(KERNEL_ERROR,
            "Length of input bufPool should be no longer than length of shareBuf, which is %u",
            shareBuf.tBufPoolImpl.maxLen_);
    });
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    auto bufferInitLen = ConstDefiner::Instance().bufferInitLen;
    ASCENDC_ASSERT((len <= bufferInitLen.at(pool)),
        { KERNEL_LOG(KERNEL_ERROR, "buffer size is %d, exceed limits %d", len, bufferInitLen.at(pool)); });
    auto bufPos = T::poolPos;
    auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(bufPos));
    AscendCTBufPoolInit(static_cast<uint8_t>(bufPos),
        reinterpret_cast<uint64_t>(bufPool.tBufPoolImpl.startAddr_ + absAddr),
        len,
        reinterpret_cast<uint64_t>(&bufPool.tBufPoolImpl));
    AscendCRecordPoolHierarchy(
        reinterpret_cast<uint64_t>(&this->tBufPoolImpl), reinterpret_cast<uint64_t>(&bufPool.tBufPoolImpl));
#endif
    return true;
}
}
#endif // ASCENDC_MODULE_TBUFPOOL_IMPL_H