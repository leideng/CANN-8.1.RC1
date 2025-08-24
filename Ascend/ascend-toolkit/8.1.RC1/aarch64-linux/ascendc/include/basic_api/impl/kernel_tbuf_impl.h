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
 * \file kernel_tbuf_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_TBUF_IMPL_H
#define ASCENDC_MODULE_TBUF_IMPL_H

namespace AscendC {
// begin impl of tbuf
template <TPosition pos>
template <typename T>
__aicore__ inline __sync_alias__ LocalTensor<T> TBuf<pos>::Get(uint32_t len)
{
    uint32_t dataLen;
    if constexpr (IsSameType<T, int4b_t>::value) {
        dataLen = len / INT4_TWO;
    } else {
        dataLen = len * sizeof(T);
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((len > 0), { KERNEL_LOG(KERNEL_ERROR, "buffer length is %u, which shoud be larger than 0", len); });
    ASCENDC_DEBUG_ASSERT((dataLen % ONE_BLK_SIZE == 0), "buffer length is %u, which shoud be times of 32 Bytes \n",
        len);
    ASCENDC_ASSERT((dataLen <= bufLen),
                   { KERNEL_LOG(KERNEL_ERROR, "len is %u, max buffer len is %u", dataLen, bufLen); });
#endif
    auto ptr = this->bufStart;
    ptr->dataLen = dataLen;
    TBuffAddr addr;
    addr.logicPos = static_cast<uint8_t>(pos);
    addr.bufferHandle = reinterpret_cast<TBufHandle>(ptr);
    addr.bufferAddr = ptr->address;
    addr.dataLen = ptr->dataLen;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(GetPhyType(pos))].absAddr;
    addr.absAddr = absAddr + addr.bufferAddr;
    AscendCBufGet(addr.logicPos, static_cast<uint8_t>(GetPhyType(pos)), reinterpret_cast<uint64_t>(addr.absAddr), len);
    if (this->bufPoolHandle != 0U) {
        AscendCUpdateTbufPoolStatus(this->bufPoolHandle, false);
        AscendCTBufPoolResetCheck(static_cast<uint8_t>(GetPhyType(pos)),
            reinterpret_cast<uint64_t>(absAddr + ptr->address),
            static_cast<uint64_t>(ptr->dataLen),
            this->bufPoolHandle);
    }
#endif
    LocalTensor<T> tensor;
    tensor.SetAddr(addr);
    return tensor;
}

template <TPosition pos> template <typename T> __aicore__ inline __sync_alias__ LocalTensor<T> TBuf<pos>::Get()
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return Get<T>(bufLen * INT4_TWO);
    } else {
        return Get<T>(bufLen / sizeof(T));
    }
}

template <TPosition pos>
template <typename T>
__aicore__ inline __sync_alias__ LocalTensor<T> TBuf<pos>::GetWithOffset(uint32_t size, uint32_t bufOffset)
{
    auto ptr = this->bufStart;
    ptr->dataLen = size * sizeof(T);
    TBuffAddr addr;
    addr.logicPos = static_cast<uint8_t>(pos);
    addr.bufferHandle = reinterpret_cast<TBufHandle>(ptr);
    addr.bufferAddr = ptr->address + bufOffset;
    addr.dataLen = ptr->dataLen;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(pos));
    addr.absAddr = absAddr + addr.bufferAddr;
#endif
    LocalTensor<T> tensor;
    tensor.SetAddr(addr);
    return tensor;
}

template <TPosition pos> __aicore__ inline void TBuf<pos>::SetTpipeBuf(TBufType* bufStartIn, uint32_t bufLenIn)
{
    this->bufStart = bufStartIn;
    this->bufLen = bufLenIn;
    this->offset = 0;
}

template <TPosition pos> template <typename T> __aicore__ inline void TBuf<pos>::EnQue(const LocalTensor<T>& tensor)
{
    (void)(0);
}

template <TPosition pos> template <typename T> __aicore__ inline LocalTensor<T> TBuf<pos>::DeQue()
{
    return Get<T>();
}

template <TPosition pos>
template <typename T>
__aicore__ inline __sync_alias__ LocalTensor<T> TBuf<pos>::AllocTensor()
{
    return Get<T>();
}

template <TPosition pos> template <typename T> __aicore__ inline void TBuf<pos>::FreeTensor(LocalTensor<T>& tensor)
{
    (void)(0);
}

template <TPosition pos>
template <typename T>
__aicore__ inline TBufState TBuf<pos>::GetState(const LocalTensor<T>& tensor) const
{
    TBufHandle handle = tensor.GetBufferHandle();
    if (handle == nullptr) {
        return TBufState::FREE;
    }
    auto ptr = reinterpret_cast<TBufType*>(handle);
    return ptr->state;
}

template <TPosition pos> __aicore__ inline bool TBuf<pos>::EnQue(TBufHandle buf)
{
    return true;
}

template <TPosition pos> __aicore__ inline TBufHandle TBuf<pos>::DeQue()
{
    return Get();
}

template <TPosition pos> __aicore__ inline TBufHandle TBuf<pos>::AllocBuffer()
{
    return Get();
}

template <TPosition pos> __aicore__ inline void TBuf<pos>::FreeBuffer(TBufHandle buf)
{
    (void)(0);
}

template <TPosition pos> __aicore__ inline TBuffAddr TBuf<pos>::GetBufferAddr(TBufHandle buf)
{
    auto ptr = reinterpret_cast<TBufType*>(buf);
    TBuffAddr addr;
    addr.logicPos = static_cast<uint8_t>(pos);
    addr.bufferHandle = buf;
    addr.bufferAddr = ptr->address;
    addr.dataLen = ptr->dataLen;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    auto absAddr = GetTPipePtr()->g_tpipeImpl.bufPoolBaseAddr_[static_cast<uint8_t>(GetPhyType(pos))].absAddr;
    addr.absAddr = absAddr + addr.bufferAddr;
#endif
    return addr;
}

template <TPosition pos> __aicore__ inline void TBuf<pos>::InitStartBufHandle(
    TBufHandle startBufhandle, uint8_t num, uint32_t len)
{
    static_assert(!isTQue, "InitTBufAddr only support TBuf class");
    ASCENDC_ASSERT((num == 1), { KERNEL_LOG(KERNEL_ERROR, "param num must be one for TBuf, current num is %d", num); });
    auto ptr = reinterpret_cast<TBufType*>(startBufhandle);
    this->bufStart = ptr;
    this->bufLen = len;
    this->offset = 0;
    return;
}

template <TPosition pos> __aicore__ inline TBufHandle TBuf<pos>::Get(uint32_t len)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((len <= bufLen), { KERNEL_LOG(KERNEL_ERROR, "len is %u, max buffer len is %u", len, bufLen); });
#endif
    this->bufStart->dataLen = len;
    return reinterpret_cast<TBufHandle>(this->bufStart);
}

template <TPosition pos> __aicore__ inline TBufHandle TBuf<pos>::Get()
{
    return Get(bufLen);
}

template <TPosition pos> __aicore__ inline uint32_t TBuf<pos>::GetBufLen() const
{
    return bufLen;
}

}
#endif // ASCENDC_MODULE_TBUF_IMPL_H