/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file simple_broadcast_base.h
 * \brief
 */

#ifndef ASCEND_SIMPLE_BROADCAST_BASE_H
#define ASCEND_SIMPLE_BROADCAST_BASE_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"


using namespace AscendC;


constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t ELEM_ALIGN = BLOCK_SIZE / sizeof(int8_t); // 最小类型是int8，需要保证int8是对齐的

class SimpleBroadcastBase {
public:
    __aicore__ inline SimpleBroadcastBase() {}
    __aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
    {
        if (y == 0){
            return x;
        }
        return (x + y - 1) / y * y;
    }

    __aicore__ inline void Init(SimpleBroadcastTilingData *tilingData)
    {
        // init tiling
        tilingData_ = tilingData;

        // init idx, offset
        nIdx_ = GetBlockIdx() % tilingData_->dimNBlockNum * tilingData_->dimNBlockSize;
        nOffset_ = nIdx_;
        bIdx_ = GetBlockIdx() / tilingData_->dimNBlockNum * tilingData_->dimBBlockSize;
        bOffset_ = bIdx_;

        if (tilingData_->dimBLoop > 1) {
            outerCount_ = tilingData_->dimBLoop * tilingData_->dimN;
            int64_t leftCount = (tilingData_->dimB - bOffset_) * tilingData_->dimN;
            outerCount_ = outerCount_ > leftCount ? leftCount : outerCount_;
            innerCount_ = tilingData_->dimN;

            outerCountAlign_ = RoundUp(tilingData_->dimN, ELEM_ALIGN) * tilingData_->dimBLoop;
            innerCountAlign_ = RoundUp(innerCount_, ELEM_ALIGN);
        } else {
            outerCount_ = tilingData_->dimNLoop;
            int64_t leftCount = (tilingData_->dimN - nOffset_);
            outerCount_ = leftCount < outerCount_ ? leftCount : outerCount_;
            innerCount_ = outerCount_;

            outerCountAlign_ = RoundUp(outerCount_, ELEM_ALIGN);
            innerCountAlign_ = RoundUp(innerCount_, ELEM_ALIGN);
        }
    }

    // 准备下一次搬入，理论数量dimBLoop * dimNLoop
    __aicore__ inline int64_t OuterNext()
    {
        // 复位b方向分次计算计数器
        innerBOffset_ = 0;
        if (tilingData_->dimBLoop > 1) {
            // 一次搬dimBLoop行
            // 判断下一个循环是否还有计算量
            bOffset_ += tilingData_->dimBLoop;
            if (bOffset_ >= tilingData_->dimB || bOffset_ - bIdx_ >= tilingData_->dimBBlockSize) {
                // 完成任务，退出
                return 0;
            }
            // 与全局尾部距离
            int64_t leftDimB = tilingData_->dimB - bOffset_;
            // 与单核尾部距离
            int64_t leftDimBBlock = tilingData_->dimBBlockSize - (bOffset_ - bIdx_);
            leftDimB = leftDimBBlock < leftDimB ? leftDimBBlock : leftDimB;
            int64_t outerDimB = tilingData_->dimBLoop < leftDimB ? tilingData_->dimBLoop : leftDimB;
            // 下个循环的处理数量
            outerCount_ = outerDimB * tilingData_->dimN;
            outerCountAlign_ = outerDimB * RoundUp(tilingData_->dimN, ELEM_ALIGN);
        } else {
            // 一次搬dimNLoop列
            nOffset_ += tilingData_->dimNLoop;
            // 判断N方向是否还有计算量
            if (nOffset_ >= tilingData_->dimN || nOffset_ - nIdx_ >= tilingData_->dimNBlockSize) {
                // 完成任务，切换到下一行
                nOffset_ = nIdx_;
                bOffset_ += 1;
                if (bOffset_ >= tilingData_->dimB || bOffset_ - bIdx_ >= tilingData_->dimBBlockSize) {
                    return 0;
                }
            }
            // 与全局尾部距离
            int64_t leftDimN = tilingData_->dimN - nOffset_;
            // 与单核尾部距离
            int64_t leftDimNBlock = tilingData_->dimNBlockSize - (nOffset_ - nIdx_);
            leftDimN = leftDimNBlock < leftDimN ? leftDimNBlock : leftDimN;
            int64_t outerDimN = tilingData_->dimNLoop < leftDimN ? tilingData_->dimNLoop : leftDimN;
            // 下个循环的处理数量
            outerCount_ = outerDimN;
            innerCount_ = outerCount_;

            outerCountAlign_ = RoundUp(outerDimN, ELEM_ALIGN);
            innerCountAlign_ = outerCountAlign_;
        }
        return outerCount_;
    }

    __aicore__ inline int64_t InnerNext()
    {
        innerBOffset_ += 1;
        if (innerBOffset_ >= tilingData_->dimBLoop) {
            return 0;
        }
        return innerCount_;
    }
    __aicore__ inline void CastFromF16ToI8(const AscendC::LocalTensor<int8_t> &out, const AscendC::LocalTensor<half> &in,
        half quantMin, uint32_t count)
    {
        Maxs(in, in, quantMin, count);
        AscendC::PipeBarrier<PIPE_V>();
        Mins(in, in, (half)127, count); // 127: limit
        AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
#else
        Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
#endif
        AscendC::PipeBarrier<PIPE_V>();
    }
    template <typename T>
    __aicore__ inline void InitBroadcastGlobalTensor(GlobalTensor<T> &gm, GM_ADDR addr)
    {
        gm.SetGlobalBuffer((__gm__ T *)addr, tilingData_->dimB * tilingData_->dimN);
    }

    template <typename T>
    __aicore__ inline void InitNormalGlobalTensor(GlobalTensor<T> &gm, GM_ADDR addr)
    {
        gm.SetGlobalBuffer((__gm__ T *)addr, tilingData_->dimN);
    }

    template <typename T, typename Q>
    __aicore__ inline void InitBroadcastQueue(Q &queue)
    {
        tpipe_.InitBuffer(queue, BUFFER_NUM, outerCountAlign_ * sizeof(T));
    }

    template <typename T, typename Q>
    __aicore__ inline void InitNormalQueue(Q &queue)
    {
        tpipe_.InitBuffer(queue, 1, innerCountAlign_ * sizeof(T));
    }

    template <typename T, typename B>
    __aicore__ inline void InitBroadcastTBuf(B &buf)
    {
        tpipe_.InitBuffer(buf, outerCountAlign_ * sizeof(T));
    }

    template <typename T, typename B>
    __aicore__ inline void InitNormalTBuf(B &buf)
    {
        tpipe_.InitBuffer(buf, innerCountAlign_ * sizeof(T));
    }

    template <typename T, typename Q>
    __aicore__ inline void CopyInBroadcast(GlobalTensor<T> &gm, Q &queue)
    {
        LocalTensor<T> local = queue.template AllocTensor<T>();
        if (tilingData_->dimBLoop > 1) {
            if (tilingData_->dimN * sizeof(T) % BLOCK_SIZE == 0) {
                DataCopy(local, gm[bOffset_ * tilingData_->dimN + nOffset_], outerCount_);
            } else {
#if __CCE_AICORE__ == 220
                DataCopyParams copyParams = { static_cast<uint16_t>(outerCount_ / tilingData_->dimN),
                    static_cast<uint16_t>(tilingData_->dimN * sizeof(T)), 0, 0 };
                DataCopyPad(local, gm[bOffset_ * tilingData_->dimN + nOffset_], copyParams, DataCopyPadParams());
#endif
            }
        } else {
            if (outerCount_ * sizeof(T) % BLOCK_SIZE == 0) {
                DataCopy(local, gm[bOffset_ * tilingData_->dimN + nOffset_], outerCount_);
            } else {
#if __CCE_AICORE__ == 220
                DataCopyParams copyParams = { 1, static_cast<uint16_t>(outerCount_ * sizeof(T)), 0, 0 };
                DataCopyPad(local, gm[bOffset_ * tilingData_->dimN + nOffset_], copyParams, DataCopyPadParams());
#endif
            }
        }

        queue.EnQue(local);
    }

    template <typename T, typename Q>
    __aicore__ inline void CopyInNormal(GlobalTensor<T> &gm, Q &queue)
    {
        LocalTensor<T> local = queue.template AllocTensor<T>();
        if (innerCount_ * sizeof(T) % BLOCK_SIZE == 0) {
            DataCopy(local, gm[nOffset_], innerCount_);
        } else {
#if __CCE_AICORE__ == 220
            DataCopyParams copyParams = { 1, static_cast<uint16_t>(innerCount_ * sizeof(T)), 0, 0 };
            DataCopyPad(local, gm[nOffset_], copyParams, DataCopyPadParams());
#endif
        }

        queue.EnQue(local);
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> GetInnerTensor(LocalTensor<T> &local)
    {
        return local[innerBOffset_ * RoundUp(innerCount_, BLOCK_SIZE / sizeof(T))];
    }

    template <typename T, typename Q>
    __aicore__ inline void CopyOut(GlobalTensor<T> &gm, Q &queue)
    {
        LocalTensor<T> local = queue.template DeQue<T>();
        if (tilingData_->dimBLoop > 1) {
            if (tilingData_->dimN * sizeof(T) % BLOCK_SIZE == 0) {
                DataCopy(gm[bOffset_ * tilingData_->dimN + nOffset_], local, outerCount_);
            } else {
#if __CCE_AICORE__ == 220
                DataCopyParams copyParams = { static_cast<uint16_t>(outerCount_ / tilingData_->dimN),
                    static_cast<uint16_t>(tilingData_->dimN * sizeof(T)), 0, 0 };
                DataCopyPad(gm[bOffset_ * tilingData_->dimN + nOffset_], local, copyParams);
#endif
            }
        } else {
            if (outerCount_ * sizeof(T) % BLOCK_SIZE == 0) {
                DataCopy(gm[bOffset_ * tilingData_->dimN + nOffset_], local, outerCount_);
            } else {
#if __CCE_AICORE__ == 220
                DataCopyParams copyParams = { 1, static_cast<uint16_t>(outerCount_ * sizeof(T)), 0, 0 };
                DataCopyPad(gm[bOffset_ * tilingData_->dimN + nOffset_], local, copyParams);
#endif
            }
        }

        queue.FreeTensor(local);
    }

protected:
    TPipe tpipe_;

    SimpleBroadcastTilingData *tilingData_ = nullptr;
    int64_t bIdx_ = 0; // 起始计算的b方向下标
    int64_t bOffset_ = 0; // 当前计算的b方向下标
    int64_t nIdx_ = 0; // 起始计算的n方向下标
    int64_t nOffset_ = 0; // 当前要计算的n方向下标
    int64_t outerCount_ = 0; // 当前准备搬入的元素数量
    int64_t innerCount_ = 0; // 当前准备计算的元素数量
    int64_t innerBOffset_ = 0; // 一次搬dimBLoop行时，用于计算的b下标计数

    int64_t outerCountAlign_ = 0; // outerCount 32字节对齐
    int64_t innerCountAlign_ = 0; // innerCount 32字节对齐
};

#endif