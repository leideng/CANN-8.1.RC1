/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file add_rms_norm_kernel.h
 * \brief
 */

#ifndef ADD_RMS_NORM_KERNEL_H
#define ADD_RMS_NORM_KERNEL_H
#include "rms_norm_base.h"
#include "add_rms_norm.h"
#include "add_rms_norm_split_d.h"
#include "add_rms_norm_multi_n.h"
#include "add_rms_norm_merge_n.h"
#include "add_rms_norm_single_n.h"

using namespace AscendC;
class AddRmsNormKernel {
public:
    __aicore__ inline AddRmsNormKernel(
            ArnGmAddrs *arnAddrs, TPipe *tPipe, uint32_t dataSize, uint32_t tileCnt, uint32_t tailCnt,
            Hccl<HCCL_SERVER_TYPE_AICPU> *hccl, AscendC::HcclHandle tileHandleId, AscendC::HcclHandle tailHandleId):
        arnAddrs_(arnAddrs), tPipe_(tPipe), dataSize_(dataSize), tileCnt_(tileCnt), tailCnt_(tailCnt), hccl_(hccl),
        tileHandleId_(tileHandleId), tailHandleId_(tailHandleId) {}

    __aicore__ inline void ComputeAddRmsNorm(AddRMSNormTilingData &addRMSNormTileTilingData,
                                             AddRMSNormTilingData &addRMSNormTailTilingData,
                                             AddRMSNormTilingeKeyData &addRmsNormTilingeKeyData, GM_ADDR rcvCntGM) {
        uint32_t lastCnt = 0;
        uint32_t addRmsNormCount = 1;

        while (true) {
            if (GetBlockIdx() == 0) {
                const uint32_t curCnt = GetCurFinishedCnt();
                if (curCnt <= lastCnt) {
                    continue;
                }
                *rcvCntGM = curCnt;
                dcci(reinterpret_cast<__gm__ int64_t *>(rcvCntGM), cache_line_t::SINGLE_CACHE_LINE,
                     dcci_dst_t::CACHELINE_OUT);
            }
            SyncAll();
            dcci(reinterpret_cast<__gm__ int64_t *>(rcvCntGM), cache_line_t::SINGLE_CACHE_LINE,
                 dcci_dst_t::CACHELINE_OUT);
            lastCnt = *rcvCntGM;
            if (lastCnt >= addRmsNormCount) {
                uint32_t rcvCntTile = lastCnt;
                if (tailCnt_ != 0U && lastCnt > tileCnt_) {
                    rcvCntTile = tileCnt_;
                }
                if (tailCnt_ == 0U || (tailCnt_ != 0U && addRmsNormCount <= tileCnt_)) {
                    ComputeAddRmsNormInner(addRMSNormTileTilingData, addRmsNormTilingeKeyData.ARNKeyTile,
                        addRmsNormTilingeKeyData.ARNBlockDimTile, rcvCntTile, addRmsNormCount);
                }

                if (tailCnt_ != 0U && addRmsNormCount > tileCnt_ && addRmsNormCount <= lastCnt) {
                    ComputeAddRmsNormInner(addRMSNormTailTilingData, addRmsNormTilingeKeyData.ARNKeyTail,
                        addRmsNormTilingeKeyData.ARNBlockDimTail, lastCnt, addRmsNormCount);
                }
            }
            if (lastCnt >= (tileCnt_+ tailCnt_)) {
                break;
            }
        }
    }

private:

    #define INVOKE_ARN_OP_IMPL(templateClass, dType)                                            \
        do {                                                                                    \
            templateClass<dType> op;                                                            \
            op.Init(arnAddrs_->gammaGM, rmsTilingData, tPipe_, blockDim);                       \
            op.ComputeProcess(arnAddrs_->normOutGM, arnAddrs_->residualGM, arnAddrs_->yGM,      \
                              rmsTilingData, addRmsNormCount, rcvCnt);                          \
        } while (0)

    __aicore__ inline void AddRmsNorm(AddRMSNormTilingData &rmsTilingData, uint32_t keyTile, uint32_t blockDim,
                                      uint32_t rcvCnt, uint32_t addRmsNormCount) {
        if (GetBlockIdx() >= blockDim) {
            return;
        }

        tPipe_->Reset();
        if (keyTile == 10) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNorm, half);
        } else if (keyTile == 30) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNorm, bfloat16_t);
        } else if (keyTile == 11) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormSplitD, half);
        } else if (keyTile == 31) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormSplitD, bfloat16_t);
        } else if (keyTile == 12) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormMergeN, half);
        } else if (keyTile == 32) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormMergeN, bfloat16_t);
        } else if (keyTile == 13) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormSingleN, half);
        } else if (keyTile == 33) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormSingleN, bfloat16_t);
        } else if (keyTile == 14) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormMultiN, half);
        }
    }

    __aicore__ inline void ComputeAddRmsNormInner(AddRMSNormTilingData &addRMSNormTilingData, uint32_t ARNKey,
                                                  uint32_t ARNBlockDim, uint32_t rcvCnt, uint32_t &addRmsNormCount) {
        uint64_t offset = CalcShapeOffset(dataSize_, addRMSNormTilingData.num_row, addRMSNormTilingData.num_col);
        uint32_t cnt = rcvCnt - addRmsNormCount + 1;

        AddRmsNorm(addRMSNormTilingData, ARNKey, ARNBlockDim, rcvCnt, addRmsNormCount);

        arnAddrs_->normOutGM += offset * cnt;
        arnAddrs_->yGM += offset * cnt;
        arnAddrs_->residualGM += offset * cnt;
        addRmsNormCount += cnt;
    }

    __aicore__ inline uint32_t GetCurFinishedCnt() {
        const int32_t tileCnt = hccl_->Query(tileHandleId_);
        if (tailCnt_ == 0U) {
            return tileCnt;
        }
        return tileCnt + hccl_->Query(tailHandleId_);
    }

    ArnGmAddrs *arnAddrs_;
    TPipe *tPipe_;
    uint32_t dataSize_;
    uint32_t tileCnt_;
    uint32_t tailCnt_;
    Hccl<HCCL_SERVER_TYPE_AICPU> *hccl_;
    AscendC::HcclHandle tileHandleId_, tailHandleId_;
};
#endif