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
 * \file mm_allreduce_add_rms_norm_weight_quant.h
 * \brief
 */
#ifndef MM_ALLREDUCE_ADD_RMS_NORM_WEIGHT_QUANT_H
#define MM_ALLREDUCE_ADD_RMS_NORM_WEIGHT_QUANT_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce/common.h"
#include "../matmul_all_reduce/matmul_all_reduce_weight_quant.h"
#include "add_rms_norm_kernel.h"

namespace MatmulAllReduceAddRmsNormImpl {
using namespace AscendC;
using MatmulAllReduceImpl::MatmulAllReduceWeightQuant;
using WeightQuantBatchMatmulV2::QuantType;
template <typename xType, typename wType, typename yType, class mmType>
class MatmulAllReduceAddRmsNormWeightQuant: public MatmulAllReduceWeightQuant<xType, wType, yType, mmType> {
public:
    __aicore__ inline MatmulAllReduceAddRmsNormWeightQuant(
            MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs, MC2TilingHeader *tilingData,
            TPipe *tPipe): MatmulAllReduceWeightQuant<xType, wType, yType, mmType>(
                    addrs, quantAddrs, arnAddrs, tilingData, tPipe) {
        WeightQuantMatmulAllReduceAddRmsNormTilingData *p = (
                WeightQuantMatmulAllReduceAddRmsNormTilingData *)tilingData;
        arnTile_ = &p->addRMSNormTileTilingData;
        arnTail_ = &p->addRMSNormTailTilingData;
        arnTilineKey_ = &p->addRmsNormTilingeKeyData;
    }

    __aicore__ inline void Process() {
#if (ORIG_DTYPE_X1 == DT_BF16)
        this->PreProcForBiasOnVector();
#endif

        this->InnerProcess(false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            this->InnerProcess(true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }

        SyncAll<false>();
        if ASCEND_IS_AIV {
            AddRmsNormKernel op(this->arnAddrs_, this->tPipe_, sizeof(yType), this->paramInTiling_->tileCnt,
                                this->paramInTiling_->tailCnt, &this->hccl_, this->tileInfo_.hcclHandleId,
                                this->tailInfo_.hcclHandleId);
            op.ComputeAddRmsNorm(*arnTile_, *arnTail_, *arnTilineKey_, this->addrs_->workspaceGM);
        }

        SyncAll<false>();
        if (this->notifyFlag_) {
            this->hccl_.Finalize();
        }
    }

private:
    AddRMSNormTilingeKeyData *arnTilineKey_;
    AddRMSNormTilingData *arnTile_;
    AddRMSNormTilingData *arnTail_;
};

#define INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(bTransFlag, quantType, offsetFlag)                          \
    do {                                                                                                    \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantMatmulAllReduceAddRmsNormTilingData, tilingData, tilingGM);  \
        using opType = WEIGH_QUANT_MATMUL_CLASS_NAME<                                                       \
                DTYPE_X1, DTYPE_X2, DTYPE_BIAS_FOR_MC2, DTYPE_Y,                                            \
                false, bTransFlag, quantType, offsetFlag, QuantType::NONE>;                                 \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, nullptr, normOutGM, workspaceGM, normOutGM};                  \
        QuantGmAddrs quantAddrs = {antiquantScaleGM, antiquantOffsetGM, nullptr, nullptr};                  \
        ArnGmAddrs arnAddrs = {residualGM, gammaGM, yGM, normOutGM};                                        \
        MatmulAllReduceAddRmsNormWeightQuant<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType> op(                       \
                &addrs, &quantAddrs, &arnAddrs, (MC2TilingHeader *)&tilingData, &tPipe);                    \
        op.Init();                                                                                          \
        op.Process();                                                                                       \
    } while (0)
}  // namespace MatmulAllReduceAddRmsNormImpl
#endif  // MM_ALLREDUCE_ADD_RMS_NORM_WEIGHT_QUANT_H