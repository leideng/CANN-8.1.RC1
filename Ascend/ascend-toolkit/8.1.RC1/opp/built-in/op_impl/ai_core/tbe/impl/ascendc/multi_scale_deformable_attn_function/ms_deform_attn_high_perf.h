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
 * \file ms_deform_attn_high_perf.h
 * \brief
 */

#ifndef MS_DEFORM_ATTN_HIGH_PERF_H_
#define MS_DEFORM_ATTN_HIGH_PERF_H_


#include "kernel_operator.h"

using namespace AscendC;

template<int32_t num_points, int32_t embed_dims>
class KernelMultiScaleDeformableAttnOpt {
public:
    __aicore__ inline KernelMultiScaleDeformableAttnOpt() = delete;

    __aicore__ inline KernelMultiScaleDeformableAttnOpt(GM_ADDR value, GM_ADDR valueSpatialShapes,
        GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
        const MultiScaleDeformableAttnFunctionTilingData* tilingData, TPipe* pipe)
        : pipe_(pipe), blkIdx_(GetBlockIdx())
    {
        InitTiling(tilingData);
        InitTask();
        InitGM(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output);
        InitBuffer();
        InitEvent();

        SetVectorMask<float>(FULL_MASK, FULL_MASK);
        SetAtomicNone();
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTask()
    {
        uint32_t avgTasks = numQueries_ / coreNum_;
        uint32_t remainTasks = numQueries_ % coreNum_;
        startOffset_ = avgTasks * blkIdx_ + (blkIdx_ < remainTasks ? blkIdx_ : remainTasks);
        endOffset_ = startOffset_ + avgTasks + (blkIdx_ < remainTasks ? 1 : 0);
    }

    __aicore__ inline void InitTiling(const MultiScaleDeformableAttnFunctionTilingData* tilingData)
    {
        batchSize_ = tilingData->batchSize;
        numKeys_ = tilingData->numKeys;
        numHeads_ = tilingData->numHeads;
        embedDims_ = embed_dims;
        numLevels_ = tilingData->numLevels;
        numQueries_ = tilingData->numQueries;
        numPoints_ = tilingData->numPoints;
        coreNum_ = tilingData->coreNum;
        pointLoops_ = tilingData->pointLoops;
        realLevels_ = tilingData->realLevels;

        oneQueryNum_ = realLevels_ * numHeads_ * numPoints_;

        alignedNumPoints_ = AlignUp(num_points, B32_DATA_NUM_PER_BLOCK);
        alignedOneHeadNum_ = numLevels_ * alignedNumPoints_;
        alignedOneQueryNum_ = AlignUp(numHeads_ * alignedOneHeadNum_, B32_DATA_NUM_PER_REPEAT);
        alignedEmbedDims_ = AlignUp(embedDims_, B32_DATA_NUM_PER_BLOCK);
        alignedCornerEmbedDims_ = AlignUp(Four * num_points * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);
        alignedHeadEmbedDims_ = AlignUp(numHeads_ * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);

        embedBlk_ = alignedEmbedDims_ / B32_DATA_NUM_PER_BLOCK;
        outDims_ = numHeads_ * embedDims_;
        outBlk_ = numHeads_ * embedBlk_;
        queryBlk_ = alignedOneQueryNum_ / B32_DATA_NUM_PER_BLOCK;
        rptTimes_ = alignedOneQueryNum_ / B32_DATA_NUM_PER_REPEAT;
        valRptTimes4_ = alignedCornerEmbedDims_ / B32_DATA_NUM_PER_REPEAT;
        valRptTimes2_ = DivCeil(TWO * num_points * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);
        valRptTimes1_ = DivCeil(num_points * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);
        outRptTimes_ = alignedHeadEmbedDims_ / B32_DATA_NUM_PER_REPEAT;

        if (num_points == Eight && pointLoops_ == 1) {
            cpSampleParams_.blockLen = DivCeil(numLevels_ * numHeads_ * num_points, B32_DATA_NUM_PER_BLOCK);
            cpDoubleSampleParams_.blockLen = DivCeil(TWO * numLevels_ * numHeads_ * num_points, B32_DATA_NUM_PER_BLOCK);
        } else {
            cpSampleParams_.blockCount = numLevels_ * numHeads_;
            cpSampleParams_.blockLen = num_points * B32_BYTE_SIZE;
            cpSampleParams_.srcStride = (numPoints_ - num_points) * B32_BYTE_SIZE;
            cpDoubleSampleParams_.blockCount = numLevels_ * numHeads_;
            cpDoubleSampleParams_.blockLen = TWO * num_points * B32_BYTE_SIZE;
            cpDoubleSampleParams_.srcStride = TWO * (numPoints_ - num_points) * B32_BYTE_SIZE;
            cpDoubleSampleParams_.dstStride = num_points == Eight ? 0 : 1;
        }

        cpOneValParams_.blockLen = embedBlk_;
        cpDoubleValParams_.blockLen = embedBlk_;
        cpDoubleValParams_.srcStride = outBlk_ - embedBlk_;
        cpDoubleValParams_.dstStride = TWO * num_points * embedBlk_ - embedBlk_;

        gatherParams_.repeatTimes = rptTimes_ * TWO;

        dstRptStride_ = num_points * embedBlk_;
    }

    __aicore__ inline void InitGM(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output)
    {
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value));
        locationGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(samplingLocations));
        attentionWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attentionWeights));

        valueSpatialShapesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueSpatialShapes));
        valueLevelStartIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueLevelStartIndex));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(shapeQue_, AlignUp(numLevels_ * TWO, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(offsetQue_, AlignUp(numLevels_, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(locationQue_, Four * alignedOneQueryNum_ * B32_BYTE_SIZE); // x, y
        pipe_->InitBuffer(attentionWeightsQue_, alignedOneQueryNum_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(valueQue_, TWO * alignedCornerEmbedDims_ * B32_BYTE_SIZE); // TWO for double buffer
        pipe_->InitBuffer(outputQue_, TWO * alignedHeadEmbedDims_ * B32_BYTE_SIZE);

        pipe_->InitBuffer(shapeBrcBuf_, TWO * alignedOneQueryNum_ * B32_BYTE_SIZE); // w, h
        pipe_->InitBuffer(locIntBuf_, Four * alignedOneQueryNum_ * B32_BYTE_SIZE);   // x0, y0, x1, y1
        pipe_->InitBuffer(locFloatBuf_, Four * alignedOneQueryNum_ * B32_BYTE_SIZE); // lw, lh
        pipe_->InitBuffer(weightBuf_, Four * alignedOneQueryNum_ * B32_BYTE_SIZE);   // w1-w4
        pipe_->InitBuffer(cornerWeightBuf_, Four * alignedNumPoints_ * B32_BYTE_SIZE);
        // NOTE: cornerWeightBrcBuf must be at the tail of ub
        pipe_->InitBuffer(cornerWeightBrcBuf_, Four * num_points * alignedEmbedDims_ * B32_BYTE_SIZE);
    }

    __aicore__ inline void InitEvent()
    {
        calEvt_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        copyEvt_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
    }

    __aicore__ inline void PrepareShape(
        const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, LocalTensor<float>& shapeBrc);

    __aicore__ inline void CopyInSample(const LocalTensor<float>& location, const LocalTensor<float>& attentionWeight,
        uint32_t batch, uint32_t query, uint32_t pl);

    __aicore__ inline void ComputeLocation(const LocalTensor<float>& location, const LocalTensor<float>& shapes,
        const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat);

    __aicore__ inline void ComputeWeight(const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat,
        const LocalTensor<float>& weight, const LocalTensor<float>& attentionWeight);

    __aicore__ inline void ComputeBilinearInterpolation(const LocalTensor<int32_t>& shapes,
        const LocalTensor<int32_t>& offset, const LocalTensor<int32_t>& locInt, const LocalTensor<float>& value,
        const LocalTensor<float>& weight, const LocalTensor<float>& cornerWeight,
        const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& output);

private:
    TPipe* pipe_;
    GlobalTensor<float> valueGm_, locationGm_, attentionWeightsGm_, outputGm_;
    GlobalTensor<int32_t> valueSpatialShapesGm_, valueLevelStartIndexGm_;

    TBuf<TPosition::VECCALC> locationQue_, attentionWeightsQue_, shapeQue_, offsetQue_, valueQue_;
    TBuf<TPosition::VECCALC> outputQue_;

    TBuf<TPosition::VECCALC> locIntBuf_, locFloatBuf_, shapeBrcBuf_, weightBuf_, cornerWeightBuf_, cornerWeightBrcBuf_;

    int32_t blkIdx_;

    uint32_t batchSize_, numKeys_, numHeads_, embedDims_, outDims_, numLevels_, numQueries_, numPoints_, coreNum_,
        pointLoops_, realLevels_;
    uint32_t startOffset_, endOffset_;
    uint32_t alignedNumPoints_, alignedOneHeadNum_, alignedOneQueryNum_, alignedEmbedDims_, alignedCornerEmbedDims_,
        alignedHeadEmbedDims_;
    uint32_t oneQueryNum_;
    uint16_t queryBlk_, embedBlk_, outBlk_, dstRptStride_;
    uint16_t rptTimes_, outRptTimes_, valRptTimes4_, valRptTimes2_, valRptTimes1_;
    uint32_t TWO = 2, Four = 4, Eight = 8;

    TEventID calEvt_, copyEvt_;

    uint32_t baseSrcOffset_, baseDstOffset_, srcOffset_;

    DataCopyParams cpOneValParams_, cpDoubleValParams_ {2, 0, 0, 0}, cpSampleParams_ {1, 0, 0, 0},
        cpDoubleSampleParams_ {1, 0, 0, 0};
    GatherMaskParams gatherParams_;
};

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points, embed_dims>::PrepareShape(
    const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, LocalTensor<float>& shapeBrc)
{
    DataCopy(shapes, valueSpatialShapesGm_,
        {1, static_cast<uint16_t>(DivCeil(2 * numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
    DataCopy(
        offset, valueLevelStartIndexGm_, {1, static_cast<uint16_t>(DivCeil(numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
    SetFlag<HardEvent::MTE2_V>(copyEvt_);
    WaitFlag<HardEvent::MTE2_V>(copyEvt_);
    // broadcast to [head*level, 8]
    for (uint32_t k = 0; k < 2; ++k) {
        for (uint32_t i = 0; i < numLevels_; ++i) {
            shapeBrc.SetValue(i + k * alignedOneQueryNum_, shapes.GetValue(2 * i + 1 - k));
        }
        Brcb(shapeBrc[k * alignedOneQueryNum_], shapeBrc[k * alignedOneQueryNum_], 1, {1, 8});
        Copy<float, false>(shapeBrc[k * alignedOneQueryNum_ + numLevels_ * 8], shapeBrc[k * alignedOneQueryNum_],
            MASK_PLACEHOLDER, numHeads_ - 1, {1, 1, static_cast<uint16_t>(numLevels_), 0});
    }
}

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points, embed_dims>::CopyInSample(
    const LocalTensor<float>& location, const LocalTensor<float>& attentionWeight, uint32_t batch, uint32_t query,
    uint32_t pl)
{
    uint32_t sampleOffset = (batch * numQueries_ + query) * oneQueryNum_;
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    if (num_points == 8 && pointLoops_ == 1) {
        DataCopy(location, locationGm_[sampleOffset * 2], cpDoubleSampleParams_);
        DataCopy(attentionWeight, attentionWeightsGm_[sampleOffset], cpSampleParams_);
    } else {
        DataCopyPad(location, locationGm_[sampleOffset * 2 + pl * num_points * 2], cpDoubleSampleParams_, {});
        DataCopyPad(attentionWeight, attentionWeightsGm_[sampleOffset + pl * num_points], cpSampleParams_, {});
    }

    SetFlag<HardEvent::MTE2_V>(copyEvt_);
}

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points, embed_dims>::ComputeLocation(
    const LocalTensor<float>& location, const LocalTensor<float>& shapes, const LocalTensor<int32_t>& locInt,
    const LocalTensor<float>& locFloat)
{
    uint64_t cnt;
    WaitFlag<HardEvent::MTE2_V>(copyEvt_);

    GatherMask(location, location[2 * alignedOneQueryNum_], 1, false, MASK_PLACEHOLDER, gatherParams_, cnt);
    GatherMask(location[alignedOneQueryNum_], location[2 * alignedOneQueryNum_], 2, false, MASK_PLACEHOLDER,
        gatherParams_, cnt);
    SetVectorMask<float>(FULL_MASK, FULL_MASK);

    Mul<float, false>(location, location, shapes, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 1, 8, 8, 8});
    Adds<float, false>(locFloat, location, 0.5f, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    Cast<int32_t, float, false>(locInt, locFloat, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
}

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points, embed_dims>::ComputeWeight(
    const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat, const LocalTensor<float>& weight,
    const LocalTensor<float>& attentionWeight)
{
    Cast<float, int32_t, false>(
        locFloat[2 * alignedOneQueryNum_], locInt, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    Sub<float, false>(locFloat, locFloat, locFloat[2 * alignedOneQueryNum_], MASK_PLACEHOLDER, 2 * rptTimes_,
        {1, 1, 1, 8, 8, 8}); // lh, lw
    Mul<float, false>(weight[3 * alignedOneQueryNum_], locFloat, locFloat[alignedOneQueryNum_], MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8}); // lh * lw
    Duplicate<float, false>(weight, 1.f, MASK_PLACEHOLDER, rptTimes_, 1, 8);
    Sub<float, false>(weight, weight, locFloat, MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weight, weight, locFloat[alignedOneQueryNum_], MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Add<float, false>(weight, weight, weight[3 * alignedOneQueryNum_], MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weight[alignedOneQueryNum_], locFloat[alignedOneQueryNum_], weight[3 * alignedOneQueryNum_],
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weight[2 * alignedOneQueryNum_], locFloat, weight[3 * alignedOneQueryNum_], MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8});

    Mul<float, false>(weight, weight, attentionWeight, MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[alignedOneQueryNum_], weight[alignedOneQueryNum_], attentionWeight, MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[2 * alignedOneQueryNum_], weight[2 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[3 * alignedOneQueryNum_], weight[3 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
}

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points, embed_dims>::ComputeBilinearInterpolation(
    const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, const LocalTensor<int32_t>& locInt,
    const LocalTensor<float>& value, const LocalTensor<float>& weight, const LocalTensor<float>& cornerWeight,
    const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& output)
{
    uint8_t ping = 0;

#pragma bisheng auto_sync parallel
    for (uint32_t head = 0; head < numHeads_; ++head) {
        uint32_t valueOffset = (baseSrcOffset_ + head) * embedDims_;
        uint32_t outOffset = head * alignedEmbedDims_;

        for (uint32_t level = 0; level < numLevels_; ++level) {
            int32_t h = shapes.GetValue(level * 2);
            int32_t w = shapes.GetValue(level * 2 + 1);
            srcOffset_ = valueOffset + offset.GetValue(level) * outDims_;

            uint32_t sx = head * alignedOneHeadNum_ + level * alignedNumPoints_;
            uint32_t sy = sx + alignedOneQueryNum_;

            uint32_t pingOffset = ping * alignedCornerEmbedDims_;
            WaitFlag<HardEvent::V_MTE2>(ping);

            for (uint32_t point = 0; point < num_points; ++point) {
                int32_t y1 = locInt.GetValue(point + sy);
                int32_t x1 = locInt.GetValue(point + sx);
                int32_t y0 = y1 - 1;
                int32_t x0 = x1 - 1;

                if (0 <= y0 && y0 < h) {
                    if (0 < x1 && x1 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y0 * w + x0) * outDims_], cpDoubleValParams_);
                    } else if (0 <= x0 && x0 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y0 * w + x0) * outDims_], cpOneValParams_);
                    } else if (0 <= x1 && x1 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_ + 2 * num_points * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y0 * w + x1) * outDims_], cpOneValParams_);
                    }
                }
                if (0 <= y1 && y1 < h) {
                    if (0 < x1 && x1 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_ + num_points * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y1 * w + x0) * outDims_], cpDoubleValParams_);
                    } else if (0 <= x0 && x0 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_ + num_points * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y1 * w + x0) * outDims_], cpOneValParams_);
                    } else if (0 <= x1 && x1 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_ + 3 * num_points * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y1 * w + x1) * outDims_], cpOneValParams_);
                    }
                }
            }
            SetFlag<HardEvent::MTE2_V>(copyEvt_);

            // broadcast to [4*8, embedDims_]
            Copy<float, false>(cornerWeight, weight[sx], MASK_PLACEHOLDER, 1, {1, queryBlk_, 8, 8});
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < 4; ++i) {
                Brcb(cornerWeightBrc[i * num_points * alignedEmbedDims_], cornerWeight[i * alignedNumPoints_], 1,
                    {embedBlk_, dstRptStride_});
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 1; i < embedBlk_; ++i) {
                Copy<float, false>(cornerWeightBrc[i * B32_DATA_NUM_PER_BLOCK], cornerWeightBrc, MASK_PLACEHOLDER, 4,
                    {embedBlk_, embedBlk_, dstRptStride_, dstRptStride_});
            }

            WaitFlag<HardEvent::MTE2_V>(copyEvt_);

            PipeBarrier<PIPE_V>();
            Mul<float, false>(cornerWeightBrc, value[pingOffset], cornerWeightBrc, MASK_PLACEHOLDER, valRptTimes4_,
                {1, 1, 1, 8, 8, 8});

            PipeBarrier<PIPE_V>();
            Duplicate<float, false>(value[pingOffset], 0.f, MASK_PLACEHOLDER, valRptTimes4_, 1, 8);
            SetFlag<HardEvent::V_MTE2>(ping);
            ping = 1 - ping;

            Add<float, false>(cornerWeightBrc, cornerWeightBrc, cornerWeightBrc[2 * num_points * alignedEmbedDims_],
                MASK_PLACEHOLDER, valRptTimes2_, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            Add<float, false>(cornerWeightBrc, cornerWeightBrc, cornerWeightBrc[num_points * alignedEmbedDims_],
                MASK_PLACEHOLDER, valRptTimes1_, {1, 1, 1, 8, 8, 8});

            SetVectorMask<float>(0, (1UL << embedDims_) - 1);
            if (num_points == 8) {
                PipeBarrier<PIPE_V>();
                Add<float, false>(cornerWeightBrc, cornerWeightBrc, cornerWeightBrc[4 * alignedEmbedDims_],
                    MASK_PLACEHOLDER, 4,
                    {1, 1, 1, static_cast<uint8_t>(embedBlk_), static_cast<uint8_t>(embedBlk_),
                        static_cast<uint8_t>(embedBlk_)});
            }
            if (num_points >= 4) {
                PipeBarrier<PIPE_V>();
                Add<float, false>(cornerWeightBrc, cornerWeightBrc, cornerWeightBrc[2 * alignedEmbedDims_],
                    MASK_PLACEHOLDER, 2,
                    {1, 1, 1, static_cast<uint8_t>(embedBlk_), static_cast<uint8_t>(embedBlk_),
                        static_cast<uint8_t>(embedBlk_)});
            }
            if (num_points >= 2) {
                PipeBarrier<PIPE_V>();
                Add<float, false>(cornerWeightBrc, cornerWeightBrc, cornerWeightBrc[alignedEmbedDims_],
                    MASK_PLACEHOLDER, 1,
                    {1, 1, 1, static_cast<uint8_t>(embedBlk_), static_cast<uint8_t>(embedBlk_),
                        static_cast<uint8_t>(embedBlk_)});
            }
            if (num_points >= 1) {
                PipeBarrier<PIPE_V>();
                Add<float, false>(output[outOffset], output[outOffset], cornerWeightBrc, MASK_PLACEHOLDER, 1,
                    {1, 1, 1, static_cast<uint8_t>(embedBlk_), static_cast<uint8_t>(embedBlk_),
                        static_cast<uint8_t>(embedBlk_)});
            }
            SetVectorMask<float>(FULL_MASK, FULL_MASK);
        }
    }
}


template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points, embed_dims>::Process()
{
    LocalTensor<float> location = locationQue_.Get<float>();
    LocalTensor<float> attentionWeight = attentionWeightsQue_.Get<float>();
    LocalTensor<int32_t> shapes = shapeQue_.Get<int32_t>();
    LocalTensor<int32_t> offset = offsetQue_.Get<int32_t>();
    LocalTensor<float> value = valueQue_.Get<float>();
    LocalTensor<float> cornerWeight = cornerWeightBuf_.Get<float>();
    LocalTensor<float> cornerWeightBrc = cornerWeightBrcBuf_.Get<float>();
    LocalTensor<float> output = outputQue_.Get<float>();

    LocalTensor<float> shapeBrc = shapeBrcBuf_.Get<float>();
    LocalTensor<int32_t> locInt = locIntBuf_.Get<int32_t>();
    LocalTensor<float> locFloat = locFloatBuf_.Get<float>();
    LocalTensor<float> weight = weightBuf_.Get<float>();

    PrepareShape(shapes, offset, shapeBrc);
    Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, 2 * valRptTimes4_, 1, 8);
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);

    uint8_t ping = 0;
    for (uint32_t batch = 0; batch < batchSize_; ++batch) {
        for (uint32_t query = startOffset_; query < endOffset_; ++query) {
            baseSrcOffset_ = batch * numKeys_ * numHeads_;
            baseDstOffset_ = (batch * numQueries_ + query) * numHeads_ * embedDims_;

            WaitFlag<HardEvent::MTE3_V>(ping);
            Duplicate<float, false>(output[ping * alignedHeadEmbedDims_], 0.f, MASK_PLACEHOLDER, outRptTimes_, 1, 8);
            for (uint32_t pl = 0; pl < pointLoops_; ++pl) {
                CopyInSample(location[2 * alignedOneQueryNum_], attentionWeight, batch, query, pl);
                ComputeLocation(location, shapeBrc, locInt, locFloat);
                ComputeWeight(locInt, locFloat, weight, attentionWeight);
                ComputeBilinearInterpolation(shapes, offset, locInt, value, weight, cornerWeight, cornerWeightBrc,
                    output[ping * alignedHeadEmbedDims_]);
            }
            SetFlag<HardEvent::V_MTE3>(calEvt_);
            WaitFlag<HardEvent::V_MTE3>(calEvt_);
            DataCopy(outputGm_[baseDstOffset_], output[ping * alignedHeadEmbedDims_], {1, outBlk_, 0, 0});
            SetFlag<HardEvent::MTE3_V>(ping);
            ping = 1 - ping;
        }
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}
#endif // MS_DEFORM_ATTN_HIGH_PERF_H_
