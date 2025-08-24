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
 * \file context_builder.h
 * \brief Api to build tiling context
 */

#ifndef CONTEXT_BUILDER_H
#define CONTEXT_BUILDER_H

#include <memory>
#include <vector>
#include <cstring>
#include "external/exe_graph/runtime/kernel_run_context.h"
#include "external/exe_graph/runtime/context_extend.h"
#include "external/exe_graph/runtime/storage_shape.h"
#include "external/exe_graph/runtime/tiling_context.h"


namespace context_ascendc {
class ContextBuilderImpl;
class ValueHolderImpl;

struct KernelRunContextHolder {
    KernelRunContextHolder();
    ~KernelRunContextHolder();
    template<typename T>
    T *GetContext() const
    {
        return reinterpret_cast<T*>(context);
    }
    gert::ComputeNodeInfo *MutableComputeNodeInfo()
    {
        return reinterpret_cast<gert::ComputeNodeInfo *>(computeNodeExtendHolder.get());
    }
    std::unique_ptr<ValueHolderImpl> valueHolder;
    std::unique_ptr<uint8_t[]> computeNodeExtendHolder;
    KernelRunContext *context {nullptr};
};

class ContextBuilder {
public:
    ContextBuilder();
    ~ContextBuilder();
    ContextBuilder(ContextBuilder &&kernelRunContextBuilder) = delete;
    ContextBuilder &operator=(ContextBuilder &&kernelRunContextBuilder) = delete;

    // kernel context builder
    ContextBuilder &Inputs(std::vector<void *> inputs);
    ContextBuilder &Outputs(std::vector<void *> outputs);
    std::shared_ptr<KernelRunContextHolder> BuildKernelRunContext();

    // Tiling Context Builder
    ContextBuilder &NodeIoNum(size_t inputNum, size_t outputNum);
    ContextBuilder &SetOpNameType(const std::string& opName, const std::string& opType);
    ContextBuilder &IrInstanceNum(std::vector<uint32_t> instanceNum);
    ContextBuilder &AddInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
        ge::Format storageFormat, gert::StorageShape storageShape);
    ContextBuilder &AddOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
        ge::Format storageFormat, gert::StorageShape storageShape);
    ContextBuilder &AddInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
        ge::Format storageFormat, gert::StorageShape storageShape, void* constValues);
    ContextBuilder &AddInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
        ge::Format storageFormat, gert::StorageShape storageShape, const std::string &filePath);
    ContextBuilder &AddAttr(const std::string& attrName, int64_t attrValue);
    ContextBuilder &AddAttr(const std::string& attrName, bool attrValue);
    ContextBuilder &AddAttr(const std::string& attrName, const std::string& attrValue);
    ContextBuilder &AddAttr(const std::string& attrName, float attrValue);
    ContextBuilder &AddAttr(const std::string& attrName, const std::vector<float>& attrValue);
    ContextBuilder &AddAttr(const std::string& attrName, const std::vector<bool>& attrValue);
    ContextBuilder &AddAttr(const std::string& attrName, const std::vector<int64_t>& attrValue);
    ContextBuilder &AddAttr(const std::string& attrName, const std::vector<std::string>& attrValue);
    ContextBuilder &AddAttr(const std::string& attrName, const std::vector<std::vector<int64_t>>& attrValue);

    ContextBuilder &CompileInfo(void *compileInfo);
    ContextBuilder &PlatformInfo(void *platformInfo);
    ContextBuilder &TilingData(void *tilingData);
    ContextBuilder &Workspace(gert::ContinuousVector *workspace);
    std::shared_ptr<KernelRunContextHolder> BuildTilingContext();

private:
    std::unique_ptr<ContextBuilderImpl> impl_;
};
}
#endif
