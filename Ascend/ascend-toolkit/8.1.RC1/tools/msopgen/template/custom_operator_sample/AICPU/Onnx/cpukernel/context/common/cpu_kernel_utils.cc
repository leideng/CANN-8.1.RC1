/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel_utils.h"

#include "attr_value_impl.h"
#include "device.h"
#include "log.h"
#include "node_def_impl.h"
#include "sharder.h"
#include "status.h"
#include "tensor_impl.h"
#include "tensor_shape_impl.h"
#include "securec.h"

namespace aicpu {
/*
 * construct Tensor for memory self-management.
 */
std::shared_ptr<Tensor> CpuKernelUtils::CreateTensor() {
  auto proto_ptr = new (std::nothrow) aicpuops::Tensor();
  KERNEL_CHECK_NULLPTR(proto_ptr, std::shared_ptr<Tensor>(nullptr),
                       "New Tensor proto failed.")

  auto wrapper_ptr = new (std::nothrow)
      TensorImpl(proto_ptr, [](aicpuops::Tensor *p) { delete p; });
  if (wrapper_ptr == nullptr) {
    KERNEL_LOG_ERROR("New TensorProto failed");
    delete proto_ptr;
    return std::shared_ptr<Tensor>(nullptr);
  }

  auto class_ptr = new (std::nothrow) Tensor(wrapper_ptr);
  if (class_ptr == nullptr) {
    KERNEL_LOG_ERROR("New Tensor failed");
    delete wrapper_ptr;
    return std::shared_ptr<Tensor>(nullptr);
  }

  return std::shared_ptr<Tensor>(class_ptr);
}

std::shared_ptr<Tensor> CpuKernelUtils::CreateTensor(TensorImpl *tensor) {
  KERNEL_CHECK_NULLPTR(tensor, std::shared_ptr<Tensor>(nullptr), "Tensor is null.")
  auto class_ptr = new (std::nothrow) Tensor(tensor);
  KERNEL_CHECK_NULLPTR(class_ptr, std::shared_ptr<Tensor>(nullptr), "New Tensor failed.")
  return std::shared_ptr<Tensor>(class_ptr);
}

/*
 * get tensor impl.
 */
std::shared_ptr<TensorImpl> CpuKernelUtils::GetImpl(const Tensor *tensor) {
  return tensor->impl_;
}

/*
 * get tensor name.
 */
std::string CpuKernelUtils::GetTensorName(const Tensor *tensor) {
  auto impl = GetImpl(tensor);
  KERNEL_CHECK_NULLPTR(impl, std::string(), "Get Tensor impl failed.")
  return impl->GetName();
}

/*
 * set tensor name.
 */
void CpuKernelUtils::SetTensorName(const std::string &name,
                                   std::shared_ptr<Tensor> &tensor) {
  KERNEL_LOG_INFO("Set tensor name[%s]", name.c_str());
  auto impl = GetImpl(tensor.get());
  KERNEL_CHECK_NULLPTR_VOID(impl, "Get Tensor impl failed.")
  impl->SetName(name);
}

std::shared_ptr<TensorShape> CpuKernelUtils::CreateTensorShape() {
  auto proto_ptr = new (std::nothrow) aicpuops::TensorShape();
  KERNEL_CHECK_NULLPTR(proto_ptr, std::shared_ptr<TensorShape>(nullptr),
                       "New TensorShape proto failed.")

  auto wrapper_ptr = new (std::nothrow)
      TensorShapeImpl(proto_ptr, [](aicpuops::TensorShape *p) { delete p; });
  if (wrapper_ptr == nullptr) {
    KERNEL_LOG_ERROR("new TensorShapeImpl failed");
    delete proto_ptr;
    return std::shared_ptr<TensorShape>(nullptr);
  }

  auto class_ptr = new (std::nothrow) TensorShape(wrapper_ptr);
  if (class_ptr == nullptr) {
    KERNEL_LOG_ERROR("new TensorShape failed");
    delete wrapper_ptr;
    return std::shared_ptr<TensorShape>(nullptr);
  }

  return std::shared_ptr<TensorShape>(class_ptr);
}

std::shared_ptr<TensorShape> CpuKernelUtils::CreateTensorShape(
    TensorShapeImpl *tensor_shape) {
  KERNEL_CHECK_NULLPTR(tensor_shape, std::shared_ptr<TensorShape>(nullptr),
                       "Tensor shape proto is null.")
  auto class_ptr = new (std::nothrow) TensorShape(tensor_shape);
  KERNEL_CHECK_NULLPTR(class_ptr, std::shared_ptr<TensorShape>(nullptr),
                       "New TensorShape failed.")
  return std::shared_ptr<TensorShape>(class_ptr);
}

/*
 * get tensor shape impl.
 */
std::shared_ptr<TensorShapeImpl> CpuKernelUtils::GetImpl(
    const TensorShape *tensorShape) {
  return tensorShape->impl_;
}

/*
 * construct AttrValue for memory self-management.
 */
std::shared_ptr<AttrValue> CpuKernelUtils::CreateAttrValue() {
  auto proto_ptr = new (std::nothrow) aicpuops::AttrValue();
  KERNEL_CHECK_NULLPTR(proto_ptr, std::shared_ptr<AttrValue>(nullptr),
                       "New AttrValue proto failed.")

  auto wrapper_ptr = new (std::nothrow)
      AttrValueImpl(proto_ptr, [](aicpuops::AttrValue *p) { delete p; });
  if (wrapper_ptr == nullptr) {
    KERNEL_LOG_ERROR("new AttrValueImpl failed");
    delete proto_ptr;
    return std::shared_ptr<AttrValue>(nullptr);
  }

  auto class_ptr = new (std::nothrow) AttrValue(wrapper_ptr);
  if (class_ptr == nullptr) {
    KERNEL_LOG_ERROR("new AttrValue failed");
    delete wrapper_ptr;
    return std::shared_ptr<AttrValue>(nullptr);
  }

  return std::shared_ptr<AttrValue>(class_ptr);
}

std::shared_ptr<AttrValue> CpuKernelUtils::CreateAttrValue(
    AttrValueImpl *impl) {
  KERNEL_CHECK_NULLPTR(impl, std::shared_ptr<AttrValue>(nullptr), "Impl is null.")
  auto class_ptr = new (std::nothrow) AttrValue(impl);
  KERNEL_CHECK_NULLPTR(class_ptr, std::shared_ptr<AttrValue>(nullptr), "New AttrValue failed.")
  return std::shared_ptr<AttrValue>(class_ptr);
}

/*
 * get attr value impl.
 */
std::shared_ptr<AttrValueImpl> CpuKernelUtils::GetImpl(
    const AttrValue *attr_value) {
  return attr_value->impl_;
}

/*
 * construct NodeDef for memory self-management.
 */
std::shared_ptr<NodeDef> CpuKernelUtils::CreateNodeDef() {
  auto proto_ptr = new (std::nothrow) aicpuops::NodeDef();
  KERNEL_CHECK_NULLPTR(proto_ptr, std::shared_ptr<NodeDef>(nullptr),
                       "New NodeDef proto failed.")

  auto wrapper_ptr = new (std::nothrow)
      NodeDefImpl(proto_ptr, [](aicpuops::NodeDef *p) { delete p; });
  if (wrapper_ptr == nullptr) {
    KERNEL_LOG_ERROR("new NodeDefImpl failed");
    delete proto_ptr;
    return std::shared_ptr<NodeDef>(nullptr);
  }

  auto class_ptr = new (std::nothrow) NodeDef(wrapper_ptr);
  if (class_ptr == nullptr) {
    KERNEL_LOG_ERROR("new NodeDef failed");
    delete wrapper_ptr;
    return std::shared_ptr<NodeDef>(nullptr);
  }

  return std::shared_ptr<NodeDef>(class_ptr);
}

/*
 * ParallelFor shards the "total" units of work.
 * @return uint32_t: 0->sucess other->failed
 */
uint32_t CpuKernelUtils::ParallelFor(
    const CpuKernelContext &ctx, int64_t total, int64_t perUnitSize,
    const std::function<void(int64_t, int64_t)> &work) {
  KERNEL_CHECK_NULLPTR(ctx.device_, KERNEL_STATUS_INNER_ERROR, "Device is null.")

  const Sharder *sharder = ctx.device_->GetSharder();
  KERNEL_CHECK_NULLPTR(sharder, KERNEL_STATUS_INNER_ERROR, "Get sharder is null.")

  sharder->ParallelFor(total, perUnitSize, work);
  return KERNEL_STATUS_OK;
}

/*
 * Get CPU number
 * @return CPU number
 */
uint32_t CpuKernelUtils::GetCPUNum(const CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.device_, 0, "Device is null.")

  const Sharder *sharder = ctx.device_->GetSharder();
  KERNEL_CHECK_NULLPTR(sharder, 0, "Get sharder is null.")

  return sharder->GetCPUNum();
}

void CpuKernelUtils::UpdateCustWorkSpaceInfo(CpuKernelContext *ctx, uint64_t workspace_size,
                                             uint64_t workspace_addr) {
  ctx->workspace_size_ = workspace_size;
  ctx->workspace_addr_ = workspace_addr;
  auto ret = memset_s(
      reinterpret_cast<void *>(static_cast<uintptr_t>(ctx->workspace_addr_)),
      ctx->workspace_size_, 0x00, ctx->workspace_size_);
  if (ret != EOK) {
    KERNEL_LOG_ERROR("UpdateCustWorkSpaceInfo memeset_s failed.");
  }
  return;
}
}  // namespace aicpu
