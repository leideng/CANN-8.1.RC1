/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/op_def.h"
#include "op_def_impl.h"
#include "common/ge_common/debug/ge_log.h"

namespace ops {
OpAICoreConfig::OpAICoreConfig() : impl_(new(std::nothrow) OpAICoreConfigImpl) {}

OpAICoreConfig::OpAICoreConfig(const OpAICoreConfig &aicore_config) : impl_(new(std::nothrow) OpAICoreConfigImpl) {
  this->impl_->op_params = aicore_config.impl_->op_params;
  this->impl_->cfg_keys = aicore_config.impl_->cfg_keys;
  this->impl_->cfg_info = aicore_config.impl_->cfg_info;
}

OpAICoreConfig::~OpAICoreConfig() = default;

OpAICoreConfig &OpAICoreConfig::operator=(const OpAICoreConfig &aicore_config) {
  if (this != &aicore_config) {
    *this->impl_ = *aicore_config.impl_;
  }
  return *this;
}

OpParamDef &OpAICoreConfig::Input(const char *name) {
  return this->impl_->op_params.Input(name);
}

OpParamDef &OpAICoreConfig::Output(const char *name) {
  return this->impl_->op_params.Output(name);
}

OpAICoreConfig &OpAICoreConfig::DynamicCompileStaticFlag(bool flag) {
  this->AddCfgItem("dynamicCompileStatic.flag", flag ? "true" : "false");
  return *this;
}

OpAICoreConfig &OpAICoreConfig::DynamicFormatFlag(bool flag) {
  this->AddCfgItem("dynamicFormat.flag", flag ? "true" : "false");
  return *this;
}

OpAICoreConfig &OpAICoreConfig::DynamicRankSupportFlag(bool flag) {
  this->AddCfgItem("dynamicRankSupport.flag", flag ? "true" : "false");
  return *this;
}

OpAICoreConfig &OpAICoreConfig::DynamicShapeSupportFlag(bool flag) {
  this->AddCfgItem("dynamicShapeSupport.flag", flag ? "true" : "false");
  return *this;
}

OpAICoreConfig &OpAICoreConfig::NeedCheckSupportFlag(bool flag) {
  this->AddCfgItem("needCheckSupport.flag", flag ? "true" : "false");
  return *this;
}

OpAICoreConfig &OpAICoreConfig::PrecisionReduceFlag(bool flag) {
  this->AddCfgItem("precision_reduce.flag", flag ? "true" : "false");
  return *this;
}

OpAICoreConfig &OpAICoreConfig::ExtendCfgInfo(const char *key, const char *value) {
  this->AddCfgItem(key, value);
  return *this;
}

std::vector<OpParamDef> &OpAICoreConfig::GetInputs(void) const {
  return this->impl_->op_params.GetInputs();
}
std::vector<OpParamDef> &OpAICoreConfig::GetOutputs(void) const {
  return this->impl_->op_params.GetOutputs();
}
void OpAICoreConfig::AddCfgItem(const char *key, const char *value) {
  auto it = this->impl_->cfg_info.find(key);
  if (it == this->impl_->cfg_info.cend()) {
    this->impl_->cfg_keys.emplace_back(key);
  } else {
    this->impl_->cfg_info.erase(key);
  }
  this->impl_->cfg_info.emplace(key, value);
}

std::vector<ge::AscendString> &OpAICoreConfig::GetCfgKeys(void) {
  return this->impl_->cfg_keys;
}

std::map<ge::AscendString, ge::AscendString> &OpAICoreConfig::GetCfgInfo(void) {
  return this->impl_->cfg_info;
}

ge::AscendString &OpAICoreConfig::GetConfigValue(const char *key) {
  return this->impl_->cfg_info[key];
}

OpAICoreDef::OpAICoreDef() : impl_(new(std::nothrow) OpAICoreDefImpl) {}

OpAICoreDef::OpAICoreDef(const OpAICoreDef &aicore_def) : impl_(new(std::nothrow) OpAICoreDefImpl) {
  this->impl_->tiling_func = aicore_def.impl_->tiling_func;
  this->impl_->tiling_parse = aicore_def.impl_->tiling_parse;
  this->impl_->ci_creator = aicore_def.impl_->ci_creator;
  this->impl_->ci_deleter = aicore_def.impl_->ci_deleter;
  this->impl_->op_chk_support = aicore_def.impl_->op_chk_support;
  this->impl_->op_sel_format = aicore_def.impl_->op_sel_format;
  this->impl_->op_get_support = aicore_def.impl_->op_get_support;
  this->impl_->op_get_spec = aicore_def.impl_->op_get_spec;
  this->impl_->op_generlize_func = aicore_def.impl_->op_generlize_func;
  this->impl_->aicore_configs = aicore_def.impl_->aicore_configs;
}

OpAICoreDef::~OpAICoreDef() = default;

OpAICoreDef &OpAICoreDef::operator=(const OpAICoreDef &aicore_def) {
  if (this != &aicore_def) {
    *this->impl_ = *aicore_def.impl_;
  }
  return *this;
}

ge::graphStatus TilingParsePlaceHolder(gert::TilingParseContext* context)
{
  (void)context;
  return ge::GRAPH_SUCCESS;
}

OpAICoreDef &OpAICoreDef::SetTiling(gert::OpImplRegisterV2::TilingKernelFunc func) {
  this->impl_->tiling_func = func;
  this->impl_->tiling_parse = TilingParsePlaceHolder;
  return *this;
}

OpAICoreDef &OpAICoreDef::SetCheckSupport(optiling::OP_CHECK_FUNC func) {
  this->impl_->op_chk_support = func;
  return *this;
}

OpAICoreDef &OpAICoreDef::SetOpSelectFormat(optiling::OP_CHECK_FUNC func) {
  this->impl_->op_sel_format = func;
  return *this;
}

OpAICoreDef &OpAICoreDef::SetOpSupportInfo(optiling::OP_CHECK_FUNC func) {
  this->impl_->op_get_support = func;
  return *this;
}

OpAICoreDef &OpAICoreDef::SetOpSpecInfo(optiling::OP_CHECK_FUNC func) {
  this->impl_->op_get_spec = func;
  return *this;
}

OpAICoreDef &OpAICoreDef::SetParamGeneralize(optiling::PARAM_GENERALIZE_FUNC func) {
  this->impl_->op_generlize_func = func;
  return *this;
}

OpAICoreDef &OpAICoreDef::AddConfig(const char *soc) {
  OpAICoreConfig aicore_config;
  aicore_config.DynamicCompileStaticFlag(true)
    .DynamicFormatFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .NeedCheckSupportFlag(false)
    .PrecisionReduceFlag(true);
  this->AddConfig(soc, aicore_config);
  return *this;
}

OpAICoreDef &OpAICoreDef::AddConfig(const char *soc, OpAICoreConfig &aicore_config) {
  this->impl_->aicore_configs.erase(ge::AscendString(soc));
  this->impl_->aicore_configs.emplace(ge::AscendString(soc), aicore_config);
  return *this;
}

std::map<ge::AscendString, OpAICoreConfig> &OpAICoreDef::GetAICoreConfigs(void) {
  return this->impl_->aicore_configs;
}

gert::OpImplRegisterV2::TilingKernelFunc &OpAICoreDef::GetTiling(void) {
  return this->impl_->tiling_func;
}

optiling::OP_CHECK_FUNC &OpAICoreDef::GetCheckSupport(void) {
  return this->impl_->op_chk_support;
}
optiling::OP_CHECK_FUNC &OpAICoreDef::GetOpSelectFormat(void) {
  return this->impl_->op_sel_format;
}
optiling::OP_CHECK_FUNC &OpAICoreDef::GetOpSupportInfo(void) {
  return this->impl_->op_get_support;
}
optiling::OP_CHECK_FUNC &OpAICoreDef::GetOpSpecInfo(void) {
  return this->impl_->op_get_spec;
}
optiling::PARAM_GENERALIZE_FUNC &OpAICoreDef::GetParamGeneralize(void) {
  return this->impl_->op_generlize_func;
}
void OpAICoreDef::Log(const char *op_type, const char *info) const {
  GELOGD("%s, op_type:%s.", info, op_type);
}
}  // namespace ops
