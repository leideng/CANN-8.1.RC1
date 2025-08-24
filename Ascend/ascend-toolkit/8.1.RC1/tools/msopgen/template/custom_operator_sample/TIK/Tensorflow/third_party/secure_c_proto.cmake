# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

include(ExternalProject)
ExternalProject_Add(secure_c_proto
  URL               https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz
  PREFIX            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(secure_c_proto SOURCE_DIR)
ExternalProject_Get_Property(secure_c_proto BINARY_DIR)

set(SECUREC_INCLUDE ${SOURCE_DIR})
add_custom_target(securec_proto_headers ALL DEPENDS secure_c_proto)