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
ExternalProject_Add(eigen
  URL               https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  PREFIX            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
  URL_MD5           4c527a9171d71a72a9d4186e65bea559
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(eigen SOURCE_DIR)
ExternalProject_Get_Property(eigen BINARY_DIR)

set(EIGEN_INCLUDE ${SOURCE_DIR})

add_custom_target(eigen_headers ALL DEPENDS eigen)
