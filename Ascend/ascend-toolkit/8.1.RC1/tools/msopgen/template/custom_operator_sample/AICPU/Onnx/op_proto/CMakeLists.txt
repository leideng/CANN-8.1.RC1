# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
if("x${CMAKE_CXX_COMPILER}" STREQUAL "x")
    set(CMAKE_CXX_COMPILER g++)
endif()
if("x${CMAKE_C_COMPILER}" STREQUAL "x")
    set(CMAKE_C_COMPILER gcc)
endif()

# add source files
aux_source_directory(. SRCS)
add_custom_target(third_proto)
include(${CMAKE_SOURCE_DIR}/third_party/secure_c_proto.cmake)
add_dependencies(third_proto securec_proto_headers)
include_directories(${CMAKE_SOURCE_DIR}/third_party/src/secure_c_proto/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/src/secure_c_kernel/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/metadef/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/util)
include_directories(${CMAKE_SOURCE_DIR}/log)
include_directories(${CMAKE_SOURCE_DIR}/metadef/inc/external/graph)
include_directories(${CMAKE_SOURCE_DIR}/metadef/inc/external)
include_directories(${CMAKE_SOURCE_DIR}/op_proto/common/inc)

if("x${SRCS}" STREQUAL "x")
    add_custom_target(${OP_PROTO_TARGET}
            COMMAND mkdir -p ${OP_PROTO_TARG  ET_OUT_DIR}
            COMMAND echo "no source to make lib${OP_PROTO_TARGET}.so")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH ${OP_PROTO_TARGET_OUT_DIR})

message( STATUS "OP_PROTO_TARGET=${OP_PROTO_TARGET}")
add_library(${OP_PROTO_TARGET} SHARED ${SRCS})
add_dependencies(${OP_PROTO_TARGET} third_proto)
target_link_libraries(${OP_PROTO_TARGET} ${ASCEND_INC}/../lib64/libgraph.so)
