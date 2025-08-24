set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )

set(PROJECT_DIR "$ENV{PROJECT_PATH}")
set(GTEST_DIR ${PROJECT_DIR}/testcases/libs/gtest)
set(ADK_DIR "$ENV{ADK_PATH}")
set(ATC_DIR ${ADK_DIR}/compiler)
set(OP_PROTO_SRC_DIR ${PROJECT_DIR}/op_proto)

message(STATUS "ATC_DIR=${ATC_DIR}")

enable_testing()

include_directories(
        "${GTEST_DIR}/include"
        "${ATC_DIR}/include"
        "${OP_PROTO_SRC_DIR}"
        )

aux_source_directory(${OP_PROTO_SRC_DIR} OP_PROTO_SOURCE_SRCS)
file(GLOB OP_PROTO_TEST_FILES **/**proto.cpp)

link_directories(
        "${ATC_DIR}/lib64"
        "${GTEST_DIR}"
)

set(CUSTOM_OBJECT_NAME "ops_proto_test")

add_executable(${CUSTOM_OBJECT_NAME}
        ${PROJECT_DIR}/testcases/ut/ops_test/test_main.cpp ${OP_PROTO_SOURCE_SRCS} ${OP_PROTO_TEST_FILES})

target_link_libraries(${CUSTOM_OBJECT_NAME} gtest c_sec ascendalog pthread error_manager graph graph_base register)
