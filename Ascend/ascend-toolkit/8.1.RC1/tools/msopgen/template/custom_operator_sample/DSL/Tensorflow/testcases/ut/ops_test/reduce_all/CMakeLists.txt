add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

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
file(GLOB OP_PROTO_TEST_FILES_CC **proto.cc)
file(GLOB OP_PROTO_TEST_FILES_CPP **proto.cpp)

link_directories(
        "${ATC_DIR}/lib64"
        "${GTEST_DIR}"
)

set(CUSTOM_OBJECT_NAME "reduce_all_proto_test")

add_executable(${CUSTOM_OBJECT_NAME}
        ${PROJECT_DIR}/testcases/ut/ops_test/test_main.cc ${OP_PROTO_SOURCE_SRCS} ${OP_PROTO_TEST_FILES_CC} ${OP_PROTO_TEST_FILES_CPP})

target_link_libraries(${CUSTOM_OBJECT_NAME} gtest c_sec ascendalog pthread error_manager graph register)
