include(ExternalProject)
include(GNUInstallDirs)
set(_protobuf_url "")
if(CANN_PKG_SERVER)
  set(_protobuf_url "${CANN_PKG_SERVER}/libs/protobuf/v3.13.0.tar.gz")
endif()

if("x${PRODUCT_SIDE}" STREQUAL "xdevice")
  if (MINRC)
    set(CMAKE_CXX_COMPILER_ /usr/bin/aarch64-linux-gnu-g++)
    set(CMAKE_C_COMPILER_ /usr/bin/aarch64-linux-gnu-gcc)
  else()
    message("make device static protobuf")
    set(CMAKE_CXX_COMPILER_ ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
    set(CMAKE_C_COMPILER_ ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)
  endif()
else()
  message("make host static protobuf")
  set(CMAKE_CXX_COMPILER_ ${CMAKE_CXX_COMPILER})
  set(CMAKE_C_COMPILER_ ${CMAKE_C_COMPILER})
endif()

set(protobuf_CXXFLAGS "-Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=1 -O2 -Dgoogle=ascend_private")
set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
set(PROTOBUF_STATIC_PKG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/src/protobuf_static)
ExternalProject_Add(protobuf_static_build
                    URL ${_protobuf_url}
                        https://github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz
                    PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
                    URL_MD5 1a6274bc4a65b55a6fa70e264d796490
                    CONFIGURE_COMMAND ${CMAKE_COMMAND}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER_}
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER_}
                    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                    -DCMAKE_LINKER=${CMAKE_LINKER}
                    -DCMAKE_AR=${CMAKE_AR}
                    -DCMAKE_RANLIB=${CMAKE_RANLIB}
                    -Dprotobuf_WITH_ZLIB=OFF
                    -DLIB_PREFIX=ascend_
                    -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_CXX_FLAGS=${protobuf_CXXFLAGS} -DCMAKE_CXX_LDFLAGS=${protobuf_LDFLAGS} -DCMAKE_INSTALL_PREFIX=${PROTOBUF_STATIC_PKG_DIR} <SOURCE_DIR>/cmake
                    BUILD_COMMAND $(MAKE)
                    INSTALL_COMMAND $(MAKE) install
                    EXCLUDE_FROM_ALL TRUE
)
include(GNUInstallDirs)

add_library(ascend_protobuf_static_lib STATIC IMPORTED)

set_target_properties(ascend_protobuf_static_lib PROPERTIES
                      IMPORTED_LOCATION ${PROTOBUF_STATIC_PKG_DIR}/${CMAKE_INSTALL_LIBDIR}/libascend_protobuf.a
)

add_library(ascend_protobuf_static INTERFACE)
target_include_directories(ascend_protobuf_static INTERFACE ${PROTOBUF_STATIC_PKG_DIR}/include)
target_link_libraries(ascend_protobuf_static INTERFACE ascend_protobuf_static_lib)

add_dependencies(ascend_protobuf_static protobuf_static_build)
