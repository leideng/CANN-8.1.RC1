set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ut/ops_test")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ut/ops_test)
endif()

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/st/out/*")
foreach(file ${files})
  message(STATUS ${file})
  if(IS_DIRECTORY ${file})
      add_subdirectory(${file})
  endif()
endforeach()