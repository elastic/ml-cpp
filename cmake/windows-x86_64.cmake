message(STATUS "In windows-x86_64.cmake")
# this must be first
include ("${CMAKE_CURRENT_LIST_DIR}/functions.cmake")

# set the os variables for windows
include ("${CMAKE_CURRENT_LIST_DIR}/os/windows.cmake")

# set the architecture bits
include ("${CMAKE_CURRENT_LIST_DIR}/architecture/x86_64.cmake")

include ("${CMAKE_CURRENT_LIST_DIR}/compiler/vs2019.cmake")

message(STATUS "windows-x86_64: ML_COMPILE_DEFINITIONS = ${ML_COMPILE_DEFINITIONS}")
message(STATUS "windows-x86_64: ML_LIBRAY_PREFIX ${ML_LIBRARY_PREFIX}")
