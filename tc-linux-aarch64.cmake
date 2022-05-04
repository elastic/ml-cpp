# the name of the target operating system
set(CMAKE_SYSTEM_NAME Linux)

message(STATUS "CMAKE_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}")

set(CROSS_TARGET_PLATFORM  aarch64-linux-gnu)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER   "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-gcc")
set(CMAKE_CXX_COMPILER "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-g++")

set(CMAKE_AR       "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-ar")
set(CMAKE_RANLIB   "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-ranlib")
set(CMAKE_STRIP    "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-strip")
set(CMAKE_LINKER   "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-ld")

SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -ru <TARGET> <OBJECTS>")

# where is the target environment located
set(CMAKE_FIND_ROOT_PATH  /usr/local/sysroot-${CROSS_TARGET_PLATFORM})
#set(CMAKE_SYSROOT  /usr/local/sysroot-${CROSS_TARGET_PLATFORM})

message(STATUS "CMAKE_SYSROOT=${CMAKE_SYSROOT}")

# adjust the default behavior of the FIND_XXX() commands:
# search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
