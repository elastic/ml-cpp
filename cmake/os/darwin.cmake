message(STATUS "Darwin detected")

set(EXE_DIR MacOS)
set(CMAKE_MACOSX_RPATH 1)
add_compile_definitions(MacOSX)
set(PLATFORM_NAME "MacOSX")
