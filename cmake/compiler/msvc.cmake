#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#


if(NOT CRT_OPT)
 set(CRT_OPT -MD)
endif()

set(VCINCLUDES "-I${VCBASE}/VC/Tools/MSVC/${VCVER}/include")
set(WINSDKINCLUDES "-I${WINSDKBASE}/10/Include/$ENV{UCRTVER}/ucrt -I${WINSDKBASE}/8.0/Include/shared -I${WINSDKBASE}/8.0/Include/um")
set(CPPFLAGS "-X -I${CMAKE_SOURCE_DIR}/3rd_party/include -I${ROOT}/usr/local/include ${VCINCLUDES} ${WINSDKINCLUDES} -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -DWIN32_LEAN_AND_MEAN -DNTDDI_VERSION=0x06010000 -D_WIN32_WINNT=0x0601 -DBUILDING_autodetect ${OPTCPPFLAGS}")

add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE WIN32_LEAN_AND_MEAN NTDDI_VERSION=0x06010000 _WIN32_WINNT=0x0601 Windows)
set(CMAKE_C_FLAGS "-nologo ${OPTCFLAGS} -W4 ${CRT_OPT} -EHsc -Zi -Gw -FS -Zc:inline -diagnostics:caret -utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -TP ${CFLAGS} -Zc:rvalueCast -Zc:strictStrings -wd4127 -we4150 -wd4201 -wd4231 -wd4251 -wd4355 -wd4512 -wd4702 -bigobj ${CPPFLAGS}")

set(CMAKE_STATIC_LIBRARY_PREFIX lib)
set(CMAKE_SHARED_LIBRARY_PREFIX lib)
set(CMAKE_IMPORT_LIBRARY_PREFIX lib)

