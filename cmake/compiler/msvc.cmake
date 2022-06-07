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

set(VCINCLUDES ${VCBASE}/VC/Tools/MSVC/${VCVER}/include)
set(WINSDKINCLUDES ${WINSDKBASE}/10/Include/${UCRTVER}/ucrt ${WINSDKBASE}/8.0/Include/shared ${WINSDKBASE}/8.0/Include/um)

list(APPEND ML_SYSTEM_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/3rd_party/include ${ROOT}/usr/local/include ${VCINCLUDES} ${WINSDKINCLUDES})

list(APPEND ML_COMPILE_DEFINITIONS
	_CRT_SECURE_NO_WARNINGS
	_CRT_NONSTDC_NO_DEPRECATE
	WIN32_LEAN_AND_MEAN
    NOMINMAX
	NTDDI_VERSION=0x06010000
	_WIN32_WINNT=0x0601 Windows)

list(APPEND ML_C_FLAGS
	"-X"
	"-nologo"
	${OPTCFLAGS}
	"-W4"
	${CRT_OPT}
	"-EHsc"
	"-Zi"
	"-Gw"
	"-FS"
	"-Zc:inline"
	"-diagnostics:caret"
	"-utf-8")

list(APPEND ML_CXX_FLAGS
	${ML_C_FLAGS}
	"-TP"
	"-Zc:rvalueCast"
	"-Zc:strictStrings"
	"-wd4127"
	"-we4150"
	"-wd4201"
	"-wd4231"
	"-wd4251"
	"-wd4355"
	"-wd4512"
	"-wd4702"
	"-bigobj"
	"${OPTCPPFLAGS}")

