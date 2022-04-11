project("ML Seccomp")

include_directories(../../include)

set(PLATFORM_SRCS
    CSystemCallFilter_${PLATFORM_NAME}.cc
)

set(SRCS
)

add_library(MlSeccomp STATIC ${SRCS} ${PLATFORM_SRCS})
