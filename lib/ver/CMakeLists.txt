project("ML Ver")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)

set(SRCS
CBuildInfo.cc
)

add_library(MlVer STATIC ${SRCS})
target_link_libraries(MlVer ${Boost_LIBRARIES})

#install(TARGETS MlVer DESTINATION lib)
