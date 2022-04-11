project("ML Core lib unit tests")

include_directories(../../../include)

# TODO make platform safe
set (PLATFORM_SRCS
)

set (SRCS
Main.cc 
CSystemCallFilterTest.cc 
)

add_executable(ml_test_seccomp ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_seccomp ${Boost_LIBRARIES})
target_link_libraries(ml_test_seccomp ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_seccomp MlCore)
target_link_libraries(ml_test_seccomp MlSeccomp)
target_link_libraries(ml_test_seccomp MlTest)

add_test(ml_test_seccomp ml_test_seccomp)
# make "make test" run the test program
add_custom_target(test_seccomp
     DEPENDS ml_test_seccomp
     COMMAND ml_test_seccomp
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
