if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/")
endif()

add_subdirectory(core)
add_subdirectory(maths/common)
add_subdirectory(maths/time_series)
add_subdirectory(maths/analytics)
add_subdirectory(model)
add_subdirectory(api)
add_subdirectory(api/dump_state)
add_subdirectory(test)
add_subdirectory(ver)
add_subdirectory(seccomp)

add_subdirectory(core/unittest)
add_subdirectory(maths/common/unittest)
add_subdirectory(maths/time_series/unittest)
add_subdirectory(maths/analytics/unittest)
add_subdirectory(model/unittest)
add_subdirectory(api/unittest)
add_subdirectory(ver/unittest)
add_subdirectory(seccomp/unittest)

add_custom_target(run_test_model
    DEPENDS ml_test_model
    COMMAND ml_test_model
    WORKING_DIRECTORY ../../lib/model/unittest
)
add_custom_target(run_test_core
    DEPENDS ml_test_core
    COMMAND ml_test_core
    WORKING_DIRECTORY ../../lib/core/unittest
)
add_custom_target(run_test_api
    DEPENDS ml_test_api
    COMMAND ml_test_api
    WORKING_DIRECTORY ../../lib/api/unittest
)
add_custom_target(run_test_seccomp
    DEPENDS ml_test_seccomp
    COMMAND ml_test_seccomp
    WORKING_DIRECTORY ../../lib/seccomp/unittest
)
add_custom_target(run_test_maths_common
    DEPENDS ml_test_maths_common
    COMMAND ml_test_maths_common
    WORKING_DIRECTORY ../../lib/maths/common/unittest
)
add_custom_target(run_test_maths_time_series
    DEPENDS ml_test_maths_time_series
    COMMAND ml_test_maths_time_series
    WORKING_DIRECTORY ../../lib/maths/time_series/unittest
)
add_custom_target(run_test_maths_analytics
    DEPENDS ml_test_maths_analytics
    COMMAND ml_test_maths_analytics
    WORKING_DIRECTORY ../../lib/maths/analytics/unittest
)
add_custom_target(run_test_ver
    DEPENDS ml_test_ver
    COMMAND ml_test_ver
    WORKING_DIRECTORY ../../lib/ver/unittest
)
