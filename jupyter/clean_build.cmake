set(SRC_DIR src)
set(TEST_DIR tests)

function(clean)
  foreach(file ${CLEAN_BUILD_FILES})
    message(STATUS ${file})
  endforeach() 
  if(CLEAN_BUILD_FILES)
    file(REMOVE_RECURSE ${CLEAN_BUILD_FILES})
  endif()
endfunction()

file(GLOB CLEAN_BUILD_FILES LIST_DIRECTORIES true ${SRC_DIR}/*.egg-info ${SRC_DIR}/**/__pycache__ ${TEST_DIR}/**/__pycache__ ${TEST_DIR}/**/*.pyc .pytest_cache .ipynb_checkpoints)

clean()