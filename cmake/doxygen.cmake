function(ml_doxygen output)
  find_program(DOXYGEN_EXECUTABLE doxygen HINTS /opt/homebrew /usr/local)
  find_package(Doxygen)
  if (NOT DOXYGEN_FOUND)
    add_custom_target(doxygen COMMAND false
                              COMMENT "Doxygen not found")
    return()
  endif()

  set(DOXYGEN_GENERATE_HTML          YES)
  set(DOXYGEN_HTML_OUTPUT            ${output})
  set(DOXYGEN_PROJECT_NAME           "Ml C++")
  set(DOXYGEN_PROJECT_NUMBER         ${ML_VERSION_NUM})
  set(DOXYGEN_PROJECT_LOGO           mk/ml.ico)
  set(DOXYGEN_OUTPUT_DIRECTORY       ${CMAKE_SOURCE_DIR}/build/doxygen)
  set(DOXYGEN_INHERIT_DOCS           NO)
  set(DOXYGEN_SEPARATE_MEMBER_PAGES  NO)
  set(DOXYGEN_TAB_SIZE               4)
  set(DOXYGEN_LOOKUP_CACHE_SIZE      1)
  set(DOXYGEN_EXTRACT_ALL            YES)
  set(DOXYGEN_EXTRACT_PRIVATE        YES)
  set(DOXYGEN_EXTRACT_STATIC         YES)
  set(DOXYGEN_EXTRACT_ANON_NSPACES   YES)
  set(DOXYGEN_FILE_PATTERNS          *.cc  *.h)
  set(DOXYGEN_RECURSIVE              YES)
  set(DOXYGEN_EXCLUDE                3rd_party)
  set(DOXYGEN_HTML_OUTPUT            cplusplus)
  set(DOXYGEN_SEARCHENGINE           NO)
  set(DOXYGEN_PAPER_TYPE             a4wide)
  set(DOXYGEN_EXTRA_PACKAGES         amsmath amssymb)
  set(DOXYGEN_LATEX_BATCHMODE        YES)
  set(DOXYGEN_HAVE_DOT               YES)
  set(DOXYGEN_DOT_FONTNAME           FreeSans)
  set(DOXYGEN_DOT_GRAPH_MAX_NODES    100)

  doxygen_add_docs(doxygen
      ${PROJECT_SOURCE_DIR}
      COMMENT "Generate HTML documentation"
  )

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${output})

endfunction()
