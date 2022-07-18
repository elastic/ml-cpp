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

#
# Create a CSV file listing the information about our 3rd party dependencies
# that is required for the stack-wide list.
#
# Usage:
# cmake -D OUTPUT_FILE=<output_file> -P dependency_report.cmake
#
# The format is that defined in https://github.com/elastic/release-manager/issues/207,
# i.e. a CSV file with the following fields:
#
# name,version,revision,url,license,copyright,sourceURL
#
# The way this script works, each component must have its own CSV file with
# those fields, and this script simply combines them into a single CSV file.
# Because of this, the field order is important - in each per-component CSV
# file the fields must be in the order shown above.

if(Windows)
  set(EOL "\r\n")
else()
  set(EOL "\n")
endif()
function(dependency_report _output_file)
  # IMPORTANT: this assumes all the *INFO.csv files have the following header:
  #
  # name,version,revision,url,license,copyright,sourceURL

  file(WRITE ${_output_file} "name,version,revision,url,license,copyright,sourceURL${EOL}")
  file(GLOB INFO_FILES ${CMAKE_CURRENT_LIST_DIR}/licenses/*INFO.csv)
  foreach(INFO_FILE ${INFO_FILES})
    file(STRINGS ${INFO_FILE} ${INFO_FILE}_CONTENTS)
    list(GET ${INFO_FILE}_CONTENTS -1 INFO_LINE)
    file(APPEND ${_output_file} "${INFO_LINE}${EOL}")
  endforeach()
endfunction()

if(OUTPUT_FILE)
  dependency_report(${OUTPUT_FILE})
else()
  message("Usage: cmake -D OUTPUT_FILE=<output_file> -P dependency_report.cmake")
  return()
endif()
