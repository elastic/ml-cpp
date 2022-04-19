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

project("ML Api")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

add_compile_definitions(BUILDING_libMlApi)

set(SRCS
CAnnotationJsonWriter.cc
CAnomalyJob.cc
CAnomalyJobConfig.cc
CAnomalyJobConfigReader.cc
CBenchMarker.cc
CBoostedTreeInferenceModelBuilder.cc
CCategoryIdMapper.cc
CCmdSkeleton.cc
CConfigUpdater.cc
CCsvInputParser.cc
CCsvOutputWriter.cc
CDataFrameAnalysisConfigReader.cc
CDataFrameAnalysisInstrumentation.cc
CDataFrameAnalysisRunner.cc
CDataFrameAnalysisSpecification.cc
CDataFrameAnalysisSpecificationJsonWriter.cc
CDataFrameAnalyzer.cc
CDataFrameOutliersRunner.cc
CDataFrameTrainBoostedTreeClassifierRunner.cc
CDataFrameTrainBoostedTreeRegressionRunner.cc
CDataFrameTrainBoostedTreeRunner.cc
CDataProcessor.cc
CDetectionRulesJsonParser.cc
CFieldDataCategorizer.cc
CForecastRunner.cc
CGlobalCategoryId.cc
CHierarchicalResultsWriter.cc
CInferenceModelDefinition.cc
CInferenceModelMetadata.cc
CInputParser.cc
CIoManager.cc
CJsonOutputWriter.cc
CLengthEncodedInputParser.cc
CMemoryUsageEstimationResultJsonWriter.cc
CModelPlotDataJsonWriter.cc
CModelSizeStatsJsonWriter.cc
CModelSnapshotJsonWriter.cc
CNdInputParser.cc
CNdJsonInputParser.cc
CNdJsonOutputWriter.cc
CNoopCategoryIdMapper.cc
CPerPartitionCategoryIdMapper.cc
CPersistenceManager.cc
CResultNormalizer.cc
CSimpleOutputWriter.cc
CSingleFieldDataCategorizer.cc
CSingleStreamDataAdder.cc
CSingleStreamSearcher.cc
CStateRestoreStreamFilter.cc
ElasticsearchStateIndex.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_library(libMlApi SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(libMlApi PUBLIC ${Boost_LIBRARIES})
target_link_libraries(libMlApi PUBLIC ${LOG4CXX_LIBRARIES})
target_link_libraries(libMlApi PUBLIC libMlCore)
target_link_libraries(libMlApi PUBLIC libMlMathsCommon)
target_link_libraries(libMlApi PUBLIC libMlMathsTimeSeries)
target_link_libraries(libMlApi PUBLIC libMlMathsAnalytics)
target_link_libraries(libMlApi PUBLIC libMlModel)
if (WIN32)
  target_link_libraries(libMlApi PUBLIC ws2_32)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(libMlApi PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS libMlApi RUNTIME DESTINATION ${DYNAMIC_LIB_DIR} ARCHIVE DESTINATION ${IMPORT_LIB_DIR})
install(FILES $<TARGET_PDB_FILE:libMlApi> DESTINATION ${DYNAMIC_LIB_DIR} OPTIONAL)
