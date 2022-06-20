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

project("ML Test")

set(ML_DEPENDENCIES third_party)
set(ML_LINK_LIBRARIES 
  ${Boost_LIBRARIES_WITH_UNIT_TEST}
  MlApi
  MlCore
  MlMathsAnalytics
  MlMathsCommon
  )

ml_add_library(MlTest SHARED
  CBoostTestXmlOutput.cc
  CDataFrameAnalysisSpecificationFactory.cc
  CDataFrameAnalyzerTrainingFactory.cc
  CMultiFileDataAdder.cc
  CMultiFileSearcher.cc
  CProgramCounterClearingFixture.cc
  CRandomNumbers.cc
  CShellCmdEscape.cc
  CTestObserver.cc
  CTestTmpDir.cc
  CThreadDataReader.cc
  CThreadDataWriter.cc
  CTimeSeriesTestData.cc
  )
