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

project("ML Maths Time Series unit tests")

set (SRCS
  Main.cc
  CAdaptiveBucketingTest.cc
  CCalendarComponentAdaptiveBucketingTest.cc
  CCalendarComponentTest.cc
  CCalendarCyclicTestTest.cc
  CCalendarFeatureTest.cc
  CCountMinSketchTest.cc
  CDecayRateControllerTest.cc
  CExpandingWindowTest.cc
  CForecastTest.cc
  CMathsMemoryTest.cc
  CSeasonalComponentAdaptiveBucketingTest.cc
  CSeasonalComponentTest.cc
  CSeasonalTimeTest.cc
  CSignalTest.cc
  CTimeSeriesDecompositionTest.cc
  CTimeSeriesModelTest.cc
  CTimeSeriesMultibucketFeaturesTest.cc
  CTimeSeriesSegmentationTest.cc
  CTimeSeriesTestForChangeTest.cc
  CTimeSeriesTestForSeasonalityTest.cc
  CTrendComponentTest.cc
  TestUtils.cc
  )

set(ML_LINK_LIBRARIES
  ${Boost_LIBRARIES_WITH_UNIT_TEST}
  MlCore
  MlMathsCommon
  MlMathsTimeSeries
  MlTest
  )

ml_add_test_executable(maths_time_series ${SRCS})
