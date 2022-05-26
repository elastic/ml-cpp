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

project("ML Maths Time Series")

set(ML_DEPENDENCIES eigen)
set(ML_LINK_LIBRARIES 
    ${Boost_LIBRARIES}
    MlMathsCommon
    MlCore)

ml_add_library(MlMathsTimeSeries SHARED
    CAdaptiveBucketing.cc 
    CCalendarComponent.cc 
    CCalendarComponentAdaptiveBucketing.cc 
    CCalendarCyclicTest.cc 
    CCalendarFeature.cc 
    CCountMinSketch.cc 
    CDecayRateController.cc 
    CDecompositionComponent.cc 
    CExpandingWindow.cc 
    CModelStateSerialiser.cc 
    CSeasonalComponent.cc 
    CSeasonalComponentAdaptiveBucketing.cc 
    CSeasonalTime.cc 
    CSignal.cc 
    CTimeSeriesDecomposition.cc 
    CTimeSeriesDecompositionDetail.cc 
    CTimeSeriesDecompositionStateSerialiser.cc 
    CTimeSeriesDecompositionStub.cc 
    CTimeSeriesModel.cc 
    CTimeSeriesMultibucketFeatureSerialiser.cc 
    CTimeSeriesSegmentation.cc 
    CTimeSeriesTestForChange.cc 
    CTimeSeriesTestForSeasonality.cc 
    CTrendComponent.cc 
)
