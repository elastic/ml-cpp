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

project("ML Maths Analytics")

set(ML_DEPENDENCIES third_party)
set(ML_LINK_LIBRARIES
  ${Boost_LIBRARIES}
  MlMathsCommon
  MlCore)

ml_add_library(MlMathsAnalytics SHARED
  CBoostedTree.cc
  CBoostedTreeFactory.cc
  CBoostedTreeHyperparameters.cc
  CBoostedTreeImpl.cc
  CBoostedTreeLeafNodeStatistics.cc
  CBoostedTreeLeafNodeStatisticsThreading.cc
  CBoostedTreeLoss.cc
  CBoostedTreeUtils.cc
  CDataFrameCategoryEncoder.cc
  CDataFramePredictiveModel.cc
  CDataFrameUtils.cc
  CMic.cc
  COutliers.cc
  CTreeShapFeatureImportance.cc
  )
