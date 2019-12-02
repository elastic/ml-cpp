/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDataFrameRegressionModel.h>

namespace ml {
namespace maths {

const std::string CDataFrameRegressionModel::SHAP_PREFIX{"shap_"};

CDataFrameRegressionModel::CDataFrameRegressionModel(core::CDataFrame& frame,
                                                     TProgressCallback recordProgress,
                                                     TMemoryUsageCallback recordMemoryUsage,
                                                     TTrainingStateCallback recordTrainingState)
    : m_Frame{frame}, m_RecordProgress{std::move(recordProgress)},
      m_RecordMemoryUsage{std::move(recordMemoryUsage)},
      m_RecordTrainingState(std::move(recordTrainingState)) {
}

core::CDataFrame& CDataFrameRegressionModel::frame() const {
    return m_Frame;
}

const CDataFrameRegressionModel::TProgressCallback&
CDataFrameRegressionModel::progressRecorder() const {
    return m_RecordProgress;
}

const CDataFrameRegressionModel::TMemoryUsageCallback&
CDataFrameRegressionModel::memoryUsageRecorder() const {
    return m_RecordMemoryUsage;
}

const CDataFrameRegressionModel::TTrainingStateCallback&
CDataFrameRegressionModel::trainingStateRecorder() const {
    return m_RecordTrainingState;
}
}
}
