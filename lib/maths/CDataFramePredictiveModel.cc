/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDataFramePredictiveModel.h>

namespace ml {
namespace maths {

const std::string CDataFramePredictiveModel::SHAP_PREFIX{"feature_importance."};

CDataFramePredictiveModel::CDataFramePredictiveModel(core::CDataFrame& frame,
                                                     TTrainingStateCallback recordTrainingState)
    : m_Frame{frame}, m_RecordTrainingState(std::move(recordTrainingState)) {
}

core::CDataFrame& CDataFramePredictiveModel::frame() const {
    return m_Frame;
}

const CDataFramePredictiveModel::TTrainingStateCallback&
CDataFramePredictiveModel::trainingStateRecorder() const {
    return m_RecordTrainingState;
}
}
}
