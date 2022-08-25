/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <maths/analytics/CDataFramePredictiveModel.h>

namespace ml {
namespace maths {
namespace analytics {
CDataFramePredictiveModel::CDataFramePredictiveModel(core::CDataFrame& frame,
                                                     TTrainingStateCallback recordTrainingState)
    : m_Frame{frame}, m_RecordTrainingState(std::move(recordTrainingState)) {
}

const core::CDataFrame& CDataFramePredictiveModel::trainingData() const {
    return m_Frame;
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
}
