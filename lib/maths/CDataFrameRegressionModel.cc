/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDataFrameRegressionModel.h>

namespace ml {
namespace maths {

CDataFrameRegressionModel::CDataFrameRegressionModel(core::CDataFrame& frame,
                                                     TProgressCallback recordProgress)
    : m_Frame{frame}, m_RecordProgress{recordProgress} {
}

core::CDataFrame& CDataFrameRegressionModel::frame() const {
    return m_Frame;
}

const CDataFrameRegressionModel::TProgressCallback&
CDataFrameRegressionModel::progressRecorder() const {
    return m_RecordProgress;
}
}
}
