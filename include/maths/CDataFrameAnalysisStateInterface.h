/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameAnalysisInterface_h
#define INCLUDED_ml_maths_CDataFrameAnalysisInterface_h

#include <cstdint>

namespace ml {
namespace maths {

class CDataFrameAnalysisStateInterface {
public:
    virtual ~CDataFrameAnalysisStateInterface() = default;
    virtual void updateMemoryUsage(std::int64_t /*delta*/){};
    virtual void updateProgress(double /*fractionalProgress*/){};
};
}
}

#endif //INCLUDED_ml_maths_CDataFrameAnalysisInterface_h
