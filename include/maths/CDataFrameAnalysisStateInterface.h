/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameAnalysisInterface_h
#define INCLUDED_ml_maths_CDataFrameAnalysisInterface_h

#include <maths/ImportExport.h>

#include <cstdint>

namespace ml {
namespace maths {

class MATHS_EXPORT CDataFrameAnalysisStateInterface {
public:
    using TProgressCallback = std::function<void(double)>;
    using TMemoryUsageCallback = std::function<void(std::int64_t)>;

public:
    virtual ~CDataFrameAnalysisStateInterface() = default;
    virtual void updateMemoryUsage(std::int64_t /*delta*/){};
    virtual void updateProgress(double /*fractionalProgress*/){};
    virtual void nextStep(uint32_t /*step*/){};
    TProgressCallback progressCallback() {
        return [&](double fractionalProgress) {
            this->updateProgress(fractionalProgress);
        };
    };
    TMemoryUsageCallback memoryUsageCallback() {
        return [&](std::int64_t delta) { this->updateMemoryUsage(delta); };
    };
};
}
}

#endif //INCLUDED_ml_maths_CDataFrameAnalysisInterface_h
