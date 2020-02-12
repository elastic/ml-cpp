/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameAnalysisInstrumentationInterface_h
#define INCLUDED_ml_maths_CDataFrameAnalysisInstrumentationInterface_h

#include <maths/ImportExport.h>

#include <cstdint>

namespace ml {
namespace maths {

//! \brief Interface class for collecting data frame analysis job statistics owned
//! by the maths module.
class MATHS_EXPORT CDataFrameAnalysisInstrumentationInterface {
public:
    using TProgressCallback = std::function<void(double)>;
    using TMemoryUsageCallback = std::function<void(std::int64_t)>;

public:
    virtual ~CDataFrameAnalysisInstrumentationInterface() = default;
    //! Adds \p delta to the memory usage statistics.
    virtual void updateMemoryUsage(std::int64_t delta) = 0;
    //! This adds \p fractionalProgess to the current progress.
    //!
    //! \note The caller should try to ensure that the sum of the values added
    //! at the end of the analysis is equal to one.
    //! \note This is converted to an integer - so we can atomically add - by
    //! scaling by 1024. Therefore, this shouldn't be called with values less
    //! than 0.001. In fact, it is unlikely that such high resolution is needed
    //! and typically this would be called significantly less frequently.
    virtual void updateProgress(double fractionalProgress) = 0;
    //! Trigger the next step of the job. This will initiate writing the job state
    //! to the results pipe.
    virtual void nextStep(std::uint32_t step) = 0;
    //! Factory for the updateProgress() callback function object.
    TProgressCallback progressCallback() {
        return [this](double fractionalProgress) {
            this->updateProgress(fractionalProgress);
        };
    }
    //! Factory for the updateMemoryUsage() callback function object.
    TMemoryUsageCallback memoryUsageCallback() {
        return [this](std::int64_t delta) { this->updateMemoryUsage(delta); };
    }
};

//! \brief Dummies out all instrumentation.
class MATHS_EXPORT CDataFrameAnalysisInstrumentationStub final : public CDataFrameAnalysisInstrumentationInterface {
    void updateMemoryUsage(std::int64_t) override {}
    void updateProgress(double) override {}
    void nextStep(std::uint32_t) override {}
};
}
}

#endif //INCLUDED_ml_maths_CDataFrameAnalysisInstrumentationInterface_h
