/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameAnalysisState_h
#define INCLUDED_ml_api_CDataFrameAnalysisState_h

#include <core/CProgramCounters.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <maths/CDataFrameAnalysisStateInterface.h>

#include <cstdint>

namespace ml {
namespace api {

class CDataFrameAnalysisState : public maths::CDataFrameAnalysisStateInterface {
public:
    CDataFrameAnalysisState();

    virtual ~CDataFrameAnalysisState() = default;

    void updateMemoryUsage(std::int64_t delta) override;

    //! This adds \p fractionalProgess to the current progress.
    //!
    //! \note The caller should try to ensure that the sum of the values added
    //! at the end of the analysis is equal to one.
    //! \note This is converted to an integer - so we can atomically add - by
    //! scaling by 1024. Therefore, this shouldn't be called with values less
    //! than 0.001. In fact, it is unlikely that such high resolution is needed
    //! and typically this would be called significantly less frequently.
    void updateProgress(double fractionalProgress) override;
    void setToFinished();
    //    virtual void updateQualityOfResults() = 0;
    //    virtual void updateParameters() = 0;
    void writeState(core::CRapidJsonConcurrentLineWriter& writer);

    //! \return True if the running analysis has finished.
    bool finished() const;

    //! \return The progress of the analysis in the range [0,1] being an estimate
    //! of the proportion of total work complete for a single run.
    double progress() const;

    void resetProgress();

protected:
    virtual counter_t::ECounterTypes memoryCounterType() = 0;

private:
    std::atomic<std::int64_t> m_Memory;
    std::atomic_size_t m_FractionalProgress;
    std::atomic_bool m_Finished;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisState_h
