/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameAnalysisInstrumentation_h
#define INCLUDED_ml_api_CDataFrameAnalysisInstrumentation_h

#include <core/CProgramCounters.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <maths/CDataFrameAnalysisInstrumentationInterface.h>

#include <api/ImportExport.h>

#include <atomic>
#include <cstdint>

namespace ml {
namespace api {

//! \brief Instrumentation class for collecting data frame analysis job statistics.
//!
//! DESCRIPTION:\n
//! Responsible for collecting data frame analysis job statistics, i.e. memory usage,
//! progress, parameters, quality of results. The class also implements the functionality to
//! write the state at different iteration into the results pipe.
class API_EXPORT CDataFrameAnalysisInstrumentation
    : public maths::CDataFrameAnalysisInstrumentationInterface {

public:
    CDataFrameAnalysisInstrumentation();

    //! Adds \p delta to the memory usage statistics.
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

    //! \return True if the running analysis has finished.
    bool finished() const;

    //! \return The progress of the analysis in the range [0,1] being an estimate
    //! of the proportion of total work complete for a single run.
    double progress() const;

    //! Reset variables related to the job progress.
    void resetProgress();

    //! Set pointer to the writer object.
    void writer(core::CRapidJsonConcurrentLineWriter* writer);

    //! Trigger the next step of the job. This will initiate writing the job state
    //! to the results pipe.
    void nextStep(std::uint32_t step) override;

    //! \return The peak memory usage.
    std::int64_t memory() const;

protected:
    virtual counter_t::ECounterTypes memoryCounterType() = 0;

private:
    void writeProgress(std::uint32_t step);
    void writeMemory(std::uint32_t step);
    void writeState(uint32_t step);

private:
    std::atomic_bool m_Finished;
    std::atomic_size_t m_FractionalProgress;
    std::atomic<std::int64_t> m_Memory;
    core::CRapidJsonConcurrentLineWriter* m_Writer;
};

class API_EXPORT CDataFrameOutliersInstrumentation final
    : public CDataFrameAnalysisInstrumentation {
protected:
    counter_t::ECounterTypes memoryCounterType() override;
};

class API_EXPORT CDataFrameTrainBoostedTreeInstrumentation final
    : public CDataFrameAnalysisInstrumentation {
protected:
    counter_t::ECounterTypes memoryCounterType() override;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisInstrumentation_h
