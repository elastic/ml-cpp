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

#include <rapidjson/document.h>

#include <atomic>
#include <cstdint>
#include <unordered_map>

namespace ml {
namespace api {

//! \brief Instrumentation class for collecting data frame analysis job statistics.
//!
//! DESCRIPTION:\n
//! Responsible for collecting data frame analysis job statistics, i.e. memory usage,
//! progress, parameters, quality of results. The class also implements the functionality to
//! write the state at different iteration into the results pipe.
class API_EXPORT CDataFrameAnalysisInstrumentation
    : virtual public maths::CDataFrameAnalysisInstrumentationInterface {
public:
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;

public:
    explicit CDataFrameAnalysisInstrumentation(const std::string& jobId);

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

    //! Record that the analysis is complete.
    void setToFinished();

    //! \return True if the running analysis has finished.
    bool finished() const;

    //! \return The progress of the analysis in the range [0,1] being an estimate
    //! of the proportion of total work complete for a single run.
    double progress() const;

    //! Reset variables related to the job progress.
    void resetProgress();

    //! Set pointer to the writer object.
    void writer(TRapidJsonWriter* writer);

    //! Trigger the next step of the job. This will initiate writing the job state
    //! to the results pipe.
    void nextStep(const std::string& phase = "") override;

    //! \return The peak memory usage.
    std::int64_t memory() const;

    const std::string& jobId() const;

protected:
    virtual counter_t::ECounterTypes memoryCounterType() = 0;
    TRapidJsonWriter* writer();

private:
    void writeMemory(std::int64_t timestamp);
    virtual void writeAnalysisStats(std::int64_t /* timestamp */){};
    virtual void writeState();

private:
    std::atomic_bool m_Finished;
    std::atomic_size_t m_FractionalProgress;
    std::atomic<std::int64_t> m_Memory;
    TRapidJsonWriter* m_Writer;
    std::string m_JobId;
};

class API_EXPORT CDataFrameOutliersInstrumentation final
    : public CDataFrameAnalysisInstrumentation,
      public maths::CDataFrameOutliersInstrumentationInterface {
public:
    explicit CDataFrameOutliersInstrumentation(const std::string& jobId)
        : CDataFrameAnalysisInstrumentation(jobId){};

protected:
    counter_t::ECounterTypes memoryCounterType() override;

private:
    void writeAnalysisStats(std::int64_t timestamp) override;
};

class API_EXPORT CDataFrameTrainBoostedTreeInstrumentation final
    : public CDataFrameAnalysisInstrumentation,
      public maths::CDataFrameTrainBoostedTreeInstrumentationInterface {
public:
    explicit CDataFrameTrainBoostedTreeInstrumentation(const std::string& jobId)
        : CDataFrameAnalysisInstrumentation(jobId){};

    void type(EStatsType type) override;
    void iteration(std::size_t iteration) override;
    void iterationTime(std::uint64_t delta) override;
    void lossType(const std::string& lossType) override;
    void lossValues(std::string fold, TDoubleVec&& lossValues) override;
    void numFolds(std::size_t numFolds) override;
    void hyperparameters(const SHyperparameters& hyperparameters) override;
    SHyperparameters& hyperparameters() override { return m_Hyperparameters; };

protected:
    counter_t::ECounterTypes memoryCounterType() override;

private:
    using TLossMap = std::unordered_map<std::string, TDoubleVec>;

private:
    void writeAnalysisStats(std::int64_t timestamp) override;
    void writeHyperparameters(rapidjson::Value& parentObject);
    void writeValidationLoss(rapidjson::Value& parentObject);
    void writeTimingStats(rapidjson::Value& parentObject);

private:
    EStatsType m_Type;
    std::size_t m_Iteration;
    std::uint64_t m_IterationTime;
    std::uint64_t m_ElapsedTime;
    std::string m_LossType;
    TLossMap m_LossValues;
    std::size_t m_NumFolds;
    SHyperparameters m_Hyperparameters;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisInstrumentation_h
