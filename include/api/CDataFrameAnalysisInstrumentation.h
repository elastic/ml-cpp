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
#include <memory>
#include <unordered_map>

namespace ml {
namespace api {

//! \brief Instrumentation class for collecting data frame analysis job statistics.
//!
//! DESCRIPTION:\n
//! Responsible for collecting data frame analysis job statistics, i.e. memory usage,
//! progress, parameters, quality of results. This also implements the functionality
//! to write the JSON statistics to a specified output stream in a thread safe manner.
class API_EXPORT CDataFrameAnalysisInstrumentation
    : virtual public maths::CDataFrameAnalysisInstrumentationInterface {
public:
    //! \brief Set the output stream for the lifetime of this object.
    class API_EXPORT CScopeSetOutputStream {
    public:
        CScopeSetOutputStream(CDataFrameAnalysisInstrumentation& instrumentation,
                              core::CJsonOutputStreamWrapper& outStream);
        ~CScopeSetOutputStream();

        CScopeSetOutputStream(const CScopeSetOutputStream&) = delete;
        CScopeSetOutputStream& operator=(const CScopeSetOutputStream&) = delete;

    private:
        CDataFrameAnalysisInstrumentation& m_Instrumentation;
    };

public:
    //! Constructs an instrumentation object an analytics job with a given \p jobId.
    explicit CDataFrameAnalysisInstrumentation(const std::string& jobId);

    //! Adds \p delta to the memory usage statistics.
    void updateMemoryUsage(std::int64_t delta) override;

    //! This adds \p fractionalProgress to the current progress.
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

    //! Trigger the next step of the job. This will initiate writing the job state
    //! to the results pipe.
    //! \todo use \p phase to tag different phases of the analysis job.
    void nextStep(const std::string& phase = "") override;

    //! \return The peak memory usage.
    std::int64_t memory() const;

    //! \return The id of the data frame analytics job.
    const std::string& jobId() const;

protected:
    using TWriter = core::CRapidJsonConcurrentLineWriter;
    using TWriterUPtr = std::unique_ptr<TWriter>;

protected:
    virtual counter_t::ECounterTypes memoryCounterType() = 0;
    TWriter* writer();

private:
    void writeMemory(std::int64_t timestamp);
    virtual void writeAnalysisStats(std::int64_t timestamp) = 0;
    virtual void writeState();

private:
    std::string m_JobId;
    std::atomic_bool m_Finished;
    std::atomic_size_t m_FractionalProgress;
    std::atomic<std::int64_t> m_Memory;
    TWriterUPtr m_Writer;
};

//! \brief Instrumentation class for Outlier Detection jobs.
class API_EXPORT CDataFrameOutliersInstrumentation final
    : public CDataFrameAnalysisInstrumentation,
      public maths::CDataFrameOutliersInstrumentationInterface {
public:
    explicit CDataFrameOutliersInstrumentation(const std::string& jobId)
        : CDataFrameAnalysisInstrumentation(jobId) {}

protected:
    counter_t::ECounterTypes memoryCounterType() override;

private:
    void writeAnalysisStats(std::int64_t timestamp) override;
};

//! \brief Instrumentation class for Supervised Learning jobs.
//!
//! DESCRIPTION:\n
//! This class extends CDataFrameAnalysisInstrumentation with setters
//! for hyperparameters, validation loss results, and job timing.
class API_EXPORT CDataFrameTrainBoostedTreeInstrumentation final
    : public CDataFrameAnalysisInstrumentation,
      public maths::CDataFrameTrainBoostedTreeInstrumentationInterface {
public:
    explicit CDataFrameTrainBoostedTreeInstrumentation(const std::string& jobId)
        : CDataFrameAnalysisInstrumentation(jobId) {}

    //! Supervised learning job \p type, can be E_Regression or E_Classification.
    void type(EStatsType type) override;
    //! Current \p iteration number.
    void iteration(std::size_t iteration) override;
    //! Run time of the iteration.
    void iterationTime(std::uint64_t delta) override;
    //! Type of the validation loss result, e.g. "mse".
    void lossType(const std::string& lossType) override;
    //! List of \p lossValues of validation error for the given \p fold.
    void lossValues(std::size_t fold, TDoubleVec&& lossValues) override;
    //! \return Structure contains hyperparameters.
    SHyperparameters& hyperparameters() override { return m_Hyperparameters; }

protected:
    counter_t::ECounterTypes memoryCounterType() override;

private:
    using TLossVec = std::vector<std::pair<std::size_t, TDoubleVec>>;

private:
    void writeAnalysisStats(std::int64_t timestamp) override;
    void writeHyperparameters(rapidjson::Value& parentObject);
    void writeValidationLoss(rapidjson::Value& parentObject);
    void writeTimingStats(rapidjson::Value& parentObject);
    void reset();

private:
    EStatsType m_Type;
    std::size_t m_Iteration;
    std::uint64_t m_IterationTime;
    std::uint64_t m_ElapsedTime = 0;
    std::string m_LossType;
    TLossVec m_LossValues;
    SHyperparameters m_Hyperparameters;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisInstrumentation_h
