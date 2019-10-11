/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameAnalysisRunner_h
#define INCLUDED_ml_api_CDataFrameAnalysisRunner_h

#include <core/CFastMutex.h>
#include <core/CStatePersistInserter.h>

#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;
class CRapidJsonConcurrentLineWriter;
namespace data_frame_detail {
class CRowRef;
}
}
namespace api {
class CDataFrameAnalysisSpecification;
class CMemoryUsageEstimationResultJsonWriter;

//! \brief Hierarchy for running a specific core::CDataFrame analyses.
//!
//! DESCRIPTION:\n
//! This hierarchy manages the running of specific analyses on a core::CDataFrame
//! object. It provides common interface for reporting progress and errors back to
//! calling code and starting an analysis.
//!
//! IMPLEMENTATION:\n
//! Particular analyses are specified by a JSON object which is passed as a header
//! to the data_frame_analyzer command before any data. It is the responsibility of
//! the CDataFrameAnalysisSpecification to parse this header although it passes off
//! the reading of the analysis parameters object to implementations of this runner.
//! Therefore CDataFrameAnalysisSpecification is also responsible for creating an
//! appropriate runner object for the data_frame_analyzer command. A particular
//! analysis is launched by the CDataFrameAnalysisSpecification::run method which
//! returns a reference to the appropriate CDataFrameAnalysisRunner implementation.
//!
//! This launches the work to do the analysis in a background thread so that the
//! main thread remains responsive and can periodically report progress and errors.
//!
//! No mechanism is provided to cancel the work (yet) because it is anticipated
//! that this will be probably be achieved by killing the process and it is too
//! early to determine how to implement a good cooperative interrupt scheme.
class API_EXPORT CDataFrameAnalysisRunner {
public:
    using TBoolVec = std::vector<bool>;
    using TStrVec = std::vector<std::string>;
    using TRowRef = core::data_frame_detail::CRowRef;
    using TProgressRecorder = std::function<void(double)>;

public:
    //! The intention is that concrete objects of this hierarchy are constructed
    //! by the factory class.
    CDataFrameAnalysisRunner(const CDataFrameAnalysisSpecification& spec);
    virtual ~CDataFrameAnalysisRunner();

    CDataFrameAnalysisRunner(const CDataFrameAnalysisRunner&) = delete;
    CDataFrameAnalysisRunner& operator=(const CDataFrameAnalysisRunner&) = delete;

    //! This computes the execution strategy for the analysis, including how
    //! the data frame will be stored, the size of the partition and the maximum
    //! number of rows per subset.
    void computeAndSaveExecutionStrategy();

    //! Estimates memory usage in two cases:
    //!   1. disk is not used (the whole data frame fits in main memory)
    //!   2. disk is used (only one partition needs to be loaded to main memory)
    void estimateMemoryUsage(CMemoryUsageEstimationResultJsonWriter& writer) const;

    //! Check if the data frame for this analysis should use in or out of core
    //! storage.
    bool storeDataFrameInMainMemory() const;

    //! \return The number of partitions to use when analysing the data frame.
    //! \note If this is greater than one then the data frame should be stored
    //! on disk. The run method is responsible for copying the relevant pieces
    //! into main memory during an analysis.
    std::size_t numberPartitions() const;

    //! Get the maximum permitted partition size in numbers of rows.
    std::size_t maximumNumberRowsPerPartition() const;

    //! \return The number of columns this analysis appends.
    virtual std::size_t numberExtraColumns() const = 0;

    //! \return Indicator of columns for which empty value should be treated as missing.
    virtual TBoolVec columnsForWhichEmptyIsMissing(const TStrVec& fieldNames) const;

    //! Write the extra columns of \p row added by the analysis to \p writer.
    //!
    //! This should create a new object of the form:
    //! <pre>
    //! {
    //!   "name of column n":   "value of column n",
    //!   "name of column n+1": "value of column n+1",
    //!   ...
    //! }
    //! </pre>
    //! with one named member for each column added.
    //!
    //! \param[in] frame The data frame for which to write results.
    //! \param[in] row The row to write the columns added by this analysis.
    //! \param[in,out] writer The stream to which to write the extra columns.
    virtual void writeOneRow(const core::CDataFrame& frame,
                             const TRowRef& row,
                             core::CRapidJsonConcurrentLineWriter& writer) const = 0;

    //! Checks whether the analysis is already running and if not launches it
    //! in the background.
    //!
    //! \note The thread calling this is expected to be nearly always idle, i.e.
    //! just progress monitoring, so this doesn't count towards the thread limit.
    void run(core::CDataFrame& frame);

    //! This waits to until the analysis has finished and joins the thread.
    void waitToFinish();

    //! \return True if the running analysis has finished.
    bool finished() const;

    //! \return The progress of the analysis in the range [0,1] being an estimate
    //! of the proportion of total work complete for a single run.
    double progress() const;

protected:
    using TStatePersister =
        std::function<void(std::function<void(core::CStatePersistInserter&)>)>;

protected:
    const CDataFrameAnalysisSpecification& spec() const;
    TProgressRecorder progressRecorder();
    std::size_t estimateMemoryUsage(std::size_t totalNumberRows,
                                    std::size_t partitionNumberRows,
                                    std::size_t numberColumns) const;

    //! \return Callback function for writing state using given persist inserter
    TStatePersister statePersister();

private:
    virtual void runImpl(core::CDataFrame& frame) = 0;
    virtual std::size_t estimateBookkeepingMemoryUsage(std::size_t numberPartitions,
                                                       std::size_t totalNumberRows,
                                                       std::size_t partitionNumberRows,
                                                       std::size_t numberColumns) const = 0;

    //! This adds \p fractionalProgess to the current progress.
    //!
    //! \note The caller should try to ensure that the sum of the values added
    //! at the end of the analysis is equal to one.
    //! \note This is converted to an integer - so we can atomically add - by
    //! scaling by 1024. Therefore, this shouldn't be called with values less
    //! than 0.001. In fact, it is unlikely that such high resolution is needed
    //! and typically this would be called significantly less frequently.
    void recordProgress(double fractionalProgress);
    void setToFinished();

private:
    const CDataFrameAnalysisSpecification& m_Spec;

    std::size_t m_NumberPartitions = 0;
    std::size_t m_MaximumNumberRowsPerPartition = 0;

    std::atomic_bool m_Finished;
    std::atomic_size_t m_FractionalProgress;

    std::thread m_Runner;
};

//! \brief Makes a core::CDataFrame analysis runner.
class API_EXPORT CDataFrameAnalysisRunnerFactory {
public:
    using TRunnerUPtr = std::unique_ptr<CDataFrameAnalysisRunner>;

public:
    virtual ~CDataFrameAnalysisRunnerFactory() = default;
    virtual const std::string& name() const = 0;

    TRunnerUPtr make(const CDataFrameAnalysisSpecification& spec) const;
    TRunnerUPtr make(const CDataFrameAnalysisSpecification& spec,
                     const rapidjson::Value& jsonParameters) const;

private:
    virtual TRunnerUPtr makeImpl(const CDataFrameAnalysisSpecification& spec) const = 0;
    virtual TRunnerUPtr makeImpl(const CDataFrameAnalysisSpecification& spec,
                                 const rapidjson::Value& jsonParameters) const = 0;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisRunner_h
