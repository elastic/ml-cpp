/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameAnalysisSpecification_h
#define INCLUDED_ml_api_CDataFrameAnalysisSpecification_h

#include <core/CFastMutex.h>

#include <api/CDataFrameAnalysisRunner.h>
#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;
class CTemporaryDirectory;
}
namespace api {

//! \brief Parses a complete specification for running a core::CDataFrame analysis
//! and supports launching that analysis on a specified frame object.
//!
//! DESCRIPTION:\n
//! This manages extracting all configuration for a particular analysis from a JSON
//! header which is passed to the data_frame_analyzer command before any data. This
//! creates and owns an analysis runner object which is also configured by the header.
//! The analysis is run asynchronously via the CDataFrameAnalysisSpecification::run
//! method which returns a handle to the runner to retrieve progress, errors and other
//! performance statistics.
class API_EXPORT CDataFrameAnalysisSpecification {
public:
    using TStrVec = std::vector<std::string>;
    using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
    using TTemporaryDirectoryPtr = std::shared_ptr<core::CTemporaryDirectory>;
    using TDataFrameUPtrTemporaryDirectoryPtrPr =
        std::pair<TDataFrameUPtr, TTemporaryDirectoryPtr>;
    using TRunnerUPtr = std::unique_ptr<CDataFrameAnalysisRunner>;
    using TRunnerFactoryUPtr = std::unique_ptr<CDataFrameAnalysisRunnerFactory>;
    using TRunnerFactoryUPtrVec = std::vector<TRunnerFactoryUPtr>;
    using TFatalErrorHandler = std::function<void(std::string)>;

public:
    //! Inititialize from a JSON object.
    //!
    //! The specification has the following expected form:
    //! <CODE>
    //! {
    //!   "rows": <integer>,
    //!   "cols": <integer>,
    //!   "memory_limit": <integer>,
    //!   "threads": <integer>,
    //!   "temp_dir": <string>,
    //!   "analysis": {
    //!     "name": <string>,
    //!     "parameters": <object>
    //!   }
    //! }
    //! </CODE>
    //!
    //! \param[in] jsonSpecification The specification as a JSON object.
    //! \note The analysis name must be one of the supported analysis types.
    //! \note All constraints must be positive.
    //! \note The parameters, if any, must be consistent for the analysis type.
    //! \note If this fails the state is set to bad and the analysis will not run.
    //! \note temp_dir Is a directory which can be used to store the data frame
    //! out-of-core if we can't meet the memory constraint for the analysis without
    //! partitioning.
    CDataFrameAnalysisSpecification(const std::string& jsonSpecification);

    //! This construtor provides support for custom analysis types and is mainly
    //! intended for testing.
    //!
    //! \param[in] runnerFactories Plugins for the supported analyses.
    CDataFrameAnalysisSpecification(TRunnerFactoryUPtrVec runnerFactories,
                                    const std::string& jsonSpecification);

    CDataFrameAnalysisSpecification(const CDataFrameAnalysisSpecification&) = delete;
    CDataFrameAnalysisSpecification& operator=(const CDataFrameAnalysisSpecification&) = delete;
    CDataFrameAnalysisSpecification(CDataFrameAnalysisSpecification&&) = delete;
    CDataFrameAnalysisSpecification& operator=(CDataFrameAnalysisSpecification&&) = delete;

    //! \return The number of rows in the frame.
    std::size_t numberRows() const;

    //! \return The number of columns in the input frame.
    std::size_t numberColumns() const;

    //! \return The number of columns the analysis configured to run will append
    //! to the data frame.
    std::size_t numberExtraColumns() const;

    //! \return The memory usage limit for the process.
    std::size_t memoryLimit() const;

    //! \return The number of threads the analysis can use.
    std::size_t numberThreads() const;

    //! Make a data frame suitable for this analysis specification.
    //!
    //! This chooses the storage strategy based on the analysis constraints and
    //! the number of rows and target number of columns and reserves capacity as
    //! appropriate.
    TDataFrameUPtrTemporaryDirectoryPtrPr makeDataFrame();

    //! Run the analysis in a background thread.
    //!
    //! This returns a handle to the object responsible for running the analysis.
    //! Destroying this object waits for the analysis to complete and joins the
    //! thread. It is expected that the caller will mainly sleep and wake up
    //! periodically to report progess, errors and see if it has finished.
    //!
    //! \return frame The data frame to analyse.
    //! \note The commit of the results of the analysis is atomic per partition.
    //! \warning This assumes that there is no access to the data frame in the
    //! calling thread until the runner has finished.
    CDataFrameAnalysisRunner* run(core::CDataFrame& frame) const;

private:
    void initializeRunner(const char* name, const rapidjson::Value& analysis);
    static TFatalErrorHandler defaultFatalErrorHandler();

private:
    std::size_t m_NumberRows = 0;
    std::size_t m_NumberColumns = 0;
    std::size_t m_MemoryLimit = 0;
    std::size_t m_NumberThreads = 0;
    std::string m_TemporaryDirectory;
    // TODO Sparse table support
    // double m_TableLoadFactor = 0.0;
    TRunnerFactoryUPtrVec m_RunnerFactories;
    TRunnerUPtr m_Runner;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisSpecification_h
