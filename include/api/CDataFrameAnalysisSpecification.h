/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameAnalysisSpecification_h
#define INCLUDED_ml_api_CDataFrameAnalysisSpecification_h

#include <core/CFastMutex.h>

#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

#include <cstddef>
#include <string>
#include <thread>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;
}
namespace api {
class CDataFrameAnalysisRunner;
class CDataFrameAnalysisRunnerFactory;

//! \brief Parses a complete specification for running a core::CDataFrame analysis
//! and supports launching that analysis on a specified frame object.
//!
//! DESCRIPTION:\n
//! This manages extracting all configuration for a particular analysis from a JSON
//! header which is passed to the data_frame_analyzer command before any data. This
//! creates and owns an analysis runner object which is also configured by the header.
//! The analysis is run asynchronously via the CDataFrameAnalysisSpecification::run
//! method which returns a handle to the runner to retrieve progress and errors.
class API_EXPORT CDataFrameAnalysisSpecification {
public:
    using TStrVec = std::vector<std::string>;
    using TRunnerPtr = std::unique_ptr<CDataFrameAnalysisRunner>;
    using TRunnerFactoryPtr = std::unique_ptr<CDataFrameAnalysisRunnerFactory>;
    using TRunnerFactoryPtrVec = std::vector<TRunnerFactoryPtr>;

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
    //!   "analysis": {
    //!     "name": <string>,
    //!     "parameters": <object>
    //!   }
    //! }
    //! </CODE>
    //!
    //! \param[in] runnerFactories Plugins for the supported analyses.
    //! \param[in] jsonSpecification The specification as a JSON object.
    //! \note The analysis name must be one of the supported analysis types.
    //! \note All constraints must be positive.
    //! \note The parameters, if any, must be consistent for the analysis type.
    //! \note If this fails the state is set to bad and the analysis will not run.
    CDataFrameAnalysisSpecification(TRunnerFactoryPtrVec runnerFactories,
                                    const std::string& jsonSpecification);

    //! Check if the specification is bad.
    bool bad() const;

    //! \return The number of rows in the frame.
    std::size_t rows() const;

    //! \return The number of columns in the input frame.
    std::size_t cols() const;

    //! \return The memory limit for the process.
    std::size_t memoryLimit() const;

    //! \return The number of threads the analysis can use.
    std::size_t threads() const;

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

private:
    bool m_Bad = false;
    std::size_t m_Rows = 0;
    std::size_t m_Cols = 0;
    std::size_t m_MemoryLimit = 0;
    std::size_t m_Threads = 0;
    // TODO Sparse table support
    // double m_TableLoadFactor = 0.0;
    TRunnerFactoryPtrVec m_RunnerFactories;
    TRunnerPtr m_Runner;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisSpecification_h
