/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameAnalysisRunner_h
#define INCLUDED_ml_api_CDataFrameAnalysisRunner_h

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
class CDataFrameAnalysisSpecification;

//! \brief Hierarchy for running a specific core::CDataFrame analyses.
//!
//! DESCRIPTION:\n
//! This hierarchy manages the running of specific analyses on a core::CDataFrame
//! object. It provides common interface for reporting progress and errors back to
//! calling code and starting an analysis.
//!
//! IMPLEMENTATION:\n
//! Particular analyses are specified by a JSON object which is passed as a header
//! before any data to the data_frame_analyzer command. It is the responsibility of
//! the CDataFrameAnalysisSpecification to parse this header although it passes of
//! the reading of the analysis parameters object to implementations of this runner.
//! Therefore CDataFrameAnalysisSpecification is also responsible for creating an
//! appropriate runner object for the data_frame_analyzer command. A particular
//! analysis is launched by the CDataFrameAnalysisSpecification::run method which
//! returns a reference to the appropriate CDataFrameAnalysisRunner implementation.
//!
//! This launches the work to do the analysis in a background thread so that the
//! main thread remains reponsive and can periodically report progress and errors.
//!
//! No mechanism is provided to cancel the work (yet) because it is anticipated
//! that this will be probably be achieved by killing the process and it is too
//! early to determine how to implement a good cooperative interrupt scheme.
class API_EXPORT CDataFrameAnalysisRunner {
public:
    using TStrVec = std::vector<std::string>;

public:
    CDataFrameAnalysisRunner(const CDataFrameAnalysisSpecification& spec);
    virtual ~CDataFrameAnalysisRunner();

    CDataFrameAnalysisRunner(const CDataFrameAnalysisRunner&) = delete;
    CDataFrameAnalysisRunner& operator=(const CDataFrameAnalysisRunner&) = delete;

    //! \return The number of partitions to use when analysing the data frame.
    //! \note If this is greater than one then the data frame should be stored
    //! on disk. The run method is responsible for copying the relevant pieces
    //! into main memory during an analysis.
    virtual std::size_t numberOfPartitions() const = 0;

    //! \return The number of columns this analysis requires. This includes
    //! the columns of the input frame plus any that the analysis will append.
    virtual std::size_t requiredFrameColumns() const = 0;

    //! Checks whether the analysis is already running and if not launches it
    //! in the background.
    //!
    //! \note The thread calling this is expected to be nearly always idle, i.e.
    //! just progress monitoring, so this doesn't count towards the thread limit.
    void run(core::CDataFrame& frame);

    //! This waits to until the analysis has finished and joins the thread.
    void waitToFinish();

    //! \return True if the analysis configuration failed and it can't be run.
    bool bad() const;

    //! \return True if the running analysis has finished.
    bool finished() const;

    //! \return The progress of the analysis in the range [0,1] being an estimate
    //! of the proportion of total work complete for a single run.
    double progress() const;

    //! \return Any errors emitted during the analysis.
    TStrVec errors() const;

protected:
    virtual void runImpl(core::CDataFrame& frame) = 0;

    const CDataFrameAnalysisSpecification& spec() const;

    void setToBad();
    void setToFinished();
    void updateProgress(double fractionalProgress);
    void addError(const std::string& error);

private:
    const CDataFrameAnalysisSpecification& m_Spec;

    bool m_Bad = false;
    std::atomic_bool m_Finished;
    std::atomic<double> m_FractionalProgress;
    TStrVec m_Errors;

    std::thread m_Runner;
    static core::CFastMutex m_Mutex;
};

//! \brief Makes a core::CDataFrame analysis runner.
class API_EXPORT CDataFrameAnalysisRunnerFactory {
public:
    using TRunnerPtr = std::unique_ptr<CDataFrameAnalysisRunner>;

public:
    virtual ~CDataFrameAnalysisRunnerFactory() = default;
    virtual const char* name() const = 0;
    virtual TRunnerPtr make(const CDataFrameAnalysisSpecification& spec) const = 0;
    virtual TRunnerPtr make(const CDataFrameAnalysisSpecification& spec,
                            const rapidjson::Value& params) const = 0;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisRunner_h
