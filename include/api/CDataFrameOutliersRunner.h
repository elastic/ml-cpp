/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisRunner.h>

#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

#include <boost/optional.hpp>

namespace ml {
namespace api {

//! \brief Runs outlier detection on a core::CDataFrame.
class API_EXPORT CDataFrameOutliersRunner : public CDataFrameAnalysisRunner {
public:
    CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec,
                             const rapidjson::Value& params);
    CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec);

    //! \return The number of partitions to use when computing outliers for the frame.
    //! \sa CDataFrameAnalysisRunner::run.
    virtual std::size_t numberOfPartitions() const;

    //! \return The number of columns this adds to the data frame.
    virtual std::size_t numberExtraColumns() const;

    //! Write the extra columns of \p row added by outlier analysis to \p writer.
    virtual void writeOneRow(TRowRef row, core::CRapidJsonConcurrentLineWriter& writer) const;

private:
    using TOptionalSize = boost::optional<std::size_t>;

private:
    virtual void runImpl(core::CDataFrame& frame);

private:
    //! \name Custom config
    //@{
    //! If non-null this overrides default number of neighbours to use when computing
    //! outlier factors.
    TOptionalSize m_NumberNeighbours;

    //! If non-null selects the method to use to compute outlier factors; the default
    //! is an ensemble of all supported types.
    //!
    //! \see maths::CLocalOutlierFactors for more details.
    TOptionalSize m_Method;
    //@}

    std::size_t m_NumberPartitions;
};

//! \brief Makes a core::CDataFrame outlier analysis runner.
class API_EXPORT CDataFrameOutliersRunnerFactory : public CDataFrameAnalysisRunnerFactory {
public:
    virtual const char* name() const;
    virtual TRunnerUPtr make(const CDataFrameAnalysisSpecification& spec) const;
    virtual TRunnerUPtr make(const CDataFrameAnalysisSpecification& spec,
                             const rapidjson::Value& params) const;
};
}
}
