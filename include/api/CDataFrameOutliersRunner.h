/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameOutliersRunner_h
#define INCLUDED_ml_api_CDataFrameOutliersRunner_h

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisInstrumentation.h>
#include <api/CDataFrameAnalysisRunner.h>
#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

namespace ml {
namespace api {

//! \brief Runs outlier detection on a core::CDataFrame.
class API_EXPORT CDataFrameOutliersRunner final : public CDataFrameAnalysisRunner {
public:
    static const std::string STANDARDIZATION_ENABLED;
    static const std::string N_NEIGHBORS;
    static const std::string METHOD;
    static const std::string COMPUTE_FEATURE_INFLUENCE;
    static const std::string FEATURE_INFLUENCE_THRESHOLD;
    static const std::string OUTLIER_FRACTION;

public:
    //! This is not intended to be called directly: use CDataFrameOutliersRunnerFactory.
    CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec,
                             const CDataFrameAnalysisParameters& parameters);

    //! \return Reference to the analysis state.
    const CDataFrameAnalysisInstrumentation& instrumentation() const override;
    //! \return Reference to the analysis state.
    CDataFrameAnalysisInstrumentation& instrumentation() override;

    //! This is not intended to be called directly: use CDataFrameOutliersRunnerFactory.
    CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec);

    //! \return The number of columns this adds to the data frame.
    std::size_t numberExtraColumns() const override;

    //! Write the extra columns of \p row added by outlier analysis to \p writer.
    void writeOneRow(const core::CDataFrame& frame,
                     const TRowRef& row,
                     core::CRapidJsonConcurrentLineWriter& writer) const override;

private:
    void runImpl(core::CDataFrame& frame) override;
    std::size_t estimateBookkeepingMemoryUsage(std::size_t numberPartitions,
                                               std::size_t totalNumberRows,
                                               std::size_t partitionNumberRows,
                                               std::size_t numberColumns) const override;

private:
    //! \name Custom config
    //@{
    //! If non-zero this overrides default number of neighbours to use when computing
    //! outlier factors.
    std::size_t m_NumberNeighbours = 0;

    //! Selects the method to use to compute outlier factors; the default is an ensemble
    //! of all supported types.
    //!
    //! \see maths::COutliers for more details.
    std::size_t m_Method;

    //! If true then standardise the feature values.
    bool m_StandardizationEnabled = true;

    //! Compute the significance of features responsible for each point being outlying.
    bool m_ComputeFeatureInfluence = true;

    //! The minimum outlier score for which we'll write out feature influence.
    double m_FeatureInfluenceThreshold = 0.1;

    //! The fraction of true outliers amoung the points.
    double m_OutlierFraction = 0.05;
    //@}

    CDataFrameOutliersInstrumentation m_Instrumentation;
};

//! \brief Makes a core::CDataFrame outlier analysis runner.
class API_EXPORT CDataFrameOutliersRunnerFactory final : public CDataFrameAnalysisRunnerFactory {
public:
    static const std::string NAME;

public:
    const std::string& name() const override;

private:
    TRunnerUPtr makeImpl(const CDataFrameAnalysisSpecification& spec) const override;
    TRunnerUPtr makeImpl(const CDataFrameAnalysisSpecification& spec,
                         const rapidjson::Value& jsonParameters) const override;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameOutliersRunner_h
