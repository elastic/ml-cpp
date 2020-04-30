/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameTrainBoostedTreeRegressionRunner_h
#define INCLUDED_ml_api_CDataFrameTrainBoostedTreeRegressionRunner_h

#include "maths/CBoostedTreeLoss.h"
#include <api/CDataFrameTrainBoostedTreeRunner.h>
#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

namespace ml {
namespace api {

//! \brief Runs boosted tree regression on a core::CDataFrame.
class API_EXPORT CDataFrameTrainBoostedTreeRegressionRunner final
    : public CDataFrameTrainBoostedTreeRunner {

public:
    using TLossFunctionUPtr = std::unique_ptr<maths::boosted_tree::CLoss>;
    using TLossFunctionType = maths::boosted_tree::ELossType;

public:
    static const std::string STRATIFIED_CROSS_VALIDATION;
    static const std::string LOSS_FUNCTION;
    static const std::string LOSS_FUNCTION_PARAMETER;
    static const std::string MSE;
    static const std::string MSLE;
    static const std::string PSEUDO_HUBER;

public:
    static const CDataFrameAnalysisConfigReader& parameterReader();

    static TLossFunctionUPtr lossFunction(const CDataFrameAnalysisParameters& parameters);

    //! This is not intended to be called directly: use CDataFrameTrainBoostedTreeRegressionRunnerFactory.
    CDataFrameTrainBoostedTreeRegressionRunner(const CDataFrameAnalysisSpecification& spec,
                                               const CDataFrameAnalysisParameters& parameters);

    //! Write the prediction for \p row to \p writer.
    void writeOneRow(const core::CDataFrame& frame,
                     const TRowRef& row,
                     core::CRapidJsonConcurrentLineWriter& writer) const override;

    //! \return A serialisable definition of the trained regression model.
    TInferenceModelDefinitionUPtr
    inferenceModelDefinition(const TStrVec& fieldNames,
                             const TStrVecVec& categoryNameMap) const override;

private:
    void validate(const core::CDataFrame& frame,
                  std::size_t dependentVariableColumn) const override;
};

//! \brief Makes a core::CDataFrame boosted tree regression runner.
class API_EXPORT CDataFrameTrainBoostedTreeRegressionRunnerFactory final
    : public CDataFrameAnalysisRunnerFactory {
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

#endif // INCLUDED_ml_api_CDataFrameTrainBoostedTreeRegressionRunner_h
