/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameTrainBoostedTreeClassifierRunner_h
#define INCLUDED_ml_api_CDataFrameTrainBoostedTreeClassifierRunner_h

#include <api/CDataFrameTrainBoostedTreeRunner.h>
#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

namespace ml {
namespace api {

//! \brief Runs boosted tree classification on a core::CDataFrame.
class API_EXPORT CDataFrameTrainBoostedTreeClassifierRunner final
    : public CDataFrameTrainBoostedTreeRunner {
public:
    static const CDataFrameAnalysisConfigReader& getParameterReader();

    //! This is not intended to be called directly: use CDataFrameTrainBoostedTreeClassifierRunnerFactory.
    CDataFrameTrainBoostedTreeClassifierRunner(const CDataFrameAnalysisSpecification& spec,
                                               const CDataFrameAnalysisParameters& parameters);

    //! This is not intended to be called directly: use CDataFrameTrainBoostedTreeClassifierRunnerFactory.
    CDataFrameTrainBoostedTreeClassifierRunner(const CDataFrameAnalysisSpecification& spec);

    //! \return Indicator of columns for which empty value should be treated as missing.
    TBoolVec columnsForWhichEmptyIsMissing(const TStrVec& fieldNames) const override;

    //! Write the prediction for \p row to \p writer.
    void writeOneRow(const core::CDataFrame& frame,
                     const TRowRef& row,
                     core::CRapidJsonConcurrentLineWriter& writer) const override;

    //! \return A serialisable definition of the trained classification model.
    TInferenceModelDefinitionUPtr
    inferenceModelDefinition(const TStrVec& fieldNames,
                             const TStrVecVec& categoryNames) const override;

private:
    TLossFunctionUPtr chooseLossFunction(const core::CDataFrame& frame,
                                         std::size_t dependentVariableColumn) const override;

private:
    std::size_t m_NumTopClasses;
};

//! \brief Makes a core::CDataFrame boosted tree classification runner.
class API_EXPORT CDataFrameTrainBoostedTreeClassifierRunnerFactory final
    : public CDataFrameAnalysisRunnerFactory {
public:
    const std::string& name() const override;

private:
    static const std::string NAME;

private:
    TRunnerUPtr makeImpl(const CDataFrameAnalysisSpecification& spec) const override;
    TRunnerUPtr makeImpl(const CDataFrameAnalysisSpecification& spec,
                         const rapidjson::Value& jsonParameters) const override;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameTrainBoostedTreeClassifierRunner_h
