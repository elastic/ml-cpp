/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameRegressionRunner_h
#define INCLUDED_ml_api_CDataFrameRegressionRunner_h

#include <core/CDataSearcher.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameBoostedTreeRunner.h>
#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

#include <atomic>

namespace ml {
namespace api {

//! \brief Runs boosted tree regression on a core::CDataFrame.
class API_EXPORT CDataFrameRegressionRunner final : public CDataFrameBoostedTreeRunner {
public:
    static const CDataFrameAnalysisConfigReader getParameterReader();

    //! This is not intended to be called directly: use CDataFrameRegressionRunnerFactory.
    CDataFrameRegressionRunner(const CDataFrameAnalysisSpecification& spec,
                               const CDataFrameAnalysisConfigReader::CParameters& parameters);

    //! This is not intended to be called directly: use CDataFrameRegressionRunnerFactory.
    CDataFrameRegressionRunner(const CDataFrameAnalysisSpecification& spec);

    //! Write the prediction for \p row to \p writer.
    void writeOneRow(const core::CDataFrame& frame,
                     const TRowRef& row,
                     core::CRapidJsonConcurrentLineWriter& writer) const override;

private:
    TLossFunctionUPtr chooseLossFunction(const core::CDataFrame& frame,
                                         std::size_t dependentVariableColumn) const override;
};

//! \brief Makes a core::CDataFrame boosted tree regression runner.
class API_EXPORT CDataFrameRegressionRunnerFactory final : public CDataFrameAnalysisRunnerFactory {
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

#endif // INCLUDED_ml_api_CDataFrameRegressionRunner_h
