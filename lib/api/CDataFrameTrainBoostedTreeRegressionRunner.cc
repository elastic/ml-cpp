/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>

#include <core/CLogger.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CDataFrameUtils.h>

#include <api/CBoostedTreeInferenceModelBuilder.h>
#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/ElasticsearchStateIndex.h>

#include <numeric>

namespace ml {
namespace api {
namespace {
// Configuration
const std::string STRATIFIED_CROSS_VALIDATION{"stratified_cross_validation"};

// Output
const std::string IS_TRAINING_FIELD_NAME{"is_training"};
}

const CDataFrameAnalysisConfigReader&
CDataFrameTrainBoostedTreeRegressionRunner::parameterReader() {
    static const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
        auto theReader = CDataFrameTrainBoostedTreeRunner::parameterReader();
        theReader.addParameter(STRATIFIED_CROSS_VALIDATION,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        return theReader;
    }()};
    return PARAMETER_READER;
}

CDataFrameTrainBoostedTreeRegressionRunner::CDataFrameTrainBoostedTreeRegressionRunner(
    const CDataFrameAnalysisSpecification& spec,
    const CDataFrameAnalysisParameters& parameters)
    : CDataFrameTrainBoostedTreeRunner{spec, parameters} {

    this->boostedTreeFactory().stratifyRegressionCrossValidation(
        parameters[STRATIFIED_CROSS_VALIDATION].fallback(true));

    const TStrVec& categoricalFieldNames{spec.categoricalFieldNames()};
    if (std::find(categoricalFieldNames.begin(), categoricalFieldNames.end(),
                  this->dependentVariableFieldName()) != categoricalFieldNames.end()) {
        HANDLE_FATAL(<< "Input error: trying to perform regression with categorical target.");
    }
}

CDataFrameTrainBoostedTreeRegressionRunner::CDataFrameTrainBoostedTreeRegressionRunner(
    const CDataFrameAnalysisSpecification& spec)
    : CDataFrameTrainBoostedTreeRunner{spec} {
}

void CDataFrameTrainBoostedTreeRegressionRunner::writeOneRow(
    const core::CDataFrame& frame,
    const TRowRef& row,
    core::CRapidJsonConcurrentLineWriter& writer) const {
    const auto& tree = this->boostedTree();
    const std::size_t columnHoldingDependentVariable = tree.columnHoldingDependentVariable();
    const std::size_t columnHoldingPrediction =
        tree.columnHoldingPrediction(row.numberColumns());

    writer.StartObject();
    writer.Key(this->predictionFieldName());
    writer.Double(row[columnHoldingPrediction]);
    writer.Key(IS_TRAINING_FIELD_NAME);
    writer.Bool(maths::CDataFrameUtils::isMissing(row[columnHoldingDependentVariable]) == false);
    if (this->topShapValues() > 0) {

        for (int i = 0; i < row.numberColumns(); ++i) {
            const std::string& columnName{frame.columnNames()[i]};
            auto res = std::mismatch(
                CDataFrameTrainBoostedTreeRunner::SHAP_PREFIX.begin(),
                CDataFrameTrainBoostedTreeRunner::SHAP_PREFIX.end(), columnName.begin());
            if (res.first == CDataFrameTrainBoostedTreeRunner::SHAP_PREFIX.end()) {
                writer.Key(columnName);
                writer.Double(row[i]);
            }
        }
    }
    writer.EndObject();
}

CDataFrameTrainBoostedTreeRegressionRunner::TLossFunctionUPtr
CDataFrameTrainBoostedTreeRegressionRunner::chooseLossFunction(const core::CDataFrame&,
                                                               std::size_t) const {
    return std::make_unique<maths::boosted_tree::CMse>();
}

CDataFrameAnalysisRunner::TInferenceModelDefinitionUPtr
CDataFrameTrainBoostedTreeRegressionRunner::inferenceModelDefinition(
    const CDataFrameAnalysisRunner::TStrVec& fieldNames,
    const CDataFrameAnalysisRunner::TStrVecVec& categoryNames) const {
    CRegressionInferenceModelBuilder builder(
        fieldNames, this->boostedTree().columnHoldingDependentVariable(), categoryNames);
    this->boostedTree().accept(builder);

    return std::make_unique<CInferenceModelDefinition>(builder.build());
}

const std::string& CDataFrameTrainBoostedTreeRegressionRunnerFactory::name() const {
    return NAME;
}

CDataFrameTrainBoostedTreeRegressionRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeRegressionRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec) const {
    return std::make_unique<CDataFrameTrainBoostedTreeRegressionRunner>(spec);
}

CDataFrameTrainBoostedTreeRegressionRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeRegressionRunnerFactory::makeImpl(
    const CDataFrameAnalysisSpecification& spec,
    const rapidjson::Value& jsonParameters) const {
    const CDataFrameAnalysisConfigReader& parameterReader{
        CDataFrameTrainBoostedTreeRegressionRunner::parameterReader()};
    auto parameters = parameterReader.read(jsonParameters);
    return std::make_unique<CDataFrameTrainBoostedTreeRegressionRunner>(spec, parameters);
}

const std::string CDataFrameTrainBoostedTreeRegressionRunnerFactory::NAME{"regression"};
}
}
