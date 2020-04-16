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
#include <maths/CBoostedTreeLoss.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CTreeShapFeatureImportance.h>

#include <api/CBoostedTreeInferenceModelBuilder.h>
#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/ElasticsearchStateIndex.h>

#include <cmath>
#include <memory>
#include <set>
#include <string>

namespace ml {
namespace api {
namespace {
// Output
const std::string IS_TRAINING_FIELD_NAME{"is_training"};

const std::set<std::string> PREDICTION_FIELD_NAME_BLACKLIST{IS_TRAINING_FIELD_NAME};
}

const CDataFrameAnalysisConfigReader&
CDataFrameTrainBoostedTreeRegressionRunner::parameterReader() {
    static const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
        auto theReader = CDataFrameTrainBoostedTreeRunner::parameterReader();
        theReader.addParameter(STRATIFIED_CROSS_VALIDATION,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(LOSS_FUNCTION, CDataFrameAnalysisConfigReader::E_OptionalParameter,
                               {{MSE, int{E_Mse}}, {MSLE, int{E_Msle}}});
        return theReader;
    }()};
    return PARAMETER_READER;
}

CDataFrameTrainBoostedTreeRegressionRunner::TLossFunctionUPtr
CDataFrameTrainBoostedTreeRegressionRunner::lossFunction(const CDataFrameAnalysisParameters& parameters) {
    ELossFunctionType lossFunctionType{parameters[LOSS_FUNCTION].fallback(E_Mse)};
    switch (lossFunctionType) {
    case E_Msle:
        return std::make_unique<maths::boosted_tree::CMsle>();
    case E_Mse:
        return std::make_unique<maths::boosted_tree::CMse>();
    }
}

CDataFrameTrainBoostedTreeRegressionRunner::CDataFrameTrainBoostedTreeRegressionRunner(
    const CDataFrameAnalysisSpecification& spec,
    const CDataFrameAnalysisParameters& parameters)
    : CDataFrameTrainBoostedTreeRunner{
          spec, parameters,
          CDataFrameTrainBoostedTreeRegressionRunner::lossFunction(parameters)} {

    this->boostedTreeFactory().stratifyRegressionCrossValidation(
        parameters[STRATIFIED_CROSS_VALIDATION].fallback(true));

    const TStrVec& categoricalFieldNames{spec.categoricalFieldNames()};
    if (std::find(categoricalFieldNames.begin(), categoricalFieldNames.end(),
                  this->dependentVariableFieldName()) != categoricalFieldNames.end()) {
        HANDLE_FATAL(<< "Input error: trying to perform regression with categorical target.")
    }
    if (PREDICTION_FIELD_NAME_BLACKLIST.count(this->predictionFieldName()) > 0) {
        HANDLE_FATAL(<< "Input error: " << PREDICTION_FIELD_NAME << " must not be equal to any of "
                     << core::CContainerPrinter::print(PREDICTION_FIELD_NAME_BLACKLIST) << ".")
    }
}

void CDataFrameTrainBoostedTreeRegressionRunner::writeOneRow(
    const core::CDataFrame&,
    const TRowRef& row,
    core::CRapidJsonConcurrentLineWriter& writer) const {

    const auto& tree = this->boostedTree();
    const std::size_t columnHoldingDependentVariable{tree.columnHoldingDependentVariable()};

    writer.StartObject();
    writer.Key(this->predictionFieldName());
    writer.Double(tree.readPrediction(row)[0]);
    writer.Key(IS_TRAINING_FIELD_NAME);
    writer.Bool(maths::CDataFrameUtils::isMissing(row[columnHoldingDependentVariable]) == false);
    auto featureImportance = tree.shap();
    if (featureImportance != nullptr) {
        featureImportance->shap(
            row, [&writer](const maths::CTreeShapFeatureImportance::TSizeVec& indices,
                           const TStrVec& names,
                           const maths::CTreeShapFeatureImportance::TVectorVec& shap) {
                writer.Key(CDataFrameTrainBoostedTreeRunner::FEATURE_IMPORTANCE_FIELD_NAME);
                writer.StartArray();
                for (auto i : indices) {
                    if (shap[i].norm() != 0.0) {
                        writer.StartObject();
                        writer.Key(CDataFrameTrainBoostedTreeRunner::FEATURE_NAME_FIELD_NAME);
                        writer.String(names[i]);
                        writer.Key(CDataFrameTrainBoostedTreeRunner::IMPORTANCE_FIELD_NAME);
                        writer.Double(shap[i](0));
                        writer.EndObject();
                    }
                }
                writer.EndArray();
            });
    }
    writer.EndObject();
}

void CDataFrameTrainBoostedTreeRegressionRunner::validate(const core::CDataFrame&,
                                                          std::size_t) const {
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

// clang-format off
const std::string CDataFrameTrainBoostedTreeRegressionRunner::STRATIFIED_CROSS_VALIDATION{"stratified_cross_validation"};
const std::string CDataFrameTrainBoostedTreeRegressionRunner::LOSS_FUNCTION{"loss_function"};
const std::string CDataFrameTrainBoostedTreeRegressionRunner::MSE{"mse"};
const std::string CDataFrameTrainBoostedTreeRegressionRunner::MSLE{"msle"};
// clang-format on

const std::string& CDataFrameTrainBoostedTreeRegressionRunnerFactory::name() const {
    return NAME;
}

CDataFrameTrainBoostedTreeRegressionRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeRegressionRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification&) const {
    HANDLE_FATAL(<< "Input error: classification has a non-optional parameter '"
                 << CDataFrameTrainBoostedTreeRunner::DEPENDENT_VARIABLE_NAME << "'.")
    return nullptr;
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
