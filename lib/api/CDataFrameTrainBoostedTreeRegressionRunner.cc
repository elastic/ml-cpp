/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>

#include <core/CDataFrame.h>
#include <core/CFloatStorage.h>
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

// This is a temporary code to catch the bug in
// https://github.com/elastic/elasticsearch/issues/59413.
using TFloatVec = std::vector<core::CFloatStorage>;
using TFloatVecVec = std::vector<TFloatVec>;
void logDataFrame(const core::CDataFrame& frame) {
    TFloatVecVec frameVec;
    frameVec.reserve(frame.numberRows());
    frame.readRows(1, [&](core::CDataFrame::TRowItr beginRows, core::CDataFrame::TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            TFloatVec rowVec;
            rowVec.reserve(row->numberColumns());
            for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                rowVec.push_back(row->data()[i]);
            }
            frameVec.emplace_back(std::move(rowVec));
        }
    });
    std::stringstream logStr;
    for (auto& rowVec : frameVec) {
        for (auto& el : rowVec) {
            logStr << el << ", ";
        }
        logStr << "\n";
    }
    LOG_DEBUG(<< "Data frame: \n" << logStr.str());
}

void logModel(CDataFrameAnalysisRunner::TInferenceModelDefinitionUPtr definition) {
    LOG_DEBUG(<< "Model definition: \n" << definition->jsonString());
}
}

const CDataFrameAnalysisConfigReader&
CDataFrameTrainBoostedTreeRegressionRunner::parameterReader() {
    static const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
        auto theReader = CDataFrameTrainBoostedTreeRunner::parameterReader();
        theReader.addParameter(STRATIFIED_CROSS_VALIDATION,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(
            LOSS_FUNCTION, CDataFrameAnalysisConfigReader::E_OptionalParameter,
            {{MSE, int{TLossFunctionType::E_MseRegression}},
             {MSLE, int{TLossFunctionType::E_MsleRegression}},
             {PSEUDO_HUBER, int{TLossFunctionType::E_HuberRegression}}});
        theReader.addParameter(LOSS_FUNCTION_PARAMETER,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        return theReader;
    }()};
    return PARAMETER_READER;
}

CDataFrameTrainBoostedTreeRegressionRunner::TLossFunctionUPtr
CDataFrameTrainBoostedTreeRegressionRunner::lossFunction(const CDataFrameAnalysisParameters& parameters) {
    TLossFunctionType lossFunctionType{
        parameters[LOSS_FUNCTION].fallback(TLossFunctionType::E_MseRegression)};
    switch (lossFunctionType) {
    case TLossFunctionType::E_MsleRegression:
        return std::make_unique<maths::boosted_tree::CMsle>(
            parameters[LOSS_FUNCTION_PARAMETER].fallback(1.0));
    case TLossFunctionType::E_MseRegression:
        return std::make_unique<maths::boosted_tree::CMse>();
    case TLossFunctionType::E_HuberRegression:
        return std::make_unique<maths::boosted_tree::CPseudoHuber>(
            parameters[LOSS_FUNCTION_PARAMETER].fallback(1.0));
    case TLossFunctionType::E_BinaryClassification:
    case TLossFunctionType::E_MulticlassClassification:
        LOG_ERROR(<< "Input error: regression loss type is expected but classification type is provided. Defaulting to MSE instead.");
        return std::make_unique<maths::boosted_tree::CMse>();
    }
    return nullptr;
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
    const core::CDataFrame& frame,
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
        bool noFeatureImportances{true};
        m_InferenceModelMetadata.columnNames(featureImportance->columnNames());
        featureImportance->shap(
            row, [&writer, &noFeatureImportances,
                  this](const maths::CTreeShapFeatureImportance::TSizeVec& indices,
                        const TStrVec& featureNames,
                        const maths::CTreeShapFeatureImportance::TVectorVec& shap) {
                // This is a temporary blog to catch the bug from
                // https://github.com/elastic/elasticsearch/issues/59413
                if (noFeatureImportances) {
                    for (auto i : indices) {
                        if (shap[i].norm() != 0.0) {
                            noFeatureImportances = false;
                            break;
                        }
                    }
                }
                writer.Key(FEATURE_IMPORTANCE_FIELD_NAME);
                writer.StartArray();
                for (auto i : indices) {
                    if (shap[i].norm() != 0.0) {
                        writer.StartObject();
                        writer.Key(FEATURE_NAME_FIELD_NAME);
                        writer.String(featureNames[i]);
                        writer.Key(IMPORTANCE_FIELD_NAME);
                        writer.Double(shap[i](0));
                        writer.EndObject();
                    }
                }
                writer.EndArray();

                for (int i = 0; i < static_cast<int>(shap.size()); ++i) {
                    if (shap[i].lpNorm<1>() != 0) {
                        const_cast<CDataFrameTrainBoostedTreeRegressionRunner*>(this)
                            ->m_InferenceModelMetadata.addToFeatureImportance(i, shap[i]);
                    }
                }
            });
        // This is temporary logging, we are interested in a special case with
        // 500 rows and 3 columns. It will be removed after we caught the bug
        // https://github.com/elastic/elasticsearch/issues/59413.
        if (noFeatureImportances && frame.numberRows() == 500 &&
            frame.numberColumns() == 3 && m_DebugInfoPrinted == false) {
            logDataFrame(frame);
            logModel(this->inferenceModelDefinition(
                frame.columnNames(), frame.categoricalColumnValues()));
            m_DebugInfoPrinted = true;
        }
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
    this->accept(builder);
    return std::make_unique<CInferenceModelDefinition>(builder.build());
}

CDataFrameAnalysisRunner::TOptionalInferenceModelMetadata
CDataFrameTrainBoostedTreeRegressionRunner::inferenceModelMetadata() const {
    const auto& featureImportance = this->boostedTree().shap();
    if (featureImportance) {
        m_InferenceModelMetadata.featureImportanceBaseline(featureImportance->baseline());
    }
    m_InferenceModelMetadata.hyperparameterImportance(
        this->boostedTree().hyperparameterImportance());
    return m_InferenceModelMetadata;
}

// clang-format off
const std::string CDataFrameTrainBoostedTreeRegressionRunner::STRATIFIED_CROSS_VALIDATION{"stratified_cross_validation"};
const std::string CDataFrameTrainBoostedTreeRegressionRunner::LOSS_FUNCTION{"loss_function"};
const std::string CDataFrameTrainBoostedTreeRegressionRunner::LOSS_FUNCTION_PARAMETER{"loss_function_parameter"};
const std::string CDataFrameTrainBoostedTreeRegressionRunner::MSE{"mse"};
const std::string CDataFrameTrainBoostedTreeRegressionRunner::MSLE{"msle"};
const std::string CDataFrameTrainBoostedTreeRegressionRunner::PSEUDO_HUBER{"huber"};
// clang-format on

const std::string& CDataFrameTrainBoostedTreeRegressionRunnerFactory::name() const {
    return NAME;
}

CDataFrameTrainBoostedTreeRegressionRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeRegressionRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification&) const {
    HANDLE_FATAL(<< "Input error: regression has a non-optional parameter '"
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
