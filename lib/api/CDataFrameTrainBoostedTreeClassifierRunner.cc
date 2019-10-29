/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameTrainBoostedTreeClassifierRunner.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CDataFrameUtils.h>
#include <maths/COrderings.h>
#include <maths/CTools.h>

#include <api/CBoostedTreeInferenceModelBuilder.h>
#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/ElasticsearchStateIndex.h>

#include <numeric>

namespace ml {
namespace api {
namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

// Configuration
const std::string NUM_TOP_CLASSES{"num_top_classes"};
const std::string BALANCED_CLASS_LOSS{"balanced_class_loss"};

// Output
const std::string IS_TRAINING_FIELD_NAME{"is_training"};
const std::string TOP_CLASSES_FIELD_NAME{"top_classes"};
const std::string CLASS_NAME_FIELD_NAME{"class_name"};
const std::string CLASS_PROBABILITY_FIELD_NAME{"class_probability"};
}

const CDataFrameAnalysisConfigReader&
CDataFrameTrainBoostedTreeClassifierRunner::getParameterReader() {
    static const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
        auto theReader = CDataFrameTrainBoostedTreeRunner::getParameterReader();
        theReader.addParameter(NUM_TOP_CLASSES, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(BALANCED_CLASS_LOSS,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        return theReader;
    }()};
    return PARAMETER_READER;
}

CDataFrameTrainBoostedTreeClassifierRunner::CDataFrameTrainBoostedTreeClassifierRunner(
    const CDataFrameAnalysisSpecification& spec,
    const CDataFrameAnalysisParameters& parameters)
    : CDataFrameTrainBoostedTreeRunner{spec, parameters} {

    m_NumTopClasses = parameters[NUM_TOP_CLASSES].fallback(std::size_t{0});
    this->boostedTreeFactory().balanceClassTrainingLoss(
        parameters[BALANCED_CLASS_LOSS].fallback(true));

    const TStrVec& categoricalFieldNames{spec.categoricalFieldNames()};
    if (std::find(categoricalFieldNames.begin(), categoricalFieldNames.end(),
                  this->dependentVariableFieldName()) == categoricalFieldNames.end()) {
        HANDLE_FATAL(<< "Input error: trying to perform classification with numeric target.");
    }
}

CDataFrameTrainBoostedTreeClassifierRunner::CDataFrameTrainBoostedTreeClassifierRunner(
    const CDataFrameAnalysisSpecification& spec)
    : CDataFrameTrainBoostedTreeRunner{spec} {
}

TBoolVec CDataFrameTrainBoostedTreeClassifierRunner::columnsForWhichEmptyIsMissing(
    const TStrVec& fieldNames) const {
    // The only field for which empty value should be treated as missing is dependent
    // variable which has empty value for non-training rows.
    TBoolVec emptyAsMissing(fieldNames.size(), false);
    auto pos = std::find(fieldNames.begin(), fieldNames.end(),
                         this->dependentVariableFieldName());
    if (pos != fieldNames.end()) {
        emptyAsMissing[pos - fieldNames.begin()] = true;
    }
    return emptyAsMissing;
}

void CDataFrameTrainBoostedTreeClassifierRunner::writeOneRow(
    const core::CDataFrame& frame,
    const TRowRef& row,
    core::CRapidJsonConcurrentLineWriter& writer) const {
    const auto& tree = this->boostedTree();
    const std::size_t columnHoldingDependentVariable{tree.columnHoldingDependentVariable()};
    const std::size_t columnHoldingPrediction{
        tree.columnHoldingPrediction(row.numberColumns())};
    const TStrVec& categoryValues{frame.categoricalColumnValues()[columnHoldingDependentVariable]};

    // TODO generalise when supporting multiple categories.

    double predictedLogOddsOfCategory1{row[columnHoldingPrediction]};
    double probabilityOfCategory1{maths::CTools::logisticFunction(predictedLogOddsOfCategory1)};
    TDoubleVec probabilityOfCategory{1.0 - probabilityOfCategory1, probabilityOfCategory1};

    double actualCategoryId{row[columnHoldingDependentVariable]};
    std::size_t predictedCategoryId(std::max_element(probabilityOfCategory.begin(),
                                                     probabilityOfCategory.end()) -
                                    probabilityOfCategory.begin());

    writer.StartObject();
    writer.Key(this->predictionFieldName());
    writer.String(categoryValues[predictedCategoryId]);
    writer.Key(IS_TRAINING_FIELD_NAME);
    writer.Bool(maths::CDataFrameUtils::isMissing(actualCategoryId) == false);

    if (m_NumTopClasses > 0) {
        TSizeVec categoryIds(probabilityOfCategory.size());
        std::iota(categoryIds.begin(), categoryIds.end(), 0);
        maths::COrderings::simultaneousSort(probabilityOfCategory, categoryIds,
                                            std::greater<double>());
        writer.Key(TOP_CLASSES_FIELD_NAME);
        writer.StartArray();
        for (std::size_t i = 0; i < std::min(categoryIds.size(), m_NumTopClasses); ++i) {
            writer.StartObject();
            writer.Key(CLASS_NAME_FIELD_NAME);
            writer.String(categoryValues[categoryIds[i]]);
            writer.Key(CLASS_PROBABILITY_FIELD_NAME);
            writer.Double(probabilityOfCategory[i]);
            writer.EndObject();
        }
        writer.EndArray();
    }
    writer.EndObject();
}

CDataFrameTrainBoostedTreeClassifierRunner::TLossFunctionUPtr
CDataFrameTrainBoostedTreeClassifierRunner::chooseLossFunction(const core::CDataFrame& frame,
                                                               std::size_t dependentVariableColumn) const {
    std::size_t categoryCount{
        frame.categoricalColumnValues()[dependentVariableColumn].size()};
    if (categoryCount == 2) {
        return std::make_unique<maths::boosted_tree::CLogistic>();
    }
    HANDLE_FATAL(<< "Input error: only binary classification is supported. "
                 << "Trying to predict '" << frame.columnNames()[dependentVariableColumn]
                 << "' which has '" << categoryCount << "' categories. "
                 << "The number of rows read is '" << frame.numberRows() << "'.");
    return nullptr;
}

CDataFrameAnalysisRunner::TInferenceModelDefinitionUPtr
CDataFrameTrainBoostedTreeClassifierRunner::inferenceModelDefinition(
    const CDataFrameAnalysisRunner::TStrVec& fieldNames,
    const CDataFrameAnalysisRunner::TStrVecVec& categoryNames) const {
    CClassificationInferenceModelBuilder builder(
        fieldNames, this->boostedTree().columnHoldingDependentVariable(), categoryNames);
    this->boostedTree().accept(builder);

    return std::make_unique<CInferenceModelDefinition>(builder.build());
}

const std::string& CDataFrameTrainBoostedTreeClassifierRunnerFactory::name() const {
    return NAME;
}

CDataFrameTrainBoostedTreeClassifierRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeClassifierRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec) const {
    return std::make_unique<CDataFrameTrainBoostedTreeClassifierRunner>(spec);
}

CDataFrameTrainBoostedTreeClassifierRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeClassifierRunnerFactory::makeImpl(
    const CDataFrameAnalysisSpecification& spec,
    const rapidjson::Value& jsonParameters) const {
    const CDataFrameAnalysisConfigReader& parameterReader{
        CDataFrameTrainBoostedTreeClassifierRunner::getParameterReader()};
    auto parameters = parameterReader.read(jsonParameters);
    return std::make_unique<CDataFrameTrainBoostedTreeClassifierRunner>(spec, parameters);
}

const std::string CDataFrameTrainBoostedTreeClassifierRunnerFactory::NAME{"classification"};
}
}
