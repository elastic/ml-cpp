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
#include <maths/CDataFramePredictiveModel.h>
#include <maths/CDataFrameUtils.h>
#include <maths/COrderings.h>
#include <maths/CTools.h>

#include <api/CBoostedTreeInferenceModelBuilder.h>
#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/ElasticsearchStateIndex.h>

#include <numeric>
#include <set>

namespace ml {
namespace api {
namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TStrSet = std::set<std::string>;

// Output
const std::string IS_TRAINING_FIELD_NAME{"is_training"};
const std::string PREDICTION_PROBABILITY_FIELD_NAME{"prediction_probability"};
const std::string PREDICTION_SCORE_FIELD_NAME{"prediction_score"};
const std::string TOP_CLASSES_FIELD_NAME{"top_classes"};
const std::string CLASS_NAME_FIELD_NAME{"class_name"};
const std::string CLASS_PROBABILITY_FIELD_NAME{"class_probability"};
const std::string CLASS_SCORE_FIELD_NAME{"class_score"};

const TStrSet PREDICTION_FIELD_NAME_BLACKLIST{
    IS_TRAINING_FIELD_NAME, PREDICTION_PROBABILITY_FIELD_NAME,
    PREDICTION_SCORE_FIELD_NAME, TOP_CLASSES_FIELD_NAME};
}

const CDataFrameAnalysisConfigReader&
CDataFrameTrainBoostedTreeClassifierRunner::parameterReader() {
    static const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
        auto theReader = CDataFrameTrainBoostedTreeRunner::parameterReader();
        theReader.addParameter(NUM_TOP_CLASSES, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        const std::string typeString{"string"};
        const std::string typeInt{"int"};
        const std::string typeBool{"bool"};
        theReader.addParameter(PREDICTION_FIELD_TYPE,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter,
                               {{typeString, int{E_PredictionFieldTypeString}},
                                {typeInt, int{E_PredictionFieldTypeInt}},
                                {typeBool, int{E_PredictionFieldTypeBool}}});
        const std::string accuracy{"maximize_accuracy"};
        const std::string minRecall{"maximize_minimum_recall"};
        theReader.addParameter(
            CLASS_ASSIGNMENT_OBJECTIVE, CDataFrameAnalysisConfigReader::E_OptionalParameter,
            {{accuracy, int{maths::CDataFramePredictiveModel::E_Accuracy}},
             {minRecall, int{maths::CDataFramePredictiveModel::E_MinimumRecall}}});
        return theReader;
    }()};
    return PARAMETER_READER;
}

CDataFrameTrainBoostedTreeClassifierRunner::CDataFrameTrainBoostedTreeClassifierRunner(
    const CDataFrameAnalysisSpecification& spec,
    const CDataFrameAnalysisParameters& parameters)
    : CDataFrameTrainBoostedTreeRunner{
          spec, parameters, std::make_unique<maths::boosted_tree::CBinomialLogistic>()} {

    m_NumTopClasses = parameters[NUM_TOP_CLASSES].fallback(std::size_t{0});
    m_PredictionFieldType =
        parameters[PREDICTION_FIELD_TYPE].fallback(E_PredictionFieldTypeString);
    this->boostedTreeFactory().classAssignmentObjective(
        parameters[CLASS_ASSIGNMENT_OBJECTIVE].fallback(maths::CBoostedTree::E_MinimumRecall));

    const TStrVec& categoricalFieldNames{spec.categoricalFieldNames()};
    if (std::find(categoricalFieldNames.begin(), categoricalFieldNames.end(),
                  this->dependentVariableFieldName()) == categoricalFieldNames.end()) {
        HANDLE_FATAL(<< "Input error: trying to perform classification with numeric target.");
    }
    if (PREDICTION_FIELD_NAME_BLACKLIST.count(this->predictionFieldName()) > 0) {
        HANDLE_FATAL(<< "Input error: " << PREDICTION_FIELD_NAME << " must not be equal to any of "
                     << core::CContainerPrinter::print(PREDICTION_FIELD_NAME_BLACKLIST)
                     << ".");
    }
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
    this->writeOneRow(frame, tree.columnHoldingDependentVariable(),
                      tree.columnHoldingPrediction(),
                      tree.probabilityAtWhichToAssignClassOne(), row, writer);
}

void CDataFrameTrainBoostedTreeClassifierRunner::writeOneRow(
    const core::CDataFrame& frame,
    std::size_t columnHoldingDependentVariable,
    std::size_t columnHoldingPrediction,
    double probabilityAtWhichToAssignClassOne,
    const TRowRef& row,
    core::CRapidJsonConcurrentLineWriter& writer) const {

    // TODO generalise when supporting multiple categories.

    // Fetch log odds of class encoded as "1" (the classes are encoded as "0" and "1").
    double predictedLogOddsOfClass1{row[columnHoldingPrediction]};
    double probabilityOfClass1{maths::CTools::logisticFunction(predictedLogOddsOfClass1)};

    // We adjust the probabilities to account for the threshold for choosing class 1.

    TDoubleVec probabilities{1.0 - probabilityOfClass1, probabilityOfClass1};
    TDoubleVec scores{0.5 / (1.0 - probabilityAtWhichToAssignClassOne) * probabilities[0],
                      0.5 / probabilityAtWhichToAssignClassOne * probabilities[1]};

    double actualClassId{row[columnHoldingDependentVariable]};
    std::size_t predictedClassId(std::max_element(scores.begin(), scores.end()) -
                                 scores.begin());

    const TStrVec& classValues{frame.categoricalColumnValues()[columnHoldingDependentVariable]};
    writer.StartObject();
    writer.Key(this->predictionFieldName());
    writePredictedCategoryValue(classValues[predictedClassId], writer);
    writer.Key(PREDICTION_PROBABILITY_FIELD_NAME);
    writer.Double(probabilities[predictedClassId]);
    writer.Key(PREDICTION_SCORE_FIELD_NAME);
    writer.Double(scores[predictedClassId]);
    writer.Key(IS_TRAINING_FIELD_NAME);
    writer.Bool(maths::CDataFrameUtils::isMissing(actualClassId) == false);

    if (m_NumTopClasses > 0) {
        TSizeVec classIds(scores.size());
        std::iota(classIds.begin(), classIds.end(), 0);
        std::sort(classIds.begin(), classIds.end(),
                  [&scores](std::size_t lhs, std::size_t rhs) {
                      return scores[lhs] > scores[rhs];
                  });
        classIds.resize(std::min(classIds.size(), m_NumTopClasses));
        writer.Key(TOP_CLASSES_FIELD_NAME);
        writer.StartArray();
        for (std::size_t i : classIds) {
            writer.StartObject();
            writer.Key(CLASS_NAME_FIELD_NAME);
            writePredictedCategoryValue(classValues[i], writer);
            writer.Key(CLASS_PROBABILITY_FIELD_NAME);
            writer.Double(probabilities[i]);
            writer.Key(CLASS_SCORE_FIELD_NAME);
            writer.Double(scores[i]);
            writer.EndObject();
        }
        writer.EndArray();
    }

    if (this->topShapValues() > 0) {
        auto largestShapValues =
            maths::CBasicStatistics::orderStatisticsAccumulator<std::size_t>(
                this->topShapValues(), [&row](std::size_t lhs, std::size_t rhs) {
                    return std::fabs(row[lhs]) > std::fabs(row[rhs]);
                });
        for (auto col : this->boostedTree().columnsHoldingShapValues()) {
            largestShapValues.add(col);
        }
        largestShapValues.sort();
        for (auto i : largestShapValues) {
            if (row[i] != 0.0) {
                writer.Key(frame.columnNames()[i]);
                writer.Double(row[i]);
            }
        }
    }
    writer.EndObject();
}

void CDataFrameTrainBoostedTreeClassifierRunner::writePredictedCategoryValue(
    const std::string& categoryValue,
    core::CRapidJsonConcurrentLineWriter& writer) const {

    double doubleValue;
    switch (m_PredictionFieldType) {
    case E_PredictionFieldTypeString:
        writer.String(categoryValue);
        break;
    case E_PredictionFieldTypeInt:
        if (core::CStringUtils::stringToType(categoryValue, doubleValue)) {
            writer.Int64(static_cast<std::int64_t>(doubleValue));
        } else {
            writer.String(categoryValue);
        }
        break;
    case E_PredictionFieldTypeBool:
        if (core::CStringUtils::stringToType(categoryValue, doubleValue)) {
            writer.Bool(doubleValue != 0.0);
        } else {
            writer.String(categoryValue);
        }
        break;
    }
}

void CDataFrameTrainBoostedTreeClassifierRunner::validate(const core::CDataFrame& frame,
                                                          std::size_t dependentVariableColumn) const {
    std::size_t categoryCount{
        frame.categoricalColumnValues()[dependentVariableColumn].size()};
    if (categoryCount != 2) {
        HANDLE_FATAL(<< "Input error: only binary classification is supported. "
                     << "Trying to predict '" << frame.columnNames()[dependentVariableColumn]
                     << "' which has '" << categoryCount << "' categories. "
                     << "The number of rows read is '" << frame.numberRows() << "'.");
    }
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

// clang-format off
const std::string CDataFrameTrainBoostedTreeClassifierRunner::NUM_TOP_CLASSES{"num_top_classes"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::PREDICTION_FIELD_TYPE{"prediction_field_type"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::CLASS_ASSIGNMENT_OBJECTIVE{"class_assignment_objective"};
// clang-format off

const std::string& CDataFrameTrainBoostedTreeClassifierRunnerFactory::name() const {
    return NAME;
}

CDataFrameTrainBoostedTreeClassifierRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeClassifierRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification&) const {
    HANDLE_FATAL(<< "Input error: classification has a non-optional parameter '" << CDataFrameTrainBoostedTreeRunner::DEPENDENT_VARIABLE_NAME << "'.")
    return nullptr;
}

CDataFrameTrainBoostedTreeClassifierRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeClassifierRunnerFactory::makeImpl(
    const CDataFrameAnalysisSpecification& spec,
    const rapidjson::Value& jsonParameters) const {
    const CDataFrameAnalysisConfigReader& parameterReader{
        CDataFrameTrainBoostedTreeClassifierRunner::parameterReader()};
    auto parameters = parameterReader.read(jsonParameters);
    return std::make_unique<CDataFrameTrainBoostedTreeClassifierRunner>(spec, parameters);
}

const std::string CDataFrameTrainBoostedTreeClassifierRunnerFactory::NAME{"classification"};
}
}
