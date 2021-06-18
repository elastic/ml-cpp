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
#include <maths/CBoostedTreeLoss.h>
#include <maths/CDataFramePredictiveModel.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/COrderings.h>
#include <maths/CTools.h>
#include <maths/CTreeShapFeatureImportance.h>

#include <api/CBoostedTreeInferenceModelBuilder.h>
#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/ElasticsearchStateIndex.h>

#include <memory>
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
        theReader.addParameter(NUM_CLASSES, CDataFrameAnalysisConfigReader::E_RequiredParameter);
        theReader.addParameter(NUM_TOP_CLASSES, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        const std::string typeString{"string"};
        const std::string typeInt{"int"};
        const std::string typeBool{"bool"};
        theReader.addParameter(PREDICTION_FIELD_TYPE,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter,
                               {{typeString, int{E_PredictionFieldTypeString}},
                                {typeInt, int{E_PredictionFieldTypeInt}},
                                {typeBool, int{E_PredictionFieldTypeBool}}});
        int accuracy{maths::CDataFramePredictiveModel::E_Accuracy};
        int recall{maths::CDataFramePredictiveModel::E_MinimumRecall};
        int custom{maths::CDataFramePredictiveModel::E_Custom};
        theReader.addParameter(
            CLASS_ASSIGNMENT_OBJECTIVE, CDataFrameAnalysisConfigReader::E_OptionalParameter,
            {{CLASS_ASSIGNMENT_OBJECTIVE_VALUES[accuracy], accuracy},
             {CLASS_ASSIGNMENT_OBJECTIVE_VALUES[recall], recall},
             {CLASS_ASSIGNMENT_OBJECTIVE_VALUES[custom], custom}});
        theReader.addParameter(CLASSIFICATION_WEIGHTS,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        return theReader;
    }()};
    return PARAMETER_READER;
}

CDataFrameTrainBoostedTreeClassifierRunner::CDataFrameTrainBoostedTreeClassifierRunner(
    const CDataFrameAnalysisSpecification& spec,
    const CDataFrameAnalysisParameters& parameters,
    TDataFrameUPtrTemporaryDirectoryPtrPr* frameAndDirectory)
    : CDataFrameTrainBoostedTreeRunner{
          spec, parameters, loss(parameters[NUM_CLASSES].as<std::size_t>()), frameAndDirectory} {

    std::size_t numberClasses{parameters[NUM_CLASSES].as<std::size_t>()};
    auto classAssignmentObjective = parameters[CLASS_ASSIGNMENT_OBJECTIVE].fallback(
        maths::CBoostedTree::E_MinimumRecall);
    m_NumTopClasses = parameters[NUM_TOP_CLASSES].fallback(std::ptrdiff_t{0});
    m_PredictionFieldType =
        parameters[PREDICTION_FIELD_TYPE].fallback(E_PredictionFieldTypeString);
    this->boostedTreeFactory().classAssignmentObjective(classAssignmentObjective);
    auto classificationWeights = parameters[CLASSIFICATION_WEIGHTS].fallback(
        CLASSIFICATION_WEIGHTS_CLASS, CLASSIFICATION_WEIGHTS_WEIGHT,
        std::vector<std::pair<std::string, double>>{});
    if (classificationWeights.size() > 0) {
        this->boostedTreeFactory().classificationWeights(classificationWeights);
    }

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
    if (classificationWeights.size() > 0 &&
        classAssignmentObjective != maths::CBoostedTree::E_Custom) {
        HANDLE_FATAL(<< "Input error: expected "
                     << CLASS_ASSIGNMENT_OBJECTIVE_VALUES[maths::CDataFramePredictiveModel::E_Custom]
                     << " for " << CLASS_ASSIGNMENT_OBJECTIVE << " if supplying "
                     << CLASSIFICATION_WEIGHTS << " but got '"
                     << CLASS_ASSIGNMENT_OBJECTIVE_VALUES[classAssignmentObjective] << "'.");
    }
    if (classificationWeights.size() > 0 && classificationWeights.size() != numberClasses) {
        HANDLE_FATAL(<< "Input error: expected " << numberClasses << " " << CLASSIFICATION_WEIGHTS
                     << " but got " << classificationWeights.size() << ".");
    }
}

void CDataFrameTrainBoostedTreeClassifierRunner::writeOneRow(
    const core::CDataFrame& frame,
    const TRowRef& row,
    core::CRapidJsonConcurrentLineWriter& writer) const {

    const auto& tree = this->boostedTree();
    this->writeOneRow(
        frame, tree.columnHoldingDependentVariable(),
        [&](const TRowRef& row_) { return tree.readPrediction(row_); },
        [&](const TRowRef& row_) { return tree.readAndAdjustPrediction(row_); },
        row, writer, tree.shap());
}

void CDataFrameTrainBoostedTreeClassifierRunner::writeOneRow(
    const core::CDataFrame& frame,
    std::size_t columnHoldingDependentVariable,
    const TReadPredictionFunc& readClassProbabilities,
    const TReadClassScoresFunc& readClassScores,
    const TRowRef& row,
    core::CRapidJsonConcurrentLineWriter& writer,
    maths::CTreeShapFeatureImportance* featureImportance) const {

    auto probabilities = readClassProbabilities(row);
    auto scores = readClassScores(row);

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

    if (m_NumTopClasses != 0) {
        TSizeVec classIds(scores.size());
        std::iota(classIds.begin(), classIds.end(), 0);
        std::sort(classIds.begin(), classIds.end(),
                  [&scores](std::size_t lhs, std::size_t rhs) {
                      return scores[lhs] > scores[rhs];
                  });
        // -1 is a special value meaning "output all the classes"
        classIds.resize(m_NumTopClasses == -1
                            ? classIds.size()
                            : std::min(classIds.size(),
                                       static_cast<std::size_t>(m_NumTopClasses)));
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

    if (featureImportance != nullptr) {
        int numberClasses{static_cast<int>(classValues.size())};
        m_InferenceModelMetadata.columnNames(featureImportance->columnNames());
        m_InferenceModelMetadata.classValues(classValues);
        m_InferenceModelMetadata.predictionFieldTypeResolverWriter(
            [this](const std::string& categoryValue,
                   core::CRapidJsonConcurrentLineWriter& writer_) {
                this->writePredictedCategoryValue(categoryValue, writer_);
            });
        featureImportance->shap(
            row, [&](const maths::CTreeShapFeatureImportance::TSizeVec& indices,
                     const TStrVec& featureNames,
                     const maths::CTreeShapFeatureImportance::TVectorVec& shap) {
                writer.Key(FEATURE_IMPORTANCE_FIELD_NAME);
                writer.StartArray();
                for (auto i : indices) {
                    if (shap[i].norm() != 0.0) {
                        writer.StartObject();
                        writer.Key(FEATURE_NAME_FIELD_NAME);
                        writer.String(featureNames[i]);
                        if (shap[i].size() == 1) {
                            // output feature importance for individual classes in binary case
                            writer.Key(CLASSES_FIELD_NAME);
                            writer.StartArray();
                            for (int j = 0; j < numberClasses; ++j) {
                                writer.StartObject();
                                writer.Key(CLASS_NAME_FIELD_NAME);
                                writePredictedCategoryValue(classValues[j], writer);
                                writer.Key(IMPORTANCE_FIELD_NAME);
                                if (j == 1) {
                                    writer.Double(shap[i](0));
                                } else {
                                    writer.Double(-shap[i](0));
                                }
                                writer.EndObject();
                            }
                            writer.EndArray();
                        } else {
                            // output feature importance for individual classes in multiclass case
                            writer.Key(CLASSES_FIELD_NAME);
                            writer.StartArray();
                            for (int j = 0; j < shap[i].size() && j < numberClasses; ++j) {
                                writer.StartObject();
                                writer.Key(CLASS_NAME_FIELD_NAME);
                                writePredictedCategoryValue(classValues[j], writer);
                                writer.Key(IMPORTANCE_FIELD_NAME);
                                writer.Double(shap[i](j));
                                writer.EndObject();
                            }
                            writer.EndArray();
                        }
                        writer.EndObject();
                    }
                }
                writer.EndArray();

                for (std::size_t i = 0; i < shap.size(); ++i) {
                    if (shap[i].lpNorm<1>() != 0) {
                        const_cast<CDataFrameTrainBoostedTreeClassifierRunner*>(this)
                            ->m_InferenceModelMetadata.addToFeatureImportance(i, shap[i]);
                    }
                }
            });
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

CDataFrameTrainBoostedTreeClassifierRunner::TLossFunctionUPtr
CDataFrameTrainBoostedTreeClassifierRunner::loss(std::size_t numberClasses) {
    using namespace maths::boosted_tree;
    return numberClasses == 2
               ? TLossFunctionUPtr{std::make_unique<CBinomialLogisticLoss>()}
               : TLossFunctionUPtr{std::make_unique<CMultinomialLogisticLoss>(numberClasses)};
}

void CDataFrameTrainBoostedTreeClassifierRunner::validate(const core::CDataFrame& frame,
                                                          std::size_t dependentVariableColumn) const {
    std::size_t categoryCount{
        frame.categoricalColumnValues()[dependentVariableColumn].size()};
    if (categoryCount < 2) {
        HANDLE_FATAL(<< "Input error: can't run classification unless there are at least "
                     << "two classes. Trying to predict '"
                     << frame.columnNames()[dependentVariableColumn] << "' which has '"
                     << categoryCount << "' categories in the training data. "
                     << "The number of rows read is '" << frame.numberRows() << "'.");
    } else if (categoryCount > MAX_NUMBER_CLASSES) {
        HANDLE_FATAL(<< "Input error: the maximum number of classes supported is "
                     << MAX_NUMBER_CLASSES << ". Trying to predict '"
                     << frame.columnNames()[dependentVariableColumn] << "' which has '"
                     << categoryCount << "' categories in the training data. "
                     << "The number of rows read is '" << frame.numberRows() << "'.");
    }
}

CDataFrameAnalysisRunner::TInferenceModelDefinitionUPtr
CDataFrameTrainBoostedTreeClassifierRunner::inferenceModelDefinition(
    const CDataFrameAnalysisRunner::TStrVec& fieldNames,
    const CDataFrameAnalysisRunner::TStrVecVec& categoryNames) const {
    CClassificationInferenceModelBuilder builder(
        fieldNames, this->boostedTree().columnHoldingDependentVariable(), categoryNames);
    this->accept(builder);
    return std::make_unique<CInferenceModelDefinition>(builder.build());
}

CDataFrameAnalysisRunner::TOptionalInferenceModelMetadata
CDataFrameTrainBoostedTreeClassifierRunner::inferenceModelMetadata() const {
    const auto& featureImportance = this->boostedTree().shap();
    if (featureImportance != nullptr) {
        m_InferenceModelMetadata.featureImportanceBaseline(featureImportance->baseline());
    }
    if (this->task() != E_Predict) {
        m_InferenceModelMetadata.hyperparameterImportance(
            this->boostedTree().hyperparameterImportance());
    }
    std::size_t dataSummarizationNumRows{static_cast<std::size_t>(this->boostedTree().dataSummarization().manhattan())};
    if (dataSummarizationNumRows > 0) {
        m_InferenceModelMetadata.dataSummarizationNumRows(dataSummarizationNumRows);
    }
    return m_InferenceModelMetadata;
}

// clang-format off
// The MAX_NUMBER_CLASSES must match the value used in the Java code. See the
// MAX_DEPENDENT_VARIABLE_CARDINALITY in the x-pack classification code.
const std::size_t CDataFrameTrainBoostedTreeClassifierRunner::MAX_NUMBER_CLASSES{30};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::NUM_CLASSES{"num_classes"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::NUM_TOP_CLASSES{"num_top_classes"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::PREDICTION_FIELD_TYPE{"prediction_field_type"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::CLASS_ASSIGNMENT_OBJECTIVE{"class_assignment_objective"};
const CDataFrameTrainBoostedTreeClassifierRunner::TStrVec
CDataFrameTrainBoostedTreeClassifierRunner::CLASS_ASSIGNMENT_OBJECTIVE_VALUES{
    "maximize_accuracy", "maximize_minimum_recall", "custom"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::CLASSIFICATION_WEIGHTS{"classification_weights"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::CLASSIFICATION_WEIGHTS_CLASS{"class"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::CLASSIFICATION_WEIGHTS_WEIGHT{"weight"};
// clang-format on

const std::string& CDataFrameTrainBoostedTreeClassifierRunnerFactory::name() const {
    return NAME;
}

CDataFrameTrainBoostedTreeClassifierRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeClassifierRunnerFactory::makeImpl(
    const CDataFrameAnalysisSpecification&,
    TDataFrameUPtrTemporaryDirectoryPtrPr*) const {
    HANDLE_FATAL(<< "Input error: classification has a non-optional parameter '"
                 << CDataFrameTrainBoostedTreeRunner::DEPENDENT_VARIABLE_NAME << "'.");
    return nullptr;
}

CDataFrameTrainBoostedTreeClassifierRunnerFactory::TRunnerUPtr
CDataFrameTrainBoostedTreeClassifierRunnerFactory::makeImpl(
    const CDataFrameAnalysisSpecification& spec,
    const rapidjson::Value& jsonParameters,
    TDataFrameUPtrTemporaryDirectoryPtrPr* frameAndDirectory) const {
    const CDataFrameAnalysisConfigReader& parameterReader{
        CDataFrameTrainBoostedTreeClassifierRunner::parameterReader()};
    auto parameters = parameterReader.read(jsonParameters);
    return std::make_unique<CDataFrameTrainBoostedTreeClassifierRunner>(
        spec, parameters, frameAndDirectory);
}

const std::string CDataFrameTrainBoostedTreeClassifierRunnerFactory::NAME{"classification"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::CLASSES_FIELD_NAME{"classes"};
const std::string CDataFrameTrainBoostedTreeClassifierRunner::CLASS_NAME_FIELD_NAME{"class_name"};
}
}
