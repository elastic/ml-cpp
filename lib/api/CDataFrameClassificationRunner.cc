/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameClassificationRunner.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CProgramCounters.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/CStateDecompressor.h>
#include <core/CStopWatch.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CDataFrameUtils.h>
#include <maths/COrderings.h>
#include <maths/CTools.h>

#include <api/CBoostedTreeInferenceModelBuilder.h>
#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameBoostedTreeRunner.h>
#include <api/ElasticsearchStateIndex.h>

#include <rapidjson/document.h>

#include <numeric>

namespace ml {
namespace api {
namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

// Configuration
const std::string NUM_TOP_CLASSES{"num_top_classes"};

// Output
const std::string IS_TRAINING_FIELD_NAME{"is_training"};
const std::string TOP_CLASSES_FIELD_NAME{"top_classes"};
const std::string CLASS_NAME_FIELD_NAME{"class_name"};
const std::string CLASS_PROBABILITY_FIELD_NAME{"class_probability"};
}

const CDataFrameAnalysisConfigReader CDataFrameClassificationRunner::getParameterReader() {
    CDataFrameAnalysisConfigReader theReader{CDataFrameBoostedTreeRunner::getParameterReader()};
    theReader.addParameter(NUM_TOP_CLASSES, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    return theReader;
}

CDataFrameClassificationRunner::CDataFrameClassificationRunner(
    const CDataFrameAnalysisSpecification& spec,
    const CDataFrameAnalysisConfigReader::CParameters& parameters)
    : CDataFrameBoostedTreeRunner{spec, parameters} {

    m_NumTopClasses = parameters[NUM_TOP_CLASSES].fallback(std::size_t{0});
}

CDataFrameClassificationRunner::CDataFrameClassificationRunner(const CDataFrameAnalysisSpecification& spec)
    : CDataFrameBoostedTreeRunner{spec} {
}

TBoolVec CDataFrameClassificationRunner::columnsForWhichEmptyIsMissing(const TStrVec& fieldNames) const {
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

void CDataFrameClassificationRunner::writeOneRow(const core::CDataFrame& frame,
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

CDataFrameClassificationRunner::TLossFunctionUPtr
CDataFrameClassificationRunner::chooseLossFunction(const core::CDataFrame& frame,
                                                   std::size_t dependentVariableColumn) const {
    std::size_t categoryCount{
        frame.categoricalColumnValues()[dependentVariableColumn].size()};
    if (categoryCount == 2) {
        return std::make_unique<maths::boosted_tree::CLogistic>();
    }
    HANDLE_FATAL(<< "Input error: only binary classification is supported. "
                 << "Trying to predict '" << categoryCount << "' categories.");
    return nullptr;
}

CDataFrameAnalysisRunner::TInferenceModelDefinitionUPtr
CDataFrameClassificationRunner::inferenceModelDefinition(
    const CDataFrameAnalysisRunner::TStrVec& fieldNames,
    const CDataFrameAnalysisRunner::TStrVecVec& categoryNames) const {
    CClassificationInferenceModelBuilder builder(
        fieldNames, this->boostedTree().columnHoldingDependentVariable(), categoryNames);
    this->boostedTree().accept(builder);

    return std::make_unique<CInferenceModelDefinition>(builder.build());
}

const std::string& CDataFrameClassificationRunnerFactory::name() const {
    return NAME;
}

CDataFrameClassificationRunnerFactory::TRunnerUPtr
CDataFrameClassificationRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec) const {
    return std::make_unique<CDataFrameClassificationRunner>(spec);
}

CDataFrameClassificationRunnerFactory::TRunnerUPtr
CDataFrameClassificationRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec,
                                                const rapidjson::Value& jsonParameters) const {
    CDataFrameAnalysisConfigReader parameterReader =
        CDataFrameClassificationRunner::getParameterReader();
    auto parameters = parameterReader.read(jsonParameters);
    return std::make_unique<CDataFrameClassificationRunner>(spec, parameters);
}

const std::string CDataFrameClassificationRunnerFactory::NAME{"classification"};
}
}
