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

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameBoostedTreeRunner.h>
#include <api/ElasticsearchStateIndex.h>

#include <rapidjson/document.h>

namespace ml {
namespace api {
namespace {
// Configuration
const std::string NUM_TOP_CLASSES{"num_top_classes"};

// Output
const std::string IS_TRAINING_FIELD_NAME{"is_training"};
const std::string TOP_CLASSES_FIELD_NAME{"top_classes"};
const std::string CLASS_NAME_FIELD_NAME{"class_name"};
const std::string CLASS_PROBABILITY_FIELD_NAME{"class_probability"};
}

const CDataFrameAnalysisConfigReader CDataFrameClassificationRunner::getParameterReader() {
    CDataFrameAnalysisConfigReader theReader =
        CDataFrameBoostedTreeRunner::getParameterReader();
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

// The only field for which empty value should be treated as missing is dependent variable
// which has empty value for non-training rows.
void CDataFrameClassificationRunner::columnsForWhichEmptyIsMissing(TStrVec& fieldNames) const {
    fieldNames.push_back(dependentVariableFieldName());
}

void CDataFrameClassificationRunner::writeOneRow(const TStrVec&,
                                                 const TStrVecVec& categoricalFieldValues,
                                                 TRowRef row,
                                                 core::CRapidJsonConcurrentLineWriter& writer) const {
    const auto& tree{boostedTree()};
    std::size_t columnHoldingDependentVariable = tree->columnHoldingDependentVariable();
    std::size_t columnHoldingPrediction = tree->columnHoldingPrediction(row.numberColumns());
    const double dependentVariable = row[columnHoldingDependentVariable];
    const std::uint64_t prediction = std::lround(row[columnHoldingPrediction]);
    if (prediction >= categoricalFieldValues[columnHoldingDependentVariable].size()) {
        HANDLE_FATAL(<< "Index out of bounds: " << prediction);
    }
    std::string predictedClassName =
        categoricalFieldValues[columnHoldingDependentVariable][prediction];

    writer.StartObject();
    writer.Key(predictionFieldName());
    writer.String(predictedClassName);
    writer.Key(IS_TRAINING_FIELD_NAME);
    writer.Bool(maths::CDataFrameUtils::isMissing(dependentVariable) == false);
    // TODO: Uncomment and adapt the code below once top classes feature is implemented
    //       See https://github.com/elastic/ml-cpp/issues/712
    /*
    if (m_NumTopClasses > 0) {
        writer.Key(TOP_CLASSES_FIELD_NAME);
        writer.StartArray();
        for (std::size_t i = 0; i < m_NumTopClasses; i++) {
            writer.StartObject();
            writer.Key(CLASS_NAME_FIELD_NAME);
            writer.String(???);
            writer.Key(CLASS_PROBABILITY_FIELD_NAME);
            writer.Double(???);
            writer.EndObject();
        }
        writer.EndArray();
    }
    */
    writer.EndObject();
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
