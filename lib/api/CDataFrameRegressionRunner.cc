/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameRegressionRunner.h>

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
// Output
const std::string IS_TRAINING_FIELD_NAME{"is_training"};
}

const CDataFrameAnalysisConfigReader CDataFrameRegressionRunner::getParameterReader() {
    return CDataFrameBoostedTreeRunner::getParameterReader();
}

CDataFrameRegressionRunner::CDataFrameRegressionRunner(
    const CDataFrameAnalysisSpecification& spec,
    const CDataFrameAnalysisConfigReader::CParameters& parameters)
    : CDataFrameBoostedTreeRunner{spec, parameters} {
}

CDataFrameRegressionRunner::CDataFrameRegressionRunner(const CDataFrameAnalysisSpecification& spec)
    : CDataFrameBoostedTreeRunner{spec} {
}

void CDataFrameRegressionRunner::writeOneRow(const TStrVec&,
                                             const TStrVecVec&,
                                             TRowRef row,
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
    writer.EndObject();
}

const std::string& CDataFrameRegressionRunnerFactory::name() const {
    return NAME;
}

CDataFrameRegressionRunnerFactory::TRunnerUPtr
CDataFrameRegressionRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec) const {
    return std::make_unique<CDataFrameRegressionRunner>(spec);
}

CDataFrameRegressionRunnerFactory::TRunnerUPtr
CDataFrameRegressionRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec,
                                            const rapidjson::Value& jsonParameters) const {
    CDataFrameAnalysisConfigReader parameterReader =
        CDataFrameRegressionRunner::getParameterReader();
    auto parameters = parameterReader.read(jsonParameters);
    return std::make_unique<CDataFrameRegressionRunner>(spec, parameters);
}

const std::string CDataFrameRegressionRunnerFactory::NAME{"regression"};
}
}
