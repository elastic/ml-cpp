/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameBoostedTreeRunner.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <maths/CBoostedTree.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>

namespace ml {
namespace api {
namespace {
// Configuration
const char* const DEPENDENT_VARIABLE{"dependent_variable"};
const char* const LAMBDA{"lambda"};
const char* const GAMMA{"gamma"};
const char* const ETA{"eta"};
const char* const MAXIMUM_NUMBER_TREES{"maximum_number_trees"};
const char* const FEATURE_BAG_FRACTION{"feature_bag_fraction"};

const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
    CDataFrameAnalysisConfigReader theReader;
    theReader.addParameter(DEPENDENT_VARIABLE,
                           CDataFrameAnalysisConfigReader::E_RequiredParameter);
    // TODO objective function, support train and predict.
    theReader.addParameter(LAMBDA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(GAMMA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(ETA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(MAXIMUM_NUMBER_TREES,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(FEATURE_BAG_FRACTION,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    return theReader;
}()};

// Output
const char* const PREDICTION{"prediction"};
}

CDataFrameBoostedTreeRunner::CDataFrameBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec,
                                                         const rapidjson::Value& jsonParameters)
    : CDataFrameBoostedTreeRunner{spec} {

    auto parameters = PARAMETER_READER.read(jsonParameters);

    std::size_t dependentVariable{parameters[DEPENDENT_VARIABLE].as<std::size_t>()};

    std::size_t maximumNumberTrees{
        parameters[MAXIMUM_NUMBER_TREES].fallback(std::size_t{0})};

    double lambda{parameters[LAMBDA].fallback(-1.0)};
    double gamma{parameters[GAMMA].fallback(-1.0)};
    double eta{parameters[ETA].fallback(-1.0)};
    double featureBagFraction{parameters[FEATURE_BAG_FRACTION].fallback(-1.0)};
    if (lambda != -1.0 && lambda < 0.0) {
        HANDLE_FATAL(<< "Input error: bad lambda value. It should be non-negative.");
    }
    if (gamma != -1.0 && gamma < 0.0) {
        HANDLE_FATAL(<< "Input error: bad gamma value. It should be non-negative.");
    }
    if (eta != -1.0 && (eta <= 0.0 || eta > 1.0)) {
        HANDLE_FATAL(<< "Input error: bad eta value. "
                     << "It should be in the range (0, 1].");
    }
    if (featureBagFraction != -1.0 &&
        (featureBagFraction <= 0.0 || featureBagFraction > 1.0)) {
        HANDLE_FATAL(<< "Input error: bad feature bag fraction. "
                     << "It should be in the range (0, 1]");
    }

    m_BoostedTree = std::make_unique<maths::CBoostedTree>(
        this->spec().numberThreads(), dependentVariable,
        std::make_unique<maths::boosted_tree::CMse>());
    if (lambda >= 0.0) {
        m_BoostedTree->lambda(lambda);
    }
    if (gamma >= 0.0) {
        m_BoostedTree->gamma(gamma);
    }
    if (eta > 0.0 && eta <= 1.0) {
        m_BoostedTree->eta(eta);
    }
    if (maximumNumberTrees > 0) {
        m_BoostedTree->maximumNumberTrees(maximumNumberTrees);
    }
    if (featureBagFraction > 0.0 && featureBagFraction <= 1.0) {
        m_BoostedTree->featureBagFraction(featureBagFraction);
    }
}

CDataFrameBoostedTreeRunner::CDataFrameBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec)
    : CDataFrameAnalysisRunner{spec} {
}

CDataFrameBoostedTreeRunner::~CDataFrameBoostedTreeRunner() {
}

std::size_t CDataFrameBoostedTreeRunner::numberExtraColumns() const {
    return m_BoostedTree->numberExtraColumnsForTrain();
}

void CDataFrameBoostedTreeRunner::writeOneRow(const TStrVec&,
                                              TRowRef row,
                                              core::CRapidJsonConcurrentLineWriter& writer) const {
    writer.StartObject();
    writer.Key(PREDICTION);
    writer.Double(row[m_BoostedTree->columnHoldingPrediction(row.numberColumns())]);
    writer.EndObject();
}

void CDataFrameBoostedTreeRunner::runImpl(core::CDataFrame& frame) {
    m_BoostedTree->train(frame, this->progressRecorder());
}

std::size_t CDataFrameBoostedTreeRunner::estimateBookkeepingMemoryUsage(
    std::size_t numberPartitions,
    std::size_t totalNumberRows,
    std::size_t partitionNumberRows,
    std::size_t numberColumns) const {
    // TODO
    return 0;
}

const char* CDataFrameBoostedTreeRunnerFactory::name() const {
    return "regression";
}

CDataFrameBoostedTreeRunnerFactory::TRunnerUPtr
CDataFrameBoostedTreeRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec) const {
    return std::make_unique<CDataFrameBoostedTreeRunner>(spec);
}

CDataFrameBoostedTreeRunnerFactory::TRunnerUPtr
CDataFrameBoostedTreeRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec,
                                             const rapidjson::Value& params) const {
    return std::make_unique<CDataFrameBoostedTreeRunner>(spec, params);
}
}
}
