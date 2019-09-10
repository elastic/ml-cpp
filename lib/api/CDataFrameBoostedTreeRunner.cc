/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameBoostedTreeRunner.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CProgramCounters.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/CStopWatch.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CDataFrameUtils.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>

#include <core/CJsonStatePersistInserter.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>

namespace ml {
namespace api {
namespace {
// Configuration
const std::string DEPENDENT_VARIABLE_NAME{"dependent_variable"};
const std::string PREDICTION_FIELD_NAME{"prediction_field_name"};
const std::string LAMBDA{"lambda"};
const std::string GAMMA{"gamma"};
const std::string ETA{"eta"};
const std::string MAXIMUM_NUMBER_TREES{"maximum_number_trees"};
const std::string FEATURE_BAG_FRACTION{"feature_bag_fraction"};
const std::string NUMBER_ROUNDS_PER_HYPERPARAMETER{"number_rounds_per_hyperparameter"};

const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
    CDataFrameAnalysisConfigReader theReader;
    theReader.addParameter(DEPENDENT_VARIABLE_NAME,
                           CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(PREDICTION_FIELD_NAME,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    // TODO objective function, support train and predict.
    theReader.addParameter(LAMBDA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(GAMMA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(ETA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(MAXIMUM_NUMBER_TREES,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(FEATURE_BAG_FRACTION,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(NUMBER_ROUNDS_PER_HYPERPARAMETER,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    return theReader;
}()};

// Output
const std::string IS_TRAINING_FIELD_NAME{"is_training"};
}

CDataFrameBoostedTreeRunner::CDataFrameBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec,
                                                         const rapidjson::Value& jsonParameters)
    : CDataFrameBoostedTreeRunner{spec} {

    auto parameters = PARAMETER_READER.read(jsonParameters);

    m_DependentVariableFieldName = parameters[DEPENDENT_VARIABLE_NAME].as<std::string>();

    m_PredictionFieldName = parameters[PREDICTION_FIELD_NAME].fallback(
        m_DependentVariableFieldName + "_prediction");

    std::size_t maximumNumberTrees{
        parameters[MAXIMUM_NUMBER_TREES].fallback(std::size_t{0})};

    std::size_t numberRoundsPerHyperparameter{
        parameters[NUMBER_ROUNDS_PER_HYPERPARAMETER].fallback(std::size_t{0})};

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
        HANDLE_FATAL(<< "Input error: bad eta value. It should be in the range (0, 1].");
    }
    if (featureBagFraction != -1.0 &&
        (featureBagFraction <= 0.0 || featureBagFraction > 1.0)) {
        HANDLE_FATAL(<< "Input error: bad feature bag fraction. "
                     << "It should be in the range (0, 1]");
    }

    m_BoostedTreeFactory = std::make_unique<maths::CBoostedTreeFactory>(
        maths::CBoostedTreeFactory::constructFromParameters(
            this->spec().numberThreads(), std::make_unique<maths::boosted_tree::CMse>()));

    (*m_BoostedTreeFactory).progressCallback(this->progressRecorder()).memoryUsageCallback([this](std::int64_t delta) {
        std::int64_t memory{m_Memory.fetch_add(delta)};
        if (memory >= 0) {
            core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage).max(memory);
        } else {
            // Something has gone wrong with memory estimation. Trap this case
            // to avoid underflowing the peak memory usage statistic.
            LOG_DEBUG(<< "Memory estimate " << memory << " is negative!");
        }
    });

    // callback for storing intermediate training state
    (*m_BoostedTreeFactory).trainingStateCallback(statePersister());
    if (lambda >= 0.0) {
        m_BoostedTreeFactory->lambda(lambda);
    }
    if (gamma >= 0.0) {
        m_BoostedTreeFactory->gamma(gamma);
    }
    if (eta > 0.0 && eta <= 1.0) {
        m_BoostedTreeFactory->eta(eta);
    }
    if (maximumNumberTrees > 0) {
        m_BoostedTreeFactory->maximumNumberTrees(maximumNumberTrees);
    }
    if (featureBagFraction > 0.0 && featureBagFraction <= 1.0) {
        m_BoostedTreeFactory->featureBagFraction(featureBagFraction);
    }
    if (numberRoundsPerHyperparameter > 0) {
        m_BoostedTreeFactory->maximumOptimisationRoundsPerHyperparameter(numberRoundsPerHyperparameter);
    }
}

CDataFrameBoostedTreeRunner::CDataFrameBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec)
    : CDataFrameAnalysisRunner{spec}, m_Memory{0} {
}

CDataFrameBoostedTreeRunner::~CDataFrameBoostedTreeRunner() = default;

std::size_t CDataFrameBoostedTreeRunner::numberExtraColumns() const {
    return m_BoostedTreeFactory->numberExtraColumnsForTrain();
}

void CDataFrameBoostedTreeRunner::writeOneRow(const TStrVec&,
                                              TRowRef row,
                                              core::CRapidJsonConcurrentLineWriter& writer) const {
    if (m_BoostedTree == nullptr) {
        HANDLE_FATAL(<< "Internal error: boosted tree object missing. Please report this error.");
    } else {
        writer.StartObject();
        writer.Key(m_PredictionFieldName);
        writer.Double(row[m_BoostedTree->columnHoldingPrediction(row.numberColumns())]);
        writer.Key(IS_TRAINING_FIELD_NAME);
        writer.Bool(maths::CDataFrameUtils::isMissing(
                        row[m_BoostedTree->columnHoldingDependentVariable()]) == false);
        writer.EndObject();
    }
}

void CDataFrameBoostedTreeRunner::runImpl(const TStrVec& featureNames,
                                          core::CDataFrame& frame) {
    auto dependentVariableColumn = std::find(
        featureNames.begin(), featureNames.end(), m_DependentVariableFieldName);
    if (dependentVariableColumn == featureNames.end()) {
        HANDLE_FATAL(<< "Input error: supplied variable to predict '"
                     << m_DependentVariableFieldName << "' is missing from training"
                     << " data " << core::CContainerPrinter::print(featureNames));
        return;
    }

    core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage) =
        this->estimateMemoryUsage(frame.numberRows(),
                                  frame.numberRows() / this->numberPartitions(),
                                  frame.numberColumns() + this->numberExtraColumns());

    core::CStopWatch watch{true};

    m_BoostedTree = m_BoostedTreeFactory->buildFor(
        frame, dependentVariableColumn - featureNames.begin());
    m_BoostedTree->train();
    m_BoostedTree->predict();

    core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) = watch.stop();
}

std::size_t CDataFrameBoostedTreeRunner::estimateBookkeepingMemoryUsage(
    std::size_t /*numberPartitions*/,
    std::size_t totalNumberRows,
    std::size_t /*partitionNumberRows*/,
    std::size_t numberColumns) const {
    return m_BoostedTreeFactory->estimateMemoryUsage(totalNumberRows, numberColumns);
}

const std::string& CDataFrameBoostedTreeRunnerFactory::name() const {
    return NAME;
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

const std::string CDataFrameBoostedTreeRunnerFactory::NAME{"regression"};
}
}
