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
#include <core/CStateDecompressor.h>
#include <core/CStopWatch.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CDataFrameUtils.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CInferenceModelDefinition.h>
#include <api/ElasticsearchStateIndex.h>

#include <api/CInferenceModelFormatter.h>
#include <core/CJsonStatePersistInserter.h>
#include <rapidjson/document.h>

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
const std::string BAYESIAN_OPTIMISATION_RESTARTS{"bayesian_optimisation_restarts"};

const std::string RESULT_INFERENCE_MODEL{"inference_model"};



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
    theReader.addParameter(BAYESIAN_OPTIMISATION_RESTARTS,
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
    std::size_t bayesianOptimisationRestarts{
        parameters[BAYESIAN_OPTIMISATION_RESTARTS].fallback(std::size_t{0})};

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

    (*m_BoostedTreeFactory)
        .progressCallback(this->progressRecorder())
        .trainingStateCallback(this->statePersister())
        .memoryUsageCallback(this->memoryEstimator());

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
    if (bayesianOptimisationRestarts > 0) {
        m_BoostedTreeFactory->bayesianOptimisationRestarts(bayesianOptimisationRestarts);
    }
}

CDataFrameBoostedTreeRunner::TMemoryEstimator CDataFrameBoostedTreeRunner::memoryEstimator() {
    return [this](int64_t delta) {
        int64_t memory{m_Memory.fetch_add(delta)};
        if (memory >= 0) {
            core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage).max(memory);
        } else {
            // Something has gone wrong with memory estimation. Trap this case
            // to avoid underflowing the peak memory usage statistic.
            LOG_DEBUG(<< "Memory estimate " << memory << " is negative!");
        }
    };
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
    auto restoreSearcher{this->spec().restoreSearcher()};
    bool treeRestored{false};
    if (restoreSearcher != nullptr) {
        treeRestored = this->restoreBoostedTree(
            frame, dependentVariableColumn - featureNames.begin(), restoreSearcher);
    }

    if (treeRestored == false) {
        m_BoostedTree = m_BoostedTreeFactory->buildFor(
            frame, dependentVariableColumn - featureNames.begin());
    }
    m_BoostedTree->train();
    m_BoostedTree->predict();

    core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) = watch.stop();
}

bool CDataFrameBoostedTreeRunner::restoreBoostedTree(core::CDataFrame& frame,
                                                     std::size_t dependentVariableColumn,
                                                     TDataSearcherUPtr& restoreSearcher) {
    // Restore from Elasticsearch compressed data
    try {
        core::CStateDecompressor decompressor(*restoreSearcher);
        decompressor.setStateRestoreSearch(
            ML_STATE_INDEX, getRegressionStateId(this->spec().jobId()));
        core::CDataSearcher::TIStreamP inputStream{decompressor.search(1, 1)}; // search arguments are ignored
        if (inputStream == nullptr) {
            LOG_ERROR(<< "Unable to connect to data store");
            return false;
        }

        if (inputStream->bad()) {
            LOG_ERROR(<< "State restoration search returned bad stream");
            return false;
        }

        if (inputStream->fail()) {
            // This is fatal. If the stream exists and has failed then state is missing
            LOG_ERROR(<< "State restoration search returned failed stream");
            return false;
        }

        m_BoostedTree = maths::CBoostedTreeFactory::constructFromString(*inputStream)
                            .progressCallback(this->progressRecorder())
                            .trainingStateCallback(this->statePersister())
                            .memoryUsageCallback(this->memoryEstimator())
                            .buildFor(frame, dependentVariableColumn);
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
        return false;
    }
    return true;
}

std::size_t CDataFrameBoostedTreeRunner::estimateBookkeepingMemoryUsage(
    std::size_t /*numberPartitions*/,
    std::size_t totalNumberRows,
    std::size_t /*partitionNumberRows*/,
    std::size_t numberColumns) const {
    return m_BoostedTreeFactory->estimateMemoryUsage(totalNumberRows, numberColumns);
}

void
CDataFrameBoostedTreeRunner::serializeRunner(const TStrVec &fieldNames, const TStrSizeUMapVec &categoryNameMap,
                                             core::CRapidJsonConcurrentLineWriter &writer) const {
    std::stringstream strm;
    {
        core::CJsonStatePersistInserter inserter(strm);
        m_BoostedTree->acceptPersistInserter(inserter);
        strm.flush();
    }
    LOG_DEBUG(<< "serializeRunner: " << strm.str());

    CInferenceModelFormatter formatter{strm.str(), fieldNames, categoryNameMap};
    LOG_DEBUG(<< "Inference model json: " << formatter.toString());

    rapidjson::Document doc = writer.makeDoc();
    doc.Parse(formatter.toString());
    writer.StartObject();
    writer.Key(RESULT_INFERENCE_MODEL);
    writer.write(doc);
    writer.EndObject();
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
