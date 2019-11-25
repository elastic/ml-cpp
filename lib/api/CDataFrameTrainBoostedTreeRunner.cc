/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameTrainBoostedTreeRunner.h>

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

#include <api/CBoostedTreeInferenceModelBuilder.h>
#include <core/CJsonStatePersistInserter.h>
#include <rapidjson/document.h>

namespace ml {
namespace api {

const CDataFrameAnalysisConfigReader& CDataFrameTrainBoostedTreeRunner::parameterReader() {
    static const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
        CDataFrameAnalysisConfigReader theReader;
        theReader.addParameter(DEPENDENT_VARIABLE_NAME,
                               CDataFrameAnalysisConfigReader::E_RequiredParameter);
        theReader.addParameter(PREDICTION_FIELD_NAME,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(DOWNSAMPLE_ROWS_PER_FEATURE,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(ALPHA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(LAMBDA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(GAMMA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(ETA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(SOFT_TREE_DEPTH_LIMIT,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(SOFT_TREE_DEPTH_TOLERANCE,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(MAXIMUM_NUMBER_TREES,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(FEATURE_BAG_FRACTION,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(NUMBER_FOLDS, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(NUMBER_ROUNDS_PER_HYPERPARAMETER,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(BAYESIAN_OPTIMISATION_RESTARTS,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        return theReader;
    }()};
    return PARAMETER_READER;
}

CDataFrameTrainBoostedTreeRunner::CDataFrameTrainBoostedTreeRunner(
    const CDataFrameAnalysisSpecification& spec,
    const CDataFrameAnalysisParameters& parameters)
    : CDataFrameTrainBoostedTreeRunner{spec} {

    m_DependentVariableFieldName = parameters[DEPENDENT_VARIABLE_NAME].as<std::string>();

    m_PredictionFieldName = parameters[PREDICTION_FIELD_NAME].fallback(
        m_DependentVariableFieldName + "_prediction");

    std::size_t downsampleRowsPerFeature{
        parameters[DOWNSAMPLE_ROWS_PER_FEATURE].fallback(std::size_t{0})};

    std::size_t maximumNumberTrees{
        parameters[MAXIMUM_NUMBER_TREES].fallback(std::size_t{0})};
    std::size_t numberFolds{parameters[NUMBER_FOLDS].fallback(std::size_t{0})};
    std::size_t numberRoundsPerHyperparameter{
        parameters[NUMBER_ROUNDS_PER_HYPERPARAMETER].fallback(std::size_t{0})};
    std::size_t bayesianOptimisationRestarts{
        parameters[BAYESIAN_OPTIMISATION_RESTARTS].fallback(std::size_t{0})};

    double alpha{parameters[ALPHA].fallback(-1.0)};
    double lambda{parameters[LAMBDA].fallback(-1.0)};
    double gamma{parameters[GAMMA].fallback(-1.0)};
    double eta{parameters[ETA].fallback(-1.0)};
    double softTreeDepthLimit{parameters[SOFT_TREE_DEPTH_LIMIT].fallback(-1.0)};
    double softTreeDepthTolerance{parameters[SOFT_TREE_DEPTH_TOLERANCE].fallback(-1.0)};
    double featureBagFraction{parameters[FEATURE_BAG_FRACTION].fallback(-1.0)};
    bool shapValues{parameters[SHAP_VALUES].fallback(false)};
    if (alpha != -1.0 && alpha < 0.0) {
        HANDLE_FATAL(<< "Input error: bad alpha value. It should be non-negative.");
    }
    if (lambda != -1.0 && lambda < 0.0) {
        HANDLE_FATAL(<< "Input error: bad lambda value. It should be non-negative.");
    }
    if (gamma != -1.0 && gamma < 0.0) {
        HANDLE_FATAL(<< "Input error: bad gamma value. It should be non-negative.");
    }
    if (eta != -1.0 && (eta <= 0.0 || eta > 1.0)) {
        HANDLE_FATAL(<< "Input error: bad eta value. It should be in the range (0, 1].");
    }
    if (softTreeDepthLimit != -1.0 && softTreeDepthLimit < 0.0) {
        HANDLE_FATAL(<< "Input error: bad tree depth limit value. It should be non-negative.")
    }
    if (softTreeDepthTolerance != -1.0 && softTreeDepthTolerance <= 0.0) {
        HANDLE_FATAL(<< "Input error: bad tree depth limit value. It should be positive.")
    }
    if (featureBagFraction != -1.0 &&
        (featureBagFraction <= 0.0 || featureBagFraction > 1.0)) {
        HANDLE_FATAL(<< "Input error: bad feature bag fraction. "
                     << "It should be in the range (0, 1]");
    }

    m_BoostedTreeFactory = std::make_unique<maths::CBoostedTreeFactory>(
        maths::CBoostedTreeFactory::constructFromParameters(this->spec().numberThreads()));

    (*m_BoostedTreeFactory)
        .progressCallback(this->progressRecorder())
        .trainingStateCallback(this->statePersister())
        .memoryUsageCallback(this->memoryMonitor(counter_t::E_DFTPMPeakMemoryUsage));

    if (downsampleRowsPerFeature > 0) {
        m_BoostedTreeFactory->initialDownsampleRowsPerFeature(
            static_cast<double>(downsampleRowsPerFeature));
    }
    if (alpha >= 0.0) {
        m_BoostedTreeFactory->depthPenaltyMultiplier(alpha);
    }
    if (gamma >= 0.0) {
        m_BoostedTreeFactory->treeSizePenaltyMultiplier(gamma);
    }
    if (lambda >= 0.0) {
        m_BoostedTreeFactory->leafWeightPenaltyMultiplier(lambda);
    }
    if (eta > 0.0 && eta <= 1.0) {
        m_BoostedTreeFactory->eta(eta);
    }
    if (softTreeDepthLimit >= 0.0) {
        m_BoostedTreeFactory->softTreeDepthLimit(softTreeDepthLimit);
    }
    if (softTreeDepthTolerance > 0.0) {
        m_BoostedTreeFactory->softTreeDepthTolerance(softTreeDepthTolerance);
    }
    if (maximumNumberTrees > 0) {
        m_BoostedTreeFactory->maximumNumberTrees(maximumNumberTrees);
    }
    if (featureBagFraction > 0.0 && featureBagFraction <= 1.0) {
        m_BoostedTreeFactory->featureBagFraction(featureBagFraction);
    }
    if (numberFolds > 1) {
        m_BoostedTreeFactory->numberFolds(numberFolds);
    }
    if (numberRoundsPerHyperparameter > 0) {
        m_BoostedTreeFactory->maximumOptimisationRoundsPerHyperparameter(numberRoundsPerHyperparameter);
    }
    if (bayesianOptimisationRestarts > 0) {
        m_BoostedTreeFactory->bayesianOptimisationRestarts(bayesianOptimisationRestarts);
    }
    if (shapValues == true) {
        m_BoostedTreeFactory->shapValues(shapValues);
    }
}

CDataFrameTrainBoostedTreeRunner::CDataFrameTrainBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec)
    : CDataFrameAnalysisRunner{spec} {
}

CDataFrameTrainBoostedTreeRunner::~CDataFrameTrainBoostedTreeRunner() = default;

std::size_t CDataFrameTrainBoostedTreeRunner::numberExtraColumns() const {
    return m_BoostedTreeFactory->numberExtraColumnsForTrain();
}

const std::string& CDataFrameTrainBoostedTreeRunner::dependentVariableFieldName() const {
    return m_DependentVariableFieldName;
}

const std::string& CDataFrameTrainBoostedTreeRunner::predictionFieldName() const {
    return m_PredictionFieldName;
}

const maths::CBoostedTree& CDataFrameTrainBoostedTreeRunner::boostedTree() const {
    if (m_BoostedTree == nullptr) {
        HANDLE_FATAL(<< "Internal error: boosted tree missing. Please report this problem.");
    }
    return *m_BoostedTree;
}

maths::CBoostedTreeFactory& CDataFrameTrainBoostedTreeRunner::boostedTreeFactory() {
    if (m_BoostedTreeFactory == nullptr) {
        HANDLE_FATAL(<< "Internal error: boosted tree factory missing. Please report this problem.");
    }
    return *m_BoostedTreeFactory;
}

void CDataFrameTrainBoostedTreeRunner::runImpl(core::CDataFrame& frame) {
    auto dependentVariablePos = std::find(frame.columnNames().begin(),
                                          frame.columnNames().end(),
                                          m_DependentVariableFieldName);
    if (dependentVariablePos == frame.columnNames().end()) {
        HANDLE_FATAL(<< "Input error: supplied variable to predict '"
                     << m_DependentVariableFieldName << "' is missing from training"
                     << " data " << core::CContainerPrinter::print(frame.columnNames()));
        return;
    }

    core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage) =
        this->estimateMemoryUsage(frame.numberRows(),
                                  frame.numberRows() / this->numberPartitions(),
                                  frame.numberColumns() + this->numberExtraColumns());

    core::CStopWatch watch{true};

    std::size_t dependentVariableColumn(dependentVariablePos -
                                        frame.columnNames().begin());

    auto restoreSearcher{this->spec().restoreSearcher()};
    bool treeRestored{false};
    if (restoreSearcher != nullptr) {
        treeRestored = this->restoreBoostedTree(frame, dependentVariableColumn, restoreSearcher);
    }
    if (treeRestored == false) {
        auto loss = this->chooseLossFunction(frame, dependentVariableColumn);
        m_BoostedTree = m_BoostedTreeFactory->buildFor(frame, std::move(loss),
                                                       dependentVariableColumn);
    }

    m_BoostedTree->train();
    m_BoostedTree->predict();

    core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) = watch.stop();
}

bool CDataFrameTrainBoostedTreeRunner::restoreBoostedTree(core::CDataFrame& frame,
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
                            .memoryUsageCallback(this->memoryMonitor(counter_t::E_DFTPMPeakMemoryUsage))
                            .restoreFor(frame, dependentVariableColumn);
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
        return false;
    }
    return true;
}

std::size_t CDataFrameTrainBoostedTreeRunner::estimateBookkeepingMemoryUsage(
    std::size_t /*numberPartitions*/,
    std::size_t totalNumberRows,
    std::size_t /*partitionNumberRows*/,
    std::size_t numberColumns) const {
    return m_BoostedTreeFactory->estimateMemoryUsage(totalNumberRows, numberColumns);
}

// clang-format off
const std::string CDataFrameTrainBoostedTreeRunner::DEPENDENT_VARIABLE_NAME{"dependent_variable"};
const std::string CDataFrameTrainBoostedTreeRunner::PREDICTION_FIELD_NAME{"prediction_field_name"};
const std::string CDataFrameTrainBoostedTreeRunner::DOWNSAMPLE_ROWS_PER_FEATURE{"downsample_rows_per_feature"};
const std::string CDataFrameTrainBoostedTreeRunner::ALPHA{"alpha"};
const std::string CDataFrameTrainBoostedTreeRunner::LAMBDA{"lambda"};
const std::string CDataFrameTrainBoostedTreeRunner::GAMMA{"gamma"};
const std::string CDataFrameTrainBoostedTreeRunner::ETA{"eta"};
const std::string CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_LIMIT{"soft_tree_depth_limit"};
const std::string CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_TOLERANCE{"soft_tree_depth_tolerance"};
const std::string CDataFrameTrainBoostedTreeRunner::MAXIMUM_NUMBER_TREES{"maximum_number_trees"};
const std::string CDataFrameTrainBoostedTreeRunner::FEATURE_BAG_FRACTION{"feature_bag_fraction"};
const std::string CDataFrameTrainBoostedTreeRunner::NUMBER_FOLDS{"number_folds"};
const std::string CDataFrameTrainBoostedTreeRunner::NUMBER_ROUNDS_PER_HYPERPARAMETER{"number_rounds_per_hyperparameter"};
const std::string CDataFrameTrainBoostedTreeRunner::BAYESIAN_OPTIMISATION_RESTARTS{"bayesian_optimisation_restarts"};
// clang-format on
}
}
