/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <api/CDataFrameTrainBoostedTreeRunner.h>

#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CLogger.h>
#include <core/CPackedBitVector.h>
#include <core/CProgramCounters.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/CStateDecompressor.h>
#include <core/CStopWatch.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeFactory.h>
#include <maths/analytics/CBoostedTreeLoss.h>
#include <maths/analytics/CDataFrameUtils.h>

#include <api/CBoostedTreeInferenceModelBuilder.h>
#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataSummarizationJsonWriter.h>
#include <api/CInferenceModelDefinition.h>
#include <api/CRetrainableModelJsonReader.h>
#include <api/ElasticsearchStateIndex.h>

#include <rapidjson/document.h>

#include <limits>

namespace ml {
namespace api {
namespace {
const std::size_t NUMBER_ROUNDS_PER_HYPERPARAMETER_IS_UNSET{
    std::numeric_limits<std::size_t>::max()};
}

const CDataFrameAnalysisConfigReader& CDataFrameTrainBoostedTreeRunner::parameterReader() {
    static const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
        CDataFrameAnalysisConfigReader theReader;
        theReader.addParameter(RANDOM_NUMBER_GENERATOR_SEED,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(DEPENDENT_VARIABLE_NAME,
                               CDataFrameAnalysisConfigReader::E_RequiredParameter);
        theReader.addParameter(PREDICTION_FIELD_NAME,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(DOWNSAMPLE_ROWS_PER_FEATURE,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(DOWNSAMPLE_FACTOR,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(ALPHA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(LAMBDA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(GAMMA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(ETA, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(ETA_GROWTH_RATE_PER_TREE,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(RETRAINED_TREE_ETA,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(SOFT_TREE_DEPTH_LIMIT,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(SOFT_TREE_DEPTH_TOLERANCE,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(MAX_TREES, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(FEATURE_BAG_FRACTION,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(PREDICTION_CHANGE_COST,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(TREE_TOPOLOGY_CHANGE_PENALTY,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(NUM_FOLDS, CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(TRAIN_FRACTION_PER_FOLD,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(STOP_CROSS_VALIDATION_EARLY,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(BAYESIAN_OPTIMISATION_RESTARTS,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(NUM_TOP_FEATURE_IMPORTANCE_VALUES,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(TRAINING_PERCENT_FIELD_NAME,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(FEATURE_PROCESSORS,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(EARLY_STOPPING_ENABLED,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(FORCE_ACCEPT_INCREMENTAL_TRAINING,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(DATA_SUMMARIZATION_FRACTION,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(TASK, CDataFrameAnalysisConfigReader::E_OptionalParameter,
                               {{TASK_TRAIN, int{ETask::E_Train}},
                                {TASK_UPDATE, int{ETask::E_Update}},
                                {TASK_PREDICT, int{ETask::E_Predict}}});
        theReader.addParameter(PREVIOUS_TRAIN_LOSS_GAP,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(PREVIOUS_TRAIN_NUM_ROWS,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        return theReader;
    }()};
    return PARAMETER_READER;
}

CDataFrameTrainBoostedTreeRunner::CDataFrameTrainBoostedTreeRunner(
    const CDataFrameAnalysisSpecification& spec,
    const CDataFrameAnalysisParameters& parameters,
    TLossFunctionUPtr loss,
    TDataFrameUPtrTemporaryDirectoryPtrPr* frameAndDirectory)
    : CDataFrameAnalysisRunner{spec}, m_NumberLossParameters{loss->numberParameters()},
      m_Instrumentation{spec.jobId(), spec.memoryLimit()} {

    if (loss == nullptr) {
        HANDLE_FATAL(<< "Internal error: must provide a loss function for training."
                     << " Please report this problem");
        return;
    }

    using TDoubleVec = std::vector<double>;

    m_DependentVariableFieldName = parameters[DEPENDENT_VARIABLE_NAME].as<std::string>();
    m_PredictionFieldName = parameters[PREDICTION_FIELD_NAME].fallback(
        m_DependentVariableFieldName + "_prediction");

    m_TrainingPercent = parameters[TRAINING_PERCENT_FIELD_NAME].fallback(100.0) / 100.0;

    m_Task = parameters[TASK].fallback(E_Train);

    std::size_t seed{parameters[RANDOM_NUMBER_GENERATOR_SEED].fallback(std::size_t{0})};

    std::size_t maxTrees{parameters[MAX_TREES].fallback(std::size_t{0})};
    std::size_t numberFolds{parameters[NUM_FOLDS].fallback(std::size_t{0})};
    std::size_t downsampleRowsPerFeature{
        parameters[DOWNSAMPLE_ROWS_PER_FEATURE].fallback(std::size_t{0})};
    double trainFractionPerFold{parameters[TRAIN_FRACTION_PER_FOLD].fallback(-1.0)};
    std::size_t numberRoundsPerHyperparameter{
        parameters[MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER].fallback(
            NUMBER_ROUNDS_PER_HYPERPARAMETER_IS_UNSET)};
    bool earlyStoppingEnabled{parameters[EARLY_STOPPING_ENABLED].fallback(true)};
    std::size_t bayesianOptimisationRestarts{
        parameters[BAYESIAN_OPTIMISATION_RESTARTS].fallback(std::size_t{0})};
    bool stopCrossValidationEarly{parameters[STOP_CROSS_VALIDATION_EARLY].fallback(true)};
    std::size_t numTopFeatureImportanceValues{
        parameters[NUM_TOP_FEATURE_IMPORTANCE_VALUES].fallback(std::size_t{0})};

    auto alpha = parameters[ALPHA].fallback(TDoubleVec{});
    auto lambda = parameters[LAMBDA].fallback(TDoubleVec{});
    auto gamma = parameters[GAMMA].fallback(TDoubleVec{});
    auto eta = parameters[ETA].fallback(TDoubleVec{});
    auto etaGrowthRatePerTree = parameters[ETA_GROWTH_RATE_PER_TREE].fallback(TDoubleVec{});
    auto retrainedTreeEta = parameters[RETRAINED_TREE_ETA].fallback(TDoubleVec{});
    auto softTreeDepthLimit = parameters[SOFT_TREE_DEPTH_LIMIT].fallback(TDoubleVec{});
    auto softTreeDepthTolerance =
        parameters[SOFT_TREE_DEPTH_TOLERANCE].fallback(TDoubleVec{});
    auto downsampleFactor = parameters[DOWNSAMPLE_FACTOR].fallback(TDoubleVec{});
    auto featureBagFraction = parameters[FEATURE_BAG_FRACTION].fallback(TDoubleVec{});
    auto predictionChangeCost = parameters[PREDICTION_CHANGE_COST].fallback(TDoubleVec{});
    auto treeTopologyChangePenalty =
        parameters[TREE_TOPOLOGY_CHANGE_PENALTY].fallback(TDoubleVec{});

    bool forceAcceptIncrementalTraining{
        parameters[FORCE_ACCEPT_INCREMENTAL_TRAINING].fallback(false)};
    double dataSummarizationFraction{parameters[DATA_SUMMARIZATION_FRACTION].fallback(-1.0)};
    double previousTrainLossGap{parameters[PREVIOUS_TRAIN_LOSS_GAP].fallback(-1.0)};
    std::size_t previousTrainNumberRows{
        parameters[PREVIOUS_TRAIN_NUM_ROWS].fallback(std::size_t{0})};

    if (parameters[FEATURE_PROCESSORS].jsonObject() != nullptr) {
        m_CustomProcessors.CopyFrom(*parameters[FEATURE_PROCESSORS].jsonObject(),
                                    m_CustomProcessors.GetAllocator());
    }

    if (std::any_of(alpha.begin(), alpha.end(), [](double x) { return x < 0.0; })) {
        HANDLE_FATAL(<< "Input error: '" << ALPHA << "' should be non-negative.");
    }
    if (std::any_of(lambda.begin(), lambda.end(),
                    [](double x) { return x < 0.0; })) {
        HANDLE_FATAL(<< "Input error: '" << LAMBDA << "' should be non-negative.");
    }
    if (std::any_of(gamma.begin(), gamma.end(), [](double x) { return x < 0.0; })) {
        HANDLE_FATAL(<< "Input error: '" << GAMMA << "' should be non-negative.");
    }
    if (std::any_of(eta.begin(), eta.end(),
                    [](double x) { return x <= 0.0 || x > 1.0; })) {
        HANDLE_FATAL(<< "Input error: '" << ETA << "' should be in the range (0, 1].");
    }
    if (std::any_of(etaGrowthRatePerTree.begin(), etaGrowthRatePerTree.end(),
                    [](double x) { return x <= 0.0; })) {
        HANDLE_FATAL(<< "Input error: '" << ETA_GROWTH_RATE_PER_TREE << "' should be positive.");
    }
    if (std::any_of(retrainedTreeEta.begin(), retrainedTreeEta.end(),
                    [](double x) { return x <= 0.0 || x > 1.0; })) {
        HANDLE_FATAL(<< "Input error: '" << RETRAINED_TREE_ETA
                     << "' should be in the range (0, 1].");
    }
    if (std::any_of(softTreeDepthLimit.begin(), softTreeDepthLimit.end(),
                    [](double x) { return x < 0.0; })) {
        HANDLE_FATAL(<< "Input error: '" << SOFT_TREE_DEPTH_LIMIT << "' should be non-negative.");
    }
    if (std::any_of(softTreeDepthTolerance.begin(), softTreeDepthTolerance.end(),
                    [](double x) { return x <= 0.0; })) {
        HANDLE_FATAL(<< "Input error: '" << SOFT_TREE_DEPTH_TOLERANCE << "' should be positive.");
    }
    if (std::any_of(downsampleFactor.begin(), downsampleFactor.end(),
                    [](double x) { return x <= 0.0 || x > 1.0; })) {
        HANDLE_FATAL(<< "Input error: '" << DOWNSAMPLE_FACTOR << "' should be in the range (0, 1]");
    }
    if (std::any_of(featureBagFraction.begin(), featureBagFraction.end(),
                    [](double x) { return x <= 0.0 || x > 1.0; })) {
        HANDLE_FATAL(<< "Input error: '" << FEATURE_BAG_FRACTION
                     << "' should be in the range (0, 1]");
    }
    if (std::any_of(predictionChangeCost.begin(), predictionChangeCost.end(),
                    [](double x) { return x < 0.0; })) {
        HANDLE_FATAL(<< "Input error: '" << PREDICTION_CHANGE_COST << "' should be non-negative");
    }
    if (std::any_of(treeTopologyChangePenalty.begin(), treeTopologyChangePenalty.end(),
                    [](double x) { return x < 0.0; })) {
        HANDLE_FATAL(<< "Input error: '" << TREE_TOPOLOGY_CHANGE_PENALTY
                     << "' should be non-negative");
    }

    this->computeAndSaveExecutionStrategy();

    m_BoostedTreeFactory = this->boostedTreeFactory(std::move(loss), frameAndDirectory);
    (*m_BoostedTreeFactory)
        .seed(seed)
        .stopCrossValidationEarly(stopCrossValidationEarly)
        .analysisInstrumentation(m_Instrumentation)
        .trainingStateCallback(this->statePersister())
        .earlyStoppingEnabled(earlyStoppingEnabled)
        .forceAcceptIncrementalTraining(forceAcceptIncrementalTraining)
        .downsampleFactor(std::move(downsampleFactor))
        .depthPenaltyMultiplier(std::move(alpha))
        .treeSizePenaltyMultiplier(std::move(gamma))
        .leafWeightPenaltyMultiplier(std::move(lambda))
        .eta(std::move(eta))
        .etaGrowthRatePerTree(std::move(etaGrowthRatePerTree))
        .retrainedTreeEta(std::move(retrainedTreeEta))
        .softTreeDepthLimit(std::move(softTreeDepthLimit))
        .softTreeDepthTolerance(std::move(softTreeDepthTolerance))
        .featureBagFraction(std::move(featureBagFraction))
        .predictionChangeCost(std::move(predictionChangeCost))
        .treeTopologyChangePenalty(std::move(treeTopologyChangePenalty));

    if (downsampleRowsPerFeature > 0) {
        m_BoostedTreeFactory->initialDownsampleRowsPerFeature(
            static_cast<double>(downsampleRowsPerFeature));
    }
    if (maxTrees > 0) {
        m_BoostedTreeFactory->maximumNumberTrees(maxTrees);
    }
    if (numberFolds > 1) {
        m_BoostedTreeFactory->numberFolds(numberFolds);
    }
    if (trainFractionPerFold > 0.0) {
        m_BoostedTreeFactory->trainFractionPerFold(trainFractionPerFold);
    }
    if (numberRoundsPerHyperparameter != NUMBER_ROUNDS_PER_HYPERPARAMETER_IS_UNSET) {
        m_BoostedTreeFactory->maximumOptimisationRoundsPerHyperparameter(numberRoundsPerHyperparameter);
    }
    if (bayesianOptimisationRestarts > 0) {
        m_BoostedTreeFactory->bayesianOptimisationRestarts(bayesianOptimisationRestarts);
    }
    if (numTopFeatureImportanceValues > 0) {
        m_BoostedTreeFactory->numberTopShapValues(numTopFeatureImportanceValues);
    }
    if (dataSummarizationFraction > 0) {
        m_BoostedTreeFactory->dataSummarizationFraction(dataSummarizationFraction);
    }
    if (previousTrainLossGap > 0.0) {
        m_BoostedTreeFactory->previousTrainLossGap(previousTrainLossGap);
    }
    if (previousTrainNumberRows > 0) {
        m_BoostedTreeFactory->previousTrainNumberRows(previousTrainNumberRows);
    }
}

CDataFrameTrainBoostedTreeRunner::~CDataFrameTrainBoostedTreeRunner() = default;

std::size_t CDataFrameTrainBoostedTreeRunner::numberExtraColumns() const {
    return maths::analytics::CBoostedTreeFactory::estimatedExtraColumnsForTrain(
        this->spec().numberColumns(), m_NumberLossParameters);
}

std::size_t CDataFrameTrainBoostedTreeRunner::dataFrameSliceCapacity() const {
    std::size_t sliceCapacity{core::dataFrameDefaultSliceCapacity(
        this->spec().numberColumns() + this->numberExtraColumns())};
    std::size_t numberThreads{this->spec().numberThreads()};
    if (numberThreads > 1) {
        std::size_t numberRows{this->spec().numberRows()};

        // Use at least one slice per thread because we parallelize work over slices.
        std::size_t capacityForOneSlicePerThread{(numberRows + numberThreads - 1) / numberThreads};
        sliceCapacity = std::min(sliceCapacity, capacityForOneSlicePerThread);

        // Round the slice size so number threads is a divisor of the number of slices.
        std::size_t numberSlices{numberRows / sliceCapacity};
        sliceCapacity = numberRows /
                        (numberThreads * ((numberSlices + numberThreads / 2) / numberThreads));
    }
    return std::max(sliceCapacity, std::size_t{128});
}

core::CPackedBitVector
CDataFrameTrainBoostedTreeRunner::rowsToWriteMask(const core::CDataFrame& frame) const {
    switch (m_Task) {
    case E_Train:
        return {frame.numberRows(), true};
    case E_Predict:
    case E_Update:
        return m_BoostedTree->newTrainingRowMask();
    }
}

const std::string& CDataFrameTrainBoostedTreeRunner::dependentVariableFieldName() const {
    return m_DependentVariableFieldName;
}

const std::string& CDataFrameTrainBoostedTreeRunner::predictionFieldName() const {
    return m_PredictionFieldName;
}

const maths::analytics::CBoostedTree& CDataFrameTrainBoostedTreeRunner::boostedTree() const {
    if (m_BoostedTree == nullptr) {
        HANDLE_FATAL(<< "Internal error: boosted tree missing. Please report this problem.");
    }
    return *m_BoostedTree;
}

maths::analytics::CBoostedTreeFactory& CDataFrameTrainBoostedTreeRunner::boostedTreeFactory() {
    if (m_BoostedTreeFactory == nullptr) {
        HANDLE_FATAL(<< "Internal error: boosted tree factory missing. Please report this problem.");
    }
    return *m_BoostedTreeFactory;
}

const maths::analytics::CBoostedTreeFactory&
CDataFrameTrainBoostedTreeRunner::boostedTreeFactory() const {
    if (m_BoostedTreeFactory == nullptr) {
        HANDLE_FATAL(<< "Internal error: boosted tree factory missing. Please report this problem.");
    }
    return *m_BoostedTreeFactory;
}

bool CDataFrameTrainBoostedTreeRunner::validate(const core::CDataFrame& frame) const {
    if (frame.numberColumns() <= 1) {
        HANDLE_FATAL(<< "Input error: analysis need at least one regressor.");
        return false;
    }
    if (frame.numberRows() > maths::analytics::CBoostedTreeFactory::maximumNumberRows()) {
        HANDLE_FATAL(<< "Input error: no more than "
                     << maths::analytics::CBoostedTreeFactory::maximumNumberRows()
                     << " are supported. You need to downsample your data.");
        return false;
    }
    return true;
}

void CDataFrameTrainBoostedTreeRunner::accept(CBoostedTreeInferenceModelBuilder& builder) const {
    if (m_CustomProcessors.IsNull() == false) {
        builder.addCustomProcessor(std::make_unique<COpaqueEncoding>(m_CustomProcessors));
    }
    this->boostedTree().accept(builder);
}

void CDataFrameTrainBoostedTreeRunner::computeAndSaveExecutionStrategy() {
    // We always use in core storage for the data frame for boosted tree training
    // because it is too slow to use disk.
    this->numberPartitions(1);
    this->maximumNumberRowsPerPartition(this->spec().numberRows());
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

    this->validate(frame, dependentVariableColumn);

    switch (m_Task) {
    case E_Train:
        m_BoostedTree = [&] {
            auto boostedTree = this->restoreBoostedTree(
                frame, dependentVariableColumn, this->spec().restoreSearcher());
            return boostedTree != nullptr
                       ? std::move(boostedTree)
                       : m_BoostedTreeFactory->buildForTrain(frame, dependentVariableColumn);
        }();
        m_BoostedTree->train();
        m_BoostedTree->predict();
        break;
    case E_Update:
        m_BoostedTree = m_BoostedTreeFactory->buildForTrainIncremental(frame, dependentVariableColumn);
        m_BoostedTree->trainIncremental();
        m_BoostedTree->predict(true /*new data only*/);
        break;
    case E_Predict:
        m_BoostedTree = m_BoostedTreeFactory->buildForPredict(frame, dependentVariableColumn);
        // prediction occurs in buildForPredict
        // m_BoostedTree->predict(true /*new data only*/);
        break;
    }

    core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) = watch.stop();
}

CDataFrameTrainBoostedTreeRunner::TBoostedTreeFactoryUPtr
CDataFrameTrainBoostedTreeRunner::boostedTreeFactory(TLossFunctionUPtr loss,
                                                     TDataFrameUPtrTemporaryDirectoryPtrPr* frameAndDirectory) const {
    switch (m_Task) {
    case E_Train:
        break;
    case E_Update:
    case E_Predict:
        if (frameAndDirectory != nullptr) {
            // This will be null if we're just computing memory usage.
            auto restoreSearcher = this->spec().restoreSearcher();
            if (restoreSearcher == nullptr) {
                HANDLE_FATAL(<< "Input error: can't predict or incrementally training without supplying a model.");
                break;
            }
            *frameAndDirectory = this->makeDataFrame();
            auto dataSummarizationRestorer = [](CRetrainableModelJsonReader::TIStreamSPtr inputStream,
                                                core::CDataFrame& frame) {
                return CRetrainableModelJsonReader::dataSummarizationFromCompressedJsonStream(
                    std::move(inputStream), frame);
            };
            auto bestForestRestorer =
                [](CRetrainableModelJsonReader::TIStreamSPtr inputStream,
                   const CRetrainableModelJsonReader::TStrSizeUMap& encodingsIndices) {
                    return CRetrainableModelJsonReader::bestForestFromCompressedJsonStream(
                        std::move(inputStream), encodingsIndices);
                };
            auto& frame = frameAndDirectory->first;
            auto result = std::make_unique<maths::analytics::CBoostedTreeFactory>(
                maths::analytics::CBoostedTreeFactory::constructFromDefinition(
                    this->spec().numberThreads(), std::move(loss), *restoreSearcher,
                    *frame, dataSummarizationRestorer, bestForestRestorer));
            result->newTrainingRowMask(core::CPackedBitVector{frame->numberRows(), false});
            return result;
        }
        break;
    }

    return std::make_unique<maths::analytics::CBoostedTreeFactory>(
        maths::analytics::CBoostedTreeFactory::constructFromParameters(
            this->spec().numberThreads(), std::move(loss)));
}

CDataFrameTrainBoostedTreeRunner::TBoostedTreeUPtr
CDataFrameTrainBoostedTreeRunner::restoreBoostedTree(core::CDataFrame& frame,
                                                     std::size_t dependentVariableColumn,
                                                     const TDataSearcherUPtr& restoreSearcher) {
    if (restoreSearcher == nullptr) {
        return nullptr;
    }

    // Restore from compressed JSON.
    try {
        core::CStateDecompressor decompressor{*restoreSearcher};
        core::CDataSearcher::TIStreamP inputStream{decompressor.search(1, 1)}; // search arguments are ignored
        if (inputStream == nullptr) {
            LOG_ERROR(<< "Unable to connect to data store");
            return nullptr;
        }

        if (inputStream->bad()) {
            LOG_ERROR(<< "State restoration search returned bad stream");
            return nullptr;
        }

        if (inputStream->fail()) {
            // This is fatal. If the stream exists and has failed then state is missing
            LOG_ERROR(<< "State restoration search returned failed stream");
            return nullptr;
        }
        return maths::analytics::CBoostedTreeFactory::constructFromString(*inputStream)
            .analysisInstrumentation(m_Instrumentation)
            .trainingStateCallback(this->statePersister())
            .restoreFor(frame, dependentVariableColumn);
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
    }
    return nullptr;
}

std::size_t CDataFrameTrainBoostedTreeRunner::estimateBookkeepingMemoryUsage(
    std::size_t /*numberPartitions*/,
    std::size_t totalNumberRows,
    std::size_t /*partitionNumberRows*/,
    std::size_t numberColumns) const {
    // TODO https://github.com/elastic/ml-cpp/issues/1790.
    return m_BoostedTreeFactory->estimateMemoryUsageTrain(
        static_cast<std::size_t>(static_cast<double>(totalNumberRows) * m_TrainingPercent + 0.5),
        numberColumns);
}

const CDataFrameAnalysisInstrumentation&
CDataFrameTrainBoostedTreeRunner::instrumentation() const {
    return m_Instrumentation;
}

CDataFrameAnalysisInstrumentation& CDataFrameTrainBoostedTreeRunner::instrumentation() {
    return m_Instrumentation;
}

CDataFrameAnalysisRunner::TDataSummarizationJsonWriterUPtr
CDataFrameTrainBoostedTreeRunner::dataSummarization() const {
    auto rowMask = this->boostedTree().dataSummarization();
    if (rowMask.manhattan() <= 0.0) {
        return {};
    }
    return std::make_unique<CDataSummarizationJsonWriter>(
        this->boostedTree().trainingData(), std::move(rowMask),
        this->spec().numberColumns(), this->boostedTree().categoryEncoder());
}

// clang-format off
const std::string CDataFrameTrainBoostedTreeRunner::RANDOM_NUMBER_GENERATOR_SEED{"seed"};
const std::string CDataFrameTrainBoostedTreeRunner::DEPENDENT_VARIABLE_NAME{"dependent_variable"};
const std::string CDataFrameTrainBoostedTreeRunner::PREDICTION_FIELD_NAME{"prediction_field_name"};
const std::string CDataFrameTrainBoostedTreeRunner::TRAINING_PERCENT_FIELD_NAME{"training_percent"};
const std::string CDataFrameTrainBoostedTreeRunner::DOWNSAMPLE_ROWS_PER_FEATURE{"downsample_rows_per_feature"};
const std::string CDataFrameTrainBoostedTreeRunner::DOWNSAMPLE_FACTOR{"downsample_factor"};
const std::string CDataFrameTrainBoostedTreeRunner::ALPHA{"alpha"};
const std::string CDataFrameTrainBoostedTreeRunner::LAMBDA{"lambda"};
const std::string CDataFrameTrainBoostedTreeRunner::GAMMA{"gamma"};
const std::string CDataFrameTrainBoostedTreeRunner::ETA{"eta"};
const std::string CDataFrameTrainBoostedTreeRunner::ETA_GROWTH_RATE_PER_TREE{"eta_growth_rate_per_tree"};
const std::string CDataFrameTrainBoostedTreeRunner::RETRAINED_TREE_ETA{"retrained_tree_eta"};
const std::string CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_LIMIT{"soft_tree_depth_limit"};
const std::string CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_TOLERANCE{"soft_tree_depth_tolerance"};
const std::string CDataFrameTrainBoostedTreeRunner::MAX_TREES{"max_trees"};
const std::string CDataFrameTrainBoostedTreeRunner::FEATURE_BAG_FRACTION{"feature_bag_fraction"};
const std::string CDataFrameTrainBoostedTreeRunner::PREDICTION_CHANGE_COST{"prediction_change_cost"};
const std::string CDataFrameTrainBoostedTreeRunner::TREE_TOPOLOGY_CHANGE_PENALTY{"tree_topology_change_penalty"};
const std::string CDataFrameTrainBoostedTreeRunner::NUM_FOLDS{"num_folds"};
const std::string CDataFrameTrainBoostedTreeRunner::TRAIN_FRACTION_PER_FOLD{"train_fraction_per_fold"};
const std::string CDataFrameTrainBoostedTreeRunner::STOP_CROSS_VALIDATION_EARLY{"stop_cross_validation_early"};
const std::string CDataFrameTrainBoostedTreeRunner::MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER{"max_optimization_rounds_per_hyperparameter"};
const std::string CDataFrameTrainBoostedTreeRunner::BAYESIAN_OPTIMISATION_RESTARTS{"bayesian_optimisation_restarts"};
const std::string CDataFrameTrainBoostedTreeRunner::NUM_TOP_FEATURE_IMPORTANCE_VALUES{"num_top_feature_importance_values"};
const std::string CDataFrameTrainBoostedTreeRunner::IS_TRAINING_FIELD_NAME{"is_training"};
const std::string CDataFrameTrainBoostedTreeRunner::FEATURE_NAME_FIELD_NAME{"feature_name"};
const std::string CDataFrameTrainBoostedTreeRunner::IMPORTANCE_FIELD_NAME{"importance"};
const std::string CDataFrameTrainBoostedTreeRunner::FEATURE_IMPORTANCE_FIELD_NAME{"feature_importance"};
const std::string CDataFrameTrainBoostedTreeRunner::FEATURE_PROCESSORS{"feature_processors"};
const std::string CDataFrameTrainBoostedTreeRunner::EARLY_STOPPING_ENABLED{"early_stopping_enabled"};
const std::string CDataFrameTrainBoostedTreeRunner::FORCE_ACCEPT_INCREMENTAL_TRAINING{"force_accept_incremental_training"};
const std::string CDataFrameTrainBoostedTreeRunner::DATA_SUMMARIZATION_FRACTION{"data_summarization_fraction"};
const std::string CDataFrameTrainBoostedTreeRunner::TASK{"task"};
const std::string CDataFrameTrainBoostedTreeRunner::TASK_TRAIN{"train"};
const std::string CDataFrameTrainBoostedTreeRunner::TASK_UPDATE{"update"};
const std::string CDataFrameTrainBoostedTreeRunner::TASK_PREDICT{"predict"};
const std::string CDataFrameTrainBoostedTreeRunner::PREVIOUS_TRAIN_LOSS_GAP{"previous_train_loss_gap"};
const std::string CDataFrameTrainBoostedTreeRunner::PREVIOUS_TRAIN_NUM_ROWS{"previous_train_num_rows"};
// clang-format on
}
}
