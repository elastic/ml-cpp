/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataFrameAnalysisInstrumentation.h>

#include <core/CTimeUtils.h>

#include <rapidjson/document.h>

#include <cstdint>
#include <string>

namespace ml {
namespace api {

namespace {

// clang-format off
const std::string CLASSIFICATION_STATS_TAG{"classification_stats"};
const std::string HYPERPARAMETERS_TAG{"hyperparameters"};
const std::string ITERATION_TAG{"iteration"};
const std::string JOB_ID_TAG{"job_id"};
const std::string MEMORY_TYPE_TAG{"analytics_memory_usage"};
const std::string OUTLIER_DETECTION_STATS{"outlier_detection_stats"};
const std::string PARAMETERS_TAG{"parameters"};
const std::string PEAK_MEMORY_USAGE_TAG{"peak_usage_bytes"};
const std::string PROGRESS_TAG{"progress"};
const std::string REGRESSION_STATS_TAG{"regression_stats"};
const std::string STEP_TAG{"step"};
const std::string TIMESTAMP_TAG{"timestamp"};
const std::string TIMING_ELAPSED_TIME_TAG{"elapsed_time"};
const std::string TIMING_ITERATION_TIME_TAG{"iteration_time"};
const std::string TIMING_STATS_TAG{"timing_stats"};
const std::string TYPE_TAG{"type"};
const std::string VALIDATION_FOLD_TAG{"fold"};
const std::string VALIDATION_FOLD_VALUES_TAG{"fold_values"};
const std::string VALIDATION_LOSS_TAG{"validation_loss"};
const std::string VALIDATION_LOSS_TYPE_TAG{"loss_type"};
const std::string VALIDATION_LOSS_VALUES_TAG{"values"};

// Hyperparameters
const std::string CLASS_ASSIGNMENT_OBJECTIVE_TAG{"class_assignment_objective"};
const std::string CLASS_ASSIGNMENT_OBJECTIVE[]{"accuracy", "minimum_recall"};
const std::string DOWNSAMPLE_FACTOR_TAG{"downsample_factor"};
const std::string ETA_GROWTH_RATE_PER_TREE_TAG{"eta_growth_rate_per_tree"};
const std::string ETA_TAG{"eta"};
const std::string FEATURE_BAG_FRACTION_TAG{"feature_bag_fraction"};
const std::string MAX_ATTEMPTS_TO_ADD_TREE_TAG{"max_attempts_to_add_tree"};
const std::string MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER_TAG{"max_optimization_rounds_per_hyperparameter"};
const std::string MAX_TREES_TAG{"max_trees"};
const std::string NUM_FOLDS_TAG{"num_folds"};
const std::string NUM_SPLITS_PER_FEATURE_TAG{"num_splits_per_feature"};
const std::string REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG{"regularization_depth_penalty_multiplier"};
const std::string REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG{"regularization_leaf_weight_penalty_multiplier"};
const std::string REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG{"regularization_soft_tree_depth_limit"};
const std::string REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG{"regularization_soft_tree_depth_tolerance"};
const std::string REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG{"regularization_tree_size_penalty_multiplier"};

// Outlier detection parameters
const std::string N_NEIGHBORS{"n_neighbors"};
const std::string METHOD{"method"};
const std::string COMPUTE_FEATURE_INFLUENCE{"compute_feature_influence"};
const std::string FEATURE_INFLUENCE_THRESHOLD{"feature_influence_threshold"};
const std::string OUTLIER_FRACTION{"outlier_fraction"};
const std::string STANDARDIZATION_ENABLED{"standardization_enabled"};

// clang-format on

const std::size_t MAXIMUM_FRACTIONAL_PROGRESS{std::size_t{1}
                                              << ((sizeof(std::size_t) - 2) * 8)};
}

void CDataFrameAnalysisInstrumentation::updateMemoryUsage(std::int64_t delta) {
    std::int64_t memory{m_Memory.fetch_add(delta)};
    if (memory >= 0) {
        core::CProgramCounters::counter(this->memoryCounterType()).max(static_cast<std::uint64_t>(memory));
    } else {
        // Something has gone wrong with memory estimation. Trap this case
        // to avoid underflowing the peak memory usage statistic.
        LOG_WARN(<< "Memory estimate " << memory << " is negative!");
    }
}

void CDataFrameAnalysisInstrumentation::updateProgress(double fractionalProgress) {
    m_FractionalProgress.fetch_add(static_cast<std::size_t>(std::max(
        static_cast<double>(MAXIMUM_FRACTIONAL_PROGRESS) * fractionalProgress + 0.5, 1.0)));
}

void CDataFrameAnalysisInstrumentation::setToFinished() {
    m_Finished.store(true);
    m_FractionalProgress.store(MAXIMUM_FRACTIONAL_PROGRESS);
}

bool CDataFrameAnalysisInstrumentation::finished() const {
    return m_Finished.load();
}

double CDataFrameAnalysisInstrumentation::progress() const {
    return this->finished()
               ? 1.0
               : static_cast<double>(std::min(m_FractionalProgress.load(),
                                              MAXIMUM_FRACTIONAL_PROGRESS - 1)) /
                     static_cast<double>(MAXIMUM_FRACTIONAL_PROGRESS);
}

CDataFrameAnalysisInstrumentation::CDataFrameAnalysisInstrumentation(const std::string& jobId)
    : m_JobId{jobId}, m_Finished{false}, m_FractionalProgress{0}, m_Memory{0}, m_Writer{nullptr} {
}

void CDataFrameAnalysisInstrumentation::resetProgress() {
    m_FractionalProgress.store(0.0);
    m_Finished.store(false);
}

void CDataFrameAnalysisInstrumentation::nextStep(const std::string& /* phase */) {
    // reactivate once java side is ready
    this->writeState();
}

void CDataFrameAnalysisInstrumentation::writeState() {
    std::int64_t timestamp{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    if (m_Writer != nullptr) {
        m_Writer->StartObject();
        m_Writer->Key(MEMORY_TYPE_TAG);
        this->writeMemory(timestamp);
        this->writeAnalysisStats(timestamp);
        m_Writer->EndObject();
    }
}

std::int64_t CDataFrameAnalysisInstrumentation::memory() const {
    return m_Memory.load();
}

void CDataFrameAnalysisInstrumentation::writeMemory(std::int64_t timestamp) {
    if (m_Writer != nullptr) {
        m_Writer->StartObject();
        m_Writer->Key(JOB_ID_TAG);
        m_Writer->String(m_JobId);
        m_Writer->Key(TIMESTAMP_TAG);
        m_Writer->Int64(timestamp);
        m_Writer->Key(PEAK_MEMORY_USAGE_TAG);
        m_Writer->Int64(m_Memory.load());
        m_Writer->EndObject();
    }
}

const std::string& CDataFrameAnalysisInstrumentation::jobId() const {
    return m_JobId;
}

CDataFrameAnalysisInstrumentation::TWriter* CDataFrameAnalysisInstrumentation::writer() {
    return m_Writer.get();
}

counter_t::ECounterTypes CDataFrameOutliersInstrumentation::memoryCounterType() {
    return counter_t::E_DFOPeakMemoryUsage;
}

counter_t::ECounterTypes CDataFrameTrainBoostedTreeInstrumentation::memoryCounterType() {
    return counter_t::E_DFTPMPeakMemoryUsage;
}

void CDataFrameOutliersInstrumentation::writeAnalysisStats(std::int64_t timestamp) {
    auto writer = this->writer();
    if (writer != nullptr) {
        writer->Key(OUTLIER_DETECTION_STATS);
        writer->StartObject();
        writer->Key(JOB_ID_TAG);
        writer->String(this->jobId());
        writer->Key(TIMESTAMP_TAG);
        writer->Int64(timestamp);

        rapidjson::Value parametersObject{writer->makeObject()};
        this->writeParameters(parametersObject);
        writer->Key(PARAMETERS_TAG);
        writer->write(parametersObject);

        rapidjson::Value timingStatsObject{writer->makeObject()};
        this->writeTimingStats(timingStatsObject);
        writer->Key(TIMING_STATS_TAG);
        writer->write(timingStatsObject);

        writer->EndObject();
    }
}

void CDataFrameOutliersInstrumentation::parameters(const maths::COutliers::SComputeParameters& parameters) {
    m_Parameters = parameters;
}

void CDataFrameOutliersInstrumentation::elapsedTime(std::uint64_t time) {
    m_ElapsedTime = time;
}

void CDataFrameOutliersInstrumentation::featureInfluenceThreshold(double featureInfluenceThreshold) {
    m_FeatureInfluenceThreshold = featureInfluenceThreshold;
}

void CDataFrameOutliersInstrumentation::writeTimingStats(rapidjson::Value& parentObject) {
    auto* writer = this->writer();
    if (writer != nullptr) {
        writer->addMember(TIMING_ELAPSED_TIME_TAG,
                          rapidjson::Value(m_ElapsedTime).Move(), parentObject);
    }
}

void CDataFrameOutliersInstrumentation::writeParameters(rapidjson::Value& parentObject) {
    auto* writer = this->writer();

    if (writer != nullptr) {

        writer->addMember(
            N_NEIGHBORS,
            rapidjson::Value(static_cast<std::uint64_t>(this->m_Parameters.s_NumberNeighbours))
                .Move(),
            parentObject);
        writer->addMember(
            COMPUTE_FEATURE_INFLUENCE,
            rapidjson::Value(this->m_Parameters.s_ComputeFeatureInfluence).Move(),
            parentObject);
        writer->addMember(OUTLIER_FRACTION,
                          rapidjson::Value(this->m_Parameters.s_OutlierFraction).Move(),
                          parentObject);
        writer->addMember(FEATURE_INFLUENCE_THRESHOLD,
                          rapidjson::Value(this->m_FeatureInfluenceThreshold).Move(),
                          parentObject);
        writer->addMember(
            STANDARDIZATION_ENABLED,
            rapidjson::Value(this->m_Parameters.s_StandardizeColumns).Move(), parentObject);
        writer->addMember(METHOD, maths::COutliers::print(this->m_Parameters.s_Method),
                          parentObject);
    }
}

void CDataFrameTrainBoostedTreeInstrumentation::type(EStatsType type) {
    m_Type = type;
}

void CDataFrameTrainBoostedTreeInstrumentation::iteration(std::size_t iteration) {
    m_Iteration = iteration;
}

void CDataFrameTrainBoostedTreeInstrumentation::iterationTime(std::uint64_t delta) {
    m_IterationTime = delta;
    m_ElapsedTime += delta;
}

void CDataFrameTrainBoostedTreeInstrumentation::lossType(const std::string& lossType) {
    m_LossType = lossType;
}

void CDataFrameTrainBoostedTreeInstrumentation::lossValues(std::size_t fold,
                                                           TDoubleVec&& lossValues) {
    m_LossValues.emplace_back(std::move(fold), std::move(lossValues));
}

void CDataFrameTrainBoostedTreeInstrumentation::writeAnalysisStats(std::int64_t timestamp) {
    auto* writer = this->writer();
    if (writer != nullptr) {
        switch (m_Type) {
        case E_Regression:
            writer->Key(REGRESSION_STATS_TAG);
            break;
        case E_Classification:
            writer->Key(CLASSIFICATION_STATS_TAG);
            break;
        }
        writer->StartObject();
        writer->Key(JOB_ID_TAG);
        writer->String(this->jobId());
        writer->Key(TIMESTAMP_TAG);
        writer->Int64(timestamp);
        writer->Key(ITERATION_TAG);
        writer->Uint64(m_Iteration);

        rapidjson::Value hyperparametersObject{writer->makeObject()};
        this->writeHyperparameters(hyperparametersObject);
        writer->Key(HYPERPARAMETERS_TAG);
        writer->write(hyperparametersObject);

        rapidjson::Value validationLossObject{writer->makeObject()};
        this->writeValidationLoss(validationLossObject);
        writer->Key(VALIDATION_LOSS_TAG);
        writer->write(validationLossObject);

        rapidjson::Value timingStatsObject{writer->makeObject()};
        this->writeTimingStats(timingStatsObject);
        writer->Key(TIMING_STATS_TAG);
        writer->write(timingStatsObject);

        writer->EndObject();
    }
    this->reset();
}

void CDataFrameTrainBoostedTreeInstrumentation::reset() {
    // Clear the map of loss values before the next iteration
    m_LossValues.clear();
}

void CDataFrameTrainBoostedTreeInstrumentation::writeHyperparameters(rapidjson::Value& parentObject) {
    auto* writer = this->writer();

    if (writer != nullptr) {

        writer->addMember(ETA_TAG,
                          rapidjson::Value(this->m_Hyperparameters.s_Eta).Move(),
                          parentObject);
        if (m_Type == E_Classification) {
            writer->addMember(CLASS_ASSIGNMENT_OBJECTIVE_TAG,
                              CLASS_ASSIGNMENT_OBJECTIVE[this->m_Hyperparameters.s_ClassAssignmentObjective],
                              parentObject);
        }
        writer->addMember(
            REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_DepthPenaltyMultiplier)
                .Move(),
            parentObject);
        writer->addMember(
            REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_SoftTreeDepthLimit)
                .Move(),
            parentObject);
        writer->addMember(
            REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_SoftTreeDepthTolerance)
                .Move(),
            parentObject);
        writer->addMember(
            REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_TreeSizePenaltyMultiplier)
                .Move(),
            parentObject);
        writer->addMember(
            REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_LeafWeightPenaltyMultiplier)
                .Move(),
            parentObject);
        writer->addMember(
            DOWNSAMPLE_FACTOR_TAG,
            rapidjson::Value(this->m_Hyperparameters.s_DownsampleFactor).Move(),
            parentObject);
        writer->addMember(
            NUM_FOLDS_TAG,
            rapidjson::Value(static_cast<std::uint64_t>(this->m_Hyperparameters.s_NumFolds))
                .Move(),
            parentObject);
        writer->addMember(
            MAX_TREES_TAG,
            rapidjson::Value(static_cast<std::uint64_t>(this->m_Hyperparameters.s_MaxTrees))
                .Move(),
            parentObject);
        writer->addMember(
            FEATURE_BAG_FRACTION_TAG,
            rapidjson::Value(this->m_Hyperparameters.s_FeatureBagFraction).Move(),
            parentObject);
        writer->addMember(
            ETA_GROWTH_RATE_PER_TREE_TAG,
            rapidjson::Value(this->m_Hyperparameters.s_EtaGrowthRatePerTree).Move(),
            parentObject);
        writer->addMember(
            MAX_ATTEMPTS_TO_ADD_TREE_TAG,
            rapidjson::Value(static_cast<std::uint64_t>(this->m_Hyperparameters.s_MaxAttemptsToAddTree))
                .Move(),
            parentObject);
        writer->addMember(
            NUM_SPLITS_PER_FEATURE_TAG,
            rapidjson::Value(static_cast<std::uint64_t>(this->m_Hyperparameters.s_NumSplitsPerFeature))
                .Move(),
            parentObject);
        writer->addMember(
            MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER_TAG,
            rapidjson::Value(static_cast<std::uint64_t>(this->m_Hyperparameters.s_MaxOptimizationRoundsPerHyperparameter))
                .Move(),
            parentObject);
    }
}
void CDataFrameTrainBoostedTreeInstrumentation::writeValidationLoss(rapidjson::Value& parentObject) {
    auto* writer = this->writer();
    if (writer != nullptr) {
        writer->addMember(VALIDATION_LOSS_TYPE_TAG, m_LossType, parentObject);
        rapidjson::Value lossValuesArray{writer->makeArray()};
        for (auto& element : m_LossValues) {
            rapidjson::Value item{writer->makeObject()};
            writer->addMember(
                VALIDATION_FOLD_TAG,
                rapidjson::Value(static_cast<std::uint64_t>(element.first)).Move(), item);
            rapidjson::Value array{writer->makeArray(element.second.size())};
            for (double lossValue : element.second) {
                array.PushBack(rapidjson::Value(lossValue).Move(),
                               writer->getRawAllocator());
            }
            writer->addMember(VALIDATION_LOSS_VALUES_TAG, array, item);
            lossValuesArray.PushBack(item, writer->getRawAllocator());
        }
        writer->addMember(VALIDATION_FOLD_VALUES_TAG, lossValuesArray, parentObject);
    }
}
void CDataFrameTrainBoostedTreeInstrumentation::writeTimingStats(rapidjson::Value& parentObject) {
    auto* writer = this->writer();
    if (writer != nullptr) {
        writer->addMember(TIMING_ELAPSED_TIME_TAG,
                          rapidjson::Value(m_ElapsedTime).Move(), parentObject);
        writer->addMember(TIMING_ITERATION_TIME_TAG,
                          rapidjson::Value(m_IterationTime).Move(), parentObject);
    }
}

CDataFrameAnalysisInstrumentation::CScopeSetOutputStream::CScopeSetOutputStream(
    CDataFrameAnalysisInstrumentation& instrumentation,
    core::CJsonOutputStreamWrapper& outStream)
    : m_Instrumentation{instrumentation} {
    instrumentation.m_Writer =
        std::make_unique<core::CRapidJsonConcurrentLineWriter>(outStream);
}

CDataFrameAnalysisInstrumentation::CScopeSetOutputStream::~CScopeSetOutputStream() {
    m_Instrumentation.m_Writer = nullptr;
}
}
}
