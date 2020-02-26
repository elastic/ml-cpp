/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataFrameAnalysisInstrumentation.h>

#include <boost/iostreams/filter/zlib.hpp>
#include <core/CTimeUtils.h>
#include <rapidjson/document.h>

namespace ml {
namespace api {

namespace {
const std::string STEP_TAG{"step"};
const std::string PROGRESS_TAG{"progress"};
const std::string PEAK_MEMORY_USAGE_TAG{"peak_usage_bytes"};
const std::string TYPE_TAG{"type"};
const std::string JOB_ID_TAG{"job_id"};
const std::string TIMESTAMP_TAG{"timestamp"};
const std::string MEMORY_TYPE_TAG{"analytics_memory_usage"};
const std::string ANALYSIS_TYPE_TAG{"analysis_stats"};
const std::string REGRESSION_STATS_TAG{"regression_stats"};
const std::string ITERATION_TAG{"iteration"};
const std::string HYPERPARAMETERS_TAG{"hyperparameters"};
const std::string VALIDATION_LOSS_TAG{"validation_loss"};
const std::string TIMING_STATS_TAG{"timing_stats"};
const std::string VALIDATION_LOSS_TYPE_TAG{"loss_type"};
const std::string VALIDATION_LOSS_VALUES_TAG{"values"};
const std::string VALIDATION_NUM_FOLDS_TAG{"num_folds"};
const std::string TIMING_ELAPSED_TIME_TAG{"elapsed_time"};
const std::string TIMING_ITERATION_TIME_TAG{"iteration_time"};

// Hyperparameters
const std::string ETA_TAG{"eta"};
const std::string CLASS_ASSIGNMENT_OBJECTIVE_TAG{"class_assignment_objective"};
const std::string REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG{"regularization_depth_penalty_multiplier"};
const std::string REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG{"regularization_soft_tree_depth_limit"};
const std::string REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG{
    "regularization_soft_tree_depth_tolerance"};
const std::string REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG{
    "regularization_tree_size_penalty_multiplier"};
const std::string REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG{
    "regularization_leaf_weight_penalty_multiplier"};
const std::string DOWNSAMPLE_FACTOR_TAG{"downsample_factor"};
const std::string NUM_FOLDS_TAG{"num_folds"};
const std::string MAX_TREES_TAG{"max_trees"};
const std::string FEATURE_BAG_FRACTION_TAG{"feature_bag_fraction"};
const std::string ETA_GROWTH_RATE_PER_TREE_TAG{"eta_growth_rate_per_tree"};
const std::string MAX_ATTEMPTS_TO_ADD_TREE_TAG{"max_attempts_to_add_tree"};
const std::string NUM_SPLITS_PER_FEATURE_TAG{"num_splits_per_feature"};
const std::string MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER_TAG{
    "max_optimization_rounds_per_hyperparameter"};

const std::size_t MAXIMUM_FRACTIONAL_PROGRESS{std::size_t{1}
                                              << ((sizeof(std::size_t) - 2) * 8)};
}

void CDataFrameAnalysisInstrumentation::updateMemoryUsage(std::int64_t delta) {
    std::int64_t memory{m_Memory.fetch_add(delta)};
    if (memory >= 0) {
        core::CProgramCounters::counter(this->memoryCounterType()).max(memory);
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
    : m_Finished{false}, m_FractionalProgress{0}, m_Memory{0}, m_Writer{nullptr}, m_JobId{jobId} {
}

void CDataFrameAnalysisInstrumentation::resetProgress() {
    m_FractionalProgress.store(0.0);
    m_Finished.store(false);
}

void CDataFrameAnalysisInstrumentation::writer(core::CRapidJsonConcurrentLineWriter* writer) {
    m_Writer = writer;
}

void CDataFrameAnalysisInstrumentation::nextStep(std::uint32_t step) {
    this->writeState(step);
}

void CDataFrameAnalysisInstrumentation::writeState(std::uint32_t step) {
    std::int64_t timestamp{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    this->writeMemory(timestamp);
    this->writeAnalysisStats(timestamp, step);
}

std::int64_t CDataFrameAnalysisInstrumentation::memory() const {
    return m_Memory.load();
}

void CDataFrameAnalysisInstrumentation::writeMemory(std::int64_t timestamp) {
    if (m_Writer != nullptr) {
        m_Writer->StartObject();
        m_Writer->Key(TYPE_TAG);
        m_Writer->String(MEMORY_TYPE_TAG);
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

core::CRapidJsonConcurrentLineWriter* CDataFrameAnalysisInstrumentation::writer() {
    return m_Writer;
}

counter_t::ECounterTypes CDataFrameOutliersInstrumentation::memoryCounterType() {
    return counter_t::E_DFOPeakMemoryUsage;
}

counter_t::ECounterTypes CDataFrameTrainBoostedTreeInstrumentation::memoryCounterType() {
    return counter_t::E_DFTPMPeakMemoryUsage;
}

void CDataFrameOutliersInstrumentation::writeAnalysisStats(std::int64_t timestamp,
                                                           std::uint32_t /*step*/) {
    auto* writer{this->writer()};
    if (writer != nullptr) {
        writer->StartObject();
        writer->Key(JOB_ID_TAG);
        writer->String(this->jobId());
        writer->Key(TIMESTAMP_TAG);
        writer->Int64(timestamp);
        writer->EndObject();
    }
}

void CDataFrameTrainBoostedTreeInstrumentation::writeAnalysisStats(std::int64_t timestamp,
                                                                   std::uint32_t step) {
    auto* writer{this->writer()};
    if (writer != nullptr) {
        writer->StartObject();
        writer->Key(JOB_ID_TAG);
        writer->String(this->jobId());
        writer->Key(TIMESTAMP_TAG);
        writer->Int64(timestamp);
        rapidjson::Value hyperparametersObject{writer->makeObject()};
        this->writeHyperparameters(hyperparametersObject);
        writer->Key(HYPERPARAMETERS_TAG);
        writer->write(hyperparametersObject);
        writer->EndObject();
    }
}

void CDataFrameTrainBoostedTreeInstrumentation::writeHyperparameters(rapidjson::Value& parentObject) {
    if (this->writer() != nullptr) {
        
    this->writer()->addMember(
        ETA_TAG, rapidjson::Value(this->m_Hyperparameters.s_Eta).Move(), parentObject);
    // TODO convert from ENUM to String
    this->writer()->addMember(
        CLASS_ASSIGNMENT_OBJECTIVE_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_ClassAssignmentObjective).Move(),
        parentObject);
    this->writer()->addMember(
        REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_DepthPenaltyMultiplier)
            .Move(),
        parentObject);
    this->writer()->addMember(
        REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_SoftTreeDepthLimit)
            .Move(),
        parentObject);
    this->writer()->addMember(
        REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_SoftTreeDepthTolerance)
            .Move(),
        parentObject);
    this->writer()->addMember(
        REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_TreeSizePenaltyMultiplier)
            .Move(),
        parentObject);
    this->writer()->addMember(
        REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_LeafWeightPenaltyMultiplier)
            .Move(),
        parentObject);
    this->writer()->addMember(
        DOWNSAMPLE_FACTOR_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_DownsampleFactor).Move(), parentObject);
    this->writer()->addMember(
        NUM_FOLDS_TAG, rapidjson::Value(this->m_Hyperparameters.s_NumFolds).Move(), parentObject);
    this->writer()->addMember(
        MAX_TREES_TAG, rapidjson::Value(this->m_Hyperparameters.s_MaxTrees).Move(), parentObject);
    this->writer()->addMember(
        FEATURE_BAG_FRACTION_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_FeatureBagFraction).Move(), parentObject);
    this->writer()->addMember(
        ETA_GROWTH_RATE_PER_TREE_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_EtaGrowthRatePerTree).Move(),
        parentObject);
    this->writer()->addMember(
        MAX_ATTEMPTS_TO_ADD_TREE_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_MaxAttemptsToAddTree).Move(),
        parentObject);
    this->writer()->addMember(
        NUM_SPLITS_PER_FEATURE_TAG,
        rapidjson::Value(this->m_Hyperparameters.s_NumSplitsPerFeature).Move(), parentObject);
    this->writer()->addMember(MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                              rapidjson::Value(this->m_Hyperparameters.s_MaxOptimizationRoundsPerHyperparameter)
                                  .Move(),
                              parentObject);
    }
}
void CDataFrameTrainBoostedTreeInstrumentation::writeValidationLoss(rapidjson::Value& /* parentObject */) {
}
void CDataFrameTrainBoostedTreeInstrumentation::writeTimingStats(rapidjson::Value& /* parentObject */) {
}
}
}
