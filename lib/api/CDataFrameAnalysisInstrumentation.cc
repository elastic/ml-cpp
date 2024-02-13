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
#include <api/CDataFrameAnalysisInstrumentation.h>

#include <core/CTimeUtils.h>
#include <core/Constants.h>

#include <maths/analytics/CBoostedTree.h>

#include <api/CDataFrameOutliersRunner.h>
#include <api/CDataFrameTrainBoostedTreeClassifierRunner.h>
#include <api/CDataFrameTrainBoostedTreeRunner.h>

#include <boost/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace json = boost::json;

namespace ml {
namespace api {

namespace {
using TStrVec = std::vector<std::string>;

const double MEMORY_LIMIT_INCREMENT{2.0}; // request 100% more memory
const std::size_t MAXIMUM_FRACTIONAL_PROGRESS{std::size_t{1}
                                              << ((sizeof(std::size_t) - 2) * 8)};
const std::int64_t BYTES_IN_KB{static_cast<std::int64_t>(core::constants::BYTES_IN_KILOBYTES)};

// clang-format off
const std::string CLASSIFICATION_STATS_TAG{"classification_stats"};
const std::string HYPERPARAMETERS_TAG{"hyperparameters"};
const std::string MEMORY_REESTIMATE_TAG{"memory_reestimate_bytes"};
const std::string ITERATION_TAG{"iteration"};
const std::string JOB_ID_TAG{"job_id"};
const std::string MEMORY_STATUS_HARD_LIMIT_TAG{"hard_limit"};
const std::string MEMORY_STATUS_OK_TAG{"ok"};
const std::string MEMORY_STATUS_TAG{"status"};
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
// TODO we should expose these in the analysis config.
const std::string MAX_ATTEMPTS_TO_ADD_TREE_TAG{"max_attempts_to_add_tree"};
const std::string NUM_SPLITS_PER_FEATURE_TAG{"num_splits_per_feature"};

// Phase progress
const std::string PHASE_PROGRESS{"phase_progress"};
const std::string PHASE{"phase"};
const std::string PROGRESS_PERCENT{"progress_percent"};
// clang-format on

std::string bytesToString(std::int64_t value) {
    std::ostringstream stream;
    stream << std::fixed;
    stream << std::setprecision(0);
    value = (value + BYTES_IN_KB - 1) / BYTES_IN_KB;
    if (value < BYTES_IN_KB) {
        stream << value;
        stream << " kb";
    } else {
        value = (value + BYTES_IN_KB - 1) / BYTES_IN_KB;
        stream << value;
        stream << " mb";
    }

    return stream.str();
}

std::string bytesToString(double bytes) {
    return bytesToString(static_cast<std::int64_t>(bytes));
}
}

CDataFrameAnalysisInstrumentation::CDataFrameAnalysisInstrumentation(const std::string& jobId,
                                                                     std::size_t memoryLimit)
    : m_JobId{jobId}, m_ProgressMonitoredTask{NO_TASK},
      m_MemoryLimit{static_cast<std::int64_t>(memoryLimit)}, m_Finished{false},
      m_FractionalProgress{0}, m_Memory{0}, m_Writer{nullptr}, m_MemoryStatus(E_Ok) {
}

void CDataFrameAnalysisInstrumentation::updateMemoryUsage(std::int64_t delta) {
    std::int64_t memory{m_Memory.fetch_add(delta) + delta};
    if (memory >= 0) {
        core::CProgramCounters::counter(this->memoryCounterType()).max(static_cast<std::uint64_t>(memory));
        if (memory > m_MemoryLimit) {
            double memoryReestimateBytes{static_cast<double>(memory) * MEMORY_LIMIT_INCREMENT};
            this->memoryReestimate(static_cast<std::int64_t>(memoryReestimateBytes));
            this->memoryStatus(E_HardLimit);
            this->flush();
            m_Writer->flush();
            LOG_INFO(<< "Required memory " << memory << " exceeds the memory limit "
                     << m_MemoryLimit << ".  New estimated limit is "
                     << memoryReestimateBytes << ".");
            HANDLE_FATAL(<< "Input error: memory limit [" << bytesToString(m_MemoryLimit)
                         << "] has been exceeded. Please force stop the job, increase to new estimated limit ["
                         << bytesToString(memoryReestimateBytes) << "] and restart.");
        }
    } else {
        // Something has gone wrong with memory estimation. Trap this case
        // to avoid underflowing the peak memory usage statistic.
        LOG_WARN(<< "Memory estimate " << memory << " is negative!");
    }
}

void CDataFrameAnalysisInstrumentation::startNewProgressMonitoredTask(const std::string& task) {
    std::string lastTask;
    {
        std::lock_guard<std::mutex> lock{m_ProgressMutex};
        lastTask = m_ProgressMonitoredTask;
        m_ProgressMonitoredTask = task;
        m_FractionalProgress.store(0.0);
    }
    writeProgress(lastTask, 100, m_Writer.get());
}

void CDataFrameAnalysisInstrumentation::updateProgress(double fractionalProgress) {
    m_FractionalProgress.fetch_add(static_cast<std::size_t>(std::max(
        static_cast<double>(MAXIMUM_FRACTIONAL_PROGRESS) * fractionalProgress + 0.5, 1.0)));
}

void CDataFrameAnalysisInstrumentation::resetProgress() {
    std::lock_guard<std::mutex> lock{m_ProgressMutex};
    m_ProgressMonitoredTask = NO_TASK;
    m_FractionalProgress.store(0);
    m_Finished.store(false);
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

void CDataFrameAnalysisInstrumentation::flush(const std::string& /* tag */) {
    // TODO use the tag.
    this->writeMemoryAndAnalysisStats();
}

std::int64_t CDataFrameAnalysisInstrumentation::memory() const {
    return m_Memory.load();
}

const std::string& CDataFrameAnalysisInstrumentation::jobId() const {
    return m_JobId;
}

void CDataFrameAnalysisInstrumentation::monitor(CDataFrameAnalysisInstrumentation& instrumentation,
                                                core::CBoostJsonConcurrentLineWriter& writer) {

    std::string lastTask{NO_TASK};
    int lastProgress{0};

    int wait{1};
    while (instrumentation.finished() == false) {
        std::this_thread::sleep_for(std::chrono::milliseconds(wait));
        std::string task{instrumentation.readProgressMonitoredTask()};
        int progress{instrumentation.percentageProgress()};
        if (task != lastTask || progress > lastProgress) {
            lastTask = task;
            lastProgress = progress;
            writeProgress(lastTask, lastProgress, &writer);
        }
        wait = std::min(2 * wait, 1024);
    }

    lastTask = instrumentation.readProgressMonitoredTask();
    lastProgress = instrumentation.percentageProgress();
    writeProgress(lastTask, lastProgress, &writer);
}

void CDataFrameAnalysisInstrumentation::memoryReestimate(std::int64_t memoryReestimate) {
    m_MemoryReestimate = memoryReestimate;
}

void CDataFrameAnalysisInstrumentation::memoryStatus(EMemoryStatus status) {
    m_MemoryStatus = status;
}

std::string CDataFrameAnalysisInstrumentation::readProgressMonitoredTask() const {
    std::lock_guard<std::mutex> lock{m_ProgressMutex};
    return m_ProgressMonitoredTask;
}

int CDataFrameAnalysisInstrumentation::percentageProgress() const {
    return static_cast<int>(std::floor(100.0 * this->progress()));
}

CDataFrameAnalysisInstrumentation::TWriter* CDataFrameAnalysisInstrumentation::writer() {
    return m_Writer.get();
}

void CDataFrameAnalysisInstrumentation::writeMemoryAndAnalysisStats() {
    if (m_Writer != nullptr) {
        std::int64_t timestamp{core::CTimeUtils::nowMs()};
        m_Writer->onObjectBegin();
        this->writeMemory(timestamp);
        this->writeAnalysisStats(timestamp);
        m_Writer->onObjectEnd();
    }
}

void CDataFrameAnalysisInstrumentation::writeMemory(std::int64_t timestamp) {
    if (m_Writer != nullptr) {
        m_Writer->onKey(MEMORY_TYPE_TAG);
        m_Writer->onObjectBegin();
        m_Writer->onKey(JOB_ID_TAG);
        m_Writer->onString(m_JobId);
        m_Writer->onKey(TIMESTAMP_TAG);
        m_Writer->onInt64(timestamp);
        m_Writer->onKey(PEAK_MEMORY_USAGE_TAG);
        m_Writer->onUint64(core::CProgramCounters::counter(this->memoryCounterType()));
        m_Writer->onKey(MEMORY_STATUS_TAG);
        switch (m_MemoryStatus) {
        case E_Ok:
            m_Writer->onString(MEMORY_STATUS_OK_TAG);
            break;
        case E_HardLimit:
            m_Writer->onString(MEMORY_STATUS_HARD_LIMIT_TAG);
            break;
        }
        if (m_MemoryReestimate) {
            m_Writer->onKey(MEMORY_REESTIMATE_TAG);
            m_Writer->onInt64(*m_MemoryReestimate);
        }
        m_Writer->onObjectEnd();
    }
}

void CDataFrameAnalysisInstrumentation::writeProgress(const std::string& task,
                                                      int progress,
                                                      core::CBoostJsonConcurrentLineWriter* writer) {
    if (writer != nullptr && task != NO_TASK) {
        writer->onObjectBegin();
        writer->onKey(PHASE_PROGRESS);
        writer->onObjectBegin();
        writer->onKey(PHASE);
        writer->onString(task);
        writer->onKey(PROGRESS_PERCENT);
        writer->onInt(progress);
        writer->onObjectEnd();
        writer->onObjectEnd();
        writer->flush();
    }
}

const std::string CDataFrameAnalysisInstrumentation::NO_TASK;

counter_t::ECounterTypes CDataFrameOutliersInstrumentation::memoryCounterType() {
    return counter_t::E_DFOPeakMemoryUsage;
}

counter_t::ECounterTypes CDataFrameTrainBoostedTreeInstrumentation::memoryCounterType() {
    return counter_t::E_DFTPMPeakMemoryUsage;
}

void CDataFrameOutliersInstrumentation::writeAnalysisStats(std::int64_t timestamp) {
    auto* writer = this->writer();
    if (writer != nullptr && m_AnalysisStatsInitialized == true) {
        writer->onKey(OUTLIER_DETECTION_STATS);
        writer->onObjectBegin();
        writer->onKey(JOB_ID_TAG);
        writer->onString(this->jobId());
        writer->onKey(TIMESTAMP_TAG);
        writer->onInt64(timestamp);

        json::object parametersObject{writer->makeObject()};
        this->writeParameters(parametersObject);
        writer->onKey(PARAMETERS_TAG);
        writer->write(parametersObject);

        json::object timingStatsObject{writer->makeObject()};
        this->writeTimingStats(timingStatsObject);
        writer->onKey(TIMING_STATS_TAG);
        writer->write(timingStatsObject);

        writer->onObjectEnd();
    }
}

void CDataFrameOutliersInstrumentation::parameters(
    const maths::analytics::COutliers::SComputeParameters& parameters) {
    if (m_AnalysisStatsInitialized == false) {
        m_AnalysisStatsInitialized = true;
    }
    m_Parameters = parameters;
}

void CDataFrameOutliersInstrumentation::elapsedTime(std::uint64_t time) {
    m_ElapsedTime = time;
}

void CDataFrameOutliersInstrumentation::featureInfluenceThreshold(double featureInfluenceThreshold) {
    m_FeatureInfluenceThreshold = featureInfluenceThreshold;
}

void CDataFrameOutliersInstrumentation::writeTimingStats(json::object& parentObject) {
    auto* writer = this->writer();
    if (writer != nullptr) {
        writer->addMember(TIMING_ELAPSED_TIME_TAG, json::value(m_ElapsedTime), parentObject);
    }
}

void CDataFrameOutliersInstrumentation::writeParameters(json::object& parentObject) {
    auto* writer = this->writer();
    if (writer != nullptr) {
        writer->addMember(CDataFrameOutliersRunner::N_NEIGHBORS,
                          json::value(static_cast<std::uint64_t>(m_Parameters.s_NumberNeighbours)),
                          parentObject);
        writer->addMember(CDataFrameOutliersRunner::COMPUTE_FEATURE_INFLUENCE,
                          json::value(m_Parameters.s_ComputeFeatureInfluence), parentObject);
        writer->addMember(CDataFrameOutliersRunner::OUTLIER_FRACTION,
                          json::value(m_Parameters.s_OutlierFraction), parentObject);
        writer->addMember(CDataFrameOutliersRunner::FEATURE_INFLUENCE_THRESHOLD,
                          json::value(m_FeatureInfluenceThreshold), parentObject);
        writer->addMember(CDataFrameOutliersRunner::STANDARDIZATION_ENABLED,
                          json::value(m_Parameters.s_StandardizeColumns), parentObject);
        writer->addMember(
            CDataFrameOutliersRunner::METHOD,
            json::value(maths::analytics::COutliers::print(m_Parameters.s_Method)),
            parentObject);
    }
}

void CDataFrameTrainBoostedTreeInstrumentation::type(EStatsType type) {
    m_Type = type;
}

void CDataFrameTrainBoostedTreeInstrumentation::iteration(std::size_t iteration) {
    if (m_AnalysisStatsInitialized == false) {
        m_AnalysisStatsInitialized = true;
    }
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
    m_LossValues.emplace_back(fold, std::move(lossValues));
}

void CDataFrameTrainBoostedTreeInstrumentation::task(api_t::EDataFrameTrainBoostedTreeTask task) {
    m_Task = task;
}

void CDataFrameTrainBoostedTreeInstrumentation::writeAnalysisStats(std::int64_t timestamp) {
    auto* writer = this->writer();
    if (writer != nullptr && m_AnalysisStatsInitialized == true) {
        switch (m_Type) {
        case E_Regression:
            writer->onKey(REGRESSION_STATS_TAG);
            break;
        case E_Classification:
            writer->onKey(CLASSIFICATION_STATS_TAG);
            break;
        }
        writer->onObjectBegin();
        writer->onKey(JOB_ID_TAG);
        writer->onString(this->jobId());
        writer->onKey(TIMESTAMP_TAG);
        writer->onInt64(timestamp);
        writer->onKey(ITERATION_TAG);
        writer->onUint64(m_Iteration);

        json::object hyperparametersObject{writer->makeObject()};
        this->writeHyperparameters(hyperparametersObject);
        writer->onKey(HYPERPARAMETERS_TAG);
        writer->write(hyperparametersObject);

        json::object validationLossObject{writer->makeObject()};
        this->writeValidationLoss(validationLossObject);
        writer->onKey(VALIDATION_LOSS_TAG);
        writer->write(validationLossObject);

        json::object timingStatsObject{writer->makeObject()};
        this->writeTimingStats(timingStatsObject);
        writer->onKey(TIMING_STATS_TAG);
        writer->write(timingStatsObject);

        writer->onObjectEnd();
    }
    this->reset();
}

void CDataFrameTrainBoostedTreeInstrumentation::reset() {
    // Clear the map of loss values before the next iteration
    m_LossValues.clear();
}

void CDataFrameTrainBoostedTreeInstrumentation::writeHyperparameters(json::object& parentObject) {
    auto* writer = this->writer();

    if (writer != nullptr) {
        writer->addMember(CDataFrameTrainBoostedTreeRunner::ETA,
                          json::value(m_Hyperparameters.s_Eta), parentObject);

        if (m_Type == E_Classification) {
            auto objective = m_Hyperparameters.s_ClassAssignmentObjective;
            writer->addMember(
                CDataFrameTrainBoostedTreeClassifierRunner::CLASS_ASSIGNMENT_OBJECTIVE,
                CDataFrameTrainBoostedTreeClassifierRunner::CLASS_ASSIGNMENT_OBJECTIVE_VALUES[objective],
                parentObject);
        }
        writer->addMember(CDataFrameTrainBoostedTreeRunner::ALPHA,
                          json::value(m_Hyperparameters.s_DepthPenaltyMultiplier),
                          parentObject);
        writer->addMember(CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_LIMIT,
                          json::value(m_Hyperparameters.s_SoftTreeDepthLimit), parentObject);
        writer->addMember(CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_TOLERANCE,
                          json::value(m_Hyperparameters.s_SoftTreeDepthTolerance),
                          parentObject);
        writer->addMember(CDataFrameTrainBoostedTreeRunner::GAMMA,
                          json::value(m_Hyperparameters.s_TreeSizePenaltyMultiplier),
                          parentObject);
        writer->addMember(CDataFrameTrainBoostedTreeRunner::LAMBDA,
                          json::value(m_Hyperparameters.s_LeafWeightPenaltyMultiplier),
                          parentObject);

        writer->addMember(CDataFrameTrainBoostedTreeRunner::DOWNSAMPLE_FACTOR,
                          json::value(m_Hyperparameters.s_DownsampleFactor), parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::NUM_FOLDS,
            json::value(static_cast<std::uint64_t>(m_Hyperparameters.s_NumFolds)),
            parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::MAX_TREES,
            json::value(static_cast<std::uint64_t>(m_Hyperparameters.s_MaxTrees)),
            parentObject);
        writer->addMember(CDataFrameTrainBoostedTreeRunner::FEATURE_BAG_FRACTION,
                          json::value(m_Hyperparameters.s_FeatureBagFraction), parentObject);
        writer->addMember(CDataFrameTrainBoostedTreeRunner::ETA_GROWTH_RATE_PER_TREE,
                          json::value(m_Hyperparameters.s_EtaGrowthRatePerTree),
                          parentObject);

        writer->addMember(MAX_ATTEMPTS_TO_ADD_TREE_TAG,
                          json::value(static_cast<std::uint64_t>(
                              m_Hyperparameters.s_MaxAttemptsToAddTree)),
                          parentObject);
        writer->addMember(NUM_SPLITS_PER_FEATURE_TAG,
                          json::value(static_cast<std::uint64_t>(
                              m_Hyperparameters.s_NumSplitsPerFeature)),
                          parentObject);
        writer->addMember(CDataFrameTrainBoostedTreeRunner::MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER,
                          json::value(static_cast<std::uint64_t>(
                              m_Hyperparameters.s_MaxOptimizationRoundsPerHyperparameter)),
                          parentObject);
        if (m_Task == api_t::E_Update) {
            writer->addMember(CDataFrameTrainBoostedTreeRunner::TREE_TOPOLOGY_CHANGE_PENALTY,
                              json::value(m_Hyperparameters.s_TreeTopologyChangePenalty),
                              parentObject);
            writer->addMember(CDataFrameTrainBoostedTreeRunner::PREDICTION_CHANGE_COST,
                              json::value(m_Hyperparameters.s_PredictionChangeCost),
                              parentObject);
            writer->addMember(CDataFrameTrainBoostedTreeRunner::RETRAINED_TREE_ETA,
                              json::value(m_Hyperparameters.s_RetrainedTreeEta),
                              parentObject);
        }
    }
}

void CDataFrameTrainBoostedTreeInstrumentation::writeValidationLoss(json::object& parentObject) {
    auto* writer = this->writer();
    if (writer != nullptr) {
        writer->addMember(VALIDATION_LOSS_TYPE_TAG, json::value(m_LossType), parentObject);
        // NOTE: Do not use brace initialization here as that will
        // result in "lossValuesArray" being created as a nested array on linux
        json::array lossValuesArray = writer->makeArray();
        for (auto& element : m_LossValues) {
            json::object item{writer->makeObject()};
            writer->addMember(VALIDATION_FOLD_TAG,
                              json::value(static_cast<std::uint64_t>(element.first)), item);
            // NOTE: Do not use brace initialization here as that will
            // result in "array" being created as a nested array on linux
            json::array array = writer->makeArray(element.second.size());
            for (double lossValue : element.second) {
                array.push_back(json::value(lossValue));
            }
            writer->addMember(VALIDATION_LOSS_VALUES_TAG, array, item);
            lossValuesArray.push_back(item);
        }
        writer->addMember(VALIDATION_FOLD_VALUES_TAG, lossValuesArray, parentObject);
    }
}

void CDataFrameTrainBoostedTreeInstrumentation::writeTimingStats(json::object& parentObject) {
    auto* writer = this->writer();
    if (writer != nullptr) {
        writer->addMember(TIMING_ELAPSED_TIME_TAG, json::value(m_ElapsedTime), parentObject);
        writer->addMember(TIMING_ITERATION_TIME_TAG, json::value(m_IterationTime), parentObject);
    }
}

CDataFrameAnalysisInstrumentation::CScopeSetOutputStream::CScopeSetOutputStream(
    CDataFrameAnalysisInstrumentation& instrumentation,
    core::CJsonOutputStreamWrapper& outStream)
    : m_Instrumentation{instrumentation} {
    instrumentation.m_Writer =
        std::make_unique<core::CBoostJsonConcurrentLineWriter>(outStream);
}

CDataFrameAnalysisInstrumentation::CScopeSetOutputStream::~CScopeSetOutputStream() {
    m_Instrumentation.m_Writer = nullptr;
}
}
}
