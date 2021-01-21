/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataFrameAnalysisInstrumentation.h>

#include <core/CTimeUtils.h>
#include <core/Constants.h>

#include <maths/CBoostedTree.h>

#include <api/CDataFrameOutliersRunner.h>
#include <api/CDataFrameTrainBoostedTreeClassifierRunner.h>
#include <api/CDataFrameTrainBoostedTreeRunner.h>

#include <rapidjson/document.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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
const std::string ETA_GROWTH_RATE_PER_TREE_TAG{"eta_growth_rate_per_tree"};
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
                         << bytesToString(memoryReestimateBytes) << "] and restart.")
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
    this->writeProgress(lastTask, 100, m_Writer.get());
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
                                                core::CRapidJsonConcurrentLineWriter& writer) {

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
        m_Writer->StartObject();
        this->writeMemory(timestamp);
        this->writeAnalysisStats(timestamp);
        m_Writer->EndObject();
    }
}

void CDataFrameAnalysisInstrumentation::writeMemory(std::int64_t timestamp) {
    if (m_Writer != nullptr) {
        m_Writer->Key(MEMORY_TYPE_TAG);
        m_Writer->StartObject();
        m_Writer->Key(JOB_ID_TAG);
        m_Writer->String(m_JobId);
        m_Writer->Key(TIMESTAMP_TAG);
        m_Writer->Int64(timestamp);
        m_Writer->Key(PEAK_MEMORY_USAGE_TAG);
        m_Writer->Uint64(core::CProgramCounters::counter(this->memoryCounterType()));
        m_Writer->Key(MEMORY_STATUS_TAG);
        switch (m_MemoryStatus) {
        case E_Ok:
            m_Writer->String(MEMORY_STATUS_OK_TAG);
            break;
        case E_HardLimit:
            m_Writer->String(MEMORY_STATUS_HARD_LIMIT_TAG);
            break;
        }
        if (m_MemoryReestimate) {
            m_Writer->Key(MEMORY_REESTIMATE_TAG);
            m_Writer->Int64(m_MemoryReestimate.get());
        }
        m_Writer->EndObject();
    }
}

void CDataFrameAnalysisInstrumentation::writeProgress(const std::string& task,
                                                      int progress,
                                                      core::CRapidJsonConcurrentLineWriter* writer) {
    if (writer != nullptr && task != NO_TASK) {
        writer->StartObject();
        writer->Key(PHASE_PROGRESS);
        writer->StartObject();
        writer->Key(PHASE);
        writer->String(task);
        writer->Key(PROGRESS_PERCENT);
        writer->Int(progress);
        writer->EndObject();
        writer->EndObject();
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
    auto writer = this->writer();
    if (writer != nullptr && m_AnalysisStatsInitialized == true) {
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
            CDataFrameOutliersRunner::N_NEIGHBORS,
            rapidjson::Value(static_cast<std::uint64_t>(this->m_Parameters.s_NumberNeighbours))
                .Move(),
            parentObject);
        writer->addMember(
            CDataFrameOutliersRunner::COMPUTE_FEATURE_INFLUENCE,
            rapidjson::Value(this->m_Parameters.s_ComputeFeatureInfluence).Move(),
            parentObject);
        writer->addMember(CDataFrameOutliersRunner::OUTLIER_FRACTION,
                          rapidjson::Value(this->m_Parameters.s_OutlierFraction).Move(),
                          parentObject);
        writer->addMember(CDataFrameOutliersRunner::FEATURE_INFLUENCE_THRESHOLD,
                          rapidjson::Value(this->m_FeatureInfluenceThreshold).Move(),
                          parentObject);
        writer->addMember(
            CDataFrameOutliersRunner::STANDARDIZATION_ENABLED,
            rapidjson::Value(this->m_Parameters.s_StandardizeColumns).Move(), parentObject);
        writer->addMember(CDataFrameOutliersRunner::METHOD,
                          maths::COutliers::print(this->m_Parameters.s_Method), parentObject);
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
    m_LossValues.emplace_back(std::move(fold), std::move(lossValues));
}

void CDataFrameTrainBoostedTreeInstrumentation::writeAnalysisStats(std::int64_t timestamp) {
    auto* writer = this->writer();
    if (writer != nullptr && m_AnalysisStatsInitialized == true) {
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

        writer->addMember(CDataFrameTrainBoostedTreeRunner::ETA,
                          rapidjson::Value(this->m_Hyperparameters.s_Eta).Move(),
                          parentObject);
        if (m_Type == E_Classification) {
            auto objective = this->m_Hyperparameters.s_ClassAssignmentObjective;
            writer->addMember(
                CDataFrameTrainBoostedTreeClassifierRunner::CLASS_ASSIGNMENT_OBJECTIVE,
                CDataFrameTrainBoostedTreeClassifierRunner::CLASS_ASSIGNMENT_OBJECTIVE_VALUES[objective],
                parentObject);
        }
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::ALPHA,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_DepthPenaltyMultiplier)
                .Move(),
            parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_LIMIT,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_SoftTreeDepthLimit)
                .Move(),
            parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_TOLERANCE,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_SoftTreeDepthTolerance)
                .Move(),
            parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::GAMMA,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_TreeSizePenaltyMultiplier)
                .Move(),
            parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::LAMBDA,
            rapidjson::Value(this->m_Hyperparameters.s_Regularization.s_LeafWeightPenaltyMultiplier)
                .Move(),
            parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::DOWNSAMPLE_FACTOR,
            rapidjson::Value(this->m_Hyperparameters.s_DownsampleFactor).Move(),
            parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::NUM_FOLDS,
            rapidjson::Value(static_cast<std::uint64_t>(this->m_Hyperparameters.s_NumFolds))
                .Move(),
            parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::MAX_TREES,
            rapidjson::Value(static_cast<std::uint64_t>(this->m_Hyperparameters.s_MaxTrees))
                .Move(),
            parentObject);
        writer->addMember(
            CDataFrameTrainBoostedTreeRunner::FEATURE_BAG_FRACTION,
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
            CDataFrameTrainBoostedTreeRunner::MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER,
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
