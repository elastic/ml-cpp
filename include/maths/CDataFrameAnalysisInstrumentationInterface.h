/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameAnalysisInstrumentationInterface_h
#define INCLUDED_ml_maths_CDataFrameAnalysisInstrumentationInterface_h

#include <maths/CBoostedTree.h>
#include <maths/ImportExport.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ml {
namespace maths {

//! \brief Interface class for collecting data frame analysis job statistics owned
//! by the maths module.
class MATHS_EXPORT CDataFrameAnalysisInstrumentationInterface {
public:
    using TProgressCallback = std::function<void(double)>;
    using TMemoryUsageCallback = std::function<void(std::int64_t)>;
    using TStepCallback = std::function<void(std::uint32_t)>;

public:
    virtual ~CDataFrameAnalysisInstrumentationInterface() = default;
    //! Adds \p delta to the memory usage statistics.
    virtual void updateMemoryUsage(std::int64_t delta) = 0;
    //! This adds \p fractionalProgess to the current progress.
    //!
    //! \note The caller should try to ensure that the sum of the values added
    //! at the end of the analysis is equal to one.
    //! \note This is converted to an integer - so we can atomically add - by
    //! scaling by 1024. Therefore, this shouldn't be called with values less
    //! than 0.001. In fact, it is unlikely that such high resolution is needed
    //! and typically this would be called significantly less frequently.
    virtual void updateProgress(double fractionalProgress) = 0;
    //! Trigger the next step of the job. This will initiate writing the job state
    //! to the results pipe.
    virtual void nextStep(std::uint32_t step) = 0;
    //! Factory for the updateProgress() callback function object.
    TProgressCallback progressCallback() {
        return [this](double fractionalProgress) {
            this->updateProgress(fractionalProgress);
        };
    }
    //! Factory for the updateMemoryUsage() callback function object.
    TMemoryUsageCallback memoryUsageCallback() {
        return [this](std::int64_t delta) { this->updateMemoryUsage(delta); };
    }
    //! Factory for the nextStep() callback function object.
    TStepCallback stepCallback() {
        return [this](std::uint32_t step) { this->nextStep(step); };
    }
};

class MATHS_EXPORT CDataFrameOutliersInstrumentationInterface
    : virtual public CDataFrameAnalysisInstrumentationInterface {};

class MATHS_EXPORT CDataFrameTrainBoostedTreeInstrumentationInterface
    : virtual public CDataFrameAnalysisInstrumentationInterface {
public:
    enum EStatsType { E_Regression, E_Classification };
    struct SRegularization {
        SRegularization() = default;
        SRegularization(double depthPenaltyMultiplier,
                        double softTreeDepthLimit,
                        double softTreeDepthTolerance,
                        double treeSizePenaltyMultiplier,
                        double leafWeightPenaltyMultiplier)
            : s_DepthPenaltyMultiplier{depthPenaltyMultiplier},
              s_SoftTreeDepthLimit{softTreeDepthLimit}, s_SoftTreeDepthTolerance{softTreeDepthTolerance},
              s_TreeSizePenaltyMultiplier{treeSizePenaltyMultiplier},
              s_LeafWeightPenaltyMultiplier{leafWeightPenaltyMultiplier} {};
        double s_DepthPenaltyMultiplier = 0.0;
        double s_SoftTreeDepthLimit = 0.0;
        double s_SoftTreeDepthTolerance = 0.0;
        double s_TreeSizePenaltyMultiplier = 0.0;
        double s_LeafWeightPenaltyMultiplier = 0.0;
    };
    struct SHyperparameters {
        double s_Eta = 0.1;
        CBoostedTree::EClassAssignmentObjective s_ClassAssignmentObjective =
            CBoostedTree::E_MinimumRecall;
        SRegularization s_Regularization;
        double s_DownsampleFactor = 0.5;
        std::size_t s_NumFolds = 4;
        std::size_t s_MaxTrees = 20;
        double s_FeatureBagFraction = 0.5;
        double s_EtaGrowthRatePerTree = 1.05;
        std::size_t s_MaxAttemptsToAddTree = 3;
        std::size_t s_NumSplitsPerFeature = 75;
        std::size_t s_MaxOptimizationRoundsPerHyperparameter = 2;
    };
    using TDoubleVec = std::vector<double>;

public:
    virtual ~CDataFrameTrainBoostedTreeInstrumentationInterface() = default;
    virtual void type(EStatsType type) = 0;
    virtual void iteration(std::size_t iteration) = 0;
    virtual void startTime(std::uint64_t timestamp) = 0;
    virtual void iterationTime(std::uint64_t delta) = 0;
    virtual void lossType(const std::string& lossType) = 0;
    virtual void lossValues(std::size_t fold, TDoubleVec&& lossValues) = 0;
    virtual void numFolds(std::size_t numFolds) = 0;
    virtual void hyperparameters(const SHyperparameters& hyperparameters) = 0;
    virtual SHyperparameters& hyperparameters() = 0;
};

//! \brief Dummies out all instrumentation.
class MATHS_EXPORT CDataFrameAnalysisInstrumentationStub final
    : public CDataFrameTrainBoostedTreeInstrumentationInterface {
public:
    void updateMemoryUsage(std::int64_t) override {}
    void updateProgress(double) override {}
    void nextStep(std::uint32_t) override {}
    void type(EStatsType /* type */) override{};
    void iteration(std::size_t /* iteration */) override{};
    void startTime(std::uint64_t /* timestamp */) override{};
    void iterationTime(std::uint64_t /* delta */) override{};
    void lossType(const std::string& /* lossType */) override{};
    void lossValues(std::size_t /* fold */, TDoubleVec&& /* lossValues */) override{};
    void numFolds(std::size_t /* numFolds */) override{};
    void hyperparameters(const SHyperparameters& /* hyperparameters */) override{};
    SHyperparameters& hyperparameters() override { return m_Hyperparameters; };

private:
    SHyperparameters m_Hyperparameters;
};
}
}

#endif //INCLUDED_ml_maths_CDataFrameAnalysisInstrumentationInterface_h
